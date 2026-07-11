from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
import fnmatch
from pathlib import Path
import time
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.agent_loop import AgentLoop, AgentLoopParams
from vibe.core.agents.models import (
    AgentType,
    BuiltinAgentName,
    profile_requires_isolation,
)
from vibe.core.config import ModelPurpose, SessionLoggingConfig, VibeConfig
from vibe.core.logger import logger
from vibe.core.tasking import (
    TaskBrief,
    TaskOutcome,
    TaskOutcomeStatus,
    compile_task_brief,
    resolve_task_outcome,
)
from vibe.core.tasking._candidate import validate_task_candidate
from vibe.core.tasking._policy import (
    BoundTaskContract,
    TaskContractAuthority,
    TaskContractError,
)
from vibe.core.tasking._process_context import TaskProcessContext
from vibe.core.teams._task_checks import run_guarded_task_checks, task_check_diagnostics
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import (
    ToolCallDisplay,
    ToolResultDisplay,
    ToolUIData,
    ToolUIDataAdapter,
)
from vibe.core.types import (
    AssistantEvent,
    Role,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
)
from vibe.core.usage import SpendProcessContext, SpendPurpose
from vibe.core.usage._session import SpendBudgetExceededError
from vibe.core.verification_contract import (
    VerificationReportError,
    parse_verification_report,
)
from vibe.core.workflows._limits import DEFAULT_ISOLATED_MAX_TURNS
from vibe.core.workflows.runtime import IsolatedResult, run_isolated_agent

if TYPE_CHECKING:
    from vibe.core.verification_state import VerificationState


def workspace_fingerprint() -> str | None:
    from vibe.core._workspace_verification import workspace_fingerprint as calculate

    return calculate()


def landing_base_sha() -> str | None:
    from vibe.core.verification_state import landing_base_sha as calculate

    return calculate()


def _configured_subagent_model(ctx: InvokeContext) -> str | None:
    if ctx.agent_manager and ctx.agent_manager.config.subagent_model:
        return ctx.agent_manager.config.subagent_model
    return None


def _configured_grunt_model(ctx: InvokeContext) -> str | None:
    if ctx.agent_manager and ctx.agent_manager.config.grunt_model:
        return ctx.agent_manager.config.grunt_model
    return None


def _effective_subagent_model(args: TaskArgs, ctx: InvokeContext) -> str | None:
    # Resolve the model a subagent spawn runs on. The grunt profile has its
    # own cheap-model default (grunt_model); other profiles fall straight
    # to subagent_model. Both then fall through to the parent session's model.
    resolved: str | None = None
    if (
        args.brief is not None
        and args.brief.manifest.name == "mechanical-edit"
        and ctx.agent_manager is not None
    ):
        resolved = ctx.agent_manager.config.model_routing.alias_for(
            ModelPurpose.MECHANICAL
        )
    if resolved is None:
        resolved = args.model or None
    if resolved is None and args.agent == BuiltinAgentName.GRUNT:
        resolved = _configured_grunt_model(ctx)
    if resolved is None:
        resolved = _configured_subagent_model(ctx)
    if resolved is None:
        resolved = ctx.active_model
    if resolved is None and ctx.agent_manager is not None:
        resolved = ctx.agent_manager.config.active_model or None
    return resolved


def _subagent_error_outcome(error: Exception) -> tuple[TaskOutcomeStatus, str]:
    if isinstance(error, SpendBudgetExceededError):
        reason = error.rejection.reason.value
        return TaskOutcomeStatus.BLOCKED, f"spend budget {reason}: {error}"
    return TaskOutcomeStatus.RETRYABLE, str(error)


def _maybe_record_verifier_pass(
    agent: str,
    response: str,
    ctx: InvokeContext,
    *,
    completed: bool = True,
    attempt: _VerificationAttempt | None = None,
) -> str | None:
    # Only the verifier subagent may set the verifier flag.
    if agent != BuiltinAgentName.VERIFIER:
        return None
    if not response or not completed:
        reason = "response was empty" if not response else "task did not complete"
        return f"Verifier result was not recorded: {reason}"
    state = ctx.verification_state
    if state is None:
        return "Verifier result was not recorded: session verification state is unavailable"
    if attempt is not None:
        changed: str | None = None
        if attempt.generation is not None and not state.is_current_verifier_attempt(
            attempt.generation
        ):
            changed = "verifier attempt was superseded"
        elif (
            attempt.workspace_fingerprint is None
            or workspace_fingerprint() != attempt.workspace_fingerprint
        ):
            changed = "workspace changed during verification"
        elif landing_base_sha() != attempt.base_sha:
            changed = "landing base changed during verification"
        if changed is not None:
            return f"Verifier result was not recorded: {changed}"
    try:
        report = parse_verification_report(response)
    except VerificationReportError as exc:
        return f"Verifier result was not recorded: {exc}"
    if report.passed:
        state.record_verifier_pass(
            report,
            verified_workspace_fingerprint=(
                attempt.workspace_fingerprint if attempt is not None else None
            ),
            verified_base_sha=attempt.base_sha if attempt is not None else None,
        )
    return (
        None
        if report.passed
        else (
            "Verifier did not authorize landing: "
            f"VERDICT: {report.verdict.value.upper()}"
        )
    )


def _with_verification_result(
    outcome: TaskOutcome, agent: str, diagnostic: str | None
) -> TaskOutcome:
    if agent != BuiltinAgentName.VERIFIER:
        return outcome
    if diagnostic is None:
        return outcome.model_copy(
            update={
                "summary": "Verifier PASS recorded for the current candidate",
                "evidence": [
                    *outcome.evidence,
                    "Session verification state recorded the evidence-backed PASS",
                ],
            }
        )
    status = TaskOutcomeStatus.RETRYABLE if outcome.succeeded else outcome.status
    summary = (
        "Verifier report did not satisfy the landing gate"
        if outcome.succeeded
        else outcome.summary
    )
    return outcome.model_copy(
        update={
            "status": status,
            "summary": summary,
            "diagnostics": [*outcome.diagnostics, diagnostic],
        }
    )


@dataclass(frozen=True, slots=True)
class _VerificationAttempt:
    workspace_fingerprint: str | None
    base_sha: str | None
    generation: int | None


def _start_verification_attempt(
    agent: str, state: VerificationState | None = None
) -> _VerificationAttempt | None:
    if agent != BuiltinAgentName.VERIFIER:
        return None
    generation = state.begin_verifier_attempt() if state is not None else None
    return _VerificationAttempt(workspace_fingerprint(), landing_base_sha(), generation)


@dataclass
class _InProcessResult:
    # IsolatedResult-shaped result for a backgrounded in-process subagent, so the
    # registry finalizer reads .output/.returncode like the isolated path.
    output: str
    returncode: int
    worktree_path: str | None = None
    branch: str | None = None
    outcome: TaskOutcome | None = None


class TaskArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task: str | TaskBrief = Field(
        description=(
            "The task to delegate. Pass a string for legacy free-form delegation "
            "or a TaskBrief object for validated metadata and explicit outcomes."
        )
    )
    agent: str = Field(
        default="explore",
        description="Name of the agent profile to use (must be a subagent)",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model alias the subagent should run on, overriding the subagent's "
            "default model for this spawn. Must be one of the configured model "
            "aliases listed in your system context (Models available for "
            "subagents). Use it to route a delegated task to a specific model "
            "(e.g. run a review on a stronger or different-provider model than "
            "the host). Omit to inherit the subagent profile's model. An "
            "unconfigured alias fails with the list of valid aliases."
        ),
    )
    async_run: bool = Field(
        default=True,
        description=(
            "Background execution (the DEFAULT): returns immediately with a "
            "task_id; the subagent runs in the background and its result is "
            "delivered to you automatically at the start of a later turn (you "
            "are auto-resumed on completion). The running task shows in the Tasks "
            "pane and the `background` tool, cancellable via `background stop "
            "<task_id>`. Set async_run=false ONLY to block and get the subagent's "
            "result inline in this same turn."
        ),
    )

    @property
    def brief(self) -> TaskBrief | None:
        return self.task if isinstance(self.task, TaskBrief) else None

    @property
    def prompt(self) -> str:
        if isinstance(self.task, TaskBrief):
            return compile_task_brief(self.task)
        return self.task

    @property
    def summary(self) -> str:
        if isinstance(self.task, TaskBrief):
            return self.task.objective
        return self.task


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: str = Field(description="The accumulated response from the subagent")
    turns_used: int | None = Field(
        default=None,
        description=(
            "Number of turns the subagent used. None when unknown (isolated "
            "subagents run in a subprocess that does not report turn count)."
        ),
    )
    completed: bool = Field(
        description="Whether the agent execution completed normally"
    )
    outcome: TaskOutcome | None = Field(
        default=None,
        description=(
            "Terminal task outcome. None only for a background launch handoff or "
            "a legacy caller-created result."
        ),
    )
    isolated: bool = Field(
        default=False, description="Whether the subagent ran in an isolated worktree."
    )
    worktree_path: str | None = Field(
        default=None,
        description=(
            "Path to a kept isolated worktree (only set when the subagent ran "
            "isolated and its work could not be delivered). Use `git -C "
            "<worktree_path> ...` or merge its branch to recover the work."
        ),
    )
    branch: str | None = Field(
        default=None,
        description=(
            "Branch of a kept isolated worktree (only set alongside "
            "worktree_path). Recover with `git merge <branch>`."
        ),
    )
    task_id: str | None = Field(
        default=None,
        description=(
            "Background task id (only set when async_run=true). Use "
            "`background` to inspect status and `background stop <task_id>` to "
            "cancel; completion surfaces as a BackgroundTaskCompletedEvent."
        ),
    )


class TaskToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    allowlist: list[str] = Field(default=[BuiltinAgentName.EXPLORE])
    isolation: Literal["off", "auto", "always"] = "auto"


class Task(
    BaseTool[TaskArgs, TaskResult, TaskToolConfig, BaseToolState],
    ToolUIData[TaskArgs, TaskResult],
):
    description: ClassVar[str] = (
        "Delegate a task to a subagent for independent execution. "
        "Useful for exploration, research, or parallel work that doesn't "
        "require user interaction. By default write-capable profiles "
        "(worker/auto-approve/editor/grunt) run in an isolated git worktree — "
        "write/edit/read are auto-approved and confined to the worktree, so "
        "edits can't race the parent tree or escape it; read-only profiles run "
        "in-memory for speed. Override with the task.isolation config "
        "(off|auto|always).\n\n"
        "Delegation hygiene: hand the subagent the concrete context you already "
        "have — the exact file:line refs you found, the diff, and the specific "
        "question to answer — so it verifies rather than re-discovers. Do NOT "
        "tell it to broadly explore a large or external repo you have already "
        "searched; point it at specific paths/symbols and scope its searches. A "
        "subagent has its own, often smaller, context window, so an open-ended "
        "broad search there can overflow it — give it targets, not a hunt.\n\n"
        "For structured work, pass a TaskBrief instead of a free-form string. "
        "The host binds its paths, trusted check IDs, budget, deadline, and tool "
        "manifest before dispatch. The worker cannot widen that contract and must "
        "return an explicit terminal outcome.\n\n"
        "Execution: subagents run in the BACKGROUND by default — the call returns "
        "a task_id immediately and the subagent's result is delivered to you "
        "automatically at the start of a later turn (you are auto-resumed when it "
        "finishes; the run is visible in the Tasks pane). Spawn what you need, "
        "then continue other work or end your turn. Pass async_run=false only "
        "when you must have the result inline in THIS turn (you block until done)."
    )

    is_subagent_spawner: ClassVar[bool] = True

    @classmethod
    def call_is_read_only(
        cls, args: BaseModel, *, agent_manager: object = None
    ) -> bool:
        # A subagent call is side-effect-free (safe to fan out concurrently)
        # only when it runs in-process read-only: not a background async_run, and
        # a profile that does not require isolation. Write-capable profiles stay
        # sequential — conservative; isolation would make them concurrent-safe
        # too, but this gate never needs the per-tool isolation config. Mirrors
        # the in-process vs isolated decision in invoke().
        if (
            getattr(args, "agent", "") == BuiltinAgentName.VERIFIER
            or getattr(args, "async_run", False)
            or agent_manager is None
        ):
            return False
        get_agent = getattr(agent_manager, "get_agent", None)
        if get_agent is None:
            return False
        try:
            profile = get_agent(getattr(args, "agent", ""))
        except Exception:
            return False
        return not profile_requires_isolation(profile)

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, TaskArgs):
            return ToolCallDisplay(
                summary=f"Running {args.agent} agent: {args.summary}"
            )
        return ToolCallDisplay(summary="Running subagent")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, TaskResult):
            # A set task_id means this is a background-launch handoff, not a
            # terminal result: completed=False here means "still running", not
            # "interrupted". The launch itself succeeded.
            if result.task_id is not None:
                return ToolResultDisplay(
                    success=True, message="Agent running in background"
                )
            if result.outcome is not None and not result.outcome.succeeded:
                return ToolResultDisplay(
                    success=False,
                    message=f"Agent outcome: {result.outcome.status.value}",
                )
            # turns_used is None for isolated subagents (subprocess doesn't
            # report it); omit the count instead of showing a misleading "0".
            if result.turns_used is None:
                msg = "Agent interrupted" if not result.completed else "Agent completed"
                return ToolResultDisplay(success=result.completed, message=msg)
            turn_word = "turn" if result.turns_used == 1 else "turns"
            if not result.completed:
                return ToolResultDisplay(
                    success=False,
                    message=f"Agent interrupted after {result.turns_used} {turn_word}",
                )
            return ToolResultDisplay(
                success=True,
                message=f"Agent completed in {result.turns_used} {turn_word}",
            )
        return ToolResultDisplay(success=True, message="Agent completed")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running subagent"

    def resolve_permission(self, args: TaskArgs) -> PermissionContext | None:
        agent_name = args.agent

        for pattern in self.config.denylist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.NEVER)

        for pattern in self.config.allowlist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.ALWAYS)

        return None

    async def _collect_async_isolated(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        run: Awaitable[IsolatedResult],
        verification_attempt: _VerificationAttempt | None,
        contract: BoundTaskContract | None,
    ) -> _InProcessResult:
        try:
            result = await run
            return await self._finalize_isolated_result(
                args,
                ctx,
                result,
                contract=contract,
                verification_attempt=verification_attempt,
            )
        except Exception as e:
            response = f"[Isolated subagent error: {e}]"
            forced_status, diagnostic = _subagent_error_outcome(e)
        return _InProcessResult(
            output=response,
            returncode=1,
            outcome=resolve_task_outcome(
                args.brief,
                response,
                completed=False,
                forced_status=forced_status,
                diagnostic=diagnostic,
            ),
        )

    @staticmethod
    def _isolated_spend_context(
        args: TaskArgs, ctx: InvokeContext, contract: BoundTaskContract | None
    ) -> SpendProcessContext | None:
        if ctx.spend_adapter is None:
            return None
        purpose = (
            SpendPurpose.VERIFICATION
            if args.agent == BuiltinAgentName.VERIFIER
            else SpendPurpose.PRIMARY
        )
        if contract is not None:
            return ctx.spend_adapter.child_task(
                limits=contract.spend_limits(),
                task_brief_hash=contract.brief_hash,
                purpose=purpose,
            ).export_process_context()
        return ctx.spend_adapter.child_agent(purpose=purpose).export_process_context()

    async def _finalize_isolated_result(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        result: IsolatedResult,
        *,
        contract: BoundTaskContract | None,
        verification_attempt: _VerificationAttempt | None,
    ) -> _InProcessResult:
        response = result.output
        completed = result.returncode == 0
        preliminary = resolve_task_outcome(args.brief, response, completed=completed)
        if contract is None:
            preliminary = _with_verification_result(
                preliminary,
                args.agent,
                _maybe_record_verifier_pass(
                    args.agent,
                    response,
                    ctx,
                    completed=completed and preliminary.succeeded,
                    attempt=verification_attempt,
                ),
            )
            return _InProcessResult(
                output=response,
                returncode=result.returncode,
                worktree_path=result.worktree_path,
                branch=result.branch,
                outcome=preliminary,
            )

        wt = result.wt
        if wt is None:
            if not preliminary.succeeded:
                return _InProcessResult(
                    output=response, returncode=result.returncode, outcome=preliminary
                )
            outcome = TaskOutcome(
                status=TaskOutcomeStatus.RETRYABLE,
                summary="Structured isolated task could not be validated",
                diagnostics=["isolated worker returned no candidate worktree"],
                manifest=args.brief.manifest if args.brief else None,
            )
            return _InProcessResult(output=response, returncode=1, outcome=outcome)

        from vibe.core.worktree.ephemeral import (
            deliver_ephemeral_worktree,
            remove_ephemeral_worktree,
        )

        delivered = False
        branch: str | None = None
        outcome = preliminary
        try:
            if preliminary.succeeded:
                validation = await asyncio.to_thread(
                    validate_task_candidate, contract, wt.path, wt.base_sha
                )
                evidence = [
                    f"{check.name}: exit {check.exit_code} ({check.duration_ms} ms)"
                    for check in validation.checks
                ]
                if not validation.passed:
                    outcome = TaskOutcome(
                        status=(
                            TaskOutcomeStatus.RETRYABLE
                            if validation.scope_passed
                            else TaskOutcomeStatus.BLOCKED
                        ),
                        summary=(
                            "Trusted acceptance checks failed"
                            if validation.scope_passed
                            else "Candidate violated the bound path scope"
                        ),
                        evidence=evidence,
                        diagnostics=list(validation.diagnostics),
                        changed_paths=list(validation.changed_paths),
                        manifest=args.brief.manifest if args.brief else None,
                    )
                else:
                    delivered = await asyncio.to_thread(deliver_ephemeral_worktree, wt)
                    outcome = TaskOutcome(
                        status=(
                            TaskOutcomeStatus.SUCCEEDED
                            if delivered
                            else TaskOutcomeStatus.RETRYABLE
                        ),
                        summary=(
                            "Structured task and trusted checks succeeded"
                            if delivered
                            else "Validated candidate could not be delivered"
                        ),
                        evidence=evidence,
                        diagnostics=(
                            []
                            if delivered
                            else ["parent repository moved or refused the fast-forward"]
                        ),
                        changed_paths=list(validation.changed_paths),
                        manifest=args.brief.manifest if args.brief else None,
                    )
        finally:
            removed = await asyncio.to_thread(
                remove_ephemeral_worktree, wt, keep_if_changed=not delivered
            )
            if not removed:
                branch = wt.branch

        outcome = _with_verification_result(
            outcome,
            args.agent,
            _maybe_record_verifier_pass(
                args.agent,
                response,
                ctx,
                completed=outcome.succeeded,
                attempt=verification_attempt,
            ),
        )
        return _InProcessResult(
            output=response,
            returncode=0 if outcome.succeeded else 1,
            branch=branch,
            outcome=outcome,
        )

    async def _run_async_isolated(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        verification_attempt: _VerificationAttempt | None = None,
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        registry = ctx.background_registry
        if registry is None:
            # No registry wired (e.g. tests, programmatic runner without TUI).
            # Fall back to the blocking isolated path rather than failing hard.
            async for result in self._run_isolated(
                args, ctx, verification_attempt=verification_attempt
            ):
                yield result
            return

        task_text = args.prompt
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{task_text}"
            )

        denied = await self._judge_isolated_spawn(task_text, args.agent, ctx)
        if denied is not None:
            response = f"[Isolated subagent denied by safety judge: {denied}]"
            yield TaskResult(
                response=response,
                completed=False,
                isolated=True,
                outcome=resolve_task_outcome(
                    args.brief,
                    response,
                    completed=False,
                    forced_status=TaskOutcomeStatus.BLOCKED,
                    diagnostic=denied,
                ),
            )
            return

        log_path = self._bg_log_path(ctx)
        if log_path is not None:
            log_path.touch()

        contract = self._bind_contract(args.brief, ctx) if args.brief else None
        effective_model = _effective_subagent_model(args, ctx)
        isolated_run = run_isolated_agent(
            task_text,
            args.agent,
            label=args.agent,
            max_turns=DEFAULT_ISOLATED_MAX_TURNS,
            deliver=contract is None,
            keep_worktree=contract is not None,
            model=effective_model,
            log_path=log_path,
            scratchpad_dir=ctx.scratchpad_dir,
            spend_context=self._isolated_spend_context(args, ctx, contract),
            task_context=(
                TaskProcessContext.from_brief(args.brief) if args.brief else None
            ),
        )
        bg_task = asyncio.create_task(
            self._collect_async_isolated(
                args, ctx, isolated_run, verification_attempt, contract
            ),
            name=f"async-task-{args.agent}",
        )
        task_id = registry.register_async_agent(
            args.agent,
            bg_task,
            label=self._subagent_label(args),
            prompt=args.prompt,
            model=effective_model,
            log_path=log_path,
        )
        yield ToolStreamEvent(
            tool_name=self.get_name(),
            message=f"Launched {args.agent} subagent in background: {task_id}",
            tool_call_id=ctx.tool_call_id,
        )
        yield TaskResult(
            response=(
                f"Background subagent {task_id} launched. Inspect with "
                f"`background`; cancel with `background stop {task_id}`. "
                f"Completion surfaces as a BackgroundTaskCompletedEvent at the "
                f"top of the next parent turn."
            ),
            completed=False,
            isolated=True,
            task_id=task_id,
        )

    async def _run_isolated(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        verification_attempt: _VerificationAttempt | None = None,
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        task_text = args.prompt
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{task_text}"
            )
        yield ToolStreamEvent(
            tool_name=self.get_name(),
            message=f"Running {args.agent} agent (isolated worktree)",
            tool_call_id=ctx.tool_call_id,
        )
        completed = True
        response_text = ""
        worktree_path: str | None = None
        branch: str | None = None
        outcome: TaskOutcome | None = None
        forced_status: TaskOutcomeStatus | None = None
        diagnostic: str | None = None
        try:
            denied = await self._judge_isolated_spawn(task_text, args.agent, ctx)
            if denied is not None:
                # Judge denied the delegation (or user declined at the approval
                # prompt). Fail the TaskResult cleanly rather than raising so the
                # tool surfaces the denial in-band; no subprocess is spawned.
                completed = False
                response_text = f"[Isolated subagent denied by safety judge: {denied}]"
                forced_status = TaskOutcomeStatus.BLOCKED
                diagnostic = denied
            else:
                contract = self._bind_contract(args.brief, ctx) if args.brief else None
                result = await run_isolated_agent(
                    task_text,
                    args.agent,
                    label=args.agent,
                    max_turns=DEFAULT_ISOLATED_MAX_TURNS,
                    deliver=contract is None,
                    keep_worktree=contract is not None,
                    # Inherit the parent's effective model (not the configured default).
                    model=_effective_subagent_model(args, ctx),
                    scratchpad_dir=ctx.scratchpad_dir,
                    spend_context=self._isolated_spend_context(args, ctx, contract),
                    task_context=(
                        TaskProcessContext.from_brief(args.brief)
                        if args.brief
                        else None
                    ),
                )
                finalized = await self._finalize_isolated_result(
                    args,
                    ctx,
                    result,
                    contract=contract,
                    verification_attempt=verification_attempt,
                )
                response_text = finalized.output
                completed = finalized.returncode == 0
                worktree_path = finalized.worktree_path
                branch = finalized.branch
                outcome = finalized.outcome
        except Exception as e:
            completed = False
            response_text = f"[Isolated subagent error: {e}]"
            forced_status = TaskOutcomeStatus.RETRYABLE
            diagnostic = str(e)

        yield TaskResult(
            response=response_text,
            turns_used=None,  # isolated subprocess doesn't report turn count
            completed=completed,
            isolated=True,
            worktree_path=worktree_path,
            branch=branch,
            outcome=(
                outcome
                if outcome is not None
                else resolve_task_outcome(
                    args.brief,
                    response_text,
                    completed=completed,
                    forced_status=forced_status,
                    diagnostic=diagnostic,
                )
            ),
        )

    async def _judge_isolated_spawn(
        self, prompt: str, agent: str, ctx: InvokeContext
    ) -> str | None:
        factory = getattr(ctx, "safety_judge_factory", None)
        if factory is None:
            return None
        try:
            judge = factory()
        except Exception:
            return None
        if judge is None:
            return None
        verdict = await judge.judge(
            "task", prompt, [f"isolated '{agent}' subagent spawn"]
        )
        if verdict.safe:
            return None
        # Deferred to the user. Surface via the host approval callback if one is
        # wired; otherwise fail closed (deny the spawn).
        approval_callback = getattr(ctx, "approval_callback", None)
        if approval_callback is None:
            return verdict.reason
        from vibe.core.types import ApprovalResponse

        response, _feedback, _modified = await approval_callback(
            f"task_isolated:{agent}",
            None,
            f"task-isolated-spawn-{agent}",
            None,
            verdict.reason,
        )
        return None if response == ApprovalResponse.YES else verdict.reason

    async def run(
        self, args: TaskArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if not ctx or not ctx.agent_manager:
            raise ToolError("Task tool requires agent_manager in context")

        if (
            args.brief is not None
            and args.brief.deadline is not None
            and args.brief.deadline <= datetime.now(UTC)
        ):
            diagnostic = (
                f"task deadline {args.brief.deadline.isoformat()} expired before "
                "subagent dispatch"
            )
            yield TaskResult(
                response=f"[Structured task blocked: {diagnostic}]",
                turns_used=0,
                completed=False,
                outcome=resolve_task_outcome(
                    args.brief,
                    "",
                    completed=False,
                    forced_status=TaskOutcomeStatus.BLOCKED,
                    diagnostic=diagnostic,
                ),
            )
            return

        if args.brief is not None:
            self._bind_contract(args.brief, ctx)

        agent_manager = ctx.agent_manager

        try:
            agent_profile = agent_manager.get_agent(args.agent)
        except ValueError as e:
            raise ToolError(f"Unknown agent: {args.agent}") from e

        if agent_profile.agent_type != AgentType.SUBAGENT:
            raise ToolError(
                f"Agent '{args.agent}' is a {agent_profile.agent_type.value} agent. "
                f"Only subagents can be used with the task tool. "
                f"This is a security constraint to prevent recursive spawning."
            )

        if args.model is not None:
            valid_aliases = {m.alias for m in agent_manager.config.models}
            if args.model not in valid_aliases:
                raise ToolError(
                    f"Unknown model alias '{args.model}'. Configured aliases: "
                    f"{', '.join(sorted(valid_aliases))}."
                )

        isolation_mode = self.config.isolation
        should_isolate = isolation_mode == "always" or (
            isolation_mode == "auto" and profile_requires_isolation(agent_profile)
        )
        verification_attempt = _start_verification_attempt(
            args.agent, ctx.verification_state
        )
        if args.async_run:
            if should_isolate:
                async for result in self._run_async_isolated(
                    args, ctx, verification_attempt=verification_attempt
                ):
                    yield result
            else:
                async for result in self._run_in_process_async(
                    args, ctx, verification_attempt=verification_attempt
                ):
                    yield result
            return
        if should_isolate:
            async for result in self._run_isolated(
                args, ctx, verification_attempt=verification_attempt
            ):
                yield result
            return
        async for result in self._run_in_process(
            args, ctx, verification_attempt=verification_attempt
        ):
            yield result

    @staticmethod
    def _subagent_label(args: TaskArgs) -> str:
        snippet = args.summary.strip().split("\n", 1)[0][:60]
        return f"{args.agent}: {snippet}" if snippet else args.agent

    @staticmethod
    def _bind_contract(brief: TaskBrief, ctx: InvokeContext) -> BoundTaskContract:
        try:
            return BoundTaskContract.bind(
                brief,
                authority=TaskContractAuthority.LEAD,
                workspace_root=Path.cwd(),
                verification_state=ctx.verification_state,
            )
        except TaskContractError as e:
            raise ToolError(f"Invalid structured task contract: {e}") from e

    def _build_subagent_loop(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> tuple[AgentLoop, str]:
        task_contract = (
            self._bind_contract(args.brief, ctx) if args.brief is not None else None
        )
        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx.session_dir else "",
            session_prefix=args.agent,
            enabled=ctx.session_dir is not None,
        )
        # A fresh VibeConfig.load() falls back to the hardcoded default (mistral),
        # which fails when the parent runs on another provider; inherit instead.
        inherited_model = _effective_subagent_model(args, ctx)
        load_overrides: dict[str, str] = {}
        if inherited_model:
            load_overrides["active_model"] = inherited_model
        base_config = VibeConfig.load(session_logging=session_logging, **load_overrides)
        try:
            resolved_provider = base_config.get_active_provider().name
        except Exception as e:
            resolved_provider = repr(e)
        logger.warning(
            "subagent model resolve: agent=%s args_model=%s grunt_model=%s"
            " subagent_model=%s ctx_active=%s has_mgr=%s -> inherited=%s"
            " loaded_active=%s provider=%s",
            args.agent,
            args.model,
            _configured_grunt_model(ctx),
            _configured_subagent_model(ctx),
            ctx.active_model,
            ctx.agent_manager is not None,
            inherited_model,
            base_config.active_model,
            resolved_provider,
        )
        spend_adapter = None
        if ctx.spend_adapter is not None:
            purpose = (
                SpendPurpose.VERIFICATION
                if args.agent == BuiltinAgentName.VERIFIER
                else SpendPurpose.PRIMARY
            )
            spend_adapter = (
                ctx.spend_adapter.child_task(
                    limits=task_contract.spend_limits(),
                    task_brief_hash=task_contract.brief_hash,
                    purpose=purpose,
                )
                if task_contract is not None
                else ctx.spend_adapter.child_agent(purpose=purpose)
            )
        # Subagents inherit the parent worktree; never call worktree_manager.enter().
        subagent_loop = AgentLoop(
            config=base_config,
            params=AgentLoopParams(
                agent_name=args.agent,
                launch_context=ctx.launch_context,
                terminal_emulator=ctx.terminal_emulator,
                is_subagent=True,
                # Stream like the host: reasoning models (k2.7-code/GLM) need stream=True.
                enable_streaming=True,
                defer_heavy_init=True,
                permission_store=ctx.permission_store,
                hook_config_result=ctx.hook_config_result,
                max_turns=DEFAULT_ISOLATED_MAX_TURNS,
                spend_adapter=spend_adapter,
                task_contract=task_contract,
            ),
        )
        if ctx.session_id:
            subagent_loop.parent_session_id = ctx.session_id
        if ctx.approval_callback:
            subagent_loop.set_approval_callback(ctx.approval_callback)
        task_text = args.prompt
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{task_text}"
            )
        return subagent_loop, task_text

    async def _run_in_process(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        verification_attempt: _VerificationAttempt | None = None,
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        subagent_loop, task_text = self._build_subagent_loop(args, ctx)
        accumulated_response: list[str] = []
        completed = True
        forced_status: TaskOutcomeStatus | None = None
        diagnostic: str | None = None
        try:
            async with aclosing(subagent_loop.act(task_text)) as events:
                async for event in events:
                    if isinstance(event, AssistantEvent) and event.content:
                        accumulated_response.append(event.content)
                        if event.stopped_by_middleware:
                            completed = False
                    elif isinstance(event, ToolResultEvent):
                        if event.skipped:
                            completed = False
                        elif event.result and event.tool_class:
                            adapter = ToolUIDataAdapter(event.tool_class)
                            display = adapter.get_result_display(event)
                            message = f"{event.tool_name}: {display.message}"
                            yield ToolStreamEvent(
                                tool_name=self.get_name(),
                                message=message,
                                tool_call_id=ctx.tool_call_id,
                            )

            turns_used = sum(
                msg.role == Role.ASSISTANT for msg in subagent_loop.messages
            )

        except Exception as e:
            completed = False
            accumulated_response.append(f"\n[Subagent error: {e}]")
            forced_status, diagnostic = _subagent_error_outcome(e)
            turns_used = sum(
                msg.role == Role.ASSISTANT for msg in subagent_loop.messages
            )
        finally:
            with suppress(Exception):
                await subagent_loop.aclose()

        response = "".join(accumulated_response)
        outcome = await self._finalize_in_process_outcome(
            args,
            ctx,
            response,
            completed=completed,
            forced_status=forced_status,
            diagnostic=diagnostic,
        )
        outcome = _with_verification_result(
            outcome,
            args.agent,
            _maybe_record_verifier_pass(
                args.agent,
                response,
                ctx,
                completed=completed and outcome.succeeded,
                attempt=verification_attempt,
            ),
        )
        yield TaskResult(
            response=response,
            turns_used=turns_used,
            completed=completed,
            outcome=outcome,
        )

    async def _finalize_in_process_outcome(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        response: str,
        *,
        completed: bool,
        forced_status: TaskOutcomeStatus | None,
        diagnostic: str | None,
    ) -> TaskOutcome:
        outcome = resolve_task_outcome(
            args.brief,
            response,
            completed=completed,
            forced_status=forced_status,
            diagnostic=diagnostic,
        )
        if args.brief is None or not outcome.succeeded:
            return outcome

        contract = self._bind_contract(args.brief, ctx)
        evidence, mutation = await asyncio.to_thread(
            run_guarded_task_checks, contract.trusted_checks, contract.workspace_root
        )
        summaries = [
            f"{item.name}: exit {item.exit_code} ({item.duration_ms} ms)"
            for item in evidence
        ]
        if mutation is not None:
            return TaskOutcome(
                status=TaskOutcomeStatus.BLOCKED,
                summary="Trusted checks violated the candidate boundary",
                evidence=summaries,
                diagnostics=[mutation],
                manifest=args.brief.manifest,
            )
        if evidence and all(item.passed for item in evidence):
            return TaskOutcome(
                status=TaskOutcomeStatus.SUCCEEDED,
                summary="Structured task and trusted checks succeeded",
                evidence=summaries,
                manifest=args.brief.manifest,
            )
        return TaskOutcome(
            status=TaskOutcomeStatus.RETRYABLE,
            summary="Trusted acceptance checks failed",
            evidence=summaries,
            diagnostics=list(task_check_diagnostics(evidence)),
            manifest=args.brief.manifest,
        )

    async def _run_in_process_collect(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        verification_attempt: _VerificationAttempt | None = None,
    ) -> _InProcessResult:
        registry = ctx.background_registry
        current_task = asyncio.current_task()
        subagent_loop, task_text = self._build_subagent_loop(args, ctx)
        accumulated_response: list[str] = []
        completed = True
        turns = 0
        forced_status: TaskOutcomeStatus | None = None
        diagnostic: str | None = None
        try:
            async with aclosing(subagent_loop.act(task_text)) as events:
                async for event in events:
                    if isinstance(event, AssistantEvent) and event.content:
                        accumulated_response.append(event.content)
                        turns += 1
                        if event.stopped_by_middleware:
                            completed = False
                        if registry is not None and current_task is not None:
                            registry.update_async_progress_by_task(
                                current_task,
                                response_so_far="".join(accumulated_response),
                                turns_used=turns,
                            )
                    elif isinstance(event, ToolResultEvent) and event.skipped:
                        completed = False
        except Exception as e:
            completed = False
            accumulated_response.append(f"\n[Subagent error: {e}]")
            forced_status, diagnostic = _subagent_error_outcome(e)
        finally:
            with suppress(Exception):
                await subagent_loop.aclose()
        response = "".join(accumulated_response)
        outcome = await self._finalize_in_process_outcome(
            args,
            ctx,
            response,
            completed=completed,
            forced_status=forced_status,
            diagnostic=diagnostic,
        )
        outcome = _with_verification_result(
            outcome,
            args.agent,
            _maybe_record_verifier_pass(
                args.agent,
                response,
                ctx,
                completed=completed and outcome.succeeded,
                attempt=verification_attempt,
            ),
        )
        return _InProcessResult(
            output=response,
            returncode=0 if completed and outcome.succeeded else 1,
            outcome=outcome,
        )

    async def _run_in_process_async(
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        verification_attempt: _VerificationAttempt | None = None,
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        registry = ctx.background_registry
        if registry is None:
            async for result in self._run_in_process(
                args, ctx, verification_attempt=verification_attempt
            ):
                yield result
            return
        effective_model = _effective_subagent_model(args, ctx)
        result_path = self._bg_log_path(ctx)
        if verification_attempt is None:
            collect = self._run_in_process_collect(args, ctx)
        else:
            collect = self._run_in_process_collect(
                args, ctx, verification_attempt=verification_attempt
            )
        bg_task = asyncio.create_task(collect, name=f"async-task-{args.agent}")
        task_id = registry.register_async_agent(
            args.agent,
            bg_task,
            label=self._subagent_label(args),
            prompt=args.prompt,
            model=effective_model,
            log_path=result_path,
        )
        yield ToolStreamEvent(
            tool_name=self.get_name(),
            message=f"Launched {args.agent} subagent in background: {task_id}",
            tool_call_id=ctx.tool_call_id,
        )
        yield TaskResult(
            response=(
                f"Background subagent {task_id} launched. Inspect with "
                f"`background`; cancel with `background stop {task_id}`. "
                f"Completion surfaces at the top of the next parent turn."
            ),
            completed=False,
            task_id=task_id,
        )

    @staticmethod
    def _bg_log_path(ctx: InvokeContext) -> Path | None:
        root = ctx.scratchpad_dir or ctx.session_dir
        if root is None:
            return None
        bg_dir = Path(str(root)) / "bg"
        bg_dir.mkdir(parents=True, exist_ok=True)
        return bg_dir / f"asub-{time.monotonic_ns()}.log"
