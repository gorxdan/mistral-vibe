from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
import fnmatch
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.agent_loop import AgentLoop, AgentLoopParams
from vibe.core.agents.models import (
    BUILTIN_AGENTS,
    AgentProfile,
    AgentType,
    BuiltinAgentName,
    profile_requires_isolation,
)
from vibe.core.candidate_delivery import CandidateDelivery, CandidateDeliveryStatus
from vibe.core.config import ModelPurpose, SessionLoggingConfig, VibeConfig
from vibe.core.logger import logger
from vibe.core.tasking import (
    TaskBrief,
    TaskManifestIdentity,
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
from vibe.core.tools._task_manifest import (
    TaskManifestError,
    resolve_task_manifest,
    validate_task_manifest_for_agent,
)
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
    VerificationReport,
    VerificationReportError,
    VerificationVerdict,
    parse_verification_report,
    report_evidence_was_observed,
)
from vibe.core.verification_state import VerifierAttemptDisposition
from vibe.core.workflows._limits import DEFAULT_ISOLATED_MAX_TURNS
from vibe.core.workflows.runtime import IsolatedResult, run_isolated_agent

if TYPE_CHECKING:
    from vibe.core.verification_state import VerificationState


_MANAGED_TASK_AGENTS = frozenset({BuiltinAgentName.REVIEWER, BuiltinAgentName.VERIFIER})


def _is_managed_session(ctx: InvokeContext) -> bool:
    state = ctx.verification_state
    recipe = state.trusted_recipe if state is not None else None
    return bool(recipe is not None and recipe.config.execution_topology is not None)


def _validate_managed_task_agent(
    ctx: InvokeContext, *, agent_name: str, agent_profile: AgentProfile
) -> bool:
    if not _is_managed_session(ctx):
        return False
    if agent_name not in _MANAGED_TASK_AGENTS:
        allowed = ", ".join(sorted(_MANAGED_TASK_AGENTS))
        raise ToolError(
            "Managed execution topology restricts task delegation to "
            f"read-only review agents: {allowed}"
        )
    if agent_profile is not BUILTIN_AGENTS[agent_name]:
        raise ToolError(
            "Managed execution topology requires the host-owned builtin "
            f"agent profile '{agent_name}'"
        )
    if profile_requires_isolation(agent_profile):
        raise ToolError(
            "Managed execution topology rejected write-capable or unjailed "
            f"agent profile '{agent_name}'"
        )
    return True


def workspace_fingerprint() -> str | None:
    from vibe.core._workspace_verification import workspace_fingerprint as calculate

    return calculate()


def landing_base_sha() -> str | None:
    from vibe.core.verification_state import landing_base_sha as calculate

    return calculate()


def _verification_base_sha(state: VerificationState | None) -> str | None:
    if state is not None and state.trusted_recipe is not None:
        topology = state.trusted_recipe.config.execution_topology
        if topology is not None:
            return topology.baseline_sha
    return landing_base_sha()


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


def _verifier_report_preflight(
    response: str,
    state: VerificationState,
    *,
    completed: bool,
    authorized: bool,
    attempt: _VerificationAttempt | None,
    evidence_hashes: tuple[str, ...],
) -> tuple[VerificationReport | None, str | None]:
    if not response or not completed:
        reason = "response was empty" if not response else "task did not complete"
        return None, f"Verifier result was not recorded: {reason}"
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
        elif _verification_base_sha(state) != attempt.base_sha:
            changed = "landing base changed during verification"
        if changed is not None:
            return None, f"Verifier result was not recorded: {changed}"
    try:
        report = parse_verification_report(response)
    except VerificationReportError as exc:
        return None, f"Verifier result was not recorded: {exc}"
    if report.passed and not authorized:
        return (
            None,
            "Verifier PASS was not recorded: trusted task outcome did not succeed",
        )
    if report.passed and not report_evidence_was_observed(report, evidence_hashes):
        return (
            None,
            "Verifier result was not recorded: PASS evidence did not match output "
            "from eligible host-observed verification commands",
        )
    return report, None


def _maybe_record_verifier_pass(
    agent: str,
    response: str,
    ctx: InvokeContext,
    *,
    completed: bool = True,
    authorized: bool = True,
    attempt: _VerificationAttempt | None = None,
    evidence_hashes: tuple[str, ...] = (),
) -> str | None:
    # Only the verifier subagent may set the verifier flag.
    if agent != BuiltinAgentName.VERIFIER:
        return None
    state = ctx.verification_state
    if state is None:
        return "Verifier result was not recorded: session verification state is unavailable"
    generation = attempt.generation if attempt is not None else None

    def reject(diagnostic: str) -> str:
        state.record_verifier_result(
            generation, VerifierAttemptDisposition.INVALID, diagnostic
        )
        return diagnostic

    report, preflight_diagnostic = _verifier_report_preflight(
        response,
        state,
        completed=completed,
        authorized=authorized,
        attempt=attempt,
        evidence_hashes=evidence_hashes,
    )
    if preflight_diagnostic is not None:
        return reject(preflight_diagnostic)
    assert report is not None
    if report.passed:
        if not state.record_verifier_result(
            generation,
            VerifierAttemptDisposition.PASS,
            "Verifier PASS was recorded for the current candidate.",
        ):
            return (
                "Verifier result was not recorded: verifier attempt already reached "
                "a terminal disposition"
            )
        state.record_verifier_pass(
            report,
            verifier_attempt_generation=(
                state.verifier_attempt_generation if generation is None else generation
            ),
            verified_workspace_fingerprint=(
                attempt.workspace_fingerprint if attempt is not None else None
            ),
            verified_base_sha=attempt.base_sha if attempt is not None else None,
        )
        recording_diagnostic = None
    else:
        recording_diagnostic = (
            "Verifier did not authorize landing: "
            f"VERDICT: {report.verdict.value.upper()}"
        )
        disposition = (
            VerifierAttemptDisposition.FAIL
            if report.verdict is VerificationVerdict.FAIL
            else VerifierAttemptDisposition.PARTIAL
        )
        state.record_verifier_result(generation, disposition, recording_diagnostic)
    return recording_diagnostic


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
    diagnostics = list(outcome.diagnostics)
    if diagnostic not in diagnostics:
        diagnostics.append(diagnostic)
    return outcome.model_copy(
        update={"status": status, "summary": summary, "diagnostics": diagnostics}
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
    base_sha = _verification_base_sha(state)
    return _VerificationAttempt(workspace_fingerprint(), base_sha, generation)


def _ensure_verifier_attempt_terminal(
    agent: str,
    state: VerificationState | None,
    attempt: _VerificationAttempt | None,
    diagnostic: str,
) -> None:
    if (
        agent != BuiltinAgentName.VERIFIER
        or state is None
        or attempt is None
        or attempt.generation is None
        or not state.is_current_verifier_attempt(attempt.generation)
    ):
        return
    latest = state.latest_verifier_attempt
    if latest is None or latest.disposition is not VerifierAttemptDisposition.PENDING:
        return
    state.record_verifier_result(
        attempt.generation, VerifierAttemptDisposition.INVALID, diagnostic
    )


@dataclass
class _InProcessResult:
    # IsolatedResult-shaped result for a backgrounded in-process subagent, so the
    # registry finalizer reads .output/.returncode like the isolated path.
    output: str
    returncode: int
    worktree_path: str | None = None
    branch: str | None = None
    candidate_delivery: CandidateDelivery | None = None
    outcome: TaskOutcome | None = None


async def _validated_structured_candidate_outcome(
    contract: BoundTaskContract, wt: Any, manifest: TaskManifestIdentity | None
) -> tuple[TaskOutcome, CandidateDelivery | None, bool]:
    from vibe.core.workflows._verified_delivery import (
        VerifiedCandidateError,
        prepare_verified_candidate,
    )
    from vibe.core.worktree.ephemeral import deliver_verified_ephemeral_worktree_result

    try:
        candidate = await asyncio.to_thread(prepare_verified_candidate, wt)
    except VerifiedCandidateError as exc:
        return (
            TaskOutcome(
                status=TaskOutcomeStatus.RETRYABLE,
                summary="Structured candidate could not be frozen for validation",
                diagnostics=[str(exc)],
                manifest=manifest,
            ),
            None,
            False,
        )

    validation = await asyncio.to_thread(
        validate_task_candidate, contract, wt.path, wt.base_sha
    )
    evidence = [
        f"{check.name}: exit {check.exit_code} ({check.duration_ms} ms)"
        for check in validation.checks
    ]
    if not validation.passed:
        return (
            TaskOutcome(
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
                manifest=manifest,
            ),
            None,
            False,
        )

    delivery = await asyncio.to_thread(
        deliver_verified_ephemeral_worktree_result,
        wt,
        expected_parent_sha=candidate.parent_head,
        expected_parent_fingerprint=candidate.parent_workspace_fingerprint,
        expected_candidate_sha=candidate.candidate_head,
        expected_candidate_fingerprint=candidate.candidate_workspace_fingerprint,
    )
    delivered = delivery.accepted
    return (
        TaskOutcome(
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
                else [
                    delivery.diagnostic or "parent repository refused exact integration"
                ]
            ),
            changed_paths=list(validation.changed_paths),
            manifest=manifest,
            candidate_delivery=delivery,
        ),
        delivery,
        delivered,
    )


def _isolated_verification_evidence_hashes(result: IsolatedResult) -> tuple[str, ...]:
    stats = getattr(result, "stats", None)
    value = stats.get("verification_evidence_hashes") if stats is not None else None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return ()
    return tuple(value)


def _resolve_initial_task_outcome(
    args: TaskArgs,
    response: str,
    *,
    completed: bool,
    forced_status: TaskOutcomeStatus | None = None,
    diagnostic: str | None = None,
) -> TaskOutcome:
    if (
        args.brief is None
        or args.agent != BuiltinAgentName.VERIFIER
        or forced_status is not None
        or not completed
    ):
        return resolve_task_outcome(
            args.brief,
            response,
            completed=completed,
            forced_status=forced_status,
            diagnostic=diagnostic,
        )

    try:
        report = parse_verification_report(response)
    except VerificationReportError as exc:
        parse_diagnostic = f"Verifier result was not recorded: {exc}"
        return TaskOutcome(
            status=TaskOutcomeStatus.RETRYABLE,
            summary="Structured verifier report was invalid",
            diagnostics=[item for item in (diagnostic, parse_diagnostic) if item],
            manifest=args.brief.manifest,
        )

    match report.verdict:
        case VerificationVerdict.PASS:
            status = TaskOutcomeStatus.SUCCEEDED
            summary = "Verifier reported that checks passed"
        case VerificationVerdict.FAIL:
            status = TaskOutcomeStatus.FAILED
            summary = "Verifier reported that checks failed"
        case VerificationVerdict.PARTIAL:
            status = TaskOutcomeStatus.RETRYABLE
            summary = "Verifier reported partial verification"
    return TaskOutcome(
        status=status,
        summary=summary,
        diagnostics=[diagnostic] if diagnostic else [],
        manifest=args.brief.manifest,
    )


def _with_scratchpad_context(
    agent: str,
    task_text: str,
    scratchpad_dir: Path | None,
    verification_state: VerificationState | None = None,
) -> str:
    context: list[str] = []
    if agent == BuiltinAgentName.VERIFIER and verification_state is not None:
        recipe = verification_state.trusted_recipe
        topology = recipe.config.execution_topology if recipe is not None else None
        if topology is not None:
            context.append(
                "Host-provisioned evidence workspace (read-only to model tools): "
                f"{topology.evidence_workspace}\n"
                f"Assigned run ID: {topology.run_id}; runner ID: {topology.runner_id}. "
                "This path, not the scratchpad or parent prose, is the evidence "
                "authority. If required evidence is absent or inaccessible, report "
                "VERDICT: PARTIAL."
            )
    if scratchpad_dir is not None:
        guidance = (
            "It is cleaned automatically. Do not create, copy, move, link, or remove "
            "files there; leave any permitted-tool artifacts in place. It is not "
            "durable verification evidence and may not share the parent's mount "
            "namespace. If a required artifact is inaccessible, report PARTIAL."
            if agent == BuiltinAgentName.VERIFIER
            else (
                "Tools permitted for your profile may use it without additional path "
                "permission prompts; do not assume unavailable write tools."
            )
        )
        context.append(f"Scratchpad directory: {scratchpad_dir}\n{guidance}")
    if not context:
        return task_text
    return f"{'\n\n'.join(context)}\n\n{task_text}"


class TaskArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task: str | TaskBrief = Field(
        description=(
            "The task to delegate. Pass a string for legacy free-form delegation "
            "or a TaskBrief object for recipe-bound metadata and explicit outcomes. "
            "Do not JSON-encode a TaskBrief into the string form. Structured tasks "
            "require a trusted verification recipe; a structured verifier uses the "
            "canonical verify@1 manifest and reports a terminal VERDICT."
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
            return compile_task_brief(
                self.task, verifier=self.agent == BuiltinAgentName.VERIFIER
            )
        return self.task

    @property
    def summary(self) -> str:
        if isinstance(self.task, TaskBrief):
            return self.task.objective
        return self.task


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
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
    response: str = Field(
        description=(
            "Untrusted accumulated prose from the subagent. Authoritative "
            "completed and outcome fields take precedence."
        )
    )
    turns_used: int | None = Field(
        default=None,
        description=(
            "Number of turns the subagent used. None when unknown (isolated "
            "subagents run in a subprocess that does not report turn count)."
        ),
    )
    isolated: bool = Field(
        default=False, description="Whether the subagent ran in an isolated worktree."
    )
    worktree_path: str | None = Field(
        default=None,
        description=(
            "Original isolated worktree path when candidate work could not be "
            "delivered. The directory may already be reclaimed; the branch and "
            "candidate SHA in candidate_delivery remain the recovery authority."
        ),
    )
    branch: str | None = Field(
        default=None,
        description=(
            "Preserved isolated candidate branch. It is not part of the parent "
            "workspace until integrated."
        ),
    )
    candidate_delivery: CandidateDelivery | None = Field(
        default=None,
        description=(
            "Host-observed candidate integration state, including the base, "
            "candidate and parent SHAs. A preserved candidate is not part of "
            "the parent workspace even when worker execution completed."
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
        "return an explicit terminal outcome. Pass the brief as an object, not a "
        "JSON string. Structured tasks require a trusted recipe; structured "
        "verifiers require the canonical verify@1 manifest and return VERDICT, "
        "while read-only profiles reject write-capable manifests.\n\n"
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
        brief = getattr(args, "brief", None)
        if isinstance(brief, TaskBrief):
            try:
                manifest = resolve_task_manifest(brief.manifest)
            except TaskManifestError:
                return False
            if manifest.write_capable_tools:
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
        except asyncio.CancelledError:
            _ensure_verifier_attempt_terminal(
                args.agent,
                ctx.verification_state,
                verification_attempt,
                "Verifier result was not recorded: background task was cancelled",
            )
            raise
        except Exception as e:
            response = f"[Isolated subagent error: {e}]"
            forced_status, diagnostic = _subagent_error_outcome(e)
            _ensure_verifier_attempt_terminal(
                args.agent,
                ctx.verification_state,
                verification_attempt,
                f"Verifier result was not recorded: isolated task failed: {diagnostic}",
            )
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
        preliminary = _resolve_initial_task_outcome(args, response, completed=completed)
        if contract is None:
            candidate_delivery = getattr(result, "candidate_delivery", None)
            if (
                candidate_delivery is None
                and not getattr(result, "delivered", False)
                and result.branch is not None
            ):
                candidate_delivery = CandidateDelivery(
                    status=CandidateDeliveryStatus.PRESERVED,
                    branch=result.branch,
                    worktree_path=result.worktree_path,
                    diagnostic="candidate was not integrated into the parent workspace",
                )
            if candidate_delivery is not None:
                preliminary = preliminary.model_copy(
                    update={"candidate_delivery": candidate_delivery}
                )
            if (
                preliminary.succeeded
                and candidate_delivery is not None
                and candidate_delivery.preserved
            ):
                remaining_work = list(preliminary.remaining_work)
                if candidate_delivery.branch is not None:
                    remaining_work.append(
                        f"Integrate candidate branch {candidate_delivery.branch}"
                    )
                preliminary = preliminary.model_copy(
                    update={
                        "status": TaskOutcomeStatus.RETRYABLE,
                        "summary": "Worker completed but candidate requires integration",
                        "diagnostics": [
                            *preliminary.diagnostics,
                            candidate_delivery.diagnostic
                            or "candidate was preserved outside the parent workspace",
                        ],
                        "remaining_work": remaining_work,
                    }
                )
            preliminary = _with_verification_result(
                preliminary,
                args.agent,
                _maybe_record_verifier_pass(
                    args.agent,
                    response,
                    ctx,
                    completed=completed,
                    authorized=preliminary.succeeded,
                    attempt=verification_attempt,
                    evidence_hashes=_isolated_verification_evidence_hashes(result),
                ),
            )
            return _InProcessResult(
                output=response,
                returncode=result.returncode,
                worktree_path=result.worktree_path,
                branch=result.branch,
                candidate_delivery=candidate_delivery,
                outcome=preliminary,
            )

        wt = result.wt
        if wt is None:
            outcome = preliminary
            if preliminary.succeeded:
                outcome = TaskOutcome(
                    status=TaskOutcomeStatus.RETRYABLE,
                    summary="Structured isolated task could not be validated",
                    diagnostics=["isolated worker returned no candidate worktree"],
                    manifest=args.brief.manifest if args.brief else None,
                )
            outcome = _with_verification_result(
                outcome,
                args.agent,
                _maybe_record_verifier_pass(
                    args.agent,
                    response,
                    ctx,
                    completed=completed,
                    authorized=outcome.succeeded,
                    attempt=verification_attempt,
                    evidence_hashes=_isolated_verification_evidence_hashes(result),
                ),
            )
            return _InProcessResult(
                output=response,
                returncode=0 if outcome.succeeded else 1,
                outcome=outcome,
            )

        from vibe.core.worktree.ephemeral import (
            describe_ephemeral_worktree,
            remove_ephemeral_worktree,
        )

        delivered = False
        branch: str | None = None
        candidate_delivery: CandidateDelivery | None = None
        outcome = preliminary
        try:
            if preliminary.succeeded:
                (
                    outcome,
                    candidate_delivery,
                    delivered,
                ) = await _validated_structured_candidate_outcome(
                    contract, wt, args.brief.manifest if args.brief else None
                )
        finally:
            if candidate_delivery is None:
                candidate_delivery = describe_ephemeral_worktree(
                    wt,
                    status=CandidateDeliveryStatus.PRESERVED,
                    diagnostic="candidate was not eligible for automatic integration",
                )
            if outcome.candidate_delivery is None:
                outcome = outcome.model_copy(
                    update={"candidate_delivery": candidate_delivery}
                )
            removed = await asyncio.to_thread(
                remove_ephemeral_worktree, wt, keep_if_changed=not delivered
            )
            if not removed:
                branch = candidate_delivery.branch or wt.branch
                if not delivered and candidate_delivery.branch == wt.branch:
                    candidate_delivery = describe_ephemeral_worktree(
                        wt,
                        status=CandidateDeliveryStatus.PRESERVED,
                        parent_sha_before=candidate_delivery.parent_sha_before,
                        diagnostic=(
                            candidate_delivery.diagnostic
                            or "candidate remains available for integration"
                        ),
                    )
                    outcome = outcome.model_copy(
                        update={"candidate_delivery": candidate_delivery}
                    )

        outcome = _with_verification_result(
            outcome,
            args.agent,
            _maybe_record_verifier_pass(
                args.agent,
                response,
                ctx,
                completed=completed,
                authorized=outcome.succeeded,
                attempt=verification_attempt,
                evidence_hashes=_isolated_verification_evidence_hashes(result),
            ),
        )
        return _InProcessResult(
            output=response,
            returncode=0 if outcome.succeeded else 1,
            worktree_path=candidate_delivery.worktree_path,
            branch=branch,
            candidate_delivery=candidate_delivery,
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
        if registry is None or not registry.supports_async_agent_delivery:
            # No registry wired (e.g. tests, programmatic runner without TUI).
            # Fall back to the blocking isolated path rather than failing hard.
            async for result in self._run_isolated(
                args, ctx, verification_attempt=verification_attempt
            ):
                yield result
            return

        task_text = _with_scratchpad_context(
            args.agent, args.prompt, ctx.scratchpad_dir, ctx.verification_state
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
        task_text = _with_scratchpad_context(
            args.agent, args.prompt, ctx.scratchpad_dir, ctx.verification_state
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
        candidate_delivery: CandidateDelivery | None = None
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
                candidate_delivery = finalized.candidate_delivery
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
            candidate_delivery=candidate_delivery,
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
        autoapprove = bool(
            ctx.agent_manager and ctx.agent_manager.config.bypass_tool_permissions
        )
        unavailable_reason = "configured safety judge is unavailable"
        try:
            judge = factory()
        except Exception as exc:
            judge = None
            unavailable_reason = f"{unavailable_reason}: {exc}"
        if judge is None:
            return unavailable_reason if autoapprove else None
        verdict = await judge.judge(
            "task", prompt, [f"isolated '{agent}' subagent spawn"]
        )
        if verdict.safe:
            return None
        if autoapprove:
            return verdict.reason
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
        managed_session = _validate_managed_task_agent(
            ctx, agent_name=args.agent, agent_profile=agent_profile
        )

        if args.brief is not None:
            contract = self._bind_contract(args.brief, ctx)
            self._validate_contract_for_agent(
                contract, agent=args.agent, agent_profile=agent_profile
            )

        if args.model is not None:
            valid_aliases = {m.alias for m in agent_manager.config.models}
            if args.model not in valid_aliases:
                raise ToolError(
                    f"Unknown model alias '{args.model}'. Configured aliases: "
                    f"{', '.join(sorted(valid_aliases))}."
                )

        isolation_mode = self.config.isolation
        should_isolate = not managed_session and (
            isolation_mode == "always"
            or (isolation_mode == "auto" and profile_requires_isolation(agent_profile))
        )
        verification_attempt = _start_verification_attempt(
            args.agent, ctx.verification_state
        )
        background_handoff = False
        try:
            if args.async_run:
                dispatch = (
                    self._run_async_isolated(
                        args, ctx, verification_attempt=verification_attempt
                    )
                    if should_isolate
                    else self._run_in_process_async(
                        args, ctx, verification_attempt=verification_attempt
                    )
                )
            elif should_isolate:
                dispatch = self._run_isolated(
                    args, ctx, verification_attempt=verification_attempt
                )
            else:
                dispatch = self._run_in_process(
                    args, ctx, verification_attempt=verification_attempt
                )
            async for result in dispatch:
                if isinstance(result, TaskResult) and result.task_id is not None:
                    background_handoff = True
                yield result
        finally:
            if not background_handoff:
                _ensure_verifier_attempt_terminal(
                    args.agent,
                    ctx.verification_state,
                    verification_attempt,
                    "Verifier result was not recorded: execution ended without a "
                    "terminal verifier disposition",
                )

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

    @staticmethod
    def _validate_contract_for_agent(
        contract: BoundTaskContract, *, agent: str, agent_profile: AgentProfile
    ) -> None:
        try:
            validate_task_manifest_for_agent(
                contract.manifest,
                agent=agent,
                read_only=not profile_requires_isolation(agent_profile),
            )
        except TaskManifestError as exc:
            raise ToolError(str(exc)) from exc

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
        load_overrides: dict[str, Any] = {}
        if inherited_model:
            load_overrides["active_model"] = inherited_model
        state = ctx.verification_state
        recipe = state.trusted_recipe if state is not None else None
        if recipe is not None:
            load_overrides["trusted_verification_recipe"] = recipe.config
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
        task_text = _with_scratchpad_context(
            args.agent, args.prompt, ctx.scratchpad_dir, ctx.verification_state
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
        verifier_tools_valid = True
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
                        if args.agent == BuiltinAgentName.VERIFIER and (
                            event.skipped or event.error or event.cancelled
                        ):
                            verifier_tools_valid = False
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
                completed=completed,
                authorized=outcome.succeeded and verifier_tools_valid,
                attempt=verification_attempt,
                evidence_hashes=getattr(
                    subagent_loop, "successful_verification_evidence_hashes", ()
                ),
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
        outcome = _resolve_initial_task_outcome(
            args,
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
        subagent_loop: AgentLoop | None = None
        accumulated_response: list[str] = []
        completed = True
        verifier_tools_valid = True
        turns = 0
        forced_status: TaskOutcomeStatus | None = None
        diagnostic: str | None = None
        try:
            subagent_loop, task_text = self._build_subagent_loop(args, ctx)
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
                    elif isinstance(event, ToolResultEvent):
                        if event.skipped:
                            completed = False
                        if args.agent == BuiltinAgentName.VERIFIER and (
                            event.skipped or event.error or event.cancelled
                        ):
                            verifier_tools_valid = False
        except asyncio.CancelledError:
            _ensure_verifier_attempt_terminal(
                args.agent,
                ctx.verification_state,
                verification_attempt,
                "Verifier result was not recorded: background task was cancelled",
            )
            raise
        except Exception as e:
            completed = False
            accumulated_response.append(f"\n[Subagent error: {e}]")
            forced_status, diagnostic = _subagent_error_outcome(e)
        finally:
            if subagent_loop is not None:
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
                completed=completed,
                authorized=outcome.succeeded and verifier_tools_valid,
                attempt=verification_attempt,
                evidence_hashes=getattr(
                    subagent_loop, "successful_verification_evidence_hashes", ()
                ),
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
        if registry is None or not registry.supports_async_agent_delivery:
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
