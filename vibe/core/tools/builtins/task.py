from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from dataclasses import dataclass
import fnmatch
from pathlib import Path
import time
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import (
    AgentType,
    BuiltinAgentName,
    profile_requires_isolation,
)
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.logger import logger
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
from vibe.core.workflows.runtime import DEFAULT_ISOLATED_MAX_TURNS, run_isolated_agent


def _configured_subagent_model(ctx: InvokeContext) -> str | None:
    if ctx.agent_manager and ctx.agent_manager.config.subagent_model:
        return ctx.agent_manager.config.subagent_model
    return None


@dataclass
class _InProcessResult:
    # IsolatedResult-shaped result for a backgrounded in-process subagent, so the
    # registry finalizer reads .output/.returncode like the isolated path.
    output: str
    returncode: int
    worktree_path: str | None = None
    branch: str | None = None


class TaskArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task: str = Field(description="The task to delegate to the subagent")
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
    completed: bool = Field(description="Whether the task completed normally")
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
    # "auto" isolates write-capable profiles (worker/auto-approve/editor and any
    # profile with write_file/edit or un-jailed bash) in their own worktree;
    # read-only and read-jailed profiles stay in-process. "always" isolates
    # every profile. "off" forces in-process (the historical behavior).
    isolation: Literal["off", "auto", "always"] = "auto"


class Task(
    BaseTool[TaskArgs, TaskResult, TaskToolConfig, BaseToolState],
    ToolUIData[TaskArgs, TaskResult],
):
    description: ClassVar[str] = (
        "Delegate a task to a subagent for independent execution. "
        "Useful for exploration, research, or parallel work that doesn't "
        "require user interaction. By default write-capable profiles "
        "(worker/auto-approve/editor) run in an isolated git worktree — "
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
        if getattr(args, "async_run", False) or agent_manager is None:
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
            return ToolCallDisplay(summary=f"Running {args.agent} agent: {args.task}")
        return ToolCallDisplay(summary="Running subagent")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, TaskResult):
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

    async def _run_async_isolated(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        # Non-blocking variant of _run_isolated: spawn the isolated subagent as
        # an asyncio.Task, register it with the background registry, and return
        # immediately with the task_id. The registry's finalizer captures the
        # IsolatedResult; the parent agent loop drains queued completions at the
        # top of each turn and emits BackgroundTaskCompletedEvent for each.
        registry = ctx.background_registry
        if registry is None:
            # No registry wired (e.g. tests, programmatic runner without TUI).
            # Fall back to the blocking isolated path rather than failing hard.
            async for result in self._run_isolated(args, ctx):
                yield result
            return

        task_text = args.task
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{args.task}"
            )

        denied = await self._judge_isolated_spawn(task_text, args.agent, ctx)
        if denied is not None:
            yield TaskResult(
                response=f"[Isolated subagent denied by safety judge: {denied}]",
                completed=False,
                isolated=True,
            )
            return

        # Stream the subprocess stdout to a log file so the Tasks pane and
        # `background` tool can tail live progress. Mirrors the bash tool's bg
        # log layout (scratchpad/bg/). None when there's nowhere to write — the
        # runtime then captures stdout in memory as before (no live tail).
        log_path = self._bg_log_path(ctx)
        if log_path is not None:
            log_path.touch()

        effective_model = (
            args.model
            or _configured_subagent_model(ctx)
            or ctx.active_model
            or (ctx.agent_manager.config.active_model if ctx.agent_manager else None)
        )
        bg_task = asyncio.create_task(
            run_isolated_agent(
                task_text,
                args.agent,
                label=args.agent,
                max_turns=DEFAULT_ISOLATED_MAX_TURNS,
                deliver=True,
                model=effective_model,
                log_path=log_path,
            ),
            name=f"async-task-{args.agent}",
        )
        task_id = registry.register_async_agent(
            args.agent,
            bg_task,
            label=self._subagent_label(args),
            prompt=args.task,
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
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        # Run the subagent as a `vibe -p` subprocess in its own git worktree so
        # its writes can't race the parent tree or siblings. The branch is
        # ff-merged back on success (deliver=True) so delegated edits land, the
        # way they do in the in-process path. On non-delivery the worktree is
        # kept and surfaced via TaskResult.worktree_path/.branch for recovery.
        task_text = args.task
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{args.task}"
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
        try:
            denied = await self._judge_isolated_spawn(task_text, args.agent, ctx)
            if denied is not None:
                # Judge denied the delegation (or user declined at the approval
                # prompt). Fail the TaskResult cleanly rather than raising so the
                # tool surfaces the denial in-band; no subprocess is spawned.
                completed = False
                response_text = f"[Isolated subagent denied by safety judge: {denied}]"
            else:
                result = await run_isolated_agent(
                    task_text,
                    args.agent,
                    label=args.agent,
                    max_turns=DEFAULT_ISOLATED_MAX_TURNS,
                    deliver=True,
                    # Inherit the parent's effective model (not the configured default).
                    model=args.model
                    or _configured_subagent_model(ctx)
                    or ctx.active_model
                    or (
                        ctx.agent_manager.config.active_model
                        if ctx.agent_manager
                        else None
                    ),
                )
                response_text = result.output
                worktree_path = result.worktree_path
                branch = result.branch
        except Exception as e:
            completed = False
            response_text = f"[Isolated subagent error: {e}]"

        yield TaskResult(
            response=response_text,
            turns_used=None,  # isolated subprocess doesn't report turn count
            completed=completed,
            isolated=True,
            worktree_path=worktree_path,
            branch=branch,
        )

    async def _judge_isolated_spawn(
        self, prompt: str, agent: str, ctx: InvokeContext
    ) -> str | None:
        """Pre-flight safety judge for an isolated subagent spawn.

        Isolated subagents run as an auto-approved ``vibe -p`` subprocess, so
        the host's per-tool judge never sees their calls. This judges the
        subagent's *prompt* (the task the lead gave it) before the subprocess
        starts. Mirrors ``WorkflowRuntime._judge_isolated_spawn``.

        Returns ``None`` to proceed; returns the judge's (or user's) denial
        reason to skip the spawn. Fail-open when no judge is configured or the
        judge is unusable — the launch-time script/CLI judge already ran.
        """
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
        if args.async_run:
            if should_isolate:
                async for result in self._run_async_isolated(args, ctx):
                    yield result
            else:
                async for result in self._run_in_process_async(args, ctx):
                    yield result
            return
        if should_isolate:
            async for result in self._run_isolated(args, ctx):
                yield result
            return
        async for result in self._run_in_process(args, ctx):
            yield result

    @staticmethod
    def _subagent_label(args: TaskArgs) -> str:
        snippet = args.task.strip().split("\n", 1)[0][:60]
        return f"{args.agent}: {snippet}" if snippet else args.agent

    def _build_subagent_loop(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> tuple[AgentLoop, str]:
        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx.session_dir else "",
            session_prefix=args.agent,
            enabled=ctx.session_dir is not None,
        )
        # A fresh VibeConfig.load() falls back to the hardcoded default (mistral),
        # which fails when the parent runs on another provider; inherit instead.
        inherited_model = (
            args.model
            or _configured_subagent_model(ctx)
            or ctx.active_model
            or (ctx.agent_manager.config.active_model if ctx.agent_manager else None)
        )
        load_overrides: dict[str, str] = {}
        if inherited_model:
            load_overrides["active_model"] = inherited_model
        base_config = VibeConfig.load(session_logging=session_logging, **load_overrides)
        try:
            resolved_provider = base_config.get_active_provider().name
        except Exception as e:
            resolved_provider = repr(e)
        logger.warning(
            "subagent model resolve: agent=%s args_model=%s subagent_model=%s"
            " ctx_active=%s has_mgr=%s -> inherited=%s loaded_active=%s provider=%s",
            args.agent,
            args.model,
            _configured_subagent_model(ctx),
            ctx.active_model,
            ctx.agent_manager is not None,
            inherited_model,
            base_config.active_model,
            resolved_provider,
        )
        # Subagents inherit the parent worktree; never call worktree_manager.enter().
        subagent_loop = AgentLoop(
            config=base_config,
            agent_name=args.agent,
            entrypoint_metadata=ctx.entrypoint_metadata,
            terminal_emulator=ctx.terminal_emulator,
            is_subagent=True,
            # Stream like the host: reasoning models (k2.7-code/GLM) need stream=True.
            enable_streaming=True,
            defer_heavy_init=True,
            permission_store=ctx.permission_store,
            hook_config_result=ctx.hook_config_result,
            max_turns=DEFAULT_ISOLATED_MAX_TURNS,
        )
        if ctx.session_id:
            subagent_loop.parent_session_id = ctx.session_id
        if ctx.approval_callback:
            subagent_loop.set_approval_callback(ctx.approval_callback)
        task_text = args.task
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{args.task}"
            )
        return subagent_loop, task_text

    async def _run_in_process(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        subagent_loop, task_text = self._build_subagent_loop(args, ctx)
        accumulated_response: list[str] = []
        completed = True
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
            turns_used = sum(
                msg.role == Role.ASSISTANT for msg in subagent_loop.messages
            )
        finally:
            with suppress(Exception):
                await subagent_loop.aclose()

        yield TaskResult(
            response="".join(accumulated_response),
            turns_used=turns_used,
            completed=completed,
        )

    async def _run_in_process_collect(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> _InProcessResult:
        # Background variant: drive to completion, return an IsolatedResult-shaped
        # object (no parent yields — the parent turn already returned). Pushes
        # live progress (partial response + turn count) to the registry so the
        # Tasks pane and `background` tool reflect streaming activity.
        registry = ctx.background_registry
        current_task = asyncio.current_task()
        subagent_loop, task_text = self._build_subagent_loop(args, ctx)
        accumulated_response: list[str] = []
        completed = True
        turns = 0
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
        finally:
            with suppress(Exception):
                await subagent_loop.aclose()
        return _InProcessResult(
            output="".join(accumulated_response), returncode=0 if completed else 1
        )

    async def _run_in_process_async(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        registry = ctx.background_registry
        if registry is None:
            async for result in self._run_in_process(args, ctx):
                yield result
            return
        effective_model = (
            args.model
            or _configured_subagent_model(ctx)
            or ctx.active_model
            or (ctx.agent_manager.config.active_model if ctx.agent_manager else None)
        )
        bg_task = asyncio.create_task(
            self._run_in_process_collect(args, ctx), name=f"async-task-{args.agent}"
        )
        task_id = registry.register_async_agent(
            args.agent,
            bg_task,
            label=self._subagent_label(args),
            prompt=args.task,
            model=effective_model,
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
        """A unique log file path for an isolated subagent's stdout, or None.

        Mirrors the bash tool's background-log layout (scratchpad/bg/, falling
        back to the session dir) so the Tasks pane can tail live progress. The
        caller is responsible for ``touch()`` before handing the path to the
        subprocess. Returns None when neither dir is available — the runtime
        then falls back to an in-memory PIPE capture.
        """
        root = ctx.scratchpad_dir or ctx.session_dir
        if root is None:
            return None
        bg_dir = Path(str(root)) / "bg"
        bg_dir.mkdir(parents=True, exist_ok=True)
        return bg_dir / f"asub-{time.monotonic_ns()}.log"
