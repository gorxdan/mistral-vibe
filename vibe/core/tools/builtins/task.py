from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
import fnmatch
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import (
    AgentType,
    BuiltinAgentName,
    profile_requires_isolation,
)
from vibe.core.config import SessionLoggingConfig, VibeConfig
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


class TaskArgs(BaseModel):
    task: str = Field(description="The task to delegate to the subagent")
    agent: str = Field(
        default="explore",
        description="Name of the agent profile to use (must be a subagent)",
    )


class TaskResult(BaseModel):
    response: str = Field(description="The accumulated response from the subagent")
    turns_used: int = Field(description="Number of turns the subagent used")
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
        "(worker/auto-approve/editor) run in an isolated git worktree so their "
        "edits can't race the parent tree; read-only profiles run in-memory for "
        "speed. Override with the task.isolation config (off|auto|always)."
    )

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
            result = await run_isolated_agent(
                task_text,
                args.agent,
                label=args.agent,
                max_turns=DEFAULT_ISOLATED_MAX_TURNS,
                deliver=True,
            )
            response_text = result.output
            worktree_path = result.worktree_path
            branch = result.branch
        except Exception as e:
            completed = False
            response_text = f"[Isolated subagent error: {e}]"

        yield TaskResult(
            response=response_text,
            turns_used=0,
            completed=completed,
            isolated=True,
            worktree_path=worktree_path,
            branch=branch,
        )

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

        isolation_mode = self.config.isolation
        should_isolate = isolation_mode == "always" or (
            isolation_mode == "auto" and profile_requires_isolation(agent_profile)
        )
        if should_isolate:
            async for result in self._run_isolated(args, ctx):
                yield result
            return

        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx.session_dir else "",
            session_prefix=args.agent,
            enabled=ctx.session_dir is not None,
        )
        base_config = VibeConfig.load(session_logging=session_logging)
        # Subagents inherit the parent worktree; never call worktree_manager.enter().
        subagent_loop = AgentLoop(
            config=base_config,
            agent_name=args.agent,
            entrypoint_metadata=ctx.entrypoint_metadata,
            terminal_emulator=ctx.terminal_emulator,
            is_subagent=True,
            defer_heavy_init=True,
            permission_store=ctx.permission_store,
            hook_config_result=ctx.hook_config_result,
        )
        if ctx.session_id:
            subagent_loop.parent_session_id = ctx.session_id

        if ctx and ctx.approval_callback:
            subagent_loop.set_approval_callback(ctx.approval_callback)

        task_text = args.task
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{args.task}"
            )

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
                msg.role == Role.assistant for msg in subagent_loop.messages
            )

        except Exception as e:
            completed = False
            accumulated_response.append(f"\n[Subagent error: {e}]")
            turns_used = sum(
                msg.role == Role.assistant for msg in subagent_loop.messages
            )

        yield TaskResult(
            response="".join(accumulated_response),
            turns_used=turns_used,
            completed=completed,
        )
