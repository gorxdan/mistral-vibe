"""Background-task management tool.

Exposes the BackgroundRegistry to the model so the host agent is aware of — and
can cancel — anything it (or the user) has backgrounded: dev servers, watchers,
workflow runs, teammates, and schedule loops. Read-only `list` plus `stop`.

Auto-allowed (ToolPermission.ALWAYS) because it only touches processes the
session itself launched; it cannot spawn or mutate files.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.background import TaskCategory, TaskEntry
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent, ToolStreamEvent


class BackgroundArgs(BaseModel):
    action: str = Field(
        description=(
            "list: enumerate running background tasks (processes, workflows, "
            "agents, teams, loops). stop: terminate one task by task_id."
        )
    )
    task_id: str | None = Field(
        default=None,
        description=(
            "Task id to stop (required for action='stop'). From the registry's "
            "id grammar: proc-N, wf-N, wf-N/live-AGENT, team:NAME, loop-LOOPID."
        ),
    )
    tail: int | None = Field(
        default=None,
        description=(
            "For action='list' only: also return the last N lines of each "
            "process's log. Useful to check a server's startup output."
        ),
    )


class BackgroundResult(BaseModel):
    response: str = Field(description="The list of tasks, or the stop outcome.")
    stopped: bool = Field(
        default=False, description="For action='stop': whether the task was stopped."
    )


class BackgroundToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


def _format_entry(entry: TaskEntry, tail: str | None) -> str:
    elapsed = entry.elapsed
    if entry.category == TaskCategory.LOOP:
        time_field = f"fires in {_fmt_seconds(elapsed)}"
    else:
        time_field = f"{_fmt_seconds(elapsed)}"
    line = (
        f"- {entry.task_id}  [{entry.category.value}]  {entry.status}  "
        f"{time_field}  {_truncate(entry.label, 60)}"
    )
    if tail:
        snippet = tail.replace("\n", "\n    ")
        line += f"\n    ```\n    {snippet}\n    ```"
    return line


def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m{s % 60}s"
    return f"{s}s"


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"


class Background(
    BaseTool[BackgroundArgs, BackgroundResult, BackgroundToolConfig, BaseToolState],
    ToolUIData[BackgroundArgs, BackgroundResult],
):
    description: ClassVar[str] = (
        "List or stop background tasks (long-running processes you launched "
        "with bash background=true, workflow runs and their in-flight agents, "
        "teammates, and scheduled loops). Use action='list' to see what is "
        "running and action='stop' with a task_id to cancel one."
    )
    read_only: ClassVar[bool] = False  # stop() has side effects

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, BackgroundArgs):
            return ToolCallDisplay(
                summary=f"background {args.action}"
                + (f" {args.task_id}" if args.task_id else "")
            )
        return ToolCallDisplay(summary="background")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, BackgroundResult):
            if result.stopped:
                return ToolResultDisplay(success=True, message=result.response)
            return ToolResultDisplay(success=True, message="Listed background tasks")
        return ToolResultDisplay(success=True, message="background")

    @classmethod
    def get_status_text(cls) -> str:
        return "Listing background tasks"

    def resolve_permission(self, args: BackgroundArgs) -> PermissionContext | None:
        # Always allowed — operates only on tasks the session itself launched.
        return PermissionContext(permission=ToolPermission.ALWAYS)

    async def run(
        self, args: BackgroundArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | BackgroundResult, None]:
        if ctx is None or getattr(ctx, "background_registry", None) is None:
            raise ToolError(
                "background tool requires a background registry in context"
            )
        registry = ctx.background_registry

        action = (args.action or "").strip().lower()
        if action == "list":
            entries = registry.list_tasks()
            if not entries:
                yield BackgroundResult(response="No background tasks running.")
                return
            tail_lines = args.tail if args.tail and args.tail > 0 else None
            lines = [
                _format_entry(
                    e,
                    registry.read_log_tail(e.task_id, lines=tail_lines)
                    if tail_lines and e.category == TaskCategory.PROCESS
                    else None,
                )
                for e in entries
            ]
            yield BackgroundResult(
                response=f"{len(entries)} background task"
                f"{'s' if len(entries) != 1 else ''}:\n" + "\n".join(lines)
            )
            return

        if action == "stop":
            if not args.task_id:
                raise ToolError("action='stop' requires a task_id")
            stopped = await registry.stop(args.task_id)
            yield BackgroundResult(
                response=(
                    f"Stopped {args.task_id}."
                    if stopped
                    else f"Could not stop {args.task_id} — not found or already "
                    "finished."
                ),
                stopped=stopped,
            )
            return

        raise ToolError(
            f"Unknown background action: {action!r}. Use 'list' or 'stop'."
        )
