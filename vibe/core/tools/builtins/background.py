"""Background-task management tool.

Exposes the BackgroundRegistry to the model so the host agent is aware of — and
can cancel — anything it (or the user) has backgrounded: dev servers, watchers,
workflow runs, teammates, and schedule loops. Read-only `list` plus `stop`.

Auto-allowed (ToolPermission.ALWAYS) because it only touches processes the
session itself launched; it cannot spawn or mutate files.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.background import TaskCategory, TaskEntry
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolCallEvent, ToolResultEvent, ToolStreamEvent

# When the model asks for a single task by id (action='list' + task_id) without
# specifying a tail, show this many recent log lines — the point of a scoped
# lookup is to inspect one task, so a small recent sample is the useful default.
_DEFAULT_SCOPED_TAIL = 20


class BackgroundArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str = Field(
        description=(
            "list: enumerate running background tasks (processes, workflows, "
            "agents, teams, loops). stop: terminate one task by task_id."
        )
    )
    task_id: str | None = Field(
        default=None,
        description=(
            "Task id. For action='stop' (required): the task to terminate. "
            "For action='list' (optional): scope the listing — and any log "
            "tail — to one task by id. Matches the id exactly plus its "
            "hierarchical children (wf-1 also pulls in wf-1/live-* agents); "
            "the '/' boundary is respected, so proc-1 never matches proc-10. "
            "From the registry's id grammar: proc-N, wf-N, wf-N/live-AGENT, "
            "team:NAME, loop-LOOPID."
        ),
    )
    tail: int | None = Field(
        default=None,
        description=(
            "For action='list' only: number of trailing log lines to show for "
            "each process entry. Omitted on a full list -> no log shown; "
            "omitted with task_id set -> a small recent-output sample is "
            "shown by default. Pass 0 to suppress."
        ),
    )


class BackgroundResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: str = Field(description="The list of tasks, or the stop outcome.")
    stopped: bool = Field(
        default=False, description="For action='stop': whether the task was stopped."
    )


class BackgroundToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


def _tail_for(registry: Any, entry: TaskEntry, tail_lines: int | None) -> str | None:
    """Resolve a log tail for an entry by category. Returns None when tails are
    suppressed (tail_lines is None) or the category has nothing to tail.
    """
    if not tail_lines:
        return None
    if entry.category == TaskCategory.PROCESS:
        return registry.read_log_tail(entry.task_id, lines=tail_lines)
    if entry.category == TaskCategory.AGENT:
        return registry.read_agent_log_tail(entry.task_id, lines=tail_lines)
    if entry.category == TaskCategory.ASYNC_AGENT:
        return registry.read_async_tail(entry.task_id, lines=tail_lines)
    return None


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
        "running — pass task_id to scope it (and its log tail) to one task — "
        "and action='stop' with a task_id to cancel one."
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
        if ctx is None or ctx.background_registry is None:
            raise ToolError("background tool requires a background registry in context")
        registry = ctx.background_registry

        action = (args.action or "").strip().lower()
        if action == "list":
            all_entries = registry.list_tasks()
            scoped = bool(args.task_id)
            if scoped:
                target = args.task_id
                # Family scoping via the '/' boundary: an exact match on the
                # id, plus any hierarchical children (e.g. wf-1 pulls in
                # wf-1/live-explore, wf-1/live-reviewer). Matching on the
                # delimiter — not a raw prefix — means proc-1 never matches
                # proc-10: there is no proc-1/... child, and 'proc-10' does
                # not start with 'proc-1/'.
                entries = [
                    e
                    for e in all_entries
                    if e.task_id == target or e.task_id.startswith(f"{target}/")
                ]
                if not entries:
                    known_ids = ", ".join(e.task_id for e in all_entries) or "none"
                    yield BackgroundResult(
                        response=(
                            f"No background task with id {args.task_id!r}. "
                            f"Known ids: {known_ids}."
                        )
                    )
                    return
            else:
                entries = all_entries
                if not entries:
                    yield BackgroundResult(response="No background tasks running.")
                    return
            # Tail resolution:
            #  - explicit tail > 0 -> that many lines
            #  - explicit tail <= 0 -> none
            #  - tail omitted + scoped lookup -> default sample (recent output
            #    is the whole point of asking for one task)
            #  - tail omitted + full list -> none (keep the overview cheap)
            if args.tail is not None:
                tail_lines = args.tail if args.tail > 0 else None
            elif scoped:
                tail_lines = _DEFAULT_SCOPED_TAIL
            else:
                tail_lines = None
            lines = [
                _format_entry(e, _tail_for(registry, e, tail_lines)) for e in entries
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

        raise ToolError(f"Unknown background action: {action!r}. Use 'list' or 'stop'.")
