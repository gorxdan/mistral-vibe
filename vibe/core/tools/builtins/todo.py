from __future__ import annotations

from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolResultEvent, ToolStreamEvent


class TodoStatus(StrEnum):
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class TodoPriority(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class TodoItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    content: str
    status: TodoStatus = TodoStatus.PENDING
    priority: TodoPriority = TodoPriority.MEDIUM


class TodoArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str = Field(description="Either 'read' or 'write'")
    todos: list[TodoItem] | None = Field(
        default=None, description="Complete list of todos when writing."
    )


class TodoResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str
    todos: list[TodoItem]
    total_count: int
    verification_nudge: bool = False


class TodoConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_todos: int = 100


class TodoState(BaseToolState):
    todos: list[TodoItem] = Field(default_factory=list)


# Minimum closed-out list size that triggers the verification nudge. Below this,
# the work is small enough that independent verification is optional.
_VERIFICATION_NUDGE_MIN_TODOS = 3


def _verification_subsystem_enabled(ctx: InvokeContext | None) -> bool:
    """Read the ``verification_subsystem`` config flag. Defaults to True (the
    feature ships on) when the config isn't reachable — e.g. a bare tool call
    without an agent manager wired up.
    """
    if ctx is None or ctx.agent_manager is None:
        return True
    return bool(getattr(ctx.agent_manager.config, "verification_subsystem", True))


def _should_nudge(todos: list[TodoItem], verification_enabled: bool) -> bool:
    """Structural completion-nudge: fires when a 3+ item list is being closed
    out as all-completed and none of the items was a verification step. Catches
    the exact loop-exit moment where verification gets skipped. Gated on the
    ``verification_subsystem`` config flag.
    """
    if not verification_enabled or len(todos) < _VERIFICATION_NUDGE_MIN_TODOS:
        return False
    if not all(t.status == TodoStatus.COMPLETED for t in todos):
        return False
    return not any("verif" in t.content.lower() for t in todos)


class Todo(
    BaseTool[TodoArgs, TodoResult, TodoConfig, TodoState],
    ToolUIData[TodoArgs, TodoResult],
):
    description: ClassVar[str] = (
        "Manage todos. Use action='read' to view, action='write' with complete list to update."
    )

    @classmethod
    def format_call_display(cls, args: TodoArgs) -> ToolCallDisplay:
        match args.action:
            case "read":
                return ToolCallDisplay(summary="Reading todos")
            case "write":
                count = len(args.todos) if args.todos else 0
                return ToolCallDisplay(summary=f"Writing {count} todos")
            case _:
                return ToolCallDisplay(summary=f"Unknown action: {args.action}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, TodoResult):
            return ToolResultDisplay(success=True, message="Success")

        result = event.result

        return ToolResultDisplay(success=True, message=result.message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Managing todos"

    async def run(
        self, args: TodoArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TodoResult, None]:
        verification_enabled = _verification_subsystem_enabled(ctx)
        match args.action:
            case "read":
                yield self._read_todos()
            case "write":
                yield self._write_todos(args.todos or [], verification_enabled)
            case _:
                raise ToolError(
                    f"Invalid action '{args.action}'. Use 'read' or 'write'."
                )

    def _read_todos(self) -> TodoResult:
        return TodoResult(
            message=f"Retrieved {len(self.state.todos)} todos",
            todos=self.state.todos,
            total_count=len(self.state.todos),
        )

    def _write_todos(
        self, todos: list[TodoItem], verification_enabled: bool
    ) -> TodoResult:
        if len(todos) > self.config.max_todos:
            raise ToolError(f"Cannot store more than {self.config.max_todos} todos")

        ids = [todo.id for todo in todos]
        if len(ids) != len(set(ids)):
            raise ToolError("Todo IDs must be unique")

        self.state.todos = todos

        message = f"Updated {len(todos)} todos"
        nudge = _should_nudge(todos, verification_enabled)
        if nudge:
            message += (
                "\n\nNOTE: you closed 3+ todos and none was a verification step. "
                "Before your final summary, run independent verification — spawn "
                "the `verifier` subagent with the task, the files that changed, "
                "and the approach. You can't self-assign done by listing caveats; "
                "only a verifier issues a verdict."
            )

        return TodoResult(
            message=message,
            todos=self.state.todos,
            total_count=len(self.state.todos),
            verification_nudge=nudge,
        )
