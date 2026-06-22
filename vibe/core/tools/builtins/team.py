from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

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

if TYPE_CHECKING:
    from vibe.core.teams.mailbox import Mailbox
    from vibe.core.teams.task_store import TaskStore


def _team_dir() -> Path | None:
    """Resolve the shared team directory from the teammate environment.

    TeamManager.spawn_teammate sets VIBE_TEAM_DIR when launching a `vibe -p`
    child. The teammate reads it here to bind the shared TaskStore/Mailbox.
    Returns None when not running as a teammate (the tool is then unavailable).
    """
    raw = os.environ.get("VIBE_TEAM_DIR")
    if not raw:
        return None
    return Path(raw)


class TeamArgs(BaseModel):
    action: str = Field(
        description=(
            "One of: list_tasks, available_tasks, claim_task, complete_task, "
            "send_message, read_messages, unread_messages."
        )
    )
    task_id: str | None = Field(default=None, description="Task id for claim/complete.")
    description: str | None = Field(
        default=None, description="Result text for complete_task."
    )
    to_name: str | None = Field(default=None, description="Recipient for send_message.")
    content: str | None = Field(
        default=None, description="Message body for send_message."
    )
    mark_read: bool = Field(
        default=True, description="Mark messages read on read_messages."
    )


class TeamResult(BaseModel):
    action: str
    message: str
    tasks: list[dict] | None = None
    messages: list[dict] | None = None
    task: dict | None = None


class TeamToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class TeamState(BaseToolState):
    pass


class Team(
    BaseTool[TeamArgs, TeamResult, TeamToolConfig, TeamState],
    ToolUIData[TeamArgs, TeamResult],
):
    description: ClassVar[str] = (
        "Interact with the shared team TaskStore and Mailbox. Only available "
        "inside a teammate (VIBE_TEAM_DIR set). Use list_tasks/available_tasks/"
        "claim_task/complete_task for work distribution and send_message/"
        "read_messages/unread_messages for inter-teammate messaging."
    )

    @classmethod
    def is_available(cls, config: object | None = None) -> bool:
        # Only expose the team tool to teammate processes spawned with a
        # shared team dir. The lead does not get it (it uses /team instead).
        return _team_dir() is not None

    @classmethod
    def format_call_display(cls, args: TeamArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"team {args.action}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, TeamResult):
            return ToolResultDisplay(success=True, message=event.result.message)
        return ToolResultDisplay(success=True, message="Success")

    @classmethod
    def get_status_text(cls) -> str:
        return "Team coordination"

    def _bind(self) -> tuple[TaskStore, Mailbox]:
        from vibe.core.teams.mailbox import Mailbox
        from vibe.core.teams.task_store import TaskStore

        team_dir = _team_dir()
        if team_dir is None:
            raise ToolError("Not running as a teammate (VIBE_TEAM_DIR is unset).")
        return TaskStore(team_dir), Mailbox(team_dir)

    @staticmethod
    def _self_name() -> str:
        # Identity is bound to the spawning environment, never to model-supplied
        # args: a teammate may only act AS ITSELF. Honouring a caller-supplied
        # name would let a (prompt-injectable, auto-approved) teammate spoof the
        # sender of a message, claim/complete tasks as someone else, or read and
        # mark-read another teammate's inbox.
        name = os.environ.get("VIBE_TEAMMATE_NAME")
        if not name:
            raise ToolError(
                "Cannot determine teammate identity (VIBE_TEAMMATE_NAME is unset)."
            )
        return name

    async def run(
        self, args: TeamArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TeamResult, None]:
        task_store, mailbox = self._bind()

        match args.action:
            case "list_tasks":
                tasks = [t.model_dump(mode="json") for t in task_store.get_all_tasks()]
                yield TeamResult(
                    action=args.action, message=f"{len(tasks)} task(s).", tasks=tasks
                )
            case "available_tasks":
                tasks = [
                    t.model_dump(mode="json") for t in task_store.get_available_tasks()
                ]
                yield TeamResult(
                    action=args.action,
                    message=f"{len(tasks)} available task(s).",
                    tasks=tasks,
                )
            case "claim_task":
                if not args.task_id:
                    raise ToolError("task_id is required for claim_task.")
                assignee = self._self_name()
                task = task_store.claim_task(args.task_id, assignee)
                if task is None:
                    raise ToolError(
                        f"Could not claim task {args.task_id} (missing, already "
                        "claimed, or dependencies unmet)."
                    )
                yield TeamResult(
                    action=args.action,
                    message=f"Claimed task {task.id}.",
                    task=task.model_dump(mode="json"),
                )
            case "complete_task":
                if not args.task_id:
                    raise ToolError("task_id is required for complete_task.")
                actor = self._self_name()
                task = task_store.complete_task(
                    args.task_id, args.description, actor=actor
                )
                if task is None:
                    raise ToolError(
                        f"Could not complete task {args.task_id} (missing, or not "
                        f"claimed by {actor})."
                    )
                yield TeamResult(
                    action=args.action,
                    message=f"Completed task {task.id}.",
                    task=task.model_dump(mode="json"),
                )
            case "send_message":
                if not args.to_name or args.content is None:
                    raise ToolError(
                        "to_name and content are required for send_message."
                    )
                from_name = self._self_name()
                msg = mailbox.send(from_name, args.to_name, args.content)
                yield TeamResult(
                    action=args.action,
                    message=f"Sent message to {args.to_name}.",
                    messages=[msg.model_dump(mode="json")],
                )
            case "read_messages":
                recipient = self._self_name()
                msgs = mailbox.read(recipient, mark_read=args.mark_read)
                yield TeamResult(
                    action=args.action,
                    message=f"{len(msgs)} message(s).",
                    messages=[m.model_dump(mode="json") for m in msgs],
                )
            case "unread_messages":
                recipient = self._self_name()
                msgs = mailbox.get_unread(recipient)
                yield TeamResult(
                    action=args.action,
                    message=f"{len(msgs)} unread message(s).",
                    messages=[m.model_dump(mode="json") for m in msgs],
                )
            case _:
                raise ToolError(
                    f"Unknown action '{args.action}'. See the tool description."
                )
