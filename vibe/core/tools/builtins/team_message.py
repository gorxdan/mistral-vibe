from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

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
from vibe.core.types import ToolResultEvent, ToolStreamEvent

# The lead's mailbox identity. Teammates reply to "lead" to reach the host.
# Unlike the teammate `team` tool (which binds identity to VIBE_TEAMMATE_NAME to
# prevent a prompt-injected teammate from spoofing others), the lead IS the
# privileged orchestrator, so a fixed name is correct — it is not a spoof.
_LEAD_NAME = "lead"


def _resolve_team_dir(ctx: InvokeContext) -> Path:
    if ctx.team_dir_callback is None:
        raise ToolError(
            "Team messaging is not available in this context (no team dir callback)."
        )
    raw = ctx.team_dir_callback()
    if not raw:
        raise ToolError(
            "No active team. Spawn a teammate first with /team spawn "
            "<name> <prompt>."
        )
    return Path(raw)


class TeamMessageArgs(BaseModel):
    action: str = Field(
        description=(
            "One of: send_message, read_messages, unread_messages. The lead "
            "sends TO a teammate by name; reads its own ('lead') inbox."
        )
    )
    to_name: str | None = Field(
        default=None, description="Recipient teammate name for send_message."
    )
    content: str | None = Field(
        default=None, description="Message body for send_message."
    )


class TeamMessageResult(BaseModel):
    action: str
    message: str
    messages: list[dict[str, Any]] | None = None


class TeamMessageConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class TeamMessage(
    BaseTool[TeamMessageArgs, TeamMessageResult, TeamMessageConfig, BaseToolState],
    ToolUIData[TeamMessageArgs, TeamMessageResult],
):
    description: ClassVar[str] = (
        "Send messages to teammates and read the lead's inbox from the shared "
        "team Mailbox. This is how the host communicates with spawned teammates "
        "(the teammate-facing `team` tool is unavailable to the lead). Use "
        "send_message to direct a teammate, read_messages to collect replies "
        "addressed to the lead. Requires an active team (/team spawn)."
    )

    @classmethod
    def is_available(cls, config: object | None = None) -> bool:
        # Always discoverable; availability is decided per-invocation by whether
        # a team is active (team_dir_callback returns a path). Mirrors how the
        # teammate `team` tool gates on VIBE_TEAM_DIR at call time.
        return True

    @classmethod
    def format_call_display(cls, args: TeamMessageArgs) -> ToolCallDisplay:
        if args.action == "send_message" and args.to_name:
            return ToolCallDisplay(summary=f"team message -> {args.to_name}")
        return ToolCallDisplay(summary=f"team {args.action}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, TeamMessageResult):
            return ToolResultDisplay(success=True, message=event.result.message)
        return ToolResultDisplay(success=True, message="Success")

    @classmethod
    def get_status_text(cls) -> str:
        return "Team messaging"

    def resolve_permission(
        self, args: TeamMessageArgs
    ) -> PermissionContext | None:
        # Sending a message steers a (trusted, auto-approved) teammate, so ask.
        # Reads are side-effect-free but routing through the same permission
        # keeps the surface simple; the host can approve both.
        return PermissionContext(permission=ToolPermission.ASK)

    async def run(
        self, args: TeamMessageArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TeamMessageResult, None]:
        if ctx is None:
            raise ToolError("Team message tool requires context")

        from vibe.core.teams.mailbox import Mailbox

        team_dir = _resolve_team_dir(ctx)
        mailbox = Mailbox(team_dir)

        match args.action:
            case "send_message":
                if not args.to_name or args.content is None:
                    raise ToolError(
                        "to_name and content are required for send_message."
                    )
                msg = mailbox.send(_LEAD_NAME, args.to_name, args.content)
                yield TeamMessageResult(
                    action=args.action,
                    message=f"Sent message to {args.to_name}.",
                    messages=[msg.model_dump(mode="json")],
                )
            case "read_messages":
                msgs = mailbox.read(_LEAD_NAME, mark_read=True)
                yield TeamMessageResult(
                    action=args.action,
                    message=f"{len(msgs)} message(s) in lead inbox.",
                    messages=[m.model_dump(mode="json") for m in msgs],
                )
            case "unread_messages":
                msgs = mailbox.get_unread(_LEAD_NAME)
                yield TeamMessageResult(
                    action=args.action,
                    message=f"{len(msgs)} unread message(s) in lead inbox.",
                    messages=[m.model_dump(mode="json") for m in msgs],
                )
            case _:
                raise ToolError(
                    f"Unknown action '{args.action}'. Use send_message, "
                    "read_messages, or unread_messages."
                )
