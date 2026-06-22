from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from vibe.core.teams.models import Message, MessageKind
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
            "No active team. Spawn a teammate first with /team spawn <name> <prompt>."
        )
    return Path(raw)


def _format_message(msg: Message) -> str:
    """Render a single inbox message for the lead, structured-kind aware.

    Structured kinds get a typed prefix so the lead can recognize and act on
    them; TEXT stays a plain prose line. A PERMISSION_REQUEST is framed as a
    question the lead should answer via team_message(send_message) with a
    PERMISSION_RESPONSE addressed back to the requester.
    """
    match msg.kind:
        case MessageKind.PERMISSION_REQUEST:
            tool = msg.payload.get("tool", "unknown")
            request_id = msg.payload.get("request_id", msg.id)
            description = msg.payload.get("description", msg.content)
            return (
                f"[PERMISSION_REQUEST id={request_id} from={msg.from_name} "
                f"tool={tool}] {description}\n"
                f"Reply with team_message(send_message, to_name={msg.from_name}, "
                f"kind=PERMISSION_RESPONSE, payload={{'request_id': '{request_id}', "
                f"'decision': 'allow' | 'deny'}})."
            )
        case MessageKind.PERMISSION_RESPONSE:
            decision = msg.payload.get("decision", "?")
            request_id = msg.payload.get("request_id", "?")
            reason = msg.payload.get("reason")
            tail = f" reason: {reason}" if reason else ""
            return (
                f"[PERMISSION_RESPONSE id={request_id} from={msg.from_name} "
                f"decision={decision}]{tail}"
            )
        case MessageKind.PLAN_APPROVAL:
            return f"[PLAN_APPROVAL from={msg.from_name}] {msg.content}"
        case MessageKind.SHUTDOWN:
            return f"[SHUTDOWN from={msg.from_name}] {msg.content}"
        case _:
            return f"[from {msg.from_name}] {msg.content}"


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
    kind: MessageKind = Field(
        default=MessageKind.TEXT,
        description=(
            "Message kind. TEXT is the default free-form prose. "
            "PERMISSION_RESPONSE is the lead's reply to a teammate's "
            "PERMISSION_REQUEST (payload must carry request_id + decision)."
        ),
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured payload for non-TEXT kinds. For PERMISSION_RESPONSE: "
            "{'request_id': str, 'decision': 'allow' | 'deny', 'reason'?: str}."
        ),
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

    def resolve_permission(self, args: TeamMessageArgs) -> PermissionContext | None:
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
                msg = mailbox.send(
                    _LEAD_NAME,
                    args.to_name,
                    args.content,
                    kind=args.kind,
                    payload=args.payload,
                )
                yield TeamMessageResult(
                    action=args.action,
                    message=f"Sent {args.kind.value} message to {args.to_name}.",
                    messages=[msg.model_dump(mode="json")],
                )
            case "read_messages":
                msgs = mailbox.read(_LEAD_NAME, mark_read=True)
                formatted = "\n".join(_format_message(m) for m in msgs)
                yield TeamMessageResult(
                    action=args.action,
                    message=(
                        f"{len(msgs)} message(s) in lead inbox:\n{formatted}"
                        if formatted
                        else "0 messages in lead inbox."
                    ),
                    messages=[m.model_dump(mode="json") for m in msgs],
                )
            case "unread_messages":
                msgs = mailbox.get_unread(_LEAD_NAME)
                formatted = "\n".join(_format_message(m) for m in msgs)
                yield TeamMessageResult(
                    action=args.action,
                    message=(
                        f"{len(msgs)} unread message(s) in lead inbox:\n{formatted}"
                        if formatted
                        else "0 unread messages in lead inbox."
                    ),
                    messages=[m.model_dump(mode="json") for m in msgs],
                )
            case _:
                raise ToolError(
                    f"Unknown action '{args.action}'. Use send_message, "
                    "read_messages, or unread_messages."
                )
