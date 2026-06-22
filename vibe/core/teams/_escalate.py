"""Teammate-side helper to escalate a decision to the lead via the team mailbox.

Runs inside a teammate subprocess (``vibe -p``). A teammate that would otherwise
auto-approve a destructive action (because it runs in isolation) can call
``escalate_to_lead`` to post a ``PERMISSION_REQUEST`` to the lead's inbox and
block on the lead's ``PERMISSION_RESPONSE``. The lead decides via the
``team_message`` tool.

This closes the structural gap where a subprocess teammate could not surface a
decision to the host: the teammate's bash is auto-approved in isolation, so
without this helper any approval flow has to invent prose. With it, the lead's
``team_message read_messages`` renders a typed prompt and the round-trip is a
well-known request/response pair, not free text.

The helper is opt-in: existing teammates keep auto-approving. Only callers that
explicitly invoke ``escalate_to_lead`` route through the lead.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from uuid import uuid4

from vibe.core.logger import logger
from vibe.core.teams.mailbox import Mailbox
from vibe.core.teams.models import MessageKind

# Poll interval for the lead's response. The mailbox is file-backed, so the
# teammate polls the lead's inbox for its PERMISSION_RESPONSE.
_POLL_INTERVAL_S = 0.5
# Default ceiling on how long to wait for the lead. A teammate that blocks
# forever ties up its subprocess; callers can pass a longer timeout if they
# genuinely need to wait longer.
_DEFAULT_TIMEOUT_S = 120.0


class EscalationDenied(Exception):
    """Raised when the lead denies the request or the escalation times out."""


def _team_env_or_raise() -> tuple[str, str, str]:
    """Read team identity from the teammate subprocess environment."""
    team_dir = os.environ.get("VIBE_TEAM_DIR")
    team_name = os.environ.get("VIBE_TEAM_NAME") or "team"
    teammate_name = os.environ.get("VIBE_TEAMMATE_NAME")
    if not team_dir or not teammate_name:
        raise EscalationDenied(
            "No active team (VIBE_TEAM_DIR / VIBE_TEAMMATE_NAME unset); the "
            "escalation helper only runs inside a teammate subprocess."
        )
    return team_dir, team_name, teammate_name


async def escalate_to_lead(
    tool: str,
    description: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> None:
    """Ask the lead to approve ``description`` for ``tool``; block until answered.

    Posts a ``PERMISSION_REQUEST`` to the lead's inbox from the teammate and
    polls the teammate's own inbox for the matching ``PERMISSION_RESPONSE``.

    Raises ``EscalationDenied`` if the lead denies, the escalation times out,
    or the teammate is not running inside a team.
    """
    team_dir_raw, _team_name, teammate_name = _team_env_or_raise()
    mailbox = Mailbox(Path(team_dir_raw))
    request_id = str(uuid4())
    payload = {
        "request_id": request_id,
        "tool": tool,
        "description": description,
        "teammate": teammate_name,
    }
    mailbox.send(
        teammate_name,
        "lead",
        f"Requesting approval for {tool}: {description}",
        kind=MessageKind.PERMISSION_REQUEST,
        payload=payload,
    )
    logger.info(
        "Escalation %s posted for tool=%s teammate=%s",
        request_id,
        tool,
        teammate_name,
    )

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for msg in mailbox.get_unread(teammate_name):
            if msg.kind is not MessageKind.PERMISSION_RESPONSE:
                continue
            if msg.payload.get("request_id") != request_id:
                continue
            # Mark consumed so it doesn't reprocess next poll.
            mailbox.read(teammate_name, mark_read=True)
            decision = str(msg.payload.get("decision", "deny")).lower()
            if decision != "allow":
                reason = msg.payload.get("reason")
                raise EscalationDenied(
                    f"Lead denied {tool} escalation{f': {reason}' if reason else ''}"
                )
            return
        await asyncio.sleep(_POLL_INTERVAL_S)

    raise EscalationDenied(
        f"Escalation {request_id} for {tool} timed out after {timeout_s}s"
    )
