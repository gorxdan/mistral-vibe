from __future__ import annotations

import os

from vibe.core.teams._escalate import escalate_to_lead
from vibe.core.teams.models import TeamSafetyMode

TEAM_SAFETY_MODE_ENV = "VIBE_TEAM_SAFETY_MODE"


def team_safety_mode_from_env() -> TeamSafetyMode:
    raw = os.environ.get(TEAM_SAFETY_MODE_ENV, TeamSafetyMode.SHARED.value)
    try:
        return TeamSafetyMode(raw)
    except ValueError:
        return TeamSafetyMode.SHARED


def shared_ask_enabled() -> bool:
    if team_safety_mode_from_env() is not TeamSafetyMode.SHARED_ASK:
        return False
    return bool(
        os.environ.get("VIBE_TEAM_DIR") and os.environ.get("VIBE_TEAMMATE_NAME")
    )


async def require_shared_ask(tool: str, description: str) -> None:
    if not shared_ask_enabled():
        return
    await escalate_to_lead(tool, description)
