from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.widgets.agent_badge import AgentProfileBadge
from vibe.core.agents import AgentSafety


def test_agent_profile_badge_renders_card_with_name() -> None:
    widget = AgentProfileBadge()

    widget.set_profile("default", AgentSafety.NEUTRAL)

    assert str(widget.render()) == "⟦● default⟧"


def test_agent_profile_badge_neutral_has_no_safety_class() -> None:
    widget = AgentProfileBadge()

    widget.set_profile("default", AgentSafety.NEUTRAL)

    assert not widget.has_class("badge-safe")
    assert not widget.has_class("badge-warning")
    assert not widget.has_class("badge-error")


def test_agent_profile_badge_yolo_marks_error() -> None:
    widget = AgentProfileBadge()

    widget.set_profile("auto approve", AgentSafety.YOLO)

    assert widget.has_class("badge-error")


def test_agent_profile_badge_destructive_marks_warning() -> None:
    widget = AgentProfileBadge()

    widget.set_profile("accept edits", AgentSafety.DESTRUCTIVE)

    assert widget.has_class("badge-warning")


def test_agent_profile_badge_swap_clears_prior_safety_class() -> None:
    widget = AgentProfileBadge()

    widget.set_profile("auto approve", AgentSafety.YOLO)
    widget.set_profile("plan", AgentSafety.SAFE)

    assert widget.has_class("badge-safe")
    assert not widget.has_class("badge-error")


def test_agent_profile_badge_empty_name_hides_content() -> None:
    widget = AgentProfileBadge()

    widget.set_profile("", AgentSafety.NEUTRAL)

    assert str(widget.render()) == ""


@pytest.mark.asyncio
async def test_agent_profile_badge_shows_initial_profile_on_startup() -> None:
    app = build_test_vibe_app()

    async with app.run_test():
        badge = app.query_one(AgentProfileBadge)
        expected = app.agent_loop.agent_profile.display_name.lower()

        assert str(badge.render()) == f"⟦● {expected}⟧"
