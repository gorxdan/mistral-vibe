from __future__ import annotations

from vibe.cli.textual_ui.widgets.subagents_badge import SubagentsBadge


def test_subagents_badge_empty_when_none_running() -> None:
    widget = SubagentsBadge()

    widget.watch_running(())

    assert str(widget.render()) == ""


def test_subagents_badge_renders_single_running_agent() -> None:
    widget = SubagentsBadge()

    widget.watch_running(("Explore",))

    assert str(widget.render()) == "⟦⧖ Explore⟧"


def test_subagents_badge_truncates_overflow_with_count() -> None:
    widget = SubagentsBadge()

    widget.watch_running(("Explore", "general-purpose", "reviewer", "debugger"))

    assert str(widget.render()) == "⟦⧖ Explore, general-purpose, reviewer +1⟧"
