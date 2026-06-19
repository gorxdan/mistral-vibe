from __future__ import annotations

import asyncio
from typing import get_args
from unittest.mock import patch

import pytest

from vibe.core.config import EFFORT_LEVELS, EffortLevel, VibeConfig


def test_effort_levels_match_literal() -> None:
    assert set(EFFORT_LEVELS) == set(get_args(EffortLevel))
    assert "normal" in EFFORT_LEVELS
    assert "le-chaton" in EFFORT_LEVELS


def test_default_effort_mode_is_normal(vibe_config: VibeConfig) -> None:
    assert vibe_config.effort_mode == "normal"


def test_set_effort_mode_le_chaton_persists_and_maxes_thinking(
    vibe_config: VibeConfig,
) -> None:
    with patch.object(VibeConfig, "save_updates") as mock_save:
        vibe_config.set_effort_mode("le-chaton")

    assert vibe_config.effort_mode == "le-chaton"
    # Persisted to disk.
    assert any(
        call.args and call.args[0].get("effort_mode") == "le-chaton"
        for call in mock_save.call_args_list
    )
    # le chaton couples max thinking.
    assert vibe_config.get_active_model().thinking == "max"


def test_set_effort_mode_normal_leaves_thinking_untouched(
    vibe_config: VibeConfig,
) -> None:
    original = vibe_config.get_active_model().thinking
    with patch.object(VibeConfig, "save_updates"):
        vibe_config.set_effort_mode("normal")

    assert vibe_config.effort_mode == "normal"
    assert vibe_config.get_active_model().thinking == original


def test_le_chaton_turn_restore_returns_to_prior_mode_and_thinking(
    vibe_config: VibeConfig,
) -> None:
    """DOC-1: the le-chaton keyword triggers the mode 'for that turn' only.

    The handler captures the prior effort_mode and thinking, switches to
    le-chaton (which maxes thinking), then restores both. Without restoration
    the switch persisted permanently across sessions, contradicting the docs.
    This test exercises the restore sequence at the config level.
    """
    assert vibe_config.effort_mode == "normal"
    prior_thinking = vibe_config.get_active_model().thinking

    with patch.object(VibeConfig, "save_updates"):
        # Turn: switch to le-chaton.
        vibe_config.set_effort_mode("le-chaton")
        assert vibe_config.effort_mode == "le-chaton"
        assert vibe_config.get_active_model().thinking == "max"
        # Turn ends: restore prior mode + thinking.
        vibe_config.set_effort_mode("normal")
        if vibe_config.get_active_model().thinking != prior_thinking:
            vibe_config.set_thinking(prior_thinking)

    assert vibe_config.effort_mode == "normal"
    assert vibe_config.get_active_model().thinking == prior_thinking


@pytest.mark.asyncio
async def test_le_chaton_boost_is_live_during_the_turn(
    vibe_config: VibeConfig,
) -> None:
    """A-1: the boost must apply to the turn the keyword triggers.

    _handle_user_message spawns the turn as a background task and returns; the
    turn reads the thinking level live when it builds its request. The handler
    must therefore wait for the turn before restoring, otherwise the boost is
    reverted before it is ever used (a silent no-op).
    """
    from vibe.cli.textual_ui.app import VibeApp

    observed: dict[str, str] = {}

    class _Stub:
        def __init__(self, config: VibeConfig) -> None:
            self.config = config
            self._agent_task: asyncio.Task[None] | None = None

        async def _reload_config(self) -> None:
            pass

        async def _handle_user_message(self, text: str) -> None:
            async def _turn() -> None:
                # Yield once so that, without the fix, the handler's finally
                # would already have reverted the boost before this read.
                await asyncio.sleep(0)
                observed["thinking"] = self.config.get_active_model().thinking

            self._agent_task = asyncio.create_task(_turn())

    stub = _Stub(vibe_config)
    prior = vibe_config.get_active_model().thinking

    with patch.object(VibeConfig, "save_updates"):
        await VibeApp._handle_le_chaton_prompt(stub, "do the thing")  # type: ignore[arg-type]

    # The boost was live while the turn ran...
    assert observed["thinking"] == "max"
    # ...and is fully restored afterwards.
    assert stub.config.effort_mode == "normal"
    assert stub.config.get_active_model().thinking == prior
