from __future__ import annotations

from typing import get_args
from unittest.mock import patch

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
