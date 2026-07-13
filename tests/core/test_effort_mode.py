from __future__ import annotations

import asyncio
from typing import get_args
from unittest.mock import AsyncMock, patch

import pytest

from vibe.core.config import EFFORT_LEVELS, EffortLevel, VibeConfig
from vibe.core.orchestration import (
    OrchestrationCapabilities,
    OrchestrationDecision,
    OrchestrationLane,
    OrchestrationRoute,
    StrategyReason,
    WorkRisk,
)
from vibe.core.types import LLMMessage, Role


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


def test_keyword_override_is_non_mutating_and_restorable(
    vibe_config: VibeConfig,
) -> None:
    from vibe.cli.textual_ui.app import VibeApp

    assert vibe_config.effort_mode == "normal"
    prior_thinking = vibe_config.get_active_model().thinking

    runtime = VibeApp._with_keyword_le_chaton_override(vibe_config)

    assert runtime.effort_mode == "le-chaton"
    assert runtime.get_active_model().thinking == "max"
    assert vibe_config.effort_mode == "normal"
    assert vibe_config.get_active_model().thinking == prior_thinking


@pytest.mark.parametrize("disable_workflows", [False, True])
@pytest.mark.asyncio
async def test_le_chaton_boost_is_live_during_the_turn(
    vibe_config: VibeConfig, disable_workflows: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import build_test_vibe_app
    from vibe.cli.textual_ui.app import VibeApp

    observed: dict[str, str] = {}
    vibe_config.disable_workflows = disable_workflows
    persisted = vibe_config.model_copy(deep=True)
    app = build_test_vibe_app(config=vibe_config)

    async def fake_reload_config(**_kwargs: object) -> None:
        runtime = persisted.model_copy(deep=True)
        if app._keyword_le_chaton_lease is not None:
            runtime = VibeApp._with_keyword_le_chaton_override(runtime)
        app.agent_loop._base_config = runtime
        app.agent_loop.agent_manager.invalidate_config()

    async def fake_handle_user_message(_text: str) -> None:
        async def turn() -> None:
            await asyncio.sleep(0)
            observed["thinking"] = app.config.get_active_model().thinking

        app._agent_task = asyncio.create_task(turn())

    monkeypatch.setattr(app, "_reload_config", fake_reload_config)
    monkeypatch.setattr(app, "_handle_user_message", fake_handle_user_message)
    prior = vibe_config.get_active_model().thinking

    with patch.object(VibeConfig, "save_updates") as save_updates:
        await app._handle_le_chaton_prompt("do the thing")

    assert observed["thinking"] == "max"
    assert app._keyword_le_chaton_lease is None
    assert app.config.effort_mode == "normal"
    assert app.config.get_active_model().thinking == prior
    assert persisted.effort_mode == "normal"
    save_updates.assert_not_called()


@pytest.mark.asyncio
async def test_keyword_lease_survives_async_orchestration_debt_and_delivery(
    vibe_config: VibeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import build_test_vibe_app

    app = build_test_vibe_app(config=vibe_config)
    reload_config = AsyncMock()
    monkeypatch.setattr(app, "_reload_config", reload_config)
    controller = app.agent_loop._orchestration
    controller.begin_turn(
        enabled=True,
        user_prompt="Investigate one independent lane.",
        capabilities=OrchestrationCapabilities(task=True, background_delivery=True),
    )
    decision = OrchestrationDecision(
        route=OrchestrationRoute.TASK,
        risk=WorkRisk.MEDIUM,
        reason=StrategyReason.INDEPENDENT_LANES,
        lanes=[
            OrchestrationLane(
                id="lane-1", objective="Inspect the independent area", profile="explore"
            )
        ],
    )
    assert controller.declare(decision).accepted is True
    task_args = {
        "agent": "explore",
        "task": "[lane:lane-1] Inspect it",
        "async_run": True,
    }
    controller.record_tool_result(
        "task", task_args, "success", {"task_id": "asub-1", "completed": False}
    )

    with patch.object(VibeConfig, "save_updates") as save_updates:
        await app._acquire_keyword_le_chaton_lease()
        await app._maybe_release_keyword_le_chaton_lease()

        assert app._keyword_le_chaton_lease is not None
        assert app.agent_loop.orchestration_requires_le_chaton is True
        reload_config.assert_awaited_once()

        controller.record_task_completion("asub-1", succeeded=True)
        app.agent_loop.stage_injected_message("async task result")
        await app._maybe_release_keyword_le_chaton_lease()

        assert app._keyword_le_chaton_lease is not None
        assert app.agent_loop.orchestration_requires_le_chaton is False
        reload_config.assert_awaited_once()

        app.agent_loop._drain_pending_injections()
        app.agent_loop.messages.append(
            LLMMessage(role=Role.ASSISTANT, content="acted on async task result")
        )
        await app._maybe_release_keyword_le_chaton_lease()

    assert app._keyword_le_chaton_lease is None
    assert reload_config.await_count == 2
    save_updates.assert_not_called()
