from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.widgets.agent_badge import ModelStatusBadge
from vibe.core.config import ModelConfig


def test_model_status_badge_renders_model_and_subagent_model() -> None:
    widget = ModelStatusBadge()

    widget.set_models("gpt-5.5", "devstral-small")

    assert str(widget.render()) == "⟦MODEL gpt-5.5 / SUB MODEL devstral-small⟧"


def test_model_status_badge_hides_when_models_are_empty() -> None:
    widget = ModelStatusBadge()

    widget.set_models("", "")

    assert str(widget.render()) == ""


def test_model_status_badge_uses_unknown_for_partial_values() -> None:
    widget = ModelStatusBadge()

    widget.set_models("gpt-5.5", "")

    assert str(widget.render()) == "⟦MODEL gpt-5.5 / SUB MODEL unknown⟧"


@pytest.mark.asyncio
async def test_model_status_badge_shows_initial_models_on_startup() -> None:
    models = [
        ModelConfig(name="gpt-5.5", provider="mistral", alias="host"),
        ModelConfig(name="devstral-small-latest", provider="mistral", alias="sub"),
    ]
    config = build_test_vibe_config(
        models=models, active_model="host", subagent_model="sub"
    )
    app = build_test_vibe_app(config=config)

    async with app.run_test():
        badge = app.query_one(ModelStatusBadge)

        assert str(badge.render()) == "⟦MODEL host / SUB MODEL sub⟧"
