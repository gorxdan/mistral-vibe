from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.widgets.agent_badge import ModelStatusBadge, SubModelBadge
from vibe.core.config import ModelConfig


def test_model_status_badge_renders_active_model() -> None:
    widget = ModelStatusBadge()

    widget.set_model("gpt-5.5")

    assert str(widget.render()) == "· gpt-5.5"


def test_model_status_badge_hides_when_model_empty() -> None:
    widget = ModelStatusBadge()

    widget.set_model("")

    assert str(widget.render()) == ""


def test_model_status_badge_elides_long_model_id() -> None:
    widget = ModelStatusBadge()

    widget.set_model("m" * 40)

    assert str(widget.render()) == "· " + "m" * 27 + "…"


def test_sub_model_badge_renders_when_distinct_from_active() -> None:
    widget = SubModelBadge()

    widget.set_model("devstral-small", "gpt-5.5")

    assert str(widget.render()) == "· sub devstral-small"


def test_sub_model_badge_hidden_when_same_as_active() -> None:
    widget = SubModelBadge()

    widget.set_model("gpt-5.5", "gpt-5.5")

    assert str(widget.render()) == ""


def test_sub_model_badge_hidden_when_empty() -> None:
    widget = SubModelBadge()

    widget.set_model("", "gpt-5.5")

    assert str(widget.render()) == ""


@pytest.mark.asyncio
async def test_model_badges_show_initial_models_on_startup() -> None:
    models = [
        ModelConfig(name="gpt-5.5", provider="mistral", alias="host"),
        ModelConfig(name="devstral-small-latest", provider="mistral", alias="sub"),
    ]
    config = build_test_vibe_config(
        models=models, active_model="host", subagent_model="sub"
    )
    app = build_test_vibe_app(config=config)

    async with app.run_test():
        assert str(app.query_one(ModelStatusBadge).render()) == "· host"
        assert str(app.query_one(SubModelBadge).render()) == "· sub sub"
