from __future__ import annotations

from rich.text import Text

from tests.conftest import build_test_vibe_config
from vibe.cli.textual_ui.widgets.config_app import ConfigApp
from vibe.core.config import SafetyJudgeConfig


def _app(**judge_kwargs) -> ConfigApp:
    config = build_test_vibe_config(safety_judge=SafetyJudgeConfig(**judge_kwargs))
    return ConfigApp(config)


def test_toggle_value_reads_nested_enabled() -> None:
    assert _app(enabled=False)._get_toggle_value("safety_judge.enabled") == "Off"
    assert _app(enabled=True)._get_toggle_value("safety_judge.enabled") == "On"


def test_pending_toggle_change_overrides_config() -> None:
    app = _app(enabled=False)
    app.changes["safety_judge.enabled"] = "On"
    assert app._get_toggle_value("safety_judge.enabled") == "On"


def test_convert_changes_expands_dotted_key_to_nested() -> None:
    app = _app(enabled=False)
    app.changes["safety_judge.enabled"] = "On"
    app.changes["autocopy_to_clipboard"] = "Off"
    converted = app._convert_changes_for_save()
    assert converted == {
        "safety_judge": {"enabled": True},
        "autocopy_to_clipboard": False,
    }


def test_judge_model_prompt_shows_none_then_alias() -> None:
    assert "none" in _app(enabled=True, model=None)._judge_model_prompt().plain.lower()
    prompt = _app(enabled=True, model="glm")._judge_model_prompt()
    assert isinstance(prompt, Text)
    assert "glm" in prompt.plain


def test_subagent_model_prompt_shows_inherit_then_alias() -> None:
    assert "inherit" in _app()._subagent_model_prompt().plain.lower()
    config = build_test_vibe_config(subagent_model="glm")
    prompt = ConfigApp(config)._subagent_model_prompt()
    assert isinstance(prompt, Text)
    assert "glm" in prompt.plain
