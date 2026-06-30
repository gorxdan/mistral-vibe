from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from textual.widgets import OptionList

from tests.conftest import build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.app import BottomApp
from vibe.cli.textual_ui.widgets.config_app import ConfigApp
from vibe.cli.textual_ui.widgets.model_picker import ModelPickerApp, _build_option_text
from vibe.cli.textual_ui.widgets.thinking_picker import ThinkingPickerApp
from vibe.core.config._settings import THINKING_LEVELS, ModelConfig, ProviderConfig


def _make_config_with_models():
    models = [
        ModelConfig(name="model-a", provider="mistral", alias="alpha"),
        ModelConfig(name="model-b", provider="mistral", alias="beta"),
        ModelConfig(name="model-c", provider="mistral", alias="gamma"),
    ]
    return build_test_vibe_config(models=models, active_model="alpha")


def _make_config_with_mixed_provider_models():
    providers = [
        ProviderConfig(name="mistral", api_base="https://mistral.example/v1"),
        ProviderConfig(name="llamacpp", api_base="http://127.0.0.1:8080/v1"),
        ProviderConfig(name="openai-chatgpt", api_base="https://chatgpt.example/codex"),
    ]
    models = [
        ModelConfig(name="model-c", provider="openai-chatgpt", alias="gamma"),
        ModelConfig(name="model-a", provider="mistral", alias="alpha"),
        ModelConfig(name="local-model", provider="llamacpp", alias="local"),
        ModelConfig(name="model-b", provider="mistral", alias="beta"),
    ]
    return build_test_vibe_config(
        providers=providers, models=models, active_model="alpha"
    )


def _selectable_option_ids(picker: ModelPickerApp) -> list[str]:
    option_list = picker.query_one(OptionList)
    ids: list[str] = []
    for option in option_list.options:
        if option.disabled or option.id is None:
            continue
        ids.append(option.id)
    return ids


def _disabled_option_labels(picker: ModelPickerApp) -> list[str]:
    option_list = picker.query_one(OptionList)
    labels: list[str] = []
    for option in option_list.options:
        if option.disabled:
            labels.append(str(option.prompt).strip())
    return labels


# --- /config command ---


@pytest.mark.asyncio
async def test_config_opens_config_app() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Config
        assert len(app.query(ConfigApp)) == 1


@pytest.mark.asyncio
async def test_config_escape_returns_to_input() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        await pilot.press("escape")
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(ConfigApp)) == 0


@pytest.mark.asyncio
async def test_config_toggle_autocopy() -> None:
    config = _make_config_with_models()
    config.autocopy_to_clipboard = False
    app = build_test_vibe_app(config=config)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Navigate down to Auto-copy (third item, after Model + Thinking) and toggle
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.1)

        # Verify the toggle happened in the widget
        config_app = app.query_one(ConfigApp)
        assert config_app.changes.get("autocopy_to_clipboard") == "On"


@pytest.mark.asyncio
async def test_config_escape_saves_changes() -> None:
    config = _make_config_with_models()
    config.autocopy_to_clipboard = False
    app = build_test_vibe_app(config=config)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Toggle auto-copy (skip Model + Thinking rows)
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.1)

        with patch("vibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("escape")
            await pilot.pause(0.2)

            mock_save.assert_called_once()
            changes = mock_save.call_args[0][0]
            assert changes["autocopy_to_clipboard"] is True


# --- /model command ---


@pytest.mark.asyncio
async def test_model_opens_model_picker() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_model()
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.ModelPicker
        assert len(app.query(ModelPickerApp)) == 1


# Discovery probes live local runtimes (e.g. a dev-machine ollama), which would
# inject extra models and make exact assertions flaky. Patch it off so these
# tests see only the configured models.
def _no_discovery() -> Any:
    return patch(
        "vibe.core.llm.model_discovery.discover_extra_models",
        new=AsyncMock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_model_picker_shows_all_models() -> None:
    with _no_discovery():
        app = build_test_vibe_app(config=_make_config_with_models())
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._show_model()
            await pilot.pause(0.2)

            picker = app.query_one(ModelPickerApp)
            assert picker._model_aliases == ["alpha", "beta", "gamma"]
            assert _selectable_option_ids(picker) == ["alpha", "beta", "gamma"]
            assert picker._current_model == "alpha"


@pytest.mark.asyncio
async def test_model_picker_maps_aliases_to_api_names() -> None:
    # The picker labels each entry with the provider's API model name (config
    # `name`) while still keying selection by the friendly alias.
    with _no_discovery():
        app = build_test_vibe_app(config=_make_config_with_models())
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._show_model()
            await pilot.pause(0.2)

            picker = app.query_one(ModelPickerApp)
            assert picker._display_names == {
                "alpha": "model-a",
                "beta": "model-b",
                "gamma": "model-c",
            }


@pytest.mark.asyncio
async def test_model_picker_groups_models_by_provider() -> None:
    with _no_discovery():
        app = build_test_vibe_app(config=_make_config_with_mixed_provider_models())
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._show_model()
            await pilot.pause(0.2)

            picker = app.query_one(ModelPickerApp)
            assert _disabled_option_labels(picker) == [
                "Provider: mistral",
                "Provider: llamacpp",
                "Provider: openai-chatgpt",
            ]
            assert _selectable_option_ids(picker) == ["alpha", "beta", "local", "gamma"]


@pytest.mark.asyncio
async def test_model_picker_filter_narrows_by_name() -> None:
    # Typing in the filter input narrows the visible models by name/alias.
    with _no_discovery():
        app = build_test_vibe_app(config=_make_config_with_models())
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._show_model()
            await pilot.pause(0.2)

            picker = app.query_one(ModelPickerApp)
            assert _selectable_option_ids(picker) == ["alpha", "beta", "gamma"]

            # "model-c" is the API name of alias "gamma"; the others are model-a/b.
            await pilot.press("m", "o", "d", "e", "l", "-", "c")
            await pilot.pause(0.2)

            assert _selectable_option_ids(picker) == ["gamma"]


@pytest.mark.asyncio
async def test_model_picker_filter_clear_restores_all() -> None:
    with _no_discovery():
        app = build_test_vibe_app(config=_make_config_with_models())
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._show_model()
            await pilot.pause(0.2)

            picker = app.query_one(ModelPickerApp)
            await pilot.press("b", "e", "t", "a")
            await pilot.pause(0.2)
            assert _selectable_option_ids(picker) == ["beta"]

            # Clear the filter; all models reappear.
            await pilot.press("ctrl+u")
            await pilot.pause(0.2)
            assert _selectable_option_ids(picker) == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_model_picker_keyboard_nav_and_select_while_filter_focused() -> None:
    # Arrow/enter keep driving the option list even though the filter has focus.
    with _no_discovery():
        app = build_test_vibe_app(config=_make_config_with_models())
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            await app._show_model()
            await pilot.pause(0.2)

            with patch("vibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
                await pilot.press("down")
                await pilot.press("enter")
                await pilot.pause(0.2)

                mock_save.assert_called_once_with({"active_model": "beta"})

            assert app._current_bottom_app == BottomApp.Input


@pytest.mark.asyncio
async def test_rate_limit_model_picker_groups_candidates_by_provider() -> None:
    app = build_test_vibe_app(config=_make_config_with_mixed_provider_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._switch_to_rate_limit_picker_app("alpha", ["alpha", "local", "beta"])
        await pilot.pause(0.2)

        picker = app.query_one(ModelPickerApp)
        assert _disabled_option_labels(picker) == [
            "Provider: mistral",
            "Provider: llamacpp",
        ]
        assert _selectable_option_ids(picker) == ["alpha", "beta", "local"]


def test_build_option_text_shows_api_name_with_alias() -> None:
    # API name is the primary label; a differing alias is appended dim.
    rendered = _build_option_text("model-a", "alpha", is_current=False).plain
    assert "model-a" in rendered
    assert "alpha" in rendered

    # When alias == name (e.g. a live-discovered model), no redundant suffix.
    same = _build_option_text("gpt-5.5", "gpt-5.5", is_current=False).plain
    assert same.strip() == "gpt-5.5"


@pytest.mark.asyncio
async def test_model_picker_escape_returns_to_input() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_model()
        await pilot.pause(0.2)

        await pilot.press("escape")
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(ModelPickerApp)) == 0


@pytest.mark.asyncio
async def test_model_picker_escape_does_not_save() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_model()
        await pilot.pause(0.2)

        with patch("vibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("escape")
            await pilot.pause(0.2)

            mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_model_picker_select_model() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_model()
        await pilot.pause(0.2)

        # Navigate down to "beta" and select
        await pilot.press("down")
        with patch("vibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("enter")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with({"active_model": "beta"})

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(ModelPickerApp)) == 0


@pytest.mark.asyncio
async def test_model_picker_select_current_model() -> None:
    """Selecting the already-active model still saves (idempotent)."""
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_model()
        await pilot.pause(0.2)

        with patch("vibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("enter")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with({"active_model": "alpha"})

        assert app._current_bottom_app == BottomApp.Input


# --- config -> model picker flow ---


@pytest.mark.asyncio
async def test_config_model_entry_opens_model_picker() -> None:
    """Pressing Enter on the Model row in /config opens the model picker."""
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Model row is the first item, already highlighted. Press enter.
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert app._current_bottom_app == BottomApp.ModelPicker
        assert len(app.query(ModelPickerApp)) == 1
        assert len(app.query(ConfigApp)) == 0


@pytest.mark.asyncio
async def test_config_to_model_picker_escape_returns_to_input() -> None:
    """Opening model picker from config, then ESC, returns to input (not config)."""
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Open model picker from config
        await pilot.press("enter")
        await pilot.pause(0.3)

        # Escape model picker
        await pilot.press("escape")
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(ModelPickerApp)) == 0
        assert len(app.query(ConfigApp)) == 0


@pytest.mark.asyncio
async def test_config_to_model_picker_select_returns_to_input() -> None:
    """Opening model picker from config, selecting a model, returns to input."""
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Open model picker from config
        await pilot.press("enter")
        await pilot.pause(0.3)

        # Select second model
        await pilot.press("down")
        with patch("vibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("enter")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with({"active_model": "beta"})

        assert app._current_bottom_app == BottomApp.Input


@pytest.mark.asyncio
async def test_config_pending_changes_saved_before_model_picker() -> None:
    """Toggle changes in config are saved before switching to model picker."""
    config = _make_config_with_models()
    config.autocopy_to_clipboard = False
    app = build_test_vibe_app(config=config)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Toggle auto-copy (third row, after Model + Thinking)
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.1)

        # Go back up to model row and open model picker
        await pilot.press("up")
        await pilot.press("up")
        with patch("vibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("enter")
            await pilot.pause(0.3)

            mock_save.assert_called_once()
            changes = mock_save.call_args[0][0]
            assert changes["autocopy_to_clipboard"] is True


# --- /thinking command ---


@pytest.mark.asyncio
async def test_thinking_opens_thinking_picker() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_thinking()
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.ThinkingPicker
        assert len(app.query(ThinkingPickerApp)) == 1


@pytest.mark.asyncio
async def test_thinking_picker_shows_all_levels() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_thinking()
        await pilot.pause(0.2)

        picker = app.query_one(ThinkingPickerApp)
        assert picker._thinking_levels == THINKING_LEVELS
        assert picker._current_thinking == "off"


@pytest.mark.asyncio
async def test_thinking_picker_escape_returns_to_input() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_thinking()
        await pilot.pause(0.2)

        await pilot.press("escape")
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(ThinkingPickerApp)) == 0


@pytest.mark.asyncio
async def test_thinking_picker_select_level() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_thinking()
        await pilot.pause(0.2)

        # Navigate down to "low" (second item) and select
        await pilot.press("down")
        with patch.object(app, "_reload_config", new=AsyncMock()):
            await pilot.press("enter")
            await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(ThinkingPickerApp)) == 0
        assert app.config.get_active_model().thinking == "low"


@pytest.mark.asyncio
async def test_thinking_picker_select_high() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_thinking()
        await pilot.pause(0.2)

        # Navigate to "high" (4th item = 3 downs from "off")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("down")
        with patch.object(app, "_reload_config", new=AsyncMock()):
            await pilot.press("enter")
            await pilot.pause(0.2)

        assert app.config.get_active_model().thinking == "high"


# --- config -> thinking picker flow ---


@pytest.mark.asyncio
async def test_config_thinking_entry_opens_thinking_picker() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Thinking row is the second item (after Model). Navigate down and press enter.
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert app._current_bottom_app == BottomApp.ThinkingPicker
        assert len(app.query(ThinkingPickerApp)) == 1
        assert len(app.query(ConfigApp)) == 0


@pytest.mark.asyncio
async def test_config_to_thinking_picker_escape_returns_to_input() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Open thinking picker from config
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.3)

        # Escape thinking picker
        await pilot.press("escape")
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(ThinkingPickerApp)) == 0
        assert len(app.query(ConfigApp)) == 0


@pytest.mark.asyncio
async def test_config_to_thinking_picker_select_returns_to_input() -> None:
    app = build_test_vibe_app(config=_make_config_with_models())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app._show_config()
        await pilot.pause(0.2)

        # Open thinking picker from config
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.3)

        # Select "medium" (3rd item = 2 downs from "off")
        await pilot.press("down")
        await pilot.press("down")
        with patch.object(app, "_reload_config", new=AsyncMock()):
            await pilot.press("enter")
            await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.Input
        assert app.config.get_active_model().thinking == "medium"
