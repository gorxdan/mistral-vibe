from __future__ import annotations

import pytest

from vibe.setup.onboarding.provider_presets import (
    PRESETS,
    apply_provider_config,
    preset_for_provider_name,
)


def test_openai_preset_present_and_keyed() -> None:
    preset = next((p for p in PRESETS if p.key == "openai"), None)
    assert preset is not None
    assert preset.requires_api_key is True
    # Keyed presets must supply both a provider and a model so the default
    # onboarding branch (_install_keyed_preset) can install the api-key screen.
    assert preset.provider is not None
    assert preset.model is not None


def test_openai_preset_provider_config() -> None:
    preset = next(p for p in PRESETS if p.key == "openai")
    provider = preset.provider
    assert provider is not None
    assert provider.name == "openai"
    assert provider.api_base == "https://api.openai.com/v1"
    assert provider.api_key_env_var == "OPENAI_API_KEY"
    # The Responses API adapter is the wired path for OpenAI.
    assert provider.api_style == "openai-responses"
    # Live model discovery should fill the picker from /v1/models.
    assert provider.discover_models is True


def test_openai_preset_model_config() -> None:
    preset = next(p for p in PRESETS if p.key == "openai")
    model = preset.model
    assert model is not None
    assert model.provider == "openai"
    assert model.supports_images is True
    # GPT-5.x are reasoning models; the adapter omits temperature for them.
    assert model.thinking != "off"


def test_preset_for_provider_name_resolves_openai() -> None:
    preset = preset_for_provider_name("openai")
    assert preset is not None
    assert preset.key == "openai"


def test_openai_chatgpt_preset_discovers_models() -> None:
    preset = next((p for p in PRESETS if p.key == "openai-chatgpt"), None)
    assert preset is not None
    assert preset.provider is not None
    # Discovery queries the codex /models endpoint via the stored OAuth token,
    # so the picker reflects the subscription's full model set.
    assert preset.provider.discover_models is True


def test_apply_openai_preset_persists_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reloading the persisted config validates the active provider's key.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    preset = next(p for p in PRESETS if p.key == "openai")
    assert preset.provider is not None and preset.model is not None

    apply_provider_config(preset.provider, preset.model)

    from vibe.core.config import VibeConfig

    config = VibeConfig.get_persisted_config()
    provider_names = {p["name"] for p in config["providers"]}
    assert "openai" in provider_names
    assert config["active_model"] == preset.model.alias
