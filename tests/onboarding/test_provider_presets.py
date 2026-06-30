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


def test_sakana_preset_present_and_keyed() -> None:
    preset = next((p for p in PRESETS if p.key == "sakana"), None)
    assert preset is not None
    assert preset.requires_api_key is True
    assert preset.provider is not None
    assert preset.model is not None


def test_sakana_preset_provider_config() -> None:
    preset = next(p for p in PRESETS if p.key == "sakana")
    provider = preset.provider
    assert provider is not None
    assert provider.name == "sakana"
    assert provider.api_base == "https://api.sakana.ai/v1"
    assert provider.api_key_env_var == "SAKANA_API_KEY"
    # Fugu documents the Responses API with reasoning.effort.
    assert provider.api_style == "openai-responses"
    assert provider.discover_models is True


def test_sakana_preset_model_config() -> None:
    preset = next(p for p in PRESETS if p.key == "sakana")
    model = preset.model
    assert model is not None
    assert model.name == "fugu"
    assert model.provider == "sakana"
    assert model.supports_images is True
    # Fugu accepts high / xhigh reasoning effort; "high" maps straight through.
    assert model.thinking == "high"
    # Fugu documents a 1M-token context window; the compaction budget sits
    # below the ceiling so we shape context before the model rejects it.
    assert model.auto_compact_threshold == 880000


def test_sakana_preset_ships_fugu_ultra() -> None:
    preset = next(p for p in PRESETS if p.key == "sakana")
    ultra = next((m for m in preset.extra_models if m.alias == "fugu-ultra"), None)
    assert ultra is not None
    assert ultra.name == "fugu-ultra"
    assert ultra.provider == "sakana"
    assert ultra.supports_images is True
    assert ultra.thinking == "high"
    # Fugu Ultra shares Fugu's 1M-token context window, so it carries the same
    # compaction budget rather than falling back to the global default.
    assert ultra.auto_compact_threshold == 880000


def test_apply_sakana_preset_persists_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAKANA_API_KEY", "sk-sakana-test")
    preset = next(p for p in PRESETS if p.key == "sakana")
    assert preset.provider is not None and preset.model is not None

    apply_provider_config(preset.provider, preset.model)

    from vibe.core.config import VibeConfig

    config = VibeConfig.get_persisted_config()
    provider_names = {p["name"] for p in config["providers"]}
    assert "sakana" in provider_names
    assert config["active_model"] == preset.model.alias
    # Both Fugu models are persisted with their 1M-window compaction budget so a
    # user can switch to fugu-ultra without it defaulting to the global threshold.
    persisted = {m["alias"]: m for m in config["models"]}
    assert persisted["fugu"]["auto_compact_threshold"] == 880000
    assert persisted["fugu-ultra"]["auto_compact_threshold"] == 880000


def test_zai_preset_does_not_discover_models() -> None:
    preset = next((p for p in PRESETS if p.key == "zai"), None)
    assert preset is not None
    assert preset.provider is not None
    assert preset.provider.discover_models is False


def test_zai_preset_is_session_keyed() -> None:
    # ZAI's prefix cache scatters across nodes under concurrency; the preset pins
    # each conversation to one partition via a per-session prompt_cache_key.
    preset = next((p for p in PRESETS if p.key == "zai"), None)
    assert preset is not None
    assert preset.provider is not None
    assert preset.provider.cache.session_keyed is True


def test_kimi_preset_does_not_discover_models() -> None:
    preset = next((p for p in PRESETS if p.key == "kimi"), None)
    assert preset is not None
    assert preset.provider is not None
    assert preset.provider.discover_models is False


def test_minimax_preset_discovers_models() -> None:
    preset = next((p for p in PRESETS if p.key == "minimax"), None)
    assert preset is not None
    assert preset.provider is not None
    assert preset.provider.discover_models is True


def test_longcat_preset_present_and_keyed() -> None:
    preset = next((p for p in PRESETS if p.key == "longcat"), None)
    assert preset is not None
    assert preset.requires_api_key is True
    assert preset.provider is not None
    assert preset.model is not None


def test_longcat_preset_provider_config() -> None:
    preset = next(p for p in PRESETS if p.key == "longcat")
    provider = preset.provider
    assert provider is not None
    assert provider.name == "longcat"
    assert provider.api_base == "https://api.longcat.chat/openai/v1"
    assert provider.api_key_env_var == "LONGCAT_API_KEY"
    # OpenAI-compatible chat-completions endpoint; the default openai adapter.
    assert provider.api_style == "openai"
    # Single-model platform; discovery is disabled.
    assert provider.discover_models is False


def test_longcat_preset_model_config() -> None:
    preset = next(p for p in PRESETS if p.key == "longcat")
    model = preset.model
    assert model is not None
    assert model.name == "LongCat-2.0"
    assert model.provider == "longcat"
    assert model.alias == "longcat"
    # 1M-token context window; compaction budget sits below the ceiling.
    assert model.auto_compact_threshold == 880000


def test_apply_longcat_preset_persists_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LONGCAT_API_KEY", "sk-longcat-test")
    preset = next(p for p in PRESETS if p.key == "longcat")
    assert preset.provider is not None and preset.model is not None

    apply_provider_config(preset.provider, preset.model)

    from vibe.core.config import VibeConfig

    config = VibeConfig.get_persisted_config()
    provider_names = {p["name"] for p in config["providers"]}
    assert "longcat" in provider_names
    assert config["active_model"] == preset.model.alias


def test_longcat_preset_resolvable_by_provider_name() -> None:
    preset = preset_for_provider_name("longcat")
    assert preset is not None
    assert preset.key == "longcat"


def test_openrouter_preset_present_and_keyed() -> None:
    preset = next((p for p in PRESETS if p.key == "openrouter"), None)
    assert preset is not None
    assert preset.requires_api_key is True
    assert preset.provider is not None
    assert preset.model is not None


def test_openrouter_preset_provider_config() -> None:
    preset = next(p for p in PRESETS if p.key == "openrouter")
    provider = preset.provider
    assert provider is not None
    assert provider.name == "openrouter"
    assert provider.api_base == "https://openrouter.ai/api/v1"
    assert provider.api_key_env_var == "OPENROUTER_API_KEY"
    # OpenAI-compatible chat-completions endpoint; the default openai adapter.
    assert provider.api_style == "openai"
    # Multi-model router; discovery fills the picker with available models.
    assert provider.discover_models is True


def test_openrouter_preset_model_config() -> None:
    preset = next(p for p in PRESETS if p.key == "openrouter")
    model = preset.model
    assert model is not None
    assert model.name == "openrouter/owl-alpha"
    assert model.provider == "openrouter"
    assert model.alias == "openrouter"
    # Default Owl Alpha has a ~1M context window; budget matches the other
    # large presets.
    assert model.auto_compact_threshold == 880000


def test_apply_openrouter_preset_persists_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-test")
    preset = next(p for p in PRESETS if p.key == "openrouter")
    assert preset.provider is not None and preset.model is not None

    apply_provider_config(preset.provider, preset.model)

    from vibe.core.config import VibeConfig

    config = VibeConfig.get_persisted_config()
    provider_names = {p["name"] for p in config["providers"]}
    assert "openrouter" in provider_names
    assert config["active_model"] == preset.model.alias


def test_openrouter_preset_resolvable_by_provider_name() -> None:
    preset = preset_for_provider_name("openrouter")
    assert preset is not None
    assert preset.key == "openrouter"


def test_bedrock_preset_present_and_keyed() -> None:
    preset = next((p for p in PRESETS if p.key == "bedrock"), None)
    assert preset is not None
    assert preset.requires_api_key is True
    assert preset.provider is not None
    assert preset.model is not None


def test_bedrock_preset_provider_config() -> None:
    preset = next(p for p in PRESETS if p.key == "bedrock")
    provider = preset.provider
    assert provider is not None
    assert provider.name == "bedrock"
    assert provider.api_base == "https://bedrock-mantle.us-east-1.api.aws/anthropic"
    assert provider.api_key_env_var == "AWS_BEARER_TOKEN_BEDROCK"
    # Bedrock Mantle speaks the Anthropic Messages API; the adapter pins the
    # region-aware base URL from `region`.
    assert provider.api_style == "bedrock-anthropic"
    assert provider.region == "us-east-1"
    # Bedrock's model catalog lives behind a separate endpoint; models are
    # added in config.
    assert provider.discover_models is False


def test_bedrock_preset_model_config() -> None:
    preset = next(p for p in PRESETS if p.key == "bedrock")
    model = preset.model
    assert model is not None
    # anthropic.<family> model ID; Haiku 4.5 is the default open model.
    assert model.name == "anthropic.claude-haiku-4-5"
    assert model.provider == "bedrock"
    assert model.alias == "bedrock"
    assert model.supports_images is True


def test_apply_bedrock_preset_persists_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-test-key")
    preset = next(p for p in PRESETS if p.key == "bedrock")
    assert preset.provider is not None and preset.model is not None

    apply_provider_config(preset.provider, preset.model)

    from vibe.core.config import VibeConfig

    config = VibeConfig.get_persisted_config()
    provider_names = {p["name"] for p in config["providers"]}
    assert "bedrock" in provider_names
    assert config["active_model"] == preset.model.alias


def test_bedrock_preset_resolvable_by_provider_name() -> None:
    preset = preset_for_provider_name("bedrock")
    assert preset is not None
    assert preset.key == "bedrock"
