from __future__ import annotations

from dataclasses import dataclass

from vibe.core.config import ModelConfig, ProviderConfig, VibeConfig

ZAI_API_BASE = "https://api.z.ai/api/coding/paas/v4"
KIMI_API_BASE = "https://api.kimi.com/coding/v1"
KIMI_USER_AGENT = "KimiCLI/1.47.0"
ZAI_HELP_URL = "https://z.ai"
KIMI_HELP_URL = "https://kimi.com"
MINIMAX_API_BASE = "https://api.minimax.io/v1"
MINIMAX_HELP_URL = "https://platform.minimax.io/user-center/payment/token-plan"

CUSTOM_PROVIDER_NAME = "custom"


@dataclass(frozen=True)
class ProviderPreset:
    key: str
    label: str
    description: str
    requires_api_key: bool
    badge: str | None = None
    help_url: str | None = None
    provider: ProviderConfig | None = None
    model: ModelConfig | None = None


PRESETS: list[ProviderPreset] = [
    ProviderPreset(
        key="mistral",
        label="Mistral",
        description=(
            "Default. Sign in to Mistral AI Studio in your browser, or paste "
            "a MISTRAL_API_KEY."
        ),
        requires_api_key=True,
        badge="Default",
    ),
    ProviderPreset(
        key="zai",
        label="GLM (ZAI / Zhipu)",
        description=(
            "GLM-5.2 via the ZAI Coding Plan endpoint. Requires a ZAI_API_KEY."
        ),
        requires_api_key=True,
        help_url=ZAI_HELP_URL,
        provider=ProviderConfig(
            name="zai", api_base=ZAI_API_BASE, api_key_env_var="ZAI_API_KEY"
        ),
        model=ModelConfig(
            name="glm-5.2",
            provider="zai",
            alias="glm",
            thinking="high",
            input_price=0.0,
            output_price=0.0,
            auto_compact_threshold=880000,
        ),
    ),
    ProviderPreset(
        key="kimi",
        label="Kimi (Moonshot)",
        description=(
            "Kimi K2.7 Code via the Kimi coding endpoint. Requires a KIMI_API_KEY."
        ),
        requires_api_key=True,
        help_url=KIMI_HELP_URL,
        provider=ProviderConfig(
            name="kimi",
            api_base=KIMI_API_BASE,
            api_key_env_var="KIMI_API_KEY",
            extra_headers={"User-Agent": KIMI_USER_AGENT},
        ),
        model=ModelConfig(
            name="kimi-k2.7-code",
            provider="kimi",
            alias="kimi",
            temperature=1.0,
            thinking="high",
            input_price=0.95,
            output_price=4.0,
            supports_images=True,
            auto_compact_threshold=200000,
        ),
    ),
    ProviderPreset(
        key="minimax",
        label="MiniMax (Token Plan)",
        description=(
            "MiniMax-M3 via the MiniMax Token Plan endpoint. Requires a "
            "MINIMAX_API_KEY (Subscription Key)."
        ),
        requires_api_key=True,
        help_url=MINIMAX_HELP_URL,
        provider=ProviderConfig(
            name="minimax",
            api_base=MINIMAX_API_BASE,
            api_key_env_var="MINIMAX_API_KEY",
            api_style="openai-responses",
        ),
        model=ModelConfig(
            name="MiniMax-M3",
            provider="minimax",
            alias="minimax",
            thinking="high",
            input_price=0.0,
            output_price=0.0,
            supports_images=True,
            auto_compact_threshold=400000,
        ),
    ),
    ProviderPreset(
        key="ollama",
        label="Ollama / local",
        description=(
            "Run models locally with Ollama. No API key required; your served "
            "models are detected automatically."
        ),
        requires_api_key=False,
        badge="Local",
    ),
    ProviderPreset(
        key="custom",
        label="Custom OpenAI-compatible",
        description=(
            "Any OpenAI-compatible /chat/completions endpoint (DeepSeek, vLLM, "
            "LM Studio, OpenRouter, ...). Provide a base URL and key."
        ),
        requires_api_key=True,
    ),
]


def preset_for_provider_name(name: str) -> ProviderPreset | None:
    return next((p for p in PRESETS if p.provider and p.provider.name == name), None)


def apply_provider_config(provider: ProviderConfig, model: ModelConfig) -> None:
    config = VibeConfig.get_persisted_config()

    provider_payload = provider.model_dump(mode="json")
    providers = [
        p
        for p in (config.get("providers") or [])
        if isinstance(p, dict) and p.get("name") != provider.name
    ]
    providers.append(provider_payload)
    config["providers"] = providers

    model_payload = model.model_dump(mode="json")
    models = [
        m
        for m in (config.get("models") or [])
        if isinstance(m, dict) and m.get("alias") != model.alias
    ]
    models.append(model_payload)
    config["models"] = models
    config["active_model"] = model.alias

    VibeConfig.dump_config(config)
