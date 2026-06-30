from __future__ import annotations

from dataclasses import dataclass

from vibe.core.auth.openai_oauth import (
    OPENAI_CHATGPT_API_BASE,
    OPENAI_CHATGPT_API_STYLE,
)
from vibe.core.config import (
    ModelConfig,
    ProviderCacheConfig,
    ProviderConfig,
    VibeConfig,
)

ZAI_API_BASE = "https://api.z.ai/api/coding/paas/v4"
KIMI_API_BASE = "https://api.kimi.com/coding/v1"
KIMI_USER_AGENT = "KimiCLI/1.47.0"
ZAI_HELP_URL = "https://z.ai"
KIMI_HELP_URL = "https://kimi.com"
MINIMAX_API_BASE = "https://api.minimax.io/v1"
MINIMAX_HELP_URL = "https://platform.minimax.io/user-center/payment/token-plan"
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_HELP_URL = "https://platform.openai.com/api-keys"
SAKANA_API_BASE = "https://api.sakana.ai/v1"
SAKANA_HELP_URL = "https://sakana.ai"
LONGCAT_API_BASE = "https://api.longcat.chat/openai/v1"
LONGCAT_HELP_URL = "https://longcat.chat/platform/api_keys"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HELP_URL = "https://openrouter.ai/keys"
# Bedrock Mantle serves Claude through the Anthropic Messages API shape; the
# region is part of the host. us-east-1 is the default region; users override
# `region` (and api_base) in config to target another region.
BEDROCK_API_BASE = "https://bedrock-mantle.us-east-1.api.aws/anthropic"
BEDROCK_HELP_URL = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api-keys.html"
)

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
    # Additional models persisted alongside the active one when this preset is
    # applied (e.g. a provider that ships sibling models the user can switch to).
    extra_models: tuple[ModelConfig, ...] = ()


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
            "GLM-5.2 via the ZAI Coding Plan endpoint. Sign in with your Z.ai "
            "account in the browser, or paste a ZAI_API_KEY."
        ),
        requires_api_key=True,
        help_url=ZAI_HELP_URL,
        provider=ProviderConfig(
            name="zai",
            api_base=ZAI_API_BASE,
            api_key_env_var="ZAI_API_KEY",
            discover_models=False,
            # ZAI's prefix cache load-balances across nodes; pin each
            # conversation to one partition so concurrent sessions stop evicting
            # each other's history tail.
            cache=ProviderCacheConfig(session_keyed=True),
        ),
        model=ModelConfig(
            name="glm-5.2",
            provider="zai",
            alias="glm",
            temperature=1.0,
            thinking="max",
            preserve_reasoning=True,
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
            # api.kimi.com serves only the kimi family; discovery adds mislabelled
            # siblings of the model already configured here.
            discover_models=False,
        ),
        model=ModelConfig(
            name="kimi-k2.7-code",
            provider="kimi",
            alias="kimi",
            temperature=None,
            thinking="max",
            preserve_reasoning=True,
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
            discover_models=True,
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
        key="openai",
        label="OpenAI (API key)",
        description=(
            "OpenAI platform models (GPT-5.x, o-series) via the Responses API. "
            "Requires an OPENAI_API_KEY from platform.openai.com (pay-per-token "
            "billing — this is NOT a ChatGPT subscription). Available models "
            "are detected automatically."
        ),
        requires_api_key=True,
        help_url=OPENAI_HELP_URL,
        provider=ProviderConfig(
            name="openai",
            api_base=OPENAI_API_BASE,
            api_key_env_var="OPENAI_API_KEY",
            api_style="openai-responses",
            discover_models=True,
        ),
        model=ModelConfig(
            name="gpt-5.5",
            provider="openai",
            alias="gpt-5.5",
            thinking="high",
            # Pricing left at 0.0; OpenAI model line evolves quickly and the
            # picker live-discovers the current /v1/models list. Override per
            # model in config if you want cost tracking.
            input_price=0.0,
            output_price=0.0,
            supports_images=True,
            auto_compact_threshold=400000,
        ),
    ),
    ProviderPreset(
        key="openai-chatgpt",
        label="Sign in with ChatGPT",
        description=(
            "Use your ChatGPT Plus/Pro/Team subscription instead of an API key "
            "(no per-token billing). Opens a browser to sign in. Unofficial: "
            "routes through OpenAI's ChatGPT backend and may break if OpenAI "
            "changes it."
        ),
        requires_api_key=False,
        badge="Subscription",
        provider=ProviderConfig(
            name="openai-chatgpt",
            api_base=OPENAI_CHATGPT_API_BASE,
            api_style=OPENAI_CHATGPT_API_STYLE,
            # OAuth tokens are resolved from the token store, not an env var.
            api_key_env_var="",
            # Discover the subscription's model set from the codex /models
            # endpoint via the stored OAuth token (no API key needed). The
            # single model block below remains the default; discovery fills the
            # picker with everything the plan permits.
            discover_models=True,
        ),
        model=ModelConfig(
            name="gpt-5.5",
            provider="openai-chatgpt",
            alias="gpt-5.5",
            thinking="high",
            supports_images=True,
            auto_compact_threshold=400000,
        ),
    ),
    ProviderPreset(
        key="sakana",
        label="Sakana Fugu",
        description=(
            "Sakana Fugu (fugu / fugu-ultra) via the Sakana API. A multi-agent "
            "model used like a standard LLM through the OpenAI-compatible "
            "Responses API. Requires a SAKANA_API_KEY."
        ),
        requires_api_key=True,
        help_url=SAKANA_HELP_URL,
        provider=ProviderConfig(
            name="sakana",
            api_base=SAKANA_API_BASE,
            api_key_env_var="SAKANA_API_KEY",
            # Fugu documents the Responses API with reasoning.effort (high /
            # xhigh), matching the openai-responses adapter's effort mapping.
            api_style="openai-responses",
            discover_models=True,
        ),
        model=ModelConfig(
            name="fugu",
            provider="sakana",
            alias="fugu",
            thinking="high",
            # Pricing left at 0.0; override per model in config for cost tracking.
            input_price=0.0,
            output_price=0.0,
            supports_images=True,
            # 1M-token context window; compact well before the ceiling.
            auto_compact_threshold=880000,
        ),
        extra_models=(
            ModelConfig(
                name="fugu-ultra",
                provider="sakana",
                alias="fugu-ultra",
                thinking="high",
                # Pricing left at 0.0; override per model in config for cost tracking.
                input_price=0.0,
                output_price=0.0,
                supports_images=True,
                # Same 1M-token context window as fugu; compact before the ceiling.
                auto_compact_threshold=880000,
            ),
        ),
    ),
    ProviderPreset(
        key="longcat",
        label="LongCat (Meituan)",
        description=(
            "LongCat-2.0 via the LongCat API Platform (OpenAI-compatible). A "
            "high-performance agentic model with a 1M-token context window. "
            "Requires a LONGCAT_API_KEY."
        ),
        requires_api_key=True,
        help_url=LONGCAT_HELP_URL,
        provider=ProviderConfig(
            name="longcat",
            api_base=LONGCAT_API_BASE,
            api_key_env_var="LONGCAT_API_KEY",
            # The platform serves only the LongCat family; discovery would just
            # echo LongCat-2.0 back (already configured below).
            discover_models=False,
        ),
        model=ModelConfig(
            name="LongCat-2.0",
            provider="longcat",
            alias="longcat",
            temperature=1.0,
            # 1M-token context window (128K output). Compact well before the
            # ceiling, matching the other 1M-window presets.
            auto_compact_threshold=880000,
        ),
    ),
    ProviderPreset(
        key="openrouter",
        label="OpenRouter",
        description=(
            "300+ models (Anthropic, OpenAI, Google, Meta, ...) through one "
            "OpenAI-compatible endpoint. Requires an OPENROUTER_API_KEY; "
            "available models are detected automatically."
        ),
        requires_api_key=True,
        help_url=OPENROUTER_HELP_URL,
        provider=ProviderConfig(
            name="openrouter",
            api_base=OPENROUTER_API_BASE,
            api_key_env_var="OPENROUTER_API_KEY",
            # OpenRouter recommends these attribution headers (optional; help
            # with app ranking / free-tier credits). The default openai adapter
            # handles the OpenAI-compatible chat-completions surface.
            extra_headers={"X-Title": "Vibe"},
            # OpenRouter fronts hundreds of models behind one key; discovery
            # fills the picker with everything the key can reach.
            discover_models=True,
        ),
        model=ModelConfig(
            # Owl Alpha: OpenRouter's own agentic coding foundation model.
            # Native tool use, ~1M-token context, $0 input / $0 output. Other
            # reachable models fill the picker via discover_models above.
            # Note: provider logs prompts/completions for this model.
            name="openrouter/owl-alpha",
            provider="openrouter",
            alias="openrouter",
            # Free first-party model; OpenRouter re-bills others at provider
            # pricing. Override per model in config for cost tracking.
            input_price=0.0,
            output_price=0.0,
            # 1.05M-token context window; compact well before the ceiling,
            # matching the other ~1M-window presets.
            auto_compact_threshold=880000,
        ),
    ),
    ProviderPreset(
        key="bedrock",
        label="Amazon Bedrock (Claude)",
        description=(
            "Claude models on Amazon Bedrock via the Mantle endpoint, using an "
            "AWS Bedrock API key (AWS_BEARER_TOKEN_BEDROCK). Runs on "
            "AWS-managed infrastructure; other open Claude models (Opus, Fable) "
            "are selectable by editing the model name in config."
        ),
        requires_api_key=True,
        help_url=BEDROCK_HELP_URL,
        provider=ProviderConfig(
            name="bedrock",
            api_base=BEDROCK_API_BASE,
            api_key_env_var="AWS_BEARER_TOKEN_BEDROCK",
            # Bedrock Mantle speaks the Anthropic Messages API; the adapter pins
            # the region-aware base URL from `region`.
            api_style="bedrock-anthropic",
            region="us-east-1",
            # Bedrock's model catalog lives behind a separate ListFoundationModels
            # endpoint with its own auth; users add models in config.
            discover_models=False,
        ),
        model=ModelConfig(
            # anthropic.<family> model IDs; default to the open Haiku 4.5. Other
            # open models: anthropic.claude-opus-4-8, anthropic.claude-fable-5.
            name="anthropic.claude-haiku-4-5",
            provider="bedrock",
            alias="bedrock",
            supports_images=True,
            auto_compact_threshold=200000,
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
            "LM Studio, ...). Provide a base URL and key."
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

    preset = preset_for_provider_name(provider.name)
    extra_models = preset.extra_models if preset else ()
    new_models = [model, *extra_models]
    new_aliases = {m.alias for m in new_models}
    models = [
        m
        for m in (config.get("models") or [])
        if isinstance(m, dict) and m.get("alias") not in new_aliases
    ]
    models.extend(m.model_dump(mode="json") for m in new_models)
    config["models"] = models
    config["active_model"] = model.alias

    VibeConfig.dump_config(config)
