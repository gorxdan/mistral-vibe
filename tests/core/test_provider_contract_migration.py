from __future__ import annotations

from typing import Any

from vibe.core.config._provider_contract_migration import (
    PROVIDER_CONTRACT_MIGRATION,
    migrate_provider_contracts,
)


def test_migrates_legacy_kimi_coding_contract() -> None:
    data: dict[str, Any] = {
        "providers": [
            {
                "name": "kimi",
                "api_base": "https://api.kimi.com/coding/v1",
                "extra_headers": {"User-Agent": "KimiCLI/1.47.0", "X-Custom": "keep"},
            }
        ],
        "models": [
            {
                "name": "kimi-k2.7-code",
                "provider": "kimi",
                "alias": "kimi",
                "input_price": 0.95,
                "output_price": 4.0,
            },
            {
                "name": "kimi-k2.7-code-highspeed",
                "provider": "kimi",
                "alias": "kimi-fast",
            },
        ],
    }

    assert migrate_provider_contracts(data) is True

    provider = data["providers"][0]
    assert provider["extra_headers"] == {"X-Custom": "keep"}
    assert provider["cache"]["session_keyed"] is True
    assert provider["cache"]["session_key_field"] == "prompt_cache_key"
    standard, highspeed = data["models"]
    assert standard["name"] == "kimi-for-coding"
    assert highspeed["name"] == "kimi-for-coding-highspeed"
    assert standard["pricing_mode"] == "subscription"
    assert standard["input_price"] == 0.0
    assert standard["output_price"] == 0.0
    assert highspeed["pricing_mode"] == "subscription"
    assert PROVIDER_CONTRACT_MIGRATION in data["applied_migrations"]
    assert migrate_provider_contracts(data) is False


def test_migrates_zai_plan_without_undocumented_key() -> None:
    data: dict[str, Any] = {
        "providers": [
            {
                "name": "zai",
                "api_base": "https://api.z.ai/api/coding/paas/v4",
                "cache": {"session_keyed": True},
            }
        ],
        "models": [{"name": "glm-5.2", "provider": "zai", "alias": "glm"}],
    }

    assert migrate_provider_contracts(data) is True

    assert data["providers"][0]["cache"]["session_keyed"] is False
    assert data["models"][0]["pricing_mode"] == "subscription"


def test_migrates_canonical_routing_and_billing_modes() -> None:
    data: dict[str, Any] = {
        "providers": [
            {
                "name": "mistral",
                "api_base": "https://api.mistral.ai/v1",
                "backend": "mistral",
            },
            {"name": "openrouter", "api_base": "https://openrouter.ai/api/v1"},
            {"name": "openai-chatgpt", "api_style": "openai-chatgpt"},
        ],
        "models": [
            {"name": "devstral", "provider": "llamacpp", "alias": "local"},
            {
                "name": "openrouter/owl-alpha",
                "provider": "openrouter",
                "alias": "openrouter",
            },
            {"name": "gpt-5.5", "provider": "openai-chatgpt", "alias": "gpt-5.5"},
        ],
    }

    assert migrate_provider_contracts(data) is True

    providers = {provider["name"]: provider for provider in data["providers"]}
    assert providers["mistral"]["cache"]["session_keyed"] is True
    assert providers["openrouter"]["cache"] == {
        "session_keyed": True,
        "session_key_field": "session_id",
    }
    assert providers["openai-chatgpt"]["cache"]["session_keyed"] is True
    models = {model["alias"]: model for model in data["models"]}
    assert models["local"]["pricing_mode"] == "free"
    assert models["openrouter"]["pricing_mode"] == "free"
    assert models["gpt-5.5"]["pricing_mode"] == "subscription"


def test_preserves_explicit_billing_and_custom_cache_contracts() -> None:
    data: dict[str, Any] = {
        "providers": [
            {
                "name": "kimi",
                "api_base": "https://api.kimi.com/coding/v1",
                "cache": {"session_keyed": True, "session_key_field": "session_id"},
            }
        ],
        "models": [
            {
                "name": "custom-model",
                "provider": "kimi",
                "alias": "custom",
                "pricing_mode": "api",
            }
        ],
    }

    assert migrate_provider_contracts(data) is True
    assert data["providers"][0]["cache"]["session_key_field"] == "session_id"
    assert data["models"][0]["pricing_mode"] == "api"
    assert PROVIDER_CONTRACT_MIGRATION in data["applied_migrations"]


def test_preserves_positive_custom_prices_across_provider_presets() -> None:
    data: dict[str, Any] = {
        "providers": [
            {"name": "minimax", "api_base": "https://api.minimax.io/v1"},
            {"name": "openrouter", "api_base": "https://openrouter.ai/api/v1"},
            {"name": "bedrock", "api_base": "https://bedrock.example/v1"},
            {"name": "sakana", "api_base": "https://api.sakana.ai/v1"},
            {"name": "longcat", "api_base": "https://api.longcat.chat/v1"},
        ],
        "models": [
            {
                "name": "MiniMax-M3",
                "provider": "minimax",
                "alias": "minimax",
                "input_price": 1.0,
                "output_price": 2.0,
            },
            {
                "name": "openrouter/owl-alpha",
                "provider": "openrouter",
                "alias": "owl",
                "input_price": 1.0,
                "output_price": 2.0,
            },
            {
                "name": "claude",
                "provider": "bedrock",
                "alias": "bedrock",
                "input_price": 1.0,
                "output_price": 2.0,
            },
            {
                "name": "fugu",
                "provider": "sakana",
                "alias": "sakana",
                "input_price": 1.0,
                "output_price": 2.0,
            },
            {
                "name": "LongCat-2.0",
                "provider": "longcat",
                "alias": "longcat",
                "input_price": 1.0,
                "output_price": 2.0,
            },
        ],
    }

    assert migrate_provider_contracts(data) is True
    assert {model["pricing_mode"] for model in data["models"]} == {"api"}


def test_backfills_cache_rates_for_legacy_default_mistral_models() -> None:
    data: dict[str, Any] = {
        "providers": [
            {
                "name": "mistral",
                "api_base": "https://api.mistral.ai/v1",
                "backend": "mistral",
            }
        ],
        "models": [
            {
                "name": "mistral-vibe-cli-latest",
                "provider": "mistral",
                "alias": "mistral-medium-3.5",
                "input_price": 1.5,
                "output_price": 7.5,
            },
            {
                "name": "devstral-small-latest",
                "provider": "mistral",
                "alias": "devstral-small",
                "input_price": 0.1,
                "output_price": 0.3,
            },
        ],
    }

    assert migrate_provider_contracts(data) is True
    medium, small = data["models"]
    assert medium["cached_input_price"] == 0.15
    assert small["cached_input_price"] == 0.01
    assert medium["pricing_mode"] == "api"
    assert small["pricing_mode"] == "api"


def test_current_contract_is_marked_once_and_later_user_edits_are_preserved() -> None:
    data: dict[str, Any] = {
        "providers": [
            {
                "name": "mistral",
                "api_base": "https://api.mistral.ai/v1",
                "backend": "mistral",
                "cache": {
                    "session_keyed": True,
                    "session_key_field": "prompt_cache_key",
                },
            }
        ],
        "models": [],
    }

    assert migrate_provider_contracts(data) is True
    data["providers"][0]["cache"]["session_keyed"] = False

    assert migrate_provider_contracts(data) is False
    assert data["providers"][0]["cache"]["session_keyed"] is False
