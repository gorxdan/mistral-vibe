from __future__ import annotations

from typing import Any

PROVIDER_CONTRACT_MIGRATION = "provider_billing_cache_contract_v1"

_KIMI_MODEL_RENAMES = {
    "kimi-k2.7-code": "kimi-for-coding",
    "kimi-k2.7-code-highspeed": "kimi-for-coding-highspeed",
}
_MISTRAL_CACHED_INPUT_PRICES = {
    ("mistral-vibe-cli-latest", 1.5, 7.5): 0.15,
    ("devstral-small-latest", 0.1, 0.3): 0.01,
}
_UNKNOWN_PRESET_PROVIDERS = frozenset({"bedrock", "longcat", "sakana"})


def _cache(provider: dict[str, Any]) -> dict[str, Any]:
    value = provider.get("cache")
    if isinstance(value, dict):
        return value
    value = {}
    provider["cache"] = value
    return value


def _set(value: dict[str, Any], key: str, expected: Any) -> bool:
    if value.get(key) == expected:
        return False
    value[key] = expected
    return True


def _positive_price(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _remove_spoofed_kimi_user_agent(provider: dict[str, Any]) -> bool:
    headers = provider.get("extra_headers")
    if not isinstance(headers, dict):
        return False
    removed = False
    for key, value in list(headers.items()):
        if (
            isinstance(key, str)
            and key.lower() == "user-agent"
            and isinstance(value, str)
            and value.startswith("KimiCLI/")
        ):
            del headers[key]
            removed = True
    return removed


def _migrate_provider(provider: dict[str, Any]) -> bool:
    name = provider.get("name")
    base = str(provider.get("api_base") or "").lower()
    style = provider.get("api_style")
    changed = False

    if name == "kimi" and "api.kimi.com/coding" in base:
        cache = _cache(provider)
        changed = _set(cache, "session_keyed", True) or changed
        if "session_key_field" not in cache:
            cache["session_key_field"] = "prompt_cache_key"
            changed = True
        return _remove_spoofed_kimi_user_agent(provider) or changed

    if name == "zai" and "/api/coding/" in base:
        cache = provider.get("cache")
        if (
            isinstance(cache, dict)
            and cache.get("session_keyed") is True
            and cache.get("session_key_field", "prompt_cache_key") == "prompt_cache_key"
            and not cache.get("cache_key")
            and not cache.get("extra_body")
        ):
            cache["session_keyed"] = False
            changed = True
        return changed

    if name == "openrouter" and "openrouter.ai/api" in base:
        cache = _cache(provider)
        changed = _set(cache, "session_keyed", True) or changed
        return _set(cache, "session_key_field", "session_id") or changed

    if name == "mistral" and (
        provider.get("backend") == "mistral" or "api.mistral.ai" in base
    ):
        cache = _cache(provider)
        changed = _set(cache, "session_keyed", True) or changed
        if "session_key_field" not in cache:
            cache["session_key_field"] = "prompt_cache_key"
            changed = True
        return changed

    if name == "openai-chatgpt" or style == "openai-chatgpt":
        cache = _cache(provider)
        changed = _set(cache, "session_keyed", True) or changed
        if "session_key_field" not in cache:
            cache["session_key_field"] = "prompt_cache_key"
            changed = True
        return changed

    if name == "openai" and "api.openai.com" in base:
        cache = _cache(provider)
        changed = _set(cache, "session_keyed", True) or changed
        if "session_key_field" not in cache:
            cache["session_key_field"] = "prompt_cache_key"
            changed = True
    return changed


def _billing_mode(
    model: dict[str, Any], providers: dict[str, dict[str, Any]]
) -> str | None:
    provider_name = model.get("provider")
    provider = (
        providers.get(provider_name, {}) if isinstance(provider_name, str) else {}
    )
    base = str(provider.get("api_base") or "").lower()

    subscription_provider = (
        provider_name == "kimi" and "api.kimi.com/coding" in base
    ) or (provider_name == "zai" and "/api/coding/" in base)
    if subscription_provider or provider_name in {"minimax", "openai-chatgpt"}:
        return "subscription"
    free_provider = provider_name in {"llamacpp", "ollama"} or (
        provider_name == "openrouter" and model.get("name") == "openrouter/owl-alpha"
    )
    if free_provider:
        return "free"
    if provider_name in _UNKNOWN_PRESET_PROVIDERS:
        return "unknown"
    if (
        provider_name == "openai"
        or _positive_price(model.get("input_price"))
        or _positive_price(model.get("output_price"))
    ):
        return "api"
    return None


def _migrate_model(model: dict[str, Any], providers: dict[str, dict[str, Any]]) -> bool:
    changed = False
    provider_name = model.get("provider")
    provider = (
        providers.get(provider_name, {}) if isinstance(provider_name, str) else {}
    )
    base = str(provider.get("api_base") or "").lower()
    if provider_name == "mistral" and (
        provider.get("backend") == "mistral" or "api.mistral.ai" in base
    ):
        name = model.get("name")
        input_price = model.get("input_price")
        output_price = model.get("output_price")
        cached_price = (
            _MISTRAL_CACHED_INPUT_PRICES.get((
                name,
                float(input_price),
                float(output_price),
            ))
            if isinstance(name, str)
            and isinstance(input_price, int | float)
            and not isinstance(input_price, bool)
            and isinstance(output_price, int | float)
            and not isinstance(output_price, bool)
            else None
        )
        if cached_price is not None and "cached_input_price" not in model:
            model["cached_input_price"] = cached_price
            changed = True

    if provider_name == "kimi" and "api.kimi.com/coding" in base:
        name = model.get("name")
        is_legacy_preset = name in _KIMI_MODEL_RENAMES
        if is_legacy_preset:
            model["name"] = _KIMI_MODEL_RENAMES[name]
            changed = True
        if "pricing_mode" not in model:
            model["pricing_mode"] = (
                "subscription"
                if is_legacy_preset
                or not (
                    _positive_price(model.get("input_price"))
                    or _positive_price(model.get("output_price"))
                )
                else "api"
            )
            changed = True
        if model.get("pricing_mode") == "subscription":
            changed = _set(model, "input_price", 0.0) or changed
            changed = _set(model, "output_price", 0.0) or changed
        return changed

    if "pricing_mode" in model:
        return changed
    if _positive_price(model.get("input_price")) or _positive_price(
        model.get("output_price")
    ):
        model["pricing_mode"] = "api"
        return True
    if mode := _billing_mode(model, providers):
        model["pricing_mode"] = mode
        changed = True
    return changed


def migrate_provider_contracts(data: dict[str, Any]) -> bool:
    applied = data.get("applied_migrations", [])
    if PROVIDER_CONTRACT_MIGRATION in applied:
        return False

    raw_providers = data.get("providers", [])
    providers = {
        provider["name"]: provider
        for provider in raw_providers
        if isinstance(provider, dict) and isinstance(provider.get("name"), str)
    }
    changed = False
    for provider in providers.values():
        changed = _migrate_provider(provider) or changed

    raw_models = data.get("models", [])
    for model in raw_models:
        if isinstance(model, dict):
            changed = _migrate_model(model, providers) or changed

    if not changed and not raw_providers and not raw_models:
        return False
    data["applied_migrations"] = [*applied, PROVIDER_CONTRACT_MIGRATION]
    return True
