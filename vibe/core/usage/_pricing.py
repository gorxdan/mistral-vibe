from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token USD pricing for a model."""

    input_price: float
    output_price: float
    # Cached input is typically discounted (OpenAI: 50% off, Anthropic: 90% off).
    # None → assume full input price for cached tokens (worst-case, matches the
    # existing AgentStats.session_cost convention).
    cached_input_price: float | None = None


# Verified per-million-token USD prices (2026), sourced from each provider's
# official pricing page. Unknown models are intentionally absent — the lookup
# returns None and the card shows — (honest) rather than a guessed figure.
# Keys are lowercased canonical model names. Extend as providers publish more.
#
# Sources:
#   Z.AI (Zhipu/GLM):   https://docs.z.ai/guides/overview/pricing
#   OpenAI:             https://platform.openai.com/docs/pricing
#   Mistral:            https://mistral.ai/pricing/
#   Kimi (Moonshot):    https://platform.moonshot.ai/docs/pricing/chat
_PRICING: dict[str, ModelPricing] = {
    # ── Z.AI / Zhipu GLM family (docs.z.ai/guides/overview/pricing) ──
    "glm-5.2": ModelPricing(input_price=1.4, output_price=4.4, cached_input_price=0.26),
    "glm-5.1": ModelPricing(input_price=1.4, output_price=4.4, cached_input_price=0.26),
    "glm-5": ModelPricing(input_price=1.0, output_price=3.2, cached_input_price=0.2),
    "glm-5-turbo": ModelPricing(
        input_price=1.2, output_price=4.0, cached_input_price=0.24
    ),
    "glm-4.7": ModelPricing(input_price=0.6, output_price=2.2, cached_input_price=0.11),
    "glm-4.6": ModelPricing(input_price=0.6, output_price=2.2, cached_input_price=0.11),
    "glm-4.5": ModelPricing(input_price=0.6, output_price=2.2, cached_input_price=0.11),
    # ── OpenAI (platform.openai.com/docs/pricing, Standard tier) ──
    "gpt-5.5": ModelPricing(input_price=5.0, output_price=30.0, cached_input_price=0.5),
    "gpt-5.4": ModelPricing(
        input_price=2.5, output_price=15.0, cached_input_price=0.25
    ),
    "gpt-5.4-mini": ModelPricing(
        input_price=0.75, output_price=4.5, cached_input_price=0.075
    ),
    "gpt-5.3-codex": ModelPricing(
        input_price=1.75, output_price=14.0, cached_input_price=0.175
    ),
    "chat-latest": ModelPricing(
        input_price=5.0, output_price=30.0, cached_input_price=0.5
    ),
    # ── Mistral (mistral.ai/pricing) ──
    "mistral-large": ModelPricing(input_price=0.5, output_price=1.5),
    "mistral-medium": ModelPricing(input_price=1.5, output_price=7.5),
    "mistral-small": ModelPricing(input_price=0.15, output_price=0.6),
    "devstral-medium": ModelPricing(input_price=0.4, output_price=2.0),
    "devstral-small": ModelPricing(input_price=0.1, output_price=0.3),
    "codestral": ModelPricing(input_price=0.3, output_price=0.9),
    # ── Kimi / Moonshot (platform.moonshot.ai/docs/pricing/chat) ──
    # K2 series coding models. Verified against Kimi's published rates.
    "kimi-k2.7": ModelPricing(input_price=0.6, output_price=2.5),
    "kimi-k2.6": ModelPricing(input_price=0.6, output_price=2.5),
    "kimi-k2.5": ModelPricing(input_price=0.6, output_price=2.5),
    "moonshot-v1-128k": ModelPricing(input_price=2.5, output_price=10.0),
    "moonshot-v1-32k": ModelPricing(input_price=0.55, output_price=2.0),
    "moonshot-v1-8k": ModelPricing(input_price=0.14, output_price=0.28),
}

# Longest-first so 'gpt-4o-mini-2024-08-06' matches 'gpt-4o-mini' (longer base)
# before 'gpt-4o' shadows it. Built once at import.
_BASE_KEYS_LONGEST_FIRST: list[str] = sorted(_PRICING.keys(), key=len, reverse=True)


def lookup_pricing(model_name: str) -> ModelPricing | None:
    """Look up pricing by model name (case-insensitive, prefix-match aware).

    Model names from config often carry provider prefixes or version suffixes
    (e.g. 'openai/gpt-4o-2024-08-06'); this strips prefixes and matches on the
    base name so 'gpt-4o-2024-08-06' → 'gpt-4o'. Longest-base-first matching
    so 'gpt-4o-mini-*' resolves to the cheaper mini tier, not 'gpt-4o'.
    """
    name = model_name.lower().strip()
    # Direct hit.
    if name in _PRICING:
        return _PRICING[name]
    # Strip a 'provider/' or 'provider:' prefix and retry the direct + prefix
    # paths, so neither prefix style needs special-casing below.
    for sep in ("/", ":"):
        if sep in name:
            stripped = name.rsplit(sep, 1)[1]
            if stripped != name:
                hit = lookup_pricing(stripped)
                if hit is not None:
                    return hit
    # Longest-base-first prefix match (handles dated versions like
    # 'gpt-4o-mini-2024-08-06' → 'gpt-4o-mini', not 'gpt-4o').
    for base in _BASE_KEYS_LONGEST_FIRST:
        if name.startswith(base):
            return _PRICING[base]
    return None


def compute_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    pricing: ModelPricing,
) -> float:
    """Cost in USD for a call, applying the cached-input discount when known.

    Cached tokens are clamped to the prompt total — a buggy/overflow upstream
    must never bill cached-rate tokens that weren't actually in the prompt.
    """
    safe_cached = max(0, min(cached_tokens, prompt_tokens))
    non_cached_input = prompt_tokens - safe_cached
    cached_price = pricing.cached_input_price
    if cached_price is None:
        cached_price = pricing.input_price
    input_cost = non_cached_input * pricing.input_price + safe_cached * cached_price
    output_cost = completion_tokens * pricing.output_price
    return (input_cost + output_cost) / 1_000_000
