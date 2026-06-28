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


# Verified per-million-token prices. Sources cited inline. Unknown models are
# intentionally absent — the lookup returns None and the card shows — (honest)
# rather than a guessed dollar figure. Extend this table as prices are verified.
_PRICING: dict[str, ModelPricing] = {
    # Mistral — verified from repo config presets (_settings.py:959-960).
    "mistral-large": ModelPricing(input_price=1.5, output_price=7.5),
    # GPT-4o family — verified from OpenAI pricing page (openai.com/api/pricing).
    "gpt-4o": ModelPricing(input_price=2.5, output_price=10.0, cached_input_price=1.25),
    "gpt-4o-mini": ModelPricing(
        input_price=0.15, output_price=0.6, cached_input_price=0.075
    ),
    # Claude 3.5 — verified from Anthropic pricing page.
    "claude-3-5-sonnet": ModelPricing(
        input_price=3.0, output_price=15.0, cached_input_price=0.3
    ),
    "claude-3-5-haiku": ModelPricing(
        input_price=0.8, output_price=4.0, cached_input_price=0.08
    ),
}


def lookup_pricing(model_name: str) -> ModelPricing | None:
    """Look up pricing by model name (case-insensitive, prefix-match aware).

    Model names from config often carry provider prefixes or version suffixes
    (e.g. 'openai/gpt-4o-2024-08-06'); this strips prefixes and matches on the
    base name so 'gpt-4o-2024-08-06' → 'gpt-4o'.
    """
    name = model_name.lower().strip()
    # Direct hit.
    if name in _PRICING:
        return _PRICING[name]
    # Try matching a known base name as a prefix (handles dated versions).
    for base, pricing in _PRICING.items():
        if name.startswith(base):
            return pricing
    # Strip a 'provider/' prefix and retry.
    if "/" in name:
        return lookup_pricing(name.split("/", 1)[1])
    return None


def compute_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    pricing: ModelPricing,
) -> float:
    """Cost in USD for a call, applying the cached-input discount when known."""
    non_cached_input = max(prompt_tokens - cached_tokens, 0)
    cached_price = pricing.cached_input_price
    if cached_price is None:
        cached_price = pricing.input_price
    input_cost = (non_cached_input * pricing.input_price + cached_tokens * cached_price)
    output_cost = completion_tokens * pricing.output_price
    return (input_cost + output_cost) / 1_000_000
