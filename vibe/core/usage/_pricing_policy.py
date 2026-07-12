from __future__ import annotations

from dataclasses import dataclass

from vibe.core.config.models import ModelConfig, PricingMode
from vibe.core.types import LLMUsage
from vibe.core.usage._pricing import ModelPricing, lookup_pricing

__all__ = ["CostQuote", "quote_cold_reservation", "quote_usage"]


@dataclass(frozen=True, slots=True)
class CostQuote:
    cost_usd: float
    pricing_mode: PricingMode
    estimated: bool
    prompt_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class _Rate:
    per_million: float
    estimated: bool


@dataclass(frozen=True, slots=True)
class _ResolvedRates:
    pricing_mode: PricingMode
    input_rate: _Rate
    cached_input_rate: _Rate
    cache_write_rate: _Rate
    output_rate: _Rate


def quote_usage(
    model: ModelConfig,
    usage: LLMUsage,
    *,
    unpriced_input_price: float,
    unpriced_output_price: float,
) -> CostQuote:
    prompt_tokens = max(usage.prompt_tokens, 0)
    cached_tokens = max(0, min(usage.cached_tokens, prompt_tokens))
    cache_write_tokens = max(
        0, min(usage.cache_write_tokens, prompt_tokens - cached_tokens)
    )
    completion_tokens = max(usage.completion_tokens, 0)
    resolved = _resolve_rates(
        model,
        unpriced_input_price=unpriced_input_price,
        unpriced_output_price=unpriced_output_price,
    )
    if resolved.pricing_mode in {"free", "subscription"}:
        return CostQuote(
            cost_usd=0.0,
            pricing_mode=resolved.pricing_mode,
            estimated=False,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            completion_tokens=completion_tokens,
        )
    if usage.reported_cost_usd is not None:
        return CostQuote(
            cost_usd=usage.reported_cost_usd,
            pricing_mode="api",
            estimated=False,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            completion_tokens=completion_tokens,
        )

    uncached_tokens = prompt_tokens - cached_tokens - cache_write_tokens
    cost_usd = (
        uncached_tokens * resolved.input_rate.per_million
        + cached_tokens * resolved.cached_input_rate.per_million
        + cache_write_tokens * resolved.cache_write_rate.per_million
        + completion_tokens * resolved.output_rate.per_million
    ) / 1_000_000
    estimated = resolved.pricing_mode == "unknown" or any((
        uncached_tokens > 0 and resolved.input_rate.estimated,
        cached_tokens > 0 and resolved.cached_input_rate.estimated,
        cache_write_tokens > 0 and resolved.cache_write_rate.estimated,
        completion_tokens > 0 and resolved.output_rate.estimated,
    ))
    return CostQuote(
        cost_usd=cost_usd,
        pricing_mode=resolved.pricing_mode,
        estimated=estimated,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        completion_tokens=completion_tokens,
    )


def quote_cold_reservation(
    model: ModelConfig,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    unpriced_input_price: float,
    unpriced_output_price: float,
) -> CostQuote:
    safe_prompt_tokens = max(prompt_tokens, 0)
    safe_completion_tokens = max(completion_tokens, 0)
    resolved = _resolve_rates(
        model,
        unpriced_input_price=unpriced_input_price,
        unpriced_output_price=unpriced_output_price,
    )
    if resolved.pricing_mode in {"free", "subscription"}:
        return CostQuote(
            cost_usd=0.0,
            pricing_mode=resolved.pricing_mode,
            estimated=False,
            prompt_tokens=safe_prompt_tokens,
            cached_tokens=0,
            cache_write_tokens=0,
            completion_tokens=safe_completion_tokens,
        )

    prompt_rate = max(
        resolved.input_rate.per_million, resolved.cache_write_rate.per_million
    )
    cost_usd = (
        safe_prompt_tokens * prompt_rate
        + safe_completion_tokens * resolved.output_rate.per_million
    ) / 1_000_000
    estimated = resolved.pricing_mode == "unknown" or any((
        safe_prompt_tokens > 0
        and (resolved.input_rate.estimated or resolved.cache_write_rate.estimated),
        safe_completion_tokens > 0 and resolved.output_rate.estimated,
    ))
    return CostQuote(
        cost_usd=cost_usd,
        pricing_mode=resolved.pricing_mode,
        estimated=estimated,
        prompt_tokens=safe_prompt_tokens,
        cached_tokens=0,
        cache_write_tokens=0,
        completion_tokens=safe_completion_tokens,
    )


def _resolve_rates(
    model: ModelConfig, *, unpriced_input_price: float, unpriced_output_price: float
) -> _ResolvedRates:
    pricing_mode = model.pricing_mode
    if pricing_mode in {"free", "subscription"}:
        zero = _Rate(per_million=0.0, estimated=False)
        return _ResolvedRates(
            pricing_mode=pricing_mode,
            input_rate=zero,
            cached_input_rate=zero,
            cache_write_rate=zero,
            output_rate=zero,
        )
    if pricing_mode == "unknown":
        return _unknown_rates(
            unpriced_input_price=unpriced_input_price,
            unpriced_output_price=unpriced_output_price,
        )

    has_configured_base_rate = model.input_price > 0 or model.output_price > 0
    if has_configured_base_rate:
        return _configured_api_rates(model)

    table = lookup_pricing(model.name)
    if table is not None:
        return _table_api_rates(model, table)
    if pricing_mode == "auto":
        return _unknown_rates(
            unpriced_input_price=unpriced_input_price,
            unpriced_output_price=unpriced_output_price,
        )
    return _unpriced_api_rates(
        unpriced_input_price=unpriced_input_price,
        unpriced_output_price=unpriced_output_price,
    )


def _unknown_rates(
    *, unpriced_input_price: float, unpriced_output_price: float
) -> _ResolvedRates:
    input_rate = _Rate(per_million=max(unpriced_input_price, 0.0), estimated=True)
    output_rate = _Rate(per_million=max(unpriced_output_price, 0.0), estimated=True)
    return _ResolvedRates(
        pricing_mode="unknown",
        input_rate=input_rate,
        cached_input_rate=input_rate,
        cache_write_rate=input_rate,
        output_rate=output_rate,
    )


def _configured_api_rates(model: ModelConfig) -> _ResolvedRates:
    input_rate = _Rate(per_million=model.input_price, estimated=False)
    output_rate = _Rate(per_million=model.output_price, estimated=False)
    return _ResolvedRates(
        pricing_mode="api",
        input_rate=input_rate,
        cached_input_rate=_cache_rate(model.cached_input_price, input_rate),
        cache_write_rate=_cache_rate(model.cache_write_input_price, input_rate),
        output_rate=output_rate,
    )


def _table_api_rates(model: ModelConfig, table: ModelPricing) -> _ResolvedRates:
    input_rate = _Rate(per_million=table.input_price, estimated=False)
    return _ResolvedRates(
        pricing_mode="api",
        input_rate=input_rate,
        cached_input_rate=_cache_rate(
            model.cached_input_price
            if model.cached_input_price is not None
            else table.cached_input_price,
            input_rate,
        ),
        cache_write_rate=_cache_rate(
            model.cache_write_input_price
            if model.cache_write_input_price is not None
            else table.cache_write_input_price,
            input_rate,
        ),
        output_rate=_Rate(per_million=table.output_price, estimated=False),
    )


def _unpriced_api_rates(
    *, unpriced_input_price: float, unpriced_output_price: float
) -> _ResolvedRates:
    input_rate = _Rate(per_million=max(unpriced_input_price, 0.0), estimated=True)
    return _ResolvedRates(
        pricing_mode="api",
        input_rate=input_rate,
        cached_input_rate=input_rate,
        cache_write_rate=input_rate,
        output_rate=_Rate(per_million=max(unpriced_output_price, 0.0), estimated=True),
    )


def _cache_rate(configured: float | None, input_rate: _Rate) -> _Rate:
    if configured is not None:
        return _Rate(per_million=configured, estimated=False)
    return _Rate(per_million=input_rate.per_million, estimated=True)
