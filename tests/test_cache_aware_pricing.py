from __future__ import annotations

import pytest

from vibe.core.config import ProviderConfig
from vibe.core.config.models import ModelConfig, PricingMode
from vibe.core.types import LLMUsage
from vibe.core.usage import (
    CallKind,
    CostQuote,
    SpendBroker,
    SpendLimits,
    UsageMeter,
    UsageRecord,
    quote_cold_reservation,
    quote_usage,
    summarize,
    usage_cost,
)
from vibe.core.usage._context import (
    SpendAmount,
    SpendContext,
    SpendEnvelope,
    SpendPurpose,
    SpendReservation,
    SpendScopeKind,
)


def _model(
    *,
    name: str = "custom-model",
    pricing_mode: PricingMode = "api",
    input_price: float = 0.0,
    output_price: float = 0.0,
    cached_input_price: float | None = None,
    cache_write_input_price: float | None = None,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="test",
        alias="test",
        pricing_mode=pricing_mode,
        input_price=input_price,
        output_price=output_price,
        cached_input_price=cached_input_price,
        cache_write_input_price=cache_write_input_price,
    )


def test_quote_usage_separates_uncached_read_and_write_prices() -> None:
    quote = quote_usage(
        _model(
            input_price=2.0,
            output_price=8.0,
            cached_input_price=0.2,
            cache_write_input_price=2.5,
        ),
        LLMUsage(
            prompt_tokens=100,
            cached_tokens=60,
            cache_write_tokens=20,
            completion_tokens=5,
        ),
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )

    assert isinstance(quote, CostQuote)
    assert quote.pricing_mode == "api"
    assert quote.estimated is False
    assert quote.prompt_tokens == 100
    assert quote.cached_tokens == 60
    assert quote.cache_write_tokens == 20
    assert quote.completion_tokens == 5
    assert quote.cost_usd == pytest.approx(142 / 1_000_000)


def test_provider_reported_cost_overrides_rate_estimates() -> None:
    quote = quote_usage(
        _model(name="openrouter/routed-model", pricing_mode="unknown"),
        LLMUsage(
            prompt_tokens=1_000,
            completion_tokens=100,
            cached_tokens=800,
            reported_cost_usd=0.0042,
        ),
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )

    assert quote.cost_usd == pytest.approx(0.0042)
    assert quote.pricing_mode == "api"
    assert quote.estimated is False


@pytest.mark.parametrize("price", [-1.0, float("inf"), float("nan")])
def test_model_prices_must_be_finite_and_nonnegative(price: float) -> None:
    with pytest.raises(ValueError):
        _model(input_price=price)


def test_auto_uses_table_and_unknown_uses_estimate() -> None:
    table_quote = quote_usage(
        _model(name="gpt-5.6-luna", pricing_mode="auto"),
        LLMUsage(
            prompt_tokens=100,
            cached_tokens=40,
            cache_write_tokens=20,
            completion_tokens=10,
        ),
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )
    unknown_quote = quote_usage(
        _model(name="not-in-the-table", pricing_mode="auto"),
        LLMUsage(prompt_tokens=100, completion_tokens=10),
        unpriced_input_price=3.0,
        unpriced_output_price=7.0,
    )

    assert table_quote.pricing_mode == "api"
    assert table_quote.estimated is False
    assert table_quote.cost_usd == pytest.approx(129 / 1_000_000)
    assert unknown_quote.pricing_mode == "unknown"
    assert unknown_quote.estimated is True
    assert unknown_quote.cost_usd == pytest.approx(370 / 1_000_000)


def test_unknown_mode_ignores_configured_and_table_prices() -> None:
    quote = quote_usage(
        _model(
            name="gpt-5.6-luna",
            pricing_mode="unknown",
            input_price=2.0,
            output_price=8.0,
        ),
        LLMUsage(prompt_tokens=100, completion_tokens=10),
        unpriced_input_price=3.0,
        unpriced_output_price=7.0,
    )

    assert quote.pricing_mode == "unknown"
    assert quote.estimated is True
    assert quote.cost_usd == pytest.approx(370 / 1_000_000)


def test_configured_base_prices_do_not_blend_with_a_known_model_table() -> None:
    quote = quote_usage(
        _model(
            name="gpt-5.6-luna", pricing_mode="auto", input_price=2.0, output_price=0.0
        ),
        LLMUsage(
            prompt_tokens=10, cached_tokens=4, cache_write_tokens=3, completion_tokens=5
        ),
        unpriced_input_price=3.0,
        unpriced_output_price=7.0,
    )

    assert quote.pricing_mode == "api"
    assert quote.estimated is True
    assert quote.cost_usd == pytest.approx(20 / 1_000_000)


def test_configured_cache_prices_override_a_known_model_table() -> None:
    quote = quote_usage(
        _model(
            name="gpt-5.6-luna",
            pricing_mode="auto",
            cached_input_price=0.25,
            cache_write_input_price=3.0,
        ),
        LLMUsage(
            prompt_tokens=10, cached_tokens=4, cache_write_tokens=3, completion_tokens=1
        ),
        unpriced_input_price=3.0,
        unpriced_output_price=7.0,
    )

    assert quote.pricing_mode == "api"
    assert quote.estimated is False
    assert quote.cost_usd == pytest.approx(19 / 1_000_000)


def test_verified_mistral_and_kimi_cached_read_prices() -> None:
    mistral_quote = quote_usage(
        _model(name="mistral-large", pricing_mode="auto"),
        LLMUsage(prompt_tokens=100, cached_tokens=50),
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )
    kimi_quote = quote_usage(
        _model(name="kimi-k2.7", pricing_mode="auto"),
        LLMUsage(prompt_tokens=100, cached_tokens=80, completion_tokens=10),
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )

    assert mistral_quote.estimated is False
    assert mistral_quote.cost_usd == pytest.approx(27.5 / 1_000_000)
    assert kimi_quote.estimated is False
    assert kimi_quote.cost_usd == pytest.approx(74.2 / 1_000_000)


@pytest.mark.parametrize("pricing_mode", ["free", "subscription"])
def test_free_and_subscription_quotes_are_exactly_zero(
    pricing_mode: PricingMode,
) -> None:
    quote = quote_usage(
        _model(name="gpt-5.5", pricing_mode=pricing_mode),
        LLMUsage(
            prompt_tokens=100,
            cached_tokens=50,
            cache_write_tokens=25,
            completion_tokens=20,
            reported_cost_usd=0.42,
        ),
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )

    assert quote.pricing_mode == pricing_mode
    assert quote.estimated is False
    assert quote.cost_usd == 0.0


def test_cold_reservation_covers_the_worst_case_cache_write_rate() -> None:
    quote = quote_cold_reservation(
        _model(
            input_price=2.0,
            output_price=8.0,
            cached_input_price=0.2,
            cache_write_input_price=2.5,
        ),
        prompt_tokens=100,
        completion_tokens=5,
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )

    assert quote.cached_tokens == 0
    assert quote.cache_write_tokens == 0
    assert quote.cost_usd == pytest.approx(290 / 1_000_000)


def test_quote_clamps_overlapping_cache_classes() -> None:
    quote = quote_usage(
        _model(
            input_price=2.0,
            output_price=8.0,
            cached_input_price=0.2,
            cache_write_input_price=2.5,
        ),
        LLMUsage(prompt_tokens=10, cached_tokens=8, cache_write_tokens=8),
        unpriced_input_price=10.0,
        unpriced_output_price=30.0,
    )

    assert quote.cached_tokens == 8
    assert quote.cache_write_tokens == 2
    assert quote.cost_usd == pytest.approx((8 * 0.2 + 2 * 2.5) / 1_000_000)


def test_llm_usage_adds_cache_write_tokens() -> None:
    usage = LLMUsage(cache_write_tokens=3, reported_cost_usd=0.1) + LLMUsage(
        cache_write_tokens=4, reported_cost_usd=0.2
    )

    assert usage.cache_write_tokens == 7
    assert usage.reported_cost_usd == pytest.approx(0.3)


def test_usage_meter_uses_configurable_unpriced_rates() -> None:
    model = _model(name="not-in-the-table", pricing_mode="auto")
    usage = LLMUsage(prompt_tokens=100, completion_tokens=10)
    meter = UsageMeter(
        "session",
        limits=SpendLimits(max_cost_usd=0.0003),
        unpriced_input_usd_per_million=3.0,
        unpriced_output_usd_per_million=7.0,
    )

    quote = meter.quote(model, usage)
    reservation_quote = meter.quote_reservation(model, usage)

    assert quote.cost_usd == pytest.approx(370 / 1_000_000)
    assert reservation_quote.cost_usd == pytest.approx(370 / 1_000_000)
    assert meter.try_reserve(110, estimated_cost_usd=quote.cost_usd) is None
    assert usage_cost(model, usage) == pytest.approx(1300 / 1_000_000)


def test_usage_meter_reports_normalized_auxiliary_usage_to_owner() -> None:
    observed: list[tuple[LLMUsage, float, bool]] = []
    meter = UsageMeter("session", on_reconcile=lambda *args: observed.append(args))
    reservation = meter.try_reserve(20, estimated_cost_usd=0.01)
    assert reservation is not None

    meter.reconcile(
        reservation,
        usage=LLMUsage(
            prompt_tokens=10,
            completion_tokens=2,
            cached_tokens=8,
            cache_write_tokens=8,
            reasoning_tokens=3,
        ),
        model=_model(input_price=1.0, output_price=2.0),
        provider=ProviderConfig(name="p", api_base="https://example.test/v1"),
        call_kind=CallKind.MEMORY_SELECT,
        duration_s=0.1,
    )

    usage, cost, estimated = observed[0]
    assert usage.cached_tokens == 8
    assert usage.cache_write_tokens == 2
    assert usage.reasoning_tokens == 2
    assert cost > 0.0
    assert estimated is True


def test_usage_record_keeps_legacy_records_and_quote_metadata() -> None:
    legacy = UsageRecord.model_validate({
        "timestamp": 1.0,
        "provider": "p",
        "model": "m",
    })
    record = UsageRecord.from_usage(
        timestamp=2.0,
        provider="p",
        model="m",
        usage=LLMUsage(prompt_tokens=20, cached_tokens=8, cache_write_tokens=4),
        cost_usd=0.25,
        cost_estimated=True,
        pricing_mode="unknown",
        duration_s=0.5,
        session_id="s",
    )

    assert legacy.cache_write_tokens == 0
    assert legacy.cost_estimated is True
    assert legacy.pricing_mode == "unknown"
    assert record.cache_write_tokens == 4
    assert record.non_cached_input == 8
    assert record.cost_estimated is True
    assert record.pricing_mode == "unknown"


def test_aggregation_carries_cache_writes_and_estimated_cost_state() -> None:
    records = [
        UsageRecord(
            timestamp=100.0,
            provider="p",
            model="m",
            prompt_tokens=20,
            cached_tokens=8,
            cache_write_tokens=4,
            completion_tokens=2,
            cost_usd=0.25,
            pricing_mode="api",
        ),
        UsageRecord(
            timestamp=100.0,
            provider="p",
            model="m",
            prompt_tokens=10,
            cache_write_tokens=3,
            completion_tokens=1,
            cost_usd=0.10,
            cost_estimated=True,
            pricing_mode="unknown",
        ),
    ]

    summary = summarize(records, now=100.0)
    model = summary.providers[0].models[0]
    provider = summary.providers[0]
    window = summary.windows[0]

    assert model.cache_write_tokens == 7
    assert model.cost_estimated is True
    assert model.pricing_modes == frozenset({"api", "unknown"})
    assert provider.cache_write_tokens == 7
    assert provider.cost_estimated is True
    assert provider.pricing_modes == frozenset({"api", "unknown"})
    assert window.cache_write_tokens == 7
    assert window.cost_estimated is True
    assert window.pricing_modes == frozenset({"api", "unknown"})
    assert summary.cost_estimated is True


def test_broker_round_trip_keeps_estimated_cache_write_usage(tmp_path) -> None:
    broker = SpendBroker(tmp_path)
    broker.define_envelope(
        SpendEnvelope(scope_id="session", kind=SpendScopeKind.SESSION)
    )
    broker.define_envelope(
        SpendEnvelope(
            scope_id="agent", kind=SpendScopeKind.AGENT, parent_scope_id="session"
        )
    )
    reservation = broker.try_reserve(
        SpendContext(scope_id="agent", purpose=SpendPurpose.PRIMARY),
        SpendAmount(prompt_tokens=100, cached_tokens=50, cache_write_tokens=20),
        lease_s=60.0,
    )
    assert isinstance(reservation, SpendReservation)

    broker.mark_dispatched(reservation)
    quote = quote_usage(
        _model(name="not-in-the-table", pricing_mode="auto"),
        LLMUsage(prompt_tokens=80, cached_tokens=30, cache_write_tokens=10),
        unpriced_input_price=3.0,
        unpriced_output_price=7.0,
    )
    settlement = broker.reconcile(
        reservation,
        SpendAmount(
            prompt_tokens=quote.prompt_tokens,
            cached_tokens=quote.cached_tokens,
            cache_write_tokens=quote.cache_write_tokens,
            completion_tokens=quote.completion_tokens,
            cost_usd=quote.cost_usd,
        ),
        estimated=quote.estimated,
    )
    duplicate = broker.reconcile(
        reservation,
        SpendAmount(
            prompt_tokens=quote.prompt_tokens,
            cached_tokens=quote.cached_tokens,
            cache_write_tokens=quote.cache_write_tokens,
            completion_tokens=quote.completion_tokens,
            cost_usd=quote.cost_usd,
        ),
        estimated=quote.estimated,
    )

    snapshot = SpendBroker(tmp_path).snapshot("session")

    assert settlement.estimated is True
    assert settlement.usage_reported is True
    assert duplicate.applied is False
    assert snapshot.spent.cache_write_tokens == 10
