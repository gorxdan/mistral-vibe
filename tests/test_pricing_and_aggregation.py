from __future__ import annotations

from vibe.core.types import LLMUsage
from vibe.core.usage import UsageRecord, compute_cost, lookup_pricing, summarize


def _rec(
    *,
    provider: str,
    model: str,
    prompt: int,
    completion: int,
    cached: int = 0,
    reasoning: int = 0,
    ts: float,
    session: str = "s1",
    cost: float = 0.0,
    harness: bool = False,
) -> UsageRecord:
    return UsageRecord.from_usage(
        timestamp=ts,
        provider=provider,
        model=model,
        usage=LLMUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
        ),
        cost_usd=cost,
        duration_s=1.0,
        session_id=session,
        harness=harness,
    )


class TestPricingTable:
    def test_direct_hit(self):
        p = lookup_pricing("gpt-4o")
        assert p is not None
        assert p.input_price == 2.5
        assert p.output_price == 10.0
        assert p.cached_input_price == 1.25

    def test_case_insensitive(self):
        assert lookup_pricing("GPT-4O") is not None
        assert lookup_pricing("Mistral-Large") is not None

    def test_dated_version_prefix(self):
        # gpt-4o-2024-08-06 → gpt-4o pricing
        p = lookup_pricing("gpt-4o-2024-08-06")
        assert p is not None
        assert p.input_price == 2.5

    def test_provider_prefix_stripped(self):
        p = lookup_pricing("openai/gpt-4o")
        assert p is not None
        assert p.input_price == 2.5

    def test_unknown_returns_none(self):
        assert lookup_pricing("glm-5.2") is None
        assert lookup_pricing("totally-fake-model") is None


class TestComputeCost:
    def test_basic_no_cache(self):
        cost = compute_cost(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            cached_tokens=0,
            pricing=lookup_pricing("gpt-4o"),
        )
        # 1M * $2.5 + 500K * $10 = $2.5 + $5.0 = $7.50
        assert abs(cost - 7.50) < 0.001

    def test_cached_discount(self):
        cost = compute_cost(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=800_000,
            pricing=lookup_pricing("gpt-4o"),
        )
        # 200K * $2.5 + 800K * $1.25 (50% off) = $0.5 + $1.0 = $1.50
        assert abs(cost - 1.50) < 0.001

    def test_no_cached_price_uses_input(self):
        from vibe.core.usage import ModelPricing

        cost = compute_cost(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=500_000,
            pricing=ModelPricing(input_price=2.0, output_price=6.0),
        )
        # No cached discount: 500K*$2 + 500K*$2 = $2.0
        assert abs(cost - 2.0) < 0.001


class TestReasoningTokensCapture:
    def test_reasoning_flows_through_breakdown(self):
        now = 1000.0
        records = [
            _rec(
                provider="zai",
                model="glm-5.2",
                prompt=1000,
                completion=500,
                reasoning=300,
                ts=now,
            )
        ]
        summary = summarize(records, now=now)
        assert summary.providers[0].reasoning_tokens == 300
        assert summary.providers[0].models[0].reasoning_tokens == 300


class TestHarnessSplit:
    def test_split_separates_harness_calls(self):
        now = 100_000.0
        records = [
            _rec(provider="p", model="m", prompt=1000, completion=500, ts=now, cost=1.0),
            _rec(
                provider="p",
                model="m",
                prompt=200,
                completion=100,
                ts=now,
                cost=0.2,
                harness=True,
            ),
        ]
        summary = summarize(records, now=now)
        split = summary.harness
        assert split.user_tokens == 1500
        assert split.user_cost == 1.0
        assert split.harness_tokens == 300
        assert split.harness_cost == 0.2

    def test_no_harness_calls(self):
        records = [_rec(provider="p", model="m", prompt=10, completion=5, ts=1.0)]
        summary = summarize(records, now=2.0)
        assert summary.harness.harness_tokens == 0
        assert summary.harness.harness_cost == 0.0


class TestDailyBuckets:
    def test_14_day_series(self):
        now = 14 * 86400.0  # day 14
        records = [
            _rec(provider="p", model="m", prompt=100, completion=50, ts=now - 86400),
            _rec(provider="p", model="m", prompt=200, completion=100, ts=now),
        ]
        summary = summarize(records, now=now)
        assert len(summary.daily) == 14
        # now-86400 = day 13 (second-to-last), now = day 14 (last).
        assert summary.daily[-1].total_tokens == 300
        assert summary.daily[-2].total_tokens == 150
        assert summary.daily[-3].total_tokens == 0

    def test_empty_days_zero(self):
        summary = summarize([], now=86400.0)
        assert len(summary.daily) == 14
        assert all(d.total_tokens == 0 for d in summary.daily)


class TestThirtyDayWindow:
    def test_window_present(self):
        summary = summarize([], now=1.0)
        labels = [w.label for w in summary.windows]
        assert "Last 30 days" in labels

    def test_30_day_includes_old_records(self):
        now = 100_000_000.0
        records = [
            _rec(provider="p", model="m", prompt=10, completion=5, ts=now - 20 * 86400),
        ]
        summary = summarize(records, now=now)
        by_label = {w.label: w for w in summary.windows}
        # 20 days old: in 30d window, NOT in 7d.
        assert by_label["Last 30 days"].calls == 1
        assert by_label["Last 7 days"].calls == 0
