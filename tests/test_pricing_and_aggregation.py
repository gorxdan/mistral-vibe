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
        p = lookup_pricing("glm-5.2")
        assert p is not None
        assert p.input_price == 1.4
        assert p.output_price == 4.4
        assert p.cached_input_price == 0.26

    def test_case_insensitive(self):
        assert lookup_pricing("GLM-5.2") is not None
        assert lookup_pricing("Mistral-Large") is not None

    def test_dated_version_prefix(self):
        # gpt-5.4-2026-01-01 → gpt-5.4 pricing
        p = lookup_pricing("gpt-5.4-2026-01-01")
        assert p is not None
        assert p.input_price == 2.5

    def test_provider_prefix_stripped(self):
        p = lookup_pricing("zai/glm-5.2")
        assert p is not None
        assert p.input_price == 1.4

    def test_mini_not_shadowed_by_parent(self):
        # gpt-5.4-mini must resolve to its own (cheaper) tier, not gpt-5.4.
        p = lookup_pricing("gpt-5.4-mini")
        assert p is not None
        assert p.input_price == 0.75
        assert p.input_price != 2.5  # not shadowed by gpt-5.4

    def test_unknown_returns_none(self):
        assert lookup_pricing("totally-fake-model") is None


class TestComputeCost:
    def test_basic_no_cache(self):
        cost = compute_cost(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            cached_tokens=0,
            pricing=lookup_pricing("glm-5.2"),
        )
        # 1M * $1.4 + 500K * $4.4 = $1.4 + $2.2 = $3.60
        assert abs(cost - 3.60) < 0.001

    def test_cached_discount(self):
        cost = compute_cost(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_tokens=800_000,
            pricing=lookup_pricing("glm-5.2"),
        )
        # 200K * $1.4 + 800K * $0.26 = $0.28 + $0.208 = $0.488
        assert abs(cost - 0.488) < 0.001

    def test_cached_clamped_to_prompt(self):
        # cached > prompt must not over-bill; clamp to prompt total.
        cost = compute_cost(
            prompt_tokens=100_000,
            completion_tokens=0,
            cached_tokens=500_000,  # exceeds prompt
            pricing=lookup_pricing("glm-5.2"),
        )
        # All 100K at cached rate: 100K * $0.26 = $0.026
        assert abs(cost - 0.026) < 0.001

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
