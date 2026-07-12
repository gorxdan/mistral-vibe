from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import time

from vibe.core.config.models import PricingMode
from vibe.core.usage.models import UsageRecord

_HOUR = 3600.0
_DAY = 86400.0


@dataclass(frozen=True)
class ModelBreakdown:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    cost_usd: float
    cost_estimated: bool
    pricing_modes: frozenset[PricingMode]
    calls: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ProviderBreakdown:
    provider: str
    models: list[ModelBreakdown] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cost_estimated: bool = False
    pricing_modes: frozenset[PricingMode] = field(default_factory=frozenset)
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class WindowRollup:
    label: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    cost_usd: float
    cost_estimated: bool
    pricing_modes: frozenset[PricingMode]
    calls: int
    sessions: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class DailyBucket:
    """One day's token volume for the sparkline. ``day`` is days since epoch."""

    day: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class HarnessSplit:
    """User-driven vs harness-internal spend."""

    user_tokens: int
    user_cost: float
    user_cost_estimated: bool
    harness_tokens: int
    harness_cost: float
    harness_cost_estimated: bool


@dataclass(frozen=True)
class UsageSummary:
    providers: list[ProviderBreakdown]
    windows: list[WindowRollup]
    daily: list[DailyBucket]
    harness: HarnessSplit
    grand_total_tokens: int
    grand_total_cost: float
    cost_estimated: bool


def _breakdown(records: list[UsageRecord]) -> list[ProviderBreakdown]:
    by_pm: dict[tuple[str, str], ModelBreakdown] = {}
    p_prompt: dict[str, int] = defaultdict(int)
    p_comp: dict[str, int] = defaultdict(int)
    p_cache: dict[str, int] = defaultdict(int)
    p_cache_write: dict[str, int] = defaultdict(int)
    p_reason: dict[str, int] = defaultdict(int)
    p_cost: dict[str, float] = defaultdict(float)
    p_estimated: dict[str, bool] = defaultdict(bool)
    p_modes: dict[str, set[PricingMode]] = defaultdict(set)
    p_calls: dict[str, int] = defaultdict(int)

    def _merge(prev: ModelBreakdown | None, r: UsageRecord) -> ModelBreakdown:
        if prev is None:
            return ModelBreakdown(
                provider=r.provider,
                model=r.model,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                cached_tokens=r.cached_tokens,
                cache_write_tokens=r.cache_write_tokens,
                reasoning_tokens=r.reasoning_tokens,
                cost_usd=r.cost_usd,
                cost_estimated=r.cost_estimated,
                pricing_modes=frozenset({r.pricing_mode}),
                calls=1,
            )
        return ModelBreakdown(
            provider=r.provider,
            model=r.model,
            prompt_tokens=prev.prompt_tokens + r.prompt_tokens,
            completion_tokens=prev.completion_tokens + r.completion_tokens,
            cached_tokens=prev.cached_tokens + r.cached_tokens,
            cache_write_tokens=prev.cache_write_tokens + r.cache_write_tokens,
            reasoning_tokens=prev.reasoning_tokens + r.reasoning_tokens,
            cost_usd=prev.cost_usd + r.cost_usd,
            cost_estimated=prev.cost_estimated or r.cost_estimated,
            pricing_modes=frozenset[PricingMode]((*prev.pricing_modes, r.pricing_mode)),
            calls=prev.calls + 1,
        )

    for r in records:
        key = (r.provider, r.model)
        by_pm[key] = _merge(by_pm.get(key), r)
        p_prompt[r.provider] += r.prompt_tokens
        p_comp[r.provider] += r.completion_tokens
        p_cache[r.provider] += r.cached_tokens
        p_cache_write[r.provider] += r.cache_write_tokens
        p_reason[r.provider] += r.reasoning_tokens
        p_cost[r.provider] += r.cost_usd
        p_estimated[r.provider] = p_estimated[r.provider] or r.cost_estimated
        p_modes[r.provider].add(r.pricing_mode)
        p_calls[r.provider] += 1

    # Provider order: descending total tokens; models within descending tokens.
    models_by_provider: dict[str, list[ModelBreakdown]] = defaultdict(list)
    for mb in by_pm.values():
        models_by_provider[mb.provider].append(mb)
    for lst in models_by_provider.values():
        lst.sort(key=lambda m: m.total_tokens, reverse=True)

    providers = [
        ProviderBreakdown(
            provider=p,
            models=models_by_provider[p],
            prompt_tokens=p_prompt[p],
            completion_tokens=p_comp[p],
            cached_tokens=p_cache[p],
            cache_write_tokens=p_cache_write[p],
            reasoning_tokens=p_reason[p],
            cost_usd=p_cost[p],
            cost_estimated=p_estimated[p],
            pricing_modes=frozenset(p_modes[p]),
            calls=p_calls[p],
        )
        for p in models_by_provider
    ]
    providers.sort(key=lambda pr: pr.total_tokens, reverse=True)
    return providers


def _window(
    records: list[UsageRecord], label: str, seconds: float, now: float
) -> WindowRollup:
    cutoff = now - seconds
    recs = [r for r in records if r.timestamp >= cutoff]
    sessions = {r.session_id for r in recs if r.session_id}
    return WindowRollup(
        label=label,
        seconds=seconds,
        prompt_tokens=sum(r.prompt_tokens for r in recs),
        completion_tokens=sum(r.completion_tokens for r in recs),
        cached_tokens=sum(r.cached_tokens for r in recs),
        cache_write_tokens=sum(r.cache_write_tokens for r in recs),
        reasoning_tokens=sum(r.reasoning_tokens for r in recs),
        cost_usd=sum(r.cost_usd for r in recs),
        cost_estimated=any(r.cost_estimated for r in recs),
        pricing_modes=frozenset(r.pricing_mode for r in recs),
        calls=len(recs),
        sessions=len(sessions),
    )


def _daily(records: list[UsageRecord], days: int, now: float) -> list[DailyBucket]:
    """Per-day buckets for the last ``days`` days, oldest-first."""
    today = int(now // _DAY)
    start_day = today - days + 1
    buckets: dict[int, DailyBucket] = {
        d: DailyBucket(day=d, prompt_tokens=0, completion_tokens=0, cost_usd=0.0)
        for d in range(start_day, today + 1)
    }
    for r in records:
        day = int(r.timestamp // _DAY)
        b = buckets.get(day)
        if b is None:
            continue
        buckets[day] = DailyBucket(
            day=day,
            prompt_tokens=b.prompt_tokens + r.prompt_tokens,
            completion_tokens=b.completion_tokens + r.completion_tokens,
            cost_usd=b.cost_usd + r.cost_usd,
        )
    return [buckets[d] for d in range(start_day, today + 1)]


def _harness_split(records: list[UsageRecord]) -> HarnessSplit:
    user_t = user_c = harness_t = harness_c = 0
    user_estimated = harness_estimated = False
    for r in records:
        if r.harness:
            harness_t += r.prompt_tokens + r.completion_tokens
            harness_c += r.cost_usd
            harness_estimated = harness_estimated or r.cost_estimated
        else:
            user_t += r.prompt_tokens + r.completion_tokens
            user_c += r.cost_usd
            user_estimated = user_estimated or r.cost_estimated
    return HarnessSplit(
        user_tokens=user_t,
        user_cost=user_c,
        user_cost_estimated=user_estimated,
        harness_tokens=harness_t,
        harness_cost=harness_c,
        harness_cost_estimated=harness_estimated,
    )


def summarize(records: list[UsageRecord], *, now: float | None = None) -> UsageSummary:
    """Build the full status summary from a record list.

    ``records`` is normally a recorder's full history; the summary derives both
    the per-provider/model breakdown (all records) and the rolling time windows.
    Pass ``now`` for deterministic tests.
    """
    if now is None:
        now = time.time()
    providers = _breakdown(records)
    windows = [
        _window(records, "Last hour", _HOUR, now),
        _window(records, "Last 24h", _DAY, now),
        _window(records, "Last 7 days", 7 * _DAY, now),
        _window(records, "Last 30 days", 30 * _DAY, now),
    ]
    return UsageSummary(
        providers=providers,
        windows=windows,
        daily=_daily(records, days=14, now=now),
        harness=_harness_split(records),
        grand_total_tokens=sum(p.total_tokens for p in providers),
        grand_total_cost=sum(p.cost_usd for p in providers),
        cost_estimated=any(p.cost_estimated for p in providers),
    )


def session_records(records: list[UsageRecord], session_id: str) -> list[UsageRecord]:
    return [r for r in records if r.session_id == session_id]
