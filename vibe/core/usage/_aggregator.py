from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import time

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
    cost_usd: float
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
    cost_usd: float = 0.0
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
    cost_usd: float
    calls: int
    sessions: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class UsageSummary:
    providers: list[ProviderBreakdown]
    windows: list[WindowRollup]
    grand_total_tokens: int
    grand_total_cost: float


def _breakdown(records: list[UsageRecord]) -> list[ProviderBreakdown]:
    by_pm: dict[tuple[str, str], ModelBreakdown] = {}
    p_prompt: dict[str, int] = defaultdict(int)
    p_comp: dict[str, int] = defaultdict(int)
    p_cache: dict[str, int] = defaultdict(int)
    p_cost: dict[str, float] = defaultdict(float)
    p_calls: dict[str, int] = defaultdict(int)

    for r in records:
        key = (r.provider, r.model)
        prev = by_pm.get(key)
        if prev is None:
            by_pm[key] = ModelBreakdown(
                provider=r.provider,
                model=r.model,
                prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens,
                cached_tokens=r.cached_tokens,
                cost_usd=r.cost_usd,
                calls=1,
            )
        else:
            by_pm[key] = ModelBreakdown(
                provider=r.provider,
                model=r.model,
                prompt_tokens=prev.prompt_tokens + r.prompt_tokens,
                completion_tokens=prev.completion_tokens + r.completion_tokens,
                cached_tokens=prev.cached_tokens + r.cached_tokens,
                cost_usd=prev.cost_usd + r.cost_usd,
                calls=prev.calls + 1,
            )
        p_prompt[r.provider] += r.prompt_tokens
        p_comp[r.provider] += r.completion_tokens
        p_cache[r.provider] += r.cached_tokens
        p_cost[r.provider] += r.cost_usd
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
            cost_usd=p_cost[p],
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
        cost_usd=sum(r.cost_usd for r in recs),
        calls=len(recs),
        sessions=len(sessions),
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
    ]
    return UsageSummary(
        providers=providers,
        windows=windows,
        grand_total_tokens=sum(p.total_tokens for p in providers),
        grand_total_cost=sum(p.cost_usd for p in providers),
    )


def session_records(records: list[UsageRecord], session_id: str) -> list[UsageRecord]:
    return [r for r in records if r.session_id == session_id]
