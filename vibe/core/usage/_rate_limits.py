from __future__ import annotations

from collections.abc import Mapping
import re
import threading

from pydantic import BaseModel, ConfigDict

# Standard OpenAI-compatible rate-limit headers. Providers (OpenAI, ZAI/Zhipu,
# Kimi/Moonshot, DeepSeek, Together, Fireworks, …) return these on chat
# completions responses. Lookup is case-insensitive: httpx preserves the wire
# case, which varies by provider.
_TOKEN_LIMIT = "x-ratelimit-limit-tokens"
_TOKEN_REMAINING = "x-ratelimit-remaining-tokens"
_TOKEN_RESET = "x-ratelimit-reset-tokens"
_REQ_LIMIT = "x-ratelimit-limit-requests"
_REQ_REMAINING = "x-ratelimit-remaining-requests"
_REQ_RESET = "x-ratelimit-reset-requests"

_DURATION_RE = re.compile(
    r"(?:(\d+)\s*d)?"  # days
    r"(?:(\d+)\s*h)?"  # hours
    r"(?:(\d+)\s*ms)?"  # milliseconds (before 'm' for minutes)
    r"(?:(\d+)\s*m)?"  # minutes
    r"(?:(\d+)\s*s)?",  # seconds
    re.IGNORECASE,
)


class RateLimitSnapshot(BaseModel):
    """A provider's rolling rate-limit state at one point in time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    captured_at: float
    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    limit_requests: int | None = None
    remaining_requests: int | None = None
    reset_tokens_in_s: float | None = None
    reset_requests_in_s: float | None = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in (
                "limit_tokens",
                "remaining_tokens",
                "limit_requests",
                "remaining_requests",
                "reset_tokens_in_s",
                "reset_requests_in_s",
            )
        )


def parse_duration_seconds(value: str) -> float | None:
    """Parse OpenAI-style reset durations: '6s', '1m', '2h', '1h30m', '850ms'.

    Returns None for unparseable input. 'ms' is matched before 'm' (minutes)
    so '850ms' doesn't read as 850 minutes.
    """
    s = value.strip().lower()
    if not s:
        return None
    match = _DURATION_RE.fullmatch(s)
    if not match:
        return None
    days, hours, ms, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    total = days * 86400.0 + hours * 3600.0 + ms / 1000.0 + minutes * 60.0 + seconds
    if total == 0 and s != "0" and not s.startswith("0"):
        return None
    return total


def _ci_get(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for k, v in headers.items():
        if k.lower() == lowered:
            return v
    return None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def from_headers(
    provider: str, headers: Mapping[str, str], captured_at: float
) -> RateLimitSnapshot | None:
    """Build a snapshot from response headers, or None if none are present.

    Returns None (rather than an empty snapshot) when no rate-limit header is
    found, so the store keeps the last good snapshot instead of overwriting it
    with blanks from a provider that doesn't emit them.
    """
    keys = {k.lower() for k in headers}
    has_any = any(
        k in keys
        for k in (
            _TOKEN_LIMIT,
            _TOKEN_REMAINING,
            _TOKEN_RESET,
            _REQ_LIMIT,
            _REQ_REMAINING,
            _REQ_RESET,
        )
    )
    if not has_any:
        return None
    reset_tokens = _ci_get(headers, _TOKEN_RESET)
    reset_reqs = _ci_get(headers, _REQ_RESET)
    snap = RateLimitSnapshot(
        provider=provider,
        captured_at=captured_at,
        limit_tokens=_to_int(_ci_get(headers, _TOKEN_LIMIT)),
        remaining_tokens=_to_int(_ci_get(headers, _TOKEN_REMAINING)),
        limit_requests=_to_int(_ci_get(headers, _REQ_LIMIT)),
        remaining_requests=_to_int(_ci_get(headers, _REQ_REMAINING)),
        reset_tokens_in_s=parse_duration_seconds(reset_tokens)
        if reset_tokens
        else None,
        reset_requests_in_s=parse_duration_seconds(reset_reqs) if reset_reqs else None,
    )
    return None if snap.is_empty() else snap


class RateLimitStore:
    """Last-write-wins per-provider rate-limit snapshots.

    The agent loop updates this from response headers after each LLM call;
    `/status` reads it. Values are rolling (a provider reports what's left in
    the *current* window), so newest-wins is the right merge — there's no
    accumulation to do.
    """

    def __init__(self) -> None:
        self._by_provider: dict[str, RateLimitSnapshot] = {}
        self._lock = threading.Lock()

    def update(self, snapshot: RateLimitSnapshot) -> None:
        if snapshot.is_empty():
            return
        with self._lock:
            self._by_provider[snapshot.provider] = snapshot

    def latest(self, provider: str) -> RateLimitSnapshot | None:
        with self._lock:
            return self._by_provider.get(provider)

    def all(self) -> dict[str, RateLimitSnapshot]:
        with self._lock:
            return dict(self._by_provider)
