from __future__ import annotations

from vibe.core.usage import (
    RateLimitSnapshot,
    RateLimitStore,
    parse_duration_seconds,
    rate_limit_from_headers,
)


class TestParseDuration:
    def test_seconds(self):
        assert parse_duration_seconds("6s") == 6.0

    def test_minutes(self):
        assert parse_duration_seconds("1m") == 60.0

    def test_hours(self):
        assert parse_duration_seconds("2h") == 7200.0

    def test_composite(self):
        assert parse_duration_seconds("1h30m") == 5400.0

    def test_milliseconds_before_minutes(self):
        # '850ms' must not read as 850 minutes — ms matches before m.
        assert parse_duration_seconds("850ms") == 0.85

    def test_days(self):
        assert parse_duration_seconds("1d") == 86400.0

    def test_zero_with_unit(self):
        assert parse_duration_seconds("0s") == 0.0

    def test_bare_number_without_unit_is_none(self):
        # Providers always send a unit; a bare "0" is ambiguous/unparseable.
        assert parse_duration_seconds("0") is None

    def test_garbage(self):
        assert parse_duration_seconds("soon") is None
        assert parse_duration_seconds("") is None


class TestFromHeaders:
    def test_full_openai_headers_case_insensitive(self):
        # Providers vary wire-case; parser must be case-insensitive.
        headers = {
            "X-RateLimit-Limit-Tokens": "900000",
            "X-RateLimit-Remaining-Tokens": "630000",
            "X-RateLimit-Reset-Tokens": "6s",
            "X-RateLimit-Limit-Requests": "100",
            "X-RateLimit-Remaining-Requests": "48",
            "X-RateLimit-Reset-Requests": "1m",
        }
        snap = rate_limit_from_headers("openai", headers, captured_at=1000.0)
        assert snap is not None
        assert snap.limit_tokens == 900000
        assert snap.remaining_tokens == 630000
        assert snap.limit_requests == 100
        assert snap.remaining_requests == 48
        assert snap.reset_tokens_in_s == 6.0
        assert snap.reset_requests_in_s == 60.0

    def test_returns_none_when_no_rate_limit_headers(self):
        assert rate_limit_from_headers("p", {"content-type": "json"}, 1.0) is None

    def test_partial_headers(self):
        snap = rate_limit_from_headers(
            "kimi",
            {
                "x-ratelimit-remaining-requests": "5",
                "x-ratelimit-reset-requests": "30s",
            },
            captured_at=2.0,
        )
        assert snap is not None
        assert snap.remaining_requests == 5
        assert snap.reset_requests_in_s == 30.0
        assert snap.limit_tokens is None

    def test_garbage_int_is_none(self):
        snap = rate_limit_from_headers(
            "p", {"x-ratelimit-limit-tokens": "not-a-number"}, captured_at=1.0
        )
        # Present-but-unparseable → None for that field; snapshot still built
        # only if at least one field parsed. Here none did → None.
        assert snap is None

    def test_garbage_reset_is_none_not_fatal(self):
        snap = rate_limit_from_headers(
            "p",
            {
                "x-ratelimit-remaining-tokens": "100",
                "x-ratelimit-reset-tokens": "whenever",
            },
            captured_at=1.0,
        )
        assert snap is not None
        assert snap.remaining_tokens == 100
        assert snap.reset_tokens_in_s is None


class TestRateLimitStore:
    def test_update_and_latest(self):
        store = RateLimitStore()
        snap = RateLimitSnapshot(
            provider="zai", captured_at=1.0, limit_tokens=100000, remaining_tokens=80000
        )
        store.update(snap)
        latest = store.latest("zai")
        assert latest is not None
        assert latest.remaining_tokens == 80000
        assert store.latest("absent") is None

    def test_last_write_wins(self):
        store = RateLimitStore()
        store.update(
            RateLimitSnapshot(provider="zai", captured_at=1.0, remaining_tokens=80000)
        )
        store.update(
            RateLimitSnapshot(provider="zai", captured_at=2.0, remaining_tokens=40000)
        )
        latest = store.latest("zai")
        assert latest is not None
        assert latest.remaining_tokens == 40000
        assert latest.captured_at == 2.0

    def test_empty_snapshot_not_stored(self):
        store = RateLimitStore()
        store.update(RateLimitSnapshot(provider="p", captured_at=1.0))
        assert store.latest("p") is None
        assert store.all() == {}

    def test_all_returns_copy(self):
        store = RateLimitStore()
        store.update(
            RateLimitSnapshot(provider="zai", captured_at=1.0, remaining_tokens=1)
        )
        all_snap = store.all()
        all_snap["injected"] = RateLimitSnapshot(provider="x", captured_at=1.0)
        # Mutating the returned dict must not affect the store.
        assert store.latest("x") is None
