from __future__ import annotations

import httpx
import pytest

from vibe.core.utils.retry import (
    _BACKOFF_JITTER,
    _MAX_RETRY_AFTER_S,
    _RETRY_AFTER_JITTER,
    _retry_after_seconds,
    _retry_delay,
)


def _http_error(headers: dict[str, str]) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://x/y")
    resp = httpx.Response(429, headers=headers, request=req)
    return httpx.HTTPStatusError("rate limited", request=req, response=resp)


def test_no_header_returns_none() -> None:
    assert _retry_after_seconds(_http_error({})) is None


def test_delta_seconds_parsed() -> None:
    assert _retry_after_seconds(_http_error({"retry-after": "12"})) == 12.0


def test_non_http_error_returns_none() -> None:
    assert _retry_after_seconds(ValueError("nope")) is None


def test_retry_delay_uses_retry_after_when_larger() -> None:
    err = _http_error({"retry-after": "30"})
    # backoff at attempt 0 is ~0.5s; Retry-After 30 wins and is jittered upward
    # (positive-only) into [30, 36], never below the server window.
    delay = _retry_delay(0, 0.5, 2.0, err)
    assert 30.0 <= delay <= 30.0 * (1.0 + _RETRY_AFTER_JITTER)


def test_retry_delay_capped() -> None:
    err = _http_error({"retry-after": "9999"})
    assert _retry_delay(0, 0.5, 2.0, err) == _MAX_RETRY_AFTER_S


def test_retry_delay_falls_back_to_backoff_without_header() -> None:
    err = _http_error({})
    base = (0.5 * 4) + 0.1
    # ±30% symmetric jitter around the computed exponential backoff.
    delay = _retry_delay(2, 0.5, 2.0, err)
    assert base * (1.0 - _BACKOFF_JITTER) <= delay <= base * (1.0 + _BACKOFF_JITTER)


def test_retry_delay_backoff_midpoint_with_mocked_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # random()=0.5 => jitter multiplier is exactly 1.0 (no shift).
    monkeypatch.setattr("vibe.core.utils.retry.random.random", lambda: 0.5)
    err = _http_error({})
    base = (0.5 * 4) + 0.1
    assert _retry_delay(2, 0.5, 2.0, err) == pytest.approx(base)


def test_retry_delay_backoff_lower_bound_with_mocked_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # random()=0.0 => -30% (lower bound of symmetric jitter).
    monkeypatch.setattr("vibe.core.utils.retry.random.random", lambda: 0.0)
    err = _http_error({})
    base = (0.5 * 4) + 0.1
    assert _retry_delay(2, 0.5, 2.0, err) == pytest.approx(base * 0.7)


def test_retry_delay_backoff_upper_bound_with_mocked_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # random()=1.0 => +30% (upper bound of symmetric jitter).
    monkeypatch.setattr("vibe.core.utils.retry.random.random", lambda: 1.0)
    err = _http_error({})
    base = (0.5 * 4) + 0.1
    assert _retry_delay(2, 0.5, 2.0, err) == pytest.approx(base * 1.3)


def test_retry_delay_retry_after_jittered_upward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # random()=1.0 => +20% on the Retry-After value.
    monkeypatch.setattr("vibe.core.utils.retry.random.random", lambda: 1.0)
    err = _http_error({"retry-after": "30"})
    assert _retry_delay(0, 0.5, 2.0, err) == pytest.approx(36.0)


def test_retry_delay_retry_after_never_below_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # random()=0.0 => no upward shift; the server window is a floor.
    monkeypatch.setattr("vibe.core.utils.retry.random.random", lambda: 0.0)
    err = _http_error({"retry-after": "30"})
    assert _retry_delay(0, 0.5, 2.0, err) == 30.0


def test_retry_delay_never_negative() -> None:
    # Even at attempt 0 with tiny base, symmetric jitter can't go negative.
    err = _http_error({})
    for _ in range(50):
        assert _retry_delay(0, 0.0, 2.0, err) >= 0.0
