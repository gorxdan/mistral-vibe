from __future__ import annotations

import httpx

from vibe.core.utils.retry import _MAX_RETRY_AFTER_S, _retry_after_seconds, _retry_delay


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
    # backoff at attempt 0 is ~0.5s; Retry-After 30 should win
    assert _retry_delay(0, 0.5, 2.0, err) == 30.0


def test_retry_delay_capped() -> None:
    err = _http_error({"retry-after": "9999"})
    assert _retry_delay(0, 0.5, 2.0, err) == _MAX_RETRY_AFTER_S


def test_retry_delay_falls_back_to_backoff_without_header() -> None:
    err = _http_error({})
    assert _retry_delay(2, 0.5, 2.0, err) == (0.5 * 4) + 0.1
