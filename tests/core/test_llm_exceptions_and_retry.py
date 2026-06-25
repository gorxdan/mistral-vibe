from __future__ import annotations

from http import HTTPStatus

import httpx
import pytest

from vibe.core.llm.exceptions import (
    BackendError,
    ErrorDetail,
    ErrorResponse,
    PayloadSummary,
)
from vibe.core.utils.retry import (
    _is_retryable_http_error,
    _retry_after_seconds,
    _retry_delay,
    async_retry,
)


def _summary() -> PayloadSummary:
    return PayloadSummary(
        model="m",
        message_count=1,
        approx_chars=10,
        temperature=0.2,
        has_tools=False,
        tool_choice=None,
    )


def _backend_error(
    *,
    status: int | None = None,
    reason: str | None = None,
    headers: dict[str, str] | None = None,
    body: str | None = None,
) -> BackendError:
    return BackendError(
        provider="mistral",
        endpoint="https://api.mistral.ai/v1",
        status=status,
        reason=reason,
        headers=headers,
        body_text=body,
        parsed_error=None,
        model="m",
        payload_summary=_summary(),
    )


# --------------------------------------------------------------------------- #
# BackendError formatting                                                     #
# --------------------------------------------------------------------------- #


def test_backend_error_unauthorized_message() -> None:
    assert "Invalid API key" in str(_backend_error(status=HTTPStatus.UNAUTHORIZED))


def test_backend_error_rate_limit_message() -> None:
    assert "Rate limit" in str(_backend_error(status=HTTPStatus.TOO_MANY_REQUESTS))


def test_backend_error_generic_includes_status_and_request_id() -> None:
    err = _backend_error(
        status=500, reason="Internal", headers={"x-request-id": "rid-1"}, body="boom"
    )
    s = str(err)
    assert "500" in s
    assert "rid-1" in s
    assert "boom" in s
    assert "mistral" in s


def test_backend_error_no_status_shows_na() -> None:
    s = str(_backend_error(status=None))
    assert "N/A" in s


def test_backend_error_invalid_status_code_handled() -> None:
    # An integer not in HTTPStatus falls back to str(int)
    s = str(_backend_error(status=999))
    assert "999" in s


def test_excerpt_truncates_long_body() -> None:
    long_body = "x" * 500
    assert BackendError._excerpt(long_body).endswith("…")
    assert BackendError._excerpt("short") == "short"


def test_backend_error_structured_output_rejected_false_for_normal_body() -> None:
    assert (
        _backend_error(status=400, body="plain error").is_structured_output_rejected
        is False
    )


# --------------------------------------------------------------------------- #
# ErrorResponse.primary_message                                               #
# --------------------------------------------------------------------------- #


def test_error_response_message_from_error_dict_message() -> None:
    resp = ErrorResponse.model_validate({"error": {"message": "from-error"}})
    assert resp.primary_message == "from-error"


def test_error_response_message_from_error_dict_type() -> None:
    resp = ErrorResponse.model_validate({"error": {"type": "bad_type"}})
    assert resp.primary_message is None


def test_error_response_message_from_error_detail_model() -> None:
    resp = ErrorResponse(error=ErrorDetail(message="detail-msg"))
    assert resp.primary_message == "detail-msg"


def test_error_response_message_from_top_level_message() -> None:
    resp = ErrorResponse(message="top-level")
    assert resp.primary_message == "top-level"


def test_error_response_message_from_detail_field() -> None:
    resp = ErrorResponse(detail="detail-field")
    assert resp.primary_message == "detail-field"


def test_error_response_message_none_when_nothing_present() -> None:
    assert ErrorResponse().primary_message is None


# --------------------------------------------------------------------------- #
# retry helpers                                                               #
# --------------------------------------------------------------------------- #


def _http_status_error(
    status: int, headers: dict[str, str] | None = None
) -> httpx.HTTPStatusError:
    resp = httpx.Response(status, headers=headers or {})
    return httpx.HTTPStatusError(
        "err", request=httpx.Request("GET", "/x"), response=resp
    )


def test_is_retryable_http_error_status_codes() -> None:
    for code in (408, 429, 500, 502, 503, 504):
        assert _is_retryable_http_error(_http_status_error(code)) is True
    assert _is_retryable_http_error(_http_status_error(400)) is False


def test_is_retryable_http_error_request_errors() -> None:
    assert _is_retryable_http_error(httpx.ConnectError("down")) is True
    assert _is_retryable_http_error(httpx.TimeoutException("slow")) is True
    assert _is_retryable_http_error(RuntimeError("x")) is False


def test_retry_after_seconds_digit_header() -> None:
    exc = _http_status_error(429, headers={"retry-after": "5"})
    assert _retry_after_seconds(exc) == 5.0


def test_retry_after_seconds_missing_returns_none() -> None:
    exc = _http_status_error(429)
    assert _retry_after_seconds(exc) is None


def test_retry_after_seconds_non_http_returns_none() -> None:
    assert _retry_after_seconds(RuntimeError("x")) is None


def test_retry_after_seconds_invalid_returns_none() -> None:
    exc = _http_status_error(429, headers={"retry-after": "not-a-date"})
    assert _retry_after_seconds(exc) is None


def test_retry_delay_uses_retry_after_when_larger() -> None:
    exc = _http_status_error(429, headers={"retry-after": "10"})
    delay = _retry_delay(0, delay_seconds=0.1, backoff_factor=2.0, exc=exc)
    assert delay == 10.0


def test_retry_delay_uses_base_when_no_retry_after() -> None:
    delay = _retry_delay(
        1, delay_seconds=1.0, backoff_factor=2.0, exc=RuntimeError("x")
    )
    assert delay > 0


@pytest.mark.asyncio
async def test_async_retry_succeeds_after_retry() -> None:
    calls = {"n": 0}

    @async_retry(tries=3, delay_seconds=0.01)
    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("down")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_async_retry_raises_after_exhausting() -> None:
    @async_retry(tries=2, delay_seconds=0.01)
    async def always_fail() -> str:
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        await always_fail()


@pytest.mark.asyncio
async def test_async_retry_non_retryable_raises_immediately() -> None:
    calls = {"n": 0}

    @async_retry(tries=3, delay_seconds=0.01)
    async def bad_request() -> str:
        calls["n"] += 1
        raise _http_status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await bad_request()
    assert calls["n"] == 1
