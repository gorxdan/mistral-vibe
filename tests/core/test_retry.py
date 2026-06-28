from __future__ import annotations

from collections.abc import AsyncGenerator
import logging

import httpx
import pytest

from vibe.core.utils.retry import (
    _http_response_detail,
    _is_retryable_http_error,
    async_generator_retry,
    async_retry,
)


def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    response = httpx.Response(
        status_code=status_code, request=httpx.Request("GET", "https://example.com")
    )
    return httpx.HTTPStatusError(
        message=f"Error {status_code}", request=response.request, response=response
    )


def _make_request(url: str = "https://example.com") -> httpx.Request:
    return httpx.Request("POST", url)


class TestIsRetryableHttpError:
    @pytest.mark.parametrize("code", [408, 409, 425, 500, 502, 503, 504, 529])
    def test_retryable_codes(self, code: int) -> None:
        assert _is_retryable_http_error(_make_http_status_error(code)) is True

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_non_retryable_codes(self, code: int) -> None:
        assert _is_retryable_http_error(_make_http_status_error(code)) is False

    def test_bare_429_is_not_retryable(self) -> None:
        # A rate limit without Retry-After is not blind-retried: re-firing at an
        # already-limited endpoint amplifies load and delays failover.
        assert _is_retryable_http_error(_make_http_status_error(429)) is False

    def test_429_with_retry_after_is_retryable(self) -> None:
        response = httpx.Response(
            status_code=429,
            headers={"retry-after": "1"},
            request=httpx.Request("GET", "https://example.com"),
        )
        exc = httpx.HTTPStatusError(
            message="Error 429", request=response.request, response=response
        )
        assert _is_retryable_http_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectTimeout("connect timed out", request=_make_request()),
            httpx.ReadTimeout("read timed out", request=_make_request()),
            httpx.WriteTimeout("write timed out", request=_make_request()),
            httpx.PoolTimeout("pool timed out", request=_make_request()),
            httpx.ConnectError("connection refused", request=_make_request()),
            httpx.ReadError("read failed", request=_make_request()),
            httpx.WriteError("write failed", request=_make_request()),
            httpx.RemoteProtocolError("server disconnected", request=_make_request()),
        ],
    )
    def test_retryable_network_errors(self, exc: Exception) -> None:
        assert _is_retryable_http_error(exc) is True

    def test_non_retryable_request_error(self) -> None:
        assert _is_retryable_http_error(httpx.InvalidURL("bad url")) is False

    def test_non_http_error_returns_false(self) -> None:
        assert _is_retryable_http_error(ValueError("not http")) is False

    def test_generic_exception_returns_false(self) -> None:
        assert _is_retryable_http_error(RuntimeError("boom")) is False


class TestHttpResponseDetail:
    def _status_error(
        self,
        status_code: int = 429,
        *,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> httpx.HTTPStatusError:
        response = httpx.Response(
            status_code=status_code,
            text=text,
            headers=headers or {},
            request=httpx.Request("POST", "https://example.com"),
        )
        return httpx.HTTPStatusError(
            message=f"Error {status_code}", request=response.request, response=response
        )

    def test_includes_status_retry_after_and_body(self) -> None:
        exc = self._status_error(
            429, text='{"error":"Rate limit exceeded"}', headers={"retry-after": "12"}
        )
        detail = _http_response_detail(exc)
        assert "status=429" in detail
        assert "retry_after=12" in detail
        assert "Rate limit exceeded" in detail

    def test_quota_body_is_distinguishable_from_transient(self) -> None:
        # A provider that surfaces credit exhaustion as 429 (see ZAI code 1113):
        # the body makes it provable as quota, not a transient rate limit.
        exc = self._status_error(
            429, text='{"error":{"code":1113,"message":"Insufficient balance"}}'
        )
        detail = _http_response_detail(exc)
        assert "Insufficient balance" in detail
        # No Retry-After => not a server-advertised transient window.
        assert "retry_after=" not in detail

    def test_truncates_long_body(self) -> None:
        exc = self._status_error(429, text="x" * 1000)
        detail = _http_response_detail(exc)
        assert detail.endswith("…")
        assert len(detail) < 1000

    def test_non_http_returns_na(self) -> None:
        assert (
            _http_response_detail(httpx.ConnectError("x", request=_make_request()))
            == "n/a"
        )


class TestRetryLogsResponseBody:
    @pytest.mark.asyncio
    async def test_async_retry_logs_response_body_on_429(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        attempts = 0

        @async_retry(tries=3, delay_seconds=0.0, backoff_factor=1.0)
        async def call() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                response = httpx.Response(
                    429,
                    text='{"error":"quota depleted"}',
                    headers={"retry-after": "0"},
                    request=httpx.Request("POST", "https://example.com"),
                )
                raise httpx.HTTPStatusError(
                    message="429", request=response.request, response=response
                )
            return "ok"

        with caplog.at_level(logging.WARNING, logger="vibe"):
            result = await call()

        assert result == "ok"
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "status=429" in joined
        assert "quota depleted" in joined


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_retries_network_error_then_succeeds(self) -> None:
        attempts = 0

        @async_retry(tries=3, delay_seconds=0.0, backoff_factor=1.0)
        async def call() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise httpx.ConnectTimeout("timeout", request=_make_request())
            return "ok"

        result = await call()
        assert result == "ok"
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable(self) -> None:
        attempts = 0

        @async_retry(tries=3, delay_seconds=0.0, backoff_factor=1.0)
        async def call() -> str:
            nonlocal attempts
            attempts += 1
            raise ValueError("nope")

        with pytest.raises(ValueError):
            await call()
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries(self) -> None:
        attempts = 0

        @async_retry(tries=3, delay_seconds=0.0, backoff_factor=1.0)
        async def call() -> str:
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout("timeout", request=_make_request())

        with pytest.raises(httpx.ReadTimeout):
            await call()
        assert attempts == 3


class TestAsyncGeneratorRetry:
    @pytest.mark.asyncio
    async def test_retries_before_first_yield(self) -> None:
        attempts = 0

        @async_generator_retry(tries=3, delay_seconds=0.0, backoff_factor=1.0)
        async def gen() -> AsyncGenerator[int]:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise httpx.ConnectError("connect failed", request=_make_request())
            yield 1
            yield 2

        items = [item async for item in gen()]
        assert items == [1, 2]
        assert attempts == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_after_first_yield(self) -> None:
        attempts = 0

        @async_generator_retry(tries=3, delay_seconds=0.0, backoff_factor=1.0)
        async def gen() -> AsyncGenerator[int]:
            nonlocal attempts
            attempts += 1
            yield 1
            raise httpx.ReadError("midstream", request=_make_request())

        items: list[int] = []
        with pytest.raises(httpx.ReadError):
            async for item in gen():
                items.append(item)

        assert items == [1]
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable_before_yield(self) -> None:
        attempts = 0

        @async_generator_retry(tries=3, delay_seconds=0.0, backoff_factor=1.0)
        async def gen() -> AsyncGenerator[int]:
            nonlocal attempts
            attempts += 1
            raise ValueError("nope")
            yield 0  # pragma: no cover

        with pytest.raises(ValueError):
            async for _ in gen():
                pass
        assert attempts == 1
