from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
import types
from typing import Protocol, Self

import httpx
from mistralai.client.errors import SDKError

from vibe.core.llm.provider_retry import (
    ProviderRetryController,
    authorize_provider_retry,
    bind_provider_retry_controller,
)
from vibe.core.utils.retry import (
    is_retryable_http_error,
    provider_retry_cause,
    provider_retry_delay,
)

__all__ = ["complete_with_retry", "stream_with_retry"]


class _AsyncStream[T](Protocol):
    response: httpx.Response

    def __aiter__(self) -> AsyncIterator[T]: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...


def _retryable_error(error: SDKError | httpx.RequestError) -> Exception | None:
    normalized: Exception
    if isinstance(error, SDKError):
        response = error.raw_response
        normalized = httpx.HTTPStatusError(
            str(error), request=response.request, response=response
        )
    else:
        normalized = error
    return normalized if is_retryable_http_error(normalized) else None


async def _authorize_retry(
    controller: ProviderRetryController,
    error: SDKError | httpx.RequestError,
    attempt: int,
) -> bool:
    retryable = _retryable_error(error)
    if retryable is None:
        return False
    delay = provider_retry_delay(attempt, 0.5, 1.5, retryable)
    with bind_provider_retry_controller(controller):
        return await authorize_provider_retry(
            provider_retry_cause(retryable), delay_s=delay
        )


async def complete_with_retry[T](
    dispatch: Callable[[], Awaitable[T]], *, max_elapsed_time: float
) -> T:
    controller = ProviderRetryController(max_elapsed_time=max_elapsed_time)
    attempt = 0
    while True:
        try:
            return await dispatch()
        except (SDKError, httpx.RequestError) as error:
            if not await _authorize_retry(controller, error, attempt):
                raise
            attempt += 1


async def stream_with_retry[T](
    dispatch: Callable[[], Awaitable[_AsyncStream[T]]], *, max_elapsed_time: float
) -> AsyncGenerator[tuple[_AsyncStream[T], T], None]:
    controller = ProviderRetryController(max_elapsed_time=max_elapsed_time)
    attempt = 0
    yielded = False
    while True:
        try:
            stream = await dispatch()
            async with stream:
                async for item in stream:
                    yielded = True
                    yield stream, item
            return
        except (SDKError, httpx.RequestError) as error:
            if yielded or not await _authorize_retry(controller, error, attempt):
                raise
            attempt += 1
