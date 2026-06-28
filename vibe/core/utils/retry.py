from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
import functools
from http import HTTPStatus
import logging
import random

import httpx

logger = logging.getLogger("vibe")

_RETRYABLE_REQUEST_ERRORS: tuple[type[httpx.RequestError], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


def _is_retryable_http_error(e: Exception) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == HTTPStatus.TOO_MANY_REQUESTS:
            # Blind-retrying a rate limit amplifies load and delays failover;
            # retry only on an explicit Retry-After, else raise to fail over.
            return _retry_after_seconds(e) is not None
        return code in {408, 409, 425, 500, 502, 503, 504, 529}
    if isinstance(e, _RETRYABLE_REQUEST_ERRORS):
        return True
    return False


# Cap on how long we'll honor a server's Retry-After (avoid pathological waits).
_MAX_RETRY_AFTER_S = 60.0

# ±30% symmetric jitter on the computed exponential backoff. Without it,
# concurrent retries (workflow fan-out, parallel secondary calls) that hit a
# transient error at the same instant re-fire on the identical computed delay
# and re-trip the limit together — the "Retries exhausted" lockstep storm.
_BACKOFF_JITTER = 0.3
# 0..+20% jitter on a server Retry-After: spreads clients that received the same
# header, always upward so no client fires before the advertised window.
_RETRY_AFTER_JITTER = 0.2

# Body bytes retained in the retry-log diagnostic so a 429's cause is provable
# from logs alone (transient rate limit vs. quota/credit exhaustion).
_RESPONSE_BODY_EXCERPT = 400


def _retry_after_seconds(e: Exception) -> float | None:
    """Parse a Retry-After header (delta-seconds or HTTP-date), if present."""
    if not isinstance(e, httpx.HTTPStatusError):
        return None
    raw = e.response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return float(raw)
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    import datetime as _dt

    now = _dt.datetime.now(when.tzinfo)
    return max(0.0, (when - now).total_seconds())


def _retry_delay(
    attempt: int, delay_seconds: float, backoff_factor: float, exc: Exception
) -> float:
    """Exponential backoff with jitter, overridden upward by a Retry-After header.

    Jitter de-synchronizes concurrent retries (workflow fan-out, parallel
    secondary calls) that would otherwise re-fire in lockstep on the identical
    computed delay. A Retry-After header is jittered upward only so concurrent
    clients never fire before the server's advertised window.
    """
    base = (delay_seconds * (backoff_factor**attempt)) + (0.05 * attempt)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        jittered = retry_after * (1.0 + _RETRY_AFTER_JITTER * random.random())
        return min(max(base, jittered), _MAX_RETRY_AFTER_S)
    return max(0.0, base * (1.0 + _BACKOFF_JITTER * (random.random() * 2.0 - 1.0)))


def _http_response_detail(exc: Exception) -> str:
    """status + Retry-After + body excerpt for an HTTP error, for retry logs.

    Lets the log prove whether a 429 is a transient rate limit (Retry-After
    header, or a bare body) versus quota/credit exhaustion (a 402/403 body some
    providers mis-surface as 429). Backends buffer the body before
    raise_for_status(), so response.text is readable here.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return "n/a"
    response = exc.response
    retry_after = response.headers.get("retry-after")
    try:
        body = (response.text or "").strip().replace("\n", " ")
    except Exception:
        body = ""
    if len(body) > _RESPONSE_BODY_EXCERPT:
        body = body[:_RESPONSE_BODY_EXCERPT] + "…"
    parts = [f"status={response.status_code}"]
    if retry_after:
        parts.append(f"retry_after={retry_after}")
    if body:
        parts.append(f"body={body}")
    return " ".join(parts)


def async_retry[T, **P](
    tries: int = 3,
    delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    is_retryable: Callable[[Exception], bool] = _is_retryable_http_error,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Args:
        tries: Number of retry attempts
        delay_seconds: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        is_retryable: Function to determine if an exception should trigger a retry
                     (defaults to checking for retryable HTTP errors from both urllib and httpx)

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc = None
            for attempt in range(tries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < tries - 1 and is_retryable(e):
                        current_delay = _retry_delay(
                            attempt, delay_seconds, backoff_factor, e
                        )
                        logger.warning(
                            "Retrying %s after error attempt=%d/%d delay=%.2fs "
                            "error=%r response=%s",
                            func.__qualname__,
                            attempt + 1,
                            tries,
                            current_delay,
                            e,
                            _http_response_detail(e),
                        )
                        await asyncio.sleep(current_delay)
                        continue
                    raise e
            raise RuntimeError(
                f"Retries exhausted. Last error: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator


def async_generator_retry[T, **P](
    tries: int = 3,
    delay_seconds: float = 0.5,
    backoff_factor: float = 2.0,
    is_retryable: Callable[[Exception], bool] = _is_retryable_http_error,
) -> Callable[[Callable[P, AsyncGenerator[T]]], Callable[P, AsyncGenerator[T]]]:
    """Retry decorator for async generators.

    Args:
        tries: Number of retry attempts
        delay_seconds: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay on each retry
        is_retryable: Function to determine if an exception should trigger a retry
                     (defaults to checking for retryable HTTP errors from both urllib and httpx)

    Returns:
        Decorated async generator function with retry logic
    """

    def decorator(
        func: Callable[P, AsyncGenerator[T]],
    ) -> Callable[P, AsyncGenerator[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[T]:
            last_exc = None
            for attempt in range(tries):
                generator = func(*args, **kwargs)
                try:
                    first_item = await anext(generator)
                except StopAsyncIteration:
                    return
                except Exception as e:
                    last_exc = e
                    await generator.aclose()
                    if attempt < tries - 1 and is_retryable(e):
                        current_delay = _retry_delay(
                            attempt, delay_seconds, backoff_factor, e
                        )
                        logger.warning(
                            "Retrying %s after error attempt=%d/%d delay=%.2fs "
                            "error=%r response=%s",
                            func.__qualname__,
                            attempt + 1,
                            tries,
                            current_delay,
                            e,
                            _http_response_detail(e),
                        )
                        await asyncio.sleep(current_delay)
                        continue
                    raise
                yield first_item
                async for item in generator:
                    yield item
                return
            raise RuntimeError(
                f"Retries exhausted. Last error: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator
