"""Agent-loop exception hierarchy and backend-error classification.

Pure functions and exception types only — no ``self``/host coupling. Extracted
from the loop module so the ``_is_*`` classifiers, the rate-limit / context /
response-too-long / content-filter mapping, and the ``AgentLoopError`` tree live
together and can be unit-tested in isolation.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import NoReturn

from vibe.core.llm.exceptions import BackendError
from vibe.core.types import (
    ContentFilterError,
    ContextTooLongError,
    LLMChunk,
    RateLimitError,
    RefusalError,
    ResponseTooLongError,
    ServerError,
    TransportError,
)


class AgentLoopError(Exception): ...


class AgentLoopStateError(AgentLoopError): ...


class AgentLoopLLMResponseError(AgentLoopError): ...


# Bounded retry count for a degenerate streamed response (no content, tool calls,
# or reasoning): one initial attempt plus a single re-request. A degenerate
# response yields inert (empty) chunks upstream, so a retry with a fresh
# accumulator is clean; two failures means something is structurally wrong.
_STREAM_DEGENERATE_RETRIES = 2


class InvalidStreamError(AgentLoopLLMResponseError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid streamed response: {reason}")


class CompactionFailedError(AgentLoopError):
    def __init__(self, reason: str) -> None:
        self.reason = reason  # "tool_call" | "empty_summary"
        super().__init__(f"Compaction did not produce a summary (reason={reason}).")


class ImagesNotSupportedError(AgentLoopError): ...


class TeleportError(AgentLoopError): ...


def _refusal_error(provider: str, model: str, chunk: LLMChunk) -> RefusalError:
    stop = chunk.stop
    return RefusalError(
        provider,
        model,
        category=stop.category if stop else None,
        explanation=stop.explanation if stop else None,
    )


def _degenerate_response_reason(chunk: LLMChunk) -> str | None:
    msg = chunk.message
    has_content = bool((msg.content or "").strip())
    has_tool_calls = bool(msg.tool_calls)
    has_reasoning = bool((msg.reasoning_content or "").strip())
    if has_content or has_tool_calls or has_reasoning:
        return None
    if chunk.usage is not None:
        return None
    return "empty response (no content, tool calls, or reasoning) and no usage"


def _raise_for_backend_error(
    e: Exception, provider_name: str, model_name: str
) -> NoReturn:
    if isinstance(e, RefusalError | ResponseTooLongError):
        raise
    if _should_raise_rate_limit_error(e):
        raise RateLimitError(provider_name, model_name) from e
    if _is_context_too_long_error(e):
        raise ContextTooLongError(provider_name, model_name) from e
    if _is_response_too_long_error(e):
        raise ResponseTooLongError(provider_name, model_name) from e
    if _is_content_filter_error(e):
        raise ContentFilterError(provider_name, model_name) from e
    if _is_non_retryable_error(e):
        raise
    if _is_transport_error(e):
        raise TransportError(provider_name, model_name) from e
    if _is_server_error(e):
        raise ServerError(provider_name, model_name) from e
    raise RuntimeError(
        f"API error from {provider_name} (model: {model_name}): {e}"
    ) from e


def _should_raise_rate_limit_error(e: Exception) -> bool:
    return isinstance(e, BackendError) and e.status == HTTPStatus.TOO_MANY_REQUESTS


_MAX_SERVER_STATUS = 599


def _is_server_error(e: Exception) -> bool:
    backend = e if isinstance(e, BackendError) else getattr(e, "__cause__", None)
    return (
        isinstance(backend, BackendError)
        and backend.status is not None
        and HTTPStatus.INTERNAL_SERVER_ERROR <= backend.status <= _MAX_SERVER_STATUS
    )


def _is_transport_error(e: Exception) -> bool:
    # A dropped connection / transport failure reaches the loop as
    # BackendError(status=None): build_request_error (exceptions.py) sets no
    # status because there is no HTTP response to classify. status=None is the
    # unique signal — any HTTP status means a response arrived and is routed by
    # the server / rate-limit / context classifiers above. Mutually exclusive
    # with _is_server_error (which requires a 5xx status).
    backend = e if isinstance(e, BackendError) else getattr(e, "__cause__", None)
    return isinstance(backend, BackendError) and backend.status is None


def _is_context_too_long_error(e: Exception) -> bool:
    if isinstance(e, BackendError):
        return e.is_context_too_long
    if isinstance(e, RuntimeError) and isinstance(e.__cause__, BackendError):
        return e.__cause__.is_context_too_long
    return False


def _is_response_too_long_error(e: Exception) -> bool:
    if isinstance(e, BackendError):
        return e.is_response_too_long
    if isinstance(e, RuntimeError) and isinstance(e.__cause__, BackendError):
        return e.__cause__.is_response_too_long
    return False


def _is_content_filter_error(e: Exception) -> bool:
    if isinstance(e, BackendError):
        return e.is_content_filtered
    if isinstance(e, RuntimeError) and isinstance(e.__cause__, BackendError):
        return e.__cause__.is_content_filtered
    return False


def _is_non_retryable_error(e: BaseException) -> bool:
    # Detect Temporal-style ``non_retryable`` flag without importing temporalio.
    # Walks ``__cause__`` so an ``ActivityError`` whose cause is a non-retryable
    # ``ApplicationError`` is detected too — that's what callers driving the
    # agent loop from a Temporal activity will see when a sub-activity has
    # already failed terminally.
    seen: set[int] = set()
    current: BaseException | None = e
    while current is not None and id(current) not in seen:
        if getattr(current, "non_retryable", False):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False
