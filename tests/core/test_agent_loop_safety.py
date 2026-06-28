from __future__ import annotations

from http import HTTPStatus
from typing import Any

import pytest

from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import (
    _is_context_too_long_error,
    _is_non_retryable_error,
    _is_response_too_long_error,
    _should_raise_rate_limit_error,
)
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.llm.exceptions import BackendError, PayloadSummary
from vibe.core.tools.base import ToolPermissionError
from vibe.core.types import (
    ContextTooLongError,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    RateLimitError,
    ResponseTooLongError,
    Role,
    ToolCall,
    ToolResultEvent,
)


def _backend_error(*, status: int | None, body: str | None = None) -> BackendError:
    return BackendError(
        provider="mistral",
        endpoint="https://api.mistral.ai/v1/chat/completions",
        status=status,
        reason=None,
        headers={},
        body_text=body,
        parsed_error=None,
        model="devstral-latest",
        payload_summary=_summary(),
    )


def _summary() -> PayloadSummary:
    return PayloadSummary(
        model="devstral-latest",
        message_count=1,
        approx_chars=10,
        temperature=0.2,
        has_tools=False,
        tool_choice=None,
    )


# --------------------------------------------------------------------------- #
# Error-classification helpers (pure functions)                               #
# --------------------------------------------------------------------------- #


def test_should_raise_rate_limit_error_only_429() -> None:
    assert (
        _should_raise_rate_limit_error(
            _backend_error(status=HTTPStatus.TOO_MANY_REQUESTS)
        )
        is True
    )
    assert (
        _should_raise_rate_limit_error(_backend_error(status=HTTPStatus.BAD_REQUEST))
        is False
    )
    assert _should_raise_rate_limit_error(RuntimeError("x")) is False


def test_is_context_too_long_error_direct_and_via_cause() -> None:
    direct = _backend_error(status=HTTPStatus.BAD_REQUEST, body="context too long")
    assert _is_context_too_long_error(direct) is True

    wrapped = RuntimeError("upstream")
    wrapped.__cause__ = _backend_error(
        status=HTTPStatus.UNPROCESSABLE_ENTITY, body="prompt_too_long"
    )
    assert _is_context_too_long_error(wrapped) is True

    assert (
        _is_context_too_long_error(
            _backend_error(status=HTTPStatus.BAD_REQUEST, body="other")
        )
        is False
    )
    assert _is_context_too_long_error(RuntimeError("plain")) is False


def test_is_response_too_long_error_direct_and_via_cause() -> None:
    direct = _backend_error(
        status=HTTPStatus.UNPROCESSABLE_ENTITY, body="max_tokens_exceeded"
    )
    assert _is_response_too_long_error(direct) is True

    wrapped = RuntimeError("up")
    wrapped.__cause__ = _backend_error(
        status=HTTPStatus.UNPROCESSABLE_ENTITY, body="finish_reason=length"
    )
    assert _is_response_too_long_error(wrapped) is True

    assert (
        _is_response_too_long_error(
            _backend_error(status=HTTPStatus.BAD_REQUEST, body="x")
        )
        is False
    )


def test_is_non_retryable_error_walks_cause_chain_and_detects_flag() -> None:
    class _Inner(Exception):
        non_retryable = True

    assert _is_non_retryable_error(_Inner("x")) is True

    outer = RuntimeError("outer")
    outer.__cause__ = _Inner("inner")
    assert _is_non_retryable_error(outer) is True

    # A self-referential cause chain must not loop forever.
    cyclic = RuntimeError("cyclic")
    cyclic.__cause__ = cyclic
    assert _is_non_retryable_error(cyclic) is False

    assert _is_non_retryable_error(RuntimeError("plain")) is False


# --------------------------------------------------------------------------- #
# _chat error translation (the real safety boundary)                          #
# --------------------------------------------------------------------------- #


class _RaisingBackend(FakeBackend):
    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self._exc = exc

    async def complete(self, **_kwargs: Any) -> LLMChunk:
        raise self._exc


def _loop_with_backend(backend: FakeBackend) -> Any:
    return build_test_agent_loop(
        backend=backend, agent_name=BuiltinAgentName.AUTO_APPROVE
    )


@pytest.mark.asyncio
async def test_chat_translates_rate_limit_to_rate_limit_error() -> None:
    loop = _loop_with_backend(
        _RaisingBackend(_backend_error(status=HTTPStatus.TOO_MANY_REQUESTS))
    )
    with pytest.raises(RateLimitError):
        await loop._chat()


@pytest.mark.asyncio
async def test_chat_translates_context_too_long() -> None:
    exc = _backend_error(status=HTTPStatus.BAD_REQUEST, body="context too long")
    loop = _loop_with_backend(_RaisingBackend(exc))
    with pytest.raises(ContextTooLongError):
        await loop._chat()


@pytest.mark.asyncio
async def test_chat_translates_response_too_long() -> None:
    exc = _backend_error(
        status=HTTPStatus.UNPROCESSABLE_ENTITY, body="max_tokens_exceeded"
    )
    loop = _loop_with_backend(_RaisingBackend(exc))
    with pytest.raises(ResponseTooLongError):
        await loop._chat()


@pytest.mark.asyncio
async def test_chat_reraises_non_retryable_error() -> None:
    class _Terminal(Exception):
        non_retryable = True

    loop = _loop_with_backend(_RaisingBackend(_Terminal("terminal")))
    with pytest.raises(_Terminal):
        await loop._chat()


@pytest.mark.asyncio
async def test_chat_wraps_unknown_backend_error_as_runtime_error() -> None:
    exc = _backend_error(status=HTTPStatus.INTERNAL_SERVER_ERROR, body="boom")
    loop = _loop_with_backend(_RaisingBackend(exc))
    with pytest.raises(RuntimeError, match="API error"):
        await loop._chat()


@pytest.mark.asyncio
async def test_chat_missing_usage_surfaces_as_api_error() -> None:
    class _NoUsageBackend(FakeBackend):
        async def complete(self, **_kwargs: Any) -> LLMChunk:
            return LLMChunk(message=LLMMessage(role=Role.ASSISTANT, content="hi"))

    loop = _loop_with_backend(_NoUsageBackend())
    with pytest.raises(RuntimeError, match="Usage data missing"):
        await loop._chat()


# --------------------------------------------------------------------------- #
# Malformed tool-call arguments -> failed-call path                           #
# --------------------------------------------------------------------------- #


def _garbage_tool_call_chunk() -> LLMChunk:
    return mock_llm_chunk(
        content="",
        tool_calls=[
            ToolCall(
                id="bad-1",
                index=0,
                function=FunctionCall(name="todo", arguments="{not valid json"),
            )
        ],
    )


@pytest.mark.asyncio
async def test_malformed_tool_args_emits_failed_result_without_invoking() -> None:
    backend = FakeBackend([_garbage_tool_call_chunk(), mock_llm_chunk(content="done")])
    loop = build_test_agent_loop(
        backend=backend, agent_name=BuiltinAgentName.AUTO_APPROVE
    )

    events: list[Any] = []
    async for event in loop.act("run it"):
        events.append(event)

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert any(e.tool_call_id == "bad-1" and e.error for e in results)
    assert loop.stats.tool_calls_failed >= 1
    # A tool-role response must be appended so the next turn does not 400.
    assert any(m.role == Role.TOOL and m.tool_call_id == "bad-1" for m in loop.messages)


# --------------------------------------------------------------------------- #
# Tool raising ToolPermissionError rolls back agreed count                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_permission_error_rolls_back_agreed_and_emits_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.stubs.fake_tool import FakeTool

    loop = build_test_agent_loop(
        backend=FakeBackend([
            mock_llm_chunk(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        index=0,
                        function=FunctionCall(name="stub_tool", arguments="{}"),
                    )
                ]
            ),
            mock_llm_chunk(content="ok"),
        ]),
        agent_name=BuiltinAgentName.AUTO_APPROVE,
    )

    # The tool manager validates tool_class is a BaseTool subclass, so register
    # the class (not an instance) and arm its class-level exception flag.
    FakeTool._exception_to_raise = ToolPermissionError("denied")  # type: ignore[attr-defined]
    loop.tool_manager._all_tools["stub_tool"] = FakeTool  # type: ignore[attr-defined]

    events: list[Any] = []
    async for event in loop.act("do it"):
        events.append(event)

    results = [
        e for e in events if isinstance(e, ToolResultEvent) and e.tool_call_id == "c1"
    ]
    assert results and results[0].error and "denied" in results[0].error
    assert loop.stats.tool_calls_rejected == 1
