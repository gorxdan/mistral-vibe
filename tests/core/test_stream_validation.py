from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import (
    _STREAM_DEGENERATE_RETRIES,
    InvalidStreamError,
    _degenerate_response_reason,
)
from vibe.core.types import FunctionCall, Role, ToolCall


def _chunk(
    *,
    content: str = "",
    reasoning: str | None = None,
    tool_calls: list[ToolCall] | None = None,
):
    return mock_llm_chunk(
        content=content, reasoning_content=reasoning, tool_calls=tool_calls
    )


class TestDegenerateResponseReason:
    def test_empty_response_is_degenerate(self) -> None:
        assert _degenerate_response_reason(_chunk(content="")) is not None

    def test_whitespace_only_is_degenerate(self) -> None:
        assert _degenerate_response_reason(_chunk(content="   \n  ")) is not None

    def test_content_is_not_degenerate(self) -> None:
        assert _degenerate_response_reason(_chunk(content="hello")) is None

    def test_tool_calls_not_degenerate(self) -> None:
        chunk = _chunk(
            tool_calls=[ToolCall(index=0, function=FunctionCall(name="read"))]
        )
        assert _degenerate_response_reason(chunk) is None

    def test_reasoning_not_degenerate(self) -> None:
        assert _degenerate_response_reason(_chunk(reasoning="thinking")) is None

    def test_reason_describes_emptiness(self) -> None:
        reason = _degenerate_response_reason(_chunk(content=""))
        assert reason is not None
        assert "empty" in reason


class TestInvalidStreamError:
    def test_carries_reason(self) -> None:
        err = InvalidStreamError("boom")
        assert err.reason == "boom"
        assert "boom" in str(err)


class TestChatStreamingRetry:
    @pytest.mark.asyncio
    async def test_retries_degenerate_then_succeeds(self) -> None:
        # Stream 1: degenerate (empty, but has usage so it passes that check).
        # Stream 2: a valid response with content.
        backend = FakeBackend(
            chunks=[[mock_llm_chunk(content="")], [mock_llm_chunk(content="Hello!")]]
        )
        loop = build_test_agent_loop(backend=backend)

        chunks = [c async for c in loop._chat_streaming()]

        # The valid content reached the caller.
        assert any((c.message.content or "") for c in chunks)
        # Exactly two streaming requests: the failed attempt + the retry.
        assert len(backend.requests_messages) == _STREAM_DEGENERATE_RETRIES
        # Only the valid assistant message was committed; the degenerate one
        # was rejected before append.
        assistants = [m for m in loop.messages if m.role == Role.assistant]
        assert len(assistants) == 1
        assert (assistants[0].content or "") == "Hello!"

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self) -> None:
        backend = FakeBackend(
            chunks=[[mock_llm_chunk(content="")], [mock_llm_chunk(content="")]]
        )
        loop = build_test_agent_loop(backend=backend)

        with pytest.raises(InvalidStreamError):
            [c async for c in loop._chat_streaming()]

        assert len(backend.requests_messages) == _STREAM_DEGENERATE_RETRIES
        # Nothing committed when every attempt was degenerate.
        assert not [m for m in loop.messages if m.role == Role.assistant]

    @pytest.mark.asyncio
    async def test_no_retry_when_response_valid(self) -> None:
        backend = FakeBackend(chunks=[[mock_llm_chunk(content="ok")]])
        loop = build_test_agent_loop(backend=backend)

        chunks = [c async for c in loop._chat_streaming()]

        assert any((c.message.content or "") for c in chunks)
        assert len(backend.requests_messages) == 1
