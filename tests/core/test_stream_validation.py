from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import InvalidStreamError, _degenerate_response_reason
from vibe.core.types import FunctionCall, LLMChunk, LLMMessage, Role, ToolCall


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
    def test_empty_response_with_no_usage_is_degenerate(self) -> None:
        # The only genuinely degenerate case: nothing usable AND no usage. The
        # streaming/non-streaming paths' own ``usage is None`` guard raises
        # before this runs, so this is the defensive backstop trigger.
        chunk = LLMChunk(
            message=LLMMessage(role=Role.assistant, content=""), usage=None
        )
        assert _degenerate_response_reason(chunk) is not None

    def test_empty_response_with_usage_is_not_degenerate(self) -> None:
        # A model can legitimately end its turn with empty content but still
        # report usage (e.g. a follow-up after tool results). Must NOT retry.
        assert _degenerate_response_reason(_chunk(content="")) is None

    def test_whitespace_only_with_usage_is_not_degenerate(self) -> None:
        assert _degenerate_response_reason(_chunk(content="   \n  ")) is None

    def test_content_is_not_degenerate(self) -> None:
        assert _degenerate_response_reason(_chunk(content="hello")) is None

    def test_tool_calls_not_degenerate(self) -> None:
        chunk = _chunk(
            tool_calls=[ToolCall(index=0, function=FunctionCall(name="read"))]
        )
        assert _degenerate_response_reason(chunk) is None

    def test_reasoning_not_degenerate(self) -> None:
        assert _degenerate_response_reason(_chunk(reasoning="thinking")) is None


class TestInvalidStreamError:
    def test_carries_reason(self) -> None:
        err = InvalidStreamError("boom")
        assert err.reason == "boom"
        assert "boom" in str(err)


class TestChatStreamingNoFalsePositive:
    """Regression guards: the degenerate-retry must not fire on legitimate
    empty/whitespace responses (which carry usage), only on the genuinely
    malformed no-usage case the usage guard already catches upstream.
    """

    @pytest.mark.asyncio
    async def test_empty_response_with_usage_not_retried(self) -> None:
        backend = FakeBackend(chunks=[[mock_llm_chunk(content="")]])
        loop = build_test_agent_loop(backend=backend)

        chunks = [c async for c in loop._chat_streaming()]

        # Accepted as a legitimate (empty) turn-end: exactly one request.
        assert len(backend.requests_messages) == 1
        assert all((c.message.content or "") == "" for c in chunks)

    @pytest.mark.asyncio
    async def test_no_retry_when_response_valid(self) -> None:
        backend = FakeBackend(chunks=[[mock_llm_chunk(content="ok")]])
        loop = build_test_agent_loop(backend=backend)

        chunks = [c async for c in loop._chat_streaming()]

        assert any((c.message.content or "") for c in chunks)
        assert len(backend.requests_messages) == 1
