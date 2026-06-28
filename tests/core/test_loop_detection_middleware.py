from __future__ import annotations

import pytest

from vibe.core.config import VibeConfig
from vibe.core.middleware import (
    ConversationContext,
    LoopDetectionMiddleware,
    MiddlewareAction,
    _canonical_tool_args,
    _trailing_tool_call_fingerprints,
)
from vibe.core.types import (
    AgentStats,
    FunctionCall,
    LLMMessage,
    MessageList,
    Role,
    ToolCall,
)


def _assistant_call(name: str, arguments: str, index: int = 0) -> LLMMessage:
    return LLMMessage(
        role=Role.ASSISTANT,
        tool_calls=[
            ToolCall(index=index, function=FunctionCall(name=name, arguments=arguments))
        ],
    )


def _tool_result() -> LLMMessage:
    return LLMMessage(role=Role.TOOL, content="ok")


@pytest.fixture
def ctx(vibe_config: VibeConfig) -> ConversationContext:
    return ConversationContext(
        messages=MessageList(), stats=AgentStats(), config=vibe_config
    )


class TestCanonicalToolArgs:
    def test_none_and_empty(self) -> None:
        assert _canonical_tool_args(None) == ""
        assert _canonical_tool_args("") == ""

    def test_key_order_insensitive(self) -> None:
        assert _canonical_tool_args('{"b": 1, "a": 2}') == _canonical_tool_args(
            '{"a": 2, "b": 1}'
        )

    def test_whitespace_insensitive(self) -> None:
        assert _canonical_tool_args('{"a": 1, "b": 2}') == _canonical_tool_args(
            '{ "a":1, "b":2 }'
        )

    def test_invalid_json_falls_back_to_raw(self) -> None:
        raw = "not json{"
        assert _canonical_tool_args(raw) == raw


class TestTrailingFingerprints:
    def test_collects_only_assistant_tool_calls(self, vibe_config: VibeConfig) -> None:
        messages = MessageList([
            _assistant_call("read", '{"path": "a.py"}'),
            _tool_result(),
            _assistant_call("grep", '{"pattern": "x"}'),
            LLMMessage(role=Role.ASSISTANT, content="thinking"),
        ])
        fps = _trailing_tool_call_fingerprints(messages, limit=10)
        assert fps == [("read", '{"path": "a.py"}'), ("grep", '{"pattern": "x"}')]

    def test_truncates_to_limit(self, vibe_config: VibeConfig) -> None:
        messages = MessageList([
            _assistant_call("read", f'{{"i": {i}}}') for i in range(8)
        ])
        fps = _trailing_tool_call_fingerprints(messages, limit=3)
        assert len(fps) == 3
        assert fps[-1] == ("read", '{"i": 7}')

    def test_canonicalizes_equivalent_args(self, vibe_config: VibeConfig) -> None:
        messages = MessageList([
            _assistant_call("read", '{"a": 1, "b": 2}'),
            _assistant_call("read", '{"b": 2, "a": 1}'),
        ])
        fps = _trailing_tool_call_fingerprints(messages, limit=10)
        assert fps[0] == fps[1]


class TestLoopDetectionMiddleware:
    @pytest.mark.asyncio
    async def test_continue_when_below_threshold(
        self, ctx: ConversationContext
    ) -> None:
        mw = LoopDetectionMiddleware(threshold=5)
        ctx.messages = MessageList([
            _assistant_call("read", '{"path": "a.py"}') for _ in range(4)
        ])
        result = await mw.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_strike_one_injects_nudge(self, ctx: ConversationContext) -> None:
        mw = LoopDetectionMiddleware(threshold=5)
        ctx.messages = MessageList([
            _assistant_call("read", '{"path": "a.py"}') for _ in range(5)
        ])
        result = await mw.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
        assert "read" in (result.message or "")
        assert "stuck" in (result.message or "").lower()

    @pytest.mark.asyncio
    async def test_strike_two_stops(self, ctx: ConversationContext) -> None:
        mw = LoopDetectionMiddleware(threshold=5)
        ctx.messages = MessageList([
            _assistant_call("read", '{"path": "a.py"}') for _ in range(5)
        ])
        await mw.before_turn(ctx)  # strike 1
        result = await mw.before_turn(ctx)  # strike 2, still looping
        assert result.action == MiddlewareAction.STOP
        assert "read" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_different_call_resets_warning(
        self, ctx: ConversationContext
    ) -> None:
        mw = LoopDetectionMiddleware(threshold=5)
        ctx.messages = MessageList([
            _assistant_call("read", '{"path": "a.py"}') for _ in range(5)
        ])
        await mw.before_turn(ctx)  # strike 1 -> warned

        # A single different call breaks the trailing-identical window.
        ctx.messages = MessageList(
            [_assistant_call("read", '{"path": "a.py"}') for _ in range(4)]
            + [_assistant_call("grep", '{"pattern": "x"}')]
        )
        result = await mw.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

        # Resuming the loop re-warns (fresh strike 1), not STOP.
        ctx.messages = MessageList([
            _assistant_call("read", '{"path": "a.py"}') for _ in range(5)
        ])
        result = await mw.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE

    @pytest.mark.asyncio
    async def test_mixed_args_not_a_loop(self, ctx: ConversationContext) -> None:
        mw = LoopDetectionMiddleware(threshold=5)
        ctx.messages = MessageList([
            _assistant_call("read", f'{{"path": "f{i}.py"}}') for i in range(5)
        ])
        result = await mw.before_turn(ctx)
        assert result.action == MiddlewareAction.CONTINUE

    @pytest.mark.asyncio
    async def test_equivalent_args_json_detected_as_loop(
        self, ctx: ConversationContext
    ) -> None:
        mw = LoopDetectionMiddleware(threshold=5)
        # Same logical args, different JSON formatting on each call.
        args_variants = [
            '{"path": "a.py", "limit": 10}',
            '{ "path":"a.py", "limit":10 }',
            '{"limit": 10, "path": "a.py"}',
            '{"path":"a.py","limit":10}',
            '{ "limit":10, "path" : "a.py" }',
        ]
        ctx.messages = MessageList([_assistant_call("read", a) for a in args_variants])
        result = await mw.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE

    @pytest.mark.asyncio
    async def test_reset_clears_warning(self, ctx: ConversationContext) -> None:
        from vibe.core.middleware import ResetReason

        mw = LoopDetectionMiddleware(threshold=5)
        ctx.messages = MessageList([
            _assistant_call("read", '{"path": "a.py"}') for _ in range(5)
        ])
        await mw.before_turn(ctx)  # strike 1 -> warned
        mw.reset(ResetReason.STOP)
        # After reset, the same looping history warns again rather than STOPs.
        result = await mw.before_turn(ctx)
        assert result.action == MiddlewareAction.INJECT_MESSAGE
