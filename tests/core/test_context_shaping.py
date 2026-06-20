from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.config import ContextShapingConfig
from vibe.core.config._settings import MicrocompactConfig, SnipConfig
from vibe.core.middleware import (
    ConversationContext,
    MicrocompactMiddleware,
    SnipMiddleware,
)
from vibe.core.types import (
    AgentStats,
    FunctionCall,
    LLMMessage,
    MessageList,
    Role,
    ToolCall,
)

THRESHOLD = 1000  # tokens; watermarks are fractions of this


def _content(tokens: int) -> str:
    return "x" * (tokens * 4)  # approx_token_count ≈ len/4


def _config(**shaping) -> object:
    shaping.setdefault(
        "snip", SnipConfig(keep_recent_turns=1, min_message_tokens=50)
    )
    shaping.setdefault("cache_prefix_guard_tokens", 50)
    cfg = build_test_vibe_config(context_shaping=ContextShapingConfig(**shaping))
    cfg.models[0].auto_compact_threshold = THRESHOLD
    cfg.active_model = cfg.models[0].alias
    return cfg


def _ctx(messages: list[LLMMessage], cfg, context_tokens: int = 0):
    stats = AgentStats()
    stats.context_tokens = context_tokens
    return ConversationContext(messages=MessageList(messages), stats=stats, config=cfg)


def _history() -> list[LLMMessage]:
    # system + real user + big assistant(tool_calls) + big tool result + big
    # assistant text + recent. Total ~1300 tokens of local estimate.
    return [
        LLMMessage(role=Role.system, content="system prompt"),
        LLMMessage(role=Role.user, content="please do the thing"),
        LLMMessage(
            role=Role.assistant,
            content=_content(350),
            tool_calls=[
                ToolCall(
                    id="call_1",
                    index=0,
                    function=FunctionCall(name="bash", arguments='{"cmd":"ls"}'),
                )
            ],
        ),
        LLMMessage(role=Role.tool, content=_content(400), tool_call_id="call_1"),
        LLMMessage(role=Role.assistant, content=_content(350)),
        LLMMessage(role=Role.assistant, content="recent reply"),
    ]


@pytest.mark.asyncio
async def test_snip_elides_oldest_large_and_preserves_bookends() -> None:
    msgs = _history()
    ctx = _ctx(msgs, _config())
    await SnipMiddleware().before_turn(ctx)

    # System + real user + most-recent are untouched.
    assert ctx.messages[0].content == "system prompt"
    assert ctx.messages[1].content == "please do the thing"
    assert ctx.messages[-1].content == "recent reply"
    # At least one big message got elided.
    assert any(
        (m.content or "").startswith("<vibe_snipped>") for m in ctx.messages
    )


@pytest.mark.asyncio
async def test_snip_preserves_tool_linkage() -> None:
    msgs = _history()
    ctx = _ctx(msgs, _config())
    await SnipMiddleware().before_turn(ctx)

    assistant = ctx.messages[2]
    tool = ctx.messages[3]
    if (assistant.content or "").startswith("<vibe_snipped>"):
        assert assistant.tool_calls is not None
        assert assistant.tool_calls[0].id == "call_1"
        assert assistant.tool_calls[0].function.name == "bash"
        assert assistant.tool_calls[0].function.arguments == "{}"
    if (tool.content or "").startswith("<vibe_snipped>"):
        assert tool.tool_call_id == "call_1"  # linkage intact


@pytest.mark.asyncio
async def test_snip_is_idempotent() -> None:
    cfg = _config()
    ctx = _ctx(_history(), cfg)
    await SnipMiddleware().before_turn(ctx)
    snapshot = [m.content for m in ctx.messages]
    await SnipMiddleware().before_turn(ctx)
    assert [m.content for m in ctx.messages] == snapshot


@pytest.mark.asyncio
async def test_snip_never_touches_real_user_message() -> None:
    msgs = [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content=_content(500)),  # big REAL user msg
        LLMMessage(role=Role.assistant, content=_content(500)),
        LLMMessage(role=Role.assistant, content="recent"),
    ]
    ctx = _ctx(msgs, _config())
    await SnipMiddleware().before_turn(ctx)
    # The real user message must survive verbatim even though it is large.
    assert ctx.messages[1].content == _content(500)


@pytest.mark.asyncio
async def test_disabled_is_noop() -> None:
    cfg = _config(snip=SnipConfig(enabled=False))
    cfg.models[0].auto_compact_threshold = THRESHOLD
    cfg.active_model = cfg.models[0].alias
    before = _history()
    ctx = _ctx(list(before), cfg)
    await SnipMiddleware().before_turn(ctx)
    assert [m.content for m in ctx.messages] == [m.content for m in before]


@pytest.mark.asyncio
async def test_threshold_zero_is_noop() -> None:
    cfg = _config()
    cfg.models[0].auto_compact_threshold = 0
    cfg.active_model = cfg.models[0].alias
    before = _history()
    ctx = _ctx(list(before), cfg)
    await SnipMiddleware().before_turn(ctx)
    assert [m.content for m in ctx.messages] == [m.content for m in before]


@pytest.mark.asyncio
async def test_below_watermark_is_noop() -> None:
    cfg = _config()
    small = [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.assistant, content=_content(50)),
        LLMMessage(role=Role.assistant, content="recent"),
    ]
    ctx = _ctx(list(small), cfg)
    await SnipMiddleware().before_turn(ctx)
    assert all(not (m.content or "").startswith("<vibe_snipped>") for m in ctx.messages)


@pytest.mark.asyncio
async def test_microcompact_truncates_oldest_oversized() -> None:
    cfg = _config(
        microcompact=MicrocompactConfig(
            enabled=True, high_watermark=0.6, target=0.5, per_message_cap_tokens=100
        )
    )
    msgs = _history()
    ctx = _ctx(msgs, cfg)
    await MicrocompactMiddleware().before_turn(ctx)
    # Exactly one block (default max_blocks_per_turn=1) compressed: the oldest
    # oversized message now contains the truncation marker and is smaller.
    compressed = [m for m in ctx.messages if "[... truncated ...]" in (m.content or "")]
    assert len(compressed) == 1
    assert ctx.messages[-1].content == "recent reply"  # suffix untouched
