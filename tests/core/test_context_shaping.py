from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.config import ContextShapingConfig, VibeConfig
from vibe.core.config._settings import MicrocompactConfig, SnipConfig
from vibe.core.middleware import (
    ContextShaperMiddleware,
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


def _config(**shaping) -> VibeConfig:
    shaping.setdefault("snip", SnipConfig(keep_recent_turns=1, min_message_tokens=50))
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
        LLMMessage(role=Role.SYSTEM, content="system prompt"),
        LLMMessage(role=Role.USER, content="please do the thing"),
        LLMMessage(
            role=Role.ASSISTANT,
            content=_content(350),
            tool_calls=[
                ToolCall(
                    id="call_1",
                    index=0,
                    function=FunctionCall(name="bash", arguments='{"cmd":"ls"}'),
                )
            ],
        ),
        LLMMessage(role=Role.TOOL, content=_content(400), tool_call_id="call_1"),
        LLMMessage(role=Role.ASSISTANT, content=_content(350)),
        LLMMessage(role=Role.ASSISTANT, content="recent reply"),
    ]


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (880_000, 256_000),  # glm / fugu — 1M window, capped
        (400_000, 256_000),  # gpt-5.5 / minimax — capped
        (256_000, 256_000),  # exactly at the cap
        (200_000, 200_000),  # kimi — below cap, unchanged
        (108_800, 108_800),  # codex-spark — below cap, unchanged
    ],
)
def test_shaping_base_is_capped(threshold: int, expected: int) -> None:
    # snip/microcompact watermarks scale off this base; it is capped so a
    # giant-window model starts proactive shaping at the same absolute point as
    # a small one instead of hoarding context up to a window-pinned threshold.
    # Full auto-compaction (AutoCompactMiddleware) reads the raw threshold and is
    # unaffected.
    cfg = build_test_vibe_config()
    cfg.models[0].auto_compact_threshold = threshold
    cfg.active_model = cfg.models[0].alias
    ctx = _ctx([LLMMessage(role=Role.SYSTEM, content="s")], cfg)
    assert ContextShaperMiddleware._threshold(ctx) == expected


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
    assert any((m.content or "").startswith("<vibe_snipped>") for m in ctx.messages)


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


def _history_with_reasoning() -> list[LLMMessage]:
    # Attach reasoning_content to the big standalone assistant turn (idx 4).
    msgs = _history()
    msgs[4] = msgs[4].model_copy(update={"reasoning_content": "step-by-step thoughts"})
    return msgs


@pytest.mark.asyncio
async def test_snip_strips_reasoning_by_default() -> None:
    ctx = _ctx(_history_with_reasoning(), _config())
    await SnipMiddleware().before_turn(ctx)

    snipped = ctx.messages[4]
    assert (snipped.content or "").startswith("<vibe_snipped>")  # was elided
    assert snipped.reasoning_content is None  # default: reasoning dropped


@pytest.mark.asyncio
async def test_snip_preserves_reasoning_when_model_requires_it() -> None:
    cfg = _config()
    cfg.models[0].preserve_reasoning = True  # Kimi/GLM Preserved Thinking
    ctx = _ctx(_history_with_reasoning(), cfg)
    await SnipMiddleware().before_turn(ctx)

    snipped = ctx.messages[4]
    assert (snipped.content or "").startswith("<vibe_snipped>")  # still elided
    assert snipped.reasoning_content == "step-by-step thoughts"  # kept verbatim


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
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.USER, content=_content(500)),  # big REAL user msg
        LLMMessage(role=Role.ASSISTANT, content=_content(500)),
        LLMMessage(role=Role.ASSISTANT, content="recent"),
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
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content=_content(50)),
        LLMMessage(role=Role.ASSISTANT, content="recent"),
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


@pytest.mark.asyncio
async def test_microcompact_tags_with_sentinel() -> None:
    cfg = _config(
        microcompact=MicrocompactConfig(
            enabled=True, high_watermark=0.6, target=0.5, per_message_cap_tokens=100
        )
    )
    ctx = _ctx(_history(), cfg)
    await MicrocompactMiddleware().before_turn(ctx)
    tagged = [
        m for m in ctx.messages if (m.content or "").startswith("<vibe_microcompacted>")
    ]
    assert len(tagged) == 1


@pytest.mark.asyncio
async def test_microcompact_is_idempotent() -> None:
    cfg = _config(
        microcompact=MicrocompactConfig(
            enabled=True,
            high_watermark=0.6,
            target=0.5,
            per_message_cap_tokens=100,
            max_blocks_per_turn=10,  # exhaust the band in one pass
        )
    )
    ctx = _ctx(_history(), cfg)
    await MicrocompactMiddleware().before_turn(ctx)
    snapshot = [m.content for m in ctx.messages]
    await MicrocompactMiddleware().before_turn(ctx)
    assert [m.content for m in ctx.messages] == snapshot


@pytest.mark.asyncio
async def test_snip_preserves_persisted_output_path() -> None:
    # A tool result carrying a persisted-output disk path must carry that path
    # into the snip placeholder so the recovery contract survives shaping.
    path = "/tmp/sess/tool_results/call_abc.txt"
    tool_with_path = LLMMessage(
        role=Role.TOOL,
        content=(
            _content(400) + "\n\n…[Full output (100,000 characters) persisted to "
            f"{path}; use the `read` tool to retrieve it.]…"
        ),
        tool_call_id="call_p",
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.USER, content="please do the thing"),
        LLMMessage(
            role=Role.ASSISTANT,
            content=_content(400),
            tool_calls=[
                ToolCall(
                    id="call_p",
                    index=0,
                    function=FunctionCall(name="bash", arguments='{"cmd":"build"}'),
                )
            ],
        ),
        tool_with_path,
        LLMMessage(role=Role.ASSISTANT, content="recent reply"),
    ]
    ctx = _ctx(msgs, _config())
    await SnipMiddleware().before_turn(ctx)
    # The snipped tool message carries the path into the placeholder.
    snipped = [
        m for m in ctx.messages if (m.content or "").startswith("<vibe_snipped>")
    ]
    assert snipped
    assert any(path in (m.content or "") for m in snipped)
