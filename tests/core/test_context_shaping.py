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
        # Big tool output -> persisted to disk, so it is snip's (recoverable) domain.
        LLMMessage(
            role=Role.TOOL,
            content=_content(400) + " persisted to /tmp/vibe/call_1.txt;",
            tool_call_id="call_1",
        ),
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


@pytest.mark.parametrize(
    ("context_window", "threshold", "expected"),
    [
        (None, 880_000, 256_000),  # undeclared window — flat cap unchanged
        (1_000_000, 880_000, 500_000),  # glm/fugu — cap lifted to 0.5x window
        (1_000_000, 400_000, 400_000),  # threshold binds; the cap no longer does
        (400_000, 400_000, 256_000),  # declared small window — max floors at 256k
        (262_144, 200_000, 200_000),  # small window AND threshold below the cap
    ],
)
def test_shaping_cap_window_relative(
    context_window: int | None, threshold: int, expected: int
) -> None:
    cfg = build_test_vibe_config()
    cfg.models[0].auto_compact_threshold = threshold
    cfg.models[0].context_window = context_window
    cfg.active_model = cfg.models[0].alias
    ctx = _ctx([LLMMessage(role=Role.SYSTEM, content="s")], cfg)
    assert ContextShaperMiddleware._threshold(ctx) == expected


def test_shaping_cap_fraction_zero_restores_flat_cap() -> None:
    cfg = build_test_vibe_config(
        context_shaping=ContextShapingConfig(cap_window_fraction=0.0)
    )
    cfg.models[0].auto_compact_threshold = 880_000
    cfg.models[0].context_window = 1_000_000
    cfg.active_model = cfg.models[0].alias
    ctx = _ctx([LLMMessage(role=Role.SYSTEM, content="s")], cfg)
    assert ContextShaperMiddleware._threshold(ctx) == 256_000


def test_protected_prefix_band_extends_past_large_system_prompt() -> None:
    # The guard band counts tokens AFTER the system prompt; a big system prompt
    # must not consume the band and leave the first history messages editable.
    msgs = MessageList([
        LLMMessage(role=Role.SYSTEM, content=_content(200)),
        LLMMessage(role=Role.ASSISTANT, content=_content(30)),
        LLMMessage(role=Role.ASSISTANT, content=_content(30)),
        LLMMessage(role=Role.ASSISTANT, content="tail"),
    ])
    assert ContextShaperMiddleware._protected_prefix_len(msgs, guard_tokens=50) == 3


@pytest.mark.asyncio
async def test_microcompact_respects_guard_band_after_large_system() -> None:
    # With a prod-sized system prompt the eligible block right after it sits in
    # the guard band (protected whole, crossing the boundary); edits start beyond.
    cfg = _config(
        microcompact=MicrocompactConfig(enabled=True, per_message_cap_tokens=100)
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content=_content(2000)),
        LLMMessage(role=Role.ASSISTANT, content=_content(300)),
        LLMMessage(role=Role.ASSISTANT, content=_content(300)),
        LLMMessage(role=Role.ASSISTANT, content="recent reply"),
    ]
    ctx = _ctx(msgs, cfg)
    await MicrocompactMiddleware().before_turn(ctx)

    assert ctx.messages[1].content == _content(300)  # in-band: survives verbatim
    assert (ctx.messages[2].content or "").startswith("<vibe_microcompacted>")


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
async def test_microcompact_strips_reasoning_by_default() -> None:
    # Reasoning-bearing assistant turns are non-recoverable -> microcompact's
    # domain. It inherits snip's rule: drop the stale reasoning by default.
    cfg = _config(
        microcompact=MicrocompactConfig(enabled=True, per_message_cap_tokens=100)
    )
    ctx = _ctx(_history_with_reasoning(), cfg)
    await MicrocompactMiddleware().before_turn(ctx)

    block = ctx.messages[4]
    assert (block.content or "").startswith("<vibe_microcompacted>")  # compressed
    assert block.reasoning_content is None  # default: reasoning dropped


@pytest.mark.asyncio
async def test_microcompact_preserves_reasoning_when_model_requires_it() -> None:
    cfg = _config(
        microcompact=MicrocompactConfig(enabled=True, per_message_cap_tokens=100)
    )
    cfg.models[0].preserve_reasoning = True  # Kimi/GLM Preserved Thinking
    ctx = _ctx(_history_with_reasoning(), cfg)
    await MicrocompactMiddleware().before_turn(ctx)

    block = ctx.messages[4]
    assert (block.content or "").startswith("<vibe_microcompacted>")  # compressed
    assert block.reasoning_content == "step-by-step thoughts"  # kept verbatim


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


def _mixed_history() -> list[LLMMessage]:
    # A recoverable (disk-backed) block and a same-size non-recoverable block,
    # both old and eligible. Routing must send each to a different shaper.
    return [
        LLMMessage(role=Role.SYSTEM, content="system prompt"),
        LLMMessage(role=Role.USER, content="do the thing"),
        LLMMessage(
            role=Role.TOOL,
            content=_content(500) + " persisted to /tmp/o1.txt;",  # recoverable
            tool_call_id="c1",
        ),
        LLMMessage(role=Role.ASSISTANT, content=_content(500)),  # non-recoverable
        LLMMessage(role=Role.ASSISTANT, content="recent reply"),
    ]


@pytest.mark.asyncio
async def test_snip_targets_recoverable_skips_nonrecoverable() -> None:
    # snip owns disk-backed content: it elides the block carrying a persisted
    # path and leaves the path-less block for microcompact.
    msgs = _mixed_history()
    ctx = _ctx(msgs, _config(cache_prefix_guard_tokens=0))
    await SnipMiddleware().before_turn(ctx)
    assert (ctx.messages[2].content or "").startswith("<vibe_snipped>")  # recoverable
    assert ctx.messages[3].content == _content(500)  # non-recoverable untouched


@pytest.mark.asyncio
async def test_microcompact_regists_oversized_prior_gist() -> None:
    # An already-microcompacted block still above the cap is re-gisted smaller
    # (reclaims the accumulated floor); the marker is not nested.
    from vibe.core.middleware import _MC_OPEN

    cfg = _config(
        cache_prefix_guard_tokens=0,
        microcompact=MicrocompactConfig(per_message_cap_tokens=100),
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content=f"{_MC_OPEN} " + _content(700)),
        LLMMessage(role=Role.ASSISTANT, content="recent"),
    ]
    ctx = _ctx(msgs, cfg)
    await MicrocompactMiddleware().before_turn(ctx)

    block = ctx.messages[1].content or ""
    assert block.startswith(_MC_OPEN)
    assert block.count(_MC_OPEN) == 1  # re-gisted, not nested
    assert len(block) < len(f"{_MC_OPEN} " + _content(700)) // 2  # shrank


@pytest.mark.asyncio
async def test_microcompact_skips_capped_gist_and_processes_new_block() -> None:
    # Regression: a block already gisted to the cap looked re-eligible (the marker
    # pushes its full content just over the cap) and was re-churned to no effect,
    # burning the per-turn block budget so NEW big content never got gisted and the
    # session climbed (blocks=4 shed=0 in a live trace). The eligibility check must
    # measure the marker-stripped body.
    from vibe.core.middleware import _MC_OPEN

    cfg = _config(
        cache_prefix_guard_tokens=0,
        microcompact=MicrocompactConfig(
            per_message_cap_tokens=100, max_blocks_per_turn=1
        ),
    )
    capped = f"{_MC_OPEN} " + _content(100)  # body already at the cap
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.USER, content="go"),
        LLMMessage(role=Role.ASSISTANT, content=capped),  # oldest, already capped
        LLMMessage(role=Role.ASSISTANT, content=_content(600)),  # new, gistable
        LLMMessage(role=Role.ASSISTANT, content="recent"),
    ]
    ctx = _ctx(msgs, cfg)
    await MicrocompactMiddleware().before_turn(ctx)

    # The one block slot went to the fresh block, not the no-op capped one.
    assert ctx.messages[2].content == capped  # already-capped gist left alone
    assert (ctx.messages[3].content or "").startswith(_MC_OPEN)  # new block gisted


def test_microcompact_per_message_cap_lowered() -> None:
    assert MicrocompactConfig().per_message_cap_tokens == 1000


def test_microcompact_target_below_watermark_invariant() -> None:
    # target must be < high_watermark; otherwise the loop's `est <= target` break
    # fires the instant the gate (`est >= high_watermark`) opens, so nothing is
    # gisted and the watermark is inert (micro would effectively engage at the
    # higher target, not the watermark).
    cfg = MicrocompactConfig()
    assert cfg.target < cfg.high_watermark
    assert cfg.target == 0.6


def test_microcompact_default_rate_raised() -> None:
    # 1/turn treaded water at the watermark in prod; several blocks/turn lets
    # microcompact actually reduce non-recoverable bloat.
    assert MicrocompactConfig().max_blocks_per_turn == 4


@pytest.mark.asyncio
async def test_microcompact_compresses_up_to_rate_per_pass() -> None:
    cfg = _config(
        cache_prefix_guard_tokens=0,
        microcompact=MicrocompactConfig(per_message_cap_tokens=100),  # default rate=4
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.USER, content="go"),
    ]
    for _ in range(6):  # 6 eligible non-recoverable blocks; rate caps the pass
        msgs.append(LLMMessage(role=Role.ASSISTANT, content=_content(400)))
    msgs.append(LLMMessage(role=Role.ASSISTANT, content="recent"))
    ctx = _ctx(msgs, cfg)
    await MicrocompactMiddleware().before_turn(ctx)
    n = sum(
        1 for m in ctx.messages if (m.content or "").startswith("<vibe_microcompacted>")
    )
    assert n == 4  # bounded by max_blocks_per_turn, not all 6


def test_microcompact_default_watermark_tightened() -> None:
    # Tightened to ~snip's 0.6 so the two shapers engage together, closing the
    # 51k unshaped band a live glm session climbed through (153.6k->204.8k).
    assert MicrocompactConfig().high_watermark == 0.65


@pytest.mark.asyncio
async def test_microcompact_engages_in_old_gap_band() -> None:
    # est ~0.7x threshold: above the new 0.65 watermark, below the old 0.8 — so
    # this would NOT have fired before and must now.
    cfg = _config(
        cache_prefix_guard_tokens=0,
        microcompact=MicrocompactConfig(per_message_cap_tokens=100),  # default 0.65
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content=_content(350)),
        LLMMessage(role=Role.ASSISTANT, content=_content(350)),
        LLMMessage(role=Role.ASSISTANT, content="recent"),
    ]
    ctx = _ctx(msgs, cfg)  # est ~703 of 1000 threshold -> 0.70x
    await MicrocompactMiddleware().before_turn(ctx)
    assert any(
        (m.content or "").startswith("<vibe_microcompacted>") for m in ctx.messages
    )


@pytest.mark.asyncio
async def test_microcompact_targets_nonrecoverable_skips_recoverable() -> None:
    # microcompact owns non-recoverable content: it gist-truncates the path-less
    # block and leaves the disk-backed block for snip.
    msgs = _mixed_history()
    cfg = _config(
        cache_prefix_guard_tokens=0,
        microcompact=MicrocompactConfig(
            enabled=True, per_message_cap_tokens=100, max_blocks_per_turn=5
        ),
    )
    ctx = _ctx(msgs, cfg)
    await MicrocompactMiddleware().before_turn(ctx)
    assert (ctx.messages[3].content or "").startswith("<vibe_microcompacted>")
    assert ctx.messages[2].content == _content(500) + " persisted to /tmp/o1.txt;"
