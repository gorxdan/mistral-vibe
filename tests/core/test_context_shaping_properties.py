from __future__ import annotations

from dataclasses import dataclass
import random

import pytest

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from vibe.core.config import ContextShapingConfig, VibeConfig
from vibe.core.config._settings import MicrocompactConfig, SnipConfig
from vibe.core.middleware import (
    _MC_OPEN,
    ConversationContext,
    MicrocompactMiddleware,
    MiddlewarePipeline,
    ResetReason,
)
from vibe.core.types import AgentStats, LLMMessage, MessageList, Role
from vibe.core.utils.tokens import approx_token_count

THRESHOLD = 1000  # tokens; watermarks are fractions of this


def _content(tokens: int) -> str:
    return "x" * (tokens * 4)  # approx_token_count == len/4 exactly


def _assistant(tokens: int) -> LLMMessage:
    return LLMMessage(role=Role.ASSISTANT, content=_content(tokens))


def _user(tokens: int) -> LLMMessage:
    return LLMMessage(role=Role.USER, content=_content(tokens))


def _config(*, min_shed: int, max_blocks: int) -> VibeConfig:
    cfg = build_test_vibe_config(
        context_shaping=ContextShapingConfig(
            snip=SnipConfig(keep_recent_turns=1, min_message_tokens=50),
            microcompact=MicrocompactConfig(
                per_message_cap_tokens=100,
                min_shed_tokens=min_shed,
                max_blocks_per_turn=max_blocks,
            ),
            cache_prefix_guard_tokens=0,
        )
    )
    cfg.models[0].auto_compact_threshold = THRESHOLD
    cfg.active_model = cfg.models[0].alias
    return cfg


def _ctx(
    messages: list[LLMMessage], cfg: VibeConfig, context_tokens: int = 0
) -> ConversationContext:
    stats = AgentStats()
    stats.context_tokens = context_tokens
    return ConversationContext(messages=MessageList(messages), stats=stats, config=cfg)


def _base_messages() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.USER, content="go"),
        _assistant(2),
    ]


def _local(messages: MessageList) -> int:
    return sum(approx_token_count(m.content or "") for m in messages)


def _est(ctx: ConversationContext) -> int:
    return max(ctx.stats.context_tokens, _local(ctx.messages))


def _grow(ctx: ConversationContext, tokens: int) -> None:
    # Insert just inside the protected keep_recent_turns=1 tail so the new
    # block is immediately eligible.
    ctx.messages.insert(len(ctx.messages) - 1, _assistant(tokens))


async def _run_pass(
    mw: MicrocompactMiddleware, ctx: ConversationContext
) -> tuple[set[int], int]:
    before = [m.content for m in ctx.messages]
    pre_local = _local(ctx.messages)
    await mw.before_turn(ctx)
    after = [m.content for m in ctx.messages]
    changed = {i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a}
    return changed, pre_local - _local(ctx.messages)


def _gisted_tokens(cap: int) -> int:
    # What one gisted block actually costs: cap + the marker's ~6 tokens.
    return approx_token_count(f"{_MC_OPEN} " + "x" * (cap * 4))


@dataclass
class _Verdict:
    fires: bool
    cooldown: int | None
    plan: list[int]
    projected: int
    shed: int
    exhausted: bool


def _model_plan(
    ctx: ConversationContext, cfg: MicrocompactConfig, plan_est: int, target: float
) -> tuple[list[int], int, bool]:
    messages = ctx.messages
    prefix = 1  # guard band is 0 in every model-checked config
    suffix = min(
        ctx.config.context_shaping.snip.keep_recent_turns,
        max(0, len(messages) - prefix),
    )
    plan: list[int] = []
    projected = 0
    for i in range(prefix, len(messages) - suffix):
        if plan_est - projected <= target:
            return plan, projected, False
        if cfg.max_blocks_per_turn > 0 and len(plan) >= cfg.max_blocks_per_turn:
            return plan, projected, False
        msg = messages[i]
        body = msg.content or ""
        if msg.role == Role.USER and not msg.injected:
            continue
        if body.startswith(_MC_OPEN):
            body = body[len(_MC_OPEN) :].lstrip()
        body_tokens = approx_token_count(body)
        if body_tokens <= cfg.per_message_cap_tokens:
            continue
        plan.append(i)
        projected += body_tokens - cfg.per_message_cap_tokens
    return plan, projected, plan_est - projected > target


def _model_pass(
    ctx: ConversationContext, cooldown: int | None, *, emergency: bool = False
) -> _Verdict:
    # Spec model of MicrocompactMiddleware.before_turn's guard state machine.
    cfg = ctx.config.context_shaping.microcompact
    local = _local(ctx.messages)
    est = max(ctx.stats.context_tokens, local)
    if est < cfg.high_watermark * THRESHOLD:
        return _Verdict(False, cooldown, [], 0, 0, False)
    guarded = not emergency and cfg.max_blocks_per_turn == 0
    if guarded and cooldown is not None:
        if est <= cooldown:
            return _Verdict(False, cooldown, [], 0, 0, False)
        cooldown = None
    target = cfg.target * THRESHOLD
    plan, projected, exhausted = _model_plan(
        ctx, cfg, est if emergency else local, target
    )
    if not plan or (guarded and projected < cfg.min_shed_tokens):
        return _Verdict(
            False, est if guarded else cooldown, [], projected, 0, exhausted
        )
    shed = sum(
        approx_token_count(ctx.messages[i].content or "")
        - _gisted_tokens(cfg.per_message_cap_tokens)
        for i in plan
    )
    if guarded and exhausted and est - shed > target:
        cooldown = est
    return _Verdict(True, cooldown, plan, projected, shed, exhausted)


@pytest.mark.asyncio
async def test_emergency_pass_never_suppressed() -> None:
    # Reactive hard-overflow shedding fires through every guard state: armed
    # cooldown, unreachable floor, batch or dribble mode, stats-pinned est.
    for seed in range(20):
        rng = random.Random(seed)
        max_blocks = rng.choice([0, 3])
        cfg = _config(min_shed=10**6, max_blocks=max_blocks)
        msgs = _base_messages()
        for _ in range(rng.randint(1, 5)):
            msgs.insert(len(msgs) - 1, _assistant(rng.randint(150, 600)))
        stats0 = rng.randint(1000, 50_000)
        ctx = _ctx(msgs, cfg, context_tokens=stats0)
        mw = MicrocompactMiddleware(emergency=True)
        armed = rng.choice([None, stats0, 10**9])
        mw._cooldown_est = armed

        changed, dropped = await _run_pass(mw, ctx)

        assert changed
        assert dropped > 0
        if max_blocks:
            assert len(changed) <= max_blocks
        assert mw._cooldown_est == armed  # emergency neither arms nor clears


@pytest.mark.asyncio
async def test_cooldown_unwedges_when_est_rises() -> None:
    for seed in range(12):
        rng = random.Random(seed)
        cfg = _config(min_shed=5000, max_blocks=0)
        mw = MicrocompactMiddleware()
        # Ineligible real-user bulk keeps every firing pass pool-exhausted
        # above target; stats pinned above local so post-fire est never drops.
        bulk = rng.randint(800, 2000)
        stats0 = rng.randint(6000, 12_000)
        msgs = [
            LLMMessage(role=Role.SYSTEM, content="sys"),
            _user(bulk),
            _assistant(rng.randint(150, 800)),
            _assistant(2),
        ]
        ctx = _ctx(msgs, cfg, context_tokens=stats0)

        changed, _ = await _run_pass(mw, ctx)  # projected < floor: skip-arms
        assert not changed
        assert mw._cooldown_est == stats0

        for _ in range(2):
            armed = mw._cooldown_est
            assert armed is not None
            changed, _ = await _run_pass(mw, ctx)  # est not risen: suppressed
            assert not changed
            if rng.random() < 0.5:
                ctx.stats.context_tokens = armed + rng.randint(1, 3000)
                _grow(ctx, rng.randint(5200, 7000))
            else:
                need = max(5200, armed - _local(ctx.messages) + 101)
                _grow(ctx, need + rng.randint(0, 1500))
            assert _est(ctx) > armed
            changed, _ = await _run_pass(mw, ctx)
            assert changed  # any rise past the armed value re-enables firing


@pytest.mark.asyncio
async def test_cooldown_unwedges_on_pipeline_reset() -> None:
    for seed in range(12):
        rng = random.Random(seed)
        cfg = _config(min_shed=5000, max_blocks=0)
        mw = MicrocompactMiddleware()
        pipeline = MiddlewarePipeline().add(mw)
        stats0 = rng.randint(10_000, 20_000)
        msgs = [
            LLMMessage(role=Role.SYSTEM, content="sys"),
            _user(rng.randint(700, 1500)),
            _assistant(rng.randint(150, 800)),
            _assistant(2),
        ]
        ctx = _ctx(msgs, cfg, context_tokens=stats0)
        changed, _ = await _run_pass(mw, ctx)
        assert not changed
        assert mw._cooldown_est == stats0

        _grow(ctx, rng.randint(5200, 6000))  # fireable pool, est still <= armed
        changed, _ = await _run_pass(mw, ctx)
        assert not changed

        pipeline.reset(rng.choice([ResetReason.STOP, ResetReason.COMPACT]))
        assert mw._cooldown_est is None
        changed, _ = await _run_pass(mw, ctx)
        assert changed  # same est: the reset alone re-enabled firing


@pytest.mark.asyncio
async def test_dribble_mode_never_gated_by_floor_or_cooldown() -> None:
    for seed in range(15):
        rng = random.Random(seed)
        max_blocks = rng.randint(1, 4)
        cfg = _config(min_shed=10**9, max_blocks=max_blocks)
        micro = cfg.context_shaping.microcompact
        mw = MicrocompactMiddleware()
        ctx = _ctx(_base_messages(), cfg)
        for _ in range(25):
            roll = rng.random()
            if roll < 0.3:
                ctx.messages.append(_assistant(rng.randint(110, 500)))
            elif roll < 0.4:
                ctx.messages.append(_user(rng.randint(100, 700)))
            elif roll < 0.5:
                ctx.stats.context_tokens += rng.randint(0, 900)
            else:
                plan, _, _ = _model_plan(
                    ctx, micro, _local(ctx.messages), micro.target * THRESHOLD
                )
                gate_open = _est(ctx) >= micro.high_watermark * THRESHOLD
                changed, _ = await _run_pass(mw, ctx)
                assert changed == (set(plan) if gate_open else set())
                assert len(changed) <= max_blocks
                assert mw._cooldown_est is None  # dribble never arms

        # Mid-session flip: a cooldown armed in batch mode must not gate the
        # dribble rollback (shapers read live config each pass).
        batch_ctx = ConversationContext(
            messages=ctx.messages,
            stats=ctx.stats,
            config=_config(min_shed=10**9, max_blocks=0),
        )
        _grow(ctx, 600)
        ctx.stats.context_tokens = max(ctx.stats.context_tokens, 2000)
        changed, _ = await _run_pass(mw, batch_ctx)
        assert not changed
        assert mw._cooldown_est is not None  # batch pass under floor: armed
        changed, _ = await _run_pass(mw, ctx)
        assert changed  # same middleware, dribble config: fires anyway


@pytest.mark.asyncio
async def test_reactive_shed_resets_persistent_shaper_cooldowns() -> None:
    # The pipeline shaper arms near overflow-level est; without the reset in
    # _try_reactive_shaping the whole regrowth stays suppressed.
    for seed in range(3):
        rng = random.Random(seed)
        cfg = build_test_vibe_config(
            models=make_test_models(auto_compact_threshold=THRESHOLD),
            context_shaping=ContextShapingConfig(
                snip=SnipConfig(keep_recent_turns=1, min_message_tokens=50),
                cache_prefix_guard_tokens=50,
            ),
        )
        loop = build_test_agent_loop(config=cfg)
        mc = next(
            mw
            for mw in loop.middleware_pipeline.middlewares
            if isinstance(mw, MicrocompactMiddleware)
        )
        n0 = len(loop.messages)
        loop.messages.append(_user(rng.randint(15_000, 25_000)))  # unshedable bulk
        for _ in range(3):
            loop.messages.append(_assistant(rng.randint(4000, 5000)))
        loop.messages.append(_assistant(2))

        changed, _ = await _run_pass(mc, loop._get_context())
        assert {n0 + 1, n0 + 2, n0 + 3} <= changed  # pool-exhausted batch fire
        armed = mc._cooldown_est
        assert armed is not None

        loop.messages.insert(
            len(loop.messages) - 1, _assistant(rng.randint(6000, 8000))
        )
        ctx = loop._get_context()
        assert _est(ctx) <= armed
        changed, _ = await _run_pass(mc, ctx)
        assert not changed  # armed cooldown suppresses the regrowth

        assert await loop._try_reactive_shaping() is True
        assert mc._cooldown_est is None

        loop.messages.insert(
            len(loop.messages) - 1, _assistant(rng.randint(6000, 7000))
        )
        changed, _ = await _run_pass(mc, loop._get_context())
        assert changed  # proactive shaping resumed below the old armed est


@pytest.mark.asyncio
async def test_batch_state_machine_matches_model_over_random_sequences() -> None:
    # Trace equivalence against the spec model on every pass: fire/suppress
    # decision, exact gisted set, exact tokens shed, exact cooldown value.
    for seed in range(25):
        rng = random.Random(seed)
        floor = rng.choice([0, 300, 800])
        cfg = _config(min_shed=floor, max_blocks=0)
        micro = cfg.context_shaping.microcompact
        mw = MicrocompactMiddleware()
        ctx = _ctx(_base_messages(), cfg)
        cooldown: int | None = None
        for _ in range(40):
            roll = rng.random()
            if roll < 0.22:
                ctx.messages.append(_assistant(rng.randint(120, 700)))
            elif roll < 0.32:
                ctx.messages.append(_assistant(rng.randint(5, 90)))
            elif roll < 0.42:
                ctx.messages.append(_user(rng.randint(100, 900)))
            elif roll < 0.5:
                ctx.stats.context_tokens += rng.randint(50, 1500)
            elif roll < 0.56:
                ctx.stats.context_tokens = max(
                    0, ctx.stats.context_tokens - rng.randint(0, 2000)
                )
            elif roll < 0.62:
                mw.reset(rng.choice([ResetReason.STOP, ResetReason.COMPACT]))
                cooldown = None
                if rng.random() < 0.5:
                    ctx.stats.context_tokens = 0
            else:
                verdict = _model_pass(ctx, cooldown)
                changed, dropped = await _run_pass(mw, ctx)
                assert bool(changed) == verdict.fires
                assert mw._cooldown_est == verdict.cooldown
                if verdict.fires:
                    assert changed == set(verdict.plan)
                    assert dropped == verdict.shed
                    assert verdict.projected >= floor
                    if not verdict.exhausted:
                        slop = (
                            _gisted_tokens(micro.per_message_cap_tokens)
                            - micro.per_message_cap_tokens
                        ) * len(verdict.plan)
                        assert _local(ctx.messages) <= micro.target * THRESHOLD + slop
                cooldown = verdict.cooldown


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "middleware.py:508 _plan_pass projects a gist as shedding body-cap "
        "tokens but _execute_plan (middleware.py:526) lands each block at "
        "cap+~6 (the '<vibe_microcompacted> ' marker); a sufficient-pool "
        "batch pass can stop short and leave est above target"
    ),
)
async def test_batch_fire_with_sufficient_pool_reaches_target() -> None:
    cfg = _config(min_shed=0, max_blocks=0)
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.USER, content="go"),
        _assistant(503),
        _assistant(493),
        LLMMessage(role=Role.ASSISTANT, content="recent"),
    ]
    ctx = _ctx(msgs, cfg)
    assert _local(ctx.messages) == 1000
    mw = MicrocompactMiddleware()

    changed, _ = await _run_pass(mw, ctx)

    # Plan stopped at 1000-403=597 <= 600 with the 493 block to spare (pool
    # sufficient), but the executed shed is only 397: local lands at 603.
    assert changed == {2}
    assert not (ctx.messages[3].content or "").startswith(_MC_OPEN)
    target = cfg.context_shaping.microcompact.target * THRESHOLD
    assert _local(ctx.messages) <= target
