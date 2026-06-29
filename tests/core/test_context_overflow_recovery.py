from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from vibe.core.types import (
    BaseEvent,
    ContextTooLongError,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
)


@pytest.mark.asyncio
async def test_context_overflow_triggers_compaction_and_retries() -> None:
    loop = build_test_agent_loop()
    calls = {"turn": 0, "compact": 0}

    async def fake_turn() -> AsyncGenerator[BaseEvent, None]:
        calls["turn"] += 1
        if calls["turn"] == 1:
            raise ContextTooLongError("prov", "model")
        return
        yield  # pragma: no cover — marks this an async generator

    async def fake_compaction(
        old_tokens: int, threshold: int, **_kwargs: object
    ) -> AsyncGenerator[BaseEvent, None]:
        calls["compact"] += 1
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]
    loop._run_compaction = fake_compaction  # type: ignore[method-assign]

    events = [e async for e in loop._conversation_loop("hello")]

    assert calls["turn"] == 2, "turn retried once after overflow"
    assert calls["compact"] == 1, "emergency compaction ran"
    assert events  # at least the UserMessageEvent


@pytest.mark.asyncio
async def test_context_overflow_twice_surfaces_error() -> None:
    loop = build_test_agent_loop()
    calls = {"compact": 0}

    async def always_overflow() -> AsyncGenerator[BaseEvent, None]:
        raise ContextTooLongError("prov", "model")
        yield  # pragma: no cover

    async def fake_compaction(
        old_tokens: int, threshold: int, **_kwargs: object
    ) -> AsyncGenerator[BaseEvent, None]:
        calls["compact"] += 1
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = always_overflow  # type: ignore[method-assign]
    loop._run_compaction = fake_compaction  # type: ignore[method-assign]

    with pytest.raises(ContextTooLongError):
        _ = [e async for e in loop._conversation_loop("hello")]

    assert calls["compact"] == 1, "compacts once, then re-raises on second overflow"


@pytest.mark.asyncio
async def test_try_reactive_shaping_compresses_old_history() -> None:
    """Reactive shaping compresses shapeable old messages and returns True."""
    from vibe.core.config import ContextShapingConfig
    from vibe.core.config._settings import SnipConfig

    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=500),
        context_shaping=ContextShapingConfig(
            snip=SnipConfig(keep_recent_turns=1, min_message_tokens=50),
            cache_prefix_guard_tokens=50,
        ),
    )
    loop = build_test_agent_loop(config=cfg)
    # Enough large messages that some fall outside the protected prefix/suffix.
    for _ in range(4):
        loop.messages.append(LLMMessage(role=Role.ASSISTANT, content="x" * 4000))

    result = await loop._try_reactive_shaping()

    assert result is True
    assert any((m.content or "").startswith("<vibe") for m in loop.messages)


@pytest.mark.asyncio
async def test_try_reactive_shaping_returns_false_when_nothing_to_shape() -> None:
    """When there is nothing to compress, shaping returns False so compaction
    fires as the fallback.
    """
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    loop = build_test_agent_loop(config=cfg)

    result = await loop._try_reactive_shaping()

    assert result is False


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Emergency compaction runs the summary _chat call on the full, "
        "already-over-window history (AgentLoop.compact). With no progressive "
        "trim/chunk fallback the summarizer is handed the same payload that just "
        "overflowed; it can itself raise ContextTooLongError, which escapes the "
        "conversation loop as a hard, user-facing error. Remove this marker once "
        "compaction reduces the summarizer input (or recovers from its overflow)."
    ),
)
async def test_emergency_compaction_recovers_when_summarizer_overflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibe.core.config import ContextShapingConfig
    from vibe.core.config._settings import MicrocompactConfig, SnipConfig

    simulated_window = 50_000

    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=500),
        # Strict mode surfaces the summarizer overflow instead of silently
        # swallowing it into the extractive fallback, exposing the hard crash.
        raise_on_compaction_failure=True,
        # Neutralize the cheap before-turn shapers so the scenario is isolated to
        # the emergency-compaction summary call; reactive shaping is forced off
        # below to model overflow trapped in the protected prefix.
        context_shaping=ContextShapingConfig(
            snip=SnipConfig(enabled=False),
            microcompact=MicrocompactConfig(enabled=False),
        ),
    )
    loop = build_test_agent_loop(config=cfg)
    loop.messages.append(
        LLMMessage(role=Role.USER, content="g" * 240_000, injected=True)
    )
    loop.messages.append(LLMMessage(role=Role.USER, content="u" * 240_000))

    def history_tokens() -> int:
        from vibe.core.utils.tokens import approx_token_count

        return sum(approx_token_count(m.content or "") for m in loop.messages)

    state = {"turns": 0, "summarizer_calls": 0}

    async def fake_turn() -> AsyncGenerator[BaseEvent, None]:
        state["turns"] += 1
        if state["turns"] == 1:
            raise ContextTooLongError("prov", "model")
        return
        yield  # pragma: no cover

    async def no_shaping() -> bool:
        return False

    async def fake_chat(**_kwargs: object) -> LLMChunk:
        state["summarizer_calls"] += 1
        if history_tokens() > simulated_window:
            raise ContextTooLongError("prov", "model")
        return LLMChunk(
            message=LLMMessage(role=Role.ASSISTANT, content="summary"),
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5),
        )

    monkeypatch.setattr(loop, "_perform_llm_turn", fake_turn)
    monkeypatch.setattr(loop, "_try_reactive_shaping", no_shaping)
    monkeypatch.setattr(loop, "_chat", fake_chat)

    _ = [e async for e in loop._conversation_loop("hello")]

    assert state["summarizer_calls"] >= 1, (
        "emergency compaction must run the summarizer"
    )
    assert state["turns"] == 2, "turn retried after the loop recovered from overflow"
