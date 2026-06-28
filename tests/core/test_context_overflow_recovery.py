from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from vibe.core.types import BaseEvent, ContextTooLongError, LLMMessage, Role


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
