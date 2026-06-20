from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.types import BaseEvent, ContextTooLongError


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
        old_tokens: int, threshold: int
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
        old_tokens: int, threshold: int
    ) -> AsyncGenerator[BaseEvent, None]:
        calls["compact"] += 1
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = always_overflow  # type: ignore[method-assign]
    loop._run_compaction = fake_compaction  # type: ignore[method-assign]

    with pytest.raises(ContextTooLongError):
        _ = [e async for e in loop._conversation_loop("hello")]

    assert calls["compact"] == 1, "compacts once, then re-raises on second overflow"
