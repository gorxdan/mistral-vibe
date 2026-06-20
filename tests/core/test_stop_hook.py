from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.hooks.manager import _HANDLERS
from vibe.core.hooks.models import (
    HookSessionContext,
    HookType,
    HookUserMessage,
    StopInvocation,
    build_invocation,
)


def test_stop_registered_and_built() -> None:
    assert HookType.STOP in _HANDLERS
    inv = build_invocation(
        HookType.STOP,
        HookSessionContext(session_id="s", transcript_path="t", cwd="/x"),
        stop_hook_active=True,
    )
    assert isinstance(inv, StopInvocation)
    assert inv.stop_hook_active is True


class _FakeManager:
    def __init__(self, *events) -> None:
        self._events = events

    def reset_retry_count(self) -> None:
        pass

    async def run(self, invocation):
        for e in self._events:
            yield e


@pytest.mark.asyncio
async def test_dispatch_stop_returns_continuation() -> None:
    loop = build_test_agent_loop()
    loop._hooks_manager = _FakeManager(HookUserMessage(content="keep going"))  # type: ignore[assignment]
    cont, _ = await loop._dispatch_stop_hooks(False)
    assert cont is not None
    assert cont.content == "keep going"
    assert cont.injected is True


class _StopOnceManager:
    """Denies the FIRST stop (injecting a continuation), allows after."""

    def __init__(self) -> None:
        self.stop_calls = 0
        self.active_seen: list[bool] = []

    def reset_retry_count(self) -> None:
        pass

    async def run(self, invocation):
        if isinstance(invocation, StopInvocation):
            self.active_seen.append(invocation.stop_hook_active)
            self.stop_calls += 1
            if self.stop_calls == 1:
                yield HookUserMessage(content="keep going")
        return
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_stop_deny_continues_then_ends() -> None:
    loop = build_test_agent_loop()
    mgr = _StopOnceManager()
    loop._hooks_manager = mgr  # type: ignore[assignment]
    turns = {"n": 0}

    async def fake_turn():
        turns["n"] += 1
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]

    _ = [e async for e in loop._conversation_loop("hi")]
    # Turn 1 ends → stop denies → continuation → turn 2 → stop allows → end.
    assert turns["n"] == 2
    # The 2nd Stop saw stop_hook_active=True (guards runaway continues).
    assert mgr.active_seen == [False, True]
    assert any("keep going" in (m.content or "") for m in loop.messages)
