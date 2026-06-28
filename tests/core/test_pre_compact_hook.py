from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.hooks.manager import _HANDLERS
from vibe.core.hooks.models import (
    HookSessionContext,
    HookType,
    PreCompactInvocation,
    build_invocation,
)


def _ctx() -> HookSessionContext:
    return HookSessionContext(session_id="s", transcript_path="t", cwd="/x")


def test_pre_compact_registered() -> None:
    assert HookType.PRE_COMPACT in _HANDLERS


def test_build_invocation_pre_compact() -> None:
    inv = build_invocation(
        HookType.PRE_COMPACT,
        _ctx(),
        trigger="auto",
        current_context_tokens=100,
        threshold=200,
    )
    assert isinstance(inv, PreCompactInvocation)
    assert inv.trigger == "auto"
    assert inv.current_context_tokens == 100
    assert inv.threshold == 200
    assert inv.hook_event_name == HookType.PRE_COMPACT


def test_build_invocation_requires_trigger() -> None:
    with pytest.raises(ValueError, match="trigger"):
        build_invocation(HookType.PRE_COMPACT, _ctx())


@pytest.mark.asyncio
async def test_runner_noop_without_hooks_manager() -> None:
    loop = build_test_agent_loop()
    loop._hooks_manager = None
    events = [e async for e in loop._run_pre_compact_hooks("auto", 1, 2)]
    assert events == []


@pytest.mark.asyncio
async def test_runner_emits_hook_events_when_manager_present() -> None:
    from vibe.core.hooks.models import HookStartEvent

    loop = build_test_agent_loop()
    seen: dict[str, object] = {}

    class _FakeManager:
        async def run(self, invocation):
            seen["invocation"] = invocation
            yield HookStartEvent(hook_name="h")

    loop._hooks_manager = _FakeManager()  # type: ignore[assignment]
    events = [e async for e in loop._run_pre_compact_hooks("emergency", 5, 9)]
    assert len(events) == 1
    inv = seen["invocation"]
    assert isinstance(inv, PreCompactInvocation)
    assert inv.trigger == "emergency"
