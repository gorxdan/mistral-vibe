from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.hooks.manager import _HANDLERS
from vibe.core.hooks.models import (
    HookSessionContext,
    HookType,
    SessionEndInvocation,
    SessionStartInvocation,
    build_invocation,
)


def test_registered_and_built() -> None:
    assert HookType.SESSION_START in _HANDLERS
    assert HookType.SESSION_END in _HANDLERS
    c = HookSessionContext(session_id="s", transcript_path="t", cwd="/x")
    start = build_invocation(HookType.SESSION_START, c, source="resume")
    end = build_invocation(HookType.SESSION_END, c, reason="clear")
    assert isinstance(start, SessionStartInvocation) and start.source == "resume"
    assert isinstance(end, SessionEndInvocation) and end.reason == "clear"


class _RecMgr:
    def __init__(self) -> None:
        self.types: list[str] = []
        self.reasons: list[str] = []

    def reset_retry_count(self) -> None:
        pass

    async def run(self, invocation):
        self.types.append(type(invocation).__name__)
        if isinstance(invocation, SessionEndInvocation):
            self.reasons.append(invocation.reason)
        return
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_session_start_runner_fires() -> None:
    loop = build_test_agent_loop()
    mgr = _RecMgr()
    loop._hooks_manager = mgr  # type: ignore[assignment]
    events = [e async for e in loop._run_session_start_hooks("startup")]
    assert events == []  # recorder yields no events
    assert mgr.types == ["SessionStartInvocation"]


@pytest.mark.asyncio
async def test_session_end_fire_is_best_effort() -> None:
    loop = build_test_agent_loop()
    mgr = _RecMgr()
    loop._hooks_manager = mgr  # type: ignore[assignment]
    await loop._fire_session_end_hooks("exit")  # must not raise
    assert mgr.reasons == ["exit"]


@pytest.mark.asyncio
async def test_reset_session_clear_fires_lifecycle_compact_does_not() -> None:
    loop = build_test_agent_loop()
    fired: list[str] = []

    async def rec(reason: str) -> None:
        fired.append(reason)

    loop._fire_session_end_hooks = rec  # type: ignore[method-assign]
    loop._session_started = True

    # /clear path: lifecycle_reason set → SessionEnd fires + flag reset.
    await loop._reset_session(keep_parent=False, lifecycle_reason="clear")
    assert fired == ["clear"]
    assert loop._session_started is False

    # compaction's internal reset: no lifecycle_reason → no session hook.
    loop._session_started = True
    await loop._reset_session()
    assert fired == ["clear"]  # unchanged
    assert loop._session_started is True
