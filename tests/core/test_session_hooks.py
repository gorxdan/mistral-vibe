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
async def test_session_start_dispatch_fires() -> None:
    loop = build_test_agent_loop()
    mgr = _RecMgr()
    loop._hooks_manager = mgr  # type: ignore[assignment]
    injected, events = await loop._dispatch_session_start_hooks("startup")
    assert injected == [] and events == []  # recorder yields nothing
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


# --------------------------------------------------------------------------- #
# SessionStart additional_context injection                                    #
# --------------------------------------------------------------------------- #


def test_session_start_uses_injecting_handler() -> None:
    from vibe.core.hooks._session_start import SessionStartHandler

    assert isinstance(_HANDLERS[HookType.SESSION_START], SessionStartHandler)


def test_session_start_handler_allow_injects_additional_context() -> None:
    from vibe.core.hooks._handler import HookRetryState
    from vibe.core.hooks._session_start import SessionStartHandler
    from vibe.core.hooks.models import (
        HookConfig,
        HookSpecificOutput,
        HookStructuredResponse,
        HookUserMessage,
    )

    hook = HookConfig(name="h", type=HookType.SESSION_START, command="echo")
    inv = SessionStartInvocation(
        session_id="s", transcript_path="t", cwd="/x", source="startup"
    )
    action = SessionStartHandler().on_structured(
        hook,
        inv,
        HookStructuredResponse(
            decision="allow",
            hook_specific_output=HookSpecificOutput(additional_context="preamble"),
        ),
        HookRetryState(),
    )
    injects = [e for e in action.events if isinstance(e, HookUserMessage)]
    assert injects and injects[0].content == "preamble"
    assert action.should_break is False


class _InjectMgr:
    def __init__(self, content: str) -> None:
        self._content = content

    def reset_retry_count(self) -> None:
        pass

    async def run(self, invocation):
        from vibe.core.hooks.models import HookUserMessage

        if isinstance(invocation, SessionStartInvocation):
            yield HookUserMessage(content=self._content)
        return


@pytest.mark.asyncio
async def test_dispatch_session_start_returns_injected() -> None:
    loop = build_test_agent_loop()
    loop._hooks_manager = _InjectMgr("hello-preamble")  # type: ignore[assignment]
    injected, _ = await loop._dispatch_session_start_hooks("startup")
    assert injected == ["hello-preamble"]


class _ActStopLoop(Exception):
    pass


@pytest.mark.asyncio
async def test_act_appends_session_start_context_before_user_prompt() -> None:
    loop = build_test_agent_loop()
    loop._hooks_manager = _InjectMgr("SESSION_PREAMBLE")  # type: ignore[assignment]

    async def fake_turn():
        raise _ActStopLoop
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]
    with pytest.raises(_ActStopLoop):
        async for _ in loop.act("the user prompt"):
            pass

    contents = [m.content or "" for m in loop.messages]
    preamble_idx = next(i for i, c in enumerate(contents) if "SESSION_PREAMBLE" in c)
    user_idx = next(i for i, c in enumerate(contents) if "the user prompt" in c)
    assert preamble_idx < user_idx, "session preamble precedes the user prompt"
