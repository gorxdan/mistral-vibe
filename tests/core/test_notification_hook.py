from __future__ import annotations

from typing import Any

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.hooks.manager import _HANDLERS
from vibe.core.hooks.models import (
    HookSessionContext,
    HookType,
    NotificationInvocation,
    build_invocation,
)
from vibe.core.types import ApprovalResponse


def test_registered_and_built() -> None:
    assert HookType.NOTIFICATION in _HANDLERS
    inv = build_invocation(
        HookType.NOTIFICATION,
        HookSessionContext(session_id="s", transcript_path="t", cwd="/x"),
        notification_type="permission_required",
        message="hi",
        tool_name="bash",
    )
    assert isinstance(inv, NotificationInvocation)
    assert inv.notification_type == "permission_required"
    assert inv.tool_name == "bash"


def test_build_requires_notification_type() -> None:
    with pytest.raises(ValueError, match="notification_type"):
        build_invocation(
            HookType.NOTIFICATION,
            HookSessionContext(session_id="s", transcript_path="t", cwd="/x"),
        )


class _RecMgr:
    def __init__(self) -> None:
        self.notifs: list[NotificationInvocation] = []

    def reset_retry_count(self) -> None:
        pass

    async def run(self, invocation):
        if isinstance(invocation, NotificationInvocation):
            self.notifs.append(invocation)
        return
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_fire_notification_records() -> None:
    loop = build_test_agent_loop()
    mgr = _RecMgr()
    loop._hooks_manager = mgr  # type: ignore[assignment]
    await loop._fire_notification_hooks("question", "pick one", "ask_user_question")
    assert len(mgr.notifs) == 1
    assert mgr.notifs[0].notification_type == "question"
    assert mgr.notifs[0].tool_name == "ask_user_question"


@pytest.mark.asyncio
async def test_ask_approval_fires_permission_notification() -> None:
    loop = build_test_agent_loop()
    mgr = _RecMgr()
    loop._hooks_manager = mgr  # type: ignore[assignment]

    async def approve(*a: Any, **k: Any):
        return ApprovalResponse.YES, None, None

    class _Args(BaseModel):
        pass

    loop.approval_callback = approve  # type: ignore[assignment]
    await loop._ask_approval("bash", _Args(), "call-1", [])
    assert mgr.notifs and mgr.notifs[0].notification_type == "permission_required"
    assert mgr.notifs[0].tool_name == "bash"


async def _drive_tool(loop, tool_name: str) -> list[tuple[str, str | None]]:
    """Run _execute_tool_call with the pipeline stubbed, recording any
    notification fired.
    """
    from vibe.core.agent_loop import ToolDecision, ToolExecutionResponse
    from vibe.core.agent_loop_hooks import _BeforeToolResolution
    from vibe.core.llm.models import ResolvedToolCall
    from vibe.core.tools.base import ToolPermission
    from vibe.core.tools.builtins.ask_user_question import AskUserQuestion

    class _A(BaseModel):
        pass

    rtc = ResolvedToolCall(
        tool_name=tool_name,
        tool_class=AskUserQuestion,
        validated_args=_A(),
        call_id="c1",
    )
    fired: list[tuple[str, str | None]] = []

    async def rec(nt: str, msg: str, tn: str | None = None) -> None:
        fired.append((nt, tn))

    async def fake_pipeline(tc, ti, *, span):
        return [], _BeforeToolResolution(tc, ti, None)

    async def fake_should(*a: Any, **k: Any) -> ToolDecision:
        return ToolDecision(
            verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
        )

    async def fake_invoke(*a: Any, **k: Any):
        return
        yield  # pragma: no cover

    loop.tool_manager.get = lambda n: object()  # type: ignore[method-assign]
    loop._fire_notification_hooks = rec  # type: ignore[method-assign]
    loop._run_before_tool_pipeline = fake_pipeline  # type: ignore[method-assign]
    loop._should_execute_tool = fake_should  # type: ignore[method-assign]
    loop._invoke_tool = fake_invoke  # type: ignore[method-assign]

    async for _ in loop._execute_tool_call(None, rtc):  # type: ignore[arg-type]
        pass
    return fired


@pytest.mark.asyncio
async def test_ask_user_question_fires_question_notification() -> None:
    loop = build_test_agent_loop()
    fired = await _drive_tool(loop, "ask_user_question")
    assert ("question", "ask_user_question") in fired


@pytest.mark.asyncio
async def test_other_tool_does_not_fire_question_notification() -> None:
    loop = build_test_agent_loop()
    fired = await _drive_tool(loop, "bash")
    assert all(nt != "question" for nt, _ in fired)


@pytest.mark.asyncio
async def test_cancel_during_question_notification_does_not_mark_tool_started() -> None:
    """Regression: the question notification fires BEFORE tool_started=True, so a
    cancellation there must NOT finalize as a started-then-cancelled tool (which
    would spuriously fire after-tool hooks).
    """
    import asyncio

    from vibe.core.agent_loop import ToolDecision, ToolExecutionResponse
    from vibe.core.agent_loop_hooks import _BeforeToolResolution
    from vibe.core.llm.models import ResolvedToolCall
    from vibe.core.tools.base import ToolPermission
    from vibe.core.tools.builtins.ask_user_question import AskUserQuestion

    class _A(BaseModel):
        pass

    loop = build_test_agent_loop()
    rtc = ResolvedToolCall(
        tool_name="ask_user_question",
        tool_class=AskUserQuestion,
        validated_args=_A(),
        call_id="c1",
    )
    invoked = {"called": False}
    captured: dict[str, Any] = {}

    async def cancel_notif(*a: Any, **k: Any) -> None:
        raise asyncio.CancelledError

    async def fake_pipeline(tc, ti, *, span):
        return [], _BeforeToolResolution(tc, ti, None)

    async def fake_should(*a: Any, **k: Any) -> ToolDecision:
        return ToolDecision(
            verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
        )

    async def fake_invoke(*a: Any, **k: Any):
        invoked["called"] = True
        return
        yield  # pragma: no cover

    async def fake_finalize(tc, ti, dec, msg, *, span, tool_started, **k):
        captured["tool_started"] = tool_started
        return
        yield  # pragma: no cover

    loop.tool_manager.get = lambda n: object()  # type: ignore[method-assign]
    loop._fire_notification_hooks = cancel_notif  # type: ignore[method-assign]
    loop._run_before_tool_pipeline = fake_pipeline  # type: ignore[method-assign]
    loop._should_execute_tool = fake_should  # type: ignore[method-assign]
    loop._invoke_tool = fake_invoke  # type: ignore[method-assign]
    loop._finalize_cancelled_tool = fake_finalize  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        async for _ in loop._execute_tool_call(None, rtc):  # type: ignore[arg-type]
            pass

    assert invoked["called"] is False, "tool never invoked"
    assert captured.get("tool_started") is False, "not finalized as started"
