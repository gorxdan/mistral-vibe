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
        return ApprovalResponse.YES, None

    loop.approval_callback = approve  # type: ignore[assignment]
    await loop._ask_approval("bash", BaseModel(), "call-1", [])
    assert mgr.notifs and mgr.notifs[0].notification_type == "permission_required"
    assert mgr.notifs[0].tool_name == "bash"
