from __future__ import annotations

import sys
from typing import Any

import pytest

from vibe.core.loop import LoopManager
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.builtins.schedule import (
    Schedule,
    ScheduleArgs,
    ScheduleConfig,
    ScheduleResult,
)
from vibe.core.tools.permissions import ToolPermission


class _FakeLogger:
    session_metadata = None

    async def persist_loops(self) -> None:
        pass


def _manager() -> LoopManager:
    return LoopManager(_FakeLogger())  # type: ignore[arg-type]


def _tool() -> Schedule:
    return Schedule(config_getter=lambda: ScheduleConfig(), state=BaseToolState())


async def _run(tool: Schedule, args: ScheduleArgs, scheduler: Any) -> ScheduleResult:
    ctx = InvokeContext(tool_call_id="c1", scheduler=scheduler)
    out: ScheduleResult | None = None
    async for ev in tool.run(args, ctx):
        assert isinstance(ev, ScheduleResult)
        out = ev
    assert out is not None
    return out


# --------------------------------------------------------------------------- #
# schedule tool                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_recurring_and_once() -> None:
    mgr = _manager()
    tool = _tool()
    r = await _run(
        tool, ScheduleArgs(action="create", interval="5m", prompt="check CI"), mgr
    )
    assert r is not None and r.id and "every" in r.message
    r2 = await _run(
        tool,
        ScheduleArgs(action="create", interval="5m", prompt="ping", recurring=False),
        mgr,
    )
    assert "once in" in r2.message
    loops = mgr.loops
    assert len(loops) == 2
    assert {lp.recurring for lp in loops} == {True, False}


@pytest.mark.asyncio
async def test_list_and_cancel() -> None:
    mgr = _manager()
    tool = _tool()
    await _run(tool, ScheduleArgs(action="create", interval="1m", prompt="a"), mgr)
    lp_id = mgr.loops[0].id
    listed = await _run(tool, ScheduleArgs(action="list"), mgr)
    assert listed.scheduled and lp_id in listed.scheduled[0]
    cancelled = await _run(tool, ScheduleArgs(action="cancel", target=lp_id), mgr)
    assert "cancelled 1" in cancelled.message
    assert mgr.loops == []


@pytest.mark.asyncio
async def test_no_scheduler_degrades_gracefully() -> None:
    r = await _run(_tool(), ScheduleArgs(action="list"), None)
    assert "unavailable" in r.message.lower() and "sleep" in r.message.lower()


@pytest.mark.asyncio
async def test_below_min_interval_is_a_tool_error() -> None:
    with pytest.raises(ToolError):
        await _run(
            _tool(),
            ScheduleArgs(action="create", interval="5s", prompt="x"),
            _manager(),
        )


@pytest.mark.asyncio
async def test_cancel_requires_target() -> None:
    with pytest.raises(ToolError):
        await _run(_tool(), ScheduleArgs(action="cancel"), _manager())


# --------------------------------------------------------------------------- #
# one-shot loop fires once then drops                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_one_shot_loop_drops_after_firing() -> None:
    mgr = _manager()
    lp = await mgr.add_loop(30, "once", recurring=False)
    fired = await mgr.pop_due(now=lp.next_fire_at + 1)
    assert fired is not None and fired.id == lp.id
    assert mgr.loops == []  # one-shot dropped


@pytest.mark.asyncio
async def test_recurring_loop_re_arms() -> None:
    mgr = _manager()
    lp = await mgr.add_loop(30, "repeat", recurring=True)
    orig_fire_at = lp.next_fire_at
    fired = await mgr.pop_due(now=orig_fire_at + 1)
    assert fired is not None
    assert len(mgr.loops) == 1  # still armed
    assert mgr.loops[0].next_fire_at > orig_fire_at  # re-armed forward


# --------------------------------------------------------------------------- #
# bash blocks long blocking sleep, points to the scheduler                    #
# --------------------------------------------------------------------------- #

pytestmark_posix = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="bash policy is POSIX-only"
)


def _bash_permission(cmd: str) -> ToolPermission | None:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    ctx = tool.resolve_permission(BashArgs(command=cmd))
    return ctx.permission if ctx is not None else None


@pytestmark_posix
def test_long_sleep_is_denied_with_scheduler_pointer() -> None:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    ctx = tool.resolve_permission(BashArgs(command="sleep 300"))
    assert ctx is not None and ctx.permission == ToolPermission.NEVER
    assert "schedule" in (ctx.reason or "").lower()


@pytestmark_posix
@pytest.mark.parametrize("cmd", ["sleep 10", "sleep 5m", "sleep 60", "sleep 1h"])
def test_blocking_sleeps_denied(cmd: str) -> None:
    assert _bash_permission(cmd) == ToolPermission.NEVER


@pytestmark_posix
@pytest.mark.parametrize("cmd", ["sleep 2", "sleep 1", "git diff"])
def test_short_sleep_and_normal_commands_not_denied(cmd: str) -> None:
    assert _bash_permission(cmd) != ToolPermission.NEVER


@pytestmark_posix
def test_sleep_smuggled_in_compound_is_denied() -> None:
    assert _bash_permission("echo hi && sleep 300") == ToolPermission.NEVER
