from __future__ import annotations

import asyncio

import pytest

from vibe.core.loop import LoopManager
from vibe.core.schedule_driver import ScheduleDriver
from vibe.core.types import ScheduledLoop


class _FakeLogger:
    session_metadata = None

    async def persist_loops(self) -> None:
        pass


def _mgr() -> LoopManager:
    return LoopManager(_FakeLogger())  # type: ignore[arg-type]


async def _arm_due(mgr: LoopManager, *, recurring: bool) -> ScheduledLoop:
    lp = await mgr.add_loop(30, "ping", recurring=recurring)
    lp.next_fire_at = 0.0  # make it immediately due so tests don't wait
    return lp


def _collector():
    fired: list[ScheduledLoop] = []

    async def fire(due: ScheduledLoop) -> None:
        fired.append(due)

    return fired, fire


@pytest.mark.asyncio
async def test_run_until_idle_fires_one_shot_then_exits() -> None:
    mgr = _mgr()
    await _arm_due(mgr, recurring=False)
    fired, fire = _collector()
    driver = ScheduleDriver(mgr, can_fire=lambda: True, fire=fire)
    await asyncio.wait_for(driver.run_until_idle(), timeout=3)
    assert [f.prompt for f in fired] == ["ping"]
    assert mgr.loops == []  # one-shot drained → run_until_idle returns


@pytest.mark.asyncio
async def test_run_until_idle_caps_recurring_at_deadline() -> None:
    mgr = _mgr()
    await _arm_due(mgr, recurring=True)
    fired, fire = _collector()
    driver = ScheduleDriver(mgr, can_fire=lambda: True, fire=fire)
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(driver.run_until_idle(deadline=loop.time() + 0.3), timeout=3)
    assert len(fired) >= 1  # fired at least once
    assert len(mgr.loops) == 1  # recurring still armed, not drained


@pytest.mark.asyncio
async def test_can_fire_gate_defers() -> None:
    mgr = _mgr()
    await _arm_due(mgr, recurring=False)
    fired, fire = _collector()
    driver = ScheduleDriver(mgr, can_fire=lambda: False, fire=fire)
    loop = asyncio.get_running_loop()
    await asyncio.wait_for(driver.run_until_idle(deadline=loop.time() + 0.3), timeout=3)
    assert fired == []  # gate closed → never fired
    assert len(mgr.loops) == 1  # still pending


@pytest.mark.asyncio
async def test_background_start_fires_and_stop_cancels() -> None:
    mgr = _mgr()
    await _arm_due(mgr, recurring=False)
    fired, fire = _collector()
    driver = ScheduleDriver(mgr, can_fire=lambda: True, fire=fire)
    driver.start()
    for _ in range(40):  # poll up to ~2s for the fire
        if fired:
            break
        await asyncio.sleep(0.05)
    await driver.stop()
    assert [f.prompt for f in fired] == ["ping"]
