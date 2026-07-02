from __future__ import annotations

import asyncio
import time

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.resource_monitor import ResourceMonitor, _TreeWalk

_SLOW_WALK_S = 0.3
_MAX_GAP_S = 0.15


@pytest.mark.asyncio
async def test_act_does_not_stall_loop_on_resource_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ResourceMonitor._owner_pids.clear()
    backend = FakeBackend(mock_llm_chunk(content="hi"))
    agent = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=[]), backend=backend
    )
    monitor = agent.resource_monitor
    if not monitor.available:
        pytest.skip("psutil unavailable")
    real_read = monitor._read_tree

    def slow_read() -> _TreeWalk | None:
        time.sleep(_SLOW_WALK_S)
        return real_read()

    monkeypatch.setattr(monitor, "_read_tree", slow_read)

    gaps: list[float] = []
    stop = asyncio.Event()

    async def probe() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.001)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    probe_task = asyncio.create_task(probe())
    await asyncio.sleep(0.01)  # probe must be inside its timer loop before act()
    try:
        events = [event async for event in agent.act("go")]
    finally:
        stop.set()
        await probe_task
        await agent.aclose()
    assert events
    assert gaps
    assert max(gaps) < _MAX_GAP_S, (
        f"event loop stalled {max(gaps) * 1000:.0f}ms during turn sampling"
    )
