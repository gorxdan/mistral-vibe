"""Reproducible performance benchmark for the agent loop.

Drives ``AgentLoop.act()`` against a deterministic ``FakeBackend`` (no network)
so the perf instruments produce comparable, attributable numbers across runs.
The instruments are wired into ``act()`` and fire only when their env vars are
set, so this file is a fast no-op in normal CI and an evaluation harness under:

    VIBE_TRACE_LOOP=0.02 VIBE_PROFILE=1 \\
        uv run pytest tests/agent_loop/test_perf_benchmark.py -s

``VIBE_TRACE_LOOP`` is in SECONDS: ``0.02`` arms the tracer at 20ms. The armed
threshold is echoed into the per-PID perf log, so an empty blocker table means
"no callback blocked past the threshold" — not that the tracer was off.

Two scenarios exercise the async hot paths most likely to monopolize the shared
event-loop thread (the "single-core heavy" hypothesis):

- ``test_perf_large_response`` — one chunk with a large content payload. Stresses
  the message-parse / append / stats path on the loop thread.
- ``test_perf_parallel_fanout`` — one chunk emitting N read-only ``glob`` calls,
  then a final reply. Stresses ``_run_tools_concurrently`` (queue + gather) and
  sibling-coroutine scheduling on the single loop thread.
"""

from __future__ import annotations

import time

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.types import FunctionCall, ToolCall

# Large payload size for the response scenario. Big enough to make any per-byte
# or per-message-list work visible in the tracer; small enough to keep the test
# sub-second. 200 KB of text.
_LARGE_CONTENT = "x" * 200_000
# Fan-out width. Each glob is read-only and runs concurrently; 24 siblings is
# enough to expose scheduling/queue contention without being a unit test.
_FANOUT = 24


def _report_blockers() -> None:
    # Module-level dict, same process: read directly so output is deterministic
    # regardless of how pytest captures the logger.
    from vibe.core import loop_tracer

    if not loop_tracer._INSTALLED or not loop_tracer._BLOCKERS:
        return
    print("\n[bench] === loop-block top 10 (by total ms) ===")
    for (label, where), (count, ms) in sorted(
        loop_tracer._BLOCKERS.items(), key=lambda kv: kv[1][1], reverse=True
    )[:10]:
        print(
            f"[bench]   {ms:7.1f}ms / {int(count):3d} calls  "
            f"avg={ms / count:5.1f}ms  {label}  @ {where}"
        )


@pytest.mark.asyncio
async def test_perf_large_response() -> None:
    backend = FakeBackend(mock_llm_chunk(content=_LARGE_CONTENT))
    agent = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=[]), backend=backend
    )
    t0 = time.perf_counter()
    events = [event async for event in agent.act("go")]
    elapsed = time.perf_counter() - t0
    assert events, "large-response turn produced no events"
    print(f"\n[bench] large_response: {len(events)} events in {elapsed:.3f}s")
    _report_blockers()


@pytest.mark.asyncio
async def test_perf_parallel_fanout() -> None:
    tool_calls = [
        ToolCall(
            id=f"call_glob_{i}",
            index=i,
            function=FunctionCall(
                name="glob", arguments='{"pattern": "ZZZ_NOMATCH_*", "path": "."}'
            ),
        )
        for i in range(_FANOUT)
    ]
    # Stream 1: the N parallel read-only glob calls. Stream 2: the final reply.
    backend = FakeBackend([
        mock_llm_chunk(content="", tool_calls=tool_calls),
        mock_llm_chunk("done"),
    ])
    agent = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["glob"]), backend=backend
    )
    t0 = time.perf_counter()
    events = [event async for event in agent.act("go")]
    elapsed = time.perf_counter() - t0
    assert events, "fanout turn produced no events"
    print(f"\n[bench] parallel_fanout: {len(events)} events in {elapsed:.3f}s")
    _report_blockers()
