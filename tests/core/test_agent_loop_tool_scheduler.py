from __future__ import annotations

import asyncio

from pydantic import BaseModel
import pytest

from vibe.core.agent_loop_tool_scheduler import (
    EventSink,
    build_tool_call_waves,
    stream_tool_call_waves,
)
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.tools.builtins.read import Read
from vibe.core.tools.builtins.write_file import WriteFile


class _Args(BaseModel):
    pass


def _call(name: str, *, read_only: bool) -> ResolvedToolCall:
    tool_class = Read if read_only else WriteFile
    return ResolvedToolCall(
        tool_name=name, tool_class=tool_class, validated_args=_Args(), call_id=name
    )


def test_build_tool_call_waves_preserves_dependencies() -> None:
    calls = [
        _call("read1", read_only=True),
        _call("read2", read_only=True),
        _call("write1", read_only=False),
        _call("write2", read_only=False),
        _call("read3", read_only=True),
        _call("write3", read_only=False),
    ]
    classified: list[str] = []

    def concurrent_safe(call: ResolvedToolCall) -> bool:
        classified.append(call.tool_name)
        return call.tool_class.read_only

    waves = build_tool_call_waves(calls, concurrent_safe=concurrent_safe)

    assert [[call.tool_name for call in wave.calls] for wave in waves] == [
        ["read1", "read2"],
        ["write1"],
        ["write2"],
        ["read3"],
        ["write3"],
    ]
    assert [wave.concurrent_safe for wave in waves] == [True, False, False, True, False]
    assert classified == [call.tool_name for call in calls]
    assert build_tool_call_waves([], concurrent_safe=concurrent_safe) == ()


@pytest.mark.asyncio
async def test_executor_failure_aborts_later_waves() -> None:
    calls = [
        _call("read1", read_only=True),
        _call("write1", read_only=False),
        _call("read2", read_only=True),
    ]
    started: list[str] = []

    async def execute(
        call: ResolvedToolCall, scheduled_read_only: bool, queue: EventSink[str]
    ) -> None:
        assert scheduled_read_only is call.tool_class.read_only
        started.append(call.tool_name)
        if call.tool_name == "write1":
            raise RuntimeError("boom")
        await queue.put(call.tool_name)

    results: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for event in stream_tool_call_waves(
            calls,
            concurrent_safe=lambda call: call.tool_class.read_only,
            execute=execute,
        ):
            results.append(event)

    assert started == ["read1", "write1"]
    assert results == ["read1"]


@pytest.mark.asyncio
async def test_concurrent_failure_drains_sibling_events_before_abort() -> None:
    calls = [
        _call("read1", read_only=True),
        _call("read2", read_only=True),
        _call("write1", read_only=False),
    ]
    started: set[str] = set()
    finalized: set[str] = set()
    read2_started = asyncio.Event()

    async def execute(
        call: ResolvedToolCall, scheduled_read_only: bool, queue: EventSink[str]
    ) -> None:
        assert scheduled_read_only is call.tool_class.read_only
        started.add(call.tool_name)
        if call.tool_name == "read1":
            await read2_started.wait()
            await queue.put("read1-event")
            raise RuntimeError("boom")
        if call.tool_name == "read2":
            read2_started.set()
            await queue.put("read2-event-1")
            await asyncio.sleep(0)
            await queue.put("read2-event-2")
            finalized.add(call.tool_name)

    results: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for event in stream_tool_call_waves(
            calls,
            concurrent_safe=lambda call: call.tool_class.read_only,
            execute=execute,
        ):
            results.append(event)

    assert started == {"read1", "read2"}
    assert finalized == {"read2"}
    assert sorted(results) == ["read1-event", "read2-event-1", "read2-event-2"]


@pytest.mark.asyncio
async def test_scheduler_binds_classification_once() -> None:
    classified = 0
    scheduled: list[bool] = []

    def concurrent_safe(_call: str) -> bool:
        nonlocal classified
        classified += 1
        return True

    async def execute(
        call: str, scheduled_read_only: bool, queue: EventSink[str]
    ) -> None:
        scheduled.append(scheduled_read_only)
        await queue.put(call)

    results = [
        event
        async for event in stream_tool_call_waves(
            ["dynamic"], concurrent_safe=concurrent_safe, execute=execute
        )
    ]

    assert results == ["dynamic"]
    assert classified == 1
    assert scheduled == [True]


@pytest.mark.asyncio
async def test_cancelled_executor_does_not_cancel_remaining_waves() -> None:
    calls = [
        _call("read1", read_only=True),
        _call("read2", read_only=True),
        _call("write1", read_only=False),
    ]
    started: set[str] = set()

    async def execute(
        call: ResolvedToolCall, scheduled_read_only: bool, queue: EventSink[str]
    ) -> None:
        assert scheduled_read_only is call.tool_class.read_only
        started.add(call.tool_name)
        if call.tool_name == "read1":
            await queue.put("read1-cancelled")
            raise asyncio.CancelledError
        await queue.put(call.tool_name)

    results = [
        event
        async for event in stream_tool_call_waves(
            calls,
            concurrent_safe=lambda call: call.tool_class.read_only,
            execute=execute,
        )
    ]

    assert started == {"read1", "read2", "write1"}
    assert set(results) == {"read1-cancelled", "read2", "write1"}


@pytest.mark.asyncio
async def test_next_wave_waits_for_executor_finalization() -> None:
    calls = ["write", "read"]
    started: list[str] = []
    finalized: list[str] = []
    release_write = asyncio.Event()

    async def execute(
        call: str, scheduled_read_only: bool, queue: EventSink[str]
    ) -> None:
        assert scheduled_read_only is (call == "read")
        started.append(call)
        await queue.put(f"{call}-result")
        if call == "write":
            await release_write.wait()
        finalized.append(call)

    stream = stream_tool_call_waves(
        calls, concurrent_safe=lambda call: call == "read", execute=execute
    )

    assert await anext(stream) == "write-result"
    assert started == ["write"]
    assert finalized == []

    release_write.set()

    assert await anext(stream) == "read-result"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert started == ["write", "read"]
    assert finalized == ["write", "read"]


@pytest.mark.asyncio
async def test_cancellation_stops_every_call_in_current_wave() -> None:
    calls = [
        _call("read1", read_only=True),
        _call("read2", read_only=True),
        _call("write1", read_only=False),
    ]
    read_calls = {"read1", "read2"}
    started: set[str] = set()
    cancelled: set[str] = set()
    all_started = asyncio.Event()

    async def execute(
        call: ResolvedToolCall, scheduled_read_only: bool, queue: EventSink[str]
    ) -> None:
        assert scheduled_read_only is call.tool_class.read_only
        del queue
        started.add(call.tool_name)
        if read_calls <= started:
            all_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(call.tool_name)

    async def collect() -> list[str]:
        return [
            event
            async for event in stream_tool_call_waves(
                calls,
                concurrent_safe=lambda call: call.tool_class.read_only,
                execute=execute,
            )
        ]

    collector = asyncio.create_task(collect())
    await all_started.wait()
    collector.cancel()

    with pytest.raises(asyncio.CancelledError):
        await collector

    assert started == read_calls
    assert cancelled == read_calls


@pytest.mark.asyncio
async def test_none_event_is_not_treated_as_wave_completion() -> None:
    async def execute(
        call: str, scheduled_read_only: bool, queue: EventSink[str | None]
    ) -> None:
        assert call == "read"
        assert scheduled_read_only
        await queue.put(None)

    results = [
        event
        async for event in stream_tool_call_waves(
            ["read"], concurrent_safe=lambda _call: True, execute=execute
        )
    ]

    assert results == [None]


@pytest.mark.asyncio
async def test_closing_stream_cancels_wave_without_starting_later_calls() -> None:
    calls = [
        _call("read1", read_only=True),
        _call("read2", read_only=True),
        _call("write1", read_only=False),
    ]
    started: set[str] = set()
    cancelled = asyncio.Event()
    read2_started = asyncio.Event()

    async def execute(
        call: ResolvedToolCall, scheduled_read_only: bool, queue: EventSink[str]
    ) -> None:
        assert scheduled_read_only
        started.add(call.tool_name)
        if call.tool_name == "read1":
            await read2_started.wait()
            await queue.put(call.tool_name)
            return
        if call.tool_name == "read2":
            read2_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        await queue.put(call.tool_name)

    stream = stream_tool_call_waves(
        calls, concurrent_safe=lambda call: call.tool_class.read_only, execute=execute
    )

    assert await anext(stream) == "read1"
    await stream.aclose()

    assert cancelled.is_set()
    assert started == {"read1", "read2"}
