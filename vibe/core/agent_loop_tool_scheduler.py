from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "EventSink",
    "ToolCallWave",
    "build_tool_call_waves",
    "stream_tool_call_waves",
]


class EventSink[EventT](Protocol):
    async def put(self, event: EventT) -> None: ...


@dataclass(frozen=True, slots=True)
class _WaveComplete:
    error: Exception | None = None


class _QueueEventSink[EventT]:
    def __init__(self, queue: asyncio.Queue[EventT | _WaveComplete]) -> None:
        self._queue = queue

    async def put(self, event: EventT) -> None:
        await self._queue.put(event)


@dataclass(frozen=True, slots=True)
class ToolCallWave[CallT]:
    calls: tuple[CallT, ...]
    concurrent_safe: bool


def build_tool_call_waves[CallT](
    tool_calls: Sequence[CallT], *, concurrent_safe: Callable[[CallT], bool]
) -> tuple[ToolCallWave[CallT], ...]:
    waves: list[ToolCallWave[CallT]] = []
    concurrent_calls: list[CallT] = []

    def flush_concurrent_calls() -> None:
        if not concurrent_calls:
            return
        waves.append(ToolCallWave(calls=tuple(concurrent_calls), concurrent_safe=True))
        concurrent_calls.clear()

    for tool_call in tool_calls:
        if concurrent_safe(tool_call):
            concurrent_calls.append(tool_call)
            continue
        flush_concurrent_calls()
        waves.append(ToolCallWave(calls=(tool_call,), concurrent_safe=False))

    flush_concurrent_calls()
    return tuple(waves)


async def _stream_wave[CallT, EventT](
    wave: ToolCallWave[CallT],
    *,
    execute: Callable[[CallT, bool, EventSink[EventT]], Awaitable[None]],
) -> AsyncGenerator[EventT, None]:
    queue: asyncio.Queue[EventT | _WaveComplete] = asyncio.Queue()
    event_sink = _QueueEventSink(queue)

    async def execute_one(tool_call: CallT) -> None:
        await execute(tool_call, wave.concurrent_safe, event_sink)

    tasks = [asyncio.create_task(execute_one(tool_call)) for tool_call in wave.calls]

    async def signal_when_done() -> None:
        error: Exception | None = None
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            error = next(
                (result for result in results if isinstance(result, Exception)), None
            )
        finally:
            await queue.put(_WaveComplete(error=error))

    monitor = asyncio.create_task(signal_when_done())

    try:
        while True:
            event = await queue.get()
            if isinstance(event, _WaveComplete):
                if event.error is not None:
                    raise event.error
                break
            yield event
    except GeneratorExit:
        for task in tasks:
            if not task.done():
                task.cancel()
        raise
    except asyncio.CancelledError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if not monitor.done():
            monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


async def stream_tool_call_waves[CallT, EventT](
    tool_calls: Sequence[CallT],
    *,
    concurrent_safe: Callable[[CallT], bool],
    execute: Callable[[CallT, bool, EventSink[EventT]], Awaitable[None]],
) -> AsyncGenerator[EventT, None]:
    for wave in build_tool_call_waves(tool_calls, concurrent_safe=concurrent_safe):
        stream = _stream_wave(wave, execute=execute)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()
