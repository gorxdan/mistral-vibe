from __future__ import annotations

from typing import Protocol, runtime_checkable

from vibe.core.types import ScheduledLoop


@runtime_checkable
class Scheduler(Protocol):
    """The slice of LoopManager the model-facing `schedule` tool needs. Passed
    into InvokeContext so the tool mutates the SAME live manager the runner
    polls (so newly-scheduled loops actually fire).
    """

    @property
    def loops(self) -> list[ScheduledLoop]: ...

    async def add_loop(
        self, interval_seconds: int, prompt: str, *, recurring: bool = True
    ) -> ScheduledLoop: ...

    async def cancel(self, target: str) -> int: ...
