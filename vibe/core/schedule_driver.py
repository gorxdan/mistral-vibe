"""Core poll-and-fire driver for scheduled loops — no UI dependency.

Owns the background loop that, every poll tick, asks a :class:`LoopManager`
whether a scheduled loop is due and (when the ``can_fire`` gate allows) invokes
``fire`` with it. Extracted from the Textual runner so the interactive TUI,
headless ``vibe -p``, and ACP sessions all share one implementation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from vibe.core.logger import logger

if TYPE_CHECKING:
    from vibe.core.loop import LoopManager
    from vibe.core.types import ScheduledLoop


class ScheduleDriver:
    def __init__(
        self,
        manager: LoopManager,
        *,
        can_fire: Callable[[], bool],
        fire: Callable[[ScheduledLoop], Awaitable[None]],
        poll_interval: float = 1.0,
    ) -> None:
        """Args:
        manager: the live scheduler (also handed to the schedule tool).
        can_fire: gate — return False to defer (e.g. a turn is already running).
        fire: called with each due loop; runs its prompt as a turn.
        poll_interval: max seconds between polls (sleep is bounded by the next
            due time so a soon-due loop fires promptly).
        """
        self._manager = manager
        self._can_fire = can_fire
        self._fire = fire
        self._poll_interval = poll_interval
        self._task: asyncio.Task[None] | None = None

    @property
    def manager(self) -> LoopManager:
        return self._manager

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run_until_idle(self, *, deadline: float | None = None) -> None:
        """Drive synchronously until no loops remain (one-shots drained) or
        *deadline* (a ``time.monotonic`` value) passes. For headless use where
        there is no foreground UI loop to host a background task.
        """
        loop = asyncio.get_running_loop()
        while self._manager.loops:
            if deadline is not None and loop.time() >= deadline:
                return
            wait = min(self._manager.next_due_in(), self._poll_interval)
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - loop.time()))
            await asyncio.sleep(max(0.05, wait))
            if not self._can_fire():
                continue
            due = await self._manager.pop_due()
            if due is not None:
                await self._fire(due)

    async def _run(self) -> None:
        while True:
            try:
                sleep_for = min(self._manager.next_due_in(), self._poll_interval)
                await asyncio.sleep(max(0.05, sleep_for))
                if not self._can_fire():
                    continue
                due = await self._manager.pop_due()
                if due is None:
                    continue
                await self._fire(due)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Error polling scheduled loops", exc_info=e)
