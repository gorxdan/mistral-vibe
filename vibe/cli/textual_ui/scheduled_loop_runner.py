from __future__ import annotations

from collections.abc import Awaitable, Callable
import time

from textual.widget import Widget

from vibe.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from vibe.core.loop import (
    USAGE_HINT,
    LoopErrorResult,
    LoopListResult,
    LoopManager,
    LoopOkResult,
    ScheduledLoop,
    format_duration,
)
from vibe.core.schedule_driver import ScheduleDriver
from vibe.core.session.session_logger import SessionLogger


def _format_loop_list(loops: list[ScheduledLoop]) -> str:
    if not loops:
        return "No scheduled loops."
    now = time.time()
    rows = ["| Prompt | Next in | Every | ID |", "|--------|------|-------|----|"]
    for loop in loops:
        remaining = format_duration(max(0, int(loop.next_fire_at - now)), short=True)
        interval = format_duration(loop.interval_seconds)
        prompt = loop.prompt.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {prompt} | {remaining} | {interval} | `{loop.id}` |")
    return "\n".join(rows)


class ScheduledLoopRunner:
    def __init__(
        self,
        session_logger: SessionLogger,
        *,
        can_fire: Callable[[], bool],
        fire: Callable[[str], Awaitable[None]],
        mount: Callable[[Widget], Awaitable[None]],
        tools_collapsed: Callable[[], bool],
    ) -> None:
        self._session_logger = session_logger
        self._manager = LoopManager(session_logger)
        self._mount = mount
        self._tools_collapsed = tools_collapsed
        # The poll-fire loop is the shared core driver; this class only adds the
        # TUI bits (mount a "fired" message, wrap command results as widgets).
        self._driver = ScheduleDriver(
            self._manager, can_fire=can_fire, fire=self._fire_and_announce(fire)
        )

    def _fire_and_announce(
        self, fire: Callable[[str], Awaitable[None]]
    ) -> Callable[[ScheduledLoop], Awaitable[None]]:
        async def _run(due: ScheduledLoop) -> None:
            await fire(due.prompt)
            await self._mount(UserCommandMessage(f"Loop `{due.id}` fired"))

        return _run

    @property
    def manager(self) -> LoopManager:
        """The live scheduler — handed to AgentLoop so the `schedule` tool
        enqueues loops this runner polls and fires.
        """
        return self._manager

    def restore_from_session(self) -> None:
        metadata = self._session_logger.session_metadata
        self._manager.restore(list(metadata.loops) if metadata is not None else [])

    def start(self) -> None:
        self._driver.start()

    async def stop(self) -> None:
        await self._driver.stop()

    async def handle_command(self, cmd_args: str) -> Widget:
        result = await self._manager.handle_command(cmd_args)
        match result:
            case LoopListResult(loops=loops):
                return UserCommandMessage(_format_loop_list(loops))
            case LoopErrorResult(message=message):
                return ErrorMessage(
                    f"{message}\n{USAGE_HINT}", collapsed=self._tools_collapsed()
                )
            case LoopOkResult(message=message):
                return UserCommandMessage(message)
