from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING

from vibe.core.types import AssistantEvent
from vibe.core.utils import ConversationLimitException

if TYPE_CHECKING:
    from vibe.core.agent_loop import AgentLoop


async def run_same_worker_repair(
    task_loop: AgentLoop, check_diagnostics: str
) -> str | None:
    summary: str | None = None
    async with aclosing(task_loop.act(check_diagnostics)) as events:
        async for event in events:
            if not isinstance(event, AssistantEvent):
                continue
            summary = event.content
            if event.stopped_by_middleware:
                raise ConversationLimitException(event.content)
    return summary


__all__ = ["run_same_worker_repair"]
