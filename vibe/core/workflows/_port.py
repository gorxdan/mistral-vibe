from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vibe.core.agent_loop import AgentLoop
    from vibe.core.tools.base import InvokeContext


class AgentLoopFactory(Protocol):
    def __call__(
        self, prompt: str, *, agent: str, parent_context: InvokeContext | None
    ) -> AgentLoop: ...
