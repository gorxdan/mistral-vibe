"""Agent conversation loop package.

Public surface re-exported from the private :mod:`_loop` implementation module
so that ``from vibe.core.agent_loop import AgentLoop`` keeps resolving after the
flat module was split into a package. Fork-only subsystem mixins live in sibling
top-level modules (``vibe.core.agent_loop_memory``, ``vibe.core.agent_loop_failover``,
``vibe.core.agent_loop_hooks``) — matching upstream's own ``agent_loop_hooks``
placement — and are composed onto ``AgentLoop``.
"""

from __future__ import annotations

from vibe.core.agent_loop._errors import (
    AgentLoopError,
    AgentLoopLLMResponseError,
    AgentLoopStateError,
    CompactionFailedError,
    ImagesNotSupportedError,
    InvalidStreamError,
    TeleportError,
    _degenerate_response_reason,
)
from vibe.core.agent_loop._loop import (
    _TELEPORT_AVAILABLE,
    AgentLoop,
    AgentLoopParams,
    _git_executable_present,
    _teleport_service_cls,
)
from vibe.core.agent_loop_models import ToolDecision, ToolExecutionResponse

__all__ = [
    "_TELEPORT_AVAILABLE",
    "AgentLoop",
    "AgentLoopError",
    "AgentLoopLLMResponseError",
    "AgentLoopParams",
    "AgentLoopStateError",
    "CompactionFailedError",
    "ImagesNotSupportedError",
    "InvalidStreamError",
    "TeleportError",
    "ToolDecision",
    "ToolExecutionResponse",
    "_degenerate_response_reason",
    "_git_executable_present",
    "_teleport_service_cls",
]
