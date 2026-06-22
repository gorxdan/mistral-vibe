from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookEvent,
    HookInvocation,
    HookPromptBlock,
    HookStructuredResponse,
    HookTextReplacement,
    HookToolDenial,
    HookToolInputRewrite,
    HookUserMessage,
)

if TYPE_CHECKING:
    # Annotations-only: the concrete retry tracker lives in _handler.py and is
    # passed in by the manager; the port does not import it at runtime.
    from vibe.core.hooks._handler import HookRetryState

_HookYield = (
    HookEvent
    | HookUserMessage
    | HookPromptBlock
    | HookToolDenial
    | HookToolInputRewrite
    | HookTextReplacement
)


class HookExternalAttrs(TypedDict, total=False):
    tool_name: str
    tool_call_id: str


class _HookAction(NamedTuple):
    events: list[_HookYield]
    # The invocation the next hook in the chain receives; ``None`` keeps
    # the current one.
    next_invocation: HookInvocation | None
    should_break: bool


class HookHandler(ABC):
    """Per-type hook semantics. Stateless singleton; per-run state is
    passed in through method parameters.
    """

    @abstractmethod
    def matches(self, hook: HookConfig, invocation: HookInvocation) -> bool: ...

    def external_attributes(self, invocation: HookInvocation) -> HookExternalAttrs:
        return {}

    def on_structured(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        if response.decision == "deny":
            return self._on_deny(hook, invocation, response, retry_state)
        return self._on_allow(hook, invocation, response, retry_state)

    @abstractmethod
    def _on_deny(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        """Read the deny reason as ``response.reason or ""`` — empty is a
        valid explicit denial.
        """

    @abstractmethod
    def _on_allow(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction: ...

    @abstractmethod
    def on_passthrough(self, hook: HookConfig, retry_state: HookRetryState) -> None:
        """Side effect of a no-op outcome (empty stdout or non-strict
        failure).
        """

    def on_strict_failure(
        self, hook: HookConfig, invocation: HookInvocation, reason: str
    ) -> _HookAction | None:
        """Return an escalation action, or ``None`` to fall through to a
        plain warning.
        """
        return None
