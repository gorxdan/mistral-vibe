from __future__ import annotations

from vibe.core.hooks._handler import HookRetryState
from vibe.core.hooks._port import HookHandler, _HookAction
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookStructuredResponse,
    HookUserMessage,
)
from vibe.core.logger import logger


class SessionStartHandler(HookHandler):
    """SESSION_START: notification + optional context injection.

    ``allow`` with ``hook_specific_output.additional_context`` injects that text
    as a user message the first turn sees (Claude Code parity). ``deny`` is
    non-binding (the session still starts) — recorded as a warning.
    """

    def matches(self, hook: HookConfig, invocation: HookInvocation) -> bool:
        return True

    def _on_deny(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        logger.warning(
            "Hook %s denied session_start (non-binding): %s",
            hook.name,
            response.reason or "(no reason)",
        )
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=response.reason or "session_start denied (non-binding)",
                )
            ],
            next_invocation=None,
            should_break=False,
        )

    def _on_allow(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        retry_state.track_no_retry(hook.name)
        events: list = []
        added = response.hook_specific_output.additional_context
        if added:
            events.append(HookUserMessage(content=added))
        if response.system_message:
            events.append(
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.OK,
                    content=response.system_message,
                )
            )
        return _HookAction(events=events, next_invocation=None, should_break=False)

    def on_passthrough(self, hook: HookConfig, retry_state: HookRetryState) -> None:
        retry_state.track_no_retry(hook.name)
