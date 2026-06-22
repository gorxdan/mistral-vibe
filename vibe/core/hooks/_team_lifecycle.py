from __future__ import annotations

from vibe.core.hooks._handler import HookRetryState
from vibe.core.hooks._port import HookHandler, _HookAction
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookStructuredResponse,
)
from vibe.core.logger import logger


class TeamLifecycleHandler(HookHandler):
    """Informational handler for team lifecycle events
    (``TEAMMATE_IDLE`` / ``TASK_CREATED`` / ``TASK_COMPLETED``).

    These events are notifications, not gates: a hook can observe them (and
    emit a structured ``allow`` system message) but cannot deny or rewrite the
    lifecycle action. A ``deny`` decision is recorded as a warning and treated
    as a passthrough -- the lifecycle proceeds regardless.
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
        # Lifecycle events are not deniable; record the hook's reason as a
        # warning but let the action proceed.
        logger.warning(
            "Hook %s denied lifecycle event %s (non-binding): %s",
            hook.name,
            invocation.hook_event_name,
            response.reason or "(no reason)",
        )
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=response.reason or "lifecycle hook denied (non-binding)",
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
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.OK,
                    content=response.system_message,
                )
            ],
            next_invocation=None,
            should_break=False,
        )

    def on_passthrough(self, hook: HookConfig, retry_state: HookRetryState) -> None:
        retry_state.track_no_retry(hook.name)
