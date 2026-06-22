from __future__ import annotations

from vibe.core.hooks._handler import HookHandler, HookRetryState, _HookAction
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookPromptBlock,
    HookStructuredResponse,
    HookUserMessage,
)


class UserPromptSubmitHandler(HookHandler):
    """Gate for ``USER_PROMPT_SUBMIT``.

    ``deny`` blocks the prompt (no LLM turn runs; the reason is surfaced).
    ``allow`` with ``hook_specific_output.additional_context`` injects extra
    context the model sees this turn.
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
        reason = response.reason or "Prompt blocked by hook."
        return _HookAction(
            events=[
                HookPromptBlock(hook_name=hook.name, content=reason),
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=reason,
                ),
            ],
            next_invocation=None,
            should_break=True,
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
