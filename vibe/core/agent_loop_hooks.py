"""Hook orchestration mixin for AgentLoop.

Provides before_tool, after_tool, and post_agent_turn hook lifecycle
methods.  Extracted from ``agent_loop.py`` to keep the main module
focused on the core conversation loop and tool execution flow.

Implicit dependencies on the host class (AgentLoop):

Attributes:
    _hooks_manager   (HooksManager | None)
    session_id       (str)
    parent_session_id (str | None)
    session_logger   (SessionLogger)
    stats            (AgentStats)
    messages         (MessageList)

Methods:
    _handle_tool_response(tool_call, text, status, decision, result, span)
    _serialize_tool_input(tool_call) -> dict[str, Any]
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from opentelemetry import trace
import orjson
from pydantic import ValidationError

from vibe.core.hooks.models import (
    AfterToolInvocation,
    BeforeToolInvocation,
    HookEvent,
    HookPromptBlock,
    HookSessionContext,
    HookTextReplacement,
    HookToolDenial,
    HookToolInputRewrite,
    HookUserMessage,
    NotificationInvocation,
    PostAgentTurnInvocation,
    PreCompactInvocation,
    SessionEndInvocation,
    SessionStartInvocation,
    StopInvocation,
    ToolStatus,
    UserPromptSubmitInvocation,
)
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.logger import logger
from vibe.core.paths import safe_cwd
from vibe.core.types import ToolResultEvent
from vibe.core.utils import (
    CANCELLATION_TAG,
    TOOL_ERROR_TAG,
    CancellationReason,
    get_user_cancellation_message,
)

if TYPE_CHECKING:
    from vibe.core.agent_loop import ToolDecision
    from vibe.core.hooks.manager import HooksManager
    from vibe.core.session.session_logger import SessionLogger
    from vibe.core.types import AgentStats, LLMMessage, MessageList


class _BeforeToolResolution(NamedTuple):
    # ``denial_event`` is non-None when the pipeline ended in a denial
    # (explicit or synthesized from a failed rewrite re-validation);
    # callers yield it and stop.  Otherwise tool_call / tool_input hold
    # the (possibly rewritten) values to use for permission + execution.
    tool_call: ResolvedToolCall
    tool_input: dict[str, Any]
    denial_event: ToolResultEvent | None


class AgentLoopHooksMixin:
    """Mixin that adds hook orchestration to AgentLoop.

    See module docstring for the implicit contract with the host class.
    """

    # Declared for type-checking only; set by AgentLoop.__init__.
    _hooks_manager: HooksManager | None
    session_id: str
    parent_session_id: str | None
    session_logger: SessionLogger
    stats: AgentStats
    messages: MessageList

    def _handle_tool_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        status: Literal["success", "failure", "skipped"],
        decision: ToolDecision | None = None,
        result: dict[str, Any] | None = None,
        span: trace.Span | None = None,
        observe_orchestration: bool = True,
    ) -> None: ...

    def _serialize_tool_input(self, tool_call: ResolvedToolCall) -> dict[str, Any]:
        return tool_call.validated_args.model_dump(mode="json")

    def _hook_session_context(self) -> HookSessionContext:
        transcript = ""
        if self.session_logger.enabled and self.session_logger.session_dir is not None:
            transcript = str(self.session_logger.messages_filepath.resolve())
        return HookSessionContext(
            session_id=self.session_id,
            transcript_path=transcript,
            cwd=str(safe_cwd()),
            parent_session_id=self.parent_session_id,
        )

    async def _run_post_agent_turn_hooks(
        self,
    ) -> AsyncGenerator[HookEvent | HookUserMessage]:
        if not self._hooks_manager:
            return
        invocation = PostAgentTurnInvocation(
            **self._hook_session_context().model_dump()
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookUserMessage)):
                yield ev

    async def _dispatch_user_prompt_submit_hooks(
        self, prompt: str, message_id: str | None, has_images: bool
    ) -> tuple[str | None, list[str], list[HookEvent]]:
        """Run user_prompt_submit hooks.

        Returns ``(block_reason, injected_contexts, events)``. ``block_reason``
        non-None means a hook denied the prompt (no LLM turn should run).
        """
        events: list[HookEvent] = []
        injected: list[str] = []
        block_reason: str | None = None
        if not self._hooks_manager:
            return None, injected, events
        invocation = UserPromptSubmitInvocation(
            **self._hook_session_context().model_dump(),
            prompt=prompt,
            message_id=message_id,
            has_images=has_images,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, HookPromptBlock):
                block_reason = ev.content
            elif isinstance(ev, HookUserMessage):
                injected.append(ev.content)
            elif isinstance(ev, HookEvent):
                events.append(ev)
        return block_reason, injected, events

    async def _dispatch_stop_hooks(
        self, stop_hook_active: bool
    ) -> tuple[LLMMessage | None, list[HookEvent]]:
        """Run stop hooks when the turn is about to end. Deny → a continuation
        user message (capped by the retry state); allow → end the turn.
        """
        from vibe.core.types import InjectedMessageKind, LLMMessage, Role

        events: list[HookEvent] = []
        continuation: LLMMessage | None = None
        if not self._hooks_manager:
            return None, events
        invocation = StopInvocation(
            **self._hook_session_context().model_dump(),
            stop_hook_active=stop_hook_active,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, HookUserMessage):
                continuation = LLMMessage(
                    role=Role.USER,
                    content=ev.content,
                    injected=True,
                    injected_kind=InjectedMessageKind.STOP_HOOK,
                )
            elif isinstance(ev, HookEvent):
                events.append(ev)
        return continuation, events

    async def _dispatch_session_start_hooks(
        self, source: str
    ) -> tuple[list[str], list[HookEvent]]:
        """Run session-start hooks. Returns (injected_contexts, events): a hook
        may inject additional_context the first turn sees.
        """
        events: list[HookEvent] = []
        injected: list[str] = []
        if not self._hooks_manager:
            return injected, events
        invocation = SessionStartInvocation(
            **self._hook_session_context().model_dump(), source=source
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, HookUserMessage):
                injected.append(ev.content)
            elif isinstance(ev, HookEvent):
                events.append(ev)
        return injected, events

    async def _fire_session_end_hooks(self, reason: str) -> None:
        """Run session-end hooks best-effort and time-bounded (the process may
        be tearing down). Events are logged, not surfaced; never raises.
        """
        manager = self._hooks_manager
        if manager is None:
            return
        invocation = SessionEndInvocation(
            **self._hook_session_context().model_dump(), reason=reason
        )

        async def _drain() -> None:
            async for _ev in manager.run(invocation):
                pass

        try:
            await asyncio.wait_for(_drain(), timeout=5.0)
        except (TimeoutError, Exception) as e:
            logger.warning("session_end hooks failed/timed out: %s", e)

    async def _fire_notification_hooks(
        self, notification_type: str, message: str, tool_name: str | None = None
    ) -> None:
        """Notify on user-attention events (permission/question). Best-effort,
        time-bounded, never raises into the caller.
        """
        manager = self._hooks_manager
        if manager is None:
            return
        invocation = NotificationInvocation(
            **self._hook_session_context().model_dump(),
            notification_type=notification_type,
            message=message,
            tool_name=tool_name,
        )

        async def _drain() -> None:
            async for _ev in manager.run(invocation):
                pass

        try:
            await asyncio.wait_for(_drain(), timeout=5.0)
        except (TimeoutError, Exception) as e:
            logger.warning("notification hooks failed/timed out: %s", e)

    async def _run_pre_compact_hooks(
        self, trigger: str, current_context_tokens: int, threshold: int
    ) -> AsyncGenerator[HookEvent]:
        """Notify pre-compaction hooks (observe-only). Never blocks compaction."""
        if not self._hooks_manager:
            return
        invocation = PreCompactInvocation(
            **self._hook_session_context().model_dump(),
            trigger=trigger,
            current_context_tokens=current_context_tokens,
            threshold=threshold,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, HookEvent):
                yield ev

    async def _run_before_tool_hooks(
        self, tool_call: ResolvedToolCall, tool_input: dict[str, Any]
    ) -> AsyncGenerator[HookEvent | HookToolDenial | HookToolInputRewrite]:
        if not self._hooks_manager:
            return
        invocation = BeforeToolInvocation(
            **self._hook_session_context().model_dump(),
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.call_id,
            tool_input=tool_input,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookToolDenial, HookToolInputRewrite)):
                yield ev

    async def _run_after_tool_hooks(
        self,
        tool_call: ResolvedToolCall,
        *,
        tool_input: dict[str, Any],
        tool_status: ToolStatus,
        tool_output: dict[str, Any] | None = None,
        tool_error: str | None = None,
        duration_ms: float = 0.0,
        initial_text: str = "",
    ) -> AsyncGenerator[HookEvent | HookTextReplacement]:
        if not self._hooks_manager:
            return
        invocation = AfterToolInvocation(
            **self._hook_session_context().model_dump(),
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.call_id,
            tool_input=tool_input,
            tool_status=tool_status,
            tool_output=tool_output,
            tool_output_text=initial_text,
            tool_error=tool_error,
            duration_ms=duration_ms,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookTextReplacement)):
                yield ev

    async def _run_after_tool_and_finalize(
        self,
        tool_call: ResolvedToolCall,
        *,
        tool_input: dict[str, Any],
        tool_status: ToolStatus,
        response_status: Literal["success", "failure", "skipped"],
        decision: ToolDecision | None = None,
        span: trace.Span,
        tool_output: dict[str, Any] | None = None,
        tool_error: str | None = None,
        duration_ms: float = 0.0,
        initial_text: str = "",
        observe_orchestration: bool = True,
    ) -> tuple[list[HookEvent], BaseException | None]:
        final_text = initial_text
        events: list[HookEvent] = []
        error: BaseException | None = None
        try:
            async for ev in self._run_after_tool_hooks(
                tool_call,
                tool_input=tool_input,
                tool_status=tool_status,
                tool_output=tool_output,
                tool_error=tool_error,
                duration_ms=duration_ms,
                initial_text=initial_text,
            ):
                if isinstance(ev, HookTextReplacement):
                    final_text = ev.text
                else:
                    events.append(ev)
        except asyncio.CancelledError as exc:
            error = exc
        except Exception as exc:
            error = exc
        self._handle_tool_response(
            tool_call,
            final_text,
            response_status,
            decision,
            tool_output,
            span=span,
            observe_orchestration=observe_orchestration,
        )
        return events, error

    async def _run_before_tool_pipeline(
        self,
        tool_call: ResolvedToolCall,
        tool_input: dict[str, Any],
        *,
        span: trace.Span,
    ) -> tuple[list[HookEvent], _BeforeToolResolution]:
        """Validate each rewrite as it arrives; first invalid one aborts the chain.

        Events are buffered (not streamed) because before_tool hooks are
        gating checks expected to complete quickly.
        """
        events: list[HookEvent] = []
        async for ev in self._run_before_tool_hooks(tool_call, tool_input):
            if isinstance(ev, HookToolDenial):
                return events, _BeforeToolResolution(
                    tool_call=tool_call,
                    tool_input=tool_input,
                    denial_event=self._handle_before_tool_denial(
                        tool_call, ev, span=span
                    ),
                )
            if isinstance(ev, HookToolInputRewrite):
                rewritten = self._apply_tool_input_rewrite(tool_call, ev)
                if isinstance(rewritten, HookToolDenial):
                    return events, _BeforeToolResolution(
                        tool_call=tool_call,
                        tool_input=tool_input,
                        denial_event=self._handle_before_tool_denial(
                            tool_call, rewritten, span=span
                        ),
                    )
                tool_call, tool_input = rewritten
                continue
            events.append(ev)

        return events, _BeforeToolResolution(
            tool_call=tool_call, tool_input=tool_input, denial_event=None
        )

    def _apply_tool_input_rewrite(
        self, tool_call: ResolvedToolCall, rewrite: HookToolInputRewrite
    ) -> tuple[ResolvedToolCall, dict[str, Any]] | HookToolDenial:
        """Re-validate a rewrite against the tool's args model.

        Rebuilds ``ResolvedToolCall`` and patches the assistant message so the
        LLM sees the validated rewrite next turn. Returns a synthesized denial
        on validation failure.
        """
        tool_class = tool_call.tool_class
        args_model, _ = tool_class._get_tool_args_results()
        try:
            new_validated = args_model.model_validate(rewrite.tool_input)
        except ValidationError as e:
            logger.warning(
                "Hook %s produced invalid tool_input for '%s': %s",
                rewrite.hook_name,
                tool_call.tool_name,
                e,
            )
            return HookToolDenial(
                hook_name=rewrite.hook_name,
                content=(
                    f"Hook '{rewrite.hook_name}' rewrote tool_input but the"
                    f" result failed validation against"
                    f" {tool_call.tool_name}: {e}"
                ),
            )

        new_tool_call = tool_call.model_copy(update={"validated_args": new_validated})
        new_tool_input = self._serialize_tool_input(new_tool_call)
        self._patch_assistant_tool_call_args(tool_call.call_id, new_tool_input)
        return new_tool_call, new_tool_input

    def _patch_assistant_tool_call_args(
        self, call_id: str, new_args: dict[str, Any]
    ) -> None:
        """Mutate the assistant message's tool_calls so the transcript reflects
        the validated rewrite evaluated by the host, not the original args.
        """
        if not call_id:
            return
        encoded = orjson.dumps(new_args).decode("utf-8")
        for message in reversed(self.messages):
            if not message.tool_calls:
                continue
            for tc in message.tool_calls:
                if tc.id == call_id:
                    tc.function.arguments = encoded
                    return

    def _handle_before_tool_denial(
        self, tool_call: ResolvedToolCall, denial: HookToolDenial, *, span: trace.Span
    ) -> ToolResultEvent:
        self.stats.tool_calls_hook_denied += 1
        denial_text = (
            f"<{TOOL_ERROR_TAG}>Tool '{tool_call.tool_name}' was denied by "
            f"hook '{denial.hook_name}': {denial.content}</{TOOL_ERROR_TAG}>"
        )
        self._handle_tool_response(tool_call, denial_text, "skipped", None, span=span)
        return ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            skipped=True,
            skip_reason=denial_text,
            cancelled=False,
            tool_call_id=tool_call.call_id,
        )

    async def _handle_tool_skip(
        self, tool_call: ResolvedToolCall, decision: ToolDecision, *, span: trace.Span
    ) -> AsyncGenerator[ToolResultEvent | HookEvent]:
        self.stats.tool_calls_rejected += 1
        skip_reason = decision.feedback or str(
            get_user_cancellation_message(
                CancellationReason.TOOL_SKIPPED, tool_call.tool_name
            )
        )
        result_event = ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            skipped=True,
            skip_reason=skip_reason,
            cancelled=f"<{CANCELLATION_TAG}>" in skip_reason,
            tool_call_id=tool_call.call_id,
        )
        self._handle_tool_response(
            tool_call, skip_reason, "skipped", decision, span=span
        )
        yield result_event

    async def _finalize_cancelled_tool(
        self,
        tool_call: ResolvedToolCall,
        tool_input: dict[str, Any],
        decision: ToolDecision | None,
        cancel_text: str,
        *,
        span: trace.Span,
        tool_started: bool,
    ) -> tuple[list[HookEvent], BaseException | None]:
        """Run cancellation hooks before publishing the terminal result."""
        if not tool_started:
            self._handle_tool_response(
                tool_call, cancel_text, "failure", decision, span=span
            )
            return [], None
        return await self._run_after_tool_and_finalize(
            tool_call,
            tool_input=tool_input,
            tool_status="cancelled",
            response_status="failure",
            decision=decision,
            span=span,
            tool_error=cancel_text,
            initial_text=cancel_text,
        )

    async def _dispatch_post_turn_hooks(
        self,
    ) -> tuple[LLMMessage | None, list[HookEvent]]:
        """Run post-agent-turn hooks and separate retry injection from events.

        Returns a ``(retry_message, events)`` tuple.  ``retry_message`` is
        an injected ``LLMMessage`` when a hook requests a retry, else ``None``.
        """
        from vibe.core.types import InjectedMessageKind, LLMMessage, Role

        events: list[HookEvent] = []
        retry_msg: LLMMessage | None = None
        async for hook_event in self._run_post_agent_turn_hooks():
            if isinstance(hook_event, HookUserMessage):
                retry_msg = LLMMessage(
                    role=Role.USER,
                    content=hook_event.content,
                    injected=True,
                    injected_kind=InjectedMessageKind.POST_AGENT_TURN_HOOK,
                )
            else:
                events.append(hook_event)
        return retry_msg, events
