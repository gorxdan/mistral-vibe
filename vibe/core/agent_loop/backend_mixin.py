"""Backend chat / streaming mixin for AgentLoop.

Provides message shaping for the LLM backend (injected-context capping, late
memory injection), the synchronous and streaming chat paths, and post-turn usage
stats. Extracted from the loop module.

Implicit dependencies on the host class (AgentLoop):

Attributes (set by AgentLoop.__init__):
    backend               (BackendLike)
    format_handler        (APIToolFormatHandler)
    telemetry_client      (TelemetryClient)
    tool_manager          (ToolManager)
    stats                 (AgentStats)
    session_id            (str)
    _usage_recorder       (UsageRecorder)
    _response_format      (Any — response-format config for the active backend)
    _codex_routing        (Any — codex/responses-API turn routing state)
    _max_output_override  (int | None)
    _fallback_model_override (ModelConfig | None — shared with failover mixin)
    _late_memory_section  (str — shared with memory mixin)
    _last_user_message    (str | None)

Properties (defined on AgentLoop):
    config                (VibeConfig)
    effective_model       (() -> ModelConfig)

Methods (defined elsewhere on AgentLoop / sibling mixins):
    _build_backend_metadata() -> dict
    _capture_codex_turn_state(...) -> None
    _capture_rate_limits(provider, sink) -> None
    _wire_temperature(active_model, provider) -> float | None
    _wrap_memories(section) -> str   [AgentLoopMemoryMixin]
    _update_stats(usage, time_seconds, ...) -> None
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
import time
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from vibe.core import stream_tracer
from vibe.core.agent_loop._errors import (
    _STREAM_DEGENERATE_RETRIES,
    AgentLoopLLMResponseError,
    InvalidStreamError,
    _degenerate_response_reason,
    _raise_for_backend_error,
    _refusal_error,
)
from vibe.core.agent_loop.safety_mixin import AgentLoopSafetyMixin
from vibe.core.agent_loop_failover import AgentLoopFailoverMixin
from vibe.core.agent_loop_memory import AgentLoopMemoryMixin
from vibe.core.baseline_scaling import baseline_tier_for, trim_tool_descriptions
from vibe.core.compaction import truncate_compaction_context_for_backend
from vibe.core.llm.backend.factory import create_backend
from vibe.core.llm.types import BackendLike, CompletionRequest
from vibe.core.logger import logger
from vibe.core.telemetry.build_metadata import build_attachment_counts
from vibe.core.tracing import (
    add_message_content_events,
    chat_span,
    set_finish_reason,
    set_usage,
)
from vibe.core.types import (
    AgentStats,
    AvailableTool,
    InjectedMessageKind,
    LLMChunk,
    LLMChunkAccumulator,
    LLMMessage,
    LLMUsage,
    ResponseTooLongError,
    Role,
)
from vibe.core.usage import UsageRecord, compute_cost, lookup_pricing
from vibe.core.utils.tokens import truncate_middle_to_tokens

if TYPE_CHECKING:
    from vibe.core.config import ModelConfig, ProviderConfig, VibeConfig
    from vibe.core.llm.format import APIToolFormatHandler
    from vibe.core.telemetry.send import TelemetryClient
    from vibe.core.telemetry.types import TelemetryCallType, TelemetryRequestMetadata
    from vibe.core.tools.manager import ToolManager
    from vibe.core.usage import UsageRecorder


class AgentLoopBackendMixin(
    AgentLoopFailoverMixin, AgentLoopMemoryMixin, AgentLoopSafetyMixin
):
    """Mixin that adds the LLM-backend chat / streaming path to AgentLoop.

    Inherits Failover (for ``_fallback_model_override``), Memory (for
    ``_wrap_memories`` / ``_late_memory_section``), and Safety (→ Hooks, for
    ``messages`` etc.) so all shared attrs/methods resolve via the inheritance
    chain without redeclaration.
    """

    # Declared for type-checking only; set by AgentLoop.__init__.
    backend: BackendLike
    format_handler: APIToolFormatHandler
    telemetry_client: TelemetryClient
    tool_manager: ToolManager
    stats: AgentStats
    session_id: str
    _usage_recorder: UsageRecorder
    _response_format: Any
    _codex_routing: Any
    _max_output_override: int | None

    @property
    def config(self) -> VibeConfig: ...

    def effective_model(self) -> ModelConfig: ...

    def _last_user_message(self) -> LLMMessage | None: ...

    def _build_backend_metadata(
        self, call_type: TelemetryCallType | None = None
    ) -> TelemetryRequestMetadata: ...

    def _capture_codex_turn_state(self, *args: Any, **kwargs: Any) -> None: ...

    def _capture_rate_limits(
        self, provider: ProviderConfig, sink: dict[str, str] | None
    ) -> None: ...

    @staticmethod
    def _wire_temperature(
        active_model: ModelConfig, provider: ProviderConfig
    ) -> float | None: ...
    def _messages_for_backend(self, active_model: ModelConfig) -> Sequence[LLMMessage]:
        msgs = self._cap_injected_messages_for_backend(self._with_late_memory())
        if active_model.supports_images:
            return msgs
        if not any(m.images for m in msgs):
            return msgs
        return [m.model_copy(update={"images": None}) if m.images else m for m in msgs]

    def _cap_injected_messages_for_backend(
        self, messages: Sequence[LLMMessage]
    ) -> Sequence[LLMMessage]:
        max_tokens = self.config.context_shaping.max_injected_message_tokens
        if max_tokens <= 0:
            return messages
        capped: list[LLMMessage] | None = None
        for idx, message in enumerate(messages):
            if not message.injected or not isinstance(message.content, str):
                continue
            if message.injected_kind == InjectedMessageKind.COMPACTION_CONTEXT:
                content = truncate_compaction_context_for_backend(
                    message.content, max_tokens
                )
            else:
                content = truncate_middle_to_tokens(message.content, max_tokens)
            if content == message.content:
                continue
            if capped is None:
                capped = list(messages)
            capped[idx] = message.model_copy(update={"content": content})
        return messages if capped is None else capped

    def _with_late_memory(self) -> Sequence[LLMMessage]:
        section = self._late_memory_section
        if self.config.memory.inject_mode != "late" or not section:
            return self.messages
        mem_msg = LLMMessage(
            role=Role.USER,
            content=self._wrap_memories(section),
            injected=True,
            injected_kind=InjectedMessageKind.MEMORY,
        )
        msgs = list(self.messages)
        if self.config.memory.late_anchor == "tail":
            # Absolute tail: adapters rely on trailing MEMORY message(s) being the
            # request's suffix when placing the history cache breakpoint.
            msgs.append(mem_msg)
            return msgs
        insert_at = next(
            (i for i in range(len(msgs) - 1, -1, -1) if msgs[i].role == Role.USER),
            len(msgs),
        )
        msgs.insert(insert_at, mem_msg)
        return msgs

    def count_history_images_unsupported_by_active_model(self) -> int:
        try:
            active_model = self.config.get_active_model()
        except ValueError:
            return 0
        if active_model.supports_images:
            return 0
        return sum(1 for m in self.messages if m.images)

    def _resolve_active_model(
        self, model_override: ModelConfig | None = None
    ) -> tuple[ModelConfig, ProviderConfig]:
        active_model = (
            model_override
            or self._fallback_model_override
            or self.config.get_active_model()
        )
        return active_model, self.config.get_provider_for_model(active_model)

    def _capture_chat_content(
        self, span: trace.Span, response_msg: LLMMessage, user_msg: LLMMessage | None
    ) -> None:
        """Attach turn prose to the chat span when content capture is opted in.

        Off by default; the only way a long session's recall behaviour (model
        restating/contradicting itself) becomes visible in a trace, since spans
        otherwise carry token counts but no message text.
        """
        if not self.config.otel_capture_content:
            return
        add_message_content_events(
            span,
            user_text=user_msg.content if user_msg else None,
            assistant_text=response_msg.content,
            reasoning_text=response_msg.reasoning_content,
            tool_call_names=[
                tc.function.name
                for tc in (response_msg.tool_calls or ())
                if tc.function.name
            ],
        )

    def _available_tools(self, active_model: ModelConfig) -> list[AvailableTool]:
        # Tool subset is tier-invariant (no tool removed); only the schema-text
        # description is trimmed on a small-window tier. Keyed on the per-turn
        # effective model so a failover to a small window trims consistently.
        tier = baseline_tier_for(active_model, self.config)
        return self.format_handler.get_available_tools(
            self.tool_manager,
            trim_descriptions=trim_tool_descriptions(tier, self.config),
            description_max_chars=self.config.baseline_scaling.tool_description_max_chars,
        )

    async def _chat(
        self,
        max_tokens: int | None = None,
        model_override: ModelConfig | None = None,
        *,
        harness: bool = False,
    ) -> LLMChunk:
        # Apply the output-escalation override only to main-turn calls: callers
        # that set model_override (e.g. compaction summary) must not inherit it.
        if max_tokens is None and model_override is None:
            max_tokens = self._max_output_override
        active_model, provider = self._resolve_active_model(model_override)
        # self.backend always serves effective_model()'s provider (init, failover,
        # and reload keep them in lockstep). A model_override (e.g. compaction)
        # may target a different provider than the current failover backend —
        # reuse self.backend only when providers match, otherwise build a one-off
        # backend so the model name + temperature reach the right endpoint
        # (gpt-5.5 reaching a kimi backend -> "invalid temperature").
        backend = self.backend
        if (
            model_override is not None
            and provider.name
            != self.config.get_provider_for_model(self.effective_model()).name
        ):
            backend = create_backend(provider=provider, timeout=self.config.api_timeout)
        backend_metadata = self._build_backend_metadata()

        available_tools = self._available_tools(active_model)
        tool_choice = self.format_handler.get_tool_choice()

        last_user_message = self._last_user_message()
        self.telemetry_client.send_request_sent(
            model=active_model.alias,
            nb_context_chars=sum(len(m.content or "") for m in self.messages),
            nb_context_messages=len(self.messages),
            nb_prompt_chars=len(last_user_message.content or "")
            if last_user_message
            else 0,
            call_type=backend_metadata.call_type,
            message_id=backend_metadata.message_id,
            attachment_counts=build_attachment_counts(
                last_user_message, supports_images=active_model.supports_images
            ),
        )

        try:
            async with chat_span(
                model=active_model.name,
                provider=provider.name,
                temperature=self._wire_temperature(active_model, provider),
                max_tokens=max_tokens,
                thinking=active_model.thinking,
            ) as _span:
                start_time = time.perf_counter()
                extra_headers, turn_state_sink = self._codex_routing(provider)
                result = await backend.complete(
                    CompletionRequest(
                        model=active_model,
                        messages=self._messages_for_backend(active_model),
                        temperature=active_model.temperature,
                        tools=available_tools,
                        tool_choice=tool_choice,
                        extra_headers=extra_headers,
                        max_tokens=max_tokens,
                        metadata=backend_metadata.model_dump(exclude_none=True),
                        response_format=self._response_format,
                    ),
                    response_headers_sink=turn_state_sink,
                )
                end_time = time.perf_counter()
                self._capture_codex_turn_state(turn_state_sink)
                self._capture_rate_limits(provider, turn_state_sink)

                if result.usage is None:
                    raise AgentLoopLLMResponseError(
                        "Usage data missing in non-streaming completion response"
                    )
                self._update_stats(
                    usage=result.usage,
                    time_seconds=end_time - start_time,
                    provider=provider,
                    model=active_model,
                    harness=harness,
                )
                set_usage(_span, result.usage)
                set_finish_reason(_span, result.stop.reason if result.stop else None)
                self._capture_chat_content(_span, result.message, last_user_message)

            if result.correlation_id:
                self.telemetry_client.last_correlation_id = result.correlation_id

            processed_message = self.format_handler.process_api_response_message(
                result.message
            )
            # Raise before committing the truncated turn to history so the
            # escalation retry (larger max_tokens) starts from a clean message list.
            if result.stop and result.stop.is_truncated:
                raise ResponseTooLongError(provider.name, active_model.name)
            self.messages.append(processed_message)
            if result.stop and result.stop.is_refusal:
                raise _refusal_error(provider.name, active_model.name, result)
            return LLMChunk(
                message=processed_message, usage=result.usage, stop=result.stop
            )

        except Exception as e:
            _raise_for_backend_error(e, provider.name, active_model.name)

    async def _chat_streaming(
        self, max_tokens: int | None = None
    ) -> AsyncGenerator[LLMChunk]:
        if max_tokens is None:
            max_tokens = self._max_output_override
        active_model, provider = self._resolve_active_model()
        backend_metadata = self._build_backend_metadata()

        available_tools = self._available_tools(active_model)
        tool_choice = self.format_handler.get_tool_choice()

        last_user_message = self._last_user_message()
        self.telemetry_client.send_request_sent(
            model=active_model.alias,
            nb_context_chars=sum(len(m.content or "") for m in self.messages),
            nb_context_messages=len(self.messages),
            nb_prompt_chars=len(last_user_message.content or "")
            if last_user_message
            else 0,
            call_type=backend_metadata.call_type,
            message_id=backend_metadata.message_id,
            attachment_counts=build_attachment_counts(
                last_user_message, supports_images=active_model.supports_images
            ),
        )

        for attempt in range(_STREAM_DEGENERATE_RETRIES):
            try:
                async with chat_span(
                    model=active_model.name,
                    provider=provider.name,
                    temperature=self._wire_temperature(active_model, provider),
                    max_tokens=max_tokens,
                    thinking=active_model.thinking,
                ) as _span:
                    start_time = time.perf_counter()
                    # Accumulate streamed deltas in O(n) instead of folding with
                    # LLMChunk.__add__ per chunk (which re-concatenates the whole
                    # message every delta -> O(n^2) over a response).
                    chunk_acc = LLMChunkAccumulator()
                    extra_headers, turn_state_sink = self._codex_routing(provider)
                    stream_tracer.stream_started(self)
                    async for chunk in self.backend.complete_streaming(
                        CompletionRequest(
                            model=active_model,
                            messages=self._messages_for_backend(active_model),
                            temperature=active_model.temperature,
                            tools=available_tools,
                            tool_choice=tool_choice,
                            extra_headers=extra_headers,
                            max_tokens=max_tokens,
                            metadata=backend_metadata.model_dump(exclude_none=True),
                            response_format=self._response_format,
                        ),
                        response_headers_sink=turn_state_sink,
                    ):
                        stream_tracer.chunk_received(self)
                        if chunk.correlation_id:
                            self.telemetry_client.last_correlation_id = (
                                chunk.correlation_id
                            )
                        processed_chunk = LLMChunk(
                            message=self.format_handler.process_api_response_message(
                                chunk.message
                            ),
                            usage=chunk.usage,
                            stop=chunk.stop,
                        )
                        chunk_acc.add(processed_chunk)
                        yield processed_chunk
                    end_time = time.perf_counter()
                    self._capture_codex_turn_state(turn_state_sink)
                    self._capture_rate_limits(provider, turn_state_sink)

                    chunk_agg = chunk_acc.build()
                    if chunk_agg is None or chunk_agg.usage is None:
                        raise AgentLoopLLMResponseError(
                            "Usage data missing in final chunk of streamed completion"
                        )
                    # Reject a degenerate no-op response (no content, tool calls,
                    # or reasoning) so it is re-requested below rather than
                    # silently ending the turn producing nothing. A degenerate
                    # response yields inert empty chunks upstream, so the retry
                    # with a fresh accumulator is clean.
                    degenerate_reason = _degenerate_response_reason(chunk_agg)
                    if degenerate_reason is not None:
                        raise InvalidStreamError(degenerate_reason)
                    self._update_stats(
                        usage=chunk_acc.usage,
                        time_seconds=end_time - start_time,
                        provider=provider,
                        model=active_model,
                    )
                    set_usage(_span, chunk_acc.usage)
                    set_finish_reason(
                        _span, chunk_agg.stop.reason if chunk_agg.stop else None
                    )
                    self._capture_chat_content(
                        _span, chunk_agg.message, last_user_message
                    )

                # Raise before committing the truncated turn so the escalation
                # retry re-streams from a clean message list (mirrors _chat).
                if chunk_agg.stop and chunk_agg.stop.is_truncated:
                    raise ResponseTooLongError(provider.name, active_model.name)
                self.messages.append(chunk_agg.message)
                if chunk_agg.stop and chunk_agg.stop.is_refusal:
                    raise _refusal_error(provider.name, active_model.name, chunk_agg)
                return

            except InvalidStreamError as e:
                if attempt < _STREAM_DEGENERATE_RETRIES - 1:
                    logger.warning(
                        "Degenerate streamed response (%s); re-requesting stream "
                        "attempt %d/%d",
                        e.reason,
                        attempt + 1,
                        _STREAM_DEGENERATE_RETRIES,
                    )
                    continue
                raise
            except Exception as e:
                _raise_for_backend_error(e, provider.name, active_model.name)

    def _update_stats(
        self,
        usage: LLMUsage,
        time_seconds: float,
        *,
        provider: ProviderConfig,
        model: ModelConfig,
        harness: bool = False,
    ) -> None:
        self.stats.last_turn_duration = time_seconds
        self.stats.last_turn_prompt_tokens = usage.prompt_tokens
        self.stats.last_turn_completion_tokens = usage.completion_tokens
        self.stats.session_prompt_tokens += usage.prompt_tokens
        self.stats.session_completion_tokens += usage.completion_tokens
        self.stats.last_turn_cached_tokens = usage.cached_tokens
        self.stats.session_cached_tokens += usage.cached_tokens
        self.stats.context_tokens = usage.prompt_tokens + usage.completion_tokens
        if time_seconds > 0 and usage.completion_tokens > 0:
            self.stats.tokens_per_second = usage.completion_tokens / time_seconds

        # Persist the call for cross-session usage windows (/status). Best-effort:
        # a recorder failure never affects the turn. Cost precedence: a user's
        # explicit per-model config prices win; otherwise the built-in pricing
        # table supplies verified rates; both absent → cost_usd=0 (card shows —).
        if model.input_price > 0 or model.output_price > 0:
            cost = (
                usage.prompt_tokens * model.input_price
                + usage.completion_tokens * model.output_price
            ) / 1_000_000
        else:
            pricing = lookup_pricing(model.name)
            if pricing is not None:
                cost = compute_cost(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                    pricing=pricing,
                )
            else:
                cost = 0.0
        self._usage_recorder.record(
            UsageRecord.from_usage(
                timestamp=time.time(),
                provider=provider.name,
                model=model.name,
                usage=usage,
                cost_usd=cost,
                duration_s=time_seconds,
                session_id=self.session_id,
                harness=harness,
            )
        )
