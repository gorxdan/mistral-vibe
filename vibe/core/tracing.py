from __future__ import annotations

import atexit
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import StatusCode

from vibe import __version__

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from vibe.core.config import VibeConfig
    from vibe.core.types import LLMUsage

from vibe.core.logger import logger
from vibe.core.paths import TRACE_LOG_DIR
from vibe.core.utils import utc_now

VIBE_TRACER_NAME = "chaton"
VIBE_AGENT_NAME = "chaton"


class _JsonlSpanExporter(SpanExporter):
    """Appends each ended span as one JSON line to a local file.

    Used with :class:`~opentelemetry.sdk.trace.export.SimpleSpanProcessor` so
    spans are written immediately on end — durable across crashes, suitable for
    local debugging without an external collector.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for span in spans:
                    f.write(span.to_json() + "\n")
        except Exception:
            logger.warning("Failed to write span to %s", self._path, exc_info=True)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _local_trace_path() -> Path:
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    TRACE_LOG_DIR.path.mkdir(parents=True, exist_ok=True)
    return TRACE_LOG_DIR.path / f"trace_{ts}_{os.getpid()}.jsonl"


def setup_tracing(config: VibeConfig) -> None:
    if not config.enable_telemetry or not config.enable_otel:
        return

    exporter_cfg = config.otel_span_exporter_config
    local_export = config.otel_local_export

    if exporter_cfg is None and not local_export:
        return

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    resource = Resource.create({
        "service.name": VIBE_AGENT_NAME,
        "service.version": __version__,
    })
    provider = TracerProvider(resource=resource)

    if exporter_cfg is not None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(**exporter_cfg.model_dump())
        provider.add_span_processor(BatchSpanProcessor(exporter))

    if local_export:
        provider.add_span_processor(
            SimpleSpanProcessor(_JsonlSpanExporter(_local_trace_path()))
        )

    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)


def _get_tracer() -> trace.Tracer:
    return trace.get_tracer(VIBE_TRACER_NAME, __version__)


@asynccontextmanager
async def _safe_span(
    name: str, attributes: dict[str, Any]
) -> AsyncGenerator[trace.Span]:
    # Tracing errors are logged, never raised.
    try:
        tracer = _get_tracer()
        cm = tracer.start_as_current_span(name, attributes=attributes)
        span = cm.__enter__()
    except Exception:
        logger.warning("Failed to create span", exc_info=True)
        yield trace.INVALID_SPAN
        return

    exc_info: BaseException | None = None
    try:
        yield span
    except BaseException as exc:
        exc_info = exc
        raise
    finally:
        try:
            if isinstance(exc_info, Exception):
                span.set_status(StatusCode.ERROR, str(exc_info))
                span.record_exception(exc_info)
            elif exc_info is None:
                span.set_status(StatusCode.OK)
        except Exception:
            logger.warning("Failed to record span status", exc_info=True)
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.warning("Failed to end span", exc_info=True)


@asynccontextmanager
async def agent_span(
    *, model: str | None = None, session_id: str | None = None
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.INVOKE_AGENT.value,
        gen_ai_attributes.GEN_AI_PROVIDER_NAME: gen_ai_attributes.GenAiProviderNameValues.MISTRAL_AI.value,
        gen_ai_attributes.GEN_AI_AGENT_NAME: VIBE_AGENT_NAME,
    }
    if model:
        attributes[gen_ai_attributes.GEN_AI_REQUEST_MODEL] = model
    if session_id:
        attributes[gen_ai_attributes.GEN_AI_CONVERSATION_ID] = session_id

    # Propagate conversation ID as OTEL baggage so descendant spans — including
    # those created by the Mistral SDK — can read and attach it.
    token = None
    if session_id:
        ctx = baggage.set_baggage(gen_ai_attributes.GEN_AI_CONVERSATION_ID, session_id)
        token = context.attach(ctx)
    try:
        async with _safe_span(f"invoke_agent {VIBE_AGENT_NAME}", attributes) as span:
            yield span
    finally:
        if token is not None:
            context.detach(token)


@asynccontextmanager
async def chat_span(*, model: str | None = None) -> AsyncGenerator[trace.Span]:
    """One LLM inference call (gen_ai `chat`).

    Nested under the loop-level ``invoke_agent`` span so per-call token usage is
    attributable. Usage attributes are attached post-call via :func:`set_usage`.
    """
    attributes: dict[str, Any] = {
        gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.CHAT.value,
        gen_ai_attributes.GEN_AI_PROVIDER_NAME: gen_ai_attributes.GenAiProviderNameValues.MISTRAL_AI.value,
    }
    if model:
        attributes[gen_ai_attributes.GEN_AI_REQUEST_MODEL] = model
    if conv_id := baggage.get_baggage(gen_ai_attributes.GEN_AI_CONVERSATION_ID):
        attributes[gen_ai_attributes.GEN_AI_CONVERSATION_ID] = conv_id

    async with _safe_span(f"chat {VIBE_AGENT_NAME}", attributes) as span:
        yield span


def set_usage(span: trace.Span, usage: LLMUsage) -> None:
    # cached_tokens has no stable gen_ai constant; emit a vibe-namespaced attr.
    try:
        span.set_attribute(
            gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens
        )
        span.set_attribute(
            gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens
        )
        span.set_attribute("gen_ai.usage.cached_input_tokens", usage.cached_tokens)
    except Exception:
        pass


@asynccontextmanager
async def tool_span(
    *, tool_name: str, call_id: str, arguments: str
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.EXECUTE_TOOL.value,
        gen_ai_attributes.GEN_AI_TOOL_NAME: tool_name,
        gen_ai_attributes.GEN_AI_TOOL_CALL_ID: call_id,
        gen_ai_attributes.GEN_AI_TOOL_CALL_ARGUMENTS: arguments,
        gen_ai_attributes.GEN_AI_TOOL_TYPE: "function",
    }
    if conv_id := baggage.get_baggage(gen_ai_attributes.GEN_AI_CONVERSATION_ID):
        attributes[gen_ai_attributes.GEN_AI_CONVERSATION_ID] = conv_id

    async with _safe_span(f"execute_tool {tool_name}", attributes) as span:
        yield span


@asynccontextmanager
async def hook_span(
    *,
    hook_name: str,
    hook_type: str,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        "vibe.hook.name": hook_name,
        "vibe.hook.type": hook_type,
    }
    if tool_name is not None:
        attributes[gen_ai_attributes.GEN_AI_TOOL_NAME] = tool_name
    if tool_call_id is not None:
        attributes[gen_ai_attributes.GEN_AI_TOOL_CALL_ID] = tool_call_id
    if conv_id := baggage.get_baggage(gen_ai_attributes.GEN_AI_CONVERSATION_ID):
        attributes[gen_ai_attributes.GEN_AI_CONVERSATION_ID] = conv_id

    async with _safe_span(f"hook {hook_type} {hook_name}", attributes) as span:
        yield span


def set_tool_result(span: trace.Span, result: str) -> None:
    try:
        span.set_attribute(gen_ai_attributes.GEN_AI_TOOL_CALL_RESULT, result)
    except Exception:
        pass
