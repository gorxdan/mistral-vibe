from __future__ import annotations

import atexit
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
import json
import logging
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

# Map internal backend identifiers to the OTel gen_ai.provider.name well-known
# values where one exists; unknown backends (zai, sakana, ...) pass through as
# free-form strings, which the convention permits.
_PROVIDER_ALIASES = {
    "mistral": gen_ai_attributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "openai-chatgpt": gen_ai_attributes.GenAiProviderNameValues.OPENAI.value,
}


def _normalize_provider(name: str | None) -> str | None:
    # Never default an unknown provider to a vendor: return None so callers omit
    # the attribute rather than misattributing spend/errors to Mistral.
    if not name:
        return None
    return _PROVIDER_ALIASES.get(name, name)


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


class _JsonlLogExporter:
    """Local JSONL sink for OTel logs (mirrors :class:`_JsonlSpanExporter`).

    A plain callable-style exporter (export/shutdown/force_flush). It is wrapped
    by :class:`_OtelLogExporterAdapter` at setup time to satisfy the
    ``LogRecordExporter`` protocol, so this class never imports the optional logs
    SDK at module load. Used so structured logs reach a durable local file
    without requiring the ``opentelemetry-exporter-otlp-proto-http`` log exporter
    subpackage (not currently a dependency). Each record is one JSON line:
    timestamp, severity, body, and attributes.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def export(self, batch: Sequence[Any]) -> object:
        from opentelemetry.sdk._logs.export import LogRecordExportResult

        try:
            with self._path.open("a", encoding="utf-8") as f:
                for log in batch:
                    f.write(_log_record_to_json(log) + "\n")
        except Exception:
            logger.warning("Failed to write log to %s", self._path, exc_info=True)
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _log_record_to_json(log: Any) -> str:
    # The batch processor hands us ReadableLogRecord wrappers (SDK 1.39) whose
    # fields live on the nested .log_record, not on the wrapper itself; reading
    # them off the wrapper yields all-None records. Fall back to the object
    # itself for exporters/tests that pass a bare LogRecord.
    record = getattr(log, "log_record", log)
    body = getattr(record, "body", None)
    severity = getattr(record, "severity_number", None)
    payload: dict[str, Any] = {
        "timestamp": getattr(record, "timestamp", None),
        "severity": getattr(severity, "value", severity),
        "body": body if isinstance(body, str) else str(body),
        "attributes": dict(getattr(record, "attributes", {}) or {}),
    }
    return json.dumps(payload, default=str)


def _make_log_processor(path: Path) -> Any:
    """Build a BatchLogRecordProcessor wrapping a local-JSONL LogRecordExporter.

    Defined lazily (only called from _setup_logging, which already imported the
    logs SDK) so the LogRecordExporter base resolves at runtime, not at module
    import — keeping the logs pillar optional.
    """
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        LogRecordExporter,
        LogRecordExportResult,
    )

    class _LocalLogExporter(LogRecordExporter):
        def __init__(self) -> None:
            self._sink = _JsonlLogExporter(path)

        def export(self, batch: Sequence[Any]) -> LogRecordExportResult:  # type: ignore[override]
            return self._sink.export(batch)  # type: ignore[return-value]

        def shutdown(self) -> None:
            self._sink.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return self._sink.force_flush(timeout_millis)

    return BatchLogRecordProcessor(_LocalLogExporter())


def _local_trace_path(prefix: str) -> Path:
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    TRACE_LOG_DIR.path.mkdir(parents=True, exist_ok=True)
    return TRACE_LOG_DIR.path / f"{prefix}_{ts}_{os.getpid()}.jsonl"


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
            SimpleSpanProcessor(_JsonlSpanExporter(_local_trace_path("trace")))
        )

    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)

    # Pillar 2 (metrics) + Pillar 3 (logs): best-effort, never fatal to startup.
    _setup_metrics(config, resource, exporter_cfg, local_export)
    _setup_logging(config, resource, exporter_cfg, local_export)

    # W3C TraceContext propagation (traceparent + tracestate) so descendant
    # spans — including those in the Mistral SDK — inherit the full W3C
    # context, not just baggage.
    try:
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        context.attach(TraceContextTextMapPropagator().extract(context.get_current()))
    except Exception:
        logger.debug("W3C TraceContext propagator unavailable; using default")


def _setup_metrics(
    config: VibeConfig, resource: Any, exporter_cfg: Any, local_export: bool
) -> None:
    """Install a MeterProvider when the metric SDK + exporter are importable."""
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    except ImportError:
        logger.debug("OTel metric exporter unavailable; metrics pillar skipped")
        return

    readers: list[Any] = []
    if exporter_cfg is not None:
        metric_endpoint = _swap_otel_path(exporter_cfg.endpoint, "metrics")
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=metric_endpoint, headers=exporter_cfg.headers
                )
            )
        )
    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    from opentelemetry import metrics

    metrics.set_meter_provider(meter_provider)
    atexit.register(meter_provider.shutdown)


def _setup_logging(
    config: VibeConfig, resource: Any, exporter_cfg: Any, local_export: bool
) -> None:
    """Install a LoggerProvider bridging stdlib logging.

    Exports structured logs locally (JSONL). The OTLP log exporter is optional —
    it lights up if ``opentelemetry-exporter-otlp-proto-http`` ships a log
    exporter in a future version, but is not a hard dependency today.
    """
    try:
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    except ImportError:
        logger.debug("OTel logging SDK unavailable; logs pillar skipped")
        return

    log_provider = LoggerProvider(resource=resource)

    # Local JSONL export mirrors the span path: durable, no collector needed.
    if local_export:
        log_provider.add_log_record_processor(
            _make_log_processor(_local_trace_path("log"))
        )

    # An OTLP log exporter would slot in here once the
    # ``opentelemetry-exporter-otlp-proto-http`` log subpackage is a dependency;
    # it is intentionally not a hard requirement today, so local logs are the
    # default export path.

    from opentelemetry import _logs

    _logs.set_logger_provider(log_provider)

    # Bridge stdlib logging so vibe's logger output reaches the OTel log stream.
    handler = LoggingHandler(logger_provider=log_provider)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    if not any(isinstance(h, LoggingHandler) for h in root.handlers):
        root.addHandler(handler)
    atexit.register(log_provider.shutdown)


def _swap_otel_path(endpoint: str, signal: str) -> str:
    """Replace the trailing OTLP signal path (traces -> metrics|logs).

    OTLP HTTP endpoints end in a signal segment (``.../v1/traces``); swap it to
    the requested signal. Falls back to appending when the pattern isn't found.
    """
    for seg in ("traces", "metrics", "logs"):
        if endpoint.rstrip("/").endswith(f"/{seg}"):
            return endpoint.rstrip("/")[: -len(seg)] + signal
    return f"{endpoint.rstrip('/')}/{signal}"


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
            else:
                # BaseException-but-not-Exception: CancelledError / GeneratorExit
                # / KeyboardInterrupt. Not a failure, so leave status non-ERROR
                # (don't inflate error rates on interrupt/shutdown), but flag it
                # so a cancelled span is distinguishable from one left unset by an
                # instrumentation gap.
                span.set_attribute("vibe.cancelled", True)
        except Exception:
            logger.warning("Failed to record span status", exc_info=True)
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                logger.warning("Failed to end span", exc_info=True)


@asynccontextmanager
async def agent_span(
    *,
    model: str | None = None,
    session_id: str | None = None,
    provider: str | None = None,
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.INVOKE_AGENT.value,
        gen_ai_attributes.GEN_AI_AGENT_NAME: VIBE_AGENT_NAME,
    }
    if (prov := _normalize_provider(provider)) is not None:
        attributes[gen_ai_attributes.GEN_AI_PROVIDER_NAME] = prov
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
async def chat_span(
    *, model: str | None = None, provider: str | None = None
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.CHAT.value,
    }
    if (prov := _normalize_provider(provider)) is not None:
        attributes[gen_ai_attributes.GEN_AI_PROVIDER_NAME] = prov
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


def set_tool_error(span: trace.Span, message: str) -> None:
    # Tool failures are caught in the agent loop and never reach _safe_span, so
    # the span would stay OK; flag it ERROR so failures are queryable.
    try:
        span.set_status(StatusCode.ERROR, message)
        span.set_attribute("gen_ai.tool.is_error", True)
    except Exception:
        pass


def set_tool_exec_duration(span: trace.Span, seconds: float) -> None:
    # The execute_tool span also wraps the approval gate + hooks; record the
    # exec-only interval so the span lifetime is not misread as command runtime.
    try:
        span.set_attribute("vibe.tool.exec_duration_s", seconds)
    except Exception:
        pass
