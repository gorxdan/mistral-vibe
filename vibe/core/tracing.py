from __future__ import annotations

import atexit
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
import gzip
import json
import logging
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any

from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import StatusCode

from vibe import __version__

if TYPE_CHECKING:
    from opentelemetry.sdk._logs.export import LogRecordExportResult
    from opentelemetry.sdk.trace import ReadableSpan

    from vibe.core.config import VibeConfig
    from vibe.core.orchestration import OrchestrationTurnSummary
    from vibe.core.types import LLMUsage

from vibe.core.logger import logger
from vibe.core.paths import TRACE_LOG_DIR
from vibe.core.utils import utc_now

VIBE_TRACER_NAME = "mistral_vibe"
VIBE_AGENT_NAME = "mistral-vibe"

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


def _span_to_jsonl(span: ReadableSpan) -> str:
    # ReadableSpan.to_json() returns multi-line indented JSON (SDK default); writing
    # it raw breaks the one-record-per-line JSONL contract this exporter depends on.
    # Round-trip through json to emit compact single-line JSON per span.
    return json.dumps(json.loads(span.to_json()))


class _JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path) -> None:
        self._path = path

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self._path.open("a", encoding="utf-8") as f:
                for span in spans:
                    f.write(_span_to_jsonl(span) + "\n")
        except Exception:
            logger.warning("Failed to write span to %s", self._path, exc_info=True)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _JsonlLogExporter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def export(self, batch: Sequence[Any]) -> LogRecordExportResult:
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
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        LogRecordExporter,
    )

    class _LocalLogExporter(LogRecordExporter):
        def __init__(self) -> None:
            self._sink = _JsonlLogExporter(path)

        def export(self, batch: Sequence[Any]) -> LogRecordExportResult:
            return self._sink.export(batch)

        def shutdown(self) -> None:
            self._sink.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return self._sink.force_flush(timeout_millis)

    return BatchLogRecordProcessor(_LocalLogExporter())


def _local_trace_path(prefix: str) -> Path:
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    TRACE_LOG_DIR.path.mkdir(parents=True, exist_ok=True)
    return TRACE_LOG_DIR.path / f"{prefix}_{ts}_{os.getpid()}.jsonl"


# Local trace/log JSONL accumulate one pair per process; the dir grew unbounded
# (hundreds of MB). Traces are valuable, so keep them all — but compress the cold
# tail: the newest N of each prefix stay hot (uncompressed, fast to read), older
# ones are gzipped in place (~10x smaller). Nothing is deleted.
_LOCAL_TRACE_HOT = 50


def _archive_old_traces(directory: Path, keep_hot: int = _LOCAL_TRACE_HOT) -> None:
    """Gzip trace/log files beyond the newest *keep_hot* of each prefix, in place.

    Best-effort and synchronous at startup. The newest files (incl. this
    process's, just-created) stay uncompressed; older ``*.jsonl`` become
    ``*.jsonl.gz`` (the data is preserved, just compressed). Already-archived
    files are skipped. Failures never block tracing setup.
    """
    try:
        if not directory.exists():
            return
        for prefix in ("trace_", "log_"):
            files = sorted(
                directory.glob(f"{prefix}*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in files[keep_hot:]:
                gz = old.with_name(old.name + ".gz")
                try:
                    if gz.exists():
                        continue
                    with old.open("rb") as src, gzip.open(gz, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    old.unlink()
                except OSError:
                    pass
    except Exception:
        logger.debug("local trace archive skipped", exc_info=True)


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
        _archive_old_traces(TRACE_LOG_DIR.path)
        provider.add_span_processor(
            SimpleSpanProcessor(_JsonlSpanExporter(_local_trace_path("trace")))
        )

    trace.set_tracer_provider(provider)
    atexit.register(provider.shutdown)

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
    try:
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    except ImportError:
        logger.debug("OTel logging SDK unavailable; logs pillar skipped")
        return

    log_provider = LoggerProvider(resource=resource)

    if local_export:
        log_provider.add_log_record_processor(
            _make_log_processor(_local_trace_path("log"))
        )

    from opentelemetry import _logs

    _logs.set_logger_provider(log_provider)

    handler = LoggingHandler(logger_provider=log_provider)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    if not any(isinstance(h, LoggingHandler) for h in root.handlers):
        root.addHandler(handler)
    atexit.register(log_provider.shutdown)


def _swap_otel_path(endpoint: str, signal: str) -> str:
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
    agent_profile: str | None = None,
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.INVOKE_AGENT.value,
        gen_ai_attributes.GEN_AI_AGENT_NAME: VIBE_AGENT_NAME,
    }
    # gen_ai.agent.name stays the app identity ("mistral-vibe"); the actual profile
    # (host vs a subagent like "Explore"/"worker") rides a separate attribute so
    # in-process subagent turns are attributable in a trace.
    if agent_profile:
        attributes["vibe.agent.profile"] = agent_profile
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
    *,
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking: str | None = None,
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.CHAT.value
    }
    if (prov := _normalize_provider(provider)) is not None:
        attributes[gen_ai_attributes.GEN_AI_PROVIDER_NAME] = prov
    if model:
        attributes[gen_ai_attributes.GEN_AI_REQUEST_MODEL] = model
    # Sampling params, so a trace can be partitioned by them (e.g. a temperature
    # A/B). temperature is recorded only when actually sent — None means the
    # adapter omits it (Moonshot k2.7-code), so absence is the truthful signal.
    if temperature is not None:
        attributes[gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE] = temperature
    if max_tokens is not None:
        attributes[gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
    if thinking:
        attributes["vibe.request.thinking"] = thinking
    if conv_id := baggage.get_baggage(gen_ai_attributes.GEN_AI_CONVERSATION_ID):
        attributes[gen_ai_attributes.GEN_AI_CONVERSATION_ID] = conv_id

    async with _safe_span(f"chat {VIBE_AGENT_NAME}", attributes) as span:
        yield span


def _set_usage_attrs(
    span: trace.Span,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> None:
    # cached/reasoning tokens have no stable gen_ai constant; emit vibe-namespaced.
    try:
        span.set_attribute(gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        span.set_attribute(gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        span.set_attribute("gen_ai.usage.cached_input_tokens", cached_tokens)
        span.set_attribute("gen_ai.usage.cache_write_input_tokens", cache_write_tokens)
        span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning_tokens)
    except Exception:
        pass


def set_usage(span: trace.Span, usage: LLMUsage) -> None:
    _set_usage_attrs(
        span,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


_CONTENT_EVENT_MAX_CHARS = 8_000


def _clip_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    elided = len(text) - max_chars
    return f"{text[:head]}…[{elided} chars elided]…{text[-tail:]}"


def add_message_content_events(
    span: trace.Span,
    *,
    user_text: str | None = None,
    assistant_text: str | None = None,
    reasoning_text: str | None = None,
    tool_call_names: Sequence[str] = (),
    max_chars: int = _CONTENT_EVENT_MAX_CHARS,
) -> None:
    """Attach prompt/response prose to a chat span as events, for recall debugging.

    Opt-in (``config.otel_capture_content``, default off): spans normally carry
    only token counts, never message text — capturing content balloons local
    trace files and records user/source bytes. When enabled, each field is
    middle-clipped to *max_chars* so one huge turn can't dominate the file. Event
    names follow the gen_ai semantic convention.
    """
    try:
        if user_text:
            span.add_event(
                "gen_ai.user.message", {"content": _clip_middle(user_text, max_chars)}
            )
        attrs: dict[str, Any] = {}
        if assistant_text:
            attrs["content"] = _clip_middle(assistant_text, max_chars)
        if reasoning_text:
            attrs["reasoning"] = _clip_middle(reasoning_text, max_chars)
        if tool_call_names:
            attrs["tool_calls"] = ", ".join(tool_call_names)
        if attrs:
            span.add_event("gen_ai.assistant.message", attrs)
    except Exception:
        logger.warning("Failed to attach message content to span", exc_info=True)


def set_finish_reason(span: trace.Span, reason: str | None) -> None:
    # gen_ai.response.finish_reasons is a list; vibe produces one stop per turn
    # (stop/length/tool_calls/refusal). 'length' marks an output-truncated turn.
    if not reason:
        return
    try:
        span.set_attribute(gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS, (reason,))
    except Exception:
        pass


def set_agent_usage(
    span: trace.Span,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> None:
    _set_usage_attrs(
        span,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def set_orchestration_summary(
    span: trace.Span, summary: OrchestrationTurnSummary
) -> None:
    attributes: dict[str, str | int | bool] = {
        "vibe.orchestration.state": summary.state.value,
        "vibe.orchestration.task_available": summary.capabilities.task,
        "vibe.orchestration.workflow_available": summary.capabilities.workflow,
        "vibe.orchestration.team_available": summary.capabilities.team,
        "vibe.orchestration.background_delivery": (
            summary.capabilities.background_delivery
        ),
        "vibe.orchestration.reconnaissance_calls": summary.reconnaissance_calls,
        "vibe.orchestration.direct_mutations": summary.direct_mutations,
        "vibe.orchestration.unique_paths": summary.unique_paths,
        "vibe.orchestration.productive_delegations": summary.productive_delegations,
        "vibe.orchestration.completed_delegations": summary.completed_delegations,
        "vibe.orchestration.pending_delegations": summary.pending_delegations,
        "vibe.orchestration.verifier_delegations": summary.verifier_delegations,
        "vibe.orchestration.required_delegations": summary.required_delegations,
        "vibe.orchestration.failed_delegations": summary.failed_delegations,
        "vibe.orchestration.scope_drift": summary.scope_drift,
        "vibe.orchestration.policy_nudges": summary.policy_nudges,
        "vibe.orchestration.user_allows_agents": summary.user_allows_agents,
        "vibe.orchestration.user_allows_workflow": summary.user_allows_workflow,
        "vibe.orchestration.user_allows_team": summary.user_allows_team,
    }
    if summary.route is not None:
        attributes["vibe.orchestration.route"] = summary.route.value
    if summary.reason is not None:
        attributes["vibe.orchestration.reason"] = summary.reason.value
    try:
        for key, value in attributes.items():
            span.set_attribute(key, value)
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


def set_tool_user_wait(span: trace.Span, seconds: float) -> None:
    # Time an interactive tool (ask_user_question, in-tool approval) spent
    # blocked on a human. Excluded from exec_duration so the latter stays
    # exec-only; recorded here so the human-wait is still visible.
    try:
        span.set_attribute("vibe.tool.user_wait_s", seconds)
    except Exception:
        pass


@asynccontextmanager
async def context_shaping_span(
    *, op: str, trigger: str = "auto"
) -> AsyncGenerator[trace.Span]:
    attributes: dict[str, Any] = {
        "vibe.context.op": op,
        "vibe.context.trigger": trigger,
    }
    if conv_id := baggage.get_baggage(gen_ai_attributes.GEN_AI_CONVERSATION_ID):
        attributes[gen_ai_attributes.GEN_AI_CONVERSATION_ID] = conv_id
    async with _safe_span(f"context_shaping {op}", attributes) as span:
        yield span


def set_context_shaping_result(
    span: trace.Span,
    *,
    tokens_before: int,
    tokens_after: int,
    threshold: int | None = None,
    blocks: int | None = None,
    status: str | None = None,
    reasoning_preserved: bool | None = None,
) -> None:
    try:
        span.set_attribute("vibe.context.tokens_before", tokens_before)
        span.set_attribute("vibe.context.tokens_after", tokens_after)
        span.set_attribute(
            "vibe.context.tokens_removed", max(0, tokens_before - tokens_after)
        )
        if threshold is not None:
            span.set_attribute("vibe.context.threshold", threshold)
        if blocks is not None:
            span.set_attribute("vibe.context.blocks", blocks)
        if status is not None:
            span.set_attribute("vibe.context.status", status)
        # Whether elided assistant turns kept reasoning_content (Preserved
        # Thinking). False here is the signal that snip stripped it.
        if reasoning_preserved is not None:
            span.set_attribute("vibe.context.reasoning_preserved", reasoning_preserved)
    except Exception:
        pass
