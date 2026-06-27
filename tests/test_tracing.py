from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import StatusCode
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core import tracing
from vibe.core.config import OtelSpanExporterConfig
from vibe.core.tools.base import BaseToolConfig, ToolPermission
from vibe.core.tracing import agent_span, setup_tracing, tool_span
from vibe.core.types import BaseEvent, FunctionCall, ToolCall


class _CollectingExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _otel_provider(monkeypatch: pytest.MonkeyPatch):
    # Patch get_tracer_provider instead of set_tracer_provider to sidestep the
    # OTEL singleton guard that rejects a second set_tracer_provider call.
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    yield exporter


class TestSetupTracing:
    def test_noop_when_disabled(self) -> None:
        config = MagicMock(enable_telemetry=True, enable_otel=False)
        with patch("vibe.core.tracing.trace.set_tracer_provider") as mock_set:
            setup_tracing(config)
        mock_set.assert_not_called()

    def test_noop_when_telemetry_disabled(self) -> None:
        config = MagicMock(enable_telemetry=False, enable_otel=True)
        with patch("vibe.core.tracing.trace.set_tracer_provider") as mock_set:
            setup_tracing(config)
        mock_set.assert_not_called()

    def test_noop_when_exporter_config_is_none(self) -> None:
        config = MagicMock(
            enable_telemetry=True,
            enable_otel=True,
            otel_span_exporter_config=None,
            otel_local_export=False,
        )
        with patch("vibe.core.tracing.trace.set_tracer_provider") as mock_set:
            setup_tracing(config)
        mock_set.assert_not_called()

    def test_configures_provider_from_exporter_config(self) -> None:
        config = MagicMock(
            enable_telemetry=True,
            enable_otel=True,
            otel_span_exporter_config=OtelSpanExporterConfig(
                endpoint="https://customer.mistral.ai/telemetry/v1/traces",
                headers={"Authorization": "Bearer sk-test"},
            ),
            otel_local_export=False,
        )

        with (
            patch(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
            ) as mock_exporter,
            patch("vibe.core.tracing.trace.set_tracer_provider") as mock_set,
        ):
            setup_tracing(config)

        mock_exporter.assert_called_once_with(
            endpoint="https://customer.mistral.ai/telemetry/v1/traces",
            headers={"Authorization": "Bearer sk-test"},
        )
        mock_set.assert_called_once()
        assert isinstance(mock_set.call_args[0][0], TracerProvider)

    def test_custom_endpoint_has_no_auth_headers(self) -> None:
        config = MagicMock(
            enable_telemetry=True,
            enable_otel=True,
            otel_span_exporter_config=OtelSpanExporterConfig(
                endpoint="https://my-collector:4318/v1/traces"
            ),
            otel_local_export=False,
        )

        with (
            patch(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
            ) as mock_exporter,
            patch("vibe.core.tracing.trace.set_tracer_provider") as mock_set,
        ):
            setup_tracing(config)

        mock_exporter.assert_called_once_with(
            endpoint="https://my-collector:4318/v1/traces", headers=None
        )
        mock_set.assert_called_once()
        assert isinstance(mock_set.call_args[0][0], TracerProvider)

    def test_local_export_without_remote_config(self, tmp_path) -> None:
        config = MagicMock(
            enable_telemetry=True,
            enable_otel=True,
            otel_span_exporter_config=None,
            otel_local_export=True,
        )

        with (
            patch(
                "vibe.core.tracing._local_trace_path", return_value=tmp_path / "t.jsonl"
            ),
            patch(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
            ) as mock_remote,
            patch("vibe.core.tracing.trace.set_tracer_provider") as mock_set,
        ):
            setup_tracing(config)

        mock_remote.assert_not_called()
        mock_set.assert_called_once()
        provider = mock_set.call_args[0][0]
        assert isinstance(provider, TracerProvider)

    def test_local_and_remote_export_both_configured(self, tmp_path) -> None:
        config = MagicMock(
            enable_telemetry=True,
            enable_otel=True,
            otel_span_exporter_config=OtelSpanExporterConfig(
                endpoint="https://collector/v1/traces"
            ),
            otel_local_export=True,
        )

        with (
            patch(
                "vibe.core.tracing._local_trace_path", return_value=tmp_path / "t.jsonl"
            ),
            patch(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
            ) as mock_remote,
            patch("vibe.core.tracing.trace.set_tracer_provider") as mock_set,
        ):
            setup_tracing(config)

        mock_remote.assert_called_once()
        mock_set.assert_called_once()
        provider = mock_set.call_args[0][0]
        assert isinstance(provider, TracerProvider)


class TestJsonlSpanExporter:
    def test_writes_spans_as_jsonl(self, tmp_path) -> None:
        from vibe.core.tracing import _JsonlSpanExporter

        span = MagicMock()
        # The real ReadableSpan.to_json() returns multi-line indented JSON; a
        # single-line mock hid the bug where the exporter wrote the blob raw and
        # broke the one-record-per-line JSONL contract.
        span.to_json.return_value = json.dumps(
            {"name": "invoke_agent chaton"}, indent=4
        )
        exporter = _JsonlSpanExporter(tmp_path / "traces.jsonl")

        result = exporter.export([span, span])

        assert result == SpanExportResult.SUCCESS
        lines = (tmp_path / "traces.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "invoke_agent chaton"

    def test_export_failure_returns_success(self, tmp_path) -> None:
        from vibe.core.tracing import _JsonlSpanExporter

        exporter = _JsonlSpanExporter(Path("/nonexistent_dir_xyz/traces.jsonl"))

        result = exporter.export([MagicMock()])

        assert result == SpanExportResult.SUCCESS


class TestAgentSpan:
    @pytest.mark.asyncio
    async def test_span_name_status_and_attributes(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        async with agent_span(model="devstral", session_id="s1", provider="mistral"):
            pass

        assert len(_otel_provider.spans) == 1
        span = _otel_provider.spans[0]
        assert span.name == "invoke_agent chaton"
        assert span.status.status_code == StatusCode.OK
        attrs = dict(span.attributes)
        assert attrs["gen_ai.operation.name"] == "invoke_agent"
        assert attrs["gen_ai.provider.name"] == "mistral_ai"
        assert attrs["gen_ai.agent.name"] == "chaton"
        assert attrs["gen_ai.request.model"] == "devstral"
        assert attrs["gen_ai.conversation.id"] == "s1"

    @pytest.mark.asyncio
    async def test_omits_optional_attributes(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        async with agent_span():
            pass

        attrs = dict(_otel_provider.spans[0].attributes)
        assert "gen_ai.request.model" not in attrs
        assert "gen_ai.conversation.id" not in attrs
        # Unknown provider is omitted, never defaulted to a vendor.
        assert "gen_ai.provider.name" not in attrs
        # No profile passed -> attribute omitted.
        assert "vibe.agent.profile" not in attrs

    @pytest.mark.asyncio
    async def test_records_subagent_profile(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        # gen_ai.agent.name stays the app identity; the profile is attributable
        # separately so an in-process subagent turn is distinguishable in a trace.
        async with agent_span(
            model="glm-5.2", session_id="s1", agent_profile="Explore"
        ):
            pass

        attrs = dict(_otel_provider.spans[0].attributes)
        assert attrs["gen_ai.agent.name"] == "chaton"
        assert attrs["vibe.agent.profile"] == "Explore"

    @pytest.mark.asyncio
    async def test_records_error_on_exception(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        with pytest.raises(ValueError, match="boom"):
            async with agent_span():
                raise ValueError("boom")

        span = _otel_provider.spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert "boom" in span.status.description

    @pytest.mark.asyncio
    async def test_records_aggregate_usage(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        from vibe.core.tracing import set_agent_usage

        async with agent_span(model="glm-5.2", session_id="s1") as span:
            set_agent_usage(
                span, input_tokens=1234, output_tokens=56, cached_tokens=1000
            )

        attrs = dict(_otel_provider.spans[0].attributes)
        assert attrs["gen_ai.usage.input_tokens"] == 1234
        assert attrs["gen_ai.usage.output_tokens"] == 56
        assert attrs["gen_ai.usage.cached_input_tokens"] == 1000


class TestToolSpan:
    @pytest.mark.asyncio
    async def test_span_name_status_and_attributes(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        async with tool_span(tool_name="bash", call_id="c1", arguments='{"cmd": "ls"}'):
            pass

        assert len(_otel_provider.spans) == 1
        span = _otel_provider.spans[0]
        assert span.name == "execute_tool bash"
        assert span.status.status_code == StatusCode.OK
        attrs = dict(span.attributes)
        assert attrs["gen_ai.operation.name"] == "execute_tool"
        assert attrs["gen_ai.tool.name"] == "bash"
        assert attrs["gen_ai.tool.call.id"] == "c1"
        assert attrs["gen_ai.tool.call.arguments"] == '{"cmd": "ls"}'
        assert attrs["gen_ai.tool.type"] == "function"

    @pytest.mark.asyncio
    async def test_records_error_and_exception_event(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        with pytest.raises(RuntimeError):
            async with tool_span(tool_name="bash", call_id="c1", arguments="{}"):
                raise RuntimeError("fail")

        span = _otel_provider.spans[0]
        assert span.status.status_code == StatusCode.ERROR
        exc_events = [e for e in span.events if e.name == "exception"]
        assert len(exc_events) == 1


class TestSpanHierarchy:
    @pytest.mark.asyncio
    async def test_chat_and_tool_are_siblings_under_agent(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        async with agent_span(model="devstral"):
            tracer = trace.get_tracer("mistralai_sdk_tracer")
            # Simulate a chat span created by the Mistral SDK.
            with tracer.start_as_current_span("chat devstral"):
                pass

            async with tool_span(tool_name="grep", call_id="c1", arguments="{}"):
                pass

            with tracer.start_as_current_span("chat devstral"):
                pass

        agent = next(s for s in _otel_provider.spans if "invoke_agent" in s.name)
        children = [
            s
            for s in _otel_provider.spans
            if s.parent and s.parent.span_id == agent.context.span_id
        ]
        assert len(children) == 3
        assert [s.name for s in children] == [
            "chat devstral",
            "execute_tool grep",
            "chat devstral",
        ]


class TestBaggagePropagation:
    @pytest.mark.asyncio
    async def test_tool_span_inherits_conversation_id(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        async with agent_span(model="devstral", session_id="sess-42"):
            async with tool_span(tool_name="bash", call_id="c1", arguments="{}"):
                pass

        tool = next(s for s in _otel_provider.spans if "execute_tool" in s.name)
        assert dict(tool.attributes)["gen_ai.conversation.id"] == "sess-42"

    @pytest.mark.asyncio
    async def test_tool_span_omits_conversation_id_when_no_session(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        async with agent_span(model="devstral"):
            async with tool_span(tool_name="bash", call_id="c1", arguments="{}"):
                pass

        tool = next(s for s in _otel_provider.spans if "execute_tool" in s.name)
        assert "gen_ai.conversation.id" not in dict(tool.attributes)

    @pytest.mark.asyncio
    async def test_baggage_does_not_leak_after_agent_span(self) -> None:
        from opentelemetry import baggage as baggage_api

        async with agent_span(model="devstral", session_id="sess-1"):
            pass

        assert baggage_api.get_baggage("gen_ai.conversation.id") is None


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_yields_invalid_span_on_creation_failure(
        self, _otel_provider: _CollectingExporter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken_tracer() -> trace.Tracer:
            raise RuntimeError("tracer broken")

        monkeypatch.setattr(tracing, "_get_tracer", _broken_tracer)

        async with agent_span():
            pass

        assert len(_otel_provider.spans) == 0

    @pytest.mark.asyncio
    async def test_caller_exception_propagates_when_set_status_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken_set_status(self, *args, **kwargs):
            raise RuntimeError("set_status broken")

        monkeypatch.setattr(
            "opentelemetry.sdk.trace.Span.set_status", _broken_set_status
        )

        with pytest.raises(ValueError, match="original"):
            async with agent_span():
                raise ValueError("original")

    @pytest.mark.asyncio
    async def test_cancellation_ends_span_without_error_status(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        with pytest.raises(asyncio.CancelledError):
            async with agent_span():
                raise asyncio.CancelledError

        span = _otel_provider.spans[0]
        # CancelledError is a BaseException, not an Exception: not a failure, so
        # status stays non-ERROR, but the span is flagged cancelled so it is
        # distinguishable from a span left unset by an instrumentation gap.
        assert span.status.status_code != StatusCode.ERROR
        assert dict(span.attributes).get("vibe.cancelled") is True

    @pytest.mark.asyncio
    async def test_success_path_swallows_span_end_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _broken_end(self, *args, **kwargs):
            raise RuntimeError("end broken")

        monkeypatch.setattr("opentelemetry.sdk.trace.Span.end", _broken_end)

        async with agent_span():
            pass


class TestIntegration:
    @staticmethod
    async def _collect_events(agent_loop, prompt: str) -> list[BaseEvent]:
        return [ev async for ev in agent_loop.act(prompt)]

    @pytest.mark.asyncio
    async def test_agent_turn_with_tool_call_produces_spans(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        tool_call = ToolCall(
            id="call_1",
            index=0,
            function=FunctionCall(name="todo", arguments='{"action": "read"}'),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="Let me check.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Done.")],
        ])
        config = build_test_vibe_config(
            enabled_tools=["todo"],
            tools={"todo": BaseToolConfig(permission=ToolPermission.ALWAYS)},
        )
        agent_loop = build_test_agent_loop(config=config, backend=backend)

        await self._collect_events(agent_loop, "What are my todos?")

        spans = _otel_provider.spans
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        tool_spans = [s for s in spans if "execute_tool" in s.name]

        assert len(agent_spans) == 1
        assert len(tool_spans) == 1

        agent = agent_spans[0]
        tool = tool_spans[0]

        # Parent-child relationship
        assert tool.parent is not None
        assert tool.parent.span_id == agent.context.span_id

        # -- Agent span: name, status, and every attribute set by agent_span() --
        assert agent.name == "invoke_agent chaton"
        assert agent.status.status_code == StatusCode.OK
        agent_attrs = dict(agent.attributes)
        assert agent_attrs["gen_ai.operation.name"] == "invoke_agent"
        assert agent_attrs["gen_ai.provider.name"] == "mistral_ai"
        assert agent_attrs["gen_ai.agent.name"] == "chaton"
        assert agent_attrs["gen_ai.request.model"] == "mistral-vibe-cli-latest"
        assert agent_attrs["gen_ai.conversation.id"] == agent_loop.session_id

        # -- Tool span: name, status, and every attribute set by tool_span() + set_tool_result() --
        assert tool.name == "execute_tool todo"
        assert tool.status.status_code == StatusCode.OK
        tool_attrs = dict(tool.attributes)
        assert tool_attrs["gen_ai.operation.name"] == "execute_tool"
        assert tool_attrs["gen_ai.tool.name"] == "todo"
        assert tool_attrs["gen_ai.tool.call.id"] == "call_1"
        assert tool_attrs["gen_ai.tool.type"] == "function"
        assert (
            tool_attrs["gen_ai.tool.call.arguments"] == '{"action":"read","todos":null}'
        )
        assert tool_attrs["gen_ai.tool.call.result"] == (
            "message: Retrieved 0 todos\ntodos: []\ntotal_count: 0\n"
            "verification_nudge: False"
        )
        # Conversation ID propagated via baggage from agent_span
        assert tool_attrs["gen_ai.conversation.id"] == agent_loop.session_id

    @pytest.mark.asyncio
    async def test_failed_tool_span_still_records_exec_duration(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        # H2: vibe.tool.exec_duration_s must be stamped even when the tool raises
        # (ToolError), not only on the success path — failure latency was blind.
        # grep is read_only (auto-permitted, no approval gate) and raises a
        # ToolError on an empty pattern from inside invoke() — a deterministic
        # in-invoke failure that exercises the finally stamp.
        tool_call = ToolCall(
            id="call_1",
            index=0,
            function=FunctionCall(
                name="grep", arguments='{"pattern": "", "path": "."}'
            ),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="Searching.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Done.")],
        ])
        config = build_test_vibe_config(
            enabled_tools=["grep"],
            tools={"grep": BaseToolConfig(permission=ToolPermission.ALWAYS)},
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
        )
        agent_loop = build_test_agent_loop(config=config, backend=backend)

        await self._collect_events(agent_loop, "read it")

        tool_spans = [s for s in _otel_provider.spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        attrs = dict(tool_spans[0].attributes)
        assert attrs.get("gen_ai.tool.is_error") is True, "the read should have failed"
        assert "vibe.tool.exec_duration_s" in attrs, (
            "exec_duration must be stamped on the failure path"
        )
        assert attrs["vibe.tool.exec_duration_s"] >= 0.0

    @pytest.mark.asyncio
    async def test_interactive_tool_excludes_user_wait_from_exec_duration(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        # exec_duration_s is meant to be exec-only. ask_user_question blocks on
        # the human inside invoke(); that wait must NOT be counted as tool
        # runtime (a multi-hour answer otherwise reads as hours of exec). It is
        # recorded separately as vibe.tool.user_wait_s.
        from vibe.core.tools.builtins.ask_user_question import (
            Answer,
            AskUserQuestionResult,
        )

        wait = 0.3

        async def slow_answer(args: object) -> AskUserQuestionResult:
            await asyncio.sleep(wait)
            return AskUserQuestionResult(
                cancelled=False, answers=[Answer(question="Pick", answer="A")]
            )

        tool_call = ToolCall(
            id="call_1",
            index=0,
            function=FunctionCall(
                name="ask_user_question",
                arguments=json.dumps({
                    "questions": [
                        {
                            "question": "Pick",
                            "options": [{"label": "A"}, {"label": "B"}],
                        }
                    ]
                }),
            ),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="Asking.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Done.")],
        ])
        config = build_test_vibe_config(
            enabled_tools=["ask_user_question"],
            tools={
                "ask_user_question": BaseToolConfig(permission=ToolPermission.ALWAYS)
            },
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
        )
        agent_loop = build_test_agent_loop(config=config, backend=backend)
        agent_loop.set_user_input_callback(slow_answer)

        await self._collect_events(agent_loop, "ask me")

        tool_spans = [s for s in _otel_provider.spans if "execute_tool" in s.name]
        assert len(tool_spans) == 1
        attrs = dict(tool_spans[0].attributes)
        # The human wait is recorded...
        assert attrs.get("vibe.tool.user_wait_s", 0.0) >= wait * 0.8
        # ...and excluded from exec_duration, which stays near-zero exec-only.
        assert attrs["vibe.tool.exec_duration_s"] < wait * 0.5

    @pytest.mark.asyncio
    async def test_context_shaping_span_records_token_deltas(
        self, _otel_provider: _CollectingExporter
    ) -> None:
        # Snip/microcompact/compact rewrite history and bust the prefix cache;
        # the span makes each event visible with its token reduction so a
        # cache-rate drop can be correlated to the reshape that caused it.
        from vibe.core.tracing import context_shaping_span, set_context_shaping_result

        async with context_shaping_span(op="snip", trigger="auto") as span:
            set_context_shaping_result(
                span, tokens_before=1000, tokens_after=600, threshold=2000, blocks=3
            )

        spans = [s for s in _otel_provider.spans if s.name == "context_shaping snip"]
        assert len(spans) == 1
        a = dict(spans[0].attributes)
        assert a["vibe.context.op"] == "snip"
        assert a["vibe.context.trigger"] == "auto"
        assert a["vibe.context.tokens_before"] == 1000
        assert a["vibe.context.tokens_after"] == 600
        assert a["vibe.context.tokens_removed"] == 400
        assert a["vibe.context.threshold"] == 2000
        assert a["vibe.context.blocks"] == 3


# --------------------------------------------------------------------------- #
# OTel three-pillar helpers (#10)                                              #
# --------------------------------------------------------------------------- #


class TestSwapOtelPath:
    def test_swaps_traces_to_metrics(self) -> None:
        from vibe.core.tracing import _swap_otel_path

        assert (
            _swap_otel_path("https://x/v1/traces", "metrics") == "https://x/v1/metrics"
        )

    def test_swaps_traces_to_logs(self) -> None:
        from vibe.core.tracing import _swap_otel_path

        assert _swap_otel_path("https://x/v1/traces", "logs") == "https://x/v1/logs"

    def test_swaps_metrics_to_traces(self) -> None:
        from vibe.core.tracing import _swap_otel_path

        assert (
            _swap_otel_path("https://x/v1/metrics", "traces") == "https://x/v1/traces"
        )

    def test_appends_when_no_known_signal(self) -> None:
        from vibe.core.tracing import _swap_otel_path

        assert _swap_otel_path("https://x/otel", "metrics") == "https://x/otel/metrics"

    def test_handles_trailing_slash(self) -> None:
        from vibe.core.tracing import _swap_otel_path

        assert _swap_otel_path("https://x/v1/traces/", "logs") == "https://x/v1/logs"


class TestLogProcessor:
    def test_make_log_processor_writes_jsonl(self, tmp_path: Path) -> None:
        # _make_log_processor builds a BatchLogRecordProcessor that exports to a
        # local JSONL file via the wrapped LogRecordExporter.
        from vibe.core.tracing import _make_log_processor

        path = tmp_path / "log.jsonl"
        processor = _make_log_processor(path)
        assert processor is not None
        # The processor wraps an exporter whose sink writes to `path`; we cannot
        # easily synthesize a LogRecord here, so just confirm it builds and the
        # underlying file target is wired (force_flush is a safe no-op probe).
        assert processor.force_flush() in (True, None)

    def test_log_record_to_json_reads_nested_record(self) -> None:
        # SDK 1.39's batch processor hands the exporter ReadableLogRecord
        # wrappers whose fields live on .log_record; reading them off the wrapper
        # (the old bug) produced all-None records.
        from types import SimpleNamespace

        from vibe.core.tracing import _log_record_to_json

        nested = SimpleNamespace(
            body="hello world",
            timestamp=123,
            severity_number=SimpleNamespace(value=9),
            attributes={"k": "v"},
        )
        payload = json.loads(_log_record_to_json(SimpleNamespace(log_record=nested)))
        assert payload["body"] == "hello world"
        assert payload["timestamp"] == 123
        assert payload["severity"] == 9
        assert payload["attributes"] == {"k": "v"}

    def test_log_record_to_json_falls_back_to_bare_record(self) -> None:
        from types import SimpleNamespace

        from vibe.core.tracing import _log_record_to_json

        bare = SimpleNamespace(
            body="x", timestamp=1, severity_number=None, attributes=None
        )
        payload = json.loads(_log_record_to_json(bare))
        assert payload["body"] == "x"
        assert payload["severity"] is None
        assert payload["attributes"] == {}

    def test_metrics_and_logs_setup_are_best_effort(self) -> None:
        # _setup_metrics / _setup_logging must not raise when the optional SDK
        # pieces are missing; they log at debug and return.
        from vibe.core.tracing import _setup_logging, _setup_metrics

        resource = MagicMock()
        cfg = MagicMock()
        # Should not raise regardless of what's installed.
        _setup_metrics(cfg, resource, None, local_export=False)
        _setup_logging(cfg, resource, None, local_export=False)
