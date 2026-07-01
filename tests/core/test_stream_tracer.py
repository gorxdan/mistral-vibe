from __future__ import annotations

from collections.abc import Iterator
import re

import pytest

from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core import perf_log, stream_tracer
from vibe.core.paths import LOG_DIR


class _FakeTime:
    def __init__(self) -> None:
        self.now = 10.0

    def monotonic(self) -> float:
        return self.now


def _reset_state() -> None:
    stream_tracer._enabled = None
    stream_tracer._turn = None
    for handler in list(stream_tracer._perf_log.handlers):
        stream_tracer._perf_log.removeHandler(handler)
        handler.close()
    if perf_log._HANDLER is not None:
        perf_log._HANDLER.close()
    perf_log._HANDLER = None


@pytest.fixture(autouse=True)
def _reset_tracer() -> Iterator[None]:
    # Reset before too: an earlier act() test in this worker may have cached
    # _enabled=False (env unset) or a handler bound to a dead tmp LOG_DIR.
    _reset_state()
    yield
    _reset_state()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeTime:
    fake = _FakeTime()
    monkeypatch.setattr(stream_tracer, "time", fake)
    return fake


def _perf_log_text() -> str:
    files = list(LOG_DIR.path.glob("vibe-perf-*.log"))
    assert len(files) == 1
    return files[0].read_text(encoding="utf-8")


def test_disabled_probes_are_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBE_TRACE_STREAM", raising=False)
    owner = object()
    stream_tracer.turn_started(owner, "t-1")
    assert stream_tracer._turn is None
    stream_tracer.stream_started(owner)
    stream_tracer.chunk_received(owner)
    stream_tracer.assistant_rendered()
    stream_tracer.turn_finished(owner)
    assert not stream_tracer._perf_log.handlers
    assert not LOG_DIR.path.exists()


def test_enabled_emits_one_summary_line(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeTime
) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    owner = object()
    stream_tracer.turn_started(owner, "abcd1234-3")
    stream_tracer.stream_started(owner)
    clock.now = 10.5
    stream_tracer.chunk_received(owner)
    clock.now = 10.6
    stream_tracer.chunk_received(owner)
    clock.now = 10.65
    stream_tracer.assistant_rendered()
    clock.now = 10.7
    stream_tracer.assistant_rendered()
    clock.now = 11.4
    stream_tracer.chunk_received(owner)
    stream_tracer.turn_finished(owner)

    text = _perf_log_text()
    assert text.count("perf stream:") == 1
    assert "turn=abcd1234-3" in text
    assert "ttfb=500ms" in text
    assert "ttfr=650ms" in text
    assert "max_gap=800ms" in text
    assert "slow_gaps=1" in text
    assert "chunks=3" in text


def test_missing_chunks_and_render_reported_as_dash(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeTime
) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    owner = object()
    stream_tracer.turn_started(owner, "t-1")
    stream_tracer.turn_finished(owner)
    text = _perf_log_text()
    assert "ttfb=- " in text
    assert "ttfr=- " in text
    assert "chunks=0" in text


def test_gap_reference_resets_between_streams(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeTime
) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    owner = object()
    stream_tracer.turn_started(owner, "t-1")
    stream_tracer.stream_started(owner)
    stream_tracer.chunk_received(owner)
    clock.now = 10.1
    stream_tracer.chunk_received(owner)
    # Second LLM call after tool execution: the pause is not an intra-stream gap.
    clock.now = 20.0
    stream_tracer.stream_started(owner)
    stream_tracer.chunk_received(owner)
    stream_tracer.turn_finished(owner)
    text = _perf_log_text()
    assert "max_gap=100ms" in text
    assert "slow_gaps=0" in text
    assert "chunks=3" in text


def test_nested_turn_is_ignored_and_attributed_to_outermost(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeTime
) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    host, subagent = object(), object()
    stream_tracer.turn_started(host, "host-1")
    stream_tracer.turn_started(subagent, "sub-1")
    stream_tracer.stream_started(subagent)
    stream_tracer.chunk_received(subagent)
    stream_tracer.stream_started(host)
    clock.now = 10.2
    stream_tracer.chunk_received(host)
    stream_tracer.turn_finished(subagent)
    assert "perf stream:" not in _perf_log_text()
    stream_tracer.turn_finished(host)
    text = _perf_log_text()
    assert "turn=host-1" in text
    assert "ttfb=200ms" in text
    assert "chunks=1" in text


def test_same_owner_restart_replaces_stale_turn(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeTime
) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    owner = object()
    stream_tracer.turn_started(owner, "t-1")
    clock.now = 30.0
    stream_tracer.turn_started(owner, "t-2")
    stream_tracer.turn_finished(owner)
    text = _perf_log_text()
    assert text.count("perf stream:") == 1
    assert "turn=t-2" in text


@pytest.mark.asyncio
async def test_act_emits_stream_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    backend = FakeBackend(
        chunks=[[mock_llm_chunk(content="hel"), mock_llm_chunk(content="lo")]]
    )
    loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    async for _event in loop.act("hi"):
        pass
    text = _perf_log_text()
    assert text.count("perf stream:") == 1
    assert f"turn={loop.session_id[:8]}-" in text
    assert re.search(r"ttfb=\d+ms", text)
    assert "chunks=2" in text


@pytest.mark.asyncio
async def test_act_when_disabled_writes_no_perf_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIBE_TRACE_STREAM", raising=False)
    monkeypatch.delenv("VIBE_TRACE_LOOP", raising=False)
    backend = FakeBackend(chunks=[[mock_llm_chunk(content="hi")]])
    loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    async for _event in loop.act("hi"):
        pass
    if LOG_DIR.path.exists():
        assert not list(LOG_DIR.path.glob("vibe-perf-*.log"))


def test_fail_soft_when_log_dir_is_a_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    LOG_DIR.path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.path.write_text("not a dir", encoding="utf-8")
    owner = object()
    stream_tracer.turn_started(owner, "t-1")
    assert stream_tracer._turn is None
    assert stream_tracer._enabled is False
    stream_tracer.chunk_received(owner)
    stream_tracer.assistant_rendered()
    stream_tracer.turn_finished(owner)
    assert not stream_tracer._perf_log.handlers


def test_background_subagent_turn_never_claims_the_slot(
    monkeypatch: pytest.MonkeyPatch, clock: _FakeTime
) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    host, subagent = object(), object()
    # In-process BACKGROUND subagent: starts while no host turn is open.
    stream_tracer.turn_started(subagent, "sub-1", is_subagent=True)
    assert stream_tracer._turn is None
    stream_tracer.turn_started(host, "host-1")
    clock.now = 10.3
    stream_tracer.chunk_received(host)
    stream_tracer.assistant_rendered()
    stream_tracer.turn_finished(host)
    text = _perf_log_text()
    assert "turn=host-1" in text
    assert "ttfb=300ms" in text
