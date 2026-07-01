from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest

from vibe.cli.textual_ui.handlers.event_handler import EventHandler
from vibe.core import perf_log, stream_tracer
from vibe.core.types import AssistantEvent, ReasoningEvent


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


def _make_handler() -> EventHandler:
    return EventHandler(mount_callback=AsyncMock(), get_tools_collapsed=lambda: False)


@pytest.mark.asyncio
async def test_first_assistant_text_latches_render_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_TRACE_STREAM", "1")
    owner = object()
    stream_tracer.turn_started(owner, "t-1")
    handler = _make_handler()

    await handler.handle_event(ReasoningEvent(content="thinking", message_id="r1"))
    turn = stream_tracer._turn
    assert turn is not None
    assert turn.first_render is None

    await handler.handle_event(AssistantEvent(content="hi", message_id="m1"))
    first = turn.first_render
    assert first is not None

    await handler.handle_event(AssistantEvent(content=" there", message_id="m1"))
    assert turn.first_render == first
