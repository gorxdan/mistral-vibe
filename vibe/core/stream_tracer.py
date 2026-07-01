"""Streaming-responsiveness tracer. Env-gated: ``VIBE_TRACE_STREAM=1``.

The repo's perf verdict is that wall-clock is model-bound (96% prompt-cache
hit), so the one harness-side latency lever is how fast streamed output
reaches the user — and neither ``loop_tracer`` (loop-thread blocking) nor
``profiler`` (cumulative CPU) measures that. This tracer times the boundaries
a user actually feels and emits ONE summary line per turn into the shared
per-PID perf log (``vibe.core.perf_log``)::

    perf stream: turn=<id> ttfb=<ms> ttfr=<ms> max_gap=<ms> slow_gaps=<n> chunks=<n>

- ``ttfb``: user submit (``AgentLoop.act`` entry) -> first ``LLMChunk`` off the
  backend stream.
- ``ttfr``: submit -> first assistant text handed to a mounted TUI widget
  (Textual paints it on the following frame; headless/ACP runs report ``-``).
- ``max_gap`` / ``slow_gaps``: worst inter-chunk gap and count of gaps > 500ms
  at the backend receive boundary, measured within a stream (tool runs between
  LLM calls are not gaps) and aggregated over the turn's streams.

A separate flag from ``VIBE_TRACE_LOOP`` on purpose: installing the loop
tracer flips ``loop.set_debug(True)`` (documented observer effect), which
would distort the very latencies measured here; and that env var's value is a
loop-block threshold, not a switch. Both tracers share the per-PID perf log
file when enabled together.

Fail-soft and near-zero cost when disabled: every probe returns after one
module-global check and takes no timestamp. Concurrent in-process turns
(subagents) are excluded via an owner token — only the outermost ``act()``
owns the trace, so subagent chunks never pollute the host turn's numbers.
All timestamps are ``time.monotonic()``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time

from vibe.core.logger import logger
from vibe.core.perf_log import perf_handler

_SLOW_GAP_S = 0.5

# propagate=False: stream summaries go only to the shared per-PID perf log.
_perf_log = logging.getLogger("vibe.perf.stream")
_perf_log.propagate = False
_perf_log.setLevel(logging.WARNING)

_enabled: bool | None = None


@dataclasses.dataclass
class _Turn:
    owner_id: int
    turn_id: str
    submitted: float
    first_chunk: float | None = None
    first_render: float | None = None
    prev_chunk: float | None = None
    max_gap: float = 0.0
    slow_gaps: int = 0
    chunks: int = 0


_turn: _Turn | None = None


def _is_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = _enable()
    return _enabled


def _enable() -> bool:
    if not os.environ.get("VIBE_TRACE_STREAM"):
        return False
    handler = perf_handler()
    if handler is None:
        logger.warning("perf stream tracer disabled: perf log unavailable")
        return False
    if handler not in _perf_log.handlers:
        _perf_log.addHandler(handler)
    logger.info("perf stream tracer installed: perf-log=%s", handler.baseFilename)
    return True


def turn_started(owner: object, turn_id: str) -> None:
    """Open a trace for *owner*'s turn. No-op when disabled or nested.

    A turn already owned by another object (an in-process subagent starting
    under the host's turn) is ignored; the same owner restarting replaces its
    own stale trace.
    """
    global _turn
    if not _is_enabled():
        return
    if _turn is not None and _turn.owner_id != id(owner):
        return
    _turn = _Turn(owner_id=id(owner), turn_id=turn_id, submitted=time.monotonic())


def stream_started(owner: object) -> None:
    """Reset the gap reference: pauses between LLM calls are not chunk gaps."""
    turn = _turn
    if turn is None or turn.owner_id != id(owner):
        return
    turn.prev_chunk = None


def chunk_received(owner: object) -> None:
    turn = _turn
    if turn is None or turn.owner_id != id(owner):
        return
    now = time.monotonic()
    turn.chunks += 1
    if turn.first_chunk is None:
        turn.first_chunk = now
    if turn.prev_chunk is not None:
        gap = now - turn.prev_chunk
        turn.max_gap = max(turn.max_gap, gap)
        if gap > _SLOW_GAP_S:
            turn.slow_gaps += 1
    turn.prev_chunk = now


def assistant_rendered() -> None:
    """Latch the first time assistant text reaches a mounted widget.

    Ownerless on purpose: the TUI event handler renders whatever the active
    (outermost) turn yields, and the latch fires once per turn.
    """
    turn = _turn
    if turn is None or turn.first_render is not None:
        return
    turn.first_render = time.monotonic()


def turn_finished(owner: object) -> None:
    """Emit the turn's summary line and close the trace. Never raises."""
    global _turn
    turn = _turn
    if turn is None or turn.owner_id != id(owner):
        return
    _turn = None
    try:
        _perf_log.warning(
            "perf stream: turn=%s ttfb=%s ttfr=%s max_gap=%.0fms slow_gaps=%d "
            "chunks=%d",
            turn.turn_id,
            _ms_since(turn.submitted, turn.first_chunk),
            _ms_since(turn.submitted, turn.first_render),
            turn.max_gap * 1000.0,
            turn.slow_gaps,
            turn.chunks,
        )
    except Exception:
        logger.debug("perf stream: summary failed", exc_info=True)


def _ms_since(start: float, end: float | None) -> str:
    return f"{(end - start) * 1000.0:.0f}ms" if end is not None else "-"
