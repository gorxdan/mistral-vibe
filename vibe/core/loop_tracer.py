"""Event-loop blocking tracer. Env-gated: ``VIBE_TRACE_LOOP=<seconds>``.

The TUI process runs one asyncio loop (Textual's) that hosts the agent loop,
tool fan-out, memory extraction, turn summaries, LSP jsonrpc dispatch,
telemetry and the render pump as sibling coroutines. The loop is
single-threaded, so one CPU-bound callback starves every other coroutine: the
TUI stalls, tool calls queue, and the whole process tree reads as
"single-core heavy" even though the CPU% is shared across children.

This tracer times every callback and logs the ones that monopolize the loop
past a threshold, attributing each to the running task and the call site that
scheduled it. It is the lens that turns "the loop is busy" into "coroutine X
blocked the shared thread for Y ms". Complementary to ``vibe.cli.profiler``
(``VIBE_PROFILE``), which attributes cumulative CPU via sampling; this answers
which coroutine hogged the loop thread.

Install is idempotent and fail-soft; a complete no-op unless the env var is set,
so production paths are untouched. When enabled it wraps ``asyncio.Handle._run``
(affects both Handle and TimerHandle) — opt-in eval tooling only.

Observer effect: install() turns on ``loop.set_debug(True)`` for scheduled-from
attribution, which makes asyncio capture a traceback for every scheduled handle
and enables its debug checks. Traced runs are measurably slower than untraced
ones — compare traced runs only against other traced runs.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import logging
import os
import time
import traceback
from typing import Any

from vibe.core.logger import logger
from vibe.core.perf_log import perf_handler

_ORIG_RUN: Any = None

# propagate=False: loop-block events go only to the shared per-PID perf log.
_perf_log = logging.getLogger("vibe.perf.loop")
_perf_log.propagate = False
_perf_log.setLevel(logging.WARNING)
_INSTALLED = False
_THRESHOLD = 0.0
# (task label, handle module) -> [count, total_ms]. Populated only while
# installed; report() summarizes so a benchmark can emit a clean top-N rather
# than grepping occurrence lines.
_BLOCKERS: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or "?"


def _attrib(handle: Any) -> tuple[str, str]:
    """Derive (task label, source location) from a handle.

    A task step is scheduled as a bound method whose ``__self__`` IS the task,
    which is stable across CPython versions and observable from outside the
    callback (unlike ``asyncio.current_task()``, which is only set while the
    coroutine body is executing — gone by the time the timing ``finally`` runs).
    """
    cb = getattr(handle, "_callback", None)
    owner = getattr(cb, "__self__", None)
    if isinstance(owner, asyncio.Task):
        coro = owner.get_coro()
        qual = getattr(coro, "__qualname__", "?")
        code = getattr(coro, "cr_code", None)
        where = _basename(getattr(code, "co_filename", "")) if code else "?"
        return f"{owner.get_name()}<{qual}>", where
    # Plain callback (timer, call_soon) — not a task step.
    module = getattr(cb, "__module__", None)
    qual = getattr(cb, "__qualname__", None)
    return "<callback>", module or qual or "?"


def _scheduled_from(handle: Any) -> str:
    tb = getattr(handle, "_source_traceback", None)
    if not tb:
        return ""
    # Tail frames; the head is asyncio's scheduler plumbing.
    return "".join(traceback.format_list(tb[-3:])).rstrip()


def _wrap_run(orig: Any) -> Any:
    threshold = _THRESHOLD

    def _run(self: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(self)
        finally:
            dt = time.perf_counter() - t0
            if dt >= threshold:
                try:
                    label, where = _attrib(self)
                    bucket = _BLOCKERS[(label, where)]
                    bucket[0] += 1
                    bucket[1] += dt * 1000.0
                    _perf_log.warning(
                        "perf loop-block: %.0fms task=%s where=%s%s",
                        dt * 1000.0,
                        label,
                        where,
                        f"\n{scheduled}"
                        if (scheduled := _scheduled_from(self))
                        else "",
                    )
                except Exception:
                    logger.debug("perf loop-block: report failed", exc_info=True)

    return _run


def install() -> None:
    """Install the tracer on the current event loop if ``VIBE_TRACE_LOOP`` is set.

    Idempotent. Reads the threshold (seconds) from the env var; an unparseable
    value falls back to 50ms. No-op when unset or already installed, and when
    no loop is running (so importing the module is always safe).
    """
    global _ORIG_RUN, _INSTALLED, _THRESHOLD
    if _INSTALLED:
        return
    raw = os.environ.get("VIBE_TRACE_LOOP")
    if not raw:
        return
    try:
        threshold = float(raw)
    except ValueError:
        threshold = 0.05
    if threshold <= 0:
        return

    # Handler before any global mutation: a bad LOG_DIR must no-op, not crash act().
    handler = perf_handler()
    if handler is None:
        logger.warning("perf loop-block tracer disabled: perf log unavailable")
        return
    if handler not in _perf_log.handlers:
        _perf_log.addHandler(handler)

    # Debug mode populates Handle._source_traceback (scheduled-from attribution).
    # slow_callback_duration is raised to infinity so asyncio's own built-in
    # slow-callback logger never fires alongside ours (we are the single source).
    try:
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        loop.slow_callback_duration = float("inf")
    except RuntimeError:
        pass  # no running loop; wrapping still applies once one is running

    _THRESHOLD = threshold
    _ORIG_RUN = asyncio.Handle._run
    asyncio.Handle._run = _wrap_run(_ORIG_RUN)
    _INSTALLED = True
    logger.info(
        "perf loop-block tracer installed: threshold=%.0fms perf-log=%s",
        threshold * 1000.0,
        handler.baseFilename,
    )


def report(top: int = 15) -> None:
    """Log a top-N summary of callbacks that blocked the loop since install."""
    if not _INSTALLED or not _BLOCKERS:
        return
    ranked = sorted(_BLOCKERS.items(), key=lambda kv: kv[1][1], reverse=True)
    _perf_log.warning(
        "perf loop-block summary (top %d by total ms):", min(top, len(ranked))
    )
    for (label, mod), (count, total_ms) in ranked[:top]:
        _perf_log.warning(
            "perf loop-block: %6.0fms / %3d calls  avg=%4.0fms  task=%s  where=%s",
            total_ms,
            int(count),
            total_ms / count if count else 0.0,
            label,
            mod,
        )


def reset() -> None:
    """Clear accumulated blocker stats (e.g. between benchmark iterations)."""
    _BLOCKERS.clear()
