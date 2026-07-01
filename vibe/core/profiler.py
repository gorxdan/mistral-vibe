"""General-purpose profiler for measuring any section of the application.

Wraps pyinstrument (dev-only dependency). Silently no-ops when not installed or
when ``VIBE_PROFILE`` is unset, so production paths are untouched.

Activated by ``VIBE_PROFILE=1``. Optionally cap how many turns are recorded with
``VIBE_PROFILE_TURNS=N`` (default 1) to avoid a file per turn on long sessions.

Usage::

    from vibe.core import profiler

    with profiler.section("startup"):
        ...  # code to profile

    # or the raw start/stop form:
    profiler.start("startup")
    ...
    profiler.stop_and_print()

Complementary to ``vibe.core.loop_tracer`` (``VIBE_TRACE_LOOP``): the tracer
flags which coroutine blocked the single loop thread and for how long; the
profiler attributes cumulative CPU across a whole section via sampling. Use both
together: the tracer points at the turn, the profiler shows the call stack.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyinstrument import Profiler


@dataclasses.dataclass
class _State:
    profiler: Profiler | None = None
    label: str = "default"


_state = _State()


def is_enabled() -> bool:
    return bool(os.environ.get("VIBE_PROFILE"))


def turn_limit() -> int:
    """Max number of turns to record once enabled (``VIBE_PROFILE_TURNS``).

    Defaults to 1 so a long session under ``VIBE_PROFILE=1`` produces one
    representative profile rather than a file per turn. 0 disables the cap.
    """
    try:
        return max(0, int(os.environ.get("VIBE_PROFILE_TURNS", "1")))
    except ValueError:
        return 1


def start(label: str = "default") -> None:
    """Start profiling. The label names the output file.

    No-op if pyinstrument is missing or ``VIBE_PROFILE`` is unset.
    """
    if not is_enabled():
        return
    try:
        from pyinstrument import Profiler
    except ImportError:
        return

    if _state.profiler is not None:
        import warnings

        warnings.warn(
            "Profiler already running; stop it before starting a new one.", stacklevel=2
        )
        return

    _state.label = label
    _state.profiler = Profiler()
    _state.profiler.start()


def stop_and_print() -> None:
    """Stop profiling, write an HTML + text report, and print a summary."""
    if _state.profiler is None:
        return
    _state.profiler.stop()

    import sys

    from vibe.core.paths import LOG_DIR

    # LOG_DIR, not CWD: profiling a session must not litter the user's project.
    out_dir = LOG_DIR.path
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{_state.label}-profile.html"
    output_path.write_text(_state.profiler.output_html(), encoding="utf-8")

    text_path = out_dir / f"{_state.label}-profile.txt"
    text_path.write_text(_state.profiler.output_text(color=False), encoding="utf-8")

    print(
        f"\n[profiler:{_state.label}] Saved HTML profile to {output_path.resolve()}",
        file=sys.stderr,
    )
    print(
        f"[profiler:{_state.label}] Saved text profile to {text_path.resolve()}",
        file=sys.stderr,
    )
    print(_state.profiler.output_text(color=True), file=sys.stderr)

    _state.profiler = None
    _state.label = "default"


@contextlib.contextmanager
def section(label: str, *, turn: int | None = None) -> Iterator[None]:
    """Profile a code section. No-op unless ``VIBE_PROFILE`` is set.

    ``turn`` optionally caps recording to the first ``VIBE_PROFILE_TURNS`` turns
    (see ``turn_limit``) so a long session does not emit a file per turn.
    """
    if not is_enabled():
        yield
        return
    if turn is not None and turn_limit() and turn >= turn_limit():
        yield
        return
    start(label)
    try:
        yield
    finally:
        stop_and_print()
