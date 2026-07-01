"""Shared per-PID perf log for the ``VIBE_TRACE_*`` instrumentation.

High-volume perf events go to a dedicated per-PID log so concurrent
instrumented sessions do not interleave in the shared vibe.log (nor race its
RotatingFileHandler on rollover). The handler is created once per process and
shared by every perf tracer (``loop_tracer``, ``stream_tracer``) so two
enabled tracers never open the same file twice — each attaches it to its own
non-propagating logger.
"""

from __future__ import annotations

from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time

from vibe.core.logger import StructuredLogFormatter, logger
from vibe.core.paths import LOG_DIR
from vibe.core.utils import utc_now

# Rotation caps each file's size, but every instrumented PID adds a new file to
# LOG_DIR forever — reap by mtime at handler creation, like the startup sweepers.
_PERF_LOG_MAX_AGE_S = 7 * 24 * 3600

_HANDLER: RotatingFileHandler | None = None


def gc_stale_perf_logs(directory: Path, max_age_s: float = _PERF_LOG_MAX_AGE_S) -> None:
    """Delete ``vibe-perf-*`` logs (incl. rotation backups) older than *max_age_s*.

    Best effort — never raises.
    """
    try:
        cutoff = time.time() - max_age_s
        for f in directory.glob("vibe-perf-*.log*"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
    except Exception:
        logger.debug("perf log gc skipped", exc_info=True)


def perf_handler() -> RotatingFileHandler | None:
    """The per-PID perf log handler, created on first call and then cached.

    Returns None when LOG_DIR cannot be opened (fail-soft: an unopenable log
    dir must no-op the tracers, not crash act(); a later call retries).
    Timestamp in the file name so a reused PID never appends to a dead
    session's log.
    """
    global _HANDLER
    if _HANDLER is not None:
        return _HANDLER
    gc_stale_perf_logs(LOG_DIR.path)
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    perf_path = LOG_DIR.path / f"vibe-perf-{ts}-{os.getpid()}.log"
    try:
        perf_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            perf_path, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
    except OSError:
        logger.warning("perf log unavailable: cannot open %s", perf_path, exc_info=True)
        return None
    handler.setFormatter(StructuredLogFormatter())
    _HANDLER = handler
    return handler
