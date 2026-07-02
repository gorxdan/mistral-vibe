"""LSP adherence telemetry: symbol-shaped greps done while LSP was available
(a routing miss) versus lsp calls (the intended choice).

The ratio lets prompt/routing changes be judged on data instead of
introspection. "Agent X under-uses lsp" is only actionable once you can see
*how often* it greps a symbol that lsp would have resolved.

Always-on by design: events go to a dedicated rotating file
(``vibe-adherence.log``) so the signal is captured without enabling
``VIBE_TRACE_*`` or wiring an otel collector — both are off by default, and
telemetry behind an opt-in switch is the failure mode this exists to fix. The
file is separate from ``vibe.log`` (WARNING-floor would drop INFO events) and
from the trace perf log (gated to instrumented runs). Fail-soft: if the log
dir cannot be opened, recording is a silent no-op, never a crash.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from vibe.core.logger import StructuredLogFormatter
from vibe.core.paths import LOG_DIR

_COUNTS: dict[str, int] = {"symbol_grep_miss": 0, "lsp_call": 0}

_log = logging.getLogger("vibe.adherence")
_log.propagate = False
_log.setLevel(logging.INFO)
# None = not yet built; _handler_unavailable means "tried, could not open".
_handler: RotatingFileHandler | None = None
_handler_unavailable = False


def _build_handler() -> RotatingFileHandler | None:
    global _handler, _handler_unavailable
    if _handler is not None:
        return _handler
    if _handler_unavailable:
        return None
    path = LOG_DIR.path / "vibe-adherence.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"
        )
    except OSError:
        _handler_unavailable = True
        return None
    handler.setFormatter(StructuredLogFormatter())
    handler.setLevel(_log.level)
    _log.addHandler(handler)
    _handler = handler
    return handler


def _emit(message: str) -> None:
    if _build_handler() is None:
        return
    _log.info(message)


def record_symbol_grep_miss() -> None:
    """A grep ran for a symbol-shaped pattern while LSP was available.

    That is the routing miss this harness tries to reduce: lsp would have
    resolved the symbol (incl. imports/re-exports) more completely.
    """
    _COUNTS["symbol_grep_miss"] += 1
    _emit("lsp_adherence miss kind=symbol_grep")


def record_lsp_call(operation: str) -> None:
    """A successful lsp call — the intended choice for a symbol query."""
    _COUNTS["lsp_call"] += 1
    _emit(f"lsp_adherence hit op={operation}")


def snapshot() -> dict[str, int]:
    """Current session counters (copy). For tests and ad-hoc inspection."""
    return dict(_COUNTS)


def reset_for_test() -> None:
    """Reset counters and handler state for test isolation."""
    global _handler, _handler_unavailable
    if _handler is not None:
        _log.removeHandler(_handler)
        _handler = None
    _handler_unavailable = False
    for k in _COUNTS:
        _COUNTS[k] = 0
