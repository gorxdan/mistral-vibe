"""LSP adherence telemetry: symbol-shaped greps done while LSP was available
(a routing miss) versus lsp calls (the intended choice).

The ratio lets prompt/routing changes be judged on data instead of
introspection. "Agent X under-uses lsp" is only actionable once you can see
*how often* it greps a symbol that lsp would have resolved.

Default-on local diagnostics: events go to a dedicated rotating file
(``vibe-adherence.log``, 2MB x2 backups, metadata-only counters, no egress)
so the signal is captured without enabling ``VIBE_TRACE_*`` or wiring an otel
collector. Gated by ``enable_telemetry``: entrypoints call :func:`configure`
at startup, and ``enable_telemetry = false`` silences the log. Unwired paths
(bare library use) keep today's default-on behavior. The file is separate
from ``vibe.log`` (WARNING-floor would drop INFO events) and from the trace
perf log (gated to instrumented runs). Fail-soft: if the log dir cannot be
opened, recording is a silent no-op, never a crash.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from vibe.core.logger import StructuredLogFormatter
from vibe.core.paths import LOG_DIR

_COUNTS: dict[str, int] = {
    "symbol_grep_miss": 0,
    "lsp_call": 0,
    # Consecutive misses since last lsp hit; resets on record_lsp_call.
    "consecutive_symbol_grep_miss": 0,
}

# Soft NOTE below this; ESCALATION at/above (no intervening lsp call).
ESCALATE_AFTER = 2

_log = logging.getLogger("vibe.adherence")
_log.propagate = False
_log.setLevel(logging.INFO)
# None = not yet built; _handler_unavailable means "tried, could not open".
_handler: RotatingFileHandler | None = None
_handler_unavailable = False
# Default True so unwired paths (bare library use) keep logging.
_enabled = True


def configure(enabled: bool) -> None:
    """Wire the enable_telemetry config flag; called at entrypoint startup."""
    global _enabled
    _enabled = enabled


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
    if not _enabled or _build_handler() is None:
        return
    _log.info(message)


def record_symbol_grep_miss() -> int:
    """A grep ran for a symbol-shaped pattern while LSP was available.

    That is the routing miss this harness tries to reduce: lsp would have
    resolved the symbol (incl. imports/re-exports) more completely.

    Returns the new consecutive-miss count (for hint escalation).
    """
    _COUNTS["symbol_grep_miss"] += 1
    _COUNTS["consecutive_symbol_grep_miss"] += 1
    consecutive = _COUNTS["consecutive_symbol_grep_miss"]
    _emit(f"lsp_adherence miss kind=symbol_grep consecutive={consecutive}")
    return consecutive


def record_lsp_call(operation: str) -> None:
    """A successful lsp call — the intended choice for a symbol query."""
    _COUNTS["lsp_call"] += 1
    # Corrected path: clear the consecutive streak so the next miss is soft again.
    _COUNTS["consecutive_symbol_grep_miss"] = 0
    _emit(f"lsp_adherence hit op={operation}")


def consecutive_symbol_grep_misses() -> int:
    return _COUNTS["consecutive_symbol_grep_miss"]


def should_escalate_symbol_grep() -> bool:
    return _COUNTS["consecutive_symbol_grep_miss"] >= ESCALATE_AFTER


def symbol_grep_hint(pattern: str, *, consecutive: int | None = None) -> str:
    """Model-visible routing hint after a symbol-shaped grep while LSP is up.

    Bare identifiers → ``workspace_symbol`` first (no file/position needed).
    Position-based ops only after a hit. Escalates after ``ESCALATE_AFTER``
    consecutive misses without an intervening lsp call.
    """
    n = (
        consecutive
        if consecutive is not None
        else _COUNTS["consecutive_symbol_grep_miss"]
    )
    bare = _is_bare_identifier(pattern)
    first_step = (
        f"`lsp` `workspace_symbol` query={pattern!r} (no file_path needed)"
        if bare
        else (
            "`lsp` `workspace_symbol` for the name, or `go_to_definition` / "
            "`find_references` once you have file_path + 1-based position"
        )
    )
    if n >= ESCALATE_AFTER:
        return (
            f"ESCALATION: {n} consecutive symbol greps while LSP is available "
            f"(pattern {pattern!r}). Stop using grep for symbols — call {first_step} "
            "next. grep misses imports, re-exports, aliases, and overloads that "
            "lsp resolves. Further symbol greps without an lsp call keep this "
            "escalation active."
        )
    return (
        f"NOTE: {pattern!r} is a symbol lookup — use {first_step} instead of "
        "grep. grep misses imports, re-exports, aliases, and overloads that "
        "lsp resolves; grepping a symbol here is the routing miss the harness "
        "flags. For a symbol question, call lsp next."
    )


def _is_bare_identifier(pattern: str) -> bool:
    return bool(pattern) and pattern.isidentifier()


def snapshot() -> dict[str, int]:
    """Current session counters (copy). For tests and ad-hoc inspection."""
    return dict(_COUNTS)


def reset_for_test() -> None:
    """Reset counters, handler, and enabled state for test isolation."""
    global _handler, _handler_unavailable, _enabled
    if _handler is not None:
        _log.removeHandler(_handler)
        _handler = None
    _handler_unavailable = False
    _enabled = True
    for k in _COUNTS:
        _COUNTS[k] = 0
