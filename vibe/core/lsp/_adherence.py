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

from collections import OrderedDict
import logging
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

from vibe.core.logger import StructuredLogFormatter
from vibe.core.paths import LOG_DIR

if TYPE_CHECKING:
    from vibe.core.tools.base import InvokeContext


def _new_counts() -> dict[str, int]:
    return {"symbol_grep_miss": 0, "lsp_call": 0, "consecutive_symbol_grep_miss": 0}


_COUNTS = _new_counts()
_MAX_SCOPED_SESSIONS = 512
_SCOPED_COUNTS: OrderedDict[tuple[str, str], dict[str, int]] = OrderedDict()

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


def _scope_key(ctx: InvokeContext | None) -> tuple[str, str] | None:
    if ctx is None:
        return None
    if ctx.session_id:
        return ("session", ctx.session_id)
    if ctx.session_dir is not None:
        return ("session_dir", str(ctx.session_dir))
    return ("tool_call", ctx.tool_call_id)


def _counts_for(ctx: InvokeContext | None) -> dict[str, int]:
    key = _scope_key(ctx)
    if key is None:
        return _COUNTS
    counts = _SCOPED_COUNTS.get(key)
    if counts is None:
        counts = _new_counts()
        _SCOPED_COUNTS[key] = counts
        while len(_SCOPED_COUNTS) > _MAX_SCOPED_SESSIONS:
            _SCOPED_COUNTS.popitem(last=False)
    else:
        _SCOPED_COUNTS.move_to_end(key)
    return counts


def _log_value(value: object) -> str:
    text = "_".join(str(value).split())
    return text[:128]


def _log_dimensions(ctx: InvokeContext | None) -> str:
    if ctx is None:
        return ""

    dimensions: list[tuple[str, object]] = []
    if ctx.session_id:
        dimensions.append(("session", ctx.session_id))
    if ctx.agent_manager is not None:
        profile = getattr(ctx.agent_manager.active_profile, "name", None)
        if profile is not None:
            dimensions.append(("profile", profile))
    if ctx.launch_context is not None:
        entrypoint = getattr(ctx.launch_context, "agent_entrypoint", None)
        if entrypoint is not None:
            dimensions.append(("entrypoint", entrypoint))
    if not dimensions:
        return ""
    return " " + " ".join(f"{name}={_log_value(value)}" for name, value in dimensions)


def record_symbol_grep_miss(*, ctx: InvokeContext | None = None) -> int:
    """A grep ran for a symbol-shaped pattern while LSP was available.

    That is the routing miss this harness tries to reduce: lsp would have
    resolved the symbol (incl. imports/re-exports) more completely.

    Returns the new consecutive-miss count (for hint escalation).
    """
    counts = _counts_for(ctx)
    counts["symbol_grep_miss"] += 1
    counts["consecutive_symbol_grep_miss"] += 1
    consecutive = counts["consecutive_symbol_grep_miss"]
    _emit(
        f"lsp_adherence miss kind=symbol_grep consecutive={consecutive}"
        f"{_log_dimensions(ctx)}"
    )
    return consecutive


def record_lsp_call(
    operation: str, *, ctx: InvokeContext | None = None, cache_hit: bool | None = None
) -> None:
    """A successful lsp call — the intended choice for a symbol query."""
    counts = _counts_for(ctx)
    counts["lsp_call"] += 1
    counts["consecutive_symbol_grep_miss"] = 0
    cache_dimension = (
        "" if cache_hit is None else f" cache_hit={str(cache_hit).lower()}"
    )
    _emit(
        f"lsp_adherence hit op={_log_value(operation)}{cache_dimension}"
        f"{_log_dimensions(ctx)}"
    )


def consecutive_symbol_grep_misses(*, ctx: InvokeContext | None = None) -> int:
    return _counts_for(ctx)["consecutive_symbol_grep_miss"]


def should_escalate_symbol_grep(*, ctx: InvokeContext | None = None) -> bool:
    return consecutive_symbol_grep_misses(ctx=ctx) >= ESCALATE_AFTER


def symbol_grep_hint(
    pattern: str, *, consecutive: int | None = None, ctx: InvokeContext | None = None
) -> str:
    """Model-visible routing hint after a symbol-shaped grep while LSP is up.

    Bare identifiers use ``workspace_symbol`` first unless a bound task contract
    requires file-scoped queries. Escalates after ``ESCALATE_AFTER`` consecutive
    misses without an intervening lsp call.
    """
    n = (
        consecutive
        if consecutive is not None
        else consecutive_symbol_grep_misses(ctx=ctx)
    )
    bare = _is_bare_identifier(pattern)
    if ctx is not None and ctx.task_contract is not None:
        first_step = (
            "`lsp` `document_symbol` with an in-scope file_path, or "
            "`go_to_definition` / `find_references` with an in-scope file_path "
            "+ 1-based position"
        )
    else:
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


def snapshot(*, ctx: InvokeContext | None = None) -> dict[str, int]:
    """Current session counters (copy). For tests and ad-hoc inspection."""
    return dict(_counts_for(ctx))


def reset_for_test() -> None:
    """Reset counters, handler, and enabled state for test isolation."""
    global _handler, _handler_unavailable, _enabled
    if _handler is not None:
        _log.removeHandler(_handler)
        _handler = None
    _handler_unavailable = False
    _enabled = True
    _SCOPED_COUNTS.clear()
    for k in _COUNTS:
        _COUNTS[k] = 0
