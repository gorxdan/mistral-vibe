from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import threading
from typing import Any

from vibe.core.lsp._stale import FILTERS_BY_SOURCE, StaleDiagnosticFilter
from vibe.core.lsp._types import Diagnostic, DiagnosticSeverity, Range, path_from_uri

MAX_DIAGNOSTICS_PER_FILE = 10
MAX_TOTAL_DIAGNOSTICS = 30
_DELIVERED_LRU_SIZE = 500


class DiagnosticRegistry:
    """Collects ``textDocument/publishDiagnostics`` notifications and yields
    them as next-turn context for the model.

    Two stores: ``_pending`` holds freshly published diagnostics grouped by
    source server, drained on each call to :meth:`consume`; ``_delivered``
    is an LRU keyed by URI that suppresses re-surfacing of identical
    diagnostics across turns.

    Stale-suppression is delegated to per-source :class:`StaleDiagnosticFilter`
    strategies (see :data:`~vibe.core.lsp._stale.FILTERS_BY_SOURCE`); the
    registry itself is server-agnostic.
    """

    def __init__(
        self,
        root_path: str | Path | None = None,
        *,
        filters: Mapping[str, StaleDiagnosticFilter] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._pending: list[tuple[str, list[Diagnostic], str]] = []
        self._delivered: OrderedDict[str, set[str]] = OrderedDict()
        self._root = Path(root_path).resolve() if root_path is not None else None
        self._filters = filters if filters is not None else FILTERS_BY_SOURCE

    def set_root(self, root_path: str | Path | None) -> None:
        """Set the workspace root the per-source filters check against."""
        self._root = Path(root_path).resolve() if root_path is not None else None

    def _filter_stale(
        self, diagnostics: list[Diagnostic], server_name: str
    ) -> list[Diagnostic]:
        """Drop diagnostics a source-specific filter proves stale.

        Servers with no registered filter pass through unchanged.
        """
        policy = self._filters.get(server_name)
        if policy is None:
            return diagnostics
        return [d for d in diagnostics if not policy.is_stale(d, self._root)]

    def publish(self, params: dict[str, Any], server_name: str) -> None:
        uri = params.get("uri", "")
        if not uri:
            return
        path = path_from_uri(uri)
        raw = params.get("diagnostics") or []
        # Only surface errors and warnings to the model. Hints/info (unused
        # params, unreachable code, style nits) are noise that burns context
        # without driving any fix the model should make.
        diagnostics = [
            Diagnostic.from_lsp(d)
            for d in raw
            if int(d.get("severity", DiagnosticSeverity.ERROR))
            <= DiagnosticSeverity.WARNING
        ]
        diagnostics = self._filter_stale(diagnostics, server_name)
        if not diagnostics:
            return
        with self._lock:
            self._pending.append((path, diagnostics, server_name))

    def consume(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._pending:
                return []
            grouped: OrderedDict[str, list[Diagnostic]] = OrderedDict()
            sources: set[str] = set()
            for path, diagnostics, server_name in self._pending:
                grouped.setdefault(path, []).extend(diagnostics)
                sources.add(server_name)
            self._pending.clear()

        files_out: list[dict[str, Any]] = []
        total = 0
        for path, diagnostics in grouped.items():
            already = self._delivered.get(path, set())
            fresh: list[Diagnostic] = []
            seen_now: set[str] = set()
            ordered = sorted(diagnostics, key=lambda d: int(d.severity))
            for diag in ordered:
                if total >= MAX_TOTAL_DIAGNOSTICS:
                    break
                if diag.dedup_key in already or diag.dedup_key in seen_now:
                    continue
                fresh.append(diag)
                seen_now.add(diag.dedup_key)
                total += 1
            if not fresh:
                continue
            self._delivered[path] = seen_now | already
            self._delivered.move_to_end(path)
            while len(self._delivered) > _DELIVERED_LRU_SIZE:
                self._delivered.popitem(last=False)
            files_out.append({
                "path": path,
                "diagnostics": fresh[:MAX_DIAGNOSTICS_PER_FILE],
            })
        if not files_out:
            return []
        return [{"sources": sorted(sources), "files": files_out}]

    def clear_for_path(self, path: str) -> None:
        with self._lock:
            self._delivered.pop(path, None)

    def clear_all(self) -> None:
        with self._lock:
            self._pending.clear()
            self._delivered.clear()


def format_diagnostics_for_model(batch: dict[str, Any]) -> str:
    lines: list[str] = ["LSP diagnostics (from " + ", ".join(batch["sources"]) + "):"]
    for file_entry in batch["files"]:
        path = file_entry["path"]
        lines.append(f"\n{path}")
        for diag in file_entry["diagnostics"]:
            start = diag.range.start
            lines.append(
                f"  line {start.line + 1}, col {start.character + 1} - "
                f"{diag.label}: {diag.message}"
            )
    return "\n".join(lines)


__all__ = [
    "DiagnosticRegistry",
    "DiagnosticSeverity",
    "Range",
    "format_diagnostics_for_model",
]
