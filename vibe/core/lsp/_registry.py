from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
import threading
from typing import Any

from vibe.core.lsp._types import Diagnostic, DiagnosticSeverity, Range, path_from_uri

MAX_DIAGNOSTICS_PER_FILE = 10
MAX_TOTAL_DIAGNOSTICS = 30
_DELIVERED_LRU_SIZE = 500

# Match the module path in pyright's import-resolution error messages, e.g.
#   'Import "vibe.core.memory.verifier" could not be resolved'
#   'Cannot import "vibe.core.memory.verifier" from ...'
# Captures the dotted module name in group 1.
_IMPORT_RESOLVE_RE = re.compile(
    r'(?:Import|import)\s+"([a-zA-Z_][\w.]*?)"\s+.*(?:could not be resolved|cannot be resolved|cannot be imported)'
)


class DiagnosticRegistry:
    """Collects ``textDocument/publishDiagnostics`` notifications and yields
    them as next-turn context for the model.

    Two stores: ``_pending`` holds freshly published diagnostics grouped by
    source server, drained on each call to :meth:`consume`; ``_delivered``
    is an LRU keyed by URI that suppresses re-surfacing of identical
    diagnostics across turns.
    """

    def __init__(self, root_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._pending: list[tuple[str, list[Diagnostic], str]] = []
        self._delivered: OrderedDict[str, set[str]] = OrderedDict()
        self._root = Path(root_path).resolve() if root_path is not None else None

    def set_root(self, root_path: str | Path | None) -> None:
        """Set the workspace root used for stale-suppression checks.

        Pass None to disable suppression. Called by the LSP manager when the
        workspace root is known so import-resolution errors can be validated
        against the live tree.
        """
        self._root = Path(root_path).resolve() if root_path is not None else None

    def _module_resolves_on_disk(self, module_path: str) -> bool:
        """True when a dotted module name maps to an existing file under root.

        This is the stale-suppression check: if the module the server claims it
        cannot resolve actually exists on disk (the new-module gap — pyright's
        ``markFilesDirty`` no-ops on never-loaded files), the diagnostic is
        provably stale and must not be staged to the model.
        """
        if self._root is None:
            return False
        parts = module_path.split(".")
        if not parts:
            return False
        rel = Path(*parts)
        # Module file (foo/bar.py) or package init (foo/bar/__init__.py).
        candidates = [
            self._root / rel.with_suffix(".py"),
            self._root / rel / "__init__.py",
        ]
        return any(p.is_file() for p in candidates)

    def _is_stale_import_error(self, diagnostic: Diagnostic) -> bool:
        """True for an import-resolution diagnostic whose target does exist.

        Only suppresses when the evidence proves the error wrong: the module
        file is on disk, so the LSP server's cached 'not found' is stale. Real
        missing-module errors (file genuinely absent) still stage.
        """
        if diagnostic.severity != DiagnosticSeverity.ERROR:
            return False
        match = _IMPORT_RESOLVE_RE.search(diagnostic.message)
        if match is None:
            return False
        return self._module_resolves_on_disk(match.group(1))

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
        # Drop provably-stale import-resolution errors: the server cached a
        # 'module not found' result, then a concurrent session added the module
        # via git. Without this guard the model is staged phantom errors it
        # would 'fix' by editing working code. See plan Phase 0b.
        diagnostics = [d for d in diagnostics if not self._is_stale_import_error(d)]
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
