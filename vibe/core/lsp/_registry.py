from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
import ntpath
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import unquote, urlsplit

from vibe.core.lsp._positions import utf16_range_to_codepoint
from vibe.core.lsp._stale import FILTERS_BY_SOURCE, StaleDiagnosticFilter
from vibe.core.lsp._types import Diagnostic, DiagnosticSeverity, Range
from vibe.core.utils.io import read_safe

MAX_DIAGNOSTICS_PER_FILE = 10
MAX_TOTAL_DIAGNOSTICS = 30
_STATE_LRU_SIZE = 500
_WINDOWS_DRIVE_RE = re.compile(r"^/?[A-Za-z]:[\\/]")
_WINDOWS_DRIVE_HOST_RE = re.compile(r"^[A-Za-z]:$")


def normalize_diagnostic_path(path_or_uri: str | Path) -> str:
    raw = str(path_or_uri)
    if not raw:
        return ""

    try:
        is_file_uri = raw.lower().startswith("file:")
        remote_file_uri = False
        if is_file_uri:
            parsed = urlsplit(raw)
            host = unquote(parsed.netloc)
            decoded = unquote(parsed.path)
            if host and host.casefold() != "localhost":
                if _WINDOWS_DRIVE_HOST_RE.fullmatch(host):
                    raw = f"{host}{decoded}"
                else:
                    raw = f"//{host}{decoded}"
                    remote_file_uri = True
            else:
                raw = decoded
            if not raw:
                return ""

        windows_drive = _WINDOWS_DRIVE_RE.match(raw)
        windows_unc = (
            remote_file_uri
            or raw.startswith("\\\\")
            or (not is_file_uri and raw.startswith("//"))
        )
        if windows_drive:
            if raw.startswith("/"):
                raw = raw[1:]
            return ntpath.normcase(ntpath.normpath(raw)).replace("\\", "/")
        if windows_unc:
            return ntpath.normcase(ntpath.normpath(raw)).replace("\\", "/")

        return str(Path(raw).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return ""


@dataclass
class _SourceState:
    diagnostics: dict[str, Diagnostic]
    delivered: set[str] = field(default_factory=set)


class DiagnosticRegistry:
    """Tracks the latest diagnostics from each server and yields new context.

    Each ``publishDiagnostics`` notification replaces the state for its
    normalized path and server. Delivery state belongs to that same source,
    so one server resolving a file cannot erase another server's diagnostics.

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
        self._states: OrderedDict[tuple[str, str], _SourceState] = OrderedDict()
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
        path = normalize_diagnostic_path(str(uri))
        if not path:
            return
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
        current = {diagnostic.dedup_key: diagnostic for diagnostic in diagnostics}
        state_key = (path, server_name)
        with self._lock:
            if not current:
                self._states.pop(state_key, None)
                return
            previous = self._states.get(state_key)
            delivered = previous.delivered.intersection(current) if previous else set()
            self._states[state_key] = _SourceState(current, delivered)
            self._states.move_to_end(state_key)
            while len(self._states) > _STATE_LRU_SIZE:
                self._states.popitem(last=False)

    def consume(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._states:
                return []
            emitted: OrderedDict[str, list[Diagnostic]] = OrderedDict()
            emitted_sources: set[str] = set()
            emitted_states: list[tuple[str, str]] = []
            per_file: dict[str, int] = {}
            total = 0
            for state_key, state in list(self._states.items()):
                if total >= MAX_TOTAL_DIAGNOSTICS:
                    break
                path, server_name = state_key
                file_count = per_file.get(path, 0)
                file_remaining = MAX_DIAGNOSTICS_PER_FILE - file_count
                if file_remaining <= 0:
                    continue
                limit = min(file_remaining, MAX_TOTAL_DIAGNOSTICS - total)
                fresh = [
                    (key, diagnostic)
                    for key, diagnostic in sorted(
                        state.diagnostics.items(),
                        key=lambda item: int(item[1].severity),
                    )
                    if key not in state.delivered
                ][:limit]
                if not fresh:
                    continue
                state.delivered.update(key for key, _ in fresh)
                emitted.setdefault(path, []).extend(
                    diagnostic for _, diagnostic in fresh
                )
                emitted_sources.add(server_name)
                emitted_states.append(state_key)
                count = len(fresh)
                per_file[path] = file_count + count
                total += count

            for state_key in emitted_states:
                self._states.move_to_end(state_key)

            if not emitted:
                return []
            files_out = [
                {"path": path, "diagnostics": diagnostics}
                for path, diagnostics in emitted.items()
            ]
            return [{"sources": sorted(emitted_sources), "files": files_out}]

    def clear_for_path(self, path: str | Path) -> None:
        normalized = normalize_diagnostic_path(path)
        if not normalized:
            return
        with self._lock:
            for state_key in list(self._states):
                if state_key[0] == normalized:
                    self._states.pop(state_key)

    def clear_all(self) -> None:
        with self._lock:
            self._states.clear()


def format_diagnostics_for_model(batch: dict[str, Any]) -> str:
    lines: list[str] = ["LSP diagnostics (from " + ", ".join(batch["sources"]) + "):"]
    for file_entry in batch["files"]:
        path = file_entry["path"]
        lines.append(f"\n{path}")
        source_text: str | None = None
        source_path = Path(path)
        if source_path.is_file():
            try:
                source_text = read_safe(source_path).text
            except OSError:
                pass
        for diag in file_entry["diagnostics"]:
            start = diag.range.start
            encoding_note = " (UTF-16 column)"
            if source_text is not None:
                try:
                    start = utf16_range_to_codepoint(source_text, diag.range).start
                    encoding_note = ""
                except (TypeError, ValueError):
                    pass
            lines.append(
                f"  line {start.line + 1}, col {start.character + 1}"
                f"{encoding_note} - "
                f"{diag.label}: {diag.message}"
            )
    return "\n".join(lines)


__all__ = [
    "DiagnosticRegistry",
    "DiagnosticSeverity",
    "Range",
    "format_diagnostics_for_model",
    "normalize_diagnostic_path",
]
