from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from vibe.core.utils.io import read_safe

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_filename(call_id: str) -> str:
    """Make a tool_call_id safe to use as a filename.

    Call ids are uuid4 hex strings in practice, but stay defensive against any
    tool-call id shape.
    """
    name = _UNSAFE_NAME_CHARS.sub("_", call_id).strip("_")
    return name or "unknown"


def truncate_middle_chars(text: str, limit: int) -> str:
    """Return head + tail of *text* with the middle replaced by a marker.

    Keeps ``limit * 3 // 4`` head and the remainder tail so both the start and
    end of a result stay visible.
    """
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit - head
    elided = len(text) - head - tail
    return f"{text[:head]}\n\n…[{elided} characters elided]…\n\n{text[-tail:]}"


class ToolResultStore:
    """Persists oversized tool results to disk so their full content survives
    beyond the inline preview, keyed by tool_call_id.

    The store follows the live session directory via a getter, so compact/fork/
    resume (which change ``session_dir``) need no rewiring. When no session
    directory is available (session logging disabled), the store is inert and
    callers fall back to permanent middle-truncation.

    Full outputs land under ``<session_dir>/tool_results/<call_id>.txt`` as
    UTF-8. The agent can recall one with the existing ``read`` tool once the
    path is surfaced in the inline preview marker.
    """

    SUBDIR = "tool_results"

    def __init__(self, session_dir_getter: Callable[[], Path | None]) -> None:
        self._get_session_dir = session_dir_getter

    @property
    def available(self) -> bool:
        return self._get_session_dir() is not None

    def _dir(self) -> Path | None:
        base = self._get_session_dir()
        if base is None:
            return None
        return base / self.SUBDIR

    def path_for(self, call_id: str) -> Path | None:
        target_dir = self._dir()
        if target_dir is None:
            return None
        return target_dir / f"{_safe_filename(call_id)}.txt"

    def persist(self, call_id: str, content: str) -> Path | None:
        """Write the full content to disk, returning the path or None on failure.

        Failures (disabled logging, unwritable disk) are non-fatal: the caller
        falls back to permanent middle-truncation, matching prior behavior.
        """
        path = self.path_for(call_id)
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError:
            return None
        return path

    def read(self, call_id: str) -> str | None:
        """Return the persisted full content for *call_id*, or None if absent."""
        path = self.path_for(call_id)
        if path is None or not path.is_file():
            return None
        return read_safe(path).text

    def shape(
        self, call_id: str, content: str, *, preview_chars: int, hard_cap: int
    ) -> str:
        """Return the inline-safe form of a tool result.

        Results at or under *hard_cap* are returned unchanged. Larger results
        are persisted in full and replaced inline with a smaller head+tail
        preview that names the persisted path. When persistence is unavailable,
        fall back to permanent middle-truncation at *hard_cap*.
        """
        if len(content) <= hard_cap:
            return content
        path = self.persist(call_id, content)
        if path is not None:
            preview = truncate_middle_chars(content, preview_chars)
            return (
                f"{preview}\n\n"
                f"…[Full output ({len(content):,} characters) persisted to {path}; "
                f"use the `read` tool with this path to retrieve it.]…"
            )
        return truncate_middle_chars(content, hard_cap)
