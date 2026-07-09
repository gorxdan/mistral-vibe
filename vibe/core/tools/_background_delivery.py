from __future__ import annotations

from pathlib import Path

from vibe.core.tools.tool_result_store import truncate_middle_chars
from vibe.core.utils.io import write_safe

BACKGROUND_COMPLETION_PREVIEW_CHARS = 4_000


def compact_background_completion(response: str, path: Path | None) -> str:
    if len(response) <= BACKGROUND_COMPLETION_PREVIEW_CHARS:
        return response

    persisted_path: Path | None = None
    if path is not None:
        try:
            write_safe(path, response)
        except OSError:
            pass
        else:
            persisted_path = path

    preview = truncate_middle_chars(response, BACKGROUND_COMPLETION_PREVIEW_CHARS)
    if persisted_path is None:
        return (
            f"{preview}\n\n"
            f"...[Background result truncated from {len(response):,} characters; "
            "full output could not be persisted.]"
        )
    return (
        f"{preview}\n\n"
        f"...[Full background result ({len(response):,} characters) persisted to "
        f"{persisted_path}; use the `read` tool to retrieve it.]"
    )
