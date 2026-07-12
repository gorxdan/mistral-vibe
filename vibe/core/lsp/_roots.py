from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vibe.core.lsp._types import path_from_uri, uri_from_path

_GLOB_MAGIC = frozenset("*?[")


@dataclass(frozen=True)
class ResolvedWorkspaceRoot:
    uri: str
    path: Path | None
    explicit: bool


def directory_matches_markers(directory: Path, markers: tuple[str, ...]) -> bool:
    for marker in markers:
        marker_path = Path(marker)
        if marker_path.is_absolute() or ".." in marker_path.parts:
            continue
        try:
            if any(character in marker for character in _GLOB_MAGIC):
                if any(directory.glob(marker)):
                    return True
            elif (directory / marker).exists():
                return True
        except (OSError, ValueError):
            continue
    return False


def nearest_manifest_root(
    file_path: str | Path, session_root: str | Path, markers: tuple[str, ...]
) -> Path:
    root = Path(session_root).expanduser().resolve()
    if not markers:
        return root
    start = Path(file_path).expanduser().resolve().parent
    try:
        start.relative_to(root)
    except ValueError:
        return root
    for candidate in (start, *start.parents):
        if directory_matches_markers(candidate, markers):
            return candidate
        if candidate == root:
            break
    return root


def resolve_workspace_root(
    file_path: str | Path,
    session_root: str | Path,
    markers: tuple[str, ...],
    *,
    explicit_root_uri: str | None = None,
) -> ResolvedWorkspaceRoot:
    if explicit_root_uri is not None:
        explicit_path = (
            Path(path_from_uri(explicit_root_uri)).expanduser().resolve()
            if explicit_root_uri.startswith("file:")
            else None
        )
        return ResolvedWorkspaceRoot(
            uri=explicit_root_uri, path=explicit_path, explicit=True
        )
    path = nearest_manifest_root(file_path, session_root, markers)
    return ResolvedWorkspaceRoot(uri=uri_from_path(path), path=path, explicit=False)


__all__ = [
    "ResolvedWorkspaceRoot",
    "directory_matches_markers",
    "nearest_manifest_root",
    "resolve_workspace_root",
]
