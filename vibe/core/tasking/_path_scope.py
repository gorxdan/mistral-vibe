from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath


def path_matches_scope(path: str, pattern: str) -> bool:
    if not any(character in pattern for character in "*?["):
        return path == pattern.rstrip("/")

    path_parts = PurePosixPath(path).parts
    pattern_parts = PurePosixPath(pattern).parts
    reachable = {0}
    for pattern_part in pattern_parts:
        if not reachable:
            return False
        if pattern_part == "**":
            reachable = set(range(min(reachable), len(path_parts) + 1))
            continue
        reachable = {
            path_index + 1
            for path_index in reachable
            if path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], pattern_part)
        }
    return len(path_parts) in reachable


__all__ = ["path_matches_scope"]
