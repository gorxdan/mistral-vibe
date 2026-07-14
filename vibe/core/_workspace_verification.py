from __future__ import annotations

from pathlib import Path

from vibe.core.worktree._trusted_git import TrustedGitError, TrustedGitWorktree


def workspace_fingerprint(path: Path | None = None) -> str | None:
    try:
        current = (path or Path.cwd()).resolve(strict=True)
        if current.is_file():
            current = current.parent
        root = next(
            (
                candidate
                for candidate in (current, *current.parents)
                if (candidate / ".git").exists() or (candidate / ".git").is_symlink()
            ),
            None,
        )
        if root is None:
            return None
        return TrustedGitWorktree.open(root).fingerprint()
    except (OSError, TrustedGitError, ValueError):
        return None
