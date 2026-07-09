from __future__ import annotations

import hashlib
from pathlib import Path

from git import Repo
from git.exc import GitError


def workspace_fingerprint(path: Path | None = None) -> str | None:
    try:
        repo = Repo(path or Path.cwd(), search_parent_directories=True)
        if repo.working_tree_dir is None:
            return None
        head = repo.head.commit.hexsha
        working_diff = repo.git.diff("--binary", "HEAD", "--")
        staged_diff = repo.git.diff("--binary", "--cached", "HEAD", "--")
        untracked = [
            (name, repo.git.hash_object("--", name).strip())
            for name in sorted(repo.untracked_files)
        ]
    except (GitError, OSError, ValueError):
        return None

    digest = hashlib.sha256()
    for value in (head, staged_diff, working_diff):
        digest.update(value.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    for name, blob_hash in untracked:
        digest.update(name.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(blob_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
