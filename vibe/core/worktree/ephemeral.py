"""Ephemeral git worktrees for per-agent isolation in workflows.

Unlike :class:`WorktreeManager` (a single process-wide worktree driven by
``os.chdir``), these helpers create and remove throwaway worktrees WITHOUT
chdir and without a global singleton, so many concurrent workflow agents can
each run in their own isolated checkout (used by ``agent(isolation="worktree")``).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time

from git import Repo
from git.exc import GitCommandError

from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME

_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class EphemeralWorktree:
    path: Path
    branch: str
    repo_root: Path
    base_sha: str


def create_ephemeral_worktree(
    repo_root: Path, label: str, *, base_dir: Path | None = None
) -> EphemeralWorktree:
    """Create a throwaway worktree off HEAD. No chdir, no global state — safe to
    call concurrently (each call yields a distinct worktree + branch).
    """
    repo = Repo(str(repo_root), search_parent_directories=True)
    if repo.working_tree_dir is None:
        raise RuntimeError(f"{repo_root} is a bare repo; cannot create a worktree")
    root = Path(repo.working_tree_dir).resolve()
    base_sha = repo.head.commit.hexsha

    safe = _SAFE_LABEL_RE.sub("-", label).strip("-")[:32] or "agent"
    leaf = f"{safe}-{os.getpid()}-{time.monotonic_ns()}"
    branch = f"vibe/iso/{leaf}"
    if base_dir is None:
        base_dir = VIBE_HOME.path / "worktrees" / root.name / "iso"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / leaf

    repo.git.worktree("add", str(path), "-b", branch, "HEAD")
    logger.info("Created isolated worktree %s on branch %s", path, branch)
    return EphemeralWorktree(
        path=path, branch=branch, repo_root=root, base_sha=base_sha
    )


def deliver_ephemeral_worktree(wt: EphemeralWorktree) -> bool:
    # Gated by a passed contract: commit the agent's work, then ff-merge its
    # branch into the parent. --ff-only never creates a merge commit and never
    # force-overwrites; if the parent HEAD moved (concurrent delivery) or the
    # merge would touch uncommitted changes, git refuses — the branch stays for
    # manual merge. Returns True only if the parent HEAD advanced.
    try:
        parent = Repo(str(wt.repo_root))
        wt_repo = Repo(str(wt.path))
        if wt_repo.is_dirty(untracked_files=True):
            wt_repo.git.add("-A")
            wt_repo.index.commit("workflow agent work")
        try:
            parent.git.merge("--ff-only", wt.branch)
        except GitCommandError as e:
            reason = str(e).strip().splitlines()[:1]
            logger.info(
                "Skipping delivery of %s: ff-merge refused (%s); branch %s kept",
                wt.path,
                reason,
                wt.branch,
            )
            return False
        logger.info("Delivered isolated worktree %s into %s", wt.path, wt.repo_root)
        return True
    except (GitCommandError, OSError) as e:
        logger.warning("Failed to deliver isolated worktree %s: %s", wt.path, e)
        return False


def remove_ephemeral_worktree(
    wt: EphemeralWorktree, *, keep_if_changed: bool = True
) -> bool:
    """Remove the worktree and delete its throwaway branch.

    If ``keep_if_changed`` and the worktree has uncommitted changes or its branch
    advanced past the base commit, keep it (no ``--force``) so an isolated agent's
    work is recoverable via ``git merge``. Returns True if removed.
    """
    try:
        repo = Repo(str(wt.repo_root))
        try:
            wt_repo = Repo(str(wt.path))
            changed = (
                wt_repo.is_dirty(untracked_files=True)
                or wt_repo.head.commit.hexsha != wt.base_sha
            )
        except (GitCommandError, OSError):
            changed = False  # worktree already gone / unreadable

        if keep_if_changed and changed:
            logger.info(
                "Keeping isolated worktree %s (branch %s) — has changes to merge",
                wt.path,
                wt.branch,
            )
            return False

        # `git worktree remove` refuses a dirty worktree without --force; when
        # the caller explicitly opts out of keeping changes, force it.
        if keep_if_changed:
            repo.git.worktree("remove", str(wt.path))
        else:
            repo.git.worktree("remove", "--force", str(wt.path))
        try:
            repo.git.branch("-D", wt.branch)
        except GitCommandError:
            pass
        return True
    except (GitCommandError, OSError) as e:
        logger.warning(
            "Failed to remove isolated worktree %s: %s. Run `git worktree prune`.",
            wt.path,
            e,
        )
        return False
