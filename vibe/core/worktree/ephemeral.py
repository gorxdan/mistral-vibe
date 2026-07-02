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


def _iso_lock_reason() -> str:
    import json

    boot_id = ""
    try:
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
    except OSError:
        pass
    start_time = 0
    try:
        stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").split()
        start_time = int(stat[21])
    except (OSError, IndexError, ValueError):
        pass
    return json.dumps(
        {
            "pid": os.getpid(),
            "boot_id": boot_id,
            "proc_start_time": start_time,
            "purpose": "iso",
            "created_at": time.time(),
        },
        separators=(",", ":"),
    )


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
    try:
        repo.git.worktree("lock", str(path), "--reason", _iso_lock_reason())
    except GitCommandError as exc:
        logger.warning("git worktree lock failed for %s: %s", path, exc)
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


def _try_unlock(repo: Repo, wt_path: Path) -> None:
    try:
        repo.git.worktree("unlock", str(wt_path))
    except GitCommandError as exc:
        logger.debug("worktree unlock (%s) no-op or failed: %s", wt_path, exc)


def remove_ephemeral_worktree(
    wt: EphemeralWorktree, *, keep_if_changed: bool = True
) -> bool:
    """Remove the worktree directory and (unless kept for recovery) its branch.

    If ``keep_if_changed`` and the worktree has uncommitted changes or its branch
    advanced past the base commit, the agent's work is preserved on the *branch*
    (committing any uncommitted changes first) and only the on-disk directory is
    reclaimed — recover via ``git merge <branch>``. Keeping the branch ref rather
    than the whole checked-out directory avoids the unbounded accumulation of
    kept worktree dirs under the worktrees base dir.

    Returns True when the worktree was fully reclaimed (directory removed AND
    branch deleted — nothing left to recover); False when the branch was kept for
    recovery (its directory is still reclaimed in that case).
    """
    try:
        repo = Repo(str(wt.repo_root))
        wt_repo: Repo | None = None
        try:
            wt_repo = Repo(str(wt.path))
            uncommitted = wt_repo.is_dirty(untracked_files=True)
            changed = uncommitted or wt_repo.head.commit.hexsha != wt.base_sha
        except (GitCommandError, OSError):
            uncommitted = False
            changed = False  # worktree already gone / unreadable

        if keep_if_changed and changed:
            # Persist work onto the BRANCH, then reclaim the on-disk directory.
            # Commit any uncommitted changes first; force-remove the dir only
            # once the work is safely committed (force-remove discards an
            # uncommitted tree, so a failed commit must keep the directory).
            if uncommitted and wt_repo is not None:
                try:
                    wt_repo.git.add("-A")
                    wt_repo.index.commit("workflow agent work (kept for recovery)")
                except (GitCommandError, OSError) as e:
                    logger.warning(
                        "Could not commit isolated worktree %s for recovery (%s); "
                        "keeping the directory so work is not lost.",
                        wt.path,
                        e,
                    )
                    return False
            try:
                _try_unlock(repo, wt.path)
                repo.git.worktree("remove", "--force", str(wt.path))
                logger.info(
                    "Reclaimed isolated worktree dir %s; work preserved on branch "
                    "%s (recover: git merge %s)",
                    wt.path,
                    wt.branch,
                    wt.branch,
                )
            except GitCommandError as e:
                logger.warning(
                    "Could not remove isolated worktree dir %s (%s); branch %s "
                    "still holds the work.",
                    wt.path,
                    e,
                    wt.branch,
                )
            return False  # branch kept for recovery

        # Nothing to recover (clean tree, caller opted out, or the directory was
        # already reclaimed on an earlier call): remove the directory if present
        # and delete the throwaway branch.
        if wt.path.exists():
            _try_unlock(repo, wt.path)
            if keep_if_changed:
                repo.git.worktree("remove", str(wt.path))
            else:
                repo.git.worktree("remove", "--force", str(wt.path))
        else:
            repo.git.worktree("prune")
        try:
            repo.git.branch("-D", wt.branch)
        except GitCommandError:
            pass
        return True
    except (GitCommandError, OSError) as e:
        logger.warning(
            "Failed to remove isolated worktree %s: %s. Run `vibe worktree list` "
            "to review branches.",
            wt.path,
            e,
        )
        return False
