"""Git worktree isolation for the vibe harness.

When active, agent writes land on a throwaway branch in a git worktree
instead of the user's live checkout.  The lever is a single ``os.chdir``
at the top-level entrypoint — subagents inherit the process cwd and
never call :meth:`WorktreeManager.enter`.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from git import Repo
from git.exc import GitCommandError

from vibe.core.config import WorktreeConfig
from vibe.core.logger import logger
from vibe.core.trusted_folders import trusted_folders_manager

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

__all__ = [
    "WorktreeHandle",
    "WorktreeManager",
    "original_working_directory",
    "worktree_enabled",
    "worktree_manager",
]

# Names of files/dirs that indicate a git operation is in progress.
_MID_OPERATION_MARKERS = [
    "MERGE_HEAD",
    "rebase-merge",
    "rebase-apply",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
]

# Default git identity for WIP commits when the user has none configured.
_FALLBACK_GIT_NAME = "vibe"
_FALLBACK_GIT_EMAIL = "vibe@local"


@dataclass(frozen=True)
class WorktreeHandle:
    """Durable anchor for an active worktree session."""

    original_repo_root: Path
    worktree_path: Path
    branch: str
    create_head_sha: str
    symlinks: list[Path] = field(default_factory=list)
    config: WorktreeConfig = field(default_factory=WorktreeConfig)


def worktree_enabled(
    config: VibeConfig, *, programmatic: bool, cli_flag: bool = False
) -> bool:
    """Return ``True`` if worktree isolation should be active."""
    m = config.worktree.mode
    if m == "off":
        return False
    if m == "on":
        return True
    # auto-by-entrypoint: programmatic ON, cli needs --worktree, acp OFF.
    return programmatic or cli_flag


def original_working_directory() -> str:
    """Return the original repo root if a worktree is active, else ``str(Path.cwd())``.

    Used by :mod:`vibe.core.session.session_logger` so that the recorded
    ``working_directory`` is the user's real checkout, not the worktree path.
    This keeps the exact-string match in :mod:`vibe.core.session.session_loader`
    working after the worktree is removed.
    """
    if worktree_manager.active is not None:
        return str(worktree_manager.active.original_repo_root)
    return str(Path.cwd())


class WorktreeManager:
    """Manages a single active git worktree for the process.

    Only one worktree can be active at a time (the nested-enter guard).
    Subagents never call :meth:`enter` — they inherit the process cwd.
    """

    def __init__(self) -> None:
        self._active: WorktreeHandle | None = None
        self._atexit_registered = False
        self._signal_handlers_installed = False
        self._signal_received: int | None = None

    @property
    def active(self) -> WorktreeHandle | None:
        return self._active

    # ------------------------------------------------------------------
    # enter
    # ------------------------------------------------------------------

    def enter(self, label: str, config: WorktreeConfig) -> WorktreeHandle | None:
        """Create a worktree, carry dirty state, and chdir into it.

        Returns ``None`` (run in-place) if the repo is mid-operation or has
        dirty submodules.  Raises :class:`RuntimeError` if a worktree is
        already active (nested-enter guard).
        """
        if self._active is not None:
            raise RuntimeError(
                "WorktreeManager.enter() called while a worktree is already active. "
                "Subagents must never call enter() — they inherit the parent's cwd."
            )

        try:
            return self._do_enter(label, config)
        except Exception:
            # Fail soft to in-place (never lose user work), but record the full
            # traceback so programming errors (AttributeError/TypeError/etc.)
            # are diagnosable instead of silently degrading isolation.
            logger.exception("Worktree creation failed, running in-place")
            # Best-effort cleanup of partial state.
            self._cleanup_partial()
            return None

    def _do_enter(self, label: str, config: WorktreeConfig) -> WorktreeHandle | None:
        # 1. Crash recovery: prune stale worktrees.
        self._prune_and_report(config)

        # 2. Resolve original repo root via git (NOT find_git_repo_ancestor,
        #    which requires .git to be a dir and misses worktree roots).
        repo = self._get_repo(Path.cwd())
        original_root = Path(repo.working_tree_dir).resolve()

        # 3. Refuse if repo mid-operation.
        if self._is_mid_operation(original_root):
            logger.warning(
                "Repo is mid-operation (merge/rebase/cherry-pick). "
                "Skipping worktree isolation — running in-place."
            )
            return None

        # 4. Refuse if dirty submodules.
        if self._has_dirty_submodules(repo):
            logger.warning(
                "Repo has dirty submodules. Skipping worktree isolation — "
                "running in-place. Submodule in-progress edits are not carried."
            )
            return None

        # 5. Record create_head_sha.
        create_head_sha = repo.head.commit.hexsha

        # 6. Collision-free branch name (nanosecond resolution).
        leaf = f"{label}-{os.getpid()}-{time.time_ns()}"
        branch = f"{config.branch_prefix}{leaf}"

        # 7. Resolve worktree path from config.base_dir (outside the repo by
        #    default, so it never appears in the user's git status).
        base_dir = Path(config.base_dir)
        repo_name = original_root.name
        worktree_path = base_dir / repo_name / leaf
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        repo.git.worktree("add", str(worktree_path), "-b", branch, "HEAD")
        logger.info("Created worktree at %s on branch %s", worktree_path, branch)

        # 8. Carry dirty state (excluding carry_ignored paths, which are
        #    symlinked in step 9 instead).
        symlinks: list[Path] = []
        if config.carry_dirty:
            self._carry_dirty(repo, worktree_path, config.carry_ignored)

        # 9. Symlink deps.
        symlinks = self._symlink_deps(original_root, worktree_path, config)

        # 10. Trust the worktree.
        trusted_folders_manager.trust_for_session(worktree_path)

        # 11. Set orientation (footer shows original root).
        # Done by the caller via config.displayed_workdir.

        # 12. chdir into worktree.
        os.chdir(worktree_path)

        # 13. Store handle and register cleanup backstops.
        handle = WorktreeHandle(
            original_repo_root=original_root,
            worktree_path=worktree_path,
            branch=branch,
            create_head_sha=create_head_sha,
            symlinks=symlinks,
            config=config,
        )
        self._active = handle
        self._register_cleanup_backstops()
        return handle

    # ------------------------------------------------------------------
    # exit
    # ------------------------------------------------------------------

    def exit(self, handle: WorktreeHandle) -> None:
        """WIP-commit dirty state, optionally merge, then remove the worktree.

        Never discards work.  If WIP-commit or removal fails, the worktree
        and branch are kept for manual recovery.
        """
        if self._active is None or self._active.branch != handle.branch:
            logger.warning(
                "WorktreeManager.exit() called with a stale handle (branch=%s). "
                "Skipping — active worktree is %s.",
                handle.branch,
                self._active.branch if self._active else "none",
            )
            return

        try:
            self._do_exit(handle)
        except Exception as exc:
            logger.error("Worktree teardown failed: %s. Worktree kept for recovery.", exc)
        finally:
            self._active = None

    def _do_exit(self, handle: WorktreeHandle) -> None:
        wt_repo = self._get_repo(handle.worktree_path)

        # 1. Unlink dep symlinks BEFORE the WIP commit so the worktree
        #    tree is clean of ephemeral symlinks (they are gitignored, but
        #    deleting them after the commit leaves the worktree dirty again
        #    and blocks git worktree remove).
        for s in handle.symlinks:
            try:
                s.unlink()
            except OSError as exc:
                logger.warning("Failed to unlink symlink %s: %s", s, exc)

        # 2. WIP-commit if dirty.
        wip_ok = True
        if self._is_dirty(wt_repo):
            wip_ok = self._wip_commit(wt_repo, handle)

        # 3. Auto-ff merge (only if configured and the WIP commit succeeded).
        #    If the WIP commit failed, the branch is missing the latest work;
        #    merging it would give a misleading "merged" result, so skip and
        #    leave the worktree for manual recovery.
        merged = False
        if handle.config.merge == "auto-ff" and wip_ok:
            merged = self._try_auto_ff(handle)

        # 4. chdir back BEFORE worktree remove (removing cwd leaves stale cwd).
        os.chdir(handle.original_repo_root)

        # 5. Remove worktree (no --force).
        if handle.config.cleanup == "remove":
            try:
                root_repo = self._get_repo(handle.original_repo_root)
                root_repo.git.worktree("remove", str(handle.worktree_path))
                logger.info("Removed worktree at %s", handle.worktree_path)
            except GitCommandError as exc:
                logger.warning(
                    "git worktree remove failed (keeping worktree): %s. "
                    "Branch %s is safe. Run `git worktree prune` later.",
                    exc,
                    handle.branch,
                )

        # 6. Print handoff.
        if not merged:
            print(
                f"\nWorktree branch: {handle.branch}\n"
                f"Worktree path: {handle.worktree_path}\n"
                f"To merge: git merge {handle.branch}\n",
                file=sys.stdout,
            )

    # ------------------------------------------------------------------
    # dirty carry
    # ------------------------------------------------------------------

    def _carry_dirty(
        self, repo: Repo, worktree_path: Path, carry_ignored: list[str]
    ) -> None:
        """Copy tracked + untracked working-tree changes into the worktree.

        Uses a COPY of the real index under ``GIT_INDEX_FILE`` so the user's
        ``.git/index`` is never touched.  Adapts the pattern from
        ``teleport/git.py:158-159`` (``add -N .`` + ``diff HEAD --binary``).

        Paths in *carry_ignored* are excluded from the diff so they can be
        symlinked instead (symlinks are cheaper than copying dep trees).

        Uses subprocess directly (not GitPython) for the diff/apply to handle
        binary patch data correctly — GitPython's string return corrupts
        binary patches.
        """
        import subprocess

        # Get the path to the real index file.
        index_path = Path(repo.git.rev_parse("--git-path", "index"))

        # Create a temp copy of the index.
        with tempfile.NamedTemporaryFile(
            suffix=".idx", delete=False, dir=str(index_path.parent)
        ) as tmp:
            tmp_idx = Path(tmp.name)
        shutil.copy2(index_path, tmp_idx)

        try:
            # add -N . and diff HEAD under the temp index, capturing raw bytes.
            env = dict(os.environ, GIT_INDEX_FILE=str(tmp_idx))
            repo_root = Path(repo.working_tree_dir)

            # add -N . adds all untracked files as intent-to-add so they
            # appear in the diff.  Gitignored files are skipped automatically.
            subprocess.run(
                ["git", "add", "-N", "."],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
            )

            # Build diff pathspec: everything EXCEPT untracked carry_ignored
            # paths (those are symlinked instead). Tracked carry_ignored paths
            # (e.g. a committed .env with uncommitted edits) must be carried as
            # a normal diff -- excluding them silently drops the user's changes
            # and leaves the worktree on the stale committed version.
            diff_pathspecs = []
            for name in carry_ignored:
                if repo.git.ls_files(name).strip():
                    # Tracked in HEAD -- carry its dirty diff, don't exclude.
                    continue
                diff_pathspecs.append(f":(exclude){name}")

            result = subprocess.run(
                ["git", "diff", "HEAD", "--binary", "--", *diff_pathspecs],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
            )
            patch_bytes = result.stdout

            if patch_bytes.strip():
                # Write the patch to a temp file and apply in the worktree.
                with tempfile.NamedTemporaryFile(
                    suffix=".patch", delete=False, dir=str(worktree_path)
                ) as pf:
                    pf.write(patch_bytes)
                    patch_file = Path(pf.name)
                try:
                    subprocess.run(
                        ["git", "apply", str(patch_file)],
                        cwd=str(worktree_path),
                        check=True,
                        capture_output=True,
                    )
                finally:
                    patch_file.unlink(missing_ok=True)
        finally:
            tmp_idx.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # symlink deps
    # ------------------------------------------------------------------

    def _symlink_deps(
        self, original_root: Path, worktree_path: Path, config: WorktreeConfig
    ) -> list[Path]:
        """Symlink gitignored dep directories from the original repo."""
        symlinks: list[Path] = []
        for name in config.carry_ignored:
            src = original_root / name
            dst = worktree_path / name
            if not src.exists() or dst.exists():
                continue
            try:
                os.symlink(src, dst)
                symlinks.append(dst)
                logger.debug("Symlinked %s -> %s", dst, src)
            except OSError as exc:
                logger.warning("Failed to symlink %s: %s", name, exc)
        return symlinks

    # ------------------------------------------------------------------
    # crash recovery
    # ------------------------------------------------------------------

    def _prune_and_report(self, config: WorktreeConfig) -> None:
        """Prune stale worktrees and report orphan branches."""
        try:
            repo = self._get_repo(Path.cwd())
            repo.git.worktree("prune")
        except Exception as exc:
            logger.debug("git worktree prune failed: %s", exc)
            return

        # Report orphan branches (do not auto-delete — never lose work).
        try:
            branches_output = repo.git.branch("--list", f"{config.branch_prefix}*")
            for line in branches_output.splitlines():
                branch_name = line.strip().lstrip("* ").strip()
                if not branch_name:
                    continue
                # Check if this branch has a live worktree.
                wt_list = repo.git.worktree("list", "--porcelain")
                if branch_name not in wt_list:
                    logger.info(
                        "Orphan worktree branch detected: %s (kept for recovery).",
                        branch_name,
                    )
        except Exception as exc:
            logger.debug("Orphan branch sweep failed: %s", exc)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_repo(self, path: Path) -> Repo:
        """Get a GitPython Repo for *path*, searching parent dirs."""
        return Repo(str(path), search_parent_directories=True)

    def _is_mid_operation(self, repo_root: Path) -> bool:
        git_dir = repo_root / ".git"
        # In a worktree, .git is a file pointing to the real git dir.
        if git_dir.is_file():
            content = git_dir.read_text().strip()
            if content.startswith("gitdir:"):
                git_dir = Path(content.split("gitdir:", 1)[1].strip())
                if not git_dir.is_absolute():
                    git_dir = (repo_root / git_dir).resolve()
        for marker in _MID_OPERATION_MARKERS:
            if (git_dir / marker).exists():
                return True
        return False

    def _has_dirty_submodules(self, repo: Repo) -> bool:
        try:
            output = repo.git.submodule("status")
            for line in output.splitlines():
                if line.startswith(("+", "-", "U")):
                    return True
            return False
        except GitCommandError:
            return False

    def _is_dirty(self, repo: Repo) -> bool:
        try:
            return bool(repo.is_dirty(untracked_files=True))
        except Exception:
            return False

    def _wip_commit(self, repo: Repo, handle: WorktreeHandle) -> bool:
        """WIP-commit dirty state onto the branch. Never discards work.

        Returns True if the commit succeeded (or there was nothing to commit),
        False if the commit failed so the caller can skip auto-ff.
        """
        import subprocess

        try:
            repo.git.add("-A")
            # Use env vars for git identity (not -c flags, which conflict with -m).
            env = dict(
                os.environ,
                GIT_AUTHOR_NAME=_FALLBACK_GIT_NAME,
                GIT_AUTHOR_EMAIL=_FALLBACK_GIT_EMAIL,
                GIT_COMMITTER_NAME=_FALLBACK_GIT_NAME,
                GIT_COMMITTER_EMAIL=_FALLBACK_GIT_EMAIL,
            )
            subprocess.run(
                ["git", "commit", "-m", "WIP: vibe session auto-save", "--no-verify"],
                cwd=str(handle.worktree_path),
                env=env,
                check=True,
                capture_output=True,
            )
            logger.info("WIP-committed dirty worktree state to branch %s", handle.branch)
            return True
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or b"").decode("utf-8", errors="replace")
            if "nothing to commit" in err.lower():
                logger.debug("Nothing to WIP-commit in worktree %s", handle.worktree_path)
                return True
            logger.warning("WIP-commit failed: %s. Worktree kept for recovery.", err)
            return False

    def _try_auto_ff(self, handle: WorktreeHandle) -> bool:
        """Attempt a fast-forward merge into the original repo.

        Returns ``True`` if merged, ``False`` if manual handoff is needed.
        """
        try:
            root_repo = self._get_repo(handle.original_repo_root)
            current_head = root_repo.head.commit.hexsha
            if current_head != handle.create_head_sha:
                logger.info(
                    "Auto-ff skipped: HEAD moved (%s != %s). Manual merge needed.",
                    current_head[:8],
                    handle.create_head_sha[:8],
                )
                return False
            if root_repo.is_dirty(untracked_files=False):
                logger.info("Auto-ff skipped: original tree is dirty. Manual merge needed.")
                return False
            root_repo.git.merge("--ff-only", handle.branch)
            logger.info("Auto-ff merged branch %s into %s", handle.branch, handle.original_repo_root)
            return True
        except (GitCommandError, Exception) as exc:
            logger.info("Auto-ff failed, manual merge needed: %s", exc)
            return False

    def _cleanup_partial(self) -> None:
        """Best-effort cleanup of a partially-created worktree."""
        # Nothing specific to clean — the worktree add either succeeded or
        # didn't. If it did, the branch persists for recovery.
        pass

    # ------------------------------------------------------------------
    # cleanup backstops
    # ------------------------------------------------------------------

    def _register_cleanup_backstops(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True
        if not self._signal_handlers_installed:
            for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                try:
                    signal.signal(sig, self._signal_handler)
                except (OSError, ValueError):
                    pass
            self._signal_handlers_installed = True

    def _atexit_cleanup(self) -> None:
        if self._active is not None:
            if self._signal_received is not None:
                logger.info(
                    "atexit (after signal %d): cleaning up worktree %s",
                    self._signal_received,
                    self._active.branch,
                )
            else:
                logger.info("atexit: cleaning up worktree %s", self._active.branch)
            self.exit(self._active)

    def _signal_handler(self, signum: int, frame: object) -> None:
        # Async-signal-safe: only set a flag and re-raise. The heavy teardown
        # (fork/exec git, chdir) is deferred to the already-registered atexit
        # handler, which runs during normal interpreter shutdown. Doing git
        # subprocess work inside the handler can deadlock if the signal
        # interrupts a non-reentrant libc/allocator call.
        self._signal_received = signum
        # Re-raise default handler so the process exits with the right status;
        # atexit handlers still run on the way down.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


worktree_manager = WorktreeManager()
