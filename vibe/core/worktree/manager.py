from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from vibe.core.config import WorktreeConfig
from vibe.core.logger import logger
from vibe.core.trusted_folders import trusted_folders_manager
from vibe.core.utils.io import read_safe

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

__all__ = [
    "WorktreeError",
    "WorktreeHandle",
    "WorktreeManager",
    "merge_lock",
    "original_working_directory",
    "worktree_enabled",
    "worktree_manager",
]

# Serializes concurrent merges into the root repo (read-head -> rebase -> ff)
# so no merge path ever ff's against a HEAD another one just moved.
_MERGE_LOCK_NAME = "vibe-merge.lock"
_MERGE_LOCK_TIMEOUT_S = 30.0


def merge_lock(repo_root: Path) -> FileLock:
    """The per-repo lock every merge-back path must hold (exit auto-ff, CLI)."""
    return FileLock(
        str(repo_root / ".git" / _MERGE_LOCK_NAME), timeout=_MERGE_LOCK_TIMEOUT_S
    )


_WORKTREE_LEAF_FIELD_COUNT = 3  # "<label>-<pid>-<time_ns>"


class WorktreeError(RuntimeError):
    pass


_MID_OPERATION_MARKERS = [
    "MERGE_HEAD",
    "rebase-merge",
    "rebase-apply",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
]

_FALLBACK_GIT_NAME = "vibe"
_FALLBACK_GIT_EMAIL = "vibe@local"


@dataclass(frozen=True)
class WorktreeHandle:
    original_repo_root: Path
    worktree_path: Path
    branch: str
    create_head_sha: str
    symlinks: list[Path] = field(default_factory=list)
    config: WorktreeConfig = field(default_factory=WorktreeConfig)
    # Fingerprint of the original tree's dirty diff at enter() (sha256 of the
    # carry patch), or None if the tree was clean. Used at exit to confirm the
    # live dirty state still matches what was carried before auto-merging over
    # it (see WorktreeManager._try_auto_ff).
    entry_dirty_fingerprint: str | None = None


@dataclass(frozen=True)
class StrandedBranch:
    branch: str
    ahead: int
    age: str


def worktree_enabled(
    config: VibeConfig, *, programmatic: bool, cli_flag: bool = False
) -> bool:
    m = config.worktree.mode
    if m == "off":
        return False
    if m == "on":
        return True
    return programmatic or cli_flag


def original_working_directory() -> str:
    if worktree_manager.active is not None:
        return str(worktree_manager.active.original_repo_root)
    return _origin_repo_root_for_cwd()


def _origin_repo_root_for_cwd() -> str:
    cwd = Path.cwd()
    try:
        repo = Repo(str(cwd), search_parent_directories=True)
        # `rev-parse --git-common-dir` resolves a linked worktree's `.git` file
        # to the shared ``<main-repo>/.git`` (GitPython's `.common_dir` does
        # not); its parent is the main working tree. The result may be relative
        # to cwd (`.git` in a normal checkout), so join before resolving.
        common = repo.git.rev_parse("--git-common-dir")
        return str((cwd / common).resolve().parent)
    except (InvalidGitRepositoryError, GitCommandError, OSError, ValueError):
        return str(cwd.resolve())


class WorktreeManager:
    def __init__(self) -> None:
        self._active: WorktreeHandle | None = None
        self._atexit_registered = False
        self._signal_handlers_installed = False
        self._signal_received: int | None = None

    @property
    def active(self) -> WorktreeHandle | None:
        return self._active

    def enter(self, label: str, config: WorktreeConfig) -> WorktreeHandle | None:
        if self._active is not None:
            raise RuntimeError(
                "WorktreeManager.enter() called while a worktree is already active. "
                "Subagents must never call enter() — they inherit the parent's cwd."
            )

        try:
            return self._do_enter(label, config)
        except Exception:
            self._cleanup_partial()
            logger.exception("Worktree creation failed")
            # mode="on" expresses an isolation requirement — fail closed so an
            # editing agent does not silently bypass isolation and run on the
            # user's live checkout. auto-by-entrypoint keeps fail-soft (best
            # effort): the requirement is opportunistic, not a guarantee.
            if config.mode == "on":
                raise WorktreeError(
                    "Worktree isolation was requested (mode='on') but could not be "
                    "established. Running in-place would bypass the isolation "
                    "guarantee; refusing to start. See the logged traceback above."
                ) from None
            logger.warning("Running in-place (worktree isolation is best-effort)")
            return None

    def _do_enter(self, label: str, config: WorktreeConfig) -> WorktreeHandle | None:
        self._prune_and_report(config)

        # Resolve original repo root via git (NOT find_git_repo_ancestor,
        # which requires .git to be a dir and misses worktree roots).
        try:
            repo = self._get_repo(Path.cwd())
        except InvalidGitRepositoryError:
            logger.warning(
                "Not a git repository: skipping worktree isolation and "
                "running in-place. Run from inside a git repo (or use "
                "mode='off' / --no-worktree) to suppress this warning."
            )
            return None
        working_tree_dir = repo.working_tree_dir
        if working_tree_dir is None:
            raise RuntimeError("Cannot resolve working tree dir (bare repo?)")
        original_root = Path(working_tree_dir).resolve()

        if self._is_mid_operation(original_root):
            logger.warning(
                "Repo is mid-operation (merge/rebase/cherry-pick). "
                "Skipping worktree isolation — running in-place."
            )
            return None

        if self._has_dirty_submodules(repo):
            logger.warning(
                "Repo has dirty submodules. Skipping worktree isolation — "
                "running in-place. Submodule in-progress edits are not carried."
            )
            return None

        create_head_sha = repo.head.commit.hexsha

        leaf = f"{label}-{os.getpid()}-{time.time_ns()}"
        branch = f"{config.branch_prefix}{leaf}"

        base_dir = Path(config.base_dir)
        repo_name = original_root.name
        worktree_path = base_dir / repo_name / leaf
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        repo.git.worktree("add", str(worktree_path), "-b", branch, "HEAD")
        logger.info("Created worktree at %s on branch %s", worktree_path, branch)

        symlinks: list[Path] = []
        entry_dirty_fingerprint: str | None = None
        if config.carry_dirty:
            entry_dirty_fingerprint = self._dirty_fingerprint(
                repo, config.carry_ignored
            )
            self._carry_dirty(repo, worktree_path, config.carry_ignored)

        symlinks = self._symlink_deps(original_root, worktree_path, config)

        trusted_folders_manager.trust_for_session(worktree_path)

        os.chdir(worktree_path)

        handle = WorktreeHandle(
            original_repo_root=original_root,
            worktree_path=worktree_path,
            branch=branch,
            create_head_sha=create_head_sha,
            symlinks=symlinks,
            config=config,
            entry_dirty_fingerprint=entry_dirty_fingerprint,
        )
        self._active = handle
        self._register_cleanup_backstops()
        return handle

    def exit(self, handle: WorktreeHandle) -> None:
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
            logger.error(
                "Worktree teardown failed: %s. Worktree kept for recovery.", exc
            )
        finally:
            self._active = None

    def _do_exit(self, handle: WorktreeHandle) -> None:
        wt_repo = self._get_repo(handle.worktree_path)

        # Unlink dep symlinks BEFORE the WIP commit so the worktree
        # tree is clean of ephemeral symlinks (they are gitignored, but
        # deleting them after the commit leaves the worktree dirty again
        # and blocks git worktree remove).
        for s in handle.symlinks:
            try:
                s.unlink()
            except OSError as exc:
                logger.warning("Failed to unlink symlink %s: %s", s, exc)

        wip_ok = True
        if self._is_dirty(wt_repo):
            wip_ok = self._wip_commit(wt_repo, handle)

        merged = False
        attempted_ff = handle.config.merge == "auto-ff" and wip_ok
        if attempted_ff:
            merged = self._try_auto_ff(handle)

        # chdir back BEFORE worktree remove (removing cwd leaves stale cwd).
        os.chdir(handle.original_repo_root)

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

        if merged:
            print(
                "\n✓ Your changes were merged into the original checkout.\n",
                file=sys.stdout,
            )
        elif attempted_ff:
            print(
                f"\nYour work is kept on branch {handle.branch} but couldn't merge "
                "automatically (it conflicts with another session's changes).\n"
                f"  Land it later: vibe worktree merge {handle.branch}\n"
                f"  Or discard:    vibe worktree discard {handle.branch}\n",
                file=sys.stdout,
            )
        else:
            print(
                f"\nYour work is kept on branch {handle.branch}.\n"
                f"  Land it:    vibe worktree merge {handle.branch}\n"
                f"  Or discard: vibe worktree discard {handle.branch}\n",
                file=sys.stdout,
            )

    def _carry_exclude_pathspecs(
        self, repo: Repo, carry_ignored: list[str]
    ) -> list[str]:
        specs: list[str] = []
        for name in carry_ignored:
            if repo.git.ls_files(name).strip():
                continue  # tracked -> carry its dirty diff, don't exclude
            specs.append(f":(exclude){name}")
        return specs

    def _compute_dirty_patch(self, repo: Repo, carry_ignored: list[str]) -> bytes:
        # Uses subprocess (not GitPython): GitPython's string return corrupts binary patches.
        import subprocess

        wtd = repo.working_tree_dir
        if wtd is None:
            raise RuntimeError("Cannot resolve working tree dir (bare repo?)")
        repo_root = Path(wtd)

        # `git rev-parse --git-path index` is resolved relative to where git ran
        # (the repo root), not the process cwd — which at exit is the worktree.
        # Anchor it to repo_root so the path is correct regardless of cwd.
        index_path = Path(repo.git.rev_parse("--git-path", "index"))
        if not index_path.is_absolute():
            index_path = repo_root / index_path
        with tempfile.NamedTemporaryFile(
            suffix=".idx", delete=False, dir=str(index_path.parent)
        ) as tmp:
            tmp_idx = Path(tmp.name)
        shutil.copy2(index_path, tmp_idx)

        try:
            env = dict(os.environ, GIT_INDEX_FILE=str(tmp_idx))

            # add -N . adds all untracked files as intent-to-add so they
            # appear in the diff.  Gitignored files are skipped automatically.
            subprocess.run(
                ["git", "add", "-N", "."],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
            )
            diff_pathspecs = self._carry_exclude_pathspecs(repo, carry_ignored)
            result = subprocess.run(
                ["git", "diff", "HEAD", "--binary", "--", *diff_pathspecs],
                cwd=str(repo_root),
                env=env,
                check=True,
                capture_output=True,
            )
            return result.stdout
        finally:
            tmp_idx.unlink(missing_ok=True)

    def _carry_dirty(
        self, repo: Repo, worktree_path: Path, carry_ignored: list[str]
    ) -> None:
        import subprocess

        patch_bytes = self._compute_dirty_patch(repo, carry_ignored)
        if not patch_bytes.strip():
            return
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

    def _dirty_fingerprint(self, repo: Repo, carry_ignored: list[str]) -> str | None:
        import hashlib

        try:
            patch = self._compute_dirty_patch(repo, carry_ignored)
        except (GitCommandError, OSError, RuntimeError) as exc:
            logger.debug("Dirty fingerprint failed: %s", exc)
            return None
        if not patch.strip():
            return None
        return hashlib.sha256(patch).hexdigest()

    def _symlink_deps(
        self, original_root: Path, worktree_path: Path, config: WorktreeConfig
    ) -> list[Path]:
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

    def _prune_and_report(self, config: WorktreeConfig) -> None:
        try:
            repo = self._get_repo(Path.cwd())
            repo.git.worktree("prune")
        except Exception as exc:
            logger.debug("git worktree prune failed: %s", exc)
            return
        self._gc_abandoned_worktrees(repo, config)
        self._reap_dead_pid_worktrees(repo, config)

    @staticmethod
    def _pid_from_leaf(leaf: str) -> int | None:
        # leaf == "<label>-<pid>-<time_ns>"; pid is the second field from the end.
        parts = leaf.rsplit("-", 2)
        if len(parts) != _WORKTREE_LEAF_FIELD_COUNT:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True  # exists but not signalable (e.g. owned by another user)
        return True

    def _reap_dead_pid_worktrees(self, repo: Repo, config: WorktreeConfig) -> None:
        """Force-remove worktree dirs left by dead sessions, older than the GC age.

        The dir leaf encodes the creating pid (``<label>-<pid>-<ns>``). If that
        process is gone and the dir is older than ``gc_age_days``, reap it with
        --force — per the configured aggressive policy this discards any
        uncommitted WIP a crashed session left behind. The live/active worktree
        and any still-running pid are never touched. (Branch GC above keeps
        unmerged branches, so merged history is still recoverable from refs.)
        """
        if config.gc_age_days <= 0:
            return
        try:
            base = Path(config.base_dir).resolve()
            out = repo.git.worktree("list", "--porcelain")
        except Exception as exc:
            logger.debug("reap: worktree list failed: %s", exc)
            return
        active = self._active.worktree_path.resolve() if self._active else None
        cutoff = time.time() - config.gc_age_days * 86400
        for line in out.splitlines():
            if not line.startswith("worktree "):
                continue
            wt = Path(line[len("worktree ") :].strip())
            try:
                rp = wt.resolve()
                if base not in rp.parents or rp == active:
                    continue
                pid = self._pid_from_leaf(rp.name)
                if pid is None or pid == os.getpid() or self._pid_alive(pid):
                    continue  # unknown / self / still-running -> never touch
                if not wt.exists() or wt.stat().st_mtime > cutoff:
                    continue
                repo.git.worktree("remove", "--force", str(wt))
                logger.info("reaped stranded worktree %s (pid %d dead)", wt, pid)
            except (OSError, GitCommandError) as exc:
                logger.debug("reap: skip %s: %s", wt, exc)

    def _gc_abandoned_worktrees(self, repo: Repo, config: WorktreeConfig) -> None:
        if config.gc_age_days <= 0:
            return
        try:
            live = self._live_worktree_branches(repo)
            names = repo.git.branch(
                "--list", f"{config.branch_prefix}*", "--format=%(refname:short)"
            )
        except GitCommandError as exc:
            logger.debug("GC: branch/worktree listing failed: %s", exc)
            return
        cutoff = time.time() - config.gc_age_days * 86400
        for raw in names.splitlines():
            name = raw.strip()
            if not name or name in live:
                continue  # live worktree -> active session, never touch
            if not self._is_ancestor(repo, name, "HEAD"):
                continue  # unmerged work -> keep for recovery, never discard
            try:
                ts = int(repo.git.log("-1", "--format=%ct", name).strip() or 0)
            except GitCommandError:
                continue
            if ts > cutoff:
                continue
            try:
                repo.git.branch("-D", name)
                logger.info(
                    "GC: deleted merged worktree branch %s (>%d days old)",
                    name,
                    config.gc_age_days,
                )
            except GitCommandError as exc:
                logger.debug("GC: could not delete branch %s: %s", name, exc)

    def list_stranded_branches(self, config: WorktreeConfig) -> list[StrandedBranch]:
        repo = self._get_repo(Path.cwd())
        live = self._live_worktree_branches(repo)
        try:
            names = repo.git.branch(
                "--list", f"{config.branch_prefix}*", "--format=%(refname:short)"
            )
        except GitCommandError:
            return []

        stranded: list[StrandedBranch] = []
        for raw in names.splitlines():
            name = raw.strip()
            if not name or name in live:
                continue
            if self._is_ancestor(repo, name, "HEAD"):
                continue
            try:
                ahead = int(repo.git.rev_list("--count", f"HEAD..{name}").strip() or 0)
            except GitCommandError:
                continue
            if ahead == 0:
                continue
            try:
                age = repo.git.log("-1", "--format=%cr", name).strip()
            except GitCommandError:
                age = "unknown age"
            stranded.append(StrandedBranch(branch=name, ahead=ahead, age=age))
        return stranded

    def print_startup_report(self, config: WorktreeConfig) -> None:
        if not config.report_on_startup:
            return
        try:
            stranded = self.list_stranded_branches(config)
        except Exception as exc:
            logger.debug("Startup worktree report failed: %s", exc)
            return
        if not stranded:
            return
        lines = [
            "",
            f"vibe: {len(stranded)} worktree branch(es) hold unmerged work "
            "from prior sessions:",
        ]
        for b in stranded:
            lines.append(f"  {b.branch}  ({b.ahead} commit(s), {b.age})")
            lines.append(
                f"    merge:  git merge {b.branch}"
                f"   (or: vibe worktree merge {b.branch})"
            )
        lines.append("  review/clean up:  vibe worktree list")
        print("\n".join(lines), file=sys.stderr)

    def _is_ancestor(self, repo: Repo, rev: str, ancestor_of: str) -> bool:
        try:
            repo.git.merge_base("--is-ancestor", rev, ancestor_of)
            return True
        except GitCommandError:
            return False

    def _live_worktree_branches(self, repo: Repo) -> set[str]:
        # Parsed from `git worktree list --porcelain` by exact prefix match, not
        # substring: a substring test false-positives when one branch name is a
        # prefix of another (e.g. `vibe/foo` inside `vibe/foobar`).
        try:
            out = repo.git.worktree("list", "--porcelain")
        except GitCommandError:
            return set()
        live: set[str] = set()
        for line in out.splitlines():
            if line.startswith("branch refs/heads/"):
                live.add(line[len("branch refs/heads/") :].strip())
        return live

    def _get_repo(self, path: Path) -> Repo:
        return Repo(str(path), search_parent_directories=True)

    def _is_mid_operation(self, repo_root: Path) -> bool:
        git_dir = repo_root / ".git"
        # In a worktree, .git is a file pointing to the real git dir.
        if git_dir.is_file():
            content = read_safe(git_dir).text.strip()
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
            logger.info(
                "WIP-committed dirty worktree state to branch %s", handle.branch
            )
            return True
        except GitCommandError as exc:
            # repo.git.add("-A") is GitPython and raises GitCommandError (not
            # CalledProcessError), e.g. on index-lock contention. Treat it as a
            # failed WIP commit so the caller skips auto-ff and falls through to
            # the recovery handoff instead of letting it escape teardown.
            logger.warning(
                "WIP-commit staging failed: %s. Worktree kept for recovery.", exc
            )
            return False
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or b"").decode("utf-8", errors="replace")
            if "nothing to commit" in err.lower():
                logger.debug(
                    "Nothing to WIP-commit in worktree %s", handle.worktree_path
                )
                return True
            logger.warning("WIP-commit failed: %s. Worktree kept for recovery.", err)
            return False

    def _try_auto_ff(self, handle: WorktreeHandle) -> bool:
        try:
            root_repo = self._get_repo(handle.original_repo_root)
            with merge_lock(Path(handle.original_repo_root)):
                return self._merge_under_lock(root_repo, handle)
        except Timeout:
            logger.info(
                "Auto-ff: merge lock busy; branch %s kept for retry.", handle.branch
            )
            return False
        except (GitCommandError, Exception) as exc:
            logger.info("Auto-ff failed, manual merge needed: %s", exc)
            return False

    def _merge_under_lock(self, root_repo: Repo, handle: WorktreeHandle) -> bool:
        current_head = root_repo.head.commit.hexsha
        if current_head != handle.create_head_sha:
            if not self._rebase_branch_onto(handle, current_head):
                return False
        # None == no carried dirt -> a plain ff is safe (carry_ignored never blocks).
        now_fp = self._dirty_fingerprint(root_repo, handle.config.carry_ignored)
        if now_fp is None:
            root_repo.git.merge("--ff-only", handle.branch)
            logger.info(
                "Auto-ff merged branch %s into %s",
                handle.branch,
                handle.original_repo_root,
            )
            return True
        if now_fp != handle.entry_dirty_fingerprint:
            logger.info(
                "Auto-ff skipped: original tree dirty and changed since enter "
                "(concurrent edit?). Manual merge needed."
            )
            return False
        return self._stash_ff_drop(root_repo, handle)

    def _rebase_branch_onto(self, handle: WorktreeHandle, base_sha: str) -> bool:
        wt_repo = self._get_repo(handle.worktree_path)
        try:
            wt_repo.git.rebase(base_sha)
            logger.info(
                "Rebased branch %s onto %s for merge-back.", handle.branch, base_sha[:8]
            )
            return True
        except GitCommandError as exc:
            logger.info(
                "Rebase of %s onto %s conflicted (%s); aborting, branch kept.",
                handle.branch,
                base_sha[:8],
                exc,
            )
            try:
                wt_repo.git.rebase("--abort")
            except GitCommandError:
                pass
            return False

    def _stash_ref_for_message(self, repo: Repo, message: str) -> str | None:
        # `git stash drop`/`pop` reject raw commit SHAs; concurrent stashes
        # shift `stash@{0}`, so locate by unique message suffix instead.
        try:
            out = repo.git.stash("list", "--format=%gd %s")
        except GitCommandError:
            return None
        for line in out.splitlines():
            ref, _, msg = line.strip().partition(" ")
            # `git stash list` formats the subject as "On <branch>: <message>",
            # so match the unique message as a suffix, not by equality.
            if msg.endswith(message):
                return ref
        return None

    def _restore_stash(self, repo: Repo, message: str) -> None:
        ref = self._stash_ref_for_message(repo, message)
        if ref is None:
            return
        try:
            repo.git.stash("pop", ref)
        except GitCommandError as exc:
            logger.warning(
                "Could not restore live changes from %s (%s); they remain stashed "
                "(`git stash list`).",
                ref,
                exc,
            )

    def _stash_ff_drop(self, root_repo: Repo, handle: WorktreeHandle) -> bool:
        # Exclude the same untracked carry_ignored paths the carried diff did, so
        # symlinked deps / an untracked .env are never swept into (and dropped
        # with) the stash.
        excludes = self._carry_exclude_pathspecs(root_repo, handle.config.carry_ignored)
        message = f"vibe-mergeback {handle.branch} {os.getpid()} {time.time_ns()}"
        pushed = False
        try:
            out = root_repo.git.stash(
                "push", "--include-untracked", "-m", message, "--", ".", *excludes
            )
            if "No local changes to save" in out:
                # Raced clean between the dirty check and here.
                root_repo.git.merge("--ff-only", handle.branch)
                return True
            pushed = True
            try:
                root_repo.git.merge("--ff-only", handle.branch)
            except GitCommandError as exc:
                # ff refused after stashing: the dirt is only in the stash now and
                # HEAD did not advance, so popping restores it cleanly.
                logger.info(
                    "Auto-ff refused after stash (%s); restoring live changes, "
                    "keeping branch %s.",
                    exc,
                    handle.branch,
                )
                self._restore_stash(root_repo, message)
                return False
            ref = self._stash_ref_for_message(root_repo, message)
            if ref is not None:
                try:
                    root_repo.git.stash("drop", ref)
                except GitCommandError as exc:
                    logger.info(
                        "Redundant mergeback stash %s kept (%s); safe to drop.",
                        ref,
                        exc,
                    )
            logger.info(
                "Auto-ff merged branch %s into %s (over dirty tree)",
                handle.branch,
                handle.original_repo_root,
            )
            return True
        except GitCommandError as exc:
            if pushed:
                self._restore_stash(root_repo, message)
            logger.info("Stash-bracketed auto-ff failed: %s. Manual merge needed.", exc)
            return False

    def _cleanup_partial(self) -> None:
        pass

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
