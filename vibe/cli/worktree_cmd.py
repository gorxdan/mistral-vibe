"""The ``vibe worktree`` maintenance subcommand.

Lists, merges, and discards the throwaway worktree branches created by worktree
isolation. Dispatched from :func:`vibe.cli.entrypoint.main` *before* the main
argument parser so it does not collide with the positional ``initial_prompt``.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from filelock import Timeout
from git import Repo
from git.exc import GitCommandError

from vibe.core.worktree.manager import merge_lock, worktree_manager

if TYPE_CHECKING:
    from vibe.core.config import WorktreeConfig

_FLAGS = {"-f", "--force"}


def run_worktree_command(argv: list[str]) -> int:
    """Handle ``vibe worktree <action> [branch]``. *argv* is ``sys.argv[2:]``."""
    force = any(a in _FLAGS for a in argv)
    rest = [a for a in argv if a not in _FLAGS]
    action = rest[0] if rest else "list"
    arg = rest[1] if len(rest) > 1 else None

    if action in {"-h", "--help", "help"}:
        _print_usage()
        return 0
    if action == "list":
        return _cmd_list()
    if action == "merge":
        return _cmd_merge(arg) if arg else _usage_error("merge")
    if action == "discard":
        return _cmd_discard(arg, force=force) if arg else _usage_error("discard")

    print(f"vibe worktree: unknown action {action!r}", file=sys.stderr)
    _print_usage()
    return 2


def _usage_error(action: str) -> int:
    print(f"usage: vibe worktree {action} <branch>", file=sys.stderr)
    return 2


def _print_usage() -> None:
    print(
        "usage: vibe worktree {list | merge <branch> | discard <branch>}\n"
        "  list              show worktree branches holding unmerged work\n"
        "  merge <branch>    rebase-then-fast-forward a worktree branch into HEAD\n"
        "  discard <branch>  delete a worktree branch and its directory "
        "(-f to skip the unmerged-work prompt)",
        file=sys.stderr,
    )


def _load_worktree_config() -> WorktreeConfig:
    """Load the user's [worktree] config; fall back to defaults (this command
    must work without an API key, so a failed full-config load is tolerated).
    """
    try:
        from vibe.core.config import VibeConfig

        return VibeConfig.load().worktree
    except Exception:
        from vibe.core.config import WorktreeConfig

        return WorktreeConfig()


def _repo() -> Repo | None:
    try:
        return Repo(str(Path.cwd()), search_parent_directories=True)
    except Exception:
        print("vibe worktree: not inside a git repository.", file=sys.stderr)
        return None


def _is_merged(repo: Repo, branch: str) -> bool:
    try:
        repo.git.merge_base("--is-ancestor", branch, "HEAD")
        return True
    except GitCommandError:
        return False


def _worktree_dir_for_branch(repo: Repo, branch: str) -> str | None:
    """Return the path of a live worktree checked out on *branch*, if any."""
    try:
        out = repo.git.worktree("list", "--porcelain")
    except GitCommandError:
        return None
    cur_path: str | None = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            cur_path = line[len("worktree ") :].strip()
        elif line.strip() == f"branch refs/heads/{branch}":
            return cur_path
    return None


def _cmd_list() -> int:
    cfg = _load_worktree_config()
    try:
        stranded = worktree_manager.list_stranded_branches(cfg)
    except Exception as exc:
        print(f"vibe worktree: {exc}", file=sys.stderr)
        return 1
    if not stranded:
        print("No worktree branches with unmerged work.")
        return 0
    print(f"{len(stranded)} worktree branch(es) hold unmerged work:")
    for b in stranded:
        print(f"  {b.branch}  ({b.ahead} commit(s), {b.age})")
        print(f"    merge:   vibe worktree merge {b.branch}")
        print(f"    discard: vibe worktree discard {b.branch}")
    return 0


def _cmd_merge(branch: str) -> int:
    repo = _repo()
    if repo is None:
        return 1
    root = Path(repo.working_tree_dir) if repo.working_tree_dir else Path.cwd()
    try:
        # Per-repo lock: two simultaneous merges would otherwise rebase/ff
        # against a HEAD the other just moved.
        with merge_lock(root):
            return _cmd_merge_locked(repo, branch)
    except Timeout:
        print(
            "vibe worktree: another merge is in progress (merge lock busy); "
            "retry shortly.",
            file=sys.stderr,
        )
        return 1


def _cmd_merge_locked(repo: Repo, branch: str) -> int:
    if repo.is_dirty(untracked_files=False):
        print(
            "vibe worktree: working tree is dirty; commit or stash before merging.",
            file=sys.stderr,
        )
        return 1
    if _is_merged(repo, branch):
        # `git merge --ff-only` of an already-merged branch exits 0 with
        # "Already up to date" and moves nothing — do not claim a merge.
        print(
            f"vibe worktree: {branch} is already merged into HEAD; nothing to merge.",
            file=sys.stdout,
        )
        return 0
    try:
        repo.git.merge("--ff-only", branch)
    except GitCommandError:
        # HEAD diverged: rebase the branch onto HEAD, then fast-forward.
        if not _rebase_then_ff(repo, branch):
            print(
                f"vibe worktree: {branch} conflicts with HEAD and can't auto-merge; "
                f"resolve manually or `vibe worktree discard {branch}`.",
                file=sys.stderr,
            )
            return 1
    print(f"Merged {branch} into HEAD.")
    return 0


def _rebase_then_ff(repo: Repo, branch: str) -> bool:
    """Rebase *branch* onto the current branch then fast-forward into it.

    Returns False (leaving the repo back on its original branch) on conflict.
    """
    try:
        original = repo.active_branch.name
    except (TypeError, GitCommandError):
        return False
    head = repo.head.commit.hexsha
    try:
        repo.git.rebase(head, branch)
    except GitCommandError:
        with suppress(GitCommandError):
            repo.git.rebase("--abort")
        with suppress(GitCommandError):
            repo.git.checkout(original)
        return False
    with suppress(GitCommandError):
        repo.git.checkout(original)
    try:
        repo.git.merge("--ff-only", branch)
    except GitCommandError:
        return False
    return True


def _cmd_discard(branch: str, *, force: bool) -> int:
    repo = _repo()
    if repo is None:
        return 1
    if not force and not _is_merged(repo, branch):
        try:
            reply = input(f"{branch} has unmerged commits — really discard? [y/N] ")
        except (EOFError, OSError):
            reply = ""  # no usable stdin (piped / non-tty) -> treat as "no"
        if reply.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1
    dir_path = _worktree_dir_for_branch(repo, branch)
    if dir_path:
        try:
            repo.git.worktree("remove", "--force", dir_path)
        except GitCommandError as exc:
            print(
                f"vibe worktree: could not remove worktree dir {dir_path}: {exc}",
                file=sys.stderr,
            )
    try:
        repo.git.branch("-D", branch)
    except GitCommandError as exc:
        print(
            f"vibe worktree: could not delete branch {branch}: {exc}", file=sys.stderr
        )
        return 1
    try:
        repo.git.worktree("prune")
    except GitCommandError:
        pass
    print(f"Discarded {branch}.")
    return 0
