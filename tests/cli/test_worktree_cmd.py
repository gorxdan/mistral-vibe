"""Tests for the `vibe worktree` maintenance subcommand."""

from __future__ import annotations

import os
from pathlib import Path

from git import Repo
import pytest

from vibe.cli.worktree_cmd import run_worktree_command


@pytest.fixture(autouse=True)
def _restore_cwd():
    original = os.getcwd()
    yield
    os.chdir(original)


def _wt_dir(root: Repo) -> Path:
    """The repo's working tree directory as a `Path` (never bare-None)."""
    working_tree_dir = root.working_tree_dir
    assert working_tree_dir is not None
    return Path(working_tree_dir)


def _repo_with_orphan_branch(tmp_path: Path) -> tuple[Repo, str]:
    """A repo with one unmerged `vibe/*` branch (no live worktree)."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    root = Repo.init(str(repo_dir))
    with root.config_writer() as cw:
        cw.set_value("user", "name", "T")
        cw.set_value("user", "email", "t@t")
    (repo_dir / "f.txt").write_text("base\n")
    root.git.add("-A")
    root.git.commit("-m", "base")

    wt = tmp_path / "wt"
    root.git.worktree("add", str(wt), "-b", "vibe/orphan", "HEAD")
    (wt / "work.txt").write_text("agent work\n")
    r = Repo(str(wt))
    r.git.add("-A")
    r.git.commit("-m", "agent work")
    root.git.worktree("remove", "--force", str(wt))
    os.chdir(str(repo_dir))
    return root, "vibe/orphan"


def test_list_shows_unmerged_branch(tmp_path: Path, capsys) -> None:
    _repo_with_orphan_branch(tmp_path)
    assert run_worktree_command(["list"]) == 0
    out = capsys.readouterr().out
    assert "vibe/orphan" in out
    assert "vibe worktree merge vibe/orphan" in out


def test_list_empty(tmp_path: Path, capsys) -> None:
    repo_dir = tmp_path / "clean"
    repo_dir.mkdir()
    root = Repo.init(str(repo_dir))
    with root.config_writer() as cw:
        cw.set_value("user", "name", "T")
        cw.set_value("user", "email", "t@t")
    (repo_dir / "f.txt").write_text("x\n")
    root.git.add("-A")
    root.git.commit("-m", "base")
    os.chdir(str(repo_dir))
    assert run_worktree_command(["list"]) == 0
    assert "No worktree branches" in capsys.readouterr().out


def test_merge_lands_branch(tmp_path: Path) -> None:
    root, branch = _repo_with_orphan_branch(tmp_path)
    assert run_worktree_command(["merge", branch]) == 0
    # The branch's work is now in HEAD.
    assert (_wt_dir(root) / "work.txt").read_text() == "agent work\n"


def test_merge_already_merged_branch_does_not_claim_success(
    tmp_path: Path, capsys
) -> None:
    # Re-merging a branch already in HEAD is a no-op: `git merge --ff-only`
    # exits 0 with "Already up to date" and moves nothing. The command must
    # NOT print a false "Merged ... into HEAD." in that case.
    root, branch = _repo_with_orphan_branch(tmp_path)
    assert run_worktree_command(["merge", branch]) == 0
    capsys.readouterr()  # drain first merge

    assert run_worktree_command(["merge", branch]) == 0
    out = capsys.readouterr().out
    assert "Merged" not in out
    assert "already merged" in out.lower() or "nothing to merge" in out.lower()


def test_merge_refuses_dirty_tree(tmp_path: Path) -> None:
    root, branch = _repo_with_orphan_branch(tmp_path)
    (_wt_dir(root) / "f.txt").write_text("dirty\n")
    assert run_worktree_command(["merge", branch]) == 1
    # Branch unmerged, work not landed.
    assert not (_wt_dir(root) / "work.txt").exists()


def test_merge_rebases_diverged_branch(tmp_path: Path) -> None:
    root, branch = _repo_with_orphan_branch(tmp_path)
    # Advance main (disjoint) so the branch diverged -> a plain ff would fail.
    (_wt_dir(root) / "main.txt").write_text("main advance\n")
    root.git.add("-A")
    root.git.commit("-m", "main advance")

    assert run_worktree_command(["merge", branch]) == 0
    # Both the branch work and the main advance landed.
    assert (_wt_dir(root) / "work.txt").read_text() == "agent work\n"
    assert (_wt_dir(root) / "main.txt").read_text() == "main advance\n"
    assert "agent work" in root.git.log("--oneline")


def test_merge_keeps_branch_on_conflict(tmp_path: Path) -> None:
    root, branch = _repo_with_orphan_branch(tmp_path)
    # Main adds the same file the branch added (different content) -> rebase
    # conflict; the merge must fail cleanly and keep the branch.
    (_wt_dir(root) / "work.txt").write_text("main version\n")
    root.git.add("-A")
    root.git.commit("-m", "main work.txt")

    assert run_worktree_command(["merge", branch]) == 1
    assert branch in [b.name for b in root.branches]
    assert (_wt_dir(root) / "work.txt").read_text() == "main version\n"


def test_discard_force_deletes_branch(tmp_path: Path) -> None:
    root, branch = _repo_with_orphan_branch(tmp_path)
    assert run_worktree_command(["discard", branch, "--force"]) == 0
    assert branch not in [b.name for b in root.branches]


def test_discard_unmerged_aborts_without_tty(tmp_path: Path) -> None:
    # No -f and no tty -> input() EOFErrors -> treated as "no" -> aborts, branch kept.
    root, branch = _repo_with_orphan_branch(tmp_path)
    assert run_worktree_command(["discard", branch]) == 1
    assert branch in [b.name for b in root.branches]


def test_discard_refuses_locked_worktree(tmp_path: Path, capsys) -> None:
    # F8: discard on a locked (live) worktree must refuse without -f.
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    root = Repo.init(str(repo_dir))
    with root.config_writer() as cw:
        cw.set_value("user", "name", "T")
        cw.set_value("user", "email", "t@t")
    (repo_dir / "f.txt").write_text("base\n")
    root.git.add("-A")
    root.git.commit("-m", "base")

    wt = tmp_path / "wt"
    root.git.worktree("add", str(wt), "-b", "vibe/live", "HEAD")
    root.git.worktree("lock", str(wt), "--reason", "vibe-test-live")
    os.chdir(str(repo_dir))

    assert run_worktree_command(["discard", "vibe/live"]) == 1
    assert "locked" in capsys.readouterr().err.lower()
    assert "vibe/live" in [b.name for b in root.branches]


def test_discard_force_unlocks_and_removes(tmp_path: Path) -> None:
    # F8: -f unlocks and removes a locked worktree.
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    root = Repo.init(str(repo_dir))
    with root.config_writer() as cw:
        cw.set_value("user", "name", "T")
        cw.set_value("user", "email", "t@t")
    (repo_dir / "f.txt").write_text("base\n")
    root.git.add("-A")
    root.git.commit("-m", "base")

    wt = tmp_path / "wt"
    root.git.worktree("add", str(wt), "-b", "vibe/live", "HEAD")
    root.git.worktree("lock", str(wt), "--reason", "vibe-test-live")
    os.chdir(str(repo_dir))

    assert run_worktree_command(["discard", "vibe/live", "--force"]) == 0
    assert "vibe/live" not in [b.name for b in root.branches]


def test_unknown_action(tmp_path: Path) -> None:
    os.chdir(str(tmp_path))
    assert run_worktree_command(["bogus"]) == 2


def test_merge_strands_gracefully_when_merge_lock_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CLI merge must honor the per-repo merge lock.
    from filelock import FileLock

    monkeypatch.setattr("vibe.core.worktree.manager._MERGE_LOCK_TIMEOUT_S", 0.2)
    root, branch = _repo_with_orphan_branch(tmp_path)

    lock_path = _wt_dir(root) / ".git" / "vibe-merge.lock"
    with FileLock(str(lock_path)):
        assert run_worktree_command(["merge", branch]) == 1

    # Nothing landed while the lock was held; branch kept for retry.
    assert not (_wt_dir(root) / "work.txt").exists()
    assert branch in [b.name for b in root.branches]


# From a linked worktree, `merge` can't land on main (targets the worktree HEAD,
# and a sandboxed session can't write main) — it must steer to push+PR, not merge.
def test_merge_from_linked_worktree_offers_push_pr(tmp_path: Path, capsys) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    root = Repo.init(str(repo_dir))
    with root.config_writer() as cw:
        cw.set_value("user", "name", "T")
        cw.set_value("user", "email", "t@t")
    (repo_dir / "f.txt").write_text("base\n")
    root.git.add("-A")
    root.git.commit("-m", "base")

    # a branch with new work we want to land on main
    root.git.branch("vibe/feature")
    fr = Repo(str(repo_dir))
    fr.git.checkout("vibe/feature")
    (repo_dir / "feat.txt").write_text("feature\n")
    fr.git.add("-A")
    fr.git.commit("-m", "feature work")
    fr.git.checkout("master" if "master" in fr.heads else "main")

    # run the command from a LINKED worktree (a sandboxed session's home)
    wt = tmp_path / "wt"
    root.git.worktree("add", str(wt), "-b", "vibe/session", "HEAD")
    session = Repo(str(wt))
    head_before = session.head.commit.hexsha
    os.chdir(str(wt))

    rc = run_worktree_command(["merge", "vibe/feature"])

    err = capsys.readouterr().err
    assert rc == 1, "must not report success from a linked worktree"
    assert "linked worktree" in err.lower()
    assert "git push" in err
    assert "vibe/feature" in err
    # and it did NOT merge vibe/feature into the worktree's HEAD
    assert Repo(str(wt)).head.commit.hexsha == head_before
