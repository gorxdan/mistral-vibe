from __future__ import annotations

from pathlib import Path

from git import Repo
import pytest

from vibe.core.worktree.ephemeral import (
    create_ephemeral_worktree,
    remove_ephemeral_worktree,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    r = Repo.init(str(repo_dir))
    with r.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "t@t.com")
    (repo_dir / "f.txt").write_text("base\n")
    r.index.add(["f.txt"])
    r.index.commit("init")
    return repo_dir


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    return tmp_path / "wt"


def test_create_yields_distinct_worktrees(repo: Path, base_dir: Path) -> None:
    a = create_ephemeral_worktree(repo, "lens:x", base_dir=base_dir)
    b = create_ephemeral_worktree(repo, "lens:x", base_dir=base_dir)
    try:
        assert a.path.exists() and b.path.exists()
        assert a.path != b.path
        assert a.branch != b.branch
        assert a.branch.startswith("vibe/iso/")
        # Each is a real checkout of the base content.
        assert (a.path / "f.txt").read_text() == "base\n"
    finally:
        remove_ephemeral_worktree(a, keep_if_changed=False)
        remove_ephemeral_worktree(b, keep_if_changed=False)


def test_clean_worktree_is_removed(repo: Path, base_dir: Path) -> None:
    wt = create_ephemeral_worktree(repo, "clean", base_dir=base_dir)
    assert remove_ephemeral_worktree(wt) is True
    assert not wt.path.exists()
    # branch is gone too
    assert wt.branch not in [h.name for h in Repo(str(repo)).heads]


def test_changed_worktree_branch_is_kept_dir_reclaimed(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "dirty", base_dir=base_dir)
    (wt.path / "new.txt").write_text("agent output")
    # Branch is kept for manual merge (returns False), but the on-disk directory
    # is reclaimed (no unbounded accumulation) and the uncommitted work is
    # committed onto the branch first.
    assert remove_ephemeral_worktree(wt) is False
    assert not wt.path.exists()
    parent = Repo(str(repo))
    assert wt.branch in [h.name for h in parent.heads]
    # The agent's uncommitted file was committed onto the branch before reclaim
    # (cat-file -e exits 0 / empty stdout iff the blob exists on the branch).
    assert parent.git.cat_file("-e", f"{wt.branch}:new.txt") == ""
    # Re-removing with keep_if_changed=False deletes the orphaned branch.
    assert remove_ephemeral_worktree(wt, keep_if_changed=False) is True
    assert wt.branch not in [h.name for h in Repo(str(repo)).heads]


def test_committed_worktree_branch_is_recoverable(repo: Path, base_dir: Path) -> None:
    wt = create_ephemeral_worktree(repo, "committed", base_dir=base_dir)
    wt_repo = Repo(str(wt.path))
    (wt.path / "feature.txt").write_text("done")
    wt_repo.index.add(["feature.txt"])
    wt_repo.index.commit("isolated agent work")
    # Advanced past base -> branch kept (returns False), dir reclaimed.
    assert remove_ephemeral_worktree(wt) is False
    assert not wt.path.exists()
    assert wt.branch in [h.name for h in Repo(str(repo)).heads]
    remove_ephemeral_worktree(wt, keep_if_changed=False)
