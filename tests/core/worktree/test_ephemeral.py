from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo

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


def test_changed_worktree_is_kept(repo: Path, base_dir: Path) -> None:
    wt = create_ephemeral_worktree(repo, "dirty", base_dir=base_dir)
    (wt.path / "new.txt").write_text("agent output")
    assert remove_ephemeral_worktree(wt) is False  # kept for manual merge
    assert wt.path.exists()
    # branch retained
    assert wt.branch in [h.name for h in Repo(str(repo)).heads]
    # cleanup
    assert remove_ephemeral_worktree(wt, keep_if_changed=False) is True


def test_committed_worktree_branch_is_recoverable(repo: Path, base_dir: Path) -> None:
    wt = create_ephemeral_worktree(repo, "committed", base_dir=base_dir)
    wt_repo = Repo(str(wt.path))
    (wt.path / "feature.txt").write_text("done")
    wt_repo.index.add(["feature.txt"])
    wt_repo.index.commit("isolated agent work")
    # Advanced past base -> kept even though tree is clean.
    assert remove_ephemeral_worktree(wt) is False
    assert wt.branch in [h.name for h in Repo(str(repo)).heads]
    remove_ephemeral_worktree(wt, keep_if_changed=False)
