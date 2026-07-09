from __future__ import annotations

from pathlib import Path

from git import Repo

from vibe.core._workspace_verification import workspace_fingerprint
from vibe.core.utils.io import write_safe


def _repo(path: Path) -> Repo:
    repo = Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")
    write_safe(path / "tracked.txt", "one\n")
    repo.index.add(["tracked.txt"])
    repo.index.commit("initial")
    return repo


def test_workspace_fingerprint_changes_with_tracked_and_untracked_content(
    tmp_path: Path, monkeypatch
) -> None:
    _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    initial = workspace_fingerprint()

    write_safe(tmp_path / "tracked.txt", "two\n")
    tracked = workspace_fingerprint()
    write_safe(tmp_path / "untracked.txt", "new\n")
    untracked = workspace_fingerprint()

    assert initial is not None
    assert tracked is not None
    assert untracked is not None
    assert len({initial, tracked, untracked}) == 3


def test_workspace_fingerprint_fails_closed_outside_git(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert workspace_fingerprint() is None
