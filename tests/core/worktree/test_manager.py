"""Tests for vibe.core.worktree.manager.WorktreeManager."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from git import Repo
from git.exc import GitCommandError

from vibe.core.config import WorktreeConfig
from vibe.core.worktree.manager import (
    WorktreeHandle,
    WorktreeManager,
    original_working_directory,
    worktree_enabled,
    worktree_manager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a real git repo with an initial commit."""
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    repo = Repo.init(str(repo_dir))
    # Configure git identity for commits.
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@test.com")
    (repo_dir / "README.md").write_text("# Test Repo\n")
    (repo_dir / "src.py").write_text("print('hello')\n")
    repo.index.add(["README.md", "src.py"])
    repo.index.commit("Initial commit")
    return repo_dir


@pytest.fixture
def manager() -> WorktreeManager:
    """Provide a fresh WorktreeManager for each test."""
    return WorktreeManager()


@pytest.fixture(autouse=True)
def save_restore_cwd():
    """Save and restore the process cwd around every test."""
    original = os.getcwd()
    yield
    os.chdir(original)
    # Also clear the global singleton if it was set.
    worktree_manager._active = None


# ---------------------------------------------------------------------------
# worktree_enabled()
# ---------------------------------------------------------------------------


class TestWorktreeEnabled:
    def test_off_mode(self):
        from vibe.core.config import VibeConfig

        config = VibeConfig.load(worktree=WorktreeConfig(mode="off"))
        assert not worktree_enabled(config, programmatic=True)
        assert not worktree_enabled(config, programmatic=False, cli_flag=True)

    def test_on_mode(self):
        from vibe.core.config import VibeConfig

        config = VibeConfig.load(worktree=WorktreeConfig(mode="on"))
        assert worktree_enabled(config, programmatic=False)
        assert worktree_enabled(config, programmatic=False, cli_flag=False)

    def test_auto_by_entrypoint_programmatic(self):
        from vibe.core.config import VibeConfig

        config = VibeConfig.load(worktree=WorktreeConfig(mode="auto-by-entrypoint"))
        assert worktree_enabled(config, programmatic=True)
        assert not worktree_enabled(config, programmatic=False)

    def test_auto_by_entrypoint_cli_flag(self):
        from vibe.core.config import VibeConfig

        config = VibeConfig.load(worktree=WorktreeConfig(mode="auto-by-entrypoint"))
        assert worktree_enabled(config, programmatic=False, cli_flag=True)


# ---------------------------------------------------------------------------
# original_working_directory()
# ---------------------------------------------------------------------------


class TestOriginalWorkingDirectory:
    def test_returns_cwd_when_no_worktree(self):
        worktree_manager._active = None
        result = original_working_directory()
        assert result == str(Path.cwd())

    def test_returns_original_root_when_active(self, tmp_path: Path):
        handle = WorktreeHandle(
            original_repo_root=tmp_path / "original",
            worktree_path=tmp_path / "worktree",
            branch="vibe/test-123",
            create_head_sha="abc123",
        )
        worktree_manager._active = handle
        result = original_working_directory()
        assert result == str(tmp_path / "original")
        worktree_manager._active = None


# ---------------------------------------------------------------------------
# enter() — basic lifecycle
# ---------------------------------------------------------------------------


class TestEnterBasic:
    def test_enter_creates_worktree_and_chdirs(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)

        assert handle is not None
        assert handle.original_repo_root == temp_repo.resolve()
        assert handle.worktree_path.exists()
        assert handle.worktree_path.is_dir()
        assert Path.cwd() == handle.worktree_path
        assert handle.branch.startswith("vibe/test-")
        assert handle.create_head_sha

        # The worktree should be a real git checkout.
        wt_repo = Repo(str(handle.worktree_path))
        assert wt_repo.head.commit.hexsha == handle.create_head_sha

    def test_enter_records_active_handle(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)

        assert manager.active is handle

    def test_enter_creates_branch_with_prefix(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", branch_prefix="wt/")
        handle = manager.enter("label", config)

        assert handle is not None
        assert handle.branch.startswith("wt/label-")

    def test_enter_uses_configured_base_dir(self, manager: WorktreeManager, temp_repo: Path, tmp_path: Path):
        os.chdir(str(temp_repo))
        custom_base = tmp_path / "custom-wt"
        config = WorktreeConfig(mode="on", base_dir=str(custom_base))
        handle = manager.enter("test", config)
        assert handle is not None
        assert str(handle.worktree_path).startswith(str(custom_base))
        manager.exit(handle)


# ---------------------------------------------------------------------------
# enter() — nested guard
# ---------------------------------------------------------------------------


class TestNestedGuard:
    def test_enter_refuses_when_already_active(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("first", config)
        assert handle is not None

        with pytest.raises(RuntimeError, match="already active"):
            manager.enter("second", config)


# ---------------------------------------------------------------------------
# enter() — graceful refusal
# ---------------------------------------------------------------------------


class TestGracefulRefusal:
    def test_refuses_mid_merge(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        # Simulate a merge in progress.
        (temp_repo / ".git" / "MERGE_HEAD").write_text("abc123\n")

        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)
        assert handle is None
        assert manager.active is None

    def test_refuses_mid_rebase(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        rebase_dir = temp_repo / ".git" / "rebase-merge"
        rebase_dir.mkdir()

        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)
        assert handle is None


# ---------------------------------------------------------------------------
# exit() — teardown
# ---------------------------------------------------------------------------


class TestExit:
    def test_exit_cleans_up_worktree(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None

        wt_path = handle.worktree_path
        branch = handle.branch
        assert wt_path.exists()

        manager.exit(handle)

        assert manager.active is None
        assert not wt_path.exists()
        # Branch should persist.
        repo = Repo(str(temp_repo))
        assert branch in [b.name for b in repo.branches]

    def test_exit_chdirs_back_to_original(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)
        assert handle is not None
        assert Path.cwd() == handle.worktree_path

        manager.exit(handle)
        assert Path.cwd() == temp_repo.resolve()

    def test_exit_wip_commits_dirty_state(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None

        # Make the worktree dirty.
        (handle.worktree_path / "new_file.txt").write_text("uncommitted content\n")
        (handle.worktree_path / "src.py").write_text("print('modified')\n")

        manager.exit(handle)

        # The branch should have a WIP commit.
        repo = Repo(str(temp_repo))
        commit_msg = repo.git.log(handle.branch, "--oneline", "-1")
        assert "WIP" in commit_msg

    def test_exit_keeps_worktree_on_keep_mode(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", cleanup="keep")
        handle = manager.enter("test", config)
        assert handle is not None

        manager.exit(handle)
        # Worktree should still exist.
        assert handle.worktree_path.exists()


# ---------------------------------------------------------------------------
# Dirty carry
# ---------------------------------------------------------------------------


class TestDirtyCarry:
    def test_carries_tracked_modifications(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        # Make the original repo dirty.
        (temp_repo / "src.py").write_text("print('modified in original')\n")

        config = WorktreeConfig(mode="on", carry_dirty=True)
        handle = manager.enter("test", config)
        assert handle is not None

        # The modification should be carried into the worktree.
        wt_content = (handle.worktree_path / "src.py").read_text()
        assert "modified in original" in wt_content

    def test_carries_untracked_files(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        # Create an untracked file.
        (temp_repo / "untracked.txt").write_text("new untracked file\n")

        config = WorktreeConfig(mode="on", carry_dirty=True)
        handle = manager.enter("test", config)
        assert handle is not None

        # The untracked file should appear in the worktree.
        assert (handle.worktree_path / "untracked.txt").exists()
        content = (handle.worktree_path / "untracked.txt").read_text()
        assert "new untracked file" in content

    def test_temp_index_not_polluted(self, manager: WorktreeManager, temp_repo: Path):
        """The user's .git/index must be byte-identical before and after enter."""
        os.chdir(str(temp_repo))
        # Make the repo dirty.
        (temp_repo / "src.py").write_text("print('dirty')\n")
        (temp_repo / "new.txt").write_text("untracked\n")

        index_path = temp_repo / ".git" / "index"
        hash_before = hashlib.sha256(index_path.read_bytes()).hexdigest()

        config = WorktreeConfig(mode="on", carry_dirty=True)
        handle = manager.enter("test", config)
        assert handle is not None

        hash_after = hashlib.sha256(index_path.read_bytes()).hexdigest()
        assert hash_before == hash_after, "User's git index was modified!"


# ---------------------------------------------------------------------------
# Symlink deps
# ---------------------------------------------------------------------------


class TestSymlinkDeps:
    def test_symlinks_node_modules(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        # Create a .gitignore so node_modules is not carried by dirty carry.
        (temp_repo / ".gitignore").write_text("node_modules/\n")
        # Create a node_modules dir (gitignored).
        nm = temp_repo / "node_modules"
        nm.mkdir()
        (nm / "some_pkg").mkdir()
        (nm / "some_pkg" / "index.js").write_text("module.exports = 1;\n")

        config = WorktreeConfig(mode="on", carry_ignored=["node_modules"])
        handle = manager.enter("test", config)
        assert handle is not None

        wt_nm = handle.worktree_path / "node_modules"
        assert wt_nm.is_symlink()
        assert (wt_nm / "some_pkg" / "index.js").read_text() == "module.exports = 1;\n"

    def test_symlinks_recorded_in_handle(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        (temp_repo / "node_modules").mkdir()

        config = WorktreeConfig(mode="on", carry_ignored=["node_modules"])
        handle = manager.enter("test", config)
        assert handle is not None
        assert len(handle.symlinks) == 1
        assert handle.symlinks[0].name == "node_modules"


# ---------------------------------------------------------------------------
# Auto-ff merge
# ---------------------------------------------------------------------------


class TestAutoFf:
    def test_auto_ff_succeeds_when_head_unchanged(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", merge="auto-ff", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None

        # Make a commit in the worktree.
        wt_repo = Repo(str(handle.worktree_path))
        (handle.worktree_path / "new.txt").write_text("content\n")
        wt_repo.git.add("-A")
        wt_repo.git.commit("-m", "Test commit")

        manager.exit(handle)

        # The commit should be merged into the original repo.
        root_repo = Repo(str(temp_repo))
        log = root_repo.git.log("--oneline", "-2")
        assert "Test commit" in log

    def test_auto_ff_falls_back_when_head_moved(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", merge="auto-ff", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None

        # Advance main after create.
        root_repo = Repo(str(temp_repo))
        (temp_repo / "other.txt").write_text("other\n")
        root_repo.git.add("-A")
        root_repo.git.commit("-m", "Advanced main")

        # Make a commit in the worktree.
        wt_repo = Repo(str(handle.worktree_path))
        (handle.worktree_path / "new.txt").write_text("content\n")
        wt_repo.git.add("-A")
        wt_repo.git.commit("-m", "Worktree commit")

        # Should fall back to manual — no crash, branch kept.
        manager.exit(handle)
        assert manager.active is None
        # Branch should still exist.
        assert handle.branch in [b.name for b in root_repo.branches]


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    def test_prune_runs_without_error(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        # Should not raise even with no stale worktrees.
        handle = manager.enter("test", config)
        assert handle is not None
        manager.exit(handle)

    def test_collision_free_branch_names(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle1 = manager.enter("test", config)
        assert handle1 is not None
        branch1 = handle1.branch
        manager.exit(handle1)

        # Second enter should get a different branch name (different timestamp).
        import time

        time.sleep(0.01)
        handle2 = manager.enter("test", config)
        assert handle2 is not None
        assert handle2.branch != branch1
        manager.exit(handle2)


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_enter_exit_round_trip(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        original_cwd = Path.cwd()

        config = WorktreeConfig(mode="on")
        handle = manager.enter("integration", config)
        assert handle is not None
        assert Path.cwd() == handle.worktree_path
        assert Path.cwd() != original_cwd

        # Write a file in the worktree.
        (handle.worktree_path / "agent_output.txt").write_text("agent wrote this\n")

        manager.exit(handle)

        # Back to original cwd.
        assert Path.cwd() == original_cwd.resolve()
        assert manager.active is None

    def test_worktree_has_tracked_files(self, manager: WorktreeManager, temp_repo: Path):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)
        assert handle is not None

        # Tracked files should be present in the worktree.
        assert (handle.worktree_path / "README.md").exists()
        assert (handle.worktree_path / "src.py").exists()
        manager.exit(handle)
