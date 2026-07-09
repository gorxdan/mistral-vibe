"""Tests for vibe.core.worktree.manager.WorktreeManager."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

from git import Repo
from git.exc import GitCommandError
import pytest

from vibe.core.config import WorktreeConfig
from vibe.core.worktree.manager import (
    WorktreeError,
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

    def test_default_mode_is_on(self):
        # Regression for the host-default flip: WorktreeConfig() with no
        # explicit mode must default to "on", and worktree_enabled() must
        # return True for both the interactive CLI and programmatic entrypoints
        # under that default. Previously defaulted to "auto-by-entrypoint"
        # (programmatic-only).
        from vibe.core.config import VibeConfig

        config = VibeConfig.load()
        assert config.worktree.mode == "on"
        assert worktree_enabled(config, programmatic=False)  # interactive CLI
        assert worktree_enabled(config, programmatic=True)  # vibe -p


# ---------------------------------------------------------------------------
# original_working_directory()
# ---------------------------------------------------------------------------


class TestOriginalWorkingDirectory:
    def test_returns_resolved_cwd_outside_repo(self, tmp_path: Path):
        worktree_manager._active = None
        os.chdir(str(tmp_path))
        result = original_working_directory()
        assert Path(result) == tmp_path.resolve()

    def test_returns_repo_root_from_main_checkout(self, temp_repo: Path):
        worktree_manager._active = None
        os.chdir(str(temp_repo))
        result = original_working_directory()
        assert Path(result) == temp_repo.resolve()

    def test_returns_origin_root_from_external_worktree(
        self, tmp_path: Path, temp_repo: Path
    ):
        # A worktree this process did NOT enter (active stays None): launched
        # directly inside it. Must still resolve to the main checkout so resume
        # finds sessions recorded under the origin repo.
        worktree_manager._active = None
        wt_path = tmp_path / "external-wt"
        subprocess.run(
            ["git", "-C", str(temp_repo), "worktree", "add", "-q", str(wt_path)],
            check=True,
            capture_output=True,
        )
        os.chdir(str(wt_path))
        result = original_working_directory()
        assert Path(result) == temp_repo.resolve()

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

    def test_prefers_isolated_root_when_env_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        worktree_manager._active = None
        iso_root = tmp_path / "iso-wt"
        iso_root.mkdir()
        monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(iso_root))
        result = original_working_directory()
        assert Path(result) == iso_root.resolve()

    def test_falls_through_when_isolated_root_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        worktree_manager._active = None
        monkeypatch.delenv("VIBE_ISOLATED_WORKTREE_ROOT", raising=False)
        os.chdir(str(tmp_path))
        result = original_working_directory()
        assert Path(result) == tmp_path.resolve()


# ---------------------------------------------------------------------------
# enter() — basic lifecycle
# ---------------------------------------------------------------------------


class TestEnterBasic:
    def test_enter_creates_worktree_and_chdirs(
        self, manager: WorktreeManager, temp_repo: Path
    ):
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

    def test_enter_records_active_handle(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)

        assert manager.active is handle

    def test_enter_creates_branch_with_prefix(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", branch_prefix="wt/")
        handle = manager.enter("label", config)

        assert handle is not None
        assert handle.branch.startswith("wt/label-")

    def test_enter_uses_configured_base_dir(
        self, manager: WorktreeManager, temp_repo: Path, tmp_path: Path
    ):
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
    def test_enter_refuses_when_already_active(
        self, manager: WorktreeManager, temp_repo: Path
    ):
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

    def test_refuses_not_a_git_repo(self, manager: WorktreeManager, tmp_path: Path):
        # Regression: launching from outside a git repo must not hard-fail
        # under mode="on". There is no live checkout to protect, so enter()
        # returns None (run in-place) instead of raising WorktreeError.
        os.chdir(str(tmp_path))

        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)
        assert handle is None
        assert manager.active is None


class TestCreationFailureIsolation:
    """mode='on' fails closed on a creation error; auto-by-entrypoint is soft."""

    def test_on_mode_raises_on_creation_failure(
        self, manager: WorktreeManager, temp_repo: Path, monkeypatch
    ):
        os.chdir(str(temp_repo))

        def _boom(self, label, config):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(WorktreeManager, "_do_enter", _boom)
        config = WorktreeConfig(mode="on")

        with pytest.raises(WorktreeError, match="isolation was requested"):
            manager.enter("test", config)

        assert manager.active is None

    def test_auto_mode_falls_back_in_place_on_creation_failure(
        self, manager: WorktreeManager, temp_repo: Path, monkeypatch
    ):
        os.chdir(str(temp_repo))

        def _boom(self, label, config):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(WorktreeManager, "_do_enter", _boom)
        config = WorktreeConfig(mode="auto-by-entrypoint")

        # Soft failure: returns None (run in-place) rather than raising.
        assert manager.enter("test", config) is None
        assert manager.active is None


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

    def test_exit_chdirs_back_to_original(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)
        assert handle is not None
        assert Path.cwd() == handle.worktree_path

        manager.exit(handle)
        assert Path.cwd() == temp_repo.resolve()

    def test_exit_wip_commits_dirty_state(
        self, manager: WorktreeManager, temp_repo: Path
    ):
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

    def test_exit_keeps_worktree_on_keep_mode(
        self, manager: WorktreeManager, temp_repo: Path
    ):
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
    def test_carries_tracked_modifications(
        self, manager: WorktreeManager, temp_repo: Path
    ):
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

    def test_carries_tracked_carry_ignored_file(
        self, manager: WorktreeManager, temp_repo: Path, tmp_path: Path
    ):
        os.chdir(str(temp_repo))
        repo = Repo(str(temp_repo))
        (temp_repo / ".env").write_text("SECRET=committed\n")
        repo.index.add([".env"])
        repo.index.commit("Add .env")
        # Uncommitted modification to the now-tracked .env.
        (temp_repo / ".env").write_text("SECRET=uncommitted\n")

        config = WorktreeConfig(
            mode="on",
            carry_dirty=True,
            carry_ignored=[".env"],
            base_dir=str(tmp_path / "wt"),
        )
        handle = manager.enter("test", config)
        assert handle is not None

        wt_env = (handle.worktree_path / ".env").read_text()
        assert "uncommitted" in wt_env, "tracked .env dirty edit must be carried"


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

    def test_symlinks_recorded_in_handle(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        (temp_repo / "node_modules").mkdir()

        config = WorktreeConfig(mode="on", carry_ignored=["node_modules"])
        handle = manager.enter("test", config)
        assert handle is not None
        assert len(handle.symlinks) == 1
        assert handle.symlinks[0].name == "node_modules"


# ---------------------------------------------------------------------------
# Stranded-branch reporting + GC
# ---------------------------------------------------------------------------


def _commit_branch_ahead(
    root: Repo, base_wt: Path, branch: str, *, base: str = "HEAD"
) -> None:
    """Create *branch* off *base* with one commit ahead, via a temp worktree."""
    wt = base_wt.parent / f"_tmp_{branch.replace('/', '_')}"
    root.git.worktree("add", str(wt), "-b", branch, base)
    (wt / f"{branch.replace('/', '_')}.txt").write_text("work\n")
    r = Repo(str(wt))
    r.git.add("-A")
    r.git.commit("-m", f"work on {branch}")
    root.git.worktree("remove", "--force", str(wt))


class TestStrandedBranches:
    def test_lists_unmerged_orphan_branch(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        root = Repo(str(temp_repo))
        _commit_branch_ahead(root, temp_repo, "vibe/old-session")

        stranded = manager.list_stranded_branches(WorktreeConfig(branch_prefix="vibe/"))
        names = [s.branch for s in stranded]
        assert "vibe/old-session" in names
        entry = next(s for s in stranded if s.branch == "vibe/old-session")
        assert entry.ahead == 1

    def test_excludes_merged_and_empty_branches(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        root = Repo(str(temp_repo))
        # Merged/empty branch pointing at HEAD — nothing to recover.
        root.git.branch("vibe/empty", "HEAD")

        stranded = manager.list_stranded_branches(WorktreeConfig(branch_prefix="vibe/"))
        assert "vibe/empty" not in [s.branch for s in stranded]


class TestGc:
    @staticmethod
    def _old_repo(tmp_path: Path) -> Repo:
        repo_dir = tmp_path / "gcrepo"
        repo_dir.mkdir()
        root = Repo.init(str(repo_dir))
        old = "2020-01-01T00:00:00"
        env = {
            **os.environ,
            "GIT_AUTHOR_DATE": old,
            "GIT_COMMITTER_DATE": old,
            "GIT_AUTHOR_NAME": "T",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "T",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        (repo_dir / "f.txt").write_text("x\n")
        root.git.add("-A")
        subprocess.run(
            ["git", "commit", "-m", "old base"],
            cwd=str(repo_dir),
            env=env,
            check=True,
            capture_output=True,
        )
        return root

    def test_gc_deletes_merged_old_branch(
        self, manager: WorktreeManager, tmp_path: Path
    ):
        root = self._old_repo(tmp_path)
        wd = root.working_tree_dir
        assert wd is not None
        root.git.branch("vibe/merged-old", "HEAD")  # merged (==HEAD), old date
        os.chdir(wd)

        manager._gc_abandoned_worktrees(root, WorktreeConfig(gc_age_days=7))
        assert "vibe/merged-old" not in [b.name for b in root.branches]

    def test_gc_keeps_unmerged_branch(self, manager: WorktreeManager, tmp_path: Path):
        root = self._old_repo(tmp_path)
        wd = root.working_tree_dir
        assert wd is not None
        _commit_branch_ahead(root, Path(wd), "vibe/unmerged-old")
        os.chdir(wd)

        manager._gc_abandoned_worktrees(root, WorktreeConfig(gc_age_days=7))
        # Unmerged work is never GC'd regardless of age.
        assert "vibe/unmerged-old" in [b.name for b in root.branches]

    def test_gc_disabled_when_age_zero(self, manager: WorktreeManager, tmp_path: Path):
        root = self._old_repo(tmp_path)
        wd = root.working_tree_dir
        assert wd is not None
        root.git.branch("vibe/merged-old", "HEAD")
        os.chdir(wd)

        manager._gc_abandoned_worktrees(root, WorktreeConfig(gc_age_days=0))
        assert "vibe/merged-old" in [b.name for b in root.branches]

    def test_lists_stranded_branch_that_is_substring_of_live(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        root = Repo(str(temp_repo))
        _commit_branch_ahead(root, temp_repo, "vibe/foo")  # stranded, unmerged
        live = temp_repo.parent / "live_foobar"
        root.git.worktree("add", str(live), "-b", "vibe/foobar", "HEAD")
        try:
            names = [
                s.branch
                for s in manager.list_stranded_branches(
                    WorktreeConfig(branch_prefix="vibe/")
                )
            ]
            assert "vibe/foo" in names  # not masked by the live "vibe/foobar"
            assert "vibe/foobar" not in names  # live -> excluded
        finally:
            root.git.worktree("remove", "--force", str(live))


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

    def test_collision_free_branch_names(
        self, manager: WorktreeManager, temp_repo: Path
    ):
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

    def test_worktree_has_tracked_files(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on")
        handle = manager.enter("test", config)
        assert handle is not None

        # Tracked files should be present in the worktree.
        assert (handle.worktree_path / "README.md").exists()
        assert (handle.worktree_path / "src.py").exists()
        manager.exit(handle)


def test_pid_from_leaf_parses_creating_pid() -> None:
    assert WorktreeManager._pid_from_leaf("cli-12345-67890") == 12345
    assert WorktreeManager._pid_from_leaf("programmatic-999-111222333") == 999
    # Unparseable leaves -> None (never reaped).
    assert WorktreeManager._pid_from_leaf("garbage") is None
    assert WorktreeManager._pid_from_leaf("cli-notapid-111") is None


def test_pid_alive_distinguishes_live_from_dead() -> None:
    assert WorktreeManager._pid_alive(os.getpid()) is True
    assert WorktreeManager._pid_alive(0) is False
    # A pid at the top of the range is essentially never live.
    assert WorktreeManager._pid_alive(2**31 - 1) is False


def test_merge_lock_from_linked_worktree_uses_common_gitdir(tmp_path: Path) -> None:
    # Linked worktree .git is a pointer FILE: the lock must resolve to the shared
    # gitdir (same file for all worktrees; FileLock mkdir must not hit the pointer).
    from vibe.core.worktree.manager import merge_lock

    main = tmp_path / "main"
    main.mkdir()
    repo = Repo.init(str(main))
    (main / "a.txt").write_text("x\n")
    repo.git.add("-A")
    repo.git.commit("-m", "init")
    wt = tmp_path / "wt"
    repo.git.worktree("add", "-b", "wt-branch", str(wt))

    lock = merge_lock(wt)
    with lock:
        pass
    assert Path(lock.lock_file).parent == (main / ".git").resolve()


# ---------------------------------------------------------------------------
# F1: lock/unlock live worktrees
# ---------------------------------------------------------------------------


def test_enter_locks_worktree(tmp_path: Path) -> None:
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    repo = Repo.init(str(repo_dir))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@test.com")
    (repo_dir / "a.txt").write_text("x\n")
    repo.git.add("-A")
    repo.git.commit("-m", "init")

    mgr = WorktreeManager()
    os.chdir(str(repo_dir))
    config = WorktreeConfig(mode="on", cleanup="remove")
    handle = mgr.enter("test", config)
    assert handle is not None

    # git worktree list --porcelain shows 'locked' with a reason.
    out = repo.git.worktree("list", "--porcelain")
    assert "locked" in out
    mgr.exit(handle)


def test_locked_worktree_survives_external_remove(tmp_path: Path) -> None:
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    repo = Repo.init(str(repo_dir))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@test.com")
    (repo_dir / "a.txt").write_text("x\n")
    repo.git.add("-A")
    repo.git.commit("-m", "init")

    wt = tmp_path / "wt"
    repo.git.worktree("add", "-b", "wt-branch", str(wt))
    repo.git.worktree("lock", str(wt), "--reason", "vibe-test")

    admin_entries = list((repo_dir / ".git" / "worktrees").iterdir())
    assert len(admin_entries) == 1

    # External remove must refuse on a locked worktree.
    with pytest.raises(GitCommandError):
        repo.git.worktree("remove", str(wt))

    # Admin entry survives.
    admin_after = list((repo_dir / ".git" / "worktrees").iterdir())
    assert len(admin_after) == 1


def test_exit_unlocks_then_removes(tmp_path: Path) -> None:
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    repo = Repo.init(str(repo_dir))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@test.com")
    (repo_dir / "a.txt").write_text("x\n")
    repo.git.add("-A")
    repo.git.commit("-m", "init")

    mgr = WorktreeManager()
    os.chdir(str(repo_dir))
    config = WorktreeConfig(mode="on", cleanup="remove")
    handle = mgr.enter("test", config)
    assert handle is not None
    wt_path = handle.worktree_path

    mgr.exit(handle)

    assert not wt_path.exists()
    # git worktree list no longer shows it.
    out = repo.git.worktree("list", "--porcelain")
    assert str(wt_path) not in out


def test_exit_handles_already_deleted_dir(tmp_path: Path) -> None:
    repo_dir = tmp_path / "myrepo"
    repo_dir.mkdir()
    repo = Repo.init(str(repo_dir))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@test.com")
    (repo_dir / "a.txt").write_text("x\n")
    repo.git.add("-A")
    repo.git.commit("-m", "init")

    mgr = WorktreeManager()
    os.chdir(str(repo_dir))
    config = WorktreeConfig(mode="on", cleanup="remove")
    handle = mgr.enter("test", config)
    assert handle is not None

    # Simulate external deletion: unlock, force-remove the dir, leaving a husk.
    repo.git.worktree("unlock", str(handle.worktree_path))
    import shutil

    shutil.rmtree(handle.worktree_path)

    # exit() must not raise (the incident's teardown crash).
    mgr.exit(handle)
    assert mgr.active is None
