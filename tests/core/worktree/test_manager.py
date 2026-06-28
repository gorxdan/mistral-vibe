"""Tests for vibe.core.worktree.manager.WorktreeManager."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

from git import Repo
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

    def test_default_merge_is_auto_ff(self):
        # Companion to the mode flip: clean sessions should auto-ff-merge on
        # exit rather than leaving a throwaway branch for manual merge.
        from vibe.core.config import VibeConfig

        config = VibeConfig.load()
        assert config.worktree.merge == "auto-ff"


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

    def test_carries_tracked_carry_ignored_file(
        self, manager: WorktreeManager, temp_repo: Path, tmp_path: Path
    ):
        """WT-3: a tracked file in carry_ignored (e.g. .env) with uncommitted
        edits must be carried into the worktree, not silently dropped to the
        stale committed version.
        """
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
# Auto-ff merge
# ---------------------------------------------------------------------------


class TestAutoFf:
    def test_auto_ff_succeeds_when_head_unchanged(
        self, manager: WorktreeManager, temp_repo: Path
    ):
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

    def test_auto_ff_rebases_when_head_moved(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", merge="auto-ff", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None

        # A concurrent session advances main (disjoint file) after we branched.
        root_repo = Repo(str(temp_repo))
        (temp_repo / "other.txt").write_text("other\n")
        root_repo.git.add("-A")
        root_repo.git.commit("-m", "Advanced main")

        # We commit our own disjoint work in the worktree.
        wt_repo = Repo(str(handle.worktree_path))
        (handle.worktree_path / "new.txt").write_text("content\n")
        wt_repo.git.add("-A")
        wt_repo.git.commit("-m", "Worktree commit")

        manager.exit(handle)

        # Rebased onto the advanced main and fast-forwarded -> BOTH land.
        assert manager.active is None
        log = root_repo.git.log("--oneline")
        assert "Worktree commit" in log
        assert "Advanced main" in log
        assert (temp_repo / "new.txt").read_text() == "content\n"
        assert (temp_repo / "other.txt").read_text() == "other\n"

    def test_auto_ff_held_on_rebase_conflict(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        os.chdir(str(temp_repo))
        config = WorktreeConfig(mode="on", merge="auto-ff", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None

        # Main advances by editing the SAME file the worktree edits -> rebase
        # cannot apply cleanly.
        root_repo = Repo(str(temp_repo))
        (temp_repo / "src.py").write_text("print('main change')\n")
        root_repo.git.add("-A")
        root_repo.git.commit("-m", "Main edits src")

        wt_repo = Repo(str(handle.worktree_path))
        (handle.worktree_path / "src.py").write_text("print('worktree change')\n")
        wt_repo.git.add("-A")
        wt_repo.git.commit("-m", "Worktree edits src")

        manager.exit(handle)

        # Conflict -> rebase aborted, branch kept (stranded), main untouched.
        assert manager.active is None
        assert handle.branch in [b.name for b in root_repo.branches]
        assert "Worktree edits src" not in root_repo.git.log("--oneline")
        assert (temp_repo / "src.py").read_text() == "print('main change')\n"

    def test_auto_ff_lands_over_dirty_start(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        """Session started with a dirty tree still auto-merges on exit: both the
        user's pre-existing edit and the agent's work land in the original tree.
        """
        os.chdir(str(temp_repo))
        root = Repo(str(temp_repo))
        # Dirty start: an uncommitted edit to a tracked file.
        (temp_repo / "src.py").write_text("print('user edit')\n")

        config = WorktreeConfig(mode="on", merge="auto-ff", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None
        assert handle.entry_dirty_fingerprint is not None

        # Agent does disjoint work in the worktree and commits.
        (handle.worktree_path / "agent.txt").write_text("agent work\n")
        wt_repo = Repo(str(handle.worktree_path))
        wt_repo.git.add("-A")
        wt_repo.git.commit("-m", "agent commit")

        manager.exit(handle)

        # Both the user's dirty edit AND the agent's work landed; tree is clean
        # (the user's previously-uncommitted edit is now committed) and the
        # worktree directory was reclaimed.
        assert (temp_repo / "src.py").read_text() == "print('user edit')\n"
        assert (temp_repo / "agent.txt").read_text() == "agent work\n"
        assert not root.is_dirty(untracked_files=True)
        assert not handle.worktree_path.exists()
        # The redundant mergeback stash was dropped (no accumulation).
        assert root.git.stash("list") == ""

    def test_auto_ff_held_when_concurrent_edit_changes_original(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        """If the original tree changes after enter (a concurrent writer), the
        merge is held so that concurrent work is never swept into the stash and
        dropped — the branch is kept for a manual merge instead.
        """
        os.chdir(str(temp_repo))
        root = Repo(str(temp_repo))
        (temp_repo / "src.py").write_text("print('user edit')\n")  # dirty start

        config = WorktreeConfig(mode="on", merge="auto-ff", cleanup="remove")
        handle = manager.enter("test", config)
        assert handle is not None

        # Concurrent writer adds work to the original tree AFTER enter — the
        # carried fingerprint no longer matches.
        (temp_repo / "concurrent.txt").write_text("concurrent\n")

        (handle.worktree_path / "agent.txt").write_text("agent\n")
        wt_repo = Repo(str(handle.worktree_path))
        wt_repo.git.add("-A")
        wt_repo.git.commit("-m", "agent")

        manager.exit(handle)

        # Merge held: branch kept, concurrent + user work preserved untouched,
        # agent work stays on the branch (not force-landed).
        assert handle.branch in [b.name for b in root.branches]
        assert (temp_repo / "concurrent.txt").read_text() == "concurrent\n"
        assert (temp_repo / "src.py").read_text() == "print('user edit')\n"
        assert not (temp_repo / "agent.txt").exists()

    def test_auto_ff_over_dirty_preserves_untracked_carry_ignored(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        """An untracked carry_ignored file (e.g. .env) is excluded from the carry
        and must survive the exit stash-bracket rather than being swept + dropped.
        """
        os.chdir(str(temp_repo))
        (temp_repo / "src.py").write_text("print('user edit')\n")  # tracked dirt
        (temp_repo / ".env").write_text("SECRET=1\n")  # untracked carry_ignored

        config = WorktreeConfig(
            mode="on", merge="auto-ff", cleanup="remove", carry_ignored=[".env"]
        )
        handle = manager.enter("test", config)
        assert handle is not None

        (handle.worktree_path / "agent.txt").write_text("agent\n")
        wt_repo = Repo(str(handle.worktree_path))
        wt_repo.git.add("-A")
        wt_repo.git.commit("-m", "agent")

        manager.exit(handle)

        # .env preserved verbatim; the agent's work landed; user edit landed.
        assert (temp_repo / ".env").read_text() == "SECRET=1\n"
        assert (temp_repo / "agent.txt").read_text() == "agent\n"
        assert (temp_repo / "src.py").read_text() == "print('user edit')\n"


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
        """A stranded branch whose name is a substring of a LIVE worktree branch
        must not be hidden by a loose substring liveness test.
        """
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
# Stash-bracket helpers (used by the dirty-start fast-forward)
# ---------------------------------------------------------------------------


class TestStashBracketHelpers:
    def test_stash_ref_resolved_by_message(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        # `git stash list` prefixes the subject with "On <branch>: ", so the
        # resolver must match the message as a suffix, not by equality.
        os.chdir(str(temp_repo))
        root = Repo(str(temp_repo))
        (temp_repo / "src.py").write_text("dirty\n")
        msg = "vibe-mergeback vibe/x 111 222"
        root.git.stash("push", "-m", msg)
        ref = manager._stash_ref_for_message(root, msg)
        assert ref == "stash@{0}"
        root.git.stash("drop", ref)

    def test_restore_stash_pops_back_dirt(
        self, manager: WorktreeManager, temp_repo: Path
    ):
        # Models the ff-refused-after-stash failure path: the live dirt must be
        # popped back, never abandoned in an orphaned stash with a clean tree.
        os.chdir(str(temp_repo))
        root = Repo(str(temp_repo))
        (temp_repo / "src.py").write_text("dirty edit\n")
        msg = "vibe-mergeback vibe/y 333 444"
        root.git.stash("push", "-m", msg)
        assert not root.is_dirty()  # stashed -> clean

        manager._restore_stash(root, msg)

        assert (temp_repo / "src.py").read_text() == "dirty edit\n"  # restored
        assert root.git.stash("list") == ""  # no orphaned stash


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
