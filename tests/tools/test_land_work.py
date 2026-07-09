from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from git import Repo
import pytest

from vibe.core.agents.manager import AgentManager
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.land_work import (
    LandWork,
    LandWorkArgs,
    LandWorkConfig,
    _require_verification_note,
)
from vibe.core.worktree.manager import WorktreeHandle, worktree_manager


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "mainrepo"
    repo_dir.mkdir()
    repo = Repo.init(str(repo_dir))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@test.com")
    (repo_dir / "README.md").write_text("# main\n")
    repo.index.add(["README.md"])
    repo.index.commit("initial")
    return repo_dir


def _commit(repo: Repo, path: Path, content: str, msg: str) -> None:
    path.write_text(content)
    repo.index.add([path.name])
    repo.index.commit(msg)


@pytest.fixture
def tool() -> LandWork:
    return LandWork(config_getter=lambda: LandWorkConfig(), state=BaseToolState())


@pytest.fixture(autouse=True)
def _clear_worktree_singleton():
    yield
    worktree_manager._active = None


def _set_active_handle(repo_dir: Path, branch: str) -> None:
    worktree_manager._active = WorktreeHandle(
        original_repo_root=repo_dir,
        worktree_path=repo_dir,
        branch=branch,
        create_head_sha="0" * 40,
    )


class TestLandWorkAvailability:
    def test_unavailable_without_active_worktree(self):
        worktree_manager._active = None
        assert LandWork.is_available() is False

    def test_available_with_active_worktree(self, temp_repo):
        _set_active_handle(temp_repo, "vibe/test")
        assert LandWork.is_available() is True


class TestLandWorkMerge:
    def test_merges_branch_into_main_no_ff(self, temp_repo, tool):
        repo = Repo(str(temp_repo))
        # Create a divergent feature branch with a commit NOT on main.
        feat = repo.create_head("feature")
        feat.checkout()
        _commit(repo, temp_repo / "feat.txt", "new feature\n", "feat")
        repo.heads.master.checkout()
        _set_active_handle(temp_repo, "feature")

        results = asyncio.run(_collect(tool, LandWorkArgs()))

        assert len(results) == 1
        r = results[0]
        assert r.merged is True
        assert r.merge_commit_sha is not None
        # --no-ff creates a merge commit with two parents
        merged = Repo(str(temp_repo))
        commit = merged.commit(r.merge_commit_sha)
        assert len(commit.parents) == 2
        assert "feat.txt" in [b.name for b in merged.head.commit.tree.blobs]

    def test_already_merged_returns_not_merged(self, temp_repo, tool):
        repo = Repo(str(temp_repo))
        main_sha = repo.head.commit.hexsha
        # branch HEAD == main HEAD -> already an ancestor
        _set_active_handle(temp_repo, repo.active_branch.name)
        results = asyncio.run(_collect(tool, LandWorkArgs()))
        r = results[0]
        assert r.merged is False
        assert "already merged" in r.message
        assert repo.head.commit.hexsha == main_sha

    def test_dirty_main_refuses(self, temp_repo, tool):
        repo = Repo(str(temp_repo))
        feat = repo.create_head("feature")
        feat.checkout()
        _commit(repo, temp_repo / "feat.txt", "x\n", "feat")
        repo.heads.master.checkout()
        (temp_repo / "README.md").write_text("# dirty\n")  # dirty main AFTER commit
        _set_active_handle(temp_repo, "feature")
        with pytest.raises(Exception, match="dirty"):
            asyncio.run(_collect(tool, LandWorkArgs()))

    def test_no_active_worktree_raises(self, tool):
        worktree_manager._active = None
        with pytest.raises(Exception, match="active worktree"):
            asyncio.run(_collect(tool, LandWorkArgs()))


async def _collect(tool: LandWork, args: LandWorkArgs):
    out = []
    async for r in tool.run(args):
        out.append(r)
    return out


class _FakeConfig:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.verification_subsystem = verification_subsystem


class _FakeAgentManager:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.config = _FakeConfig(verification_subsystem)


class TestLandWorkVerificationNote:
    def test_skips_when_no_ctx(self):
        _require_verification_note(LandWorkArgs(), None)

    def test_skips_when_subsystem_off(self):
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=False)
            ),
        )
        _require_verification_note(LandWorkArgs(), ctx)

    def test_requires_note_when_subsystem_on(self):
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=True)
            ),
        )
        with pytest.raises(ToolError, match="verification_note"):
            _require_verification_note(LandWorkArgs(), ctx)

    def test_accepts_trivial_note_with_reason(self):
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=True)
            ),
        )
        _require_verification_note(
            LandWorkArgs(verification_note="trivial: docs-only"),
            ctx,
            changed_paths=["docs/guide.md"],
        )

    def test_rejects_trivial_note_for_code_diff(self):
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=True)
            ),
        )
        with pytest.raises(ToolError, match="documentation-only"):
            _require_verification_note(
                LandWorkArgs(verification_note="trivial: small fix"),
                ctx,
                changed_paths=["vibe/core/agent_loop.py"],
            )

    def test_rejects_arbitrary_nonempty_note(self):
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=True)
            ),
        )
        with pytest.raises(ToolError, match="cannot authorize"):
            _require_verification_note(
                LandWorkArgs(verification_note="I ran the tests and they passed"), ctx
            )

    def test_rejects_trivial_note_without_reason(self):
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=True)
            ),
        )
        with pytest.raises(ToolError, match="cannot authorize"):
            _require_verification_note(LandWorkArgs(verification_note="trivial:"), ctx)
