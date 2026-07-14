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
    _changed_paths,
    _require_verification_note,
)
from vibe.core.utils.io import write_safe
from vibe.core.verification_state import (
    VerificationReceiptReference,
    VerificationState,
    VerifierAttemptDisposition,
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


def _set_active_handle(
    repo_dir: Path, branch: str, *, worktree_path: Path | None = None
) -> None:
    worktree_manager._active = WorktreeHandle(
        original_repo_root=repo_dir,
        worktree_path=worktree_path or repo_dir,
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
    def test_refuses_without_invoke_context(self, temp_repo, tool):
        repo = Repo(str(temp_repo))
        _set_active_handle(temp_repo, repo.active_branch.name)

        with pytest.raises(ToolError, match="authenticated invocation context"):
            asyncio.run(_collect(tool, LandWorkArgs()))

    def test_merges_branch_into_main_no_ff(self, temp_repo, tool):
        repo = Repo(str(temp_repo))
        base_sha = repo.head.commit.hexsha
        # Create a divergent feature branch with a commit NOT on main.
        feat = repo.create_head("feature")
        feat.checkout()
        _commit(repo, temp_repo / "feat.txt", "new feature\n", "feat")
        candidate_sha = repo.head.commit.hexsha
        repo.heads.master.checkout()
        candidate_path = temp_repo.parent / "feature-worktree"
        repo.git.worktree("add", str(candidate_path), "feature")
        _set_active_handle(temp_repo, "feature", worktree_path=candidate_path)

        results = asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))

        assert len(results) == 1
        r = results[0]
        assert r.merged is True
        assert r.merge_commit_sha is not None
        # --no-ff creates a merge commit with two parents
        merged = Repo(str(temp_repo))
        commit = merged.commit(r.merge_commit_sha)
        assert len(commit.parents) == 2
        assert [parent.hexsha for parent in commit.parents] == [base_sha, candidate_sha]
        assert "feat.txt" in [b.name for b in merged.head.commit.tree.blobs]

    def test_already_merged_returns_not_merged(self, temp_repo, tool):
        repo = Repo(str(temp_repo))
        main_sha = repo.head.commit.hexsha
        # branch HEAD == main HEAD -> already an ancestor
        _set_active_handle(temp_repo, repo.active_branch.name)
        results = asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))
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
        with pytest.raises(ToolError, match="dirty"):
            asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))

    def test_no_active_worktree_raises(self, tool):
        worktree_manager._active = None
        with pytest.raises(ToolError, match="active worktree"):
            asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))

    def test_trivial_authorization_freezes_exact_candidate_sha(
        self, temp_repo, tool, monkeypatch
    ):
        import vibe.core.tools.builtins.land_work as landing

        repo = Repo(str(temp_repo))
        base_sha = repo.head.commit.hexsha
        feature = repo.create_head("feature")
        feature.checkout()
        (temp_repo / "docs").mkdir()
        (temp_repo / "docs" / "guide.md").write_text("guide\n")
        repo.index.add(["docs/guide.md"])
        repo.index.commit("docs")
        candidate_sha = repo.head.commit.hexsha
        repo.heads.master.checkout()
        _set_active_handle(temp_repo, "feature")
        ctx = InvokeContext(
            tool_call_id="land",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=True)
            ),
            verification_state=VerificationState(),
        )
        original = landing._require_verification_note

        def authorize_then_move(*args, **kwargs):
            result = original(*args, **kwargs)
            tree_sha = repo.commit(candidate_sha).tree.hexsha
            moved_sha = repo.git.commit_tree(
                tree_sha, "-p", candidate_sha, "-m", "late candidate movement"
            )
            repo.git.update_ref("refs/heads/feature", moved_sha, candidate_sha)
            return result

        monkeypatch.setattr(landing, "_require_verification_note", authorize_then_move)

        with pytest.raises(ToolError, match="candidate branch changed"):
            asyncio.run(
                _collect(
                    tool,
                    LandWorkArgs(verification_note="trivial: documentation only"),
                    ctx,
                )
            )

        assert repo.head.commit.hexsha == base_sha
        assert not (temp_repo / "docs" / "guide.md").exists()

    def test_compare_and_swap_rejects_base_move_before_ref_update(
        self, temp_repo, tool, monkeypatch
    ):
        import vibe.core.tools.builtins.land_work as landing

        repo = Repo(str(temp_repo))
        base_sha = repo.head.commit.hexsha
        feature = repo.create_head("feature")
        feature.checkout()
        _commit(repo, temp_repo / "feat.txt", "feature\n", "feature")
        repo.heads.master.checkout()
        candidate_path = temp_repo.parent / "candidate-cas"
        repo.git.worktree("add", str(candidate_path), "feature")
        _set_active_handle(temp_repo, "feature", worktree_path=candidate_path)
        original = landing._merge_no_ff
        moved_sha: str | None = None

        def move_base_then_merge(*args, **kwargs):
            nonlocal moved_sha
            tree_sha = repo.commit(base_sha).tree.hexsha
            moved_sha = repo.git.commit_tree(
                tree_sha, "-p", base_sha, "-m", "concurrent base movement"
            )
            repo.git.update_ref("refs/heads/master", moved_sha, base_sha)
            return original(*args, **kwargs)

        monkeypatch.setattr(landing, "_merge_no_ff", move_base_then_merge)

        with pytest.raises(ToolError, match="recent main work"):
            asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))

        assert moved_sha is not None
        assert repo.head.commit.hexsha == moved_sha
        assert len(repo.head.commit.parents) == 1
        assert not (temp_repo / "feat.txt").exists()

    def test_verifier_authority_rejects_dirty_candidate_after_gate(
        self, temp_repo, tool, monkeypatch
    ):
        import vibe.core.tools.builtins.land_work as landing

        repo = Repo(str(temp_repo))
        base_sha = repo.head.commit.hexsha
        candidate_path = temp_repo.parent / "candidate"
        repo.git.worktree("add", str(candidate_path), "-b", "feature", base_sha)
        candidate = Repo(str(candidate_path))
        (candidate_path / "feat.txt").write_text("feature\n")
        candidate.index.add(["feat.txt"])
        candidate.index.commit("feature")
        worktree_manager._active = WorktreeHandle(
            original_repo_root=temp_repo,
            worktree_path=candidate_path,
            branch="feature",
            create_head_sha=base_sha,
        )
        state = VerificationState()
        ctx = InvokeContext(
            tool_call_id="land",
            agent_manager=cast(
                AgentManager, _FakeAgentManager(verification_subsystem=True)
            ),
            verification_state=state,
        )

        def authorize_then_dirty(*args, **kwargs):
            (candidate_path / "late.txt").write_text("late\n")
            return None

        monkeypatch.setattr(landing, "_require_verification_note", authorize_then_dirty)

        with pytest.raises(ToolError, match="verified candidate became dirty"):
            asyncio.run(_collect(tool, LandWorkArgs(), ctx))

        assert repo.head.commit.hexsha == base_sha
        assert not (temp_repo / "feat.txt").exists()

    def test_landing_ignores_ambient_git_redirection(
        self, temp_repo, tool, monkeypatch, tmp_path
    ):
        repo = Repo(str(temp_repo))
        base_sha = repo.head.commit.hexsha
        candidate_path = temp_repo.parent / "candidate-ambient"
        repo.git.worktree("add", str(candidate_path), "-b", "feature", base_sha)
        candidate = Repo(str(candidate_path))
        (candidate_path / "feat.txt").write_text("feature\n")
        candidate.index.add(["feat.txt"])
        candidate.index.commit("feature")
        _set_active_handle(temp_repo, "feature", worktree_path=candidate_path)

        redirected_index = tmp_path / "redirected-index"
        redirected_worktree = tmp_path / "redirected-worktree"
        redirected_worktree.mkdir()
        monkeypatch.setenv("GIT_INDEX_FILE", str(redirected_index))
        monkeypatch.setenv("GIT_WORK_TREE", str(redirected_worktree))
        monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "hostile.so"))
        monkeypatch.setenv("PATH", str(tmp_path))

        results = asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))

        assert results[0].merged is True
        assert (temp_repo / "feat.txt").read_text() == "feature\n"
        assert not redirected_index.exists()
        assert not any(redirected_worktree.iterdir())

    def test_landing_rejects_executable_local_git_config(self, temp_repo, tool):
        repo = Repo(str(temp_repo))
        base_sha = repo.head.commit.hexsha
        candidate_path = temp_repo.parent / "candidate-unsafe-config"
        repo.git.worktree("add", str(candidate_path), "-b", "feature", base_sha)
        candidate = Repo(str(candidate_path))
        (candidate_path / "feat.txt").write_text("feature\n")
        candidate.index.add(["feat.txt"])
        candidate.index.commit("feature")
        _set_active_handle(temp_repo, "feature", worktree_path=candidate_path)
        repo.git.config("--local", "filter.evil.process", "false")

        with pytest.raises(ToolError, match="unsafe local executable Git"):
            asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))

        assert repo.head.commit.hexsha == base_sha
        assert not (temp_repo / "feat.txt").exists()

    def test_revalidates_managed_topology_at_both_authority_boundaries(
        self, temp_repo, tool, monkeypatch
    ):
        import vibe.core.tools.builtins.land_work as landing

        repo = Repo(str(temp_repo))
        base_sha = repo.head.commit.hexsha
        candidate_path = temp_repo.parent / "candidate-revalidate"
        repo.git.worktree("add", str(candidate_path), "-b", "feature", base_sha)
        candidate = Repo(str(candidate_path))
        (candidate_path / "feat.txt").write_text("feature\n")
        candidate.index.add(["feat.txt"])
        candidate.index.commit("feature")
        _set_active_handle(temp_repo, "feature", worktree_path=candidate_path)
        calls: list[tuple[str, str]] = []

        def observe(ctx, request, *, expected_base_sha, expected_candidate_sha):
            calls.append((expected_base_sha, expected_candidate_sha))

        monkeypatch.setattr(landing, "_revalidate_managed_topology", observe)
        results = asyncio.run(_collect(tool, LandWorkArgs(), _passing_context()))

        assert results[0].merged is True
        assert calls == [
            (base_sha, candidate.head.commit.hexsha),
            (base_sha, candidate.head.commit.hexsha),
        ]


async def _collect(
    tool: LandWork, args: LandWorkArgs, ctx: InvokeContext | None = None
):
    out = []
    async for r in tool.run(args, ctx):
        out.append(r)
    return out


class _FakeConfig:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.verification_subsystem = verification_subsystem


class _FakeAgentManager:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.config = _FakeConfig(verification_subsystem)


class _PassingVerificationState(VerificationState):
    def __init__(self) -> None:
        super().__init__()
        generation = self.begin_verifier_attempt()
        assert self.record_verifier_result(
            generation,
            VerifierAttemptDisposition.PASS,
            "Test receipt authority is current.",
        )
        self.receipt_reference = VerificationReceiptReference(
            receipt_id="a" * 64,
            repository_identity="repository",
            base_sha="b" * 40,
            candidate_head="c" * 40,
            task_brief_hash="d" * 64,
            contract_hash="e" * 64,
            configuration_hash="f" * 64,
            checks_hash="1" * 64,
            recipe_version="test-v1",
            verifier_attempt_generation=generation,
        )

    def has_valid_receipt(
        self,
        *,
        repository_path: Path,
        expected_base_sha: str,
        expected_candidate_head: str | None = None,
        receipt_id: str | None = None,
    ) -> bool:
        return receipt_id == "a" * 64


def _passing_context(*, verification_subsystem: bool = True) -> InvokeContext:
    return InvokeContext(
        tool_call_id="land",
        agent_manager=cast(
            AgentManager,
            _FakeAgentManager(verification_subsystem=verification_subsystem),
        ),
        verification_state=_PassingVerificationState(),
    )


def _empty_context(*, verification_subsystem: bool = True) -> InvokeContext:
    ctx = _passing_context(verification_subsystem=verification_subsystem)
    ctx.verification_state = VerificationState()
    return ctx


class TestLandWorkVerificationNote:
    def test_helper_skips_when_no_ctx(self):
        _require_verification_note(LandWorkArgs(), None)

    def test_helper_skips_when_subsystem_off(self):
        ctx = _empty_context(verification_subsystem=False)
        _require_verification_note(LandWorkArgs(), ctx)

    def test_requires_note_when_subsystem_on(self):
        ctx = _empty_context()
        with pytest.raises(ToolError, match="trusted verification receipt"):
            _require_verification_note(LandWorkArgs(), ctx)

    def test_accepts_trivial_note_with_reason(self):
        ctx = _empty_context()
        _require_verification_note(
            LandWorkArgs(verification_note="trivial: docs-only"),
            ctx,
            changed_paths=["docs/guide.md"],
        )

    def test_rejects_trivial_note_for_code_diff(self):
        ctx = _empty_context()
        with pytest.raises(ToolError, match="documentation-only"):
            _require_verification_note(
                LandWorkArgs(verification_note="trivial: small fix"),
                ctx,
                changed_paths=["vibe/core/agent_loop.py"],
            )

    def test_rejects_trivial_note_for_code_renamed_into_docs(self, temp_repo):
        repo = Repo(str(temp_repo))
        write_safe(temp_repo / "outside.py", "value = 1\n")
        repo.index.add(["outside.py"])
        repo.index.commit("add source")
        feature = repo.create_head("feature")
        feature.checkout()
        (temp_repo / "docs").mkdir()
        repo.git.mv("outside.py", "docs/inside.md")
        repo.index.commit("rename into docs")
        repo.heads.master.checkout()
        changed_paths = _changed_paths(repo, "master", "feature")
        ctx = _empty_context()

        assert changed_paths == ["docs/inside.md", "outside.py"]
        with pytest.raises(ToolError, match="documentation-only"):
            _require_verification_note(
                LandWorkArgs(verification_note="trivial: docs-only"),
                ctx,
                changed_paths=changed_paths,
            )

    def test_rejects_arbitrary_nonempty_note(self):
        ctx = _empty_context()
        with pytest.raises(ToolError, match="cannot authorize"):
            _require_verification_note(
                LandWorkArgs(verification_note="I ran the tests and they passed"), ctx
            )

    def test_rejects_trivial_note_without_reason(self):
        ctx = _empty_context()
        with pytest.raises(ToolError, match="cannot authorize"):
            _require_verification_note(LandWorkArgs(verification_note="trivial:"), ctx)
