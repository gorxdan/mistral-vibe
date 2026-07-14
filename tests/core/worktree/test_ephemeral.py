from __future__ import annotations

from pathlib import Path

from git import Repo
from git.exc import GitCommandError
from pydantic import ValidationError
import pytest

from vibe.core._workspace_verification import workspace_fingerprint
from vibe.core.candidate_delivery import (
    CandidateDelivery,
    CandidateDeliveryStatus,
    CandidateIntegrationMethod,
)
from vibe.core.worktree._trusted_git import TrustedGitWorktree
from vibe.core.worktree.ephemeral import (
    create_ephemeral_worktree,
    deliver_ephemeral_worktree,
    deliver_ephemeral_worktree_result,
    deliver_verified_ephemeral_worktree_result,
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


def test_default_path_namespaced_by_real_repo(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F2: default base_dir must derive the repo name via git-common-dir, not
    # root.name (which is the worktree leaf when called from inside a session
    # worktree). Override VIBE_HOME so the path is deterministic.
    fake_home = tmp_path / "vibehome"
    fake_home.mkdir()
    monkeypatch.setenv("VIBE_HOME", str(fake_home))
    wt = create_ephemeral_worktree(repo, "iso")
    try:
        assert "repo" in str(wt.path)
        assert str(wt.path).startswith(str(fake_home / "worktrees" / "repo" / "iso"))
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


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


def test_concurrent_nonconflicting_candidates_both_deliver(
    repo: Path, base_dir: Path
) -> None:
    first = create_ephemeral_worktree(repo, "first", base_dir=base_dir)
    second = create_ephemeral_worktree(repo, "second", base_dir=base_dir)
    try:
        first_repo = Repo(str(first.path))
        (first.path / "first.txt").write_text("first\n")
        first_repo.index.add(["first.txt"])
        first_repo.index.commit("first candidate")

        second_repo = Repo(str(second.path))
        (second.path / "second.txt").write_text("second\n")
        second_repo.index.add(["second.txt"])
        second_repo.index.commit("second candidate")

        assert deliver_ephemeral_worktree(first) is True
        delivery = deliver_ephemeral_worktree_result(second)
        assert delivery.status is CandidateDeliveryStatus.LANDED
        assert delivery.integration_method is CandidateIntegrationMethod.MERGE
        assert delivery.base_sha == second.base_sha
        assert delivery.candidate_sha == second_repo.head.commit.hexsha
        assert delivery.parent_sha_before is not None
        assert delivery.parent_sha_after == Repo(str(repo)).head.commit.hexsha
        assert delivery.branch == second.branch
        assert (repo / "first.txt").read_text() == "first\n"
        assert (repo / "second.txt").read_text() == "second\n"
    finally:
        remove_ephemeral_worktree(first, keep_if_changed=False)
        remove_ephemeral_worktree(second, keep_if_changed=False)


def test_conflicting_candidate_is_aborted_and_preserved(
    repo: Path, base_dir: Path
) -> None:
    first = create_ephemeral_worktree(repo, "first-conflict", base_dir=base_dir)
    second = create_ephemeral_worktree(repo, "second-conflict", base_dir=base_dir)
    try:
        first_repo = Repo(str(first.path))
        (first.path / "f.txt").write_text("first\n")
        first_repo.index.add(["f.txt"])
        first_repo.index.commit("first candidate")

        second_repo = Repo(str(second.path))
        (second.path / "f.txt").write_text("second\n")
        second_repo.index.add(["f.txt"])
        second_repo.index.commit("second candidate")
        second_sha = second_repo.head.commit.hexsha

        assert deliver_ephemeral_worktree(first) is True
        parent = Repo(str(repo))
        parent_sha = parent.head.commit.hexsha
        delivery = deliver_ephemeral_worktree_result(second)

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.candidate_sha == second_sha
        assert delivery.parent_sha_before == parent_sha
        assert delivery.parent_sha_after == parent_sha
        assert delivery.branch == second.branch
        assert delivery.diagnostic is not None
        assert parent.head.commit.hexsha == parent_sha
        assert not parent.is_dirty(untracked_files=True)
        assert (repo / "f.txt").read_text() == "first\n"
        assert second.branch in [head.name for head in parent.heads]
    finally:
        remove_ephemeral_worktree(first, keep_if_changed=False)
        remove_ephemeral_worktree(second, keep_if_changed=False)


def test_rewritten_candidate_history_is_not_merged(repo: Path, base_dir: Path) -> None:
    parent = Repo(str(repo))
    (repo / "parent.txt").write_text("required parent state\n")
    parent.index.add(["parent.txt"])
    previous_sha = parent.head.commit.hexsha
    parent.index.commit("second parent commit")

    candidate = create_ephemeral_worktree(repo, "rewritten", base_dir=base_dir)
    try:
        candidate_repo = Repo(str(candidate.path))
        candidate_repo.git.checkout("--detach", previous_sha)
        (candidate.path / "candidate.txt").write_text("candidate\n")
        candidate_repo.index.add(["candidate.txt"])
        candidate_repo.index.commit("rewritten candidate")
        parent.git.branch("-f", candidate.branch, candidate_repo.head.commit.hexsha)

        delivery = deliver_ephemeral_worktree_result(candidate)

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert (repo / "parent.txt").read_text() == "required parent state\n"
        assert not (repo / "candidate.txt").exists()
        assert candidate.branch in [head.name for head in parent.heads]
    finally:
        remove_ephemeral_worktree(candidate, keep_if_changed=False)


def test_detached_committed_candidate_is_preserved_on_recovery_ref(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "detached-committed", base_dir=base_dir)
    recovery_branch: str | None = None
    try:
        candidate = Repo(str(wt.path))
        candidate.git.checkout("--detach")
        (wt.path / "detached.txt").write_text("detached\n")
        candidate.index.add(["detached.txt"])
        candidate.index.commit("detached candidate")
        detached_sha = candidate.head.commit.hexsha

        delivery = deliver_ephemeral_worktree_result(wt)
        recovery_branch = delivery.branch

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.candidate_sha == detached_sha
        assert recovery_branch is not None
        assert Repo(str(repo)).commit(recovery_branch).hexsha == detached_sha
        assert not (repo / "detached.txt").exists()
        assert remove_ephemeral_worktree(wt) is False
        assert Repo(str(repo)).commit(recovery_branch).hexsha == detached_sha
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)
        if recovery_branch is not None and recovery_branch != wt.branch:
            try:
                Repo(str(repo)).git.branch("-D", recovery_branch)
            except GitCommandError:
                pass


def test_detached_uncommitted_candidate_is_committed_and_preserved(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "detached-dirty", base_dir=base_dir)
    recovery_branch: str | None = None
    try:
        candidate = Repo(str(wt.path))
        candidate.git.checkout("--detach")
        (wt.path / "dirty-detached.txt").write_text("recover me\n")

        delivery = deliver_ephemeral_worktree_result(wt)
        recovery_branch = delivery.branch

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.candidate_sha is not None
        assert recovery_branch is not None
        parent = Repo(str(repo))
        assert parent.commit(recovery_branch).hexsha == delivery.candidate_sha
        assert parent.git.show(f"{recovery_branch}:dirty-detached.txt") == "recover me"
        assert not (repo / "dirty-detached.txt").exists()
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)
        if recovery_branch is not None and recovery_branch != wt.branch:
            try:
                Repo(str(repo)).git.branch("-D", recovery_branch)
            except GitCommandError:
                pass


def test_switched_candidate_branch_is_preserved_without_delivery(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "switched", base_dir=base_dir)
    recovery_branch: str | None = None
    try:
        candidate = Repo(str(wt.path))
        candidate.git.checkout("-b", "unexpected-candidate-branch")
        (wt.path / "switched.txt").write_text("switched\n")
        candidate.index.add(["switched.txt"])
        candidate.index.commit("switched candidate")
        switched_sha = candidate.head.commit.hexsha

        delivery = deliver_ephemeral_worktree_result(wt)
        recovery_branch = delivery.branch

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.candidate_sha == switched_sha
        assert recovery_branch is not None
        assert Repo(str(repo)).commit(recovery_branch).hexsha == switched_sha
        assert not (repo / "switched.txt").exists()
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)
        parent = Repo(str(repo))
        for branch in {recovery_branch, "unexpected-candidate-branch"}:
            if branch is None or branch == wt.branch:
                continue
            try:
                parent.git.branch("-D", branch)
            except GitCommandError:
                pass


def test_rewound_parent_is_rejected_before_fast_forward(
    repo: Path, base_dir: Path
) -> None:
    parent = Repo(str(repo))
    previous_sha = parent.head.commit.hexsha
    (repo / "required.txt").write_text("required parent state\n")
    parent.index.add(["required.txt"])
    parent.index.commit("required parent commit")
    wt = create_ephemeral_worktree(repo, "rewound-parent", base_dir=base_dir)
    try:
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        parent.head.reference = parent.commit(previous_sha)
        parent.head.reset(index=True, working_tree=True)

        delivery = deliver_ephemeral_worktree_result(wt)

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert parent.head.commit.hexsha == previous_sha
        assert not (repo / "required.txt").exists()
        assert not (repo / "candidate.txt").exists()
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_dirty_parent_is_rejected_before_fast_forward(
    repo: Path, base_dir: Path, dirty_kind: str
) -> None:
    wt = create_ephemeral_worktree(
        repo, f"dirty-parent-{dirty_kind}", base_dir=base_dir
    )
    try:
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        parent = Repo(str(repo))
        parent_sha = parent.head.commit.hexsha
        dirty_path = repo / ("f.txt" if dirty_kind == "tracked" else "local.txt")
        dirty_path.write_text("local work\n")

        delivery = deliver_ephemeral_worktree_result(wt)

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert parent.head.commit.hexsha == parent_sha
        assert parent.is_dirty(untracked_files=True)
        assert dirty_path.read_text() == "local work\n"
        assert not (repo / "candidate.txt").exists()
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_replaced_parent_branch_is_rejected(repo: Path, base_dir: Path) -> None:
    wt = create_ephemeral_worktree(repo, "replaced-parent", base_dir=base_dir)
    try:
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        parent = Repo(str(repo))
        parent.git.checkout("-b", "replacement-parent")

        delivery = deliver_ephemeral_worktree_result(wt)

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert parent.active_branch.name == "replacement-parent"
        assert parent.head.commit.hexsha == wt.base_sha
        assert not (repo / "candidate.txt").exists()
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_landed_delivery_requires_authoritative_fields() -> None:
    with pytest.raises(ValidationError, match="landed delivery requires"):
        CandidateDelivery(status=CandidateDeliveryStatus.LANDED)


@pytest.mark.parametrize(
    "field", ["base_sha", "candidate_sha", "parent_sha_before", "parent_sha_after"]
)
def test_delivery_rejects_noncanonical_sha(field: str) -> None:
    with pytest.raises(ValidationError, match="full lowercase Git SHAs"):
        CandidateDelivery.model_validate({
            "status": CandidateDeliveryStatus.PRESERVED,
            field: "not-a-git-sha",
        })


def test_no_changes_delivery_rejects_parent_movement() -> None:
    with pytest.raises(ValidationError, match="must not change parent HEAD"):
        CandidateDelivery(
            status=CandidateDeliveryStatus.NO_CHANGES,
            base_sha="a" * 40,
            candidate_sha="a" * 40,
            parent_sha_before="b" * 40,
            parent_sha_after="c" * 40,
        )


def test_candidate_already_contained_by_parent_is_accepted(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "already-contained", base_dir=base_dir)
    try:
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        candidate_sha = candidate.head.commit.hexsha

        parent = Repo(str(repo))
        parent.git.merge("--ff-only", candidate_sha)
        (repo / "parent.txt").write_text("later parent work\n")
        parent.index.add(["parent.txt"])
        parent.index.commit("later parent work")
        parent_sha = parent.head.commit.hexsha

        delivery = deliver_ephemeral_worktree_result(wt)

        assert delivery.status is CandidateDeliveryStatus.LANDED
        assert (
            delivery.integration_method is CandidateIntegrationMethod.ALREADY_CONTAINED
        )
        assert delivery.candidate_sha == candidate_sha
        assert delivery.parent_sha_before == parent_sha
        assert delivery.parent_sha_after == parent_sha
        assert parent.head.commit.hexsha == parent_sha
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_cleanup_reports_branch_deletion_failure(repo: Path, base_dir: Path) -> None:
    wt = create_ephemeral_worktree(repo, "cleanup-ref", base_dir=base_dir)
    parent = Repo(str(repo))
    holder = base_dir / "branch-holder"
    parent.git.worktree("unlock", str(wt.path))
    parent.git.worktree("remove", str(wt.path))
    parent.git.worktree("add", str(holder), wt.branch)
    try:
        assert remove_ephemeral_worktree(wt, keep_if_changed=False) is False
        assert wt.branch in [head.name for head in parent.heads]
    finally:
        parent.git.worktree("remove", "--force", str(holder))
        parent.git.branch("-D", wt.branch)


def test_verified_candidate_delivery_uses_exact_fast_forward(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "verified", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "verified.txt").write_text("verified\n")
        candidate.index.add(["verified.txt"])
        candidate.index.commit("verified candidate")
        candidate_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None

        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=candidate_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )

        parent = Repo(str(repo))
        assert delivery.status is CandidateDeliveryStatus.LANDED
        assert delivery.integration_method is CandidateIntegrationMethod.FAST_FORWARD
        assert delivery.parent_sha_before == wt.base_sha
        assert delivery.parent_sha_after == candidate_sha
        assert parent.head.commit.hexsha == candidate_sha
        assert not parent.is_dirty(untracked_files=True)
        trusted_parent = TrustedGitWorktree.open(repo)
        assert trusted_parent.index_tree() == trusted_parent.tree_sha(candidate_sha)
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_verified_candidate_delivery_rejects_parent_advance(
    repo: Path, base_dir: Path
) -> None:
    first = create_ephemeral_worktree(repo, "first-verified", base_dir=base_dir)
    second = create_ephemeral_worktree(repo, "second-verified", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        second_repo = Repo(str(second.path))
        (second.path / "second.txt").write_text("second\n")
        second_repo.index.add(["second.txt"])
        second_repo.index.commit("second candidate")
        second_sha = second_repo.head.commit.hexsha
        second_fingerprint = workspace_fingerprint(second.path)
        assert parent_fingerprint is not None
        assert second_fingerprint is not None

        first_repo = Repo(str(first.path))
        (first.path / "first.txt").write_text("first\n")
        first_repo.index.add(["first.txt"])
        first_repo.index.commit("first candidate")
        assert deliver_ephemeral_worktree(first)
        advanced_sha = Repo(str(repo)).head.commit.hexsha

        delivery = deliver_verified_ephemeral_worktree_result(
            second,
            expected_parent_sha=second.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=second_sha,
            expected_candidate_fingerprint=second_fingerprint,
        )

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.candidate_sha == second_sha
        assert delivery.parent_sha_after == advanced_sha
        assert delivery.diagnostic == "parent HEAD changed after verification"
        assert Repo(str(repo)).head.commit.hexsha == advanced_sha
        assert not (repo / "second.txt").exists()
    finally:
        remove_ephemeral_worktree(first, keep_if_changed=False)
        remove_ephemeral_worktree(second, keep_if_changed=False)


def test_verified_candidate_delivery_rejects_post_verifier_dirty_state(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "dirty-verified", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        candidate_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None
        (wt.path / "late.txt").write_text("late mutation\n")

        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=candidate_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.diagnostic == "candidate workspace changed after verification"
        assert Repo(str(repo)).head.commit.hexsha == wt.base_sha
        assert candidate.head.commit.hexsha == candidate_sha
        assert candidate.is_dirty(untracked_files=True)
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_verified_candidate_delivery_rejects_candidate_head_change(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "moved-verified", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        verified_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None
        candidate.git.commit("--allow-empty", "-m", "late candidate movement")
        moved_sha = candidate.head.commit.hexsha

        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=verified_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.candidate_sha == verified_sha
        assert delivery.diagnostic == "candidate HEAD changed after verification"
        assert Repo(str(repo)).head.commit.hexsha == wt.base_sha
        assert candidate.head.commit.hexsha == moved_sha
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_verified_delivery_ignores_ambient_git_redirection(
    repo: Path, base_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = create_ephemeral_worktree(repo, "ambient-git", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        candidate_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None

        redirected_index = tmp_path / "redirected-index"
        redirected_worktree = tmp_path / "redirected-worktree"
        redirected_worktree.mkdir()
        monkeypatch.setenv("GIT_INDEX_FILE", str(redirected_index))
        monkeypatch.setenv("GIT_WORK_TREE", str(redirected_worktree))
        monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "hostile.so"))
        monkeypatch.setenv("PATH", str(tmp_path))

        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=candidate_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )
        monkeypatch.delenv("GIT_INDEX_FILE")
        monkeypatch.delenv("GIT_WORK_TREE")
        monkeypatch.delenv("LD_PRELOAD")
        monkeypatch.delenv("PATH")

        assert delivery.status is CandidateDeliveryStatus.LANDED
        assert Repo(str(repo)).head.commit.hexsha == candidate_sha
        assert not redirected_index.exists()
        assert not any(redirected_worktree.iterdir())
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_verified_delivery_rejects_executable_local_git_config(
    repo: Path, base_dir: Path
) -> None:
    wt = create_ephemeral_worktree(repo, "unsafe-config", base_dir=base_dir)
    parent = Repo(str(repo))
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        candidate_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None
        parent.git.config("--local", "filter.evil.process", "false")

        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=candidate_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )
        parent.git.config("--local", "--unset-all", "filter.evil.process")

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert Repo(str(repo)).head.commit.hexsha == wt.base_sha
        assert not (repo / "candidate.txt").exists()
    finally:
        try:
            parent.git.config("--local", "--unset-all", "filter.evil.process")
        except GitCommandError:
            pass
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_pre_cas_failure_restores_transaction_owned_tree(
    repo: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe.core.worktree._trusted_git as trusted_git

    wt = create_ephemeral_worktree(repo, "restore-pre-cas", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        candidate_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None

        def reject_cas(self, target_ref, new_sha, old_sha):
            raise trusted_git.TrustedGitError("injected CAS failure")

        monkeypatch.setattr(trusted_git.TrustedGitWorktree, "update_ref", reject_cas)
        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=candidate_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )

        parent = Repo(str(repo))
        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert parent.head.commit.hexsha == wt.base_sha
        assert not parent.is_dirty(untracked_files=True)
        assert not (repo / "candidate.txt").exists()
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_pre_cas_failure_preserves_concurrent_worktree_edit(
    repo: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe.core.worktree._trusted_git as trusted_git

    wt = create_ephemeral_worktree(repo, "preserve-concurrent", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "f.txt").write_text("candidate\n")
        candidate.index.add(["f.txt"])
        candidate.index.commit("candidate")
        candidate_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None

        def edit_then_reject(self, target_ref, new_sha, old_sha):
            (self.work_tree / "f.txt").write_text("concurrent edit\n")
            raise trusted_git.TrustedGitError("injected CAS failure")

        monkeypatch.setattr(
            trusted_git.TrustedGitWorktree, "update_ref", edit_then_reject
        )
        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=candidate_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )

        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.diagnostic is not None
        assert "automatic restoration was skipped after state diverged" in (
            delivery.diagnostic
        )
        assert Repo(str(repo)).head.commit.hexsha == wt.base_sha
        assert (repo / "f.txt").read_text() == "concurrent edit\n"
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)


def test_post_cas_validation_failure_is_rolled_back_and_not_landed(
    repo: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe.core.worktree._ref_transaction as ref_transaction

    wt = create_ephemeral_worktree(repo, "post-cas-validation", base_dir=base_dir)
    try:
        parent_fingerprint = workspace_fingerprint(repo)
        candidate = Repo(str(wt.path))
        (wt.path / "candidate.txt").write_text("candidate\n")
        candidate.index.add(["candidate.txt"])
        candidate.index.commit("candidate")
        candidate_sha = candidate.head.commit.hexsha
        candidate_fingerprint = workspace_fingerprint(wt.path)
        assert parent_fingerprint is not None
        assert candidate_fingerprint is not None
        original = ref_transaction._require_materialized_state
        calls = 0

        def fail_second_validation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ref_transaction.CheckedOutRefUpdateError(
                    "injected post-CAS validation failure"
                )
            return original(*args, **kwargs)

        monkeypatch.setattr(
            ref_transaction, "_require_materialized_state", fail_second_validation
        )
        delivery = deliver_verified_ephemeral_worktree_result(
            wt,
            expected_parent_sha=wt.base_sha,
            expected_parent_fingerprint=parent_fingerprint,
            expected_candidate_sha=candidate_sha,
            expected_candidate_fingerprint=candidate_fingerprint,
        )

        parent = Repo(str(repo))
        assert delivery.status is CandidateDeliveryStatus.PRESERVED
        assert delivery.diagnostic is not None
        assert "rolled back" in delivery.diagnostic
        assert parent.head.commit.hexsha == wt.base_sha
        assert not parent.is_dirty(untracked_files=True)
        assert not (repo / "candidate.txt").exists()
    finally:
        remove_ephemeral_worktree(wt, keep_if_changed=False)
