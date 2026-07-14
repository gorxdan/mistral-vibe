"""Ephemeral git worktrees for per-agent isolation in workflows.

Unlike :class:`WorktreeManager` (a single process-wide worktree driven by
``os.chdir``), these helpers create and remove throwaway worktrees WITHOUT
chdir and without a global singleton, so many concurrent workflow agents can
each run in their own isolated checkout (used by ``agent(isolation="worktree")``).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time

from git import Repo
from git.exc import GitCommandError, GitError

from vibe.core.candidate_delivery import (
    CandidateDelivery,
    CandidateDeliveryStatus,
    CandidateIntegrationMethod,
)
from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME
from vibe.core.worktree._ref_transaction import (
    CheckedOutRefUpdateError,
    update_checked_out_branch,
)
from vibe.core.worktree._trusted_git import TrustedGitError, TrustedGitWorktree
from vibe.core.worktree.manager import merge_lock

_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _iso_lock_reason() -> str:
    import json

    boot_id = ""
    try:
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        )
    except OSError:
        pass
    start_time = 0
    try:
        stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").split()
        start_time = int(stat[21])
    except (OSError, IndexError, ValueError):
        pass
    return json.dumps(
        {
            "pid": os.getpid(),
            "boot_id": boot_id,
            "proc_start_time": start_time,
            "purpose": "iso",
            "created_at": time.time(),
        },
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class EphemeralWorktree:
    path: Path
    branch: str
    repo_root: Path
    base_sha: str
    parent_branch: str | None = None


def create_ephemeral_worktree(
    repo_root: Path, label: str, *, base_dir: Path | None = None
) -> EphemeralWorktree:
    """Create a throwaway worktree off HEAD. No chdir, no global state — safe to
    call concurrently (each call yields a distinct worktree + branch).
    """
    repo = Repo(str(repo_root), search_parent_directories=True)
    if repo.working_tree_dir is None:
        raise RuntimeError(f"{repo_root} is a bare repo; cannot create a worktree")
    root = Path(repo.working_tree_dir).resolve()
    base_sha = repo.head.commit.hexsha
    try:
        parent_branch = repo.active_branch.name
    except (GitError, TypeError):
        parent_branch = None

    # F2: derive the real repo name via the git common-dir (the pattern in
    # manager.py:113-124), not root.name — inside a session worktree root.name
    # is the worktree leaf (e.g. "cli-1830853-..."), which produces un-namespaced
    # paths and permanent husk parents.
    try:
        common = repo.git.rev_parse("--git-common-dir")
        common_dir = (root / common).resolve()
        repo_name = common_dir.parent.name
    except Exception:
        repo_name = root.name

    safe = _SAFE_LABEL_RE.sub("-", label).strip("-")[:32] or "agent"
    leaf = f"{safe}-{os.getpid()}-{time.time_ns()}"
    branch = f"vibe/iso/{leaf}"
    resolved_base = base_dir or VIBE_HOME.path / "worktrees" / repo_name / "iso"
    resolved_base.mkdir(parents=True, exist_ok=True)
    path = resolved_base / leaf

    repo.git.worktree("add", str(path), "-b", branch, "HEAD")
    try:
        repo.git.worktree("lock", str(path), "--reason", _iso_lock_reason())
    except GitCommandError as exc:
        logger.warning("git worktree lock failed for %s: %s", path, exc)
    logger.info("Created isolated worktree %s on branch %s", path, branch)
    return EphemeralWorktree(
        path=path,
        branch=branch,
        repo_root=root,
        base_sha=base_sha,
        parent_branch=parent_branch,
    )


def deliver_ephemeral_worktree(wt: EphemeralWorktree) -> bool:
    return _deliver_ephemeral_worktree_result(wt).accepted


def _active_branch_name(repo: Repo) -> str | None:
    try:
        return repo.active_branch.name
    except (GitError, TypeError):
        return None


def _is_ancestor(repo: Repo, ancestor: str, descendant: str) -> bool:
    try:
        repo.git.merge_base("--is-ancestor", ancestor, descendant)
    except (GitError, OSError, ValueError):
        return False
    return True


def _same_common_repository(parent: Repo, candidate: Repo) -> bool:
    try:
        return Path(parent.common_dir).resolve() == Path(candidate.common_dir).resolve()
    except (OSError, TypeError, ValueError):
        return False


def _candidate_state_diagnostic(
    wt: EphemeralWorktree, parent: Repo, candidate: Repo
) -> str | None:
    if not _same_common_repository(parent, candidate):
        return "candidate worktree no longer belongs to the parent repository"
    if _active_branch_name(candidate) != wt.branch:
        return "candidate checked-out branch changed before delivery"
    head_sha = _revision_sha(candidate, "HEAD")
    branch_sha = _revision_sha(parent, wt.branch)
    if head_sha is None or head_sha != branch_sha:
        return "candidate HEAD no longer matches its delivery branch"
    if not _is_ancestor(parent, wt.base_sha, head_sha):
        return "candidate no longer descends from its isolated base"
    return None


def _parent_state_diagnostic(
    wt: EphemeralWorktree, parent: Repo, *, expected_head: str | None = None
) -> str | None:
    current_head = _revision_sha(parent, "HEAD")
    if current_head is None:
        return "parent HEAD was unavailable before delivery"
    if expected_head is not None and current_head != expected_head:
        return "parent HEAD changed during delivery"
    expected_branch = getattr(wt, "parent_branch", None)
    if expected_branch is not None and _active_branch_name(parent) != expected_branch:
        return "parent checked-out branch changed before delivery"
    if parent.is_dirty(untracked_files=True):
        return "parent workspace is dirty; candidate was not integrated"
    if not _is_ancestor(parent, wt.base_sha, current_head):
        return "parent no longer descends from the isolated base"
    return None


def _recovery_branch_name(wt: EphemeralWorktree, candidate_sha: str) -> str:
    return f"{wt.branch}-recovery-{candidate_sha[:12]}"


class _CandidatePreservationError(RuntimeError):
    pass


def _preserve_candidate_head(
    wt: EphemeralWorktree, parent: Repo, candidate: Repo
) -> tuple[str | None, str]:
    candidate_sha = _revision_sha(candidate, "HEAD")
    if candidate_sha is None:
        return None, wt.branch
    if (
        _same_common_repository(parent, candidate)
        and _revision_sha(parent, wt.branch) == candidate_sha
    ):
        return candidate_sha, wt.branch
    recovery_branch = _recovery_branch_name(wt, candidate_sha)
    try:
        candidate.git.update_ref(f"refs/heads/{recovery_branch}", candidate_sha)
        if _revision_sha(candidate, recovery_branch) == candidate_sha:
            return candidate_sha, recovery_branch
    except (GitError, OSError, ValueError):
        pass
    active_branch = _active_branch_name(candidate)
    if (
        active_branch is not None
        and _revision_sha(candidate, active_branch) == candidate_sha
    ):
        return candidate_sha, active_branch
    raise _CandidatePreservationError(
        "candidate HEAD could not be bound to a recovery branch"
    )


def _candidate_delivery(
    wt: EphemeralWorktree,
    *,
    status: CandidateDeliveryStatus,
    parent_sha_before: str | None,
    parent_sha_after: str | None,
    candidate_sha: str | None,
    branch: str | None = None,
    integration_method: CandidateIntegrationMethod | None = None,
    diagnostic: str | None = None,
) -> CandidateDelivery:
    return CandidateDelivery(
        status=status,
        base_sha=wt.base_sha,
        candidate_sha=candidate_sha,
        parent_sha_before=parent_sha_before,
        parent_sha_after=parent_sha_after,
        branch=branch or wt.branch,
        worktree_path=str(wt.path),
        integration_method=integration_method,
        diagnostic=diagnostic,
    )


def _preserved_candidate_delivery(
    wt: EphemeralWorktree,
    parent: Repo,
    candidate: Repo,
    *,
    parent_sha_before: str | None,
    diagnostic: str,
) -> CandidateDelivery:
    try:
        candidate_sha, recovery_branch = _preserve_candidate_head(wt, parent, candidate)
    except _CandidatePreservationError as exc:
        candidate_sha = _revision_sha(candidate, "HEAD")
        recovery_branch = wt.branch
        diagnostic = f"{diagnostic}; {exc}; worktree must be retained"
    logger.info(
        "Preserving isolated candidate %s on %s: %s",
        wt.path,
        recovery_branch,
        diagnostic,
    )
    return _candidate_delivery(
        wt,
        status=CandidateDeliveryStatus.PRESERVED,
        parent_sha_before=parent_sha_before,
        parent_sha_after=_revision_sha(parent, "HEAD"),
        candidate_sha=candidate_sha,
        branch=recovery_branch,
        diagnostic=diagnostic,
    )


class _GenericDeliveryRefusal(RuntimeError):
    pass


def _integrate_generic_candidate(
    wt: EphemeralWorktree,
    parent: Repo,
    candidate_sha: str,
    *,
    parent_sha_before: str | None,
) -> CandidateIntegrationMethod:
    if parent_sha_before is not None and _is_ancestor(
        parent, candidate_sha, parent_sha_before
    ):
        return CandidateIntegrationMethod.ALREADY_CONTAINED
    try:
        parent.git.merge("--ff-only", candidate_sha)
        return CandidateIntegrationMethod.FAST_FORWARD
    except GitCommandError as ff_error:
        parent_diagnostic = _parent_state_diagnostic(
            wt, parent, expected_head=parent_sha_before
        )
        if parent_diagnostic is not None:
            raise _GenericDeliveryRefusal(parent_diagnostic) from ff_error
        try:
            parent.git.merge(
                "--no-edit",
                "-m",
                f"Integrate isolated candidate {wt.branch}",
                candidate_sha,
            )
            return CandidateIntegrationMethod.MERGE
        except GitCommandError as merge_error:
            try:
                parent.git.merge("--abort")
            except GitCommandError:
                pass
            reason = str(merge_error).strip().splitlines()[:1]
            raise _GenericDeliveryRefusal(
                f"automatic integration failed: {reason}"
            ) from merge_error


def _deliver_ephemeral_worktree_result(wt: EphemeralWorktree) -> CandidateDelivery:
    parent: Repo | None = None
    candidate: Repo | None = None
    parent_sha_before: str | None = None
    try:
        parent = Repo(str(wt.repo_root))
        candidate = Repo(str(wt.path))
        with merge_lock(wt.repo_root):
            parent_sha_before = _revision_sha(parent, "HEAD")
            parent_diagnostic = _parent_state_diagnostic(wt, parent)
            if parent_diagnostic is not None:
                raise _GenericDeliveryRefusal(parent_diagnostic)

            candidate_diagnostic = _candidate_state_diagnostic(wt, parent, candidate)
            if candidate.is_dirty(untracked_files=True):
                candidate.git.add("-A")
                candidate.index.commit("workflow agent work")
            post_commit_diagnostic = _candidate_state_diagnostic(wt, parent, candidate)
            candidate_diagnostic = candidate_diagnostic or post_commit_diagnostic
            if candidate_diagnostic is not None:
                raise _GenericDeliveryRefusal(candidate_diagnostic)

            candidate_sha = _revision_sha(candidate, "HEAD")
            if candidate_sha is None:
                raise _GenericDeliveryRefusal(
                    "candidate HEAD was unavailable before delivery"
                )
            if candidate_sha == wt.base_sha:
                return _candidate_delivery(
                    wt,
                    status=CandidateDeliveryStatus.NO_CHANGES,
                    parent_sha_before=parent_sha_before,
                    parent_sha_after=parent_sha_before,
                    candidate_sha=candidate_sha,
                )

            parent_diagnostic = _parent_state_diagnostic(
                wt, parent, expected_head=parent_sha_before
            )
            if parent_diagnostic is not None:
                raise _GenericDeliveryRefusal(parent_diagnostic)
            integration_method = _integrate_generic_candidate(
                wt, parent, candidate_sha, parent_sha_before=parent_sha_before
            )
            parent_sha_after = _revision_sha(parent, "HEAD")
            if parent_sha_after is None:
                raise _GenericDeliveryRefusal(
                    "parent HEAD was unavailable after delivery"
                )
            logger.info("Delivered isolated worktree %s into %s", wt.path, wt.repo_root)
            return _candidate_delivery(
                wt,
                status=CandidateDeliveryStatus.LANDED,
                parent_sha_before=parent_sha_before,
                parent_sha_after=parent_sha_after,
                candidate_sha=candidate_sha,
                integration_method=integration_method,
            )
    except _GenericDeliveryRefusal as exc:
        assert parent is not None and candidate is not None
        return _preserved_candidate_delivery(
            wt,
            parent,
            candidate,
            parent_sha_before=parent_sha_before,
            diagnostic=str(exc),
        )
    except (GitError, OSError, ValueError) as exc:
        logger.warning("Failed to deliver isolated worktree %s: %s", wt.path, exc)
        if parent is not None and candidate is not None:
            return _preserved_candidate_delivery(
                wt,
                parent,
                candidate,
                parent_sha_before=parent_sha_before,
                diagnostic="candidate repository state prevented automatic integration",
            )
        return _candidate_delivery(
            wt,
            status=CandidateDeliveryStatus.PRESERVED,
            parent_sha_before=parent_sha_before,
            parent_sha_after=(
                _revision_sha(parent, "HEAD") if parent is not None else None
            ),
            candidate_sha=None,
            diagnostic="candidate repository state prevented automatic integration",
        )


def _revision_sha(repo: Repo, revision: str) -> str | None:
    try:
        return repo.commit(revision).hexsha
    except (GitError, OSError, ValueError):
        return None


def describe_ephemeral_worktree(
    wt: EphemeralWorktree,
    *,
    status: CandidateDeliveryStatus,
    parent_sha_before: str | None = None,
    diagnostic: str | None = None,
) -> CandidateDelivery:
    repo_root = getattr(wt, "repo_root", None)
    try:
        parent = Repo(str(repo_root)) if repo_root is not None else None
    except (GitError, OSError, ValueError):
        parent = None
    candidate_sha = _revision_sha(parent, wt.branch) if parent is not None else None
    parent_sha_after = _revision_sha(parent, "HEAD") if parent is not None else None
    base_sha = getattr(wt, "base_sha", None)
    resolved_status = status
    try:
        worktree_dirty = Repo(str(wt.path)).is_dirty(untracked_files=True)
    except (GitError, OSError, ValueError, AttributeError):
        worktree_dirty = False
    if candidate_sha is not None and candidate_sha == base_sha and not worktree_dirty:
        resolved_status = CandidateDeliveryStatus.NO_CHANGES
        parent_sha_before = parent_sha_after
    return CandidateDelivery(
        status=resolved_status,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
        parent_sha_before=parent_sha_before,
        parent_sha_after=parent_sha_after,
        branch=getattr(wt, "branch", None),
        worktree_path=str(getattr(wt, "path", "")) or None,
        diagnostic=diagnostic,
    )


def deliver_ephemeral_worktree_result(wt: EphemeralWorktree) -> CandidateDelivery:
    return _deliver_ephemeral_worktree_result(wt)


def deliver_verified_ephemeral_worktree_result(
    wt: EphemeralWorktree,
    *,
    expected_parent_sha: str,
    expected_parent_fingerprint: str,
    expected_candidate_sha: str,
    expected_candidate_fingerprint: str,
) -> CandidateDelivery:
    parent: TrustedGitWorktree | None = None
    try:
        parent = TrustedGitWorktree.open(wt.repo_root)
        candidate = TrustedGitWorktree.open(wt.path)
        with merge_lock(wt.repo_root):
            _require_verified_delivery_state(
                wt,
                parent,
                candidate,
                expected_parent_sha=expected_parent_sha,
                expected_parent_fingerprint=expected_parent_fingerprint,
                expected_candidate_sha=expected_candidate_sha,
                expected_candidate_fingerprint=expected_candidate_fingerprint,
            )
            if expected_candidate_sha == expected_parent_sha:
                return _verified_delivery_result(
                    wt,
                    status=CandidateDeliveryStatus.NO_CHANGES,
                    expected_parent_sha=expected_parent_sha,
                    expected_candidate_sha=expected_candidate_sha,
                    parent_sha_after=expected_parent_sha,
                )
            try:
                descends_from_base = parent.is_ancestor(
                    expected_parent_sha, expected_candidate_sha
                )
            except TrustedGitError as exc:
                raise _VerifiedDeliveryRefusal(
                    "verified candidate does not descend from the exact parent base"
                ) from exc
            if not descends_from_base:
                raise _VerifiedDeliveryRefusal(
                    "verified candidate does not descend from the exact parent base"
                )
            _require_verified_delivery_state(
                wt,
                parent,
                candidate,
                expected_parent_sha=expected_parent_sha,
                expected_parent_fingerprint=expected_parent_fingerprint,
                expected_candidate_sha=expected_candidate_sha,
                expected_candidate_fingerprint=expected_candidate_fingerprint,
            )
            update_checked_out_branch(
                parent,
                expected_head=expected_parent_sha,
                new_commit=expected_candidate_sha,
            )
    except (_VerifiedDeliveryRefusal, CheckedOutRefUpdateError) as exc:
        return _verified_delivery_result(
            wt,
            status=CandidateDeliveryStatus.PRESERVED,
            expected_parent_sha=expected_parent_sha,
            expected_candidate_sha=expected_candidate_sha,
            parent_sha_after=(
                _trusted_head_sha(parent) if parent is not None else None
            ),
            diagnostic=str(exc),
        )
    except (TrustedGitError, OSError, ValueError) as exc:
        logger.warning("Failed to deliver verified candidate %s: %s", wt.path, exc)
        return _verified_delivery_result(
            wt,
            status=CandidateDeliveryStatus.PRESERVED,
            expected_parent_sha=expected_parent_sha,
            expected_candidate_sha=expected_candidate_sha,
            parent_sha_after=None,
            diagnostic="verified candidate repository state was unavailable",
        )

    return _verified_delivery_result(
        wt,
        status=CandidateDeliveryStatus.LANDED,
        expected_parent_sha=expected_parent_sha,
        expected_candidate_sha=expected_candidate_sha,
        parent_sha_after=expected_candidate_sha,
        integration_method=CandidateIntegrationMethod.FAST_FORWARD,
    )


class _VerifiedDeliveryRefusal(RuntimeError):
    pass


def _require_verified_delivery_state(
    wt: EphemeralWorktree,
    parent: TrustedGitWorktree,
    candidate: TrustedGitWorktree,
    *,
    expected_parent_sha: str,
    expected_parent_fingerprint: str,
    expected_candidate_sha: str,
    expected_candidate_fingerprint: str,
) -> None:
    if wt.base_sha != expected_parent_sha:
        raise _VerifiedDeliveryRefusal(
            "verified parent base does not match the isolated worktree base"
        )
    if parent.common_dir != candidate.common_dir:
        raise _VerifiedDeliveryRefusal(
            "verified candidate no longer belongs to the parent repository"
        )
    expected_parent_branch = getattr(wt, "parent_branch", None)
    if (
        expected_parent_branch is not None
        and parent.head_ref() != f"refs/heads/{expected_parent_branch}"
    ):
        raise _VerifiedDeliveryRefusal(
            "parent checked-out branch changed after verification"
        )
    if parent.head_sha() != expected_parent_sha:
        raise _VerifiedDeliveryRefusal("parent HEAD changed after verification")
    if not parent.clean_against(expected_parent_sha, include_untracked=True):
        raise _VerifiedDeliveryRefusal("parent workspace changed after verification")
    if parent.fingerprint() != expected_parent_fingerprint:
        raise _VerifiedDeliveryRefusal("parent workspace changed after verification")
    if candidate.head_sha() != expected_candidate_sha:
        raise _VerifiedDeliveryRefusal("candidate HEAD changed after verification")
    if candidate.head_ref() != f"refs/heads/{wt.branch}":
        raise _VerifiedDeliveryRefusal("candidate branch changed after verification")
    if parent.branch_sha(wt.branch) != expected_candidate_sha:
        raise _VerifiedDeliveryRefusal("candidate branch changed after verification")
    if not candidate.clean_against(expected_candidate_sha, include_untracked=True):
        raise _VerifiedDeliveryRefusal("candidate workspace changed after verification")
    if candidate.fingerprint() != expected_candidate_fingerprint:
        raise _VerifiedDeliveryRefusal("candidate workspace changed after verification")


def _trusted_head_sha(repository: TrustedGitWorktree) -> str | None:
    try:
        return repository.head_sha()
    except TrustedGitError:
        return None


def _verified_delivery_result(
    wt: EphemeralWorktree,
    *,
    status: CandidateDeliveryStatus,
    expected_parent_sha: str,
    expected_candidate_sha: str,
    parent_sha_after: str | None,
    integration_method: CandidateIntegrationMethod | None = None,
    diagnostic: str | None = None,
) -> CandidateDelivery:
    return CandidateDelivery(
        status=status,
        base_sha=expected_parent_sha,
        candidate_sha=expected_candidate_sha,
        parent_sha_before=expected_parent_sha,
        parent_sha_after=parent_sha_after,
        branch=wt.branch,
        worktree_path=str(wt.path),
        integration_method=integration_method,
        diagnostic=diagnostic,
    )


def _try_unlock(repo: Repo, wt_path: Path) -> None:
    try:
        repo.git.worktree("unlock", str(wt_path))
    except GitCommandError as exc:
        logger.debug("worktree unlock (%s) no-op or failed: %s", wt_path, exc)


def _reclaim_changed_worktree(
    repo: Repo, wt: EphemeralWorktree, wt_repo: Repo | None, *, uncommitted: bool
) -> bool:
    if uncommitted and wt_repo is not None:
        try:
            wt_repo.git.add("-A")
            wt_repo.index.commit("workflow agent work (kept for recovery)")
        except (GitCommandError, OSError) as exc:
            logger.warning(
                "Could not commit isolated worktree %s for recovery (%s); "
                "keeping the directory so work is not lost.",
                wt.path,
                exc,
            )
            return False
    recovery_branch = wt.branch
    if wt_repo is None:
        return False
    try:
        _, recovery_branch = _preserve_candidate_head(wt, repo, wt_repo)
    except _CandidatePreservationError as exc:
        logger.warning(
            "Could not bind isolated worktree %s to a recovery ref (%s); "
            "keeping the directory so work is not lost.",
            wt.path,
            exc,
        )
        return False
    try:
        _try_unlock(repo, wt.path)
        repo.git.worktree("remove", "--force", str(wt.path))
        logger.info(
            "Reclaimed isolated worktree dir %s; work preserved on branch "
            "%s (recover: git merge %s)",
            wt.path,
            recovery_branch,
            recovery_branch,
        )
    except GitCommandError as exc:
        logger.warning(
            "Could not remove isolated worktree dir %s (%s); branch %s "
            "still holds the work.",
            wt.path,
            exc,
            recovery_branch,
        )
    return False


def remove_ephemeral_worktree(
    wt: EphemeralWorktree, *, keep_if_changed: bool = True
) -> bool:
    """Remove the worktree directory and (unless kept for recovery) its branch.

    If ``keep_if_changed`` and the worktree has uncommitted changes or its branch
    advanced past the base commit, the agent's work is preserved on the *branch*
    (committing any uncommitted changes first) and only the on-disk directory is
    reclaimed — recover via ``git merge <branch>``. Keeping the branch ref rather
    than the whole checked-out directory avoids the unbounded accumulation of
    kept worktree dirs under the worktrees base dir.

    Returns True when the worktree was fully reclaimed (directory removed AND
    branch deleted — nothing left to recover); False when recovery state remains
    or any cleanup step fails.
    """
    try:
        repo = Repo(str(wt.repo_root))
        wt_repo: Repo | None = None
        try:
            wt_repo = Repo(str(wt.path))
            uncommitted = wt_repo.is_dirty(untracked_files=True)
            changed = uncommitted or wt_repo.head.commit.hexsha != wt.base_sha
        except (GitCommandError, OSError):
            uncommitted = False
            changed = False  # worktree already gone / unreadable

        if keep_if_changed and changed:
            return _reclaim_changed_worktree(repo, wt, wt_repo, uncommitted=uncommitted)

        # Nothing to recover (clean tree, caller opted out, or the directory was
        # already reclaimed on an earlier call): remove the directory if present
        # and delete the throwaway branch.
        if wt.path.exists():
            _try_unlock(repo, wt.path)
            if keep_if_changed:
                repo.git.worktree("remove", str(wt.path))
            else:
                repo.git.worktree("remove", "--force", str(wt.path))
        else:
            repo.git.worktree("prune")
        try:
            repo.git.branch("-D", wt.branch)
        except GitCommandError as exc:
            logger.warning(
                "Reclaimed isolated worktree %s but could not delete branch %s: %s",
                wt.path,
                wt.branch,
                exc,
            )
            return False
        return True
    except (GitCommandError, OSError) as e:
        logger.warning(
            "Failed to remove isolated worktree %s: %s. Run `vibe worktree list` "
            "to review branches.",
            wt.path,
            e,
        )
        return False
