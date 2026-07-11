from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from git import Repo
from git.exc import GitError

from vibe.core._workspace_verification import workspace_fingerprint


class VerifiedCandidateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifiedCandidate:
    parent_path: Path
    parent_head: str
    parent_workspace_fingerprint: str
    candidate_path: Path
    candidate_head: str
    candidate_workspace_fingerprint: str


def prepare_verified_candidate(wt: Any) -> VerifiedCandidate:
    try:
        parent_path = Path(wt.repo_root).resolve()
        candidate_path = Path(wt.path).resolve()
        parent = Repo(str(parent_path))
        candidate = Repo(str(candidate_path))
        parent_head = parent.head.commit.hexsha
        parent_dirty = parent.is_dirty(untracked_files=True)
        base_sha = wt.base_sha
    except (AttributeError, GitError, OSError, ValueError) as exc:
        raise VerifiedCandidateError(
            "candidate repository state was unavailable before verification"
        ) from exc

    parent_fingerprint = workspace_fingerprint(parent_path)
    if parent_fingerprint is None:
        raise VerifiedCandidateError(
            "parent workspace fingerprint was unavailable before verification"
        )
    if parent_dirty:
        raise VerifiedCandidateError("parent workspace was dirty before verification")
    if parent_head != base_sha:
        raise VerifiedCandidateError("parent HEAD changed before verification")

    try:
        if candidate.is_dirty(untracked_files=True):
            candidate.git.add("-A")
            candidate.index.commit("workflow agent work")
        candidate_head = candidate.head.commit.hexsha
        candidate.git.merge_base("--is-ancestor", parent_head, candidate_head)
        candidate_dirty = candidate.is_dirty(untracked_files=True)
    except (GitError, OSError, ValueError) as exc:
        raise VerifiedCandidateError(
            "isolated worker candidate could not be prepared for verification"
        ) from exc

    candidate_fingerprint = workspace_fingerprint(candidate_path)
    if candidate_fingerprint is None:
        raise VerifiedCandidateError(
            "candidate workspace fingerprint was unavailable before verification"
        )
    if candidate_dirty:
        raise VerifiedCandidateError(
            "candidate workspace remained dirty after preparation"
        )
    return VerifiedCandidate(
        parent_path=parent_path,
        parent_head=parent_head,
        parent_workspace_fingerprint=parent_fingerprint,
        candidate_path=candidate_path,
        candidate_head=candidate_head,
        candidate_workspace_fingerprint=candidate_fingerprint,
    )


def verified_candidate_diagnostic(
    candidate: VerifiedCandidate, *, delivered: bool = False
) -> str | None:
    try:
        candidate_repo = Repo(str(candidate.candidate_path))
        candidate_matches = (
            not candidate_repo.is_dirty(untracked_files=True)
            and candidate_repo.head.commit.hexsha == candidate.candidate_head
            and workspace_fingerprint(candidate.candidate_path)
            == candidate.candidate_workspace_fingerprint
        )
    except (GitError, OSError, ValueError):
        candidate_matches = False
    if not candidate_matches:
        return "candidate workspace changed during verification"

    try:
        parent = Repo(str(candidate.parent_path))
        parent_head = parent.head.commit.hexsha
        parent_fingerprint = workspace_fingerprint(candidate.parent_path)
        parent_clean = not parent.is_dirty(untracked_files=True)
    except (GitError, OSError, ValueError):
        return (
            "delivered workspace does not match the verified candidate"
            if delivered
            else "parent workspace changed during verification"
        )

    if delivered:
        if (
            parent_clean
            and parent_head == candidate.candidate_head
            and parent_fingerprint == candidate.candidate_workspace_fingerprint
        ):
            return None
        return "delivered workspace does not match the verified candidate"

    if (
        parent_clean
        and parent_head == candidate.parent_head
        and parent_fingerprint == candidate.parent_workspace_fingerprint
    ):
        return None
    return "parent workspace changed during verification"


__all__ = [
    "VerifiedCandidate",
    "VerifiedCandidateError",
    "prepare_verified_candidate",
    "verified_candidate_diagnostic",
]
