from __future__ import annotations

from vibe.core.worktree._trusted_git import TrustedGitError, TrustedGitWorktree


class CheckedOutRefUpdateError(RuntimeError):
    pass


def update_checked_out_branch(
    repository: TrustedGitWorktree, *, expected_head: str, new_commit: str
) -> str:
    try:
        old_sha = repository.resolve_commit(expected_head, "expected branch HEAD")
        new_sha = repository.resolve_commit(new_commit, "candidate commit")
        target_ref = repository.head_ref()
        _require_initial_state(repository, target_ref=target_ref, old_sha=old_sha)
        if old_sha == new_sha:
            return new_sha

        try:
            repository.read_tree(new_sha)
            _require_materialized_state(
                repository, target_ref=target_ref, head_sha=old_sha, tree_sha=new_sha
            )
        except (CheckedOutRefUpdateError, TrustedGitError) as exc:
            restored = _restore_before_cas(
                repository,
                target_ref=target_ref,
                old_sha=old_sha,
                transaction_sha=new_sha,
            )
            detail = (
                ""
                if restored
                else "; automatic restoration was skipped after state diverged"
            )
            raise CheckedOutRefUpdateError(
                f"could not materialize the exact candidate tree{detail}"
            ) from exc

        try:
            repository.update_ref(target_ref, new_sha, old_sha)
        except TrustedGitError as exc:
            restored = _restore_before_cas(
                repository,
                target_ref=target_ref,
                old_sha=old_sha,
                transaction_sha=new_sha,
            )
            detail = (
                ""
                if restored
                else "; automatic restoration was skipped after state diverged"
            )
            raise CheckedOutRefUpdateError(
                "checked-out branch changed before the compare-and-swap completed"
                f"{detail}"
            ) from exc

        try:
            _require_materialized_state(
                repository, target_ref=target_ref, head_sha=new_sha, tree_sha=new_sha
            )
        except (CheckedOutRefUpdateError, TrustedGitError) as exc:
            rolled_back = _rollback_after_cas(
                repository, target_ref=target_ref, old_sha=old_sha, new_sha=new_sha
            )
            detail = " and was rolled back" if rolled_back else ""
            raise CheckedOutRefUpdateError(
                "compare-and-swap completed, but the checked-out tree failed exact "
                f"post-landing validation{detail}"
            ) from exc
        return new_sha
    except TrustedGitError as exc:
        raise CheckedOutRefUpdateError(str(exc)) from exc


def _require_initial_state(
    repository: TrustedGitWorktree, *, target_ref: str, old_sha: str
) -> None:
    if repository.head_ref() != target_ref or repository.head_sha() != old_sha:
        raise CheckedOutRefUpdateError(
            "checked-out branch moved before the ref transaction"
        )
    if repository.index_tree() != repository.tree_sha(old_sha):
        raise CheckedOutRefUpdateError(
            "checked-out index changed before the ref transaction"
        )
    if not repository.clean_against(old_sha, include_untracked=True):
        raise CheckedOutRefUpdateError(
            "checked-out branch became dirty before the ref transaction"
        )


def _require_materialized_state(
    repository: TrustedGitWorktree, *, target_ref: str, head_sha: str, tree_sha: str
) -> None:
    if repository.head_ref() != target_ref or repository.head_sha() != head_sha:
        raise CheckedOutRefUpdateError(
            "checked-out branch changed while materializing the candidate tree"
        )
    if repository.index_tree() != repository.tree_sha(tree_sha):
        raise CheckedOutRefUpdateError(
            "checked-out index does not match the exact candidate tree"
        )
    if not repository.clean_against(tree_sha, include_untracked=True):
        raise CheckedOutRefUpdateError(
            "checked-out worktree does not match the exact candidate tree"
        )


def _restore_before_cas(
    repository: TrustedGitWorktree,
    *,
    target_ref: str,
    old_sha: str,
    transaction_sha: str,
) -> bool:
    try:
        if repository.head_ref() != target_ref or repository.head_sha() != old_sha:
            return False
        owned_state = repository.clean_against(
            old_sha, include_untracked=True
        ) or repository.clean_against(transaction_sha, include_untracked=True)
        if not owned_state:
            return False
        repository.read_tree(old_sha)
        return repository.index_tree() == repository.tree_sha(
            old_sha
        ) and repository.clean_against(old_sha, include_untracked=True)
    except TrustedGitError:
        return False


def _rollback_after_cas(
    repository: TrustedGitWorktree, *, target_ref: str, old_sha: str, new_sha: str
) -> bool:
    try:
        if repository.head_ref() != target_ref or repository.head_sha() != new_sha:
            return False
        if repository.index_tree() != repository.tree_sha(new_sha):
            return False
        if not repository.clean_against(new_sha, include_untracked=True):
            return False
        repository.update_ref(target_ref, old_sha, new_sha)
        repository.read_tree(old_sha)
        return (
            repository.head_sha() == old_sha
            and repository.index_tree() == repository.tree_sha(old_sha)
            and repository.clean_against(old_sha, include_untracked=True)
        )
    except TrustedGitError:
        return False


__all__ = ["CheckedOutRefUpdateError", "update_checked_out_branch"]
