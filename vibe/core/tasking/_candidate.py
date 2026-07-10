from __future__ import annotations

from pathlib import Path

from git import Repo
from git.exc import GitError
from pydantic import BaseModel, ConfigDict

from vibe.core.tasking._policy import BoundTaskContract, TaskContractViolation
from vibe.core.teams._task_checks import (
    TaskCheckEvidence,
    run_guarded_task_checks,
    task_check_diagnostics,
)

__all__ = [
    "TaskCandidateValidation",
    "inspect_candidate_changed_paths",
    "validate_task_candidate",
    "validate_task_workspace_scope",
]


class TaskCandidateValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    changed_paths: tuple[str, ...]
    scope_passed: bool
    checks: tuple[TaskCheckEvidence, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.scope_passed
            and not self.diagnostics
            and all(check.passed for check in self.checks)
        )


def inspect_candidate_changed_paths(
    workspace_root: Path, base_sha: str
) -> tuple[str, ...]:
    repo = Repo(str(workspace_root))
    outputs = (
        repo.git.diff("--no-renames", "--name-only", base_sha, "HEAD", "--"),
        repo.git.diff("--no-renames", "--name-only", "--cached", "HEAD", "--"),
        repo.git.diff("--no-renames", "--name-only", "HEAD", "--"),
    )
    paths = {line for output in outputs for line in output.splitlines() if line}
    paths.update(repo.untracked_files)
    return tuple(sorted(paths))


def validate_task_candidate(
    contract: BoundTaskContract, workspace_root: Path, base_sha: str
) -> TaskCandidateValidation:
    root = workspace_root.resolve()
    initial = validate_task_workspace_scope(contract, root, base_sha)
    if not initial.scope_passed:
        return initial

    checks, mutation = run_guarded_task_checks(contract.trusted_checks, root)
    if mutation is not None:
        return TaskCandidateValidation(
            changed_paths=initial.changed_paths,
            scope_passed=False,
            checks=checks,
            diagnostics=(mutation,),
        )
    failed = tuple(check for check in checks if not check.passed)
    final = validate_task_workspace_scope(contract, root, base_sha, post_check=True)
    if not final.scope_passed:
        return final.model_copy(update={"checks": checks})
    return TaskCandidateValidation(
        changed_paths=final.changed_paths,
        scope_passed=True,
        checks=checks,
        diagnostics=task_check_diagnostics(failed),
    )


def validate_task_workspace_scope(
    contract: BoundTaskContract,
    workspace_root: Path,
    base_sha: str,
    *,
    post_check: bool = False,
) -> TaskCandidateValidation:
    root = workspace_root.resolve()
    prefix = "post-check " if post_check else ""
    try:
        changed_paths = inspect_candidate_changed_paths(root, base_sha)
    except (GitError, OSError, ValueError) as e:
        return TaskCandidateValidation(
            changed_paths=(),
            scope_passed=False,
            diagnostics=(f"{prefix}candidate inspection failed: {e}",),
        )
    try:
        contract.validate_changed_paths(changed_paths)
    except (ValueError, TaskContractViolation) as e:
        return TaskCandidateValidation(
            changed_paths=changed_paths,
            scope_passed=False,
            diagnostics=(f"{prefix}candidate validation failed: {e}",),
        )
    return TaskCandidateValidation(changed_paths=changed_paths, scope_passed=True)
