from __future__ import annotations

from pathlib import Path
import sys

from git import Repo
import pytest

from vibe.core._verification_runner import TrustedCheck
from vibe.core.tasking._candidate import validate_task_candidate
from vibe.core.tasking._policy import BoundTaskContract, TaskContractAuthority
from vibe.core.tasking.models import TaskManifestIdentity
from vibe.core.tools._task_manifest import resolve_task_manifest
from vibe.core.utils.io import write_safe


def _repository(tmp_path: Path) -> tuple[Repo, str]:
    repo = Repo.init(tmp_path)
    write_safe(tmp_path / "src" / "feature.py", "before\n")
    write_safe(tmp_path / "outside.py", "before\n")
    repo.git.add("-A")
    repo.index.commit("base")
    return repo, repo.head.commit.hexsha


def _contract(root: Path, check: TrustedCheck) -> BoundTaskContract:
    return BoundTaskContract(
        authority=TaskContractAuthority.LEAD,
        workspace_root=root.resolve(),
        objective="update feature",
        allowed_paths=("src/**",),
        denied_paths=("src/private/**",),
        acceptance_check_ids=(check.name,),
        trusted_checks=(check,),
        manifest=resolve_task_manifest(
            TaskManifestIdentity(name="implement-verify", version="1")
        ),
        brief_hash="brief",
        max_tokens=None,
        max_cost_usd=None,
        max_calls=None,
        deadline=None,
    )


def _check() -> TrustedCheck:
    return TrustedCheck(
        name="focused",
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        cwd=".",
        timeout_seconds=5,
    )


@pytest.mark.parametrize("mutation", ["edit", "delete", "rename", "untracked"])
def test_candidate_rejects_every_out_of_scope_change(
    tmp_path: Path, mutation: str
) -> None:
    repo, base_sha = _repository(tmp_path)
    outside = tmp_path / "outside.py"
    match mutation:
        case "edit":
            write_safe(outside, "after\n")
        case "delete":
            outside.unlink()
        case "rename":
            outside.rename(tmp_path / "renamed.py")
        case "untracked":
            write_safe(tmp_path / "new.py", "new\n")

    result = validate_task_candidate(_contract(tmp_path, _check()), tmp_path, base_sha)

    assert not result.passed
    assert "outside the task contract" in result.diagnostics[0]
    assert repo.head.commit.hexsha == base_sha


def test_candidate_runs_prebound_check_for_allowed_change(tmp_path: Path) -> None:
    _, base_sha = _repository(tmp_path)
    write_safe(tmp_path / "src" / "feature.py", "after\n")

    result = validate_task_candidate(_contract(tmp_path, _check()), tmp_path, base_sha)

    assert result.passed
    assert result.changed_paths == ("src/feature.py",)
    assert result.checks[0].name == "focused"


def test_candidate_returns_exact_failed_check_diagnostic(tmp_path: Path) -> None:
    _, base_sha = _repository(tmp_path)
    write_safe(tmp_path / "src" / "feature.py", "after\n")
    check = TrustedCheck(
        name="focused",
        argv=(sys.executable, "-c", "import sys; print('exact'); sys.exit(7)"),
        cwd=".",
        timeout_seconds=5,
    )

    result = validate_task_candidate(_contract(tmp_path, check), tmp_path, base_sha)

    assert not result.passed
    assert "check 'focused': exit 7" in result.diagnostics[0]
    assert "exact" in result.diagnostics[0]


def test_candidate_contains_trusted_check_workspace_mutation(tmp_path: Path) -> None:
    _, base_sha = _repository(tmp_path)
    write_safe(tmp_path / "src" / "feature.py", "after\n")
    check = TrustedCheck(
        name="focused",
        argv=(
            sys.executable,
            "-c",
            "from pathlib import Path; Path('generated.txt').write_text('x')",
        ),
        cwd=".",
        timeout_seconds=5,
    )

    result = validate_task_candidate(_contract(tmp_path, check), tmp_path, base_sha)

    assert not result.passed
    assert not (tmp_path / "generated.txt").exists()
    assert "check 'focused'" in result.diagnostics[0]
