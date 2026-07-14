from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from tests.trusted_verification import (
    HOST_ENVIRONMENT as _HOST_ENVIRONMENT,
    HOST_ENVIRONMENT_SHA256 as _HOST_ENVIRONMENT_SHA256,
    HOST_PYTHON as _HOST_PYTHON,
    HOST_PYTHON_SHA256 as _HOST_PYTHON_SHA256,
)
from vibe.core.config import (
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.tasking import TaskBrief, TaskBudget, TaskManifestIdentity
from vibe.core.tasking._policy import (
    BoundTaskContract,
    TaskContractAuthority,
    TaskContractError,
    TaskContractViolation,
)
from vibe.core.tools._task_manifest import TaskManifestError, resolve_task_manifest
from vibe.core.verification_state import VerificationState

_FOCUSED_ARGV = (str(_HOST_PYTHON), "-m", "pytest", "tests/test_focused.py")
_TYPES_ARGV = (str(_HOST_PYTHON), "-c", "raise SystemExit(0)")


def _set_attribute(instance: object, name: str, value: object) -> None:
    setattr(instance, name, value)


def _recipe(
    *, allowed_paths: tuple[str, ...] = ("src/**", "tests/**")
) -> TrustedVerificationRecipeConfig:
    return TrustedVerificationRecipeConfig(
        recipe_version="contract-v1",
        task_brief="Implement the scoped change",
        acceptance_contract="Focused checks must pass",
        allowed_paths=allowed_paths,
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=_FOCUSED_ARGV,
                executable_sha256=_HOST_PYTHON_SHA256,
                environment_attestation_path=str(_HOST_ENVIRONMENT),
                environment_attestation_sha256=_HOST_ENVIRONMENT_SHA256,
            ),
            TrustedVerificationCheckConfig(
                name="types",
                argv=_TYPES_ARGV,
                executable_sha256=_HOST_PYTHON_SHA256,
                environment_attestation_path=str(_HOST_ENVIRONMENT),
                environment_attestation_sha256=_HOST_ENVIRONMENT_SHA256,
            ),
        ),
    )


def _brief(**overrides: object) -> TaskBrief:
    values: dict[str, object] = {
        "objective": "Implement the scoped change",
        "allowed_paths": ["src/**", "tests/test_focused.py"],
        "denied_paths": ["src/private/**"],
        "acceptance_checks": ["focused"],
        "budget": {"max_tokens": 4_000, "max_cost_usd": 0.2, "max_calls": 5},
        "deadline": "2030-01-02T03:04:05Z",
        "manifest": {"name": "implement-verify", "version": "1"},
    }
    values.update(overrides)
    return TaskBrief.model_validate(values)


def _bind(brief: TaskBrief, root: Path) -> BoundTaskContract:
    return BoundTaskContract.bind(
        brief,
        authority=TaskContractAuthority.LEAD,
        workspace_root=root,
        verification_state=VerificationState.from_recipe(_recipe()),
    )


def test_manifest_registry_is_canonical_and_bounded() -> None:
    for name in ("investigate", "implement-verify", "verify", "mechanical-edit"):
        manifest = resolve_task_manifest(TaskManifestIdentity(name=name, version="1"))
        assert 5 <= len(manifest.tools) <= 8
        assert len(manifest.digest) == 64
        assert resolve_task_manifest(manifest.identity) == manifest


def test_manifest_registry_rejects_unknown_identity_and_digest() -> None:
    with pytest.raises(TaskManifestError, match="untrusted"):
        resolve_task_manifest(TaskManifestIdentity(name="full-access", version="99"))
    with pytest.raises(TaskManifestError, match="digest mismatch"):
        resolve_task_manifest(
            TaskManifestIdentity(
                name="implement-verify", version="1", digest="untrusted"
            )
        )


def test_contract_binds_only_recipe_owned_check_ids(tmp_path: Path) -> None:
    contract = _bind(_brief(), tmp_path)

    assert contract.acceptance_check_ids == ("focused",)
    assert contract.trusted_checks[0].argv == _FOCUSED_ARGV

    with pytest.raises(TaskContractError, match="untrusted acceptance check IDs"):
        _bind(
            _brief(acceptance_checks=["uv run pytest tests/test_focused.py"]), tmp_path
        )


def test_contract_requires_trusted_recipe_and_narrower_paths(tmp_path: Path) -> None:
    with pytest.raises(TaskContractError, match="trusted verification recipe"):
        BoundTaskContract.bind(
            _brief(),
            authority=TaskContractAuthority.USER,
            workspace_root=tmp_path,
            verification_state=VerificationState(),
        )

    with pytest.raises(TaskContractError, match="exceed"):
        _bind(_brief(allowed_paths=["outside/**"]), tmp_path)

    with pytest.raises(TaskContractError, match="exceed"):
        BoundTaskContract.bind(
            _brief(allowed_paths=["src/private/secret.py"], denied_paths=[]),
            authority=TaskContractAuthority.LEAD,
            workspace_root=tmp_path,
            verification_state=VerificationState.from_recipe(
                _recipe(allowed_paths=("src/*.py",))
            ),
        )


def test_contract_snapshots_mutable_brief_components(tmp_path: Path) -> None:
    brief = _brief()
    contract = _bind(brief, tmp_path)

    brief.allowed_paths.append("outside/**")

    assert contract.allowed_paths == ("src/**", "tests/test_focused.py")
    with pytest.raises(FrozenInstanceError):
        _set_attribute(contract, "objective", "widened")
    assert brief.budget is not None
    with pytest.raises(ValidationError):
        _set_attribute(brief.budget, "max_calls", 100)


def test_contract_enforces_manifest_and_denied_path_precedence(tmp_path: Path) -> None:
    contract = _bind(_brief(), tmp_path)

    contract.enforce_tool_call("write_file", SimpleNamespace(path="src/feature.py"))
    with pytest.raises(TaskContractViolation, match="denied"):
        contract.enforce_tool_call(
            "write_file", SimpleNamespace(path="src/private/secret.py")
        )
    with pytest.raises(TaskContractViolation, match="allowlist"):
        contract.enforce_tool_call(
            "edit", SimpleNamespace(file_path="docs/architecture.md")
        )
    with pytest.raises(TaskContractViolation, match="outside manifest"):
        contract.enforce_tool_call("bash", SimpleNamespace(command="true"))


def test_contract_enforces_read_and_search_roots(tmp_path: Path) -> None:
    contract = _bind(_brief(), tmp_path)

    contract.enforce_tool_call("read", SimpleNamespace(file_path="src/feature.py"))
    contract.enforce_tool_call("grep", SimpleNamespace(path="src"))
    with pytest.raises(TaskContractViolation, match="allowlist"):
        contract.enforce_tool_call(
            "read", SimpleNamespace(file_path="docs/architecture.md")
        )
    with pytest.raises(TaskContractViolation, match="denied"):
        contract.enforce_tool_call("glob", SimpleNamespace(path="src/private"))
    with pytest.raises(TaskContractViolation, match="valid path"):
        contract.enforce_tool_call("lsp", SimpleNamespace(file_path=None))
    with pytest.raises(TaskContractViolation, match="absolute glob"):
        contract.enforce_tool_call(
            "glob", SimpleNamespace(path="src", pattern="/etc/**")
        )
    with pytest.raises(TaskContractViolation, match="workspace-wide LSP"):
        contract.enforce_tool_call(
            "lsp",
            SimpleNamespace(operation="workspace_symbol", file_path="src/feature.py"),
        )


def test_single_segment_glob_does_not_cross_directories(tmp_path: Path) -> None:
    contract = BoundTaskContract.bind(
        _brief(allowed_paths=["src/*.py"], denied_paths=[]),
        authority=TaskContractAuthority.LEAD,
        workspace_root=tmp_path,
        verification_state=VerificationState.from_recipe(
            _recipe(allowed_paths=("src/**",))
        ),
    )

    contract.enforce_tool_call("write_file", SimpleNamespace(path="src/feature.py"))
    with pytest.raises(TaskContractViolation, match="allowlist"):
        contract.enforce_tool_call(
            "write_file", SimpleNamespace(path="src/private/secret.py")
        )
    with pytest.raises(TaskContractViolation, match="src/private/secret.py"):
        contract.validate_changed_paths(("src/private/secret.py",))


def test_segment_glob_handles_repeated_double_stars_without_backtracking(
    tmp_path: Path,
) -> None:
    repeated = "/".join("**" for _ in range(1_500))
    contract = BoundTaskContract.bind(
        _brief(allowed_paths=[f"src/{repeated}/*.py"], denied_paths=[]),
        authority=TaskContractAuthority.LEAD,
        workspace_root=tmp_path,
        verification_state=VerificationState.from_recipe(
            _recipe(allowed_paths=("src/**",))
        ),
    )
    nested = "src/" + "/".join(f"part-{index}" for index in range(32))

    contract.enforce_tool_call(
        "write_file", SimpleNamespace(path=f"{nested}/feature.py")
    )


@pytest.mark.parametrize(
    "path",
    [
        ".vibe/config.toml",
        ".VIBE/config.toml",
        ".agents/skills/override/SKILL.md",
        ".git/hooks/pre-commit",
        ".Git/config",
        "src/AGENTS.md",
        "src/agents.MD",
    ],
)
def test_contract_hard_denies_harness_control_plane(tmp_path: Path, path: str) -> None:
    contract = BoundTaskContract.bind(
        _brief(allowed_paths=["**"]),
        authority=TaskContractAuthority.LEAD,
        workspace_root=tmp_path,
        verification_state=VerificationState.from_recipe(
            _recipe(allowed_paths=("**",))
        ),
    )

    with pytest.raises(TaskContractViolation, match="control-plane"):
        contract.enforce_tool_call("write_file", SimpleNamespace(path=path))
    with pytest.raises(TaskContractViolation, match="control-plane"):
        contract.validate_changed_paths((path,))


def test_contract_hard_denies_control_plane_glob_pattern(tmp_path: Path) -> None:
    contract = BoundTaskContract.bind(
        _brief(allowed_paths=["**"]),
        authority=TaskContractAuthority.LEAD,
        workspace_root=tmp_path,
        verification_state=VerificationState.from_recipe(
            _recipe(allowed_paths=("**",))
        ),
    )

    with pytest.raises(TaskContractViolation, match="control-plane"):
        contract.enforce_tool_call(
            "glob", SimpleNamespace(path=".", pattern=".VIBE/**")
        )


def test_denied_paths_are_casefolded_fail_closed(tmp_path: Path) -> None:
    contract = _bind(_brief(), tmp_path)

    with pytest.raises(TaskContractViolation, match="denied"):
        contract.enforce_tool_call(
            "write_file", SimpleNamespace(path="src/PRIVATE/secret.py")
        )
    with pytest.raises(TaskContractViolation, match="src/PRIVATE/secret.py"):
        contract.validate_changed_paths(("src/PRIVATE/secret.py",))


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("glob", SimpleNamespace(path=".vibe/team")),
        ("grep", SimpleNamespace(path=".vibe/team")),
        ("lsp", SimpleNamespace(file_path=".vibe/team/tasks.json")),
    ],
)
def test_contract_hard_denies_team_metadata_for_search_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    arguments: SimpleNamespace,
) -> None:
    team_dir = tmp_path / ".vibe" / "team"
    team_dir.mkdir(parents=True)
    monkeypatch.setenv("VIBE_TEAM_DIR", str(team_dir))
    contract = BoundTaskContract.bind(
        _brief(allowed_paths=["**"]),
        authority=TaskContractAuthority.LEAD,
        workspace_root=tmp_path,
        verification_state=VerificationState.from_recipe(
            _recipe(allowed_paths=("**",))
        ),
    )

    with pytest.raises(TaskContractViolation, match="host-owned"):
        contract.enforce_tool_call(tool_name, arguments)


def test_contract_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "src").symlink_to(outside, target_is_directory=True)
    contract = _bind(_brief(), workspace)

    with pytest.raises(TaskContractViolation, match="escapes"):
        contract.enforce_tool_call("write_file", SimpleNamespace(path="src/escaped.py"))


def test_contract_maps_task_budget_to_child_spend_limits(tmp_path: Path) -> None:
    contract = _bind(_brief(), tmp_path)

    assert contract.deadline == datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert contract.deadline is not None
    assert contract.spend_limits().model_dump(exclude_none=True) == {
        "max_total_tokens": 4_000,
        "max_cost_usd": 0.2,
        "max_calls": 5,
        "deadline_at": contract.deadline.timestamp(),
    }


def test_contract_validates_candidate_changed_paths(tmp_path: Path) -> None:
    contract = _bind(_brief(), tmp_path)

    contract.validate_changed_paths(("src/feature.py", "tests/test_focused.py"))
    with pytest.raises(TaskContractViolation, match="src/private/secret.py"):
        contract.validate_changed_paths(("src/private/secret.py",))
    with pytest.raises(TaskContractViolation, match="docs/architecture.md"):
        contract.validate_changed_paths(("docs/architecture.md",))


def test_task_contract_models_prevent_limit_reassignment() -> None:
    budget = TaskBudget(max_calls=2)
    manifest = TaskManifestIdentity(name="verify", version="1")

    with pytest.raises(ValidationError):
        _set_attribute(budget, "max_calls", 3)
    with pytest.raises(ValidationError):
        _set_attribute(manifest, "name", "full-access")
