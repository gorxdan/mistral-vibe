from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from tests.mock.utils import collect_result
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
from vibe.core.tasking import TaskBrief
from vibe.core.tasking._policy import BoundTaskContract, TaskContractAuthority
from vibe.core.teams._task_checks import TaskCheckEvidence
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.task_checks import (
    TaskChecks,
    TaskChecksArgs,
    TaskChecksConfig,
)
from vibe.core.verification_state import VerificationState

_FOCUSED_ARGV = (str(_HOST_PYTHON), "-c", "raise SystemExit(0)")


def _contract(root: Path) -> BoundTaskContract:
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="checks-v1",
        task_brief="Implement the scoped change",
        acceptance_contract="The focused check passes",
        allowed_paths=("src/**",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=_FOCUSED_ARGV,
                executable_sha256=_HOST_PYTHON_SHA256,
                environment_attestation_path=str(_HOST_ENVIRONMENT),
                environment_attestation_sha256=_HOST_ENVIRONMENT_SHA256,
            ),
        ),
    )
    brief = TaskBrief.model_validate({
        "objective": "Implement the scoped change",
        "allowed_paths": ["src/**"],
        "acceptance_checks": ["focused"],
        "manifest": {"name": "verify", "version": "1"},
    })
    return BoundTaskContract.bind(
        brief,
        authority=TaskContractAuthority.LEAD,
        workspace_root=root,
        verification_state=VerificationState.from_recipe(recipe),
    )


@pytest.mark.asyncio
async def test_task_checks_runs_only_the_bound_checks(tmp_path: Path) -> None:
    evidence = TaskCheckEvidence(
        name="focused",
        argv=_FOCUSED_ARGV,
        cwd=str(tmp_path),
        exit_code=0,
        timed_out=False,
        duration_ms=12,
        stdout="passed",
        stderr="",
    )
    tool = TaskChecks(config_getter=lambda: TaskChecksConfig(), state=BaseToolState())
    contract = _contract(tmp_path)
    ctx = InvokeContext(tool_call_id="checks", task_contract=contract)

    with patch(
        "vibe.core.tools.builtins.task_checks.run_guarded_task_checks",
        return_value=((evidence,), None),
    ) as run_checks:
        result = await collect_result(tool.run(TaskChecksArgs(), ctx))

    assert result.passed is True
    assert result.checks == (evidence,)
    assert result.diagnostics == ("check 'focused': exit 0\nstdout:\npassed",)
    run_checks.assert_called_once_with(contract.trusted_checks, tmp_path.resolve())


@pytest.mark.asyncio
async def test_task_checks_rejects_unbound_and_model_selected_commands() -> None:
    with pytest.raises(ValidationError):
        TaskChecksArgs.model_validate({"command": "pytest"})

    tool = TaskChecks(config_getter=lambda: TaskChecksConfig(), state=BaseToolState())
    with pytest.raises(ToolError, match="host-bound"):
        await collect_result(
            tool.run(TaskChecksArgs(), InvokeContext(tool_call_id="checks"))
        )
