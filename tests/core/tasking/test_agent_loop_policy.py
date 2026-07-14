from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.trusted_verification import (
    HOST_ENVIRONMENT as _HOST_ENVIRONMENT,
    HOST_ENVIRONMENT_SHA256 as _HOST_ENVIRONMENT_SHA256,
    HOST_PYTHON as _HOST_PYTHON,
    HOST_PYTHON_SHA256 as _HOST_PYTHON_SHA256,
)
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import (
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.tasking import TaskBrief, TaskManifestIdentity
from vibe.core.tasking._policy import BoundTaskContract, TaskContractAuthority
from vibe.core.tools.permissions import RequiredPermission
from vibe.core.types import ApprovalResponse, FunctionCall, ToolCall, ToolResultEvent
from vibe.core.verification_state import VerificationState


def _contract(workspace_root: Path) -> BoundTaskContract:
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="loop-policy-v1",
        task_brief="Write only under allowed",
        acceptance_contract="The focused check must pass",
        allowed_paths=("**",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=(str(_HOST_PYTHON), "-c", "raise SystemExit(0)"),
                executable_sha256=_HOST_PYTHON_SHA256,
                environment_attestation_path=str(_HOST_ENVIRONMENT),
                environment_attestation_sha256=_HOST_ENVIRONMENT_SHA256,
            ),
        ),
    )
    brief = TaskBrief(
        objective="Write only under allowed",
        allowed_paths=["allowed/**"],
        acceptance_checks=["focused"],
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )
    return BoundTaskContract.bind(
        brief,
        authority=TaskContractAuthority.LEAD,
        workspace_root=workspace_root,
        verification_state=VerificationState.from_recipe(recipe),
    )


@pytest.mark.asyncio
async def test_agent_loop_rejects_out_of_scope_write_before_invocation() -> None:
    tool_call = ToolCall(
        id="contract-write",
        index=0,
        function=FunctionCall(
            name="write_file",
            arguments='{"path":"outside.txt","content":"not written"}',
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Writing.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Stopped.")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["write_file"]),
        backend=backend,
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        task_contract=_contract(Path.cwd()),
    )

    events = [event async for event in loop.act("write outside the contract")]

    failures = [event for event in events if isinstance(event, ToolResultEvent)]
    assert failures
    assert "outside the task contract allowlist" in (failures[0].error or "")
    assert not Path("outside.txt").exists()


@pytest.mark.asyncio
async def test_agent_loop_rechecks_user_modified_arguments() -> None:
    tool_call = ToolCall(
        id="contract-modified-write",
        index=0,
        function=FunctionCall(
            name="write_file",
            arguments='{"path":"allowed/file.txt","content":"not written"}',
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Writing.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Stopped.")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["write_file"], tools={"write_file": {"permission": "ask"}}
        ),
        backend=backend,
        agent_name=BuiltinAgentName.DEFAULT,
        task_contract=_contract(Path.cwd()),
    )

    async def modify(
        _tool_name: str,
        _args: BaseModel,
        _tool_call_id: str,
        _required_permissions: list[RequiredPermission] | None = None,
        _judge_note: str | None = None,
    ) -> tuple[ApprovalResponse, str | None, dict[str, Any] | None]:
        return (
            ApprovalResponse.MODIFY,
            None,
            {"path": "outside.txt", "content": "not written"},
        )

    loop.set_approval_callback(modify)
    events = [event async for event in loop.act("write the file")]

    failures = [event for event in events if isinstance(event, ToolResultEvent)]
    assert failures
    assert "outside the task contract allowlist" in (failures[0].error or "")
    assert not Path("outside.txt").exists()
    assert not Path("allowed/file.txt").exists()
