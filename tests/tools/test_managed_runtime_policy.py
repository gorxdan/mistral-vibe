from __future__ import annotations

from typing import Literal

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import collect_result
from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import (
    MCPStdio,
    TrustedExecutionTopologyConfig,
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.hooks.models import HookConfig, HookConfigResult, HookType
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.task import Task, TaskArgs, TaskToolConfig
from vibe.core.verification_state import VerificationState

_ACTIVE_ROOT_TOOLS = {
    "bash",
    "edit",
    "glob",
    "grep",
    "read",
    "skill",
    "task",
    "todo",
    "write_file",
}
_VERIFICATION_ROOT_TOOLS = {"glob", "grep", "read", "skill", "task", "verify_work"}
_SUBAGENT_TOOLS = {"bash", "glob", "grep", "read", "skill"}
_FORBIDDEN_MANAGED_TOOLS = {
    "land_work",
    "launch_workflow",
    "managed_mcp_fake_tool",
    "team",
    "team_message",
    "team_spawn",
    "tool_search",
    "web_fetch",
    "web_search",
    "unknown_future_mutator",
}


def _recipe(
    state: Literal["active", "verification"] = "active",
) -> TrustedVerificationRecipeConfig:
    sha = "a" * 40
    topology = TrustedExecutionTopologyConfig(
        packet_id="I00-P01",
        packet_path="docs/design/fork-maintenance/packets/I00-P01.md",
        state=state,
        control_worktree="/control",
        control_sha=sha,
        candidate_worktree="/candidate",
        candidate_branch="candidate",
        baseline_sha=sha,
        candidate_sha="b" * 40 if state == "verification" else None,
        upstream_sha=sha,
        evidence_workspace="/durable/evidence",
        run_id="i00-p01-test",
        runner_id="local-test",
        evidence_manifest_sha256="c" * 64 if state == "verification" else None,
    )
    return TrustedVerificationRecipeConfig(
        recipe_version="managed-v1",
        task_brief="Implement the managed packet",
        acceptance_contract="The focused check must pass",
        allowed_paths=("vibe/core/target.py",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=("/usr/bin/true",),
                executable_sha256="0" * 64,
                environment_attestation_path="/usr/bin/true",
                environment_attestation_sha256="1" * 64,
            ),
        ),
        execution_topology=topology,
    )


def _managed_config(state: Literal["active", "verification"] = "active"):
    return build_test_vibe_config(
        trusted_verification_recipe=_recipe(state),
        installed_components=["lsp"],
        disabled_tools=["read", "task", "verify_work"],
        mcp_servers=[
            MCPStdio(name="managed_mcp", transport="stdio", command="fake-cmd")
        ],
    )


@pytest.mark.asyncio
async def test_managed_active_allowlist_is_locked_on_initialization_and_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.agent_loop_verification.AgentLoopVerificationMixin."
        "_validate_trusted_execution_topology",
        lambda *_args, **_kwargs: None,
    )
    hook_result = HookConfigResult(
        hooks=[
            HookConfig(
                name="project-startup",
                type=HookType.SESSION_START,
                command="untrusted-project-command",
            )
        ],
        issues=[],
    )
    loop = build_test_agent_loop(
        config=_managed_config(), hook_config_result=hook_result
    )

    try:
        assert set(loop.tool_manager.available_tools) == _ACTIVE_ROOT_TOOLS
        assert not (_FORBIDDEN_MANAGED_TOOLS & set(loop.tool_manager.registered_tools))
        assert loop.hooks_count == 0
        assert loop.hooks_manager is None

        await loop.reload_with_initial_messages()

        assert set(loop.tool_manager.available_tools) == _ACTIVE_ROOT_TOOLS
        assert not (_FORBIDDEN_MANAGED_TOOLS & set(loop.tool_manager.registered_tools))

        await loop.reload_with_initial_messages(max_turns=10_000)

        assert loop._max_turns == 80
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_managed_verification_and_subagent_allowlists_are_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.agent_loop_verification.AgentLoopVerificationMixin."
        "_validate_trusted_execution_topology",
        lambda *_args, **_kwargs: None,
    )
    root = build_test_agent_loop(config=_managed_config("verification"))
    subagent = build_test_agent_loop(
        config=_managed_config("verification"),
        agent_name=BuiltinAgentName.VERIFIER,
        is_subagent=True,
    )

    try:
        assert set(root.tool_manager.available_tools) == _VERIFICATION_ROOT_TOOLS
        assert set(subagent.tool_manager.available_tools) == _SUBAGENT_TOOLS
        assert root._guard_managed_completion_claims() is True
        assert subagent._guard_managed_completion_claims() is False
        assert not (_FORBIDDEN_MANAGED_TOOLS & set(root.tool_manager.registered_tools))
        assert not (
            _FORBIDDEN_MANAGED_TOOLS & set(subagent.tool_manager.registered_tools)
        )
    finally:
        await root.aclose()
        await subagent.aclose()


@pytest.mark.asyncio
async def test_managed_task_rejects_writer_and_inherits_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _managed_config()
    state = VerificationState.from_recipe(config.trusted_verification_recipe)
    manager = AgentManager(lambda: config)
    ctx = InvokeContext(
        tool_call_id="managed-task",
        agent_manager=manager,
        active_model=config.active_model,
        verification_state=state,
    )
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    with pytest.raises(ToolError, match="restricts task delegation"):
        await collect_result(
            tool.run(TaskArgs(task="edit the candidate", agent="worker"), ctx)
        )
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.profile_requires_isolation",
        lambda _profile: True,
    )
    with pytest.raises(ToolError, match="write-capable or unjailed"):
        await collect_result(
            tool.run(TaskArgs(task="review the candidate", agent="reviewer"), ctx)
        )

    subagent, _ = tool._build_subagent_loop(
        TaskArgs(task="review the candidate", agent="reviewer", async_run=False), ctx
    )
    try:
        assert (
            subagent.base_config.trusted_verification_recipe
            == config.trusted_verification_recipe
        )
        assert set(subagent.tool_manager.available_tools) <= _SUBAGENT_TOOLS
        assert {"bash", "glob", "grep", "read"} <= set(
            subagent.tool_manager.available_tools
        )
    finally:
        await subagent.aclose()
