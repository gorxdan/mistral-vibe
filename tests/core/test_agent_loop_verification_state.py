from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import build_test_agent_loop
from tests.stubs.fake_tool import FakeTool, FakeToolArgs, FakeToolState
from vibe.core.agent_loop import ToolDecision, ToolExecutionResponse
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.tools.background import BackgroundRegistry
from vibe.core.tools.base import BaseToolConfig, ToolAuthorizationSource, ToolError
from vibe.core.tools.builtins.bash import Bash, BashArgs
from vibe.core.tools.builtins.task import Task, TaskArgs
from vibe.core.tools.builtins.write_file import WriteFileArgs
from vibe.core.tools.permissions import (
    PermissionContext,
    ToolPermission,
    authorization_context_fingerprint,
)
from vibe.core.tracing import tool_span
from vibe.core.verification_contract import (
    CommandEvidence,
    VerificationReport,
    VerificationVerdict,
)
from vibe.core.verification_state import VerifierAttemptDisposition


def _verifier_pass() -> VerificationReport:
    return VerificationReport(
        verdict=VerificationVerdict.PASS,
        evidence=(
            CommandEvidence(
                check="focused",
                command="pytest -q",
                output="all passed",
                result=VerificationVerdict.PASS,
            ),
        ),
    )


def _record_current_verifier_pass(state) -> None:
    generation = state.begin_verifier_attempt()
    assert state.record_verifier_result(
        generation,
        VerifierAttemptDisposition.PASS,
        "Verifier PASS was recorded for the current candidate.",
    )
    state.record_verifier_pass(_verifier_pass(), verifier_attempt_generation=generation)


def _authorized_decision(tool, args) -> ToolDecision:
    permission = tool.resolve_permission(args) or PermissionContext(
        permission=tool.config.permission
    )
    return ToolDecision(
        verdict=ToolExecutionResponse.EXECUTE,
        approval_type=ToolPermission.ASK,
        authorization_source=ToolAuthorizationSource.USER,
        authorization_fingerprint=authorization_context_fingerprint(
            tool.get_name(), args, permission
        ),
    )


def _register_fake_tool(loop, tool: FakeTool) -> None:
    loop.tool_manager._all_tools[tool.get_name()] = type(tool)
    loop.tool_manager._instances[tool.get_name()] = tool


def test_read_only_verifier_task_is_not_a_candidate_mutation() -> None:
    loop = build_test_agent_loop()
    call = ResolvedToolCall(
        tool_name="task",
        tool_class=Task,
        validated_args=TaskArgs(
            task="Inspect and report a strict verdict.",
            agent="verifier",
            async_run=False,
        ),
        call_id="verifier",
    )

    assert not loop._verification_tool_may_mutate_candidate(call)


def test_background_bash_invalidates_authorization_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "workspace"
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    _record_current_verifier_pass(state)
    call = ResolvedToolCall(
        tool_name="bash",
        tool_class=Bash,
        validated_args=BashArgs(command="touch delayed", background=True),
        call_id="background-write",
    )

    tracked = loop._verification_tool_may_mutate_candidate(call)

    assert tracked
    assert loop._preinvalidate_async_candidate_tool(call, tracked=tracked)
    assert state.verification_required
    assert state.last_verifier_pass is None


@pytest.mark.asyncio
async def test_non_read_only_noop_keeps_current_verifier_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "workspace"
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    _record_current_verifier_pass(state)
    tool_instance = FakeTool(
        lambda: BaseToolConfig(permission=ToolPermission.ALWAYS), FakeToolState()
    )
    _register_fake_tool(loop, tool_instance)
    tc = ResolvedToolCall(
        tool_name="stub_tool",
        tool_class=FakeTool,
        validated_args=FakeToolArgs(text="noop"),
        call_id="c1",
    )
    decision = _authorized_decision(tool_instance, tc.validated_args)

    async with tool_span(
        tool_name="stub_tool", call_id="c1", arguments='{"text":"noop"}'
    ) as span:
        async for _ in loop._invoke_tool(
            tc, tool_instance, tc.args_dict, decision, span=span
        ):
            pass

    assert state.has_verifier_pass()


@pytest.mark.asyncio
async def test_workspace_mutation_invalidates_preserved_verifier_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    state = loop._verification_state
    tool_instance = loop.tool_manager.get("write_file")
    marker = Path("mutation-marker")
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint",
        lambda: "changed" if marker.exists() else "workspace",
    )
    _record_current_verifier_pass(state)
    tc = ResolvedToolCall(
        tool_name="write_file",
        tool_class=tool_instance.__class__,
        validated_args=WriteFileArgs(path=str(marker), content="changed"),
        call_id="c2",
    )
    decision = _authorized_decision(tool_instance, tc.validated_args)

    async with tool_span(tool_name="write_file", call_id="c2", arguments="{}") as span:
        async for _ in loop._invoke_tool(
            tc, tool_instance, tc.args_dict, decision, span=span
        ):
            pass

    assert state.last_verifier_pass is not None
    assert not state.has_verifier_pass()


@pytest.mark.asyncio
async def test_effectful_tool_requires_verification_when_git_fingerprint_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: None
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    tool_instance = loop.tool_manager.get("write_file")
    tc = ResolvedToolCall(
        tool_name="write_file",
        tool_class=tool_instance.__class__,
        validated_args=WriteFileArgs(path="outside-git.txt", content="changed"),
        call_id="outside-git",
    )
    decision = _authorized_decision(tool_instance, tc.validated_args)

    async with tool_span(
        tool_name="write_file", call_id="outside-git", arguments="{}"
    ) as span:
        async for _ in loop._invoke_tool(
            tc, tool_instance, tc.args_dict, decision, span=span
        ):
            pass

    constraint = state.completion_constraint(receipt_valid=False)
    assert state.verification_required
    assert constraint is not None
    assert constraint.status.value == "unverified"


@pytest.mark.asyncio
async def test_failed_effectful_tool_is_observed_when_fingerprint_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: None
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    _record_current_verifier_pass(state)
    tool_instance = FakeTool(
        lambda: BaseToolConfig(permission=ToolPermission.ALWAYS), FakeToolState()
    )
    _register_fake_tool(loop, tool_instance)
    tool_instance._exception_to_raise = ToolError("failed after a partial write")
    call = ResolvedToolCall(
        tool_name="stub_tool",
        tool_class=FakeTool,
        validated_args=FakeToolArgs(text="mutate"),
        call_id="failed-write",
    )
    decision = _authorized_decision(tool_instance, call.validated_args)

    with pytest.raises(ToolError, match="partial write"):
        async with tool_span(
            tool_name="stub_tool", call_id="failed-write", arguments="{}"
        ) as span:
            async for _ in loop._invoke_tool(
                call, tool_instance, call.args_dict, decision, span=span
            ):
                pass

    assert state.verification_required
    assert state.last_verifier_pass is None


@pytest.mark.parametrize(
    ("category", "status"),
    [
        ("process", "running"),
        ("async_agent", "running"),
        ("workflow", "paused"),
        ("agent", "running"),
        ("team", "running"),
        ("loop", "waiting"),
    ],
)
def test_pending_background_work_revokes_reverification_authority(
    monkeypatch: pytest.MonkeyPatch, category: str, status: str
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "workspace"
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    _record_current_verifier_pass(state)
    registry = BackgroundRegistry()
    loop.background_registry = registry
    monkeypatch.setattr(
        registry,
        "list_tasks",
        lambda **_kwargs: [SimpleNamespace(category=category, status=status)],
    )

    constraint = loop._verification_completion_constraint(receipt_valid=True)

    assert constraint is not None
    assert constraint.status.value == "unverified"
    assert state.last_verifier_pass is None


@pytest.mark.parametrize(
    "status",
    [
        "blocked",
        "cancelled",
        "completed",
        "completed_with_failures",
        "failed",
        "stopped",
    ],
)
def test_terminal_background_work_preserves_reverification_authority(
    monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "workspace"
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    _record_current_verifier_pass(state)
    registry = BackgroundRegistry()
    loop.background_registry = registry
    monkeypatch.setattr(
        registry, "list_tasks", lambda **_kwargs: [SimpleNamespace(status=status)]
    )

    constraint = loop._verification_completion_constraint(receipt_valid=True)

    assert constraint is None
    assert state.last_verifier_pass is not None


def test_background_registry_failure_revokes_reverification_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "workspace"
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    _record_current_verifier_pass(state)
    registry = BackgroundRegistry()
    loop.background_registry = registry

    def fail_list_tasks(**_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry, "list_tasks", fail_list_tasks)

    constraint = loop._verification_completion_constraint(receipt_valid=True)

    assert constraint is not None
    assert constraint.status.value == "unverified"
    assert state.last_verifier_pass is None
