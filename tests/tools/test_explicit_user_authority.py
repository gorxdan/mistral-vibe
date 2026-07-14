from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop
from tests.mock.utils import collect_result
from vibe.cli.textual_ui.app import VibeApp
from vibe.core.agent_loop_models import ToolExecutionResponse
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.programmatic import _isolated_auto_approve
from vibe.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolAuthorizationSource,
    ToolError,
    ToolPermission,
    ToolPermissionError,
)
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.builtins.write_file import WriteFileArgs
from vibe.core.tools.permissions import ApprovedRule, PermissionContext, PermissionScope
from vibe.core.types import ApprovalResponse


class _RecordingApproval:
    def __init__(self, response: ApprovalResponse) -> None:
        self.response = response
        self.calls = 0

    async def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[ApprovalResponse, None, None]:
        self.calls += 1
        return self.response, None, None


def _always_bash() -> Bash:
    return Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ALWAYS),
        state=BaseToolState(),
    )


@pytest.mark.asyncio
async def test_covered_carrier_rule_cannot_replace_explicit_user_approval() -> None:
    loop = build_test_agent_loop()
    loop._permission_store.add_rule(
        ApprovedRule(
            tool_name="bash",
            scope=PermissionScope.COMMAND_PATTERN,
            session_pattern="find *",
        )
    )
    tool = _always_bash()
    args = BashArgs(command=r"find . -exec npm install \;")
    context = tool.resolve_permission(args)
    approval = _RecordingApproval(ApprovalResponse.NO)
    loop.approval_callback = approval

    assert context is not None
    assert context.requires_explicit_user_approval
    assert context.required_permissions
    assert all(
        loop._permission_store.covers("bash", required)
        for required in context.required_permissions
    )

    decision = await loop._should_execute_tool(tool, args, "covered-carrier")

    assert decision.verdict is ToolExecutionResponse.SKIP
    assert approval.calls == 1


@pytest.mark.asyncio
async def test_subagent_can_forward_explicit_gate_to_root_user() -> None:
    loop = build_test_agent_loop(is_subagent=True)
    approval = _RecordingApproval(ApprovalResponse.YES)
    loop.approval_callback = approval

    decision = await loop._should_execute_tool(
        _always_bash(), BashArgs(command="npm install"), "forwarded-authority"
    )

    assert decision.verdict is ToolExecutionResponse.EXECUTE
    assert decision.authorization_source is ToolAuthorizationSource.USER
    assert approval.calls == 1


@pytest.mark.asyncio
async def test_autoapprove_forwards_explicit_gate_to_present_root_user() -> None:
    loop = build_test_agent_loop(agent_name=BuiltinAgentName.AUTO_APPROVE)
    approval = _RecordingApproval(ApprovalResponse.YES)
    loop.approval_callback = approval

    decision = await loop._should_execute_tool(
        _always_bash(), BashArgs(command="npm install"), "autoapprove-root-gate"
    )

    assert decision.verdict is ToolExecutionResponse.EXECUTE
    assert decision.authorization_source is ToolAuthorizationSource.USER
    assert approval.calls == 1


@pytest.mark.asyncio
async def test_in_place_approval_argument_mutation_invalidates_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    permission = PermissionContext(permission=ToolPermission.ASK)
    tool = Bash(
        config_getter=lambda: BashToolConfig(permission=ToolPermission.ASK),
        state=BaseToolState(),
    )
    args = BashArgs(command="cat README.md")

    async def mutate(
        _tool_name: str,
        approval_args: BaseModel,
        _tool_call_id: str,
        _required_permissions: list[Any] | None,
        _judge_note: str | None,
    ) -> tuple[ApprovalResponse, None, None]:
        assert isinstance(approval_args, BashArgs)
        approval_args.command = "echo substituted"
        return ApprovalResponse.YES, None, None

    async def fail_if_spawned(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("spawn must not be reached")

    loop.approval_callback = mutate
    monkeypatch.setattr(tool, "resolve_permission", lambda _args: permission)
    monkeypatch.setattr(tool, "_start_foreground", fail_if_spawned)

    decision = await loop._should_execute_tool(tool, args, "mutated-approval")
    ctx = InvokeContext(
        tool_call_id="mutated-approval",
        authorization_source=decision.authorization_source,
        authorization_fingerprint=decision.authorization_fingerprint,
    )

    with pytest.raises(ToolError, match="authorization context changed"):
        await collect_result(tool.run(args, ctx))


@pytest.mark.asyncio
async def test_non_bash_bypass_revocation_is_checked_at_invoke(
    tmp_working_directory,
) -> None:
    loop = build_test_agent_loop(agent_name=BuiltinAgentName.AUTO_APPROVE)
    tool = loop.tool_manager.get("write_file")
    target = tmp_working_directory / "must-not-exist.txt"
    args = WriteFileArgs(path=str(target), content="blocked\n")

    decision = await loop._should_execute_tool(tool, args, "revoked-bypass")
    assert decision.authorization_source is ToolAuthorizationSource.BYPASS
    loop.agent_manager.config.bypass_tool_permissions = False
    ctx = InvokeContext(
        tool_call_id="revoked-bypass",
        agent_manager=loop.agent_manager,
        authorization_source=decision.authorization_source,
        authorization_fingerprint=decision.authorization_fingerprint,
    )

    with pytest.raises(ToolPermissionError, match="auto-approve authority changed"):
        async for _item in tool.invoke(ctx=ctx, **args.model_dump()):
            pass

    assert not target.exists()


@pytest.mark.asyncio
async def test_isolated_autoapprove_refuses_a_deferred_gate() -> None:
    response, feedback, modified = await _isolated_auto_approve(
        "bash",
        BashArgs(command="npm install"),
        "isolated-authority",
        None,
        "Package changes require explicit user approval.",
    )

    assert response is ApprovalResponse.NO
    assert feedback == "Package changes require explicit user approval."
    assert modified is None


@pytest.mark.parametrize("use_callback_note", [False, True])
@pytest.mark.asyncio
async def test_tui_autoapprove_does_not_consume_a_deferred_gate(
    use_callback_note: bool,
) -> None:
    reason = "Package changes require explicit user approval."
    shown_notes: list[str | None] = []

    async def wait_for_typing_pause() -> None:
        return None

    async def switch_to_approval(
        _tool: str,
        _args: BaseModel,
        _required_permissions: list[Any] | None,
        *,
        judge_note: str | None,
    ) -> None:
        shown_notes.append(judge_note)
        host._pending_approval.set_result((ApprovalResponse.NO, None, None))

    async def switch_to_input() -> None:
        return None

    host: Any = SimpleNamespace(
        agent_loop=SimpleNamespace(
            config=SimpleNamespace(bypass_tool_permissions=True),
            pending_judge_deferral=None if use_callback_note else reason,
        ),
        _user_interaction_lock=asyncio.Lock(),
        _wait_for_typing_pause=wait_for_typing_pause,
        _pending_approval=None,
        _terminal_notifier=SimpleNamespace(notify=lambda _context: None),
        _loading_widget=None,
        _switch_to_approval_app=switch_to_approval,
        _switch_to_input_app=switch_to_input,
    )

    response = await VibeApp._approval_callback(
        host,
        "bash",
        BashArgs(command="npm install"),
        "tui-authority",
        None,
        reason if use_callback_note else None,
    )

    assert response[0] is ApprovalResponse.NO
    assert shown_notes == [reason]
