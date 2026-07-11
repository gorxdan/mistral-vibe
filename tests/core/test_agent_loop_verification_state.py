from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_agent_loop
from tests.stubs.fake_tool import FakeTool, FakeToolArgs, FakeToolState
from vibe.core.agent_loop import ToolDecision, ToolExecutionResponse
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.tools.base import BaseToolConfig
from vibe.core.tools.builtins.write_file import WriteFileArgs
from vibe.core.tools.permissions import ToolPermission
from vibe.core.tracing import tool_span
from vibe.core.verification_contract import (
    CommandEvidence,
    VerificationReport,
    VerificationVerdict,
)


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


@pytest.mark.asyncio
async def test_non_read_only_noop_keeps_current_verifier_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "workspace"
    )
    loop = build_test_agent_loop()
    state = loop._verification_state
    state.record_verifier_pass(_verifier_pass())
    tool_instance = FakeTool(
        lambda: BaseToolConfig(permission=ToolPermission.ALWAYS), FakeToolState()
    )
    tc = ResolvedToolCall(
        tool_name="stub_tool",
        tool_class=FakeTool,
        validated_args=FakeToolArgs(text="noop"),
        call_id="c1",
    )
    decision = ToolDecision(
        verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
    )

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
    state.record_verifier_pass(_verifier_pass())
    tc = ResolvedToolCall(
        tool_name="write_file",
        tool_class=tool_instance.__class__,
        validated_args=WriteFileArgs(path=str(marker), content="changed"),
        call_id="c2",
    )
    decision = ToolDecision(
        verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
    )

    async with tool_span(tool_name="write_file", call_id="c2", arguments="{}") as span:
        async for _ in loop._invoke_tool(
            tc, tool_instance, tc.args_dict, decision, span=span
        ):
            pass

    assert state.last_verifier_pass is not None
    assert not state.has_verifier_pass()
