from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.agent_loop import ToolDecision, ToolExecutionResponse
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.tools.base import ToolError
from vibe.core.tools.builtins.workflow_results import (
    WorkflowResults,
    WorkflowResultsArgs,
    WorkflowResultsResult,
)
from vibe.core.tools.permissions import ToolPermission
from vibe.core.tracing import tool_span
from vibe.core.types import ToolResultEvent


@pytest.mark.asyncio
async def test_workflow_results_callback_forwarded_into_tool_context() -> None:
    # Regression: AgentLoop set workflow_results_callback on itself (app.py) but
    # never forwarded it into the InvokeContext built inside _invoke_tool, so the
    # workflow_results tool always raised "no results callback wired" and the host
    # could never pull a run's results -- forcing hand-parsing of agent logs even
    # for completed runs. Mirrors how the status/stop callbacks are already wired.
    loop = build_test_agent_loop()

    def cb(run_id: str, *, phase: str | None = None, raw: bool = False) -> dict:
        return {
            "run_id": run_id,
            "status": "completed",
            "phases": [],
            "agent_results": [],
            "return_value": {"recovered": True},
        }

    loop.workflow_results_callback = cb
    tool_instance = loop.tool_manager.get("workflow_results")

    tc = ResolvedToolCall(
        tool_name="workflow_results",
        tool_class=WorkflowResults,
        validated_args=WorkflowResultsArgs(run_id="wf-1"),
        call_id="c1",
    )
    decision = ToolDecision(
        verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
    )

    async with tool_span(
        tool_name="workflow_results", call_id="c1", arguments="{}"
    ) as span:
        events = [
            e
            async for e in loop._invoke_tool(
                tc, tool_instance, tc.args_dict, decision, span=span
            )
        ]

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results, "tool produced a result event"
    result = results[0].result
    assert isinstance(result, WorkflowResultsResult)
    assert result.return_value == {"recovered": True}


@pytest.mark.asyncio
async def test_workflow_results_without_callback_raises_specific_error() -> None:
    # The negative case: with no callback wired, the tool must fail with the
    # specific, documented error (not a silent None / empty result). This pins
    # the contract so the forwarding fix above is what makes the tool usable.
    loop = build_test_agent_loop()
    tool_instance = loop.tool_manager.get("workflow_results")

    tc = ResolvedToolCall(
        tool_name="workflow_results",
        tool_class=WorkflowResults,
        validated_args=WorkflowResultsArgs(run_id="wf-1"),
        call_id="c1",
    )
    decision = ToolDecision(
        verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
    )

    async with tool_span(
        tool_name="workflow_results", call_id="c1", arguments="{}"
    ) as span:
        with pytest.raises(ToolError, match="no results callback wired"):
            async for _ in loop._invoke_tool(
                tc, tool_instance, tc.args_dict, decision, span=span
            ):
                pass
