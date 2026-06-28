from __future__ import annotations

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.workflow_results import (
    WorkflowResults,
    WorkflowResultsArgs,
    WorkflowResultsConfig,
)


def _make_tool() -> WorkflowResults:
    return WorkflowResults(
        config_getter=lambda: WorkflowResultsConfig(), state=BaseToolState()
    )


def _ctx(callback) -> InvokeContext:
    return InvokeContext(tool_call_id="t1", workflow_results_callback=callback)


@pytest.mark.asyncio
async def test_returns_phases_and_agent_outputs() -> None:
    payload = {
        "run_id": "wf-1",
        "status": "completed_with_failures",
        "phases": [{"name": "audit", "agents": 2, "completed": 1, "failed": 1}],
        "agent_results": [
            {
                "label": "ok",
                "agent": "explore",
                "phase": "audit",
                "completed": True,
                "response": "found it",
                "error": None,
                "tokens_in": 100,
                "tokens_out": 20,
            },
            {
                "label": "bad",
                "agent": "explore",
                "phase": "audit",
                "completed": False,
                "response": "partial work",
                "error": "boom",
                "tokens_in": 100,
                "tokens_out": 0,
            },
        ],
    }
    result = await collect_result(
        _make_tool().run(
            WorkflowResultsArgs(run_id="wf-1"), ctx=_ctx(lambda *a, **k: payload)
        )
    )
    assert result.run_id == "wf-1"
    assert result.status == "completed_with_failures"
    assert result.phases[0]["failed"] == 1
    assert len(result.agent_results) == 2
    assert result.agent_results[0]["completed"] is True
    assert result.agent_results[1]["error"] == "boom"
    # The callback receives the raw/phase kwargs forwarded by the tool.
    assert result.agent_results[0]["agent"] == "explore"


@pytest.mark.asyncio
async def test_forwards_phase_and_raw_kwargs_to_callback() -> None:
    seen: dict[str, object] = {}

    def cb(run_id, *, phase=None, raw=False):
        seen.update(run_id=run_id, phase=phase, raw=raw)
        return {
            "run_id": run_id,
            "status": "running",
            "phases": [],
            "agent_results": [],
        }

    await collect_result(
        _make_tool().run(
            WorkflowResultsArgs(run_id="wf-2", phase="verify", raw=True), ctx=_ctx(cb)
        )
    )
    assert seen == {"run_id": "wf-2", "phase": "verify", "raw": True}


@pytest.mark.asyncio
async def test_errors_without_callback() -> None:
    with pytest.raises(ToolError, match="not available"):
        await collect_result(
            _make_tool().run(
                WorkflowResultsArgs(run_id="wf-1"), ctx=InvokeContext(tool_call_id="t")
            )
        )


@pytest.mark.asyncio
async def test_errors_without_context() -> None:
    with pytest.raises(ToolError, match="requires context"):
        await collect_result(_make_tool().run(WorkflowResultsArgs(run_id="wf-1")))


def test_hidden_when_workflows_disabled() -> None:
    from tests.conftest import build_test_vibe_config

    cfg = build_test_vibe_config(disable_workflows=True)
    assert WorkflowResults.is_available(cfg) is False


def test_permission_is_always_allow() -> None:
    # Read-only retrieval of run data the host already owns. No ASK gate.
    tool = _make_tool()
    ctx = tool.resolve_permission(WorkflowResultsArgs(run_id="wf-1"))
    from vibe.core.tools.base import ToolPermission

    assert ctx is not None
    assert ctx.permission == ToolPermission.ALWAYS
