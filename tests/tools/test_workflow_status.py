from __future__ import annotations

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.workflow_status import (
    WorkflowStatus,
    WorkflowStatusArgs,
    WorkflowStatusConfig,
)


def _make_tool() -> WorkflowStatus:
    return WorkflowStatus(
        config_getter=lambda: WorkflowStatusConfig(), state=BaseToolState()
    )


def _ctx(callback) -> InvokeContext:
    return InvokeContext(tool_call_id="t1", workflow_status_callback=callback)


@pytest.mark.asyncio
async def test_returns_all_runs_when_no_id() -> None:
    payload = [
        {"run_id": "wf-1", "status": "running", "live_agent_count": 2},
        {"run_id": "wf-2", "status": "completed", "live_agent_count": 0},
    ]
    result = await collect_result(
        _make_tool().run(WorkflowStatusArgs(), ctx=_ctx(lambda _id=None: payload))
    )
    assert [r["run_id"] for r in result.runs] == ["wf-1", "wf-2"]


@pytest.mark.asyncio
async def test_filters_to_one_run_id() -> None:
    seen: list[str | None] = []

    def cb(run_id=None):
        seen.append(run_id)
        return [{"run_id": run_id or "wf-9", "status": "running"}]

    result = await collect_result(
        _make_tool().run(WorkflowStatusArgs(run_id="wf-3"), ctx=_ctx(cb))
    )
    assert seen == ["wf-3"]
    assert result.runs[0]["run_id"] == "wf-3"


@pytest.mark.asyncio
async def test_errors_without_callback() -> None:
    with pytest.raises(ToolError, match="not available"):
        await collect_result(
            _make_tool().run(WorkflowStatusArgs(), ctx=InvokeContext(tool_call_id="t"))
        )


@pytest.mark.asyncio
async def test_errors_without_context() -> None:
    with pytest.raises(ToolError, match="requires context"):
        await collect_result(_make_tool().run(WorkflowStatusArgs()))


def test_hidden_when_workflows_disabled() -> None:
    from tests.conftest import build_test_vibe_config

    cfg = build_test_vibe_config(disable_workflows=True)
    assert WorkflowStatus.is_available(cfg) is False
