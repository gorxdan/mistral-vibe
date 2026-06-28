from __future__ import annotations

from typing import cast

from pydantic import ValidationError
import pytest

from tests.mock.utils import collect_result
from vibe.core.config import VibeConfig
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.workflow_stop import (
    WorkflowStop,
    WorkflowStopArgs,
    WorkflowStopConfig,
)


def _make_tool() -> WorkflowStop:
    return WorkflowStop(
        config_getter=lambda: WorkflowStopConfig(), state=BaseToolState()
    )


def _ctx(callback) -> InvokeContext:
    return InvokeContext(tool_call_id="t1", workflow_stop_callback=callback)


@pytest.mark.asyncio
async def test_stops_one_run_id() -> None:
    seen: list[tuple] = []

    async def cb(run_id, all_runs):
        seen.append((run_id, all_runs))
        return {
            "stopped": True,
            "stopped_run_ids": [run_id],
            "message": f"Stopped workflow `{run_id}`.",
        }

    result = await collect_result(
        _make_tool().run(WorkflowStopArgs(run_id="wf-1"), ctx=_ctx(cb))
    )
    assert seen == [("wf-1", False)]
    assert result.stopped is True
    assert result.stopped_run_ids == ["wf-1"]


@pytest.mark.asyncio
async def test_stops_all() -> None:
    seen: list[tuple] = []

    async def cb(run_id, all_runs):
        seen.append((run_id, all_runs))
        return {
            "stopped": True,
            "stopped_run_ids": ["wf-1", "wf-2"],
            "message": "Stopped 2 workflow run(s).",
        }

    result = await collect_result(
        _make_tool().run(WorkflowStopArgs(all=True), ctx=_ctx(cb))
    )
    assert seen == [(None, True)]
    assert result.stopped_run_ids == ["wf-1", "wf-2"]


@pytest.mark.asyncio
async def test_reports_already_finished_run() -> None:
    async def cb(run_id, all_runs):
        return {
            "stopped": False,
            "stopped_run_ids": [],
            "message": f"Could not stop `{run_id}` — not found or already finished.",
        }

    result = await collect_result(
        _make_tool().run(WorkflowStopArgs(run_id="wf-9"), ctx=_ctx(cb))
    )
    assert result.stopped is False
    assert result.stopped_run_ids == []


@pytest.mark.asyncio
async def test_errors_without_callback() -> None:
    with pytest.raises(ToolError, match="not available"):
        await collect_result(
            _make_tool().run(
                WorkflowStopArgs(run_id="wf-1"), ctx=InvokeContext(tool_call_id="t")
            )
        )


@pytest.mark.asyncio
async def test_errors_without_context() -> None:
    with pytest.raises(ToolError, match="requires context"):
        await collect_result(_make_tool().run(WorkflowStopArgs(run_id="wf-1")))


def test_requires_target() -> None:
    with pytest.raises(ValidationError):
        WorkflowStopArgs()


def test_hidden_when_workflows_disabled() -> None:
    class _Cfg:
        disable_workflows = True

    assert WorkflowStop.is_available(cast(VibeConfig, _Cfg())) is False


def test_available_by_default() -> None:
    class _Cfg:
        disable_workflows = False

    assert WorkflowStop.is_available(cast(VibeConfig, _Cfg())) is True
