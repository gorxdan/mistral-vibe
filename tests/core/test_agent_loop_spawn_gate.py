from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.agent_loop_spawn_gate import (
    AGENT_SPAWN_BATCH_DENIAL,
    plan_agent_spawn_batch,
)
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.tools.builtins.launch_workflow import LaunchWorkflow, LaunchWorkflowArgs
from vibe.core.tools.builtins.task import Task, TaskArgs
from vibe.core.tools.builtins.team_spawn import TeamSpawn, TeamSpawnArgs
from vibe.core.types import ToolResultEvent


class _Tool:
    is_subagent_spawner = False


class _Spawner:
    is_subagent_spawner = True


@dataclass(frozen=True)
class _Call:
    tool_name: str
    tool_class: type = _Tool


def _names(calls: tuple[_Call, ...]) -> list[str]:
    return [call.tool_name for call in calls]


def test_third_task_is_rejected_before_the_batch_runs() -> None:
    calls = [
        _Call("task-1", tool_class=_Spawner),
        _Call("task-2", tool_class=_Spawner),
        _Call("task-3", tool_class=_Spawner),
    ]

    plan = plan_agent_spawn_batch(calls)

    assert _names(plan.accepted) == ["task-1", "task-2"]
    assert _names(plan.rejected) == ["task-3"]


def test_task_team_and_workflow_share_the_same_agent_budget() -> None:
    calls = [
        _Call("task", tool_class=_Spawner),
        _Call("team_spawn"),
        _Call("launch_workflow"),
    ]

    plan = plan_agent_spawn_batch(calls)

    assert _names(plan.accepted) == ["task", "team_spawn"]
    assert _names(plan.rejected) == ["launch_workflow"]


def test_workflow_reserves_both_available_agent_slots() -> None:
    calls = [
        _Call("launch_workflow"),
        _Call("task", tool_class=_Spawner),
        _Call("team_spawn"),
    ]

    plan = plan_agent_spawn_batch(calls)

    assert _names(plan.accepted) == ["launch_workflow"]
    assert _names(plan.rejected) == ["task", "team_spawn"]


def test_task_agents_consume_capacity_but_nonspawning_tools_do_not() -> None:
    calls = [
        _Call("task", _Spawner),
        _Call("read"),
        _Call("task", _Spawner),
        _Call("team_spawn"),
    ]

    plan = plan_agent_spawn_batch(calls)

    assert _names(plan.accepted) == ["task", "read", "task"]
    assert _names(plan.rejected) == ["team_spawn"]


def test_a_later_batch_gets_a_fresh_bounded_budget() -> None:
    first = plan_agent_spawn_batch([
        _Call("task-1", tool_class=_Spawner),
        _Call("task-2", tool_class=_Spawner),
        _Call("task-3", tool_class=_Spawner),
    ])
    later = plan_agent_spawn_batch([_Call("task-4", tool_class=_Spawner)])

    assert _names(first.rejected) == ["task-3"]
    assert _names(later.accepted) == ["task-4"]


def _task_call(call_id: str) -> ResolvedToolCall:
    return ResolvedToolCall(
        tool_name="task",
        tool_class=Task,
        validated_args=TaskArgs(task=f"Inspect {call_id}", agent="explore"),
        call_id=call_id,
    )


@pytest.mark.asyncio
async def test_loop_invokes_only_two_tasks_from_one_assistant_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    invoked: list[str] = []

    async def fake_process(
        call: ResolvedToolCall, _scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        invoked.append(call.call_id)
        yield ToolResultEvent(
            tool_name=call.tool_name,
            tool_class=call.tool_class,
            tool_call_id=call.call_id,
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)

    results = [
        event
        async for event in loop._run_tools_concurrently([
            _task_call("task-1"),
            _task_call("task-2"),
            _task_call("task-3"),
        ])
        if isinstance(event, ToolResultEvent)
    ]

    assert invoked == ["task-1", "task-2"]
    assert {event.tool_call_id for event in results} == {"task-1", "task-2", "task-3"}
    rejected = next(event for event in results if event.tool_call_id == "task-3")
    assert AGENT_SPAWN_BATCH_DENIAL in (rejected.error or "")


@pytest.mark.asyncio
async def test_loop_uses_one_budget_for_mixed_productive_spawners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    invoked: list[str] = []

    async def fake_process(
        call: ResolvedToolCall, _scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        invoked.append(call.call_id)
        yield ToolResultEvent(
            tool_name=call.tool_name,
            tool_class=call.tool_class,
            tool_call_id=call.call_id,
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)
    calls = [
        _task_call("task"),
        ResolvedToolCall(
            tool_name="team_spawn",
            tool_class=TeamSpawn,
            validated_args=TeamSpawnArgs(name="team", prompt="Inspect it"),
            call_id="team",
        ),
        ResolvedToolCall(
            tool_name="launch_workflow",
            tool_class=LaunchWorkflow,
            validated_args=LaunchWorkflowArgs(
                script="async def main():\n    return []"
            ),
            call_id="workflow",
        ),
    ]

    results = [
        event
        async for event in loop._run_tools_concurrently(calls)
        if isinstance(event, ToolResultEvent)
    ]

    assert invoked == ["task", "team"]
    rejected = next(event for event in results if event.tool_call_id == "workflow")
    assert AGENT_SPAWN_BATCH_DENIAL in (rejected.error or "")
