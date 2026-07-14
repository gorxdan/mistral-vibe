from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
import sys
from unittest.mock import Mock

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import AgentLoop
from vibe.core.agent_loop_models import ToolDecision, ToolExecutionResponse
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import MemoryConfig
from vibe.core.hooks.config import HookConfigResult
from vibe.core.hooks.models import HookConfig, HookStartEvent, HookType
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.orchestration import OrchestrationDecision
from vibe.core.tools.background import BackgroundRegistry
from vibe.core.tools.base import ToolPermission
from vibe.core.tools.builtins.edit import Edit, EditArgs
from vibe.core.tools.builtins.grep import Grep
from vibe.core.tools.builtins.launch_workflow import (
    LaunchWorkflow,
    LaunchWorkflowArgs,
    LaunchWorkflowResult,
)
from vibe.core.tools.builtins.manage_memory import ManageMemory, ManageMemoryArgs
from vibe.core.tools.builtins.read import Read, ReadArgs
from vibe.core.tools.builtins.task import Task, TaskArgs, TaskResult
from vibe.core.tools.builtins.work_strategy import WorkStrategy
from vibe.core.tools.builtins.write_file import WriteFile, WriteFileArgs
from vibe.core.types import (
    ApprovalResponse,
    FunctionCall,
    LLMMessage,
    Role,
    ToolCall,
    ToolResultEvent,
)
from vibe.core.utils.io import read_safe, write_safe
from vibe.core.workflows.models import WorkflowLaneAttestation


class _Args(BaseModel):
    pass


def _call(name: str, tool_class: type) -> ResolvedToolCall:
    return ResolvedToolCall(
        tool_name=name, tool_class=tool_class, validated_args=_Args(), call_id=name
    )


def test_read_tools_are_read_only_writers_are_not() -> None:
    assert Read.read_only is True
    assert Grep.read_only is True
    assert WriteFile.read_only is False


def test_manage_memory_list_is_observational_but_mutations_are_not() -> None:
    assert ManageMemory.read_only is False
    assert ManageMemory.call_is_read_only(ManageMemoryArgs(action="list")) is True
    for action in ("add", "update", "delete"):
        assert ManageMemory.call_is_read_only(ManageMemoryArgs(action=action)) is False


@pytest.mark.asyncio
async def test_tool_calls_run_in_ordered_waves(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = build_test_agent_loop()
    events: list[str] = []

    async def fake_process(
        tc: ResolvedToolCall, scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        assert scheduled_read_only is tc.tool_class.read_only
        events.append(f"start:{tc.tool_name}")
        await asyncio.sleep(0.02)
        events.append(f"end:{tc.tool_name}")
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)

    calls = [
        _call("read1", Read),
        _call("read2", Grep),
        _call("write1", WriteFile),
        _call("read3", Read),
        _call("write2", WriteFile),
        _call("read4", Read),
        _call("read5", Grep),
    ]
    collected = [e async for e in loop._run_tools_concurrently(calls)]
    assert len(collected) == 7

    assert max(events.index("start:read1"), events.index("start:read2")) < min(
        events.index("end:read1"), events.index("end:read2")
    )
    assert max(events.index("end:read1"), events.index("end:read2")) < events.index(
        "start:write1"
    )
    assert events.index("end:write1") < events.index("start:read3")
    assert events.index("end:read3") < events.index("start:write2")
    assert events.index("end:write2") < min(
        events.index("start:read4"), events.index("start:read5")
    )
    assert max(events.index("start:read4"), events.index("start:read5")) < min(
        events.index("end:read4"), events.index("end:read5")
    )


@pytest.mark.asyncio
async def test_unexpected_executor_failure_emits_result_and_aborts_later_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    started: list[str] = []

    async def fake_process(
        tc: ResolvedToolCall, scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        assert scheduled_read_only is tc.tool_class.read_only
        started.append(tc.tool_name)
        if tc.tool_name == "write1":
            raise RuntimeError("executor crashed")
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)
    calls = [
        _call("read1", Read).model_copy(update={"call_id": ""}),
        _call("write1", WriteFile).model_copy(update={"call_id": ""}),
        _call("read2", Grep).model_copy(update={"call_id": ""}),
    ]
    collected: list[ToolResultEvent] = []

    with pytest.raises(RuntimeError, match="executor crashed"):
        async for event in loop._run_tools_concurrently(calls):
            assert isinstance(event, ToolResultEvent)
            collected.append(event)

    assert started == ["read1", "write1"]
    assert [event.tool_name for event in collected] == ["read1", "write1", "read2"]
    assert collected[0].error is None
    assert all(event.error is not None for event in collected[1:])
    assert all("executor crashed" in (event.error or "") for event in collected[1:])
    assert loop.stats.tool_calls_failed == 2
    tool_messages = [
        message for message in loop.messages if message.role.value == "tool"
    ]
    assert [message.tool_call_id for message in tool_messages] == ["", ""]


@pytest.mark.asyncio
async def test_self_cancelled_executor_emits_failure_and_aborts_later_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    started: set[str] = set()

    async def fake_process(
        tc: ResolvedToolCall, scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        assert scheduled_read_only is tc.tool_class.read_only
        started.add(tc.tool_name)
        if tc.tool_name == "read1":
            raise asyncio.CancelledError
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)
    calls = [_call("read1", Read), _call("read2", Grep), _call("write1", WriteFile)]
    collected: list[ToolResultEvent] = []

    with pytest.raises(RuntimeError, match="tool executor was cancelled unexpectedly"):
        async for event in loop._run_tools_concurrently(calls):
            assert isinstance(event, ToolResultEvent)
            collected.append(event)

    assert started == {"read1", "read2"}
    assert {event.tool_name for event in collected} == {"read1", "read2", "write1"}
    assert (
        next(event for event in collected if event.tool_name == "read2").error is None
    )
    assert all(
        event.error is not None
        for event in collected
        if event.tool_name in {"read1", "write1"}
    )
    assert loop.stats.tool_calls_failed == 2


@pytest.mark.asyncio
async def test_unexpected_work_strategy_failure_completes_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    started: list[str] = []

    async def fake_process(
        tc: ResolvedToolCall, scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        assert scheduled_read_only is tc.tool_class.read_only
        started.append(tc.tool_name)
        if tc.tool_name == "work_strategy":
            raise RuntimeError("strategy crashed")
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)
    calls = [
        _call("read1", Read),
        _call("work_strategy", WorkStrategy),
        _call("read2", Grep),
    ]
    collected: list[ToolResultEvent] = []

    with pytest.raises(RuntimeError, match="strategy crashed"):
        async for event in loop._run_tools_concurrently(calls):
            assert isinstance(event, ToolResultEvent)
            collected.append(event)

    assert started == ["work_strategy"]
    assert [event.tool_name for event in collected] == [
        "work_strategy",
        "read1",
        "read2",
    ]
    assert all(event.error is not None for event in collected)
    assert loop.stats.tool_calls_failed == 3


@pytest.mark.asyncio
async def test_same_turn_read_finishes_before_edit(tmp_working_directory: Path) -> None:
    target = tmp_working_directory / "ordered.txt"
    write_safe(target, "before\n")
    config = build_test_vibe_config(
        enabled_tools=["read", "edit"],
        tools={"read": {"permission": "always"}, "edit": {"permission": "always"}},
    )
    loop = build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.AUTO_APPROVE
    )
    calls = [
        ResolvedToolCall(
            tool_name="read",
            tool_class=Read,
            validated_args=ReadArgs(file_path=str(target)),
            call_id="read",
        ),
        ResolvedToolCall(
            tool_name="edit",
            tool_class=Edit,
            validated_args=EditArgs(
                file_path=str(target), old_string="before", new_string="after"
            ),
            call_id="edit",
        ),
    ]

    results = [
        event
        async for event in loop._run_tools_concurrently(calls)
        if isinstance(event, ToolResultEvent)
    ]

    assert [event.tool_name for event in results] == ["read", "edit"]
    assert all(event.error is None for event in results)
    assert read_safe(target).text == "after\n"


@pytest.mark.asyncio
async def test_unchanged_call_uses_frozen_scheduling_classification(
    monkeypatch: pytest.MonkeyPatch, tmp_working_directory: Path
) -> None:
    target = tmp_working_directory / "stable.txt"
    write_safe(target, "stable\n")
    classifications = 0

    def classify(
        _cls: type[Read], _args: BaseModel, *, agent_manager: object = None
    ) -> bool:
        nonlocal classifications
        del agent_manager
        classifications += 1
        return classifications == 1

    monkeypatch.setattr(Read, "call_is_read_only", classmethod(classify))
    config = build_test_vibe_config(
        enabled_tools=["read"], tools={"read": {"permission": "always"}}
    )
    loop = build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.AUTO_APPROVE
    )
    call = ResolvedToolCall(
        tool_name="read",
        tool_class=Read,
        validated_args=ReadArgs(file_path=str(target)),
        call_id="read",
    )

    results = [
        event
        async for event in loop._run_tools_concurrently([call])
        if isinstance(event, ToolResultEvent)
    ]

    assert len(results) == 1
    assert results[0].error is None
    assert classifications > 1


@pytest.mark.asyncio
async def test_modified_read_only_call_cannot_become_mutating(config_dir: Path) -> None:
    tool_call = ToolCall(
        id="memory",
        index=0,
        function=FunctionCall(name="manage_memory", arguments='{"action":"list"}'),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Checking memory.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="The mutation was rejected.")],
    ])
    config = build_test_vibe_config(
        enabled_tools=["manage_memory"],
        memory=MemoryConfig(enabled=True),
        tools={"manage_memory": {"permission": "ask"}},
    )
    loop = build_test_agent_loop(config=config, backend=backend)

    async def modify(
        _tool_name: str,
        _args: BaseModel,
        _tool_call_id: str,
        _required_permissions: list | None = None,
        _judge_note: str | None = None,
    ) -> tuple[ApprovalResponse, str | None, dict[str, str] | None]:
        return (
            ApprovalResponse.MODIFY,
            None,
            {"action": "add", "title": "unsafe", "body": "must not be written"},
        )

    loop.set_approval_callback(modify)

    events = [event async for event in loop.act("Check memory")]

    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.error is not None
    assert "changed from read-only to mutating" in result.error
    assert loop.stats.tool_calls_rejected == 1
    assert list((config_dir / "memory").rglob("*.md")) == []


@pytest.mark.asyncio
async def test_in_place_approval_mutation_cannot_change_non_bash_arguments(
    tmp_working_directory: Path,
) -> None:
    original = tmp_working_directory / "original.txt"
    substituted = tmp_working_directory / "substituted.txt"
    write_safe(original, "original\n")
    write_safe(substituted, "substituted\n")
    call = ToolCall(
        id="read",
        index=0,
        function=FunctionCall(
            name="read", arguments=ReadArgs(file_path=str(original)).model_dump_json()
        ),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Read the file.", tool_calls=[call])],
        [mock_llm_chunk(content="The substituted call was rejected.")],
    ])
    config = build_test_vibe_config(
        enabled_tools=["read"], tools={"read": {"permission": "ask"}}
    )
    loop = build_test_agent_loop(config=config, backend=backend)

    async def mutate(
        _tool_name: str,
        args: BaseModel,
        _tool_call_id: str,
        _required_permissions: list | None = None,
        _judge_note: str | None = None,
    ) -> tuple[ApprovalResponse, None, None]:
        assert isinstance(args, ReadArgs)
        args.file_path = str(substituted)
        return ApprovalResponse.YES, None, None

    loop.set_approval_callback(mutate)

    events = [event async for event in loop.act("Read the original file")]

    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.error is not None
    assert "arguments or authorization context changed" in result.error
    assert result.result is None
    assert loop.stats.tool_calls_rejected == 1


@pytest.mark.asyncio
async def test_mode_switch_during_approval_disables_resolved_tool(
    tmp_working_directory: Path,
) -> None:
    target = tmp_working_directory / "must-not-exist.txt"
    config = build_test_vibe_config(
        enabled_agents=[BuiltinAgentName.DEFAULT, BuiltinAgentName.COORDINATOR],
        enabled_tools=["write_file"],
        tools={"write_file": {"permission": "ask"}},
    )
    loop = build_test_agent_loop(config=config)

    async def switch_to_coordinator(
        _tool_name: str,
        _args: BaseModel,
        _tool_call_id: str,
        _required_permissions: list | None = None,
        _judge_note: str | None = None,
    ) -> tuple[ApprovalResponse, None, None]:
        await loop.switch_agent(BuiltinAgentName.COORDINATOR)
        return ApprovalResponse.YES, None, None

    loop.set_approval_callback(switch_to_coordinator)
    call = ResolvedToolCall(
        tool_name="write_file",
        tool_class=WriteFile,
        validated_args=WriteFileArgs(path=str(target), content="unsafe"),
        call_id="write-after-mode-switch",
    )

    events = [event async for event in loop._process_one_tool_call(call, False)]

    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert "no longer enabled" in (result.error or "")
    assert not target.exists()
    assert loop.stats.tool_calls_rejected == 1


@pytest.mark.asyncio
async def test_modified_task_rejection_releases_orchestration_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(effort_mode="le-chaton", enabled_tools=["task"])
    loop = build_test_agent_loop(config=config)
    loop._begin_orchestration_turn("Investigate one independent lane.")
    loop._declare_orchestration_strategy(
        OrchestrationDecision.model_validate({
            "route": "task",
            "risk": "medium",
            "reason": "independent_lanes",
            "lanes": [
                {
                    "id": "lane-1",
                    "objective": "Inspect the independent area",
                    "owner": "agent",
                    "profile": "explore",
                }
            ],
        })
    )
    decision_count = 0

    async def decide(
        _tool: object, args: BaseModel, _tool_call_id: str
    ) -> ToolDecision:
        nonlocal decision_count
        decision_count += 1
        assert isinstance(args, TaskArgs)
        if decision_count == 1:
            return ToolDecision(
                verdict=ToolExecutionResponse.EXECUTE,
                approval_type=ToolPermission.ASK,
                modified_args={
                    "task": args.task,
                    "agent": "worker",
                    "async_run": False,
                },
            )
        return ToolDecision(
            verdict=ToolExecutionResponse.SKIP,
            approval_type=ToolPermission.ASK,
            feedback="Stop after the retry preflight",
        )

    monkeypatch.setattr(loop, "_should_execute_tool", decide)

    def task_call(call_id: str) -> ResolvedToolCall:
        return ResolvedToolCall(
            tool_name="task",
            tool_class=Task,
            validated_args=TaskArgs(
                task="[lane:lane-1] Inspect it", agent="explore", async_run=False
            ),
            call_id=call_id,
        )

    first = [
        event
        async for event in loop._process_one_tool_call(task_call("modified"), True)
    ]
    retry = [
        event async for event in loop._process_one_tool_call(task_call("retry"), True)
    ]

    first_result = next(event for event in first if isinstance(event, ToolResultEvent))
    retry_result = next(event for event in retry if isinstance(event, ToolResultEvent))
    assert "changed from read-only to mutating" in (first_result.error or "")
    assert "already reserved" not in (
        retry_result.error or retry_result.skip_reason or ""
    )
    assert decision_count == 2


@pytest.mark.asyncio
async def test_hook_modified_read_only_call_cannot_become_mutating(
    config_dir: Path,
) -> None:
    tool_call = ToolCall(
        id="memory",
        index=0,
        function=FunctionCall(name="manage_memory", arguments='{"action":"list"}'),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Checking memory.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="The mutation was rejected.")],
    ])
    script = (
        f'{sys.executable} -c "'
        "import json,sys; "
        "json.dump({'hook_specific_output': {'tool_input': "
        "{'action': 'add', 'title': 'unsafe', "
        "'body': 'must not be written'}}}, sys.stdout)"
        '"'
    )
    hooks = [
        HookConfig(
            name="mutating-rewrite",
            type=HookType.BEFORE_TOOL,
            command=script,
            match="manage_memory",
        ),
        HookConfig(
            name="must-not-run",
            type=HookType.AFTER_TOOL,
            command="true",
            match="manage_memory",
        ),
    ]
    config = build_test_vibe_config(
        enabled_tools=["manage_memory"],
        memory=MemoryConfig(enabled=True),
        tools={"manage_memory": {"permission": "always"}},
    )
    loop = build_test_agent_loop(
        config=config,
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
        hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
    )

    events = [event async for event in loop.act("Check memory")]

    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.error is not None
    assert "changed from read-only to mutating" in result.error
    assert loop.stats.tool_calls_rejected == 1
    assert list((config_dir / "memory").rglob("*.md")) == []
    assert not any(
        isinstance(event, HookStartEvent) and event.hook_name == "must-not-run"
        for event in events
    )


@pytest.mark.asyncio
async def test_work_strategy_is_a_batch_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    events: list[str] = []

    async def fake_process(
        tc: ResolvedToolCall, scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        assert scheduled_read_only is tc.tool_class.read_only
        events.append(f"start:{tc.tool_name}")
        await asyncio.sleep(0.01)
        events.append(f"end:{tc.tool_name}")
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)
    calls = [
        _call("read1", Read),
        _call("work_strategy", WorkStrategy),
        _call("read2", Grep),
    ]

    collected = [event async for event in loop._run_tools_concurrently(calls)]

    assert len(collected) == 3
    strategy_end = events.index("end:work_strategy")
    assert strategy_end < events.index("start:read1")
    assert strategy_end < events.index("start:read2")


@pytest.mark.asyncio
async def test_closing_outer_tool_stream_cancels_active_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    started: set[str] = set()
    cancelled = asyncio.Event()
    read2_started = asyncio.Event()

    async def fake_process(
        tc: ResolvedToolCall, scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        assert scheduled_read_only is tc.tool_class.read_only
        started.add(tc.tool_name)
        if tc.tool_name == "read1":
            await read2_started.wait()
            yield ToolResultEvent(
                tool_name=tc.tool_name,
                tool_class=tc.tool_class,
                tool_call_id=tc.call_id,
            )
            return
        if tc.tool_name == "read2":
            read2_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)
    calls = [_call("read1", Read), _call("read2", Grep), _call("write1", WriteFile)]
    stream = loop._run_tools_concurrently(calls)

    result = await anext(stream)
    assert isinstance(result, ToolResultEvent)
    assert result.tool_name == "read1"
    await stream.aclose()

    assert cancelled.is_set()
    assert started == {"read1", "read2"}


@pytest.mark.asyncio
async def test_closing_outer_stream_closes_work_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    started: set[str] = set()
    finalized = asyncio.Event()

    async def fake_process(
        tc: ResolvedToolCall, scheduled_read_only: bool
    ) -> AsyncGenerator[ToolResultEvent]:
        assert scheduled_read_only is tc.tool_class.read_only
        started.add(tc.tool_name)
        if tc.tool_name == "work_strategy":
            try:
                yield ToolResultEvent(
                    tool_name=tc.tool_name,
                    tool_class=tc.tool_class,
                    tool_call_id=tc.call_id,
                )
                await asyncio.Event().wait()
            finally:
                finalized.set()
            return
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    monkeypatch.setattr(loop, "_process_one_tool_call", fake_process)
    calls = [_call("work_strategy", WorkStrategy), _call("read1", Read)]
    stream = loop._run_tools_concurrently(calls)

    result = await anext(stream)
    assert isinstance(result, ToolResultEvent)
    assert result.tool_name == "work_strategy"
    await stream.aclose()

    assert finalized.is_set()
    assert started == {"work_strategy"}


@pytest.mark.asyncio
async def test_task_launch_is_recorded_before_terminal_result_stream_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        effort_mode="le-chaton", enabled_tools=["task", "work_strategy"]
    )
    loop = build_test_agent_loop(config=config)
    registry = BackgroundRegistry()

    async def wake_host() -> None:
        return None

    async def hold_agent(*_args: object, **_kwargs: object) -> object:
        await asyncio.Event().wait()

    registry.attach_completion_callback(wake_host)
    loop.background_registry = registry
    monkeypatch.setattr(Task, "_run_in_process_collect", hold_agent)
    loop._begin_orchestration_turn("Investigate one independent task lane.")
    loop._declare_orchestration_strategy(
        OrchestrationDecision.model_validate({
            "route": "task",
            "risk": "medium",
            "reason": "independent_lanes",
            "lanes": [
                {
                    "id": "lane-1",
                    "objective": "Inspect the independent area",
                    "owner": "agent",
                    "profile": "explore",
                }
            ],
        })
    )
    args = TaskArgs(
        task="[lane:lane-1] Inspect the independent area",
        agent="explore",
        async_run=True,
    )
    call = ResolvedToolCall(
        tool_name="task", tool_class=Task, validated_args=args, call_id="task-launch"
    )
    stream = loop._process_one_tool_call(call, False)
    events: list[object] = []

    try:
        while True:
            event = await anext(stream)
            events.append(event)
            if isinstance(event, ToolResultEvent):
                break
        await stream.aclose()

        terminals = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(terminals) == 1
        result = terminals[0].result
        assert isinstance(result, TaskResult)
        assert result.task_id is not None
        assert loop._orchestration._reserved_lanes_by_call == {}
        assert loop._orchestration._task_lanes_by_id[result.task_id][1] == {"lane-1"}

        retry = call.model_copy(update={"call_id": "task-relaunch"})
        denial = loop._orchestration_before_tool(retry)
        assert "already launched" in (denial or "")

        loop._observe_task_completion(result.task_id, succeeded=True)
        assert result.task_id not in loop._orchestration._task_lanes_by_id
        assert loop.orchestration_summary.completed_delegations == 1
        assert loop.orchestration_summary.pending_delegations == 0
    finally:
        await registry.shutdown()


def _workflow_launch_case(launch_id: str) -> tuple[AgentLoop, ResolvedToolCall]:
    config = build_test_vibe_config(
        effort_mode="le-chaton", enabled_tools=["launch_workflow", "work_strategy"]
    )
    loop = build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.AUTO_APPROVE
    )
    loop.launch_workflow_callback = lambda _script, _name, _expected_lanes: launch_id
    loop._begin_orchestration_turn("Inspect one independent workflow lane.")
    receipt = loop._declare_orchestration_strategy(
        OrchestrationDecision.model_validate({
            "route": "workflow",
            "risk": "medium",
            "reason": "independent_lanes",
            "lanes": [
                {
                    "id": "lane-1",
                    "objective": "Inspect the independent area",
                    "owner": "agent",
                    "profile": "explore",
                },
                {
                    "id": "lane-2",
                    "objective": "Inspect the adjacent area",
                    "owner": "agent",
                    "profile": "explore",
                },
            ],
        })
    )
    assert receipt.accepted is True
    script = (
        "async def main():\n"
        "    first = await agent('Inspect the area', label='lane-1')\n"
        "    second = await agent('Inspect the adjacent area', label='lane-2')\n"
        "    return [first, second]\n"
    )
    args = LaunchWorkflowArgs(script=script, name="race-check")
    return loop, ResolvedToolCall(
        tool_name="launch_workflow", tool_class=LaunchWorkflow, validated_args=args
    )


@pytest.mark.asyncio
async def test_workflow_finalization_records_launch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, call = _workflow_launch_case("wf-once")
    call = call.model_copy(update={"call_id": "workflow-once"})
    record_result = Mock(wraps=loop._record_orchestration_tool_result)
    monkeypatch.setattr(loop, "_record_orchestration_tool_result", record_result)

    events = [event async for event in loop._process_one_tool_call(call, False)]

    terminals = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(terminals) == 1
    assert record_result.call_count == 1
    assert loop._orchestration._workflow_lanes_by_id["wf-once"][1] == {
        "lane-1",
        "lane-2",
    }


@pytest.mark.asyncio
async def test_workflow_launch_survives_after_hook_failure_without_duplicate_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, call = _workflow_launch_case("wf-race")
    call = call.model_copy(update={"call_id": "workflow-launch"})
    loop.messages.append(
        LLMMessage(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(
                    id=call.call_id,
                    index=0,
                    function=FunctionCall(
                        name=call.tool_name,
                        arguments=call.validated_args.model_dump_json(),
                    ),
                )
            ],
        )
    )

    async def fail_after_tool(*_args: object, **_kwargs: object):
        if False:
            yield
        raise RuntimeError("after hook failed")

    monkeypatch.setattr(loop, "_run_after_tool_hooks", fail_after_tool)
    finished = Mock()
    monkeypatch.setattr(loop.telemetry_client, "send_tool_call_finished", finished)
    record_result = Mock(wraps=loop._record_orchestration_tool_result)
    monkeypatch.setattr(loop, "_record_orchestration_tool_result", record_result)
    events: list[object] = []

    with pytest.raises(RuntimeError, match="after hook failed"):
        async for event in loop._process_one_tool_call(call, False):
            events.append(event)

    terminals = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(terminals) == 1
    result = terminals[0].result
    assert isinstance(result, LaunchWorkflowResult)
    assert result.run_id == "wf-race"
    assert record_result.call_count == 1
    responses = [
        message
        for message in loop.messages
        if message.role is Role.TOOL and message.tool_call_id == call.call_id
    ]
    assert len(responses) == 1
    assert loop.stats.tool_calls_succeeded == 1
    assert loop.stats.tool_calls_failed == 0
    finished.assert_called_once()
    loop._fill_missing_tool_responses()
    assert (
        len([
            message
            for message in loop.messages
            if message.role is Role.TOOL and message.tool_call_id == call.call_id
        ])
        == 1
    )
    assert loop._orchestration._reserved_lanes_by_call == {}
    assert loop._orchestration._workflow_lanes_by_id["wf-race"][1] == {
        "lane-1",
        "lane-2",
    }

    retry = call.model_copy(update={"call_id": "workflow-relaunch"})
    denial = loop._orchestration_before_tool(retry)
    assert "already launched" in (denial or "")

    expected = loop._orchestration._workflow_expectations_by_id["wf-race"]
    labels = tuple(lane.label for lane in expected)
    loop.observe_workflow_completion(
        "wf-race",
        succeeded=True,
        attestation=WorkflowLaneAttestation(
            expected=expected,
            attempted_labels=labels,
            started_labels=labels,
            successful_labels=labels,
        ),
    )
    assert "wf-race" not in loop._orchestration._workflow_lanes_by_id
    assert loop.orchestration_summary.completed_delegations == 2
    assert loop.orchestration_summary.pending_delegations == 0


@pytest.mark.asyncio
async def test_workflow_result_is_committed_before_terminal_publication(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    loop, call = _workflow_launch_case("wf-immediate-close")
    call = call.model_copy(update={"call_id": "workflow-immediate-close"})

    async def fail_after_tool(*_args: object, **_kwargs: object):
        yield HookStartEvent(
            hook_name="audit", scope=HookType.AFTER_TOOL, tool_call_id=call.call_id
        )
        raise RuntimeError("after hook failed before close")

    monkeypatch.setattr(loop, "_run_after_tool_hooks", fail_after_tool)
    finished = Mock()
    monkeypatch.setattr(loop.telemetry_client, "send_tool_call_finished", finished)
    stream = loop._process_one_tool_call(call, False)

    hook_event = await anext(stream)
    event = await anext(stream)

    assert isinstance(hook_event, HookStartEvent)
    assert isinstance(event, ToolResultEvent)
    assert any(
        message.role is Role.TOOL and message.tool_call_id == call.call_id
        for message in loop.messages
    )
    assert loop.stats.tool_calls_succeeded == 1
    assert loop._orchestration._workflow_lanes_by_id["wf-immediate-close"][1] == {
        "lane-1",
        "lane-2",
    }
    finished.assert_called_once()
    assert "after hook failed before close" in caplog.text

    await stream.aclose()
