from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
import sys

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import MemoryConfig
from vibe.core.hooks.config import HookConfigResult
from vibe.core.hooks.models import HookConfig, HookStartEvent, HookType
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.tools.builtins.edit import Edit, EditArgs
from vibe.core.tools.builtins.grep import Grep
from vibe.core.tools.builtins.manage_memory import ManageMemory, ManageMemoryArgs
from vibe.core.tools.builtins.read import Read, ReadArgs
from vibe.core.tools.builtins.work_strategy import WorkStrategy
from vibe.core.tools.builtins.write_file import WriteFile
from vibe.core.types import ApprovalResponse, FunctionCall, ToolCall, ToolResultEvent
from vibe.core.utils.io import read_safe, write_safe


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
