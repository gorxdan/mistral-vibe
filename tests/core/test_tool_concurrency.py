from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from pydantic import BaseModel
import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.llm.models import ResolvedToolCall
from vibe.core.tools.builtins.grep import Grep
from vibe.core.tools.builtins.manage_memory import ManageMemory, ManageMemoryArgs
from vibe.core.tools.builtins.read import Read
from vibe.core.tools.builtins.work_strategy import WorkStrategy
from vibe.core.tools.builtins.write_file import WriteFile
from vibe.core.types import ToolResultEvent


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
async def test_writers_run_sequentially_readers_run_parallel() -> None:
    loop = build_test_agent_loop()
    events: list[str] = []  # interleaving trace: "start:x" / "end:x"

    async def fake_process(tc: ResolvedToolCall) -> AsyncGenerator[ToolResultEvent]:
        events.append(f"start:{tc.tool_name}")
        await asyncio.sleep(0.02)
        events.append(f"end:{tc.tool_name}")
        yield ToolResultEvent(
            tool_name=tc.tool_name, tool_class=tc.tool_class, tool_call_id=tc.call_id
        )

    loop._process_one_tool_call = fake_process  # type: ignore[method-assign]

    calls = [
        _call("read1", Read),
        _call("write1", WriteFile),
        _call("read2", Grep),
        _call("write2", WriteFile),
    ]
    collected = [e async for e in loop._run_tools_concurrently(calls)]
    assert len(collected) == 4

    # Writers must not overlap: write1 fully finishes before write2 starts.
    assert events.index("end:write1") < events.index("start:write2")

    # Readers run concurrently with the writer chain: both readers start before
    # the first writer finishes (i.e. they are not serialized behind writers).
    first_writer_end = events.index("end:write1")
    assert events.index("start:read1") < first_writer_end
    assert events.index("start:read2") < first_writer_end


@pytest.mark.asyncio
async def test_work_strategy_is_a_batch_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = build_test_agent_loop()
    events: list[str] = []

    async def fake_process(tc: ResolvedToolCall) -> AsyncGenerator[ToolResultEvent]:
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
