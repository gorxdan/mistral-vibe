from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from pathlib import Path

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.llm.types import CompletionRequest
from vibe.core.middleware import LoopDetectionMiddleware
from vibe.core.types import (
    AssistantEvent,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    Role,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)


class RecordingBackend(FakeBackend):
    def __init__(self, chunks: list[list[LLMChunk]]) -> None:
        super().__init__(chunks)
        self.requests: list[CompletionRequest] = []

    async def complete(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> LLMChunk:
        self.requests.append(request)
        return await super().complete(
            request, response_headers_sink=response_headers_sink
        )

    async def complete_streaming(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        self.requests.append(request)
        async for chunk in super().complete_streaming(
            request, response_headers_sink=response_headers_sink
        ):
            yield chunk


def _read_call(path: Path, index: int) -> ToolCall:
    return ToolCall(
        id=f"read-{index}",
        index=0,
        function=FunctionCall(
            name="read", arguments=json.dumps({"file_path": str(path)})
        ),
    )


def _bash_call(index: int) -> ToolCall:
    return ToolCall(
        id=f"bash-{index}",
        index=0,
        function=FunctionCall(
            name="bash", arguments=json.dumps({"command": f"printf evidence-{index}"})
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_repeated_tool_calls_end_with_tool_free_synthesis(
    tmp_path: Path, enable_streaming: bool
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("enough evidence", encoding="utf-8")
    backend = RecordingBackend(
        [
            [mock_llm_chunk(content="", tool_calls=[_read_call(evidence, index)])]
            for index in range(6)
        ]
        + [[mock_llm_chunk(content="Final synthesis from collected evidence.")]]
    )
    config = build_test_vibe_config(enabled_tools=["read"])
    loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=enable_streaming
    )

    events = [event async for event in loop.act("Review the evidence deeply.")]

    assert len(backend.requests) == 7
    assert all(request.tools for request in backend.requests[:-1])
    assert backend.requests[-1].tools is None
    assert backend.requests[-1].tool_choice is None
    assert len([event for event in events if isinstance(event, ToolResultEvent)]) == 6
    final_events = [event for event in events if isinstance(event, AssistantEvent)]
    assert final_events[-1].content == "Final synthesis from collected evidence."
    assert not final_events[-1].stopped_by_middleware
    assert not loop._force_toolless_response


@pytest.mark.asyncio
async def test_varying_bash_calls_remain_available_after_advisory_warning() -> None:
    backend = RecordingBackend(
        [
            [mock_llm_chunk(content="", tool_calls=[_bash_call(index)])]
            for index in range(8)
        ]
        + [[mock_llm_chunk(content="Complete review after distinct checks.")]]
    )
    config = build_test_vibe_config(
        enabled_tools=["bash"], tools={"bash": {"permission": "always"}}
    )
    loop = build_test_agent_loop(config=config, backend=backend)

    events = [event async for event in loop.act("Review several distinct targets.")]

    assert len(backend.requests) == 9
    assert all(request.tools for request in backend.requests)
    assert len([event for event in events if isinstance(event, ToolResultEvent)]) == 8
    final = [event for event in events if isinstance(event, AssistantEvent)][-1]
    assert final.content == "Complete review after distinct checks."


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_tool_free_recovery_does_not_execute_disallowed_model_tool_call(
    tmp_path: Path, enable_streaming: bool
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("enough evidence", encoding="utf-8")
    backend = RecordingBackend([
        [mock_llm_chunk(content="", tool_calls=[_read_call(evidence, 0)])],
        [mock_llm_chunk(content="", tool_calls=[_read_call(evidence, 1)])],
        [mock_llm_chunk(content="", tool_calls=[_read_call(evidence, 2)])],
    ])
    config = build_test_vibe_config(enabled_tools=["read"])
    loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=enable_streaming
    )
    loop.middleware_pipeline.clear()
    loop.middleware_pipeline.add(LoopDetectionMiddleware(threshold=1))

    events = [event async for event in loop.act("Review the evidence deeply.")]

    assert backend.requests[-1].tools is None
    assert {
        event.tool_call_id for event in events if isinstance(event, ToolCallEvent)
    } == {"read-0", "read-1"}
    assert len([event for event in events if isinstance(event, ToolResultEvent)]) == 2
    final = [event for event in events if isinstance(event, AssistantEvent)][-1]
    assert "Tool-free recovery did not produce a final response" in final.content
    assert loop.messages[-1].tool_calls is None


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
@pytest.mark.parametrize("reasoning_content", [None, "internal reasoning only"])
async def test_tool_free_recovery_falls_back_when_response_has_no_visible_content(
    tmp_path: Path, enable_streaming: bool, reasoning_content: str | None
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("enough evidence", encoding="utf-8")
    backend = RecordingBackend([
        [mock_llm_chunk(content="", tool_calls=[_read_call(evidence, 0)])],
        [mock_llm_chunk(content="", tool_calls=[_read_call(evidence, 1)])],
        [mock_llm_chunk(content="", reasoning_content=reasoning_content)],
    ])
    config = build_test_vibe_config(enabled_tools=["read"])
    loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=enable_streaming
    )
    loop.middleware_pipeline.clear()
    loop.middleware_pipeline.add(LoopDetectionMiddleware(threshold=1))

    events = [event async for event in loop.act("Review the evidence deeply.")]

    assert backend.requests[-1].tools is None
    final = [event for event in events if isinstance(event, AssistantEvent)][-1]
    assert "Tool-free recovery did not produce a final response" in final.content
    assert loop.messages[-1].content == final.content
    assert not loop._force_toolless_response


@pytest.mark.asyncio
async def test_tool_free_recovery_survives_failed_llm_attempt(monkeypatch) -> None:
    loop = build_test_agent_loop()
    loop._force_toolless_response = True

    async def fail_turn():
        raise RuntimeError("backend failed")
        yield AssistantEvent(content="unreachable")

    monkeypatch.setattr(loop, "_perform_llm_turn", fail_turn)

    with pytest.raises(RuntimeError, match="backend failed"):
        _ = [event async for event in loop._perform_llm_turn_and_reset_recovery()]

    assert loop._force_toolless_response

    async def complete_turn():
        loop.messages.append(LLMMessage(role=Role.ASSISTANT, content="recovered"))
        yield AssistantEvent(content="recovered")

    monkeypatch.setattr(loop, "_perform_llm_turn", complete_turn)
    _ = [event async for event in loop._perform_llm_turn_and_reset_recovery()]

    assert not loop._force_toolless_response
