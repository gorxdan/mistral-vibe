from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import CompactionFailedError
from vibe.core.config import MemoryConfig, ModelConfig
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import (
    AssistantEvent,
    CompactEndEvent,
    CompactStartEvent,
    FunctionCall,
    LLMMessage,
    Role,
    ToolCall,
    UserMessageEvent,
)


def _get_auto_compact_properties(
    telemetry_events: list[dict[str, object]],
) -> dict[str, object]:
    auto_compact = [
        event
        for event in telemetry_events
        if event.get("event_name") == "vibe.auto_compact_triggered"
    ]
    assert len(auto_compact) == 1
    return cast(dict[str, object], auto_compact[0]["properties"])


def _get_compaction_failed_properties(
    telemetry_events: list[dict[str, object]],
) -> dict[str, object]:
    failed = [
        event
        for event in telemetry_events
        if event.get("event_name") == "vibe.compaction_failed"
    ]
    assert len(failed) == 1
    return cast(dict[str, object], failed[0]["properties"])


@pytest.mark.asyncio
async def test_auto_compact_emits_correct_events(telemetry_events: list[dict]) -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="<summary>")],
        [mock_llm_chunk(content="<final>")],
    ])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=1))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.stats.context_tokens = 2
    old_session_id = agent.session_id

    events = [ev async for ev in agent.act("Hello")]

    assert len(events) == 4
    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[1], CompactStartEvent)
    assert isinstance(events[2], CompactEndEvent)
    assert isinstance(events[3], AssistantEvent)
    start: CompactStartEvent = events[1]
    end: CompactEndEvent = events[2]
    final: AssistantEvent = events[3]
    assert start.current_context_tokens == 2
    assert start.threshold == 1
    assert isinstance(end, CompactEndEvent)
    assert final.content == "<final>"

    properties = _get_auto_compact_properties(telemetry_events)
    assert properties["nb_context_tokens_before"] == 2
    assert properties["auto_compact_threshold"] == 1
    assert properties["status"] == "success"
    assert properties["session_id"] == old_session_id
    assert properties["parent_session_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side_effect", "expected_exception", "match", "expected_status"),
    [
        pytest.param(
            RuntimeError("boom"), RuntimeError, "boom", "failure", id="failure"
        ),
        pytest.param(
            asyncio.CancelledError(),
            asyncio.CancelledError,
            None,
            "cancelled",
            id="cancelled",
        ),
    ],
)
async def test_auto_compact_emits_terminal_telemetry(
    side_effect: BaseException,
    expected_exception: type[BaseException],
    match: str | None,
    expected_status: str,
    telemetry_events: list[dict],
) -> None:
    backend = FakeBackend([[mock_llm_chunk(content="<final>")]])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=1))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.stats.context_tokens = 2
    old_session_id = agent.session_id

    events = []
    with patch.object(agent, "compact", AsyncMock(side_effect=side_effect)):
        if match is None:
            with pytest.raises(expected_exception):
                async for event in agent.act("Hello"):
                    events.append(event)
        else:
            with pytest.raises(expected_exception, match=match):
                async for event in agent.act("Hello"):
                    events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], UserMessageEvent)
    assert isinstance(events[1], CompactStartEvent)

    properties = _get_auto_compact_properties(telemetry_events)
    assert properties["nb_context_tokens_before"] == 2
    assert properties["auto_compact_threshold"] == 1
    assert properties["status"] == expected_status
    assert properties["session_id"] == old_session_id
    assert properties["parent_session_id"] is None


@pytest.mark.asyncio
async def test_auto_compact_observer_sees_user_msg_not_summary() -> None:
    """Observer sees the original user message and final response.

    Compact internals (summary request, LLM summary) are invisible
    to the observer because they happen inside silent() / reset().
    """
    observed: list[tuple[Role, str | None]] = []

    def observer(msg: LLMMessage) -> None:
        observed.append((msg.role, msg.content))

    backend = FakeBackend([
        [mock_llm_chunk(content="<summary>")],
        [mock_llm_chunk(content="<final>")],
    ])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=1))
    agent = build_test_agent_loop(
        config=cfg, message_observer=observer, backend=backend
    )
    agent.stats.context_tokens = 2

    [_ async for _ in agent.act("Hello")]

    roles = [r for r, _ in observed]
    assert roles == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert observed[1][1] == "Hello"
    assert observed[2][1] == "<final>"


@pytest.mark.asyncio
async def test_auto_compact_observer_does_not_see_summary_request() -> None:
    """The compact summary request and LLM response must not leak to observer."""
    observed: list[tuple[Role, str | None]] = []

    def observer(msg: LLMMessage) -> None:
        observed.append((msg.role, msg.content))

    backend = FakeBackend([
        [mock_llm_chunk(content="<summary>")],
        [mock_llm_chunk(content="<final>")],
    ])
    # Disable the always-on config-reference section: it lists the `/compact`
    # slash command in the system prompt, which the observer legitimately sees
    # and would trip the "no compaction content leaked" check below.
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=1),
        include_config_reference=False,
    )
    agent = build_test_agent_loop(
        config=cfg, message_observer=observer, backend=backend
    )
    agent.stats.context_tokens = 2

    [_ async for _ in agent.act("Hello")]

    contents = [c for _, c in observed]
    assert "<summary>" not in contents
    assert all("compact" not in (c or "").lower() for c in contents)


@pytest.mark.asyncio
async def test_compact_replaces_messages_with_context() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="<summary>")],
        [mock_llm_chunk(content="<final>")],
    ])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=1))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.stats.context_tokens = 2

    [_ async for _ in agent.act("Hello")]

    # After compact + final response: system, compaction context, final.
    assert agent.messages[0].role == Role.SYSTEM
    assert agent.messages[-1].role == Role.ASSISTANT
    assert agent.messages[-1].content == "<final>"


class _ModelTrackingBackend(FakeBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requested_models: list[ModelConfig] = []
        self.requests: list[CompletionRequest] = []

    async def complete(self, request, *, response_headers_sink=None):
        self.requested_models.append(request.model)
        self.requests.append(request)
        return await super().complete(
            request, response_headers_sink=response_headers_sink
        )


@pytest.mark.asyncio
async def test_compact_uses_compaction_model() -> None:
    """When compaction_model is set, compact() uses it instead of active_model."""
    compaction = ModelConfig(
        name="compaction-model",
        provider="mistral",
        alias="compaction",
        auto_compact_threshold=1,
    )
    backend = _ModelTrackingBackend([
        [mock_llm_chunk(content="<summary>")],
        [mock_llm_chunk(content="<final>")],
    ])
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=1), compaction_model=compaction
    )
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.stats.context_tokens = 2

    [_ async for _ in agent.act("Hello")]

    assert backend.requested_models[0].name == "compaction-model"
    assert backend.requested_models[1].name != "compaction-model"
    compaction_request = backend.requests[0]
    assert compaction_request.tools is None
    assert compaction_request.tool_choice is None
    assert "Summarize a coding-agent transcript" in (
        compaction_request.messages[0].content or ""
    )
    assert "super useful programming assistant" not in (
        compaction_request.messages[0].content or ""
    )


@pytest.mark.asyncio
async def test_compact_omits_task_schema_and_late_memory() -> None:
    backend = _ModelTrackingBackend([[mock_llm_chunk(content="<summary>")]])
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        memory=MemoryConfig(inject_mode="late"),
    )
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(LLMMessage(role=Role.USER, content="Hello"))
    agent._response_format = {"type": "json_schema", "json_schema": {"name": "task"}}
    agent._late_memory_section = "late memory must not enter the summary request"

    await agent.compact()

    request = backend.requests[0]
    assert request.response_format is None
    assert all(
        "late memory must not enter" not in (message.content or "")
        for message in request.messages
    )


@pytest.mark.asyncio
async def test_compact_uses_active_model_when_no_compaction_model() -> None:
    """Without compaction_model, compact() falls back to the active model."""
    backend = _ModelTrackingBackend([
        [mock_llm_chunk(content="<summary>")],
        [mock_llm_chunk(content="<final>")],
    ])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=1))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.stats.context_tokens = 2

    [_ async for _ in agent.act("Hello")]

    active = cfg.get_active_model()
    assert backend.requested_models[0].name == active.name
    assert backend.requested_models[1].name == active.name


@pytest.mark.asyncio
async def test_compact_appends_extra_instructions_to_prompt() -> None:
    backend = FakeBackend([[mock_llm_chunk(content="<summary>")]])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(LLMMessage(role=Role.USER, content="Hello"))
    agent.stats.context_tokens = 100

    await agent.compact(extra_instructions="focus on auth")

    compaction_prompt = backend.requests_messages[0][-1].content
    assert compaction_prompt is not None
    assert "## Additional Instructions" in compaction_prompt
    assert "focus on auth" in compaction_prompt


@pytest.mark.asyncio
async def test_compact_uses_configured_compaction_prompt(
    mock_prompts_dirs: tuple[Path, Path],
) -> None:
    project_prompts, _ = mock_prompts_dirs
    (project_prompts / "theorem_compact.md").write_text("Summarize theorem progress")

    backend = FakeBackend([[mock_llm_chunk(content="<summary>")]])
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        compaction_prompt_id="theorem_compact",
    )
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(LLMMessage(role=Role.USER, content="Hello"))
    agent.stats.context_tokens = 100

    await agent.compact()

    compaction_prompt = backend.requests_messages[0][-1].content
    assert compaction_prompt == "Summarize theorem progress"


@pytest.mark.asyncio
async def test_compact_without_extra_instructions_has_no_additional_section() -> None:
    backend = FakeBackend([[mock_llm_chunk(content="<summary>")]])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(LLMMessage(role=Role.USER, content="Hello"))
    agent.stats.context_tokens = 100

    await agent.compact()

    compaction_prompt = backend.requests_messages[0][-1].content
    assert compaction_prompt is not None
    assert "## Additional Instructions" not in compaction_prompt


@pytest.mark.asyncio
async def test_compact_raises_on_tool_call_when_flag_enabled(
    telemetry_events: list[dict],
) -> None:
    """With the flag on, a compaction that returns a tool call raises."""
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        index=0,
                        function=FunctionCall(name="bash", arguments="{}"),
                    )
                ],
            )
        ]
    ])
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=True,
    )
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(LLMMessage(role=Role.USER, content="Hello"))
    agent.stats.context_tokens = 100

    with pytest.raises(CompactionFailedError) as exc_info:
        await agent.compact()
    assert exc_info.value.reason == "tool_call"
    assert _get_compaction_failed_properties(telemetry_events)["reason"] == "tool_call"


@pytest.mark.asyncio
async def test_compact_raises_on_empty_summary_when_flag_enabled(
    telemetry_events: list[dict],
) -> None:
    """With the flag on, a compaction with empty content raises."""
    backend = FakeBackend([[mock_llm_chunk(content="   ")]])
    cfg = build_test_vibe_config(
        models=make_test_models(auto_compact_threshold=999),
        raise_on_compaction_failure=True,
    )
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(LLMMessage(role=Role.USER, content="Hello"))
    agent.stats.context_tokens = 100

    with pytest.raises(CompactionFailedError) as exc_info:
        await agent.compact()
    assert exc_info.value.reason == "empty_summary"
    assert (
        _get_compaction_failed_properties(telemetry_events)["reason"] == "empty_summary"
    )


@pytest.mark.asyncio
async def test_compact_falls_back_when_flag_disabled() -> None:
    """With the flag off (default), empty content falls back to the extractive
    structural trace rather than failing.
    """
    backend = FakeBackend([[mock_llm_chunk(content="")]])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(LLMMessage(role=Role.USER, content="Hello"))
    agent.stats.context_tokens = 100

    summary = await agent.compact()
    assert "Structural trace of prior turns" in summary


@pytest.mark.asyncio
async def test_compact_falls_back_on_llm_error() -> None:
    """When the compaction LLM call raises, the extractive fallback keeps the
    session alive instead of propagating the failure.
    """

    class _RaisingBackend(FakeBackend):
        async def complete(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("compaction model unavailable")

    backend = _RaisingBackend([[mock_llm_chunk(content="unused")]])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    agent.messages.append(
        LLMMessage(
            role=Role.ASSISTANT,
            content="I will read the file.",
            tool_calls=[
                ToolCall(
                    id="c1",
                    index=0,
                    function=FunctionCall(name="read", arguments='{"file_path":"/x"}'),
                )
            ],
        )
    )
    agent.messages.append(
        LLMMessage(role=Role.TOOL, content="file contents here", tool_call_id="c1")
    )
    agent.stats.context_tokens = 100

    summary = await agent.compact()
    assert "Structural trace" in summary
    assert "I will read the file." in summary
    assert "read" in summary


@pytest.mark.asyncio
async def test_compact_message_shape_preserves_prior_user_messages() -> None:
    from vibe.core.compaction import parse_previous_user_messages
    from vibe.core.prompts import UtilityPrompt

    summary_prefix = UtilityPrompt.COMPACT_SUMMARY_PREFIX.read()
    backend = FakeBackend([[mock_llm_chunk(content="fresh summary body")]])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    agent = build_test_agent_loop(config=cfg, backend=backend)
    system_message_before = agent.messages[0]

    agent.messages.append(LLMMessage(role=Role.USER, content="first real ask"))
    agent.messages.append(
        LLMMessage(role=Role.USER, content="middleware ping", injected=True)
    )
    agent.messages.append(LLMMessage(role=Role.ASSISTANT, content="ack"))
    agent.messages.append(
        LLMMessage(
            role=Role.USER,
            content=f"{summary_prefix}\nprior summary blob",
            injected=True,
        )
    )
    agent.messages.append(LLMMessage(role=Role.USER, content="follow-up ask"))
    agent.stats.context_tokens = 100

    await agent.compact()

    final = list(agent.messages)
    assert len(final) == 2  # [system, compaction_context]
    assert final[0] is system_message_before
    assert final[1].role == Role.USER
    assert final[1].injected is True
    assert parse_previous_user_messages(final[1].content or "") == [
        "first real ask",
        "follow-up ask",
    ]
    assert "Here are some of the most recent previous user messages" in (
        final[1].content or ""
    )
    assert "<compaction_summary>" in (final[1].content or "")
    assert "fresh summary body" in (final[1].content or "")
    # Injected and prior-summary user messages must be filtered out.
    assert all("middleware ping" not in (m.content or "") for m in final)
    assert sum("prior summary blob" in (m.content or "") for m in final) == 0


@pytest.mark.asyncio
async def test_compact_preserves_user_messages_across_repeated_compactions() -> None:
    from vibe.core.compaction import parse_previous_user_messages

    backend = FakeBackend([
        [mock_llm_chunk(content="summary one")],
        [mock_llm_chunk(content="summary two")],
    ])
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    agent = build_test_agent_loop(config=cfg, backend=backend)

    agent.messages.append(LLMMessage(role=Role.USER, content="first ask"))
    agent.stats.context_tokens = 100
    await agent.compact()

    agent.messages.append(LLMMessage(role=Role.USER, content="second ask"))
    agent.stats.context_tokens = 100
    await agent.compact()

    final = list(agent.messages)
    assert len(final) == 2
    assert parse_previous_user_messages(final[1].content or "") == [
        "first ask",
        "second ask",
    ]
