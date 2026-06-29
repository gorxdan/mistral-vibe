from __future__ import annotations

import contextlib
from typing import Any

import pytest

import vibe.core.tracing as tracing


@contextlib.asynccontextmanager
async def _capture(attrs_sink: dict[str, Any]):
    @contextlib.asynccontextmanager
    async def fake_safe_span(name: str, attributes: dict[str, Any]):
        attrs_sink.update(attributes)
        yield object()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tracing, "_safe_span", fake_safe_span)
        yield


@pytest.mark.asyncio
async def test_chat_span_omits_temperature_when_none() -> None:
    attrs: dict[str, Any] = {}
    async with _capture(attrs):
        async with tracing.chat_span(
            model="kimi-k2.7-code", provider="kimi",
            temperature=None, max_tokens=16384, thinking="max",
        ):
            pass
    # None temperature is omitted on the wire, so the span must not claim one.
    assert "gen_ai.request.temperature" not in attrs
    assert attrs["gen_ai.request.max_tokens"] == 16384
    assert attrs["vibe.request.thinking"] == "max"


@pytest.mark.asyncio
async def test_chat_span_records_sampling_params() -> None:
    attrs: dict[str, Any] = {}
    async with _capture(attrs):
        async with tracing.chat_span(
            model="glm-5.2", provider="zai",
            temperature=1.0, max_tokens=None, thinking="max",
        ):
            pass
    assert attrs["gen_ai.request.temperature"] == 1.0
    assert "gen_ai.request.max_tokens" not in attrs  # None omitted
    assert attrs["vibe.request.thinking"] == "max"


class _FakeSpan:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))


def test_set_usage_emits_reasoning_tokens() -> None:
    from vibe.core.types import LLMUsage

    span = _FakeSpan()
    tracing.set_usage(
        span,  # type: ignore[arg-type]
        LLMUsage(
            prompt_tokens=100, completion_tokens=50,
            cached_tokens=20, reasoning_tokens=999,
        ),
    )
    assert span.attrs["gen_ai.usage.reasoning_tokens"] == 999


def test_otel_capture_content_defaults_off() -> None:
    # Content capture must be opt-in: it balloons trace files and records
    # user/source bytes, so the default has to stay False.
    from vibe.core.config import VibeConfig

    assert VibeConfig().otel_capture_content is False


def test_add_message_content_events_captures_and_clips() -> None:
    span = _FakeSpan()
    tracing.add_message_content_events(
        span,  # type: ignore[arg-type]
        user_text="hello",
        assistant_text="x" * 50,
        reasoning_text="think",
        tool_call_names=["read", "bash"],
        max_chars=10,
    )
    assert [n for n, _ in span.events] == [
        "gen_ai.user.message",
        "gen_ai.assistant.message",
    ]
    assert span.events[0][1]["content"] == "hello"
    asst = span.events[1][1]
    # 50-char assistant text middle-clipped near max_chars with an elision marker.
    assert "chars elided" in asst["content"]
    assert len(asst["content"]) < 50
    assert asst["reasoning"] == "think"
    assert asst["tool_calls"] == "read, bash"


def test_add_message_content_events_skips_empty_fields() -> None:
    # No content at all -> no events (an empty assistant turn isn't recorded).
    blank = _FakeSpan()
    tracing.add_message_content_events(blank)  # type: ignore[arg-type]
    assert blank.events == []

    # Tool-call-only turn -> a single assistant event carrying just the names.
    tools_only = _FakeSpan()
    tracing.add_message_content_events(
        tools_only,  # type: ignore[arg-type]
        tool_call_names=["read"],
    )
    assert [n for n, _ in tools_only.events] == ["gen_ai.assistant.message"]
    assert tools_only.events[0][1] == {"tool_calls": "read"}


def test_set_finish_reason_records_and_omits_none() -> None:
    span = _FakeSpan()
    tracing.set_finish_reason(span, "length")  # type: ignore[arg-type]
    assert span.attrs["gen_ai.response.finish_reasons"] == ("length",)

    blank = _FakeSpan()
    tracing.set_finish_reason(blank, None)  # type: ignore[arg-type]
    assert "gen_ai.response.finish_reasons" not in blank.attrs


def test_context_shaping_records_reasoning_preserved() -> None:
    span = _FakeSpan()
    tracing.set_context_shaping_result(
        span,  # type: ignore[arg-type]
        tokens_before=1000, tokens_after=500, reasoning_preserved=False,
    )
    assert span.attrs["vibe.context.reasoning_preserved"] is False


def test_wire_temperature_reflects_what_is_sent() -> None:
    from vibe.core.agent_loop import AgentLoop
    from vibe.core.config import ModelConfig, ProviderConfig

    responses = ProviderConfig(
        name="openai", api_base="x", api_key_env_var="",
        api_style="openai-responses",
    )
    chat = ProviderConfig(name="zai", api_base="x", api_key_env_var="")
    gpt55 = ModelConfig(name="gpt-5.5", provider="openai", alias="g", temperature=0.2)
    gpt4 = ModelConfig(name="gpt-4o", provider="openai", alias="g4", temperature=0.5)
    glm = ModelConfig(name="glm-5.2", provider="zai", alias="glm", temperature=1.0)

    # Reasoning model on the Responses API omits temperature -> not over-reported.
    assert AgentLoop._wire_temperature(gpt55, responses) is None
    # gpt-4 family does send it.
    assert AgentLoop._wire_temperature(gpt4, responses) == 0.5
    # Chat-completions provider sends the configured value.
    assert AgentLoop._wire_temperature(glm, chat) == 1.0
