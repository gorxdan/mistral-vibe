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

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


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
