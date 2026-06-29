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
