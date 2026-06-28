"""Reasoning-effort downgrade: send the top tier (xhigh for `max`) and, when a
model rejects it with a 400 listing supported values, retry with the nearest
supported effort — "xhigh if available, else the model's highest level".
"""

from __future__ import annotations

import httpx
import orjson
import pytest
import respx

from tests.constants import OPENAI_BASE_URL, OPENAI_RESPONSES_PATH
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.llm.backend.generic import (
    GenericBackend,
    _nearest_supported_effort,
    _patch_reasoning_effort,
)
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import LLMMessage, Role

_XHIGH_REJECTED = (
    "Unsupported value: 'xhigh' is not supported with the 'gpt-4o' model. "
    "Supported values are: 'low', 'medium', 'high'."
)
_NONE_REJECTED = (
    "Unsupported value: 'none' is not supported with the 'gpt-5.3-codex-spark' "
    "model. Supported values are: 'low', 'medium', 'high', and 'xhigh'."
)


class TestNearestSupportedEffort:
    def test_downgrades_xhigh_to_highest_available(self):
        assert _nearest_supported_effort(_XHIGH_REJECTED) == "high"

    def test_lifts_unsupported_none_to_floor(self):
        # none is below the supported floor -> nearest is the lowest offered.
        assert _nearest_supported_effort(_NONE_REJECTED) == "low"

    def test_non_effort_400_is_ignored(self):
        err = "Unsupported value: 'banana'. Supported values are: 'apple', 'pear'."
        assert _nearest_supported_effort(err) is None

    def test_no_supported_list_is_ignored(self):
        assert _nearest_supported_effort("400 Bad Request") is None


class TestPatchReasoningEffort:
    def test_patches_responses_shape(self):
        body = orjson.dumps({"reasoning": {"effort": "xhigh"}})
        out = _patch_reasoning_effort(body, "high")
        assert out is not None
        assert orjson.loads(out)["reasoning"]["effort"] == "high"

    def test_patches_chat_completions_shape(self):
        body = orjson.dumps({"reasoning_effort": "xhigh"})
        out = _patch_reasoning_effort(body, "high")
        assert out is not None
        assert orjson.loads(out)["reasoning_effort"] == "high"

    def test_noop_when_already_target(self):
        assert (
            _patch_reasoning_effort(orjson.dumps({"reasoning_effort": "high"}), "high")
            is None
        )

    def test_noop_when_no_effort_field(self):
        assert _patch_reasoning_effort(orjson.dumps({"model": "x"}), "high") is None


_OK_STREAM = (
    b'data: {"type":"response.output_text.delta","output_index":0,'
    b'"content_index":0,"delta":"ok"}\n\n'
    b'data: {"type":"response.completed","response":{"id":"r","output":'
    b'[{"type":"message","content":[{"type":"output_text","text":"ok"}],'
    b'"role":"assistant"}],"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.mark.asyncio
async def test_streaming_retries_with_downgraded_effort():
    provider = ProviderConfig(
        name="openai", api_base=f"{OPENAI_BASE_URL}/v1", api_style="openai-responses"
    )
    model = ModelConfig(
        name="gpt-4o", provider="openai", alias="gpt-4o", thinking="max"
    )
    with respx.mock(base_url=OPENAI_BASE_URL) as mock_api:
        route = mock_api.post(OPENAI_RESPONSES_PATH).mock(
            side_effect=[
                httpx.Response(
                    400, content=orjson.dumps({"error": {"message": _XHIGH_REJECTED}})
                ),
                httpx.Response(
                    200,
                    stream=httpx.ByteStream(_OK_STREAM),
                    headers={"Content-Type": "text/event-stream"},
                ),
            ]
        )
        backend = GenericBackend(provider=provider)
        out = [
            chunk
            async for chunk in backend.complete_streaming(
                CompletionRequest(
                    model=model,
                    messages=[LLMMessage(role=Role.USER, content="hi")],
                    temperature=0.2,
                    tools=None,
                    max_tokens=None,
                    tool_choice=None,
                    extra_headers=None,
                )
            )
        ]

    # First request sent the top tier; it was rejected, so we retried.
    assert len(route.calls) == 2
    first = orjson.loads(route.calls[0].request.content)
    second = orjson.loads(route.calls[1].request.content)
    assert first["reasoning"]["effort"] == "xhigh"
    assert second["reasoning"]["effort"] == "high"  # nearest supported to xhigh
    assert any(c.message.content == "ok" for c in out)
