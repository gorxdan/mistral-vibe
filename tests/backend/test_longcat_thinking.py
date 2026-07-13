from __future__ import annotations

import json

import pytest

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.adapter_port import RequestParams
from vibe.core.llm.backend.generic import OpenAIAdapter
from vibe.core.types import LLMMessage, Role


def _payload(thinking: str, extra_body: dict | None = None) -> dict:
    provider = ProviderConfig(
        name="longcat",
        api_base="https://api.longcat.chat/openai/v1",
        api_key_env_var="LONGCAT_API_KEY",
    )
    request = OpenAIAdapter().prepare_request(
        RequestParams(
            model_name="LongCat-2.0",
            messages=[LLMMessage(role=Role.USER, content="hi")],
            temperature=1.0,
            tools=None,
            max_tokens=None,
            tool_choice=None,
            enable_streaming=True,
            provider=provider,
            thinking=thinking,
            extra_body=extra_body,
        )
    )
    return json.loads(request.body)


@pytest.mark.parametrize("thinking", ["low", "medium", "high", "xhigh", "max"])
def test_longcat_maps_enabled_thinking_to_binary_field(thinking: str) -> None:
    payload = _payload(thinking)

    assert payload["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in payload


def test_longcat_maps_disabled_thinking_to_binary_field() -> None:
    payload = _payload("off")

    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload


def test_longcat_preserves_explicit_thinking_body() -> None:
    payload = _payload("max", extra_body={"thinking": {"type": "disabled"}})

    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
