"""Regression tests for the OpenAI-compatible (generic) message serialization.

ollama's /v1/chat/completions rejects a message whose ``content`` field is
absent with ``invalid message content type: <nil>`` (verified live against
gemma4:26b on ollama 0.30.10). It tolerates a missing content ONLY on an
assistant message that carries tool_calls; every other role (system, user,
and tool) must include content.

A reasoning-only assistant turn cannot be sent as an empty message because
some OpenAI-compatible providers reject it. It carries no tool result or visible
assistant output, so serialization drops it. Other messages with
``content=None`` receive an empty string so their required content key remains
present.
"""

from __future__ import annotations

import json

import pytest

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.adapter_port import RequestParams
from vibe.core.llm.backend.generic import OpenAIAdapter
from vibe.core.types import Backend, FunctionCall, LLMMessage, Role, ToolCall


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="ollama",
        api_base="http://127.0.0.1:11434/v1",
        backend=Backend.GENERIC,
        api_style="openai",
        reasoning_field_name="reasoning",
    )


def _openrouter_provider() -> ProviderConfig:
    return ProviderConfig(
        name="openrouter",
        api_base="https://openrouter.example/v1",
        backend=Backend.GENERIC,
        api_style="openai",
        reasoning_field_name="reasoning",
    )


def _serialized_messages(
    messages: list[LLMMessage], provider: ProviderConfig | None = None
) -> list[dict[str, object]]:
    req = OpenAIAdapter().prepare_request(
        RequestParams(
            model_name="m",
            messages=messages,
            temperature=0.2,
            tools=None,
            max_tokens=None,
            tool_choice=None,
            enable_streaming=False,
            provider=provider or _provider(),
        )
    )
    return json.loads(req.body)["messages"]


def test_assistant_tool_call_message_keeps_content_key() -> None:
    msg = LLMMessage(
        role=Role.ASSISTANT,
        content=None,
        reasoning_content="hidden reasoning",
        tool_calls=[
            ToolCall(
                id="call_1", index=0, function=FunctionCall(name="f", arguments="{}")
            )
        ],
    )
    assert msg.content is None  # precondition: the bug source

    out = _serialized_messages([msg])
    assert out[0]["content"] == ""  # present, not dropped -> ollama accepts it
    assert "tool_calls" in out[0]


def test_normal_message_content_is_preserved() -> None:
    out = _serialized_messages([
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.USER, content="hi"),
    ])
    assert [m["content"] for m in out] == ["sys", "hi"]


@pytest.mark.parametrize("content", [None, "", " \n"])
def test_reasoning_only_assistant_message_is_omitted(content: str | None) -> None:
    out = _serialized_messages(
        [
            LLMMessage(role=Role.USER, content="before"),
            LLMMessage(
                role=Role.ASSISTANT,
                content=content,
                reasoning_content="hidden reasoning",
            ),
            LLMMessage(role=Role.USER, content="after"),
        ],
        _openrouter_provider(),
    )

    assert out == [
        {"role": "user", "content": "before"},
        {"role": "user", "content": "after"},
    ]


def test_reasoning_only_assistant_message_is_preserved_for_ollama() -> None:
    out = _serialized_messages([
        LLMMessage(
            role=Role.ASSISTANT, content=None, reasoning_content="hidden reasoning"
        )
    ])

    assert out == [
        {"role": "assistant", "content": "", "reasoning": "hidden reasoning"}
    ]


def test_remaining_messages_have_a_content_key() -> None:
    # ollama rejects an absent content field on EVERY role except an assistant
    # carrying tool_calls; verified live against gemma4:26b. content must never
    # be absent for any remaining role.
    msgs = [
        LLMMessage(role=Role.SYSTEM, content=None),
        LLMMessage(role=Role.USER, content="q"),
        LLMMessage(role=Role.ASSISTANT, content=None, tool_calls=[ToolCall(id="c")]),
        LLMMessage(role=Role.TOOL, content="result", tool_call_id="c"),
    ]
    out = _serialized_messages(msgs)
    assert [m.get("content") for m in out] == ["", "q", "", "result"]
