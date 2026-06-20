"""Regression tests for the OpenAI-compatible (generic) message serialization.

ollama's /v1/chat/completions rejects a message whose ``content`` field is
absent with ``invalid message content type: <nil>``. Assistant messages that
carry only tool_calls have ``content=None`` (see ``LLMMessage.__add__``), which
``model_dump(exclude_none=True)`` would otherwise drop entirely.
"""

from __future__ import annotations

import json

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.generic import OpenAIAdapter
from vibe.core.types import FunctionCall, LLMMessage, Role, ToolCall


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="ollama",
        api_base="http://127.0.0.1:11434/v1",
        backend="generic",  # type: ignore[arg-type]
        api_style="openai",
        reasoning_field_name="reasoning",
    )


def _serialized_messages(messages: list[LLMMessage]) -> list[dict[str, object]]:
    req = OpenAIAdapter().prepare_request(
        model_name="m",
        messages=messages,
        temperature=0.2,
        tools=None,
        max_tokens=None,
        tool_choice=None,
        enable_streaming=False,
        provider=_provider(),
    )
    return json.loads(req.body)["messages"]


def test_assistant_tool_call_message_keeps_content_key() -> None:
    msg = LLMMessage(
        role=Role.assistant,
        content=None,
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
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="hi"),
    ])
    assert [m["content"] for m in out] == ["sys", "hi"]


def test_every_message_has_a_content_key() -> None:
    # Whatever the role, content must never be absent for OpenAI-compatible
    # servers; guards against a future exclude_none regression.
    msgs = [
        LLMMessage(role=Role.assistant, content=None, tool_calls=[ToolCall(id="c")]),
        LLMMessage(role=Role.tool, content="result", tool_call_id="c"),
        LLMMessage(role=Role.user, content="q"),
    ]
    out = _serialized_messages(msgs)
    assert all("content" in m for m in out)
