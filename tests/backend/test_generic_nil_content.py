"""Regression tests for the OpenAI-compatible (generic) message serialization.

ollama's /v1/chat/completions rejects a message whose ``content`` field is
absent with ``invalid message content type: <nil>`` (verified live against
gemma4:26b on ollama 0.30.10). It tolerates a missing content ONLY on an
assistant message that carries tool_calls; every other role (system, user,
tool, and a reasoning-only assistant turn) must include content.

Any ``LLMMessage`` with ``content=None`` — a reasoning-only assistant turn on a
thinking model, an empty tool result, or an assistant whose accumulated content
collapsed to None in ``LLMMessage.__add__`` — was dropped by
``model_dump(exclude_none=True)``, producing the 400. The fix sends an empty
string so the key is always present.
"""

from __future__ import annotations

import json

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.adapter_port import RequestParams
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
        RequestParams(
        model_name="m",
        messages=messages,
        temperature=0.2,
        tools=None,
        max_tokens=None,
        tool_choice=None,
        enable_streaming=False,
        provider=_provider(),
    
        )
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
    # ollama rejects an absent content field on EVERY role except an assistant
    # carrying tool_calls; verified live against gemma4:26b. The real-world
    # trigger is a reasoning-only assistant turn (thinking models) whose content
    # is None. content must never be absent for any role.
    msgs = [
        LLMMessage(role=Role.system, content=None),
        LLMMessage(role=Role.user, content="q"),
        # reasoning-only assistant turn: no content, no tool_calls -> the bug.
        LLMMessage(role=Role.assistant, content=None, reasoning_content="hmm"),
        LLMMessage(role=Role.assistant, content=None, tool_calls=[ToolCall(id="c")]),
        LLMMessage(role=Role.tool, content="result", tool_call_id="c"),
    ]
    out = _serialized_messages(msgs)
    assert [m.get("content") for m in out] == ["", "q", "", "", "result"]
