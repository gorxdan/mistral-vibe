"""Tests for the ChatGPT-subscription Responses adapter + credential seam.

Covers the ``api_style="openai-chatgpt"`` path: system->instructions hoisting,
encrypted-reasoning include, and the GenericBackend credential resolver that
injects the OAuth bearer + ChatGPT-Account-ID headers.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import json

import httpx
import pytest
import respx

from vibe.core.auth import openai_oauth as oauth
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.llm.backend.adapter_port import RequestParams
from vibe.core.llm.backend.generic import GenericBackend
from vibe.core.llm.backend.openai_responses import (
    ChatGPTResponsesAdapter,
    OpenAIResponsesAdapter,
)
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import AvailableFunction, AvailableTool, LLMMessage, Role

CHATGPT_BASE = "https://chatgpt.test/backend-api/codex"
RESPONSES_URL = f"{CHATGPT_BASE}/responses"


def _b64url(data: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")


def _fake_jwt(account_id: str = "acct_123") -> str:
    return (
        f"{_b64url({'alg': 'none'})}."
        f"{_b64url({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}."
        "sig"
    )


def _store_tokens(account_id: str = "acct_123") -> None:
    oauth.save_tokens(
        oauth.OpenAIOAuthTokens(
            access_token="access-1",
            refresh_token="refresh-1",
            account_id=account_id,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            id_token=_fake_jwt(account_id),
        )
    )


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="openai-chatgpt", api_base=CHATGPT_BASE, api_style="openai-chatgpt"
    )


def _prepare(
    messages, *, thinking="high", tools=None, model_name="gpt-5.1-codex"
) -> dict:
    adapter = ChatGPTResponsesAdapter()
    params = RequestParams(
        model_name=model_name,
        messages=messages,
        temperature=0.2,
        tools=tools,
        max_tokens=None,
        tool_choice=None,
        enable_streaming=False,
        provider=_provider(),
        api_key="access-1",
        thinking=thinking,
    )
    return json.loads(adapter.prepare_request(params).body)


def test_system_message_hoisted_to_instructions() -> None:
    payload = _prepare([
        LLMMessage(role=Role.SYSTEM, content="You are helpful."),
        LLMMessage(role=Role.USER, content="hi"),
    ])
    assert payload["instructions"] == "You are helpful."
    # System must not also appear in input items.
    roles = [item.get("role") for item in payload["input"]]
    assert "system" not in roles
    assert "user" in roles


def test_instructions_falls_back_when_no_system() -> None:
    payload = _prepare([LLMMessage(role=Role.USER, content="hi")])
    assert payload["instructions"]  # non-empty (backend rejects empty)


def test_encrypted_reasoning_included_when_thinking_on() -> None:
    payload = _prepare([LLMMessage(role=Role.USER, content="hi")], thinking="high")
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"]["summary"] == "auto"
    assert payload["store"] is False


def test_no_include_when_thinking_off() -> None:
    # Codex models floor effort to 'low' even with thinking off (they 400 on
    # effort 'none'), so encrypted reasoning is still requested for them. Use a
    # platform model where thinking off maps to effort 'none' to exercise the
    # no-include branch.
    payload = _prepare(
        [LLMMessage(role=Role.USER, content="hi")], thinking="off", model_name="gpt-5.5"
    )
    assert "include" not in payload


def test_tool_choice_defaults_to_auto() -> None:
    tool = AvailableTool(
        function=AvailableFunction(
            name="run", description="run", parameters={"type": "object"}
        )
    )
    payload = _prepare([LLMMessage(role=Role.USER, content="hi")], tools=[tool])
    assert payload["tool_choice"] == "auto"


def _params(*, max_tokens: int | None) -> RequestParams:
    return RequestParams(
        model_name="gpt-5.1-codex",
        messages=[LLMMessage(role=Role.USER, content="hi")],
        temperature=0.2,
        tools=None,
        max_tokens=max_tokens,
        tool_choice=None,
        enable_streaming=False,
        provider=_provider(),
        api_key="access-1",
        thinking="high",
    )


def test_max_output_tokens_stripped_for_codex() -> None:
    # The codex backend rejects max_output_tokens ("Unsupported parameter:
    # max_output_tokens", HTTP 400). The adapter must drop it even when the
    # caller passes a limit, otherwise every codex call carrying a max_tokens
    # (safety judge, memory selector, max-output escalation) fails closed.
    adapter = ChatGPTResponsesAdapter()
    payload = json.loads(adapter.prepare_request(_params(max_tokens=4096)).body)
    assert "max_output_tokens" not in payload


def test_platform_responses_keeps_max_output_tokens() -> None:
    # Contract guard: only the codex variant strips the param. The platform
    # Responses API accepts max_output_tokens and must keep honoring it.
    adapter = OpenAIResponsesAdapter()
    payload = json.loads(adapter.prepare_request(_params(max_tokens=4096)).body)
    assert payload["max_output_tokens"] == 4096


@pytest.mark.asyncio
@respx.mock
async def test_backend_injects_oauth_bearer_and_account_header() -> None:
    _store_tokens("acct_xyz")
    # ChatGPT backend requires streaming; complete() now routes through the
    # streaming path, so the mock returns an SSE response.
    sse_body = (
        "\n\n".join([
            'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","role":"assistant","content":[]}}',
            'data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"hello"}',
            'data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"hello"}',
            'data: {"type":"response.completed","response":{"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":3,"output_tokens":1}}}',
            "data: [DONE]",
        ])
        + "\n\n"
    )
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            content=sse_body.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )
    )

    backend = GenericBackend(provider=_provider())
    model = ModelConfig(
        name="gpt-5.1-codex",
        provider="openai-chatgpt",
        alias="gpt-5.1-codex",
        thinking="high",
    )
    chunk = await backend.complete(
        CompletionRequest(
            model=model,
            messages=[
                LLMMessage(role=Role.SYSTEM, content="sys"),
                LLMMessage(role=Role.USER, content="hi"),
            ],
        )
    )

    assert chunk.message.content == "hello"
    assert route.called
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer access-1"
    assert request.headers["ChatGPT-Account-ID"] == "acct_xyz"
    assert request.headers["originator"] == oauth.OPENAI_ORIGINATOR
    body = json.loads(request.content)
    assert body["store"] is False
    assert body["instructions"] == "sys"


@pytest.mark.asyncio
async def test_backend_raises_when_not_signed_in() -> None:
    # No token store saved.
    backend = GenericBackend(provider=_provider())
    model = ModelConfig(
        name="gpt-5.1-codex", provider="openai-chatgpt", alias="gpt-5.1-codex"
    )
    with pytest.raises(oauth.OpenAINotAuthenticatedError):
        await backend.complete(
            CompletionRequest(
                model=model, messages=[LLMMessage(role=Role.USER, content="hi")]
            )
        )
