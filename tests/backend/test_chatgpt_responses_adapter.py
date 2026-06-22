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
from vibe.core.llm.backend.openai_responses import ChatGPTResponsesAdapter
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


def _prepare(messages, *, thinking="high", tools=None) -> dict:
    adapter = ChatGPTResponsesAdapter()
    params = RequestParams(
        model_name="gpt-5.1-codex",
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
        LLMMessage(role=Role.system, content="You are helpful."),
        LLMMessage(role=Role.user, content="hi"),
    ])
    assert payload["instructions"] == "You are helpful."
    # System must not also appear in input items.
    roles = [item.get("role") for item in payload["input"]]
    assert "system" not in roles
    assert "user" in roles


def test_instructions_falls_back_when_no_system() -> None:
    payload = _prepare([LLMMessage(role=Role.user, content="hi")])
    assert payload["instructions"]  # non-empty (backend rejects empty)


def test_encrypted_reasoning_included_when_thinking_on() -> None:
    payload = _prepare([LLMMessage(role=Role.user, content="hi")], thinking="high")
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["reasoning"]["summary"] == "auto"
    assert payload["store"] is False


def test_no_include_when_thinking_off() -> None:
    payload = _prepare([LLMMessage(role=Role.user, content="hi")], thinking="off")
    assert "include" not in payload


def test_tool_choice_defaults_to_auto() -> None:
    tool = AvailableTool(
        function=AvailableFunction(
            name="run", description="run", parameters={"type": "object"}
        )
    )
    payload = _prepare([LLMMessage(role=Role.user, content="hi")], tools=[tool])
    assert payload["tool_choice"] == "auto"


@pytest.mark.asyncio
@respx.mock
async def test_backend_injects_oauth_bearer_and_account_header() -> None:
    _store_tokens("acct_xyz")
    route = respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
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
        model=model,
        messages=[
            LLMMessage(role=Role.system, content="sys"),
            LLMMessage(role=Role.user, content="hi"),
        ],
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
            model=model, messages=[LLMMessage(role=Role.user, content="hi")]
        )
