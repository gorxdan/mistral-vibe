from __future__ import annotations

import json

import httpx
import pytest
import respx

from tests.backend.data.mistral import mistral_completion
from vibe.core.config import ModelConfig, ProviderCacheConfig, ProviderConfig
from vibe.core.config._settings import DEFAULT_PROVIDERS
from vibe.core.llm.backend.mistral import MistralBackend
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import Backend, LLMMessage, Role


def _provider(cache: ProviderCacheConfig) -> ProviderConfig:
    return ProviderConfig(
        name="mistral",
        api_base="https://api.mistral.ai/v1",
        backend=Backend.MISTRAL,
        cache=cache,
    )


def _request() -> CompletionRequest:
    return CompletionRequest(
        model=ModelConfig(name="mistral-large-latest", provider="mistral", alias="m"),
        messages=[LLMMessage(role=Role.USER, content="hello")],
        metadata={"session_id": "mistral-session-42"},
    )


def test_default_mistral_provider_enables_session_cache_routing() -> None:
    provider = next(
        provider for provider in DEFAULT_PROVIDERS if provider.name == "mistral"
    )
    assert provider.cache.session_keyed is True


@pytest.mark.asyncio
async def test_mistral_sends_session_prompt_cache_key() -> None:
    backend = MistralBackend(_provider(ProviderCacheConfig(session_keyed=True)))
    with respx.mock(base_url="https://api.mistral.ai") as mock_api:
        route = mock_api.post("/v1/chat/completions").mock(
            return_value=httpx.Response(status_code=200, json=mistral_completion("ok"))
        )
        await backend.complete(_request())

    payload = json.loads(route.calls.last.request.content)
    assert payload["prompt_cache_key"] == "mistral-session-42"


@pytest.mark.asyncio
async def test_mistral_honors_cache_mode_off() -> None:
    backend = MistralBackend(
        _provider(ProviderCacheConfig(mode="off", session_keyed=True))
    )
    with respx.mock(base_url="https://api.mistral.ai") as mock_api:
        route = mock_api.post("/v1/chat/completions").mock(
            return_value=httpx.Response(status_code=200, json=mistral_completion("ok"))
        )
        await backend.complete(_request())

    payload = json.loads(route.calls.last.request.content)
    assert "prompt_cache_key" not in payload
