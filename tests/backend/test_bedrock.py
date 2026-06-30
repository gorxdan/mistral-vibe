from __future__ import annotations

import json

import pytest

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.adapter_port import RequestParams
from vibe.core.llm.backend.bedrock import (
    DEFAULT_BEDROCK_REGION,
    BedrockAnthropicAdapter,
    build_bedrock_base_url,
)
from vibe.core.llm.backend.generic import _get_adapter
from vibe.core.types import AvailableFunction, AvailableTool, LLMMessage, Role


@pytest.fixture
def adapter() -> BedrockAnthropicAdapter:
    return BedrockAnthropicAdapter()


@pytest.fixture
def provider() -> ProviderConfig:
    return ProviderConfig(
        name="bedrock",
        api_base="https://bedrock-mantle.us-east-1.api.aws/anthropic",
        api_key_env_var="AWS_BEARER_TOKEN_BEDROCK",
        api_style="bedrock-anthropic",
        region="eu-west-1",
    )


def _params(
    provider: ProviderConfig,
    *,
    api_key: str | None = "test-bedrock-key",
    enable_streaming: bool = False,
    max_tokens: int | None = 1024,
    messages: list[LLMMessage] | None = None,
) -> RequestParams:
    return RequestParams(
        model_name="anthropic.claude-haiku-4-5",
        messages=messages or [LLMMessage(role=Role.USER, content="Hello")],
        temperature=0.5,
        tools=None,
        max_tokens=max_tokens,
        tool_choice=None,
        enable_streaming=enable_streaming,
        provider=provider,
        api_key=api_key,
    )


class TestBuildBaseUrl:
    def test_region_substituted(self) -> None:
        assert (
            build_bedrock_base_url("eu-west-1")
            == "https://bedrock-mantle.eu-west-1.api.aws/anthropic"
        )

    def test_us_east_1(self) -> None:
        assert (
            build_bedrock_base_url("us-east-1")
            == "https://bedrock-mantle.us-east-1.api.aws/anthropic"
        )

    def test_default_region_is_us_east_1(self) -> None:
        assert DEFAULT_BEDROCK_REGION == "us-east-1"


class TestPrepareRequest:
    def test_base_url_set_from_region(self, adapter, provider) -> None:
        req = adapter.prepare_request(_params(provider))

        assert req.base_url == "https://bedrock-mantle.eu-west-1.api.aws/anthropic"
        assert req.endpoint == "/v1/messages"

    def test_api_key_in_x_api_key_header(self, adapter, provider) -> None:
        req = adapter.prepare_request(_params(provider, api_key="bedrock-bearer"))

        assert req.headers["x-api-key"] == "bedrock-bearer"

    def test_payload_is_standard_anthropic(self, adapter, provider) -> None:
        req = adapter.prepare_request(_params(provider))

        payload = json.loads(req.body)
        assert payload["model"] == "anthropic.claude-haiku-4-5"
        # The parent AnthropicAdapter wraps the last user message in a text block
        # with a prompt-caching cache_control marker; assert the text survives.
        user_msg = payload["messages"][0]
        assert user_msg["role"] == "user"
        assert "Hello" in json.dumps(user_msg["content"])
        assert payload["max_tokens"] == 1024

    def test_defaults_to_us_east_1_when_region_unset(self, adapter) -> None:
        provider = ProviderConfig(
            name="bedrock",
            api_base="https://bedrock-mantle.us-east-1.api.aws/anthropic",
            api_style="bedrock-anthropic",
        )
        req = adapter.prepare_request(_params(provider))

        assert req.base_url == "https://bedrock-mantle.us-east-1.api.aws/anthropic"

    def test_stream_flag_flows_through(self, adapter, provider) -> None:
        req = adapter.prepare_request(_params(provider, enable_streaming=True))

        payload = json.loads(req.body)
        assert payload.get("stream") is True

    def test_anthropic_version_header_present(self, adapter, provider) -> None:
        # Bedrock Mantle requires anthropic-version, inherited from the parent.
        req = adapter.prepare_request(_params(provider))

        assert "anthropic-version" in req.headers

    def test_tools_passthrough(self, adapter, provider) -> None:
        tools = [
            AvailableTool(
                function=AvailableFunction(
                    name="search",
                    description="search the web",
                    parameters={"type": "object", "properties": {}},
                )
            )
        ]
        req = adapter.prepare_request(
            RequestParams(
                model_name="anthropic.claude-haiku-4-5",
                messages=[LLMMessage(role=Role.USER, content="Hi")],
                temperature=None,
                tools=tools,
                max_tokens=1024,
                tool_choice=None,
                enable_streaming=False,
                provider=provider,
                api_key="k",
            )
        )

        payload = json.loads(req.body)
        assert payload["tools"][0]["name"] == "search"


class TestGetAdapterRegistration:
    def test_bedrock_anthropic_returns_bedrock_adapter(self) -> None:
        resolved = _get_adapter("bedrock-anthropic")
        assert isinstance(resolved, BedrockAnthropicAdapter)

    def test_cached_after_first_resolution(self) -> None:
        first = _get_adapter("bedrock-anthropic")
        second = _get_adapter("bedrock-anthropic")
        assert first is second
