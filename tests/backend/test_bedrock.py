from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from vibe.core.config import ProviderCacheConfig, ProviderConfig
from vibe.core.llm.backend.adapter_port import RequestParams
from vibe.core.llm.backend.bedrock import (
    BEDROCK_MANTLE_SERVICE,
    DEFAULT_BEDROCK_REGION,
    BedrockAnthropicAdapter,
    BedrockAuthError,
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
    def test_cache_mode_off_removes_markers(self, adapter, provider) -> None:
        provider = provider.model_copy(
            update={"cache": ProviderCacheConfig(mode="off")}
        )
        req = adapter.prepare_request(
            _params(
                provider,
                messages=[
                    LLMMessage(role=Role.SYSTEM, content="Be helpful."),
                    LLMMessage(role=Role.USER, content="Hello"),
                ],
            )
        )

        payload = json.loads(req.body)
        assert "cache_control" not in payload["system"][0]
        assert "cache_control" not in payload["messages"][0]["content"][-1]

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


class _FakeFrozenCreds:
    def __init__(self) -> None:
        self.access_key = "AKIATEST"
        self.secret_key = "secret"
        self.token = "sessiontok"


class _FakeCreds:
    def get_frozen_credentials(self) -> _FakeFrozenCreds:
        return _FakeFrozenCreds()


class TestSigV4Fallback:
    def test_no_api_key_signs_with_sigv4(self, adapter, provider) -> None:
        # No api_key -> adapter must produce a SigV4 Authorization header.
        with patch(
            "vibe.core.llm.backend.bedrock._BEDROCK_CREDS.resolve",
            return_value=_FakeCreds(),
        ):
            req = adapter.prepare_request(_params(provider, api_key=None))

        assert req.base_url == "https://bedrock-mantle.eu-west-1.api.aws/anthropic"
        auth = req.headers.get("Authorization", "")
        assert auth.startswith("AWS4-HMAC-SHA256 ")
        # Session token forwarded; region + service come from the SigV4 scope.
        assert req.headers.get("X-Amz-Security-Token") == "sessiontok"
        # Bearer path header must NOT be present.
        assert "x-api-key" not in req.headers

    def test_no_api_key_no_credentials_raises(self, adapter, provider) -> None:
        with patch(
            "vibe.core.llm.backend.bedrock._BEDROCK_CREDS.resolve", return_value=None
        ):
            with pytest.raises(BedrockAuthError, match="No Amazon Bedrock credentials"):
                adapter.prepare_request(_params(provider, api_key=None))

    def test_api_key_takes_precedence_over_sigv4(self, adapter, provider) -> None:
        # When the bearer token resolves, x-api-key is set and SigV4 is skipped.
        with patch(
            "vibe.core.llm.backend.bedrock._BEDROCK_CREDS.resolve", return_value=None
        ) as mock_resolve:
            req = adapter.prepare_request(_params(provider, api_key="bearer-tok"))

        assert req.headers.get("x-api-key") == "bearer-tok"
        assert not req.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256")
        # SigV4 resolution never runs on the bearer path.
        mock_resolve.assert_not_called()

    def test_bedrock_mantle_service_constant(self) -> None:
        assert BEDROCK_MANTLE_SERVICE == "bedrock-mantle"
