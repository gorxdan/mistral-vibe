from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
import os
import types
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import httpx
import orjson

from vibe.core.llm.backend._image import to_data_uri as _to_data_uri
from vibe.core.llm.backend.adapter_port import (
    APIAdapter,
    PreparedRequest,
    RequestParams,
)
from vibe.core.llm.backend.anthropic import AnthropicAdapter
from vibe.core.llm.backend.openai_responses import (
    ChatGPTResponsesAdapter,
    OpenAIResponsesAdapter,
)
from vibe.core.llm.backend.reasoning_adapter import ReasoningAdapter
from vibe.core.llm.exceptions import BackendErrorBuilder
from vibe.core.llm.provider_limiter import provider_slot
from vibe.core.types import (
    AvailableTool,
    LLMChunk,
    LLMChunkAccumulator,
    LLMMessage,
    LLMUsage,
    Role,
    StrToolChoice,
)
from vibe.core.utils import async_generator_retry, async_retry
from vibe.core.utils.http import build_ssl_context
from vibe.core.utils.sse import iter_sse_lines

if TYPE_CHECKING:
    from vibe.core.config import ModelConfig, ProviderConfig


class OpenAIAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/chat/completions"

    def build_payload(
        self,
        model_name: str,
        converted_messages: list[dict[str, Any]],
        temperature: float,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model_name,
            "messages": converted_messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = [tool.model_dump(exclude_none=True) for tool in tools]
        if tool_choice:
            payload["tool_choice"] = (
                tool_choice
                if isinstance(tool_choice, str)
                else tool_choice.model_dump()
            )
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            # Provider-specific extras merged last (e.g. GLM's
            # {"thinking": {"type": "disabled"}}). Caller owns the format.
            payload.update(extra_body)

        return payload

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _reasoning_to_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and "reasoning_content" in msg_dict:
            msg_dict[field_name] = msg_dict.pop("reasoning_content")
        return msg_dict

    def _to_api_message(self, msg: LLMMessage, field_name: str) -> dict[str, Any]:
        msg_dict = msg.model_dump(
            exclude_none=True,
            exclude={
                "message_id",
                "reasoning_message_id",
                "reasoning_state",
                "injected",
                "images",
            },
        )
        # OpenAI-compatible servers (notably ollama) reject a message whose
        # content field is absent with "invalid message content type: <nil>".
        # Assistant messages carrying only tool_calls have content=None, which
        # exclude_none drops entirely. Send an empty string so the key is always
        # present (accepted by OpenAI/vLLM/llama.cpp/LM Studio alike).
        msg_dict.setdefault("content", "")
        return self._user_with_images_to_parts(
            self._reasoning_to_api(msg_dict, field_name), msg
        )

    def _reasoning_from_api(
        self, msg_dict: dict[str, Any], field_name: str
    ) -> dict[str, Any]:
        if field_name != "reasoning_content" and field_name in msg_dict:
            msg_dict["reasoning_content"] = msg_dict.pop(field_name)
        return msg_dict

    def _user_with_images_to_parts(
        self, msg_dict: dict[str, Any], source: LLMMessage
    ) -> dict[str, Any]:
        if source.role != Role.user or not source.images:
            return msg_dict
        parts: list[dict[str, Any]] = []
        text = msg_dict.get("content")
        if isinstance(text, str) and text:
            parts.append({"type": "text", "text": text})
        parts.extend(
            {"type": "image_url", "image_url": {"url": _to_data_uri(att)}}
            for att in source.images
        )
        msg_dict["content"] = parts
        return msg_dict

    def prepare_request(self, params: RequestParams) -> PreparedRequest:
        messages = params.messages
        enable_streaming = params.enable_streaming
        provider = params.provider
        extra_body = params.extra_body
        field_name = provider.reasoning_field_name
        converted_messages = [self._to_api_message(msg, field_name) for msg in messages]

        # Provider cache hints. Default is explicit/passthrough but inert unless
        # the provider sets extra_body/cache_key (empty fragment is skipped below).
        # May tag converted_messages in place; the returned fragment merges into
        # extra_body (caller wins).
        from vibe.core.llm.backend.cache_hints import build_cache_hint

        hint = build_cache_hint(provider, converted_messages)
        if hint:
            merged = dict(extra_body or {})
            for key, value in hint.items():
                merged.setdefault(key, value)
            extra_body = merged

        payload = self.build_payload(
            params.model_name,
            converted_messages,
            params.temperature,
            params.tools,
            params.max_tokens,
            params.tool_choice,
            params.response_format,
            extra_body,
        )

        if enable_streaming:
            payload["stream"] = True
            stream_options = {"include_usage": True}
            if provider.name == "mistral":
                stream_options["stream_tool_calls"] = True
            payload["stream_options"] = stream_options

        headers = self.build_headers(params.api_key)
        body = orjson.dumps(payload)

        return PreparedRequest(self.endpoint, headers, body)

    def _parse_message(
        self, data: dict[str, Any], field_name: str
    ) -> LLMMessage | None:
        if data.get("choices"):
            choice = data["choices"][0]
            if "message" in choice:
                msg_dict = self._reasoning_from_api(choice["message"], field_name)
                return LLMMessage.model_validate(msg_dict)
            if "delta" in choice:
                msg_dict = self._reasoning_from_api(choice["delta"], field_name)
                return LLMMessage.model_validate(msg_dict)
            raise ValueError("Invalid response data: missing message or delta")

        if "message" in data:
            msg_dict = self._reasoning_from_api(data["message"], field_name)
            return LLMMessage.model_validate(msg_dict)
        if "delta" in data:
            msg_dict = self._reasoning_from_api(data["delta"], field_name)
            return LLMMessage.model_validate(msg_dict)

        return None

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        message = self._parse_message(data, provider.reasoning_field_name)
        if message is None:
            message = LLMMessage(role=Role.assistant, content="")

        usage_data = data.get("usage") or {}
        details = usage_data.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or usage_data.get("cached_tokens") or 0
        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            cached_tokens=cached,
        )

        return LLMChunk(message=message, usage=usage)


_ADAPTERS: dict[str, APIAdapter] = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "reasoning": ReasoningAdapter(),
}


def _get_adapter(api_style: str) -> APIAdapter:
    """Load the adapter for the given API style."""
    if api_style == "openai-responses":
        return OpenAIResponsesAdapter()
    if api_style == "openai-chatgpt":
        return ChatGPTResponsesAdapter()
    if api_style not in _ADAPTERS:
        if api_style == "vertex-anthropic":
            from vibe.core.llm.backend.vertex import VertexAnthropicAdapter

            _ADAPTERS["vertex-anthropic"] = VertexAnthropicAdapter()
        else:
            raise KeyError(api_style)
    return _ADAPTERS[api_style]


# Network-op timeouts kept short so a dead/unreachable endpoint fails fast and
# failover can engage, instead of inheriting the long generation timeout (which
# blanket-applied would let an unreachable host hang for minutes). `read` stays
# at the full generation timeout so slow reasoning streams are not killed.
_CONNECT_TIMEOUT = 15.0
_NETWORK_OP_TIMEOUT = 60.0


class GenericBackend:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        provider: ProviderConfig,
        timeout: float = 720.0,
    ) -> None:
        """Initialize the backend.

        Args:
            client: Optional httpx client to use. If not provided, one will be created.
        """
        self._client = client
        self._owns_client = client is None
        self._provider = provider
        self._timeout = timeout

    async def __aenter__(self) -> GenericBackend:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._timeout,
                    connect=_CONNECT_TIMEOUT,
                    write=_NETWORK_OP_TIMEOUT,
                    pool=_NETWORK_OP_TIMEOUT,
                ),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                verify=build_ssl_context(),
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

    async def _resolve_auth(self) -> tuple[str | None, dict[str, str]]:
        """Resolve the request credential and any auth-bound headers.

        For normal providers this is just the static env-var API key. For the
        ChatGPT-subscription provider (``api_style="openai-chatgpt"``) it loads
        (and refreshes) the OAuth access token and returns the identity headers
        the ChatGPT backend requires (account id, originator, version).
        """
        if getattr(self._provider, "api_style", "openai") == "openai-chatgpt":
            from vibe.core.auth.openai_oauth import resolve_chatgpt_credentials

            creds = await resolve_chatgpt_credentials()
            return creds.access_token, creds.auth_headers()

        api_key = (
            os.getenv(self._provider.api_key_env_var)
            if self._provider.api_key_env_var
            else None
        )
        return api_key, {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._timeout,
                    connect=_CONNECT_TIMEOUT,
                    write=_NETWORK_OP_TIMEOUT,
                    pool=_NETWORK_OP_TIMEOUT,
                ),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                verify=build_ssl_context(),
            )
            self._owns_client = True
        return self._client

    async def complete(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMChunk:
        # The ChatGPT-subscription backend (codex) rejects non-streaming
        # requests with "Stream must be set to true", so route through the
        # streaming path and aggregate the chunks into a single LLMChunk.
        if getattr(self._provider, "api_style", "openai") == "openai-chatgpt":
            # Aggregate streamed deltas in O(n); folding with LLMChunk.__add__
            # per chunk re-concatenates the whole message every delta (O(n^2)).
            accumulator = LLMChunkAccumulator()
            async for chunk in self.complete_streaming(
                model=model,
                messages=messages,
                temperature=temperature,
                tools=tools,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                extra_headers=extra_headers,
                metadata=metadata,
                response_format=response_format,
                extra_body=extra_body,
            ):
                accumulator.add(chunk)
            aggregated = accumulator.build()
            if aggregated is None:
                raise BackendErrorBuilder.build_request_error(
                    provider=self._provider.name,
                    endpoint=self._provider.api_base,
                    error=httpx.RequestError("empty stream from ChatGPT backend"),
                    model=model.name,
                    messages=messages,
                    temperature=temperature,
                    has_tools=bool(tools),
                    tool_choice=tool_choice,
                )
            return aggregated

        api_key, auth_headers = await self._resolve_auth()

        api_style = getattr(self._provider, "api_style", "openai")
        adapter = _get_adapter(api_style)

        req = adapter.prepare_request(
            RequestParams(
                model_name=model.name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                enable_streaming=False,
                provider=self._provider,
                api_key=api_key,
                thinking=model.thinking,
                response_format=response_format,
                extra_body=extra_body,
            )
        )

        headers = req.headers
        if auth_headers:
            headers.update(auth_headers)
        if extra_headers:
            headers.update(extra_headers)

        base = req.base_url or self._provider.api_base
        url = f"{base}{req.endpoint}"

        try:
            res_data, _ = await self._make_request(url, req.body, headers)
            return adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    async def complete_streaming(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        temperature: float = 0.2,
        tools: list[AvailableTool] | None = None,
        max_tokens: int | None = None,
        tool_choice: StrToolChoice | AvailableTool | None = None,
        extra_headers: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        api_key, auth_headers = await self._resolve_auth()

        api_style = getattr(self._provider, "api_style", "openai")
        adapter = _get_adapter(api_style)

        req = adapter.prepare_request(
            RequestParams(
                model_name=model.name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                enable_streaming=True,
                provider=self._provider,
                api_key=api_key,
                thinking=model.thinking,
                response_format=response_format,
                extra_body=extra_body,
            )
        )

        headers = req.headers
        if auth_headers:
            headers.update(auth_headers)
        if extra_headers:
            headers.update(extra_headers)

        base = req.base_url or self._provider.api_base
        url = f"{base}{req.endpoint}"

        try:
            async for res_data in self._make_streaming_request(url, req.body, headers):
                yield adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=messages,
                temperature=temperature,
                has_tools=bool(tools),
                tool_choice=tool_choice,
            ) from e

    class HTTPResponse(NamedTuple):
        data: dict[str, Any]
        headers: dict[str, str]

    @async_retry(tries=3)
    async def _make_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> HTTPResponse:
        async with provider_slot(self._provider):
            client = self._get_client()
            response = await client.post(url, content=data, headers=headers)
            response.raise_for_status()

            response_headers = dict(response.headers.items())
            response_body = response.json()
            return self.HTTPResponse(response_body, response_headers)

    @async_generator_retry(tries=3)
    async def _make_streaming_request(
        self, url: str, data: bytes, headers: dict[str, str]
    ) -> AsyncGenerator[dict[str, Any]]:
        # Slot spans the whole stream — a live response stays in-flight until drained.
        async with provider_slot(self._provider):
            client = self._get_client()
            async with client.stream(
                method="POST", url=url, content=data, headers=headers
            ) as response:
                if not response.is_success:
                    await response.aread()
                response.raise_for_status()
                async for line in iter_sse_lines(response):
                    if line.strip() == "":
                        continue

                    # SSE comment/keepalive lines start with ':'
                    if line.startswith(":"):
                        continue
                    delim_index = line.find(":")
                    if delim_index == -1:
                        continue
                    key = line[:delim_index]
                    value = line[delim_index + 1 :]
                    if value.startswith(" "):
                        value = value[1:]
                    if key != "data":
                        # This might be the case with openrouter, so we just ignore it
                        continue
                    if value == "[DONE]":
                        return
                    yield orjson.loads(value.strip())

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None
