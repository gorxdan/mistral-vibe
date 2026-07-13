from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from http import HTTPStatus
import re
import types
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import httpx
import orjson

from vibe.core.config import resolve_api_key
from vibe.core.llm.backend._image import to_data_uri as _to_data_uri
from vibe.core.llm.backend.adapter_port import (
    APIAdapter,
    PreparedRequest,
    RequestParams,
    trailing_ephemeral_count,
)
from vibe.core.llm.backend.anthropic import AnthropicAdapter
from vibe.core.llm.backend.openai_responses import (
    ChatGPTResponsesAdapter,
    OpenAIResponsesAdapter,
)
from vibe.core.llm.backend.reasoning_adapter import ReasoningAdapter
from vibe.core.llm.exceptions import BackendErrorBuilder
from vibe.core.llm.provider_limiter import provider_slot
from vibe.core.llm.provider_retry import (
    ProviderRetryController,
    SpendRetryCause,
    authorize_provider_retry,
    bind_provider_retry_controller,
    iterate_provider_stream,
)
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import (
    AvailableTool,
    LLMChunk,
    LLMChunkAccumulator,
    LLMMessage,
    LLMUsage,
    Role,
    StopInfo,
    StrToolChoice,
)
from vibe.core.utils import async_generator_retry, async_retry
from vibe.core.utils.http import build_ssl_context
from vibe.core.utils.sse import iter_sse_lines

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig


class OpenAIAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/chat/completions"

    def build_payload(
        self,
        model_name: str,
        converted_messages: list[dict[str, Any]],
        temperature: float | None,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model_name, "messages": converted_messages}
        if temperature is not None:
            payload["temperature"] = temperature

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
                "injected_kind",
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
        if source.role != Role.USER or not source.images:
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
        converted_messages = [
            self._to_api_message(msg, field_name)
            for msg in messages
            if not (
                # OpenRouter can route to Cohere, which rejects this otherwise
                # invisible assistant turn. Other generic providers replay it.
                provider.name == "openrouter"
                and msg.role == Role.ASSISTANT
                and not (msg.content or "").strip()
                and not msg.tool_calls
            )
        ]

        # Provider cache hints. Default is explicit/passthrough but inert unless
        # the provider sets extra_body/cache_key (empty fragment is skipped below).
        # May tag converted_messages in place; the returned fragment merges into
        # extra_body (caller wins).
        from vibe.core.llm.backend.cache_hints import build_cache_hint

        hint = build_cache_hint(
            provider,
            converted_messages,
            session_id=params.cache_session_id,
            skip_trailing=trailing_ephemeral_count(messages),
        )
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

        from vibe.core.llm.backend._provider_compat import apply_openai_chat_thinking

        provider_handles_thinking = apply_openai_chat_thinking(
            payload, provider_name=provider.name, level=params.thinking
        )
        # Verbatim: xhigh/max are real API tiers. 400-rejection self-heals.
        # Skip when the caller set it via extra_body (caller owns the format).
        if (
            not provider_handles_thinking
            and params.thinking != "off"
            and "reasoning_effort" not in payload
        ):
            payload["reasoning_effort"] = params.thinking

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
            message = LLMMessage(role=Role.ASSISTANT, content="")

        usage_data = data.get("usage") or {}
        from vibe.core.llm.backend._usage_fields import (
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            reported_cost_usd,
        )

        usage = LLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            cached_tokens=cache_read_tokens(usage_data),
            cache_write_tokens=cache_write_tokens(usage_data),
            reasoning_tokens=reasoning_tokens(usage_data),
            reported_cost_usd=reported_cost_usd(usage_data),
        )

        return LLMChunk(
            message=message, usage=usage, stop=StopInfo.from_chat_choices(data)
        )


_ADAPTERS: dict[str, APIAdapter] = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "reasoning": ReasoningAdapter(),
}


def _get_adapter(api_style: str) -> APIAdapter:
    if api_style == "openai-responses":
        return OpenAIResponsesAdapter()
    if api_style == "openai-chatgpt":
        return ChatGPTResponsesAdapter()
    if api_style not in _ADAPTERS:
        if api_style == "vertex-anthropic":
            from vibe.core.llm.backend.vertex import VertexAnthropicAdapter

            _ADAPTERS["vertex-anthropic"] = VertexAnthropicAdapter()
        elif api_style == "bedrock-anthropic":
            from vibe.core.llm.backend.bedrock import BedrockAnthropicAdapter

            _ADAPTERS["bedrock-anthropic"] = BedrockAnthropicAdapter()
        else:
            raise KeyError(api_style)
    return _ADAPTERS[api_style]


def adapter_supports_max_output_escalation(api_style: str) -> bool:
    try:
        adapter = _get_adapter(api_style)
    except KeyError:
        return True
    return adapter.supports_max_output_escalation


# Bound connect/write/pool so a dead endpoint fails fast (failover can engage);
# `read` keeps the full generation timeout so slow reasoning isn't killed.
_CONNECT_TIMEOUT = 15.0
_NETWORK_OP_TIMEOUT = 60.0

# Time-to-first-byte budget: caps stream-open + status so a provider that accepts
# the connection but never streams fails fast (releasing the slot) instead of
# hanging the full `read` budget. Disarmed once the first SSE byte arrives, so
# slow generation is unaffected.
_OPEN_TIMEOUT = 90.0

# Reasoning-effort tiers, highest → lowest. Used to recover when a backend
# rejects an effort value (a model without "max", or codex rejecting "none"):
# retry with the supported value nearest the rejected one. Nearest-by-tier
# downgrades a too-high request to the model's ceiling and lifts a too-low one
# to its floor — i.e. "max if available, else the highest level the model has".
_EFFORT_ORDER: tuple[str, ...] = (
    "max",
    "xhigh",
    "high",
    "medium",
    "low",
    "minimal",
    "none",
)
_REJECTED_EFFORT_FIELD_RE = re.compile(
    r"unsupported value:\s*'reasoning_effort'\s+does not support\s*'([^']+)'",
    re.IGNORECASE,
)
_REJECTED_EFFORT_RE = re.compile(r"unsupported value:\s*'([^']+)'", re.IGNORECASE)
_SUPPORTED_EFFORTS_RE = re.compile(r"supported values are:\s*([^.]+)", re.IGNORECASE)


def _nearest_supported_effort(error_text: str) -> str | None:
    """For an 'unsupported reasoning effort' 400 body, return the supported
    effort nearest the rejected one (ties favour the higher tier), else None.
    """
    rejected_m = _REJECTED_EFFORT_FIELD_RE.search(
        error_text
    ) or _REJECTED_EFFORT_RE.search(error_text)
    supported_m = _SUPPORTED_EFFORTS_RE.search(error_text)
    if not rejected_m or not supported_m:
        return None
    rejected = rejected_m.group(1).lower()
    if rejected not in _EFFORT_ORDER:
        return None
    supported = [
        v.lower()
        for v in re.findall(r"'([^']+)'", supported_m.group(1))
        if v.lower() in _EFFORT_ORDER
    ]
    if not supported:
        return None
    ri = _EFFORT_ORDER.index(rejected)
    return min(
        supported,
        key=lambda s: (abs(_EFFORT_ORDER.index(s) - ri), _EFFORT_ORDER.index(s)),
    )


def _patch_reasoning_effort(body: bytes, effort: str) -> bytes | None:
    """Return *body* with its reasoning effort set to *effort*; None if the body
    has no effort field or already uses it. Handles both the Responses shape
    (``reasoning.effort``) and chat-completions (``reasoning_effort``).
    """
    payload = orjson.loads(body)
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        if reasoning["effort"] == effort:
            return None
        reasoning["effort"] = effort
    elif "reasoning_effort" in payload:
        if payload["reasoning_effort"] == effort:
            return None
        payload["reasoning_effort"] = effort
    else:
        return None
    return orjson.dumps(payload)


class GenericBackend:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        provider: ProviderConfig,
        timeout: float = 720.0,
        retry_max_elapsed_time: float = 300.0,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._provider = provider
        self._timeout = timeout
        self._retry_max_elapsed_time = retry_max_elapsed_time

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

        For normal providers this is the configured API key, resolved from the
        environment or keyring. For the ChatGPT-subscription provider
        (``api_style="openai-chatgpt"``) it loads (and refreshes) the OAuth access
        token and returns the identity headers the ChatGPT backend requires
        (account id, originator, version).
        """
        if getattr(self._provider, "api_style", "openai") == "openai-chatgpt":
            from vibe.core.auth.openai_oauth import resolve_chatgpt_credentials

            creds = await resolve_chatgpt_credentials()
            return creds.access_token, creds.auth_headers()

        api_key = (
            resolve_api_key(self._provider.api_key_env_var)
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

    def _retry_body_for_effort(
        self, error: httpx.HTTPStatusError, body: bytes
    ) -> bytes | None:
        """On a 400 that rejects the reasoning effort, return a body patched to
        the nearest supported effort (so ``max``→``xhigh`` falls back to the
        model's ceiling); None when the error is unrelated.
        """
        response = error.response
        if response is None or response.status_code != HTTPStatus.BAD_REQUEST:
            return None
        try:
            effort = _nearest_supported_effort(response.text)
        except Exception:
            return None
        return _patch_reasoning_effort(body, effort) if effort else None

    async def complete(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> LLMChunk:
        model = request.model
        # The ChatGPT-subscription backend (codex) rejects non-streaming
        # requests with "Stream must be set to true", so route through the
        # streaming path and aggregate the chunks into a single LLMChunk.
        if getattr(self._provider, "api_style", "openai") == "openai-chatgpt":
            # Aggregate streamed deltas in O(n); folding with LLMChunk.__add__
            # per chunk re-concatenates the whole message every delta (O(n^2)).
            accumulator = LLMChunkAccumulator()
            async for chunk in self.complete_streaming(
                request, response_headers_sink=response_headers_sink
            ):
                accumulator.add(chunk)
            aggregated = accumulator.build()
            if aggregated is None:
                raise BackendErrorBuilder.build_request_error(
                    provider=self._provider.name,
                    endpoint=self._provider.api_base,
                    error=httpx.RequestError("empty stream from ChatGPT backend"),
                    model=model.name,
                    messages=request.messages,
                    temperature=request.temperature,
                    has_tools=bool(request.tools),
                    tool_choice=request.tool_choice,
                )
            return aggregated

        api_key, auth_headers = await self._resolve_auth()

        adapter = _get_adapter(getattr(self._provider, "api_style", "openai"))

        req = adapter.prepare_request(
            RequestParams(
                model_name=model.name,
                messages=request.messages,
                temperature=request.temperature,
                tools=request.tools,
                max_tokens=request.max_tokens,
                tool_choice=request.tool_choice,
                enable_streaming=False,
                provider=self._provider,
                api_key=api_key,
                thinking=model.thinking,
                verbosity=model.verbosity,
                response_format=request.response_format,
                extra_body=request.extra_body,
                cache_session_id=(request.metadata or {}).get("session_id"),
            )
        )

        headers = req.headers
        if auth_headers:
            headers.update(auth_headers)
        if request.extra_headers:
            headers.update(request.extra_headers)

        url = f"{req.base_url or self._provider.api_base}{req.endpoint}"

        controller = ProviderRetryController(
            max_elapsed_time=self._retry_max_elapsed_time
        )
        try:
            with bind_provider_retry_controller(controller):
                res_data, resp_headers = await self._make_request(
                    url, req.body, headers
                )
            if response_headers_sink is not None:
                response_headers_sink.update(resp_headers)
            return adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            retry_body = self._retry_body_for_effort(e, req.body)
            if retry_body is not None:
                with bind_provider_retry_controller(controller):
                    allowed = await authorize_provider_retry(
                        SpendRetryCause.REASONING_EFFORT, delay_s=0.0
                    )
                    if allowed:
                        res_data, resp_headers = await self._make_request(
                            url, retry_body, headers
                        )
                        if response_headers_sink is not None:
                            response_headers_sink.update(resp_headers)
                        return adapter.parse_response(res_data, self._provider)
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=request.messages,
                temperature=request.temperature,
                has_tools=bool(request.tools),
                tool_choice=request.tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=request.messages,
                temperature=request.temperature,
                has_tools=bool(request.tools),
                tool_choice=request.tool_choice,
            ) from e

    async def complete_streaming(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        model = request.model
        api_key, auth_headers = await self._resolve_auth()

        adapter = _get_adapter(getattr(self._provider, "api_style", "openai"))

        req = adapter.prepare_request(
            RequestParams(
                model_name=model.name,
                messages=request.messages,
                temperature=request.temperature,
                tools=request.tools,
                max_tokens=request.max_tokens,
                tool_choice=request.tool_choice,
                enable_streaming=True,
                provider=self._provider,
                api_key=api_key,
                thinking=model.thinking,
                verbosity=model.verbosity,
                response_format=request.response_format,
                extra_body=request.extra_body,
                cache_session_id=(request.metadata or {}).get("session_id"),
            )
        )

        headers = req.headers
        if auth_headers:
            headers.update(auth_headers)
        if request.extra_headers:
            headers.update(request.extra_headers)

        url = f"{req.base_url or self._provider.api_base}{req.endpoint}"

        controller = ProviderRetryController(
            max_elapsed_time=self._retry_max_elapsed_time
        )
        try:
            async for res_data in iterate_provider_stream(
                self._make_streaming_request(
                    url, req.body, headers, response_headers_sink=response_headers_sink
                ),
                controller,
            ):
                yield adapter.parse_response(res_data, self._provider)

        except httpx.HTTPStatusError as e:
            retry_body = self._retry_body_for_effort(e, req.body)
            if retry_body is not None:
                with bind_provider_retry_controller(controller):
                    allowed = await authorize_provider_retry(
                        SpendRetryCause.REASONING_EFFORT, delay_s=0.0
                    )
                if allowed:
                    async for res_data in iterate_provider_stream(
                        self._make_streaming_request(
                            url,
                            retry_body,
                            headers,
                            response_headers_sink=response_headers_sink,
                        ),
                        controller,
                    ):
                        yield adapter.parse_response(res_data, self._provider)
                    return
            raise BackendErrorBuilder.build_http_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=request.messages,
                temperature=request.temperature,
                has_tools=bool(request.tools),
                tool_choice=request.tool_choice,
            ) from e
        except httpx.RequestError as e:
            raise BackendErrorBuilder.build_request_error(
                provider=self._provider.name,
                endpoint=url,
                error=e,
                model=model.name,
                messages=request.messages,
                temperature=request.temperature,
                has_tools=bool(request.tools),
                tool_choice=request.tool_choice,
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
        self,
        url: str,
        data: bytes,
        headers: dict[str, str],
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        # Slot spans the whole stream — a live response stays in-flight until drained.
        async with provider_slot(self._provider):
            client = self._get_client()
            ttft_armed = True
            try:
                async with (
                    asyncio.timeout(_OPEN_TIMEOUT) as open_deadline,
                    client.stream(
                        method="POST", url=url, content=data, headers=headers
                    ) as response,
                ):
                    if not response.is_success:
                        await response.aread()
                    response.raise_for_status()
                    # Surface response headers (e.g. the codex `x-codex-turn-state`
                    # sticky-routing token) to the caller as soon as the stream
                    # opens, before any chunk — the caller replays it on the next
                    # request.
                    if response_headers_sink is not None:
                        response_headers_sink.update(dict(response.headers))
                    async for line in iter_sse_lines(response):
                        if ttft_armed:
                            # First byte: provider is streaming — lift the cap so
                            # slow reasoning generation is not killed.
                            open_deadline.reschedule(None)
                            ttft_armed = False
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
            except TimeoutError as exc:
                # Re-raise as a retryable network timeout so existing backoff
                # tries a fresh node and the slot is freed between attempts.
                raise httpx.ConnectTimeout(
                    f"time-to-first-byte exceeded {_OPEN_TIMEOUT:.0f}s"
                ) from exc

    async def close(self) -> None:
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None
