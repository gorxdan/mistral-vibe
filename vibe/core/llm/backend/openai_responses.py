from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

import orjson
from pydantic import TypeAdapter

from vibe.core.llm.backend._image import to_data_uri as _to_data_uri
from vibe.core.llm.backend.adapter_port import (
    APIAdapter,
    PreparedRequest,
    RequestParams,
)
from vibe.core.logger import logger
from vibe.core.types import (
    AvailableTool,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StopInfo,
    StopReason,
    StrToolChoice,
    ToolCall,
)

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig

_EMPTY_USAGE = LLMUsage(prompt_tokens=0, completion_tokens=0)


def responses_temperature_supported(model_name: str) -> bool:
    # The Responses API accepts temperature only for gpt-4/gpt-3.5; reasoning
    # models (gpt-5.x, o-series, fugu) omit it entirely — OpenAI's own codex CLI
    # sends no temperature for these. So a sent-temperature is the exception.
    return model_name.startswith(("gpt-4", "gpt-3.5"))


class _ResponsesInputTokensDetails(TypedDict, total=False):
    cached_tokens: int


class _ResponsesOutputTokensDetails(TypedDict, total=False):
    reasoning_tokens: int


class _ResponsesUsageData(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    input_tokens_details: _ResponsesInputTokensDetails
    output_tokens_details: _ResponsesOutputTokensDetails


class _ResponsesFunctionCallItem(TypedDict, total=False):
    type: str
    id: str
    call_id: str
    name: str
    arguments: str


class _ResponsesContentBlock(TypedDict, total=False):
    type: str
    text: str


class _ResponsesSummaryBlock(TypedDict, total=False):
    type: str
    text: str


class _ResponsesMessageItem(TypedDict, total=False):
    type: str
    id: str
    role: str
    phase: str
    content: list[_ResponsesContentBlock]


class _ResponsesReasoningItem(TypedDict, total=False):
    type: str
    encrypted_content: str
    summary: list[_ResponsesSummaryBlock]


class _ResponsesIncompleteDetails(TypedDict, total=False):
    reason: str


class _ResponsesObject(TypedDict, total=False):
    usage: _ResponsesUsageData | None
    output: list[dict[str, Any]]
    status: str
    incomplete_details: _ResponsesIncompleteDetails | None


# Responses API signals truncation via status/incomplete_details, not finish_reason;
# map to chat-completions StopInfo so backend_mixin's self-heal and refusal fire.
def _stop_info_from_response(response_obj: _ResponsesObject) -> StopInfo | None:
    if response_obj.get("status") != "incomplete":
        return None
    reason = (response_obj.get("incomplete_details") or {}).get("reason")
    match reason:
        case "max_output_tokens":
            return StopInfo(reason=StopReason.LENGTH)
        case "content_filter":
            return StopInfo(reason=StopReason.REFUSAL)
        case _:
            return StopInfo(reason=reason)


class _ResponsesErrorData(TypedDict, total=False):
    type: str
    message: str


class _ResponsesStreamEvent(TypedDict, total=False):
    type: str
    output_index: int
    delta: str
    call_id: str
    name: str
    arguments: str
    item: dict[str, Any]
    response: _ResponsesObject
    error: _ResponsesErrorData


_RESPONSES_OBJECT_ADAPTER = TypeAdapter(_ResponsesObject)
_RESPONSES_STREAM_EVENT_ADAPTER = TypeAdapter(_ResponsesStreamEvent)
_RESPONSES_FUNCTION_CALL_ITEM_ADAPTER = TypeAdapter(_ResponsesFunctionCallItem)
_RESPONSES_MESSAGE_ITEM_ADAPTER = TypeAdapter(_ResponsesMessageItem)
_RESPONSES_REASONING_ITEM_ADAPTER = TypeAdapter(_ResponsesReasoningItem)
_RESPONSES_ERROR_DATA_ADAPTER = TypeAdapter(_ResponsesErrorData)


@dataclass(slots=True)
class _ResponsesToolCallState:
    call_id: str | None = None
    name: str | None = None
    name_emitted: bool = False
    arguments_emitted: bool = False
    # Arguments arrive as a stream of deltas. Buffer the fragments and join on
    # read instead of re-concatenating the whole string on every delta, which is
    # O(n^2) over a large tool-call payload.
    _arg_base: str = ""
    _arg_parts: list[str] = field(default_factory=list)

    @property
    def arguments(self) -> str:
        if self._arg_parts:
            self._arg_base += "".join(self._arg_parts)
            self._arg_parts.clear()
        return self._arg_base

    @arguments.setter
    def arguments(self, value: str) -> None:
        self._arg_base = value
        self._arg_parts.clear()

    def append_arguments(self, delta: str) -> None:
        if delta:
            self._arg_parts.append(delta)


class _OpenAIResponsesStreamParser:
    def __init__(self) -> None:
        self._commentary_indices: set[int] = set()
        self._ignored_event_types: set[str] = set()
        self._tool_call_states: dict[int, _ResponsesToolCallState] = {}

    def reset(self) -> None:
        self._commentary_indices.clear()
        self._ignored_event_types.clear()
        self._tool_call_states.clear()

    def parse(self, data: _ResponsesStreamEvent) -> LLMChunk:
        handler = self._EVENT_HANDLERS.get(data.get("type", ""))
        if handler is not None:
            return handler(self, data)
        return self._on_unknown_event(data)

    @staticmethod
    def _is_commentary_message(item: dict[str, Any]) -> bool:
        return item.get("type") == "message" and item.get("phase") == "commentary"

    @staticmethod
    def _usage_from_response(usage_data: _ResponsesUsageData | None) -> LLMUsage:
        usage = usage_data or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return LLMUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            cached_tokens=input_details.get("cached_tokens", 0),
            reasoning_tokens=output_details.get("reasoning_tokens", 0),
        )

    @staticmethod
    def _reasoning_state_from_output(output: list[dict[str, Any]]) -> list[str] | None:
        reasoning_state: list[str] = []
        for item in output:
            if item.get("type") != "reasoning":
                continue
            reasoning_item = _RESPONSES_REASONING_ITEM_ADAPTER.validate_python(item)
            encrypted_content = reasoning_item.get("encrypted_content")
            if encrypted_content:
                reasoning_state.append(encrypted_content)
        return reasoning_state or None

    @staticmethod
    def _tool_call_from_item(
        item: _ResponsesFunctionCallItem, *, index: int | None = None
    ) -> ToolCall:
        item = _RESPONSES_FUNCTION_CALL_ITEM_ADAPTER.validate_python(item)
        return ToolCall(
            id=item.get("call_id") or item.get("id"),
            index=index,
            function=FunctionCall(
                name=item.get("name"), arguments=item.get("arguments", "")
            ),
        )

    @staticmethod
    def _empty_chunk() -> LLMChunk:
        return LLMChunk(
            message=LLMMessage(role=Role.ASSISTANT, content=""), usage=_EMPTY_USAGE
        )

    @staticmethod
    def _assistant_text_chunk(text: str) -> LLMChunk:
        return LLMChunk(
            message=LLMMessage(role=Role.ASSISTANT, content=text), usage=_EMPTY_USAGE
        )

    @staticmethod
    def _tool_call_chunk(
        call_id: str | None, name: str | None, arguments: str, index: int | None
    ) -> LLMChunk:
        if index is None:
            raise ValueError("Tool call chunk missing index")
        return LLMChunk(
            message=LLMMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        index=index,
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                ],
            ),
            usage=_EMPTY_USAGE,
        )

    @staticmethod
    def _reasoning_chunk(reasoning_content: str) -> LLMChunk:
        return LLMChunk(
            message=LLMMessage(
                role=Role.ASSISTANT, content="", reasoning_content=reasoning_content
            ),
            usage=_EMPTY_USAGE,
        )

    def _remember_tool_call_state(
        self,
        *,
        index: int,
        call_id: str | None,
        name: str | None,
        arguments: str | None,
        name_emitted: bool | None = None,
        arguments_emitted: bool | None = None,
    ) -> None:
        state = self._tool_call_states.setdefault(index, _ResponsesToolCallState())
        if call_id:
            state.call_id = call_id
        if name:
            state.name = name
        if arguments is not None:
            state.arguments = arguments
        if name_emitted is not None:
            state.name_emitted = name_emitted
        if arguments_emitted is not None:
            state.arguments_emitted = arguments_emitted

    def _finalize_tool_call(
        self,
        *,
        index: int | None,
        call_id: str | None,
        name: str | None,
        arguments: str | None,
    ) -> LLMChunk:
        if index is None:
            raise ValueError("Tool call chunk missing index")

        state = self._tool_call_states.get(index, _ResponsesToolCallState())
        resolved_call_id = call_id or state.call_id
        resolved_name = name or state.name
        previous_arguments = state.arguments
        final_arguments = arguments if arguments is not None else previous_arguments
        if (
            previous_arguments
            and final_arguments
            and not final_arguments.startswith(previous_arguments)
        ):
            logger.warning(
                "OpenAI Responses tool call arguments mismatch; using full final arguments from done event. previous=%r current=%r",
                previous_arguments,
                final_arguments,
            )

        should_emit_name = bool(resolved_name and not state.name_emitted)
        should_emit_arguments = bool(final_arguments) and not state.arguments_emitted

        self._remember_tool_call_state(
            index=index,
            call_id=resolved_call_id,
            name=resolved_name,
            arguments=final_arguments,
            name_emitted=state.name_emitted or should_emit_name,
            arguments_emitted=state.arguments_emitted or should_emit_arguments,
        )

        if not should_emit_name and not should_emit_arguments:
            return self._empty_chunk()

        return self._tool_call_chunk(
            call_id=resolved_call_id,
            name=resolved_name,
            arguments=final_arguments if should_emit_arguments else "",
            index=index,
        )

    def _on_response_created(self, _data: _ResponsesStreamEvent) -> LLMChunk:
        self.reset()
        return self._empty_chunk()

    def _on_text_delta(self, data: _ResponsesStreamEvent) -> LLMChunk:
        delta = data.get("delta", "")
        if data.get("output_index", 0) not in self._commentary_indices:
            return self._assistant_text_chunk(delta)
        return self._reasoning_chunk(delta)

    def _on_reasoning_delta(self, data: _ResponsesStreamEvent) -> LLMChunk:
        return self._reasoning_chunk(data.get("delta", ""))

    def _on_tool_call_delta(self, data: _ResponsesStreamEvent) -> LLMChunk:
        delta = data.get("delta", "")
        if not delta and not data.get("name") and not data.get("call_id"):
            return self._empty_chunk()

        index = data.get("output_index")
        if index is None:
            raise ValueError("Tool call chunk missing index")

        state = self._tool_call_states.setdefault(index, _ResponsesToolCallState())
        state.append_arguments(delta)
        self._remember_tool_call_state(
            index=index,
            call_id=data.get("call_id"),
            name=data.get("name"),
            arguments=None,
            name_emitted=state.name_emitted,
            arguments_emitted=state.arguments_emitted,
        )
        return self._empty_chunk()

    def _on_output_item_added(self, data: _ResponsesStreamEvent) -> LLMChunk:
        item = data.get("item") or {}
        match item.get("type"):
            case "message" if self._is_commentary_message(item):
                self._commentary_indices.add(data.get("output_index", 0))
            case "function_call":
                item = _RESPONSES_FUNCTION_CALL_ITEM_ADAPTER.validate_python(item)
                index = data.get("output_index")
                if index is not None:
                    self._remember_tool_call_state(
                        index=index,
                        call_id=item.get("call_id") or item.get("id"),
                        name=item.get("name"),
                        arguments=item.get("arguments", ""),
                        name_emitted=bool(item.get("name")),
                        arguments_emitted=False,
                    )
                tool_call = self._tool_call_from_item(
                    cast(_ResponsesFunctionCallItem, item), index=index
                )
                return self._tool_call_chunk(
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments="",
                    index=tool_call.index,
                )
        return self._empty_chunk()

    def _on_tool_call_done(self, data: _ResponsesStreamEvent) -> LLMChunk:
        return self._finalize_tool_call(
            index=data.get("output_index"),
            call_id=data.get("call_id"),
            name=data.get("name"),
            arguments=data.get("arguments"),
        )

    def _on_output_item_done(self, data: _ResponsesStreamEvent) -> LLMChunk:
        item = data.get("item") or {}
        match item.get("type"):
            case "message" if self._is_commentary_message(item):
                self._commentary_indices.add(data.get("output_index", 0))
            case "function_call":
                item = _RESPONSES_FUNCTION_CALL_ITEM_ADAPTER.validate_python(item)
                return self._finalize_tool_call(
                    index=data.get("output_index"),
                    call_id=item.get("call_id") or item.get("id"),
                    name=item.get("name"),
                    arguments=item.get("arguments"),
                )
        return self._empty_chunk()

    def _on_response_terminal(self, data: _ResponsesStreamEvent) -> LLMChunk:
        response_obj = cast(_ResponsesObject, data.get("response") or {})
        self.reset()
        output = response_obj.get("output") or []
        return LLMChunk(
            message=LLMMessage(
                role=Role.ASSISTANT,
                content="",
                reasoning_state=self._reasoning_state_from_output(output),
            ),
            usage=self._usage_from_response(response_obj.get("usage")),
            stop=_stop_info_from_response(response_obj),
        )

    def _on_error(self, data: _ResponsesStreamEvent) -> LLMChunk:
        self.reset()
        error = _RESPONSES_ERROR_DATA_ADAPTER.validate_python(data.get("error") or {})
        error_type = error.get("type", "unknown_error")
        error_message = error.get("message", "Unknown streaming error")
        raise RuntimeError(
            f"OpenAI Responses stream error ({error_type}): {error_message}"
        )

    def _on_unknown_event(self, data: _ResponsesStreamEvent) -> LLMChunk:
        if event_type := data.get("type"):
            if event_type not in self._ignored_event_types:
                logger.debug(
                    "Ignoring OpenAI Responses stream event type: %s", event_type
                )
                self._ignored_event_types.add(event_type)
        return self._empty_chunk()

    _EVENT_HANDLERS: ClassVar[
        dict[
            str,
            Callable[[_OpenAIResponsesStreamParser, _ResponsesStreamEvent], LLMChunk],
        ]
    ] = {
        "response.created": _on_response_created,
        "response.output_text.delta": _on_text_delta,
        "response.reasoning_summary_text.delta": _on_reasoning_delta,
        "response.summary_text.delta": _on_reasoning_delta,
        "response.function_call_arguments.delta": _on_tool_call_delta,
        "response.function_call_arguments.done": _on_tool_call_done,
        "response.output_item.added": _on_output_item_added,
        "response.output_item.done": _on_output_item_done,
        "response.completed": _on_response_terminal,
        "response.incomplete": _on_response_terminal,
        "error": _on_error,
    }


def _normalize_schema_for_strict(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    result = dict(node)

    props = result.get("properties")
    if isinstance(props, dict):
        result["additionalProperties"] = False
        result["properties"] = {
            k: _normalize_schema_for_strict(v) for k, v in props.items()
        }
        result["required"] = list(result["properties"].keys())

    if isinstance(result.get("items"), dict):
        result["items"] = _normalize_schema_for_strict(result["items"])

    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(result.get(key), list):
            result[key] = [_normalize_schema_for_strict(s) for s in result[key]]

    return result


class OpenAIResponsesAdapter(APIAdapter):
    endpoint: ClassVar[str] = "/responses"

    def __init__(self) -> None:
        self._stream_parser = _OpenAIResponsesStreamParser()

    @staticmethod
    def _is_temperature_supported(model_name: str) -> bool:
        return responses_temperature_supported(model_name)

    @staticmethod
    def _map_reasoning_effort(thinking: str, model_name: str = "") -> str:
        if thinking == "off":
            # codex-tuned models (e.g. gpt-5.3-codex-spark) reject effort 'none'
            # with a 400 — their minimum is 'low'. Platform/mini models (gpt-5.5,
            # gpt-5.4-mini) accept 'none' for genuinely no reasoning.
            return "low" if "codex" in model_name.lower() else "none"
        if thinking == "max":
            return "xhigh"
        return thinking

    @staticmethod
    def _to_responses_text_format(response_format: dict[str, Any]) -> dict[str, Any]:
        # The shared response_format carries the Chat Completions shape
        # ({type, json_schema: {name, schema}}). The Responses API requires the
        # fields flat under text.format: {type, name, schema}. Un-nest so the
        # server does not reject with "Missing required parameter: text.format.name".
        # The schema is also normalized for strict mode (additionalProperties:
        # false, all properties required) so the server does not reject with
        # "Invalid schema ... additionalProperties is required to be false".
        if nested := response_format.get("json_schema"):
            return {
                "type": response_format.get("type", "json_schema"),
                "name": nested.get("name", "workflow_output"),
                "schema": _normalize_schema_for_strict(nested.get("schema", {})),
            }
        return response_format

    def _convert_messages(self, messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []

        for msg in messages:
            match msg.role:
                case Role.SYSTEM:
                    input_items.append({"role": "system", "content": msg.content or ""})

                case Role.USER:
                    if msg.images:
                        parts: list[dict[str, Any]] = []
                        if msg.content:
                            parts.append({"type": "input_text", "text": msg.content})
                        parts.extend(
                            {"type": "input_image", "image_url": _to_data_uri(att)}
                            for att in msg.images
                        )
                        input_items.append({"role": "user", "content": parts})
                    else:
                        input_items.append({
                            "role": "user",
                            "content": msg.content or "",
                        })

                case Role.ASSISTANT:
                    for encrypted_content in msg.reasoning_state or []:
                        # `summary` is required by the /responses schema even when
                        # empty; omitting it 400s on replayed reasoning items.
                        input_items.append({
                            "type": "reasoning",
                            "encrypted_content": encrypted_content,
                            "summary": [],
                        })
                    input_items.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg.content or ""}],
                    })
                    for tc in msg.tool_calls or []:
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.id or "",
                            "name": tc.function.name or "",
                            "arguments": tc.function.arguments or "",
                        })

                case Role.TOOL:
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content or "",
                    })

                case _:
                    raise ValueError(f"Unsupported role: {msg.role}")

        return input_items

    def _convert_tool_for_responses(self, tool: AvailableTool) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters,
        }

    def build_payload(  # noqa: PLR0913
        self,
        *,
        model_name: str,
        input_items: list[dict[str, Any]],
        temperature: float | None,
        tools: list[AvailableTool] | None,
        max_tokens: int | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        thinking: str,
        enable_streaming: bool,
        verbosity: str | None = None,
        response_format: dict[str, Any] | None = None,
        cache_session_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_name,
            "input": input_items,
            "store": False,
        }
        # Pin the conversation to one cache partition (OpenAI's prefix auto-cache
        # load-balances across machines and misses without a routing key; Sakana
        # shares the same need). Prefer the stable per-conversation session id
        # (codex keys prompt_cache_key on its thread_id, and we send the same id
        # as the thread-id header so routing and the body key agree); fall back
        # to a content hash of the prefix for one-shot callers with no session.
        # Responses adapters cover OpenAI and Sakana, so no gating needed.
        from vibe.core.llm.backend.cache_hints import prefix_cache_key

        if cache_key := (cache_session_id or prefix_cache_key(input_items)):
            payload["prompt_cache_key"] = cache_key

        if temperature is not None and self._is_temperature_supported(model_name):
            payload["temperature"] = temperature

        effort = self._map_reasoning_effort(thinking, model_name)
        payload["reasoning"] = {"effort": effort}
        # Request encrypted reasoning so it can be echoed back across turns
        # (via reasoning_state in _convert_messages) instead of re-reasoned —
        # privacy-safe (no store needed). Only meaningful when reasoning is on.
        if effort != "none":
            payload["include"] = ["reasoning.encrypted_content"]

        if tools:
            payload["tools"] = [
                self._convert_tool_for_responses(tool) for tool in tools
            ]

        if tools and tool_choice:
            if isinstance(tool_choice, str):
                payload["tool_choice"] = tool_choice
            else:
                payload["tool_choice"] = {
                    "type": "function",
                    "name": tool_choice.function.name,
                }

        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens

        # text.verbosity (gpt-5.x output-length dial) and text.format share the
        # one `text` object, so merge rather than overwrite.
        text: dict[str, Any] = {}
        if response_format is not None:
            text["format"] = self._to_responses_text_format(response_format)
        if verbosity:
            text["verbosity"] = verbosity
        if text:
            payload["text"] = text

        if enable_streaming:
            payload["stream"] = True

        return payload

    def build_headers(self, api_key: str | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def prepare_request(self, params: RequestParams) -> PreparedRequest:
        model_name = params.model_name
        messages = params.messages
        temperature = params.temperature
        tools = params.tools
        max_tokens = params.max_tokens
        tool_choice = params.tool_choice
        enable_streaming = params.enable_streaming
        api_key = params.api_key
        thinking = params.thinking
        response_format = params.response_format
        extra_body = params.extra_body
        del extra_body  # generic-backend feature; not used by this path
        input_items = self._convert_messages(messages)

        payload = self.build_payload(
            model_name=model_name,
            input_items=input_items,
            temperature=temperature,
            tools=tools,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            thinking=thinking,
            verbosity=params.verbosity,
            enable_streaming=enable_streaming,
            response_format=response_format,
            cache_session_id=params.cache_session_id,
        )

        headers = self.build_headers(api_key)
        body = orjson.dumps(payload)

        return PreparedRequest(self.endpoint, headers, body)

    def _parse_output_items(self, output: list[dict[str, Any]]) -> LLMMessage:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for index, item in enumerate(output):
            match item.get("type"):
                case "message":
                    msg = _RESPONSES_MESSAGE_ITEM_ADAPTER.validate_python(item)
                    item_text_parts: list[str] = []
                    item_reasoning_parts: list[str] = []
                    is_commentary = self._stream_parser._is_commentary_message(item)

                    for block in msg.get("content", []):
                        block_type = block.get("type")
                        if is_commentary and block_type in {
                            "output_text",
                            "summary_text",
                            "reasoning_summary_text",
                        }:
                            item_reasoning_parts.append(block.get("text", ""))
                            continue

                        if block_type == "output_text":
                            item_text_parts.append(block.get("text", ""))

                    text = "".join(item_text_parts)
                    reasoning_content = "".join(item_reasoning_parts)
                    if is_commentary:
                        if reasoning_content:
                            reasoning_parts.append(reasoning_content)
                        continue
                    if text:
                        text_parts.append(text)
                    if reasoning_content:
                        reasoning_parts.append(reasoning_content)

                case "reasoning":
                    item = _RESPONSES_REASONING_ITEM_ADAPTER.validate_python(item)
                    for summary in item.get("summary", []):
                        if summary.get("type") in {
                            "summary_text",
                            "reasoning_summary_text",
                        }:
                            reasoning_parts.append(summary.get("text", ""))

                case "function_call":
                    tool_calls.append(
                        self._stream_parser._tool_call_from_item(
                            cast(_ResponsesFunctionCallItem, item), index=index
                        )
                    )

        return LLMMessage(
            role=Role.ASSISTANT,
            content="".join(text_parts),
            reasoning_content="".join(reasoning_parts) or None,
            reasoning_state=self._stream_parser._reasoning_state_from_output(output),
            tool_calls=tool_calls or None,
        )

    def parse_response(
        self, data: dict[str, Any], provider: ProviderConfig
    ) -> LLMChunk:
        event_type = data.get("type", "")

        if "output" in data and not event_type:
            response_data = _RESPONSES_OBJECT_ADAPTER.validate_python(data)
            output = response_data.get("output")
            if output is None:
                raise ValueError("OpenAI Responses response missing output")
            return LLMChunk(
                message=self._parse_output_items(output),
                usage=self._stream_parser._usage_from_response(
                    response_data.get("usage")
                ),
                stop=_stop_info_from_response(response_data),
            )

        return self._stream_parser.parse(
            _RESPONSES_STREAM_EVENT_ADAPTER.validate_python(data)
        )


class ChatGPTResponsesAdapter(OpenAIResponsesAdapter):
    _DEFAULT_INSTRUCTIONS: ClassVar[str] = (
        "You are a coding assistant operating in a terminal-based coding harness."
    )

    @staticmethod
    def _split_system(messages: Sequence[LLMMessage]) -> tuple[str, list[LLMMessage]]:
        system_parts: list[str] = []
        rest: list[LLMMessage] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)
            else:
                rest.append(msg)
        return "\n\n".join(system_parts), rest

    def prepare_request(self, params: RequestParams) -> PreparedRequest:
        base = super().prepare_request(params)
        body = orjson.loads(base.body)

        instructions, conversation = self._split_system(params.messages)
        body["input"] = self._convert_messages(conversation)
        body["instructions"] = instructions or self._DEFAULT_INSTRUCTIONS

        # Codex rejects max_output_tokens (HTTP 400 "Unsupported parameter");
        # the platform Responses API accepts it. Strip it only here.
        body.pop("max_output_tokens", None)

        effort = body.get("reasoning", {}).get("effort")
        if effort and effort != "none":
            body["include"] = ["reasoning.encrypted_content"]
            body["reasoning"]["summary"] = "auto"

        if params.tools and "tool_choice" not in body:
            body["tool_choice"] = "auto"

        new_body = orjson.dumps(body)
        return PreparedRequest(base.endpoint, base.headers, new_body)
