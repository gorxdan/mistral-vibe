from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson
from pydantic import ValidationError

from vibe.core.llm.models import (
    FailedToolCall,
    ParsedMessage,
    ParsedToolCall,
    ResolvedMessage,
    ResolvedToolCall,
)
from vibe.core.types import (
    AvailableFunction,
    AvailableTool,
    LLMMessage,
    Role,
    StrToolChoice,
)

if TYPE_CHECKING:
    from vibe.core.tools.manager import ToolManager


def _trim_description(description: str, max_chars: int) -> str:
    """Shorten a tool description to its first sentence, falling back to a
    max_chars word-boundary cut. Keeps the lead intent the model needs to choose
    the tool while dropping the elaboration that bloats a small-window prompt.
    """
    text = description.strip()
    first_period = text.find(". ")
    if 0 <= first_period < max_chars:
        return text[: first_period + 1]
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    space = head.rfind(" ")
    return (head[:space] if space > 0 else head).rstrip() + "…"


class APIToolFormatHandler:
    @property
    def name(self) -> str:
        return "api"

    def get_available_tools(
        self,
        tool_manager: ToolManager,
        *,
        trim_descriptions: bool = False,
        description_max_chars: int = 220,
    ) -> list[AvailableTool]:
        # The param schema (names/types/enums) is always sent verbatim — only the
        # prose description is shortened, and only on small-window tiers, so tool
        # selection stays accurate while the per-turn schema cost drops.
        return [
            AvailableTool(
                function=AvailableFunction(
                    name=tool_class.get_name(),
                    description=_trim_description(
                        tool_class.description, description_max_chars
                    )
                    if trim_descriptions
                    else tool_class.description,
                    parameters=tool_class.get_parameters(),
                )
            )
            for tool_class in tool_manager.manifest_tools.values()
        ]

    def get_tool_choice(self) -> StrToolChoice | AvailableTool:
        return "auto"

    def process_api_response_message(self, message: Any) -> LLMMessage:
        clean_message = {
            "role": message.role,
            "content": message.content,
            "reasoning_content": getattr(message, "reasoning_content", None),
            "reasoning_state": getattr(message, "reasoning_state", None),
            "reasoning_signature": getattr(message, "reasoning_signature", None),
        }

        if message.tool_calls:
            clean_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "index": tc.index,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return LLMMessage.model_validate(clean_message)

    def parse_message(self, message: LLMMessage) -> ParsedMessage:
        tool_calls = []

        api_tool_calls = message.tool_calls or []
        for tc in api_tool_calls:
            if not (function_call := tc.function):
                continue
            try:
                args = orjson.loads(function_call.arguments or "{}")
            except orjson.JSONDecodeError:
                args = {}

            tool_calls.append(
                ParsedToolCall(
                    tool_name=function_call.name or "",
                    raw_args=args,
                    call_id=tc.id or "",
                )
            )

        return ParsedMessage(tool_calls=tool_calls)

    def resolve_tool_calls(
        self, parsed: ParsedMessage, tool_manager: ToolManager
    ) -> ResolvedMessage:
        resolved_calls = []
        failed_calls = []

        active_tools = tool_manager.manifest_tools

        for parsed_call in parsed.tool_calls:
            tool_class = active_tools.get(parsed_call.tool_name)
            if not tool_class:
                failed_calls.append(
                    FailedToolCall(
                        tool_name=parsed_call.tool_name,
                        call_id=parsed_call.call_id,
                        error=f"Unknown tool '{parsed_call.tool_name}'",
                    )
                )
                continue

            args_model, _ = tool_class._get_tool_args_results()
            try:
                validated_args = args_model.model_validate(parsed_call.raw_args)
                resolved_calls.append(
                    ResolvedToolCall(
                        tool_name=parsed_call.tool_name,
                        tool_class=tool_class,
                        validated_args=validated_args,
                        call_id=parsed_call.call_id,
                    )
                )
            except ValidationError as e:
                failed_calls.append(
                    FailedToolCall(
                        tool_name=parsed_call.tool_name,
                        call_id=parsed_call.call_id,
                        error=f"Invalid arguments: {e}",
                    )
                )

        return ResolvedMessage(tool_calls=resolved_calls, failed_calls=failed_calls)

    def create_tool_response_message(
        self, tool_call: ResolvedToolCall, result_text: str
    ) -> LLMMessage:
        return LLMMessage(
            role=Role.TOOL,
            tool_call_id=tool_call.call_id,
            name=tool_call.tool_name,
            content=result_text,
        )

    def create_failed_tool_response_message(
        self, failed: FailedToolCall, error_content: str
    ) -> LLMMessage:
        return LLMMessage(
            role=Role.TOOL,
            tool_call_id=failed.call_id,
            name=failed.tool_name,
            content=error_content,
        )
