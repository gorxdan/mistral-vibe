from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from vibe.core.llm.models import (
    FailedToolCall,
    ParsedMessage,
    ParsedToolCall,
    ResolvedMessage,
    ResolvedToolCall,
)
from vibe.core.llm.tool_call_repair import (
    parse_tool_arguments,
    tool_argument_schema_diagnostic,
)
from vibe.core.logger import logger
from vibe.core.types import (
    AvailableFunction,
    AvailableTool,
    LLMMessage,
    Role,
    StrToolChoice,
)
from vibe.core.utils import first_sentence

if TYPE_CHECKING:
    from vibe.core.tools.manager import ToolManager


def _trim_description(description: str, max_chars: int) -> str:
    return first_sentence(description, max_chars)


# logger.isEnabledFor(10) is always True (logger.py forces DEBUG); gate on env instead.
_TELEMETRY_DEBUG = os.environ.get("DEBUG_MODE") == "true" or (
    os.environ.get("LOG_LEVEL", "WARNING").upper() == "DEBUG"
)


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
        stub_suffix = self._deferred_stub_suffix(tool_manager)
        tools: list[AvailableTool] = []
        for tool_class in tool_manager.manifest_tools.values():
            description = (
                _trim_description(tool_class.description, description_max_chars)
                if trim_descriptions
                else tool_class.description
            )
            # Trim-then-append: a small-tier first-sentence trim must not eat
            # the deferred-tool stub listing.
            if stub_suffix and tool_class.get_name() == "tool_search":
                description = f"{description}{stub_suffix}"
            tools.append(
                AvailableTool(
                    function=AvailableFunction(
                        name=tool_class.get_name(),
                        description=description,
                        parameters=tool_class.get_parameters(),
                    )
                )
            )
        # debug-only: scope namespace-grouping necessity by manifest size.
        if _TELEMETRY_DEBUG:
            hidden = len(tool_manager.hidden_tool_names())
            logger.debug(
                "manifest_tools size=%s hidden=%s has_remote=%s",
                len(tools),
                hidden,
                hidden > 0
                or any(t.function.name.startswith(("mcp", "conn")) for t in tools),
            )
        return tools

    @staticmethod
    def _deferred_stub_suffix(tool_manager: ToolManager) -> str:
        stubs = tool_manager.deferred_builtin_stubs()
        if not stubs:
            return ""
        listing = "; ".join(f"`{name}` — {summary}" for name, summary in stubs)
        return f" Hidden builtin tools you can activate here: {listing}"

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
            parsed_args = parse_tool_arguments(function_call.arguments)

            tool_calls.append(
                ParsedToolCall(
                    tool_name=function_call.name or "",
                    raw_args=parsed_args.arguments,
                    raw_text=parsed_args.raw_text,
                    parse_error=parsed_args.diagnostic,
                    repaired=parsed_args.repaired,
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
            if parsed_call.parse_error is not None:
                failed_calls.append(
                    FailedToolCall(
                        tool_name=parsed_call.tool_name,
                        call_id=parsed_call.call_id,
                        error=parsed_call.parse_error.for_model(),
                        diagnostic=parsed_call.parse_error,
                    )
                )
                continue
            tool_class = active_tools.get(parsed_call.tool_name)
            if not tool_class:
                # tool_search itself can be hidden (small MCP catalogs); the
                # activation hint would then send the model in a circle.
                hint = (
                    " — it exists but is not activated; call tool_search with "
                    "its name first"
                    if parsed_call.tool_name != "tool_search"
                    and parsed_call.tool_name in tool_manager.hidden_tool_names()
                    else ""
                )
                failed_calls.append(
                    FailedToolCall(
                        tool_name=parsed_call.tool_name,
                        call_id=parsed_call.call_id,
                        error=f"Unknown tool '{parsed_call.tool_name}'{hint}",
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
                diagnostic = tool_argument_schema_diagnostic(parsed_call.tool_name, e)
                failed_calls.append(
                    FailedToolCall(
                        tool_name=parsed_call.tool_name,
                        call_id=parsed_call.call_id,
                        error=diagnostic.for_model(),
                        diagnostic=diagnostic,
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
