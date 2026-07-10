from __future__ import annotations

from tests.conftest import build_test_vibe_config
from vibe.core.failure_diagnostic import FailureCategory
from vibe.core.llm.format import APIToolFormatHandler
from vibe.core.tools.manager import ToolManager
from vibe.core.types import FunctionCall, LLMMessage, Role, ToolCall


def _message(arguments: str) -> LLMMessage:
    return LLMMessage(
        role=Role.ASSISTANT,
        tool_calls=[
            ToolCall(
                id="call-1",
                index=0,
                function=FunctionCall(name="bash", arguments=arguments),
            )
        ],
    )


def test_malformed_arguments_preserve_raw_text_and_parser_error() -> None:
    [parsed] = (
        APIToolFormatHandler().parse_message(_message('{"command": "ls"')).tool_calls
    )

    assert parsed.raw_text == '{"command": "ls"'
    assert parsed.parse_error is not None
    assert parsed.parse_error.category == FailureCategory.TOOL_ARGUMENT_PARSE
    assert "line 1" in parsed.parse_error.message
    assert parsed.raw_args == {}


def test_malformed_arguments_resolve_to_exact_failure() -> None:
    config = build_test_vibe_config(enabled_tools=["bash"])
    manager = ToolManager(lambda: config, defer_mcp=True)
    parsed = APIToolFormatHandler().parse_message(_message('{"command":'))

    resolved = APIToolFormatHandler().resolve_tool_calls(parsed, manager)

    assert resolved.tool_calls == []
    [failed] = resolved.failed_calls
    assert failed.diagnostic is not None
    assert "Malformed tool argument JSON" in failed.error
    assert "Expected: valid JSON object" in failed.error
    assert '{"command":' in failed.error
    assert "Next action" in failed.error


def test_unambiguous_local_repairs_do_not_invent_fields() -> None:
    raw = 'prefix ```json\n{"command": "pwd",}\n``` suffix'
    [parsed] = APIToolFormatHandler().parse_message(_message(raw)).tool_calls

    assert parsed.raw_args == {"command": "pwd"}
    assert parsed.parse_error is None
    assert parsed.repaired is True
    assert parsed.raw_text == raw


def test_non_object_arguments_report_expected_shape() -> None:
    [parsed] = APIToolFormatHandler().parse_message(_message('["pwd"]')).tool_calls

    assert parsed.parse_error is not None
    assert parsed.parse_error.expected == "object"
    assert parsed.parse_error.actual == "list"


def test_multiple_json_objects_are_not_repaired_to_the_first() -> None:
    raw = '{"command":"echo first"} {"command":"echo second"}'

    [parsed] = APIToolFormatHandler().parse_message(_message(raw)).tool_calls

    assert parsed.raw_args == {}
    assert parsed.parse_error is not None
    assert parsed.repaired is False
    assert parsed.raw_text == raw


def test_tool_argument_schema_failure_is_structured() -> None:
    config = build_test_vibe_config(enabled_tools=["bash"])
    manager = ToolManager(lambda: config, defer_mcp=True)
    parsed = APIToolFormatHandler().parse_message(_message('{"command": 42}'))

    resolved = APIToolFormatHandler().resolve_tool_calls(parsed, manager)

    [failed] = resolved.failed_calls
    assert failed.diagnostic is not None
    assert failed.diagnostic.category == FailureCategory.TOOL_ARGUMENT_SCHEMA
    assert failed.diagnostic.field == "arguments.command"
    assert "valid string" in failed.error
    assert "correct only arguments.command" in failed.error
