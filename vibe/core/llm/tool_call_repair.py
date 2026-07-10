from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from vibe.core.failure_diagnostic import (
    FailureCategory,
    FailureDiagnostic,
    build_failure_diagnostic,
)
from vibe.core.repair import repair_json_object


@dataclass(frozen=True, slots=True)
class ToolArgumentParse:
    arguments: dict[str, Any]
    raw_text: str
    diagnostic: FailureDiagnostic | None = None
    repaired: bool = False


def parse_tool_arguments(raw: str | None) -> ToolArgumentParse:
    text = raw or "{}"
    repaired = repair_json_object(text)
    if repaired.value is not None:
        return ToolArgumentParse(
            repaired.value, repaired.raw_text, repaired=repaired.repaired
        )
    if repaired.actual_type is not None:
        diagnostic = build_failure_diagnostic(
            category=FailureCategory.TOOL_ARGUMENT_PARSE,
            message="Tool arguments must decode to a JSON object",
            field="arguments",
            expected="object",
            actual=repaired.actual_type,
            evidence_pointer="tool_call.arguments",
            suggested_action="emit one JSON object containing the tool fields",
        )
        return ToolArgumentParse({}, repaired.raw_text, diagnostic)
    diagnostic = build_failure_diagnostic(
        category=FailureCategory.TOOL_ARGUMENT_PARSE,
        message=f"Malformed tool argument JSON: {repaired.error or 'invalid JSON'}",
        field="arguments",
        expected="valid JSON object",
        actual=repaired.raw_text,
        evidence_pointer="tool_call.arguments",
        suggested_action="correct only the JSON syntax and retry the same tool call",
    )
    return ToolArgumentParse({}, repaired.raw_text, diagnostic)


def tool_argument_schema_diagnostic(
    tool_name: str, error: ValidationError
) -> FailureDiagnostic:
    issues = error.errors(include_url=False)
    first = issues[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "arguments"
    details = "; ".join(
        f"{'.'.join(str(part) for part in issue.get('loc', ())) or 'arguments'}: "
        f"{issue.get('msg', 'invalid value')}"
        for issue in issues
    )
    issue_type = str(first.get("type", "validation_error"))
    context = first.get("ctx")
    expected = (
        str(context.get("expected"))
        if isinstance(context, dict) and context.get("expected") is not None
        else str(first.get("msg", "value matching the tool schema"))
    )
    actual = "missing" if issue_type == "missing" else repr(first.get("input"))[:1_000]
    return build_failure_diagnostic(
        category=FailureCategory.TOOL_ARGUMENT_SCHEMA,
        message=f"Invalid arguments for tool '{tool_name}': {details}",
        field=f"arguments.{location}",
        expected=expected,
        actual=actual,
        evidence_pointer=f"tool_call.arguments.{location}",
        suggested_action=(
            f"correct only arguments.{location} and retry the same tool call"
        ),
    )


__all__ = [
    "ToolArgumentParse",
    "parse_tool_arguments",
    "tool_argument_schema_diagnostic",
]
