from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import orjson

from vibe.core.failure_diagnostic import (
    FailureCategory,
    FailureDiagnostic,
    build_failure_diagnostic,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_RAW_PREVIEW_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class ToolArgumentParse:
    arguments: dict[str, Any]
    raw_text: str
    diagnostic: FailureDiagnostic | None = None
    repaired: bool = False


def parse_tool_arguments(raw: str | None) -> ToolArgumentParse:
    text = raw or "{}"
    first_error: orjson.JSONDecodeError | None = None
    for candidate in _repair_candidates(text):
        try:
            value = orjson.loads(candidate)
        except orjson.JSONDecodeError as exc:
            if first_error is None:
                first_error = exc
            continue
        if not isinstance(value, dict):
            diagnostic = build_failure_diagnostic(
                category=FailureCategory.TOOL_ARGUMENT_PARSE,
                message="Tool arguments must decode to a JSON object",
                field="arguments",
                expected="object",
                actual=type(value).__name__,
                evidence_pointer="tool_call.arguments",
                suggested_action="emit one JSON object containing the tool fields",
            )
            return ToolArgumentParse({}, _bounded_raw(text), diagnostic)
        return ToolArgumentParse(value, _bounded_raw(text), repaired=candidate != text)

    detail = str(first_error) if first_error is not None else "invalid JSON"
    diagnostic = build_failure_diagnostic(
        category=FailureCategory.TOOL_ARGUMENT_PARSE,
        message=f"Malformed tool argument JSON: {detail}",
        field="arguments",
        expected="valid JSON object",
        actual=_bounded_raw(text),
        evidence_pointer="tool_call.arguments",
        suggested_action="correct only the JSON syntax and retry the same tool call",
    )
    return ToolArgumentParse({}, _bounded_raw(text), diagnostic)


def _repair_candidates(text: str) -> tuple[str, ...]:
    candidates: list[str] = [text]
    candidates.extend(match.group(1).strip() for match in _FENCE_RE.finditer(text))
    if balanced := _first_balanced_object(text):
        candidates.append(balanced)
    for candidate in tuple(candidates):
        repaired = _remove_trailing_commas(candidate)
        if repaired != candidate:
            candidates.append(repaired)
    return tuple(dict.fromkeys(candidates))


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _remove_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        result.append(char)
        index += 1
    return "".join(result)


def _bounded_raw(text: str) -> str:
    if len(text) <= _RAW_PREVIEW_CHARS:
        return text
    omitted = len(text) - _RAW_PREVIEW_CHARS
    return f"{text[:_RAW_PREVIEW_CHARS]}...[{omitted} chars omitted]"
