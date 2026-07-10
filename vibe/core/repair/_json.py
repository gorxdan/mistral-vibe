from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import orjson

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_RAW_PREVIEW_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class JsonObjectRepair:
    value: dict[str, Any] | None
    raw_text: str
    error: str | None = None
    actual_type: str | None = None
    repaired: bool = False


def repair_json_object(raw: str) -> JsonObjectRepair:
    first_error: orjson.JSONDecodeError | None = None
    for candidate in _repair_candidates(raw):
        try:
            value = orjson.loads(candidate)
        except orjson.JSONDecodeError as exc:
            if first_error is None:
                first_error = exc
            continue
        if not isinstance(value, dict):
            return JsonObjectRepair(
                value=None,
                raw_text=_bounded_raw(raw),
                error="JSON value is not an object",
                actual_type=type(value).__name__,
            )
        return JsonObjectRepair(
            value=value, raw_text=_bounded_raw(raw), repaired=candidate != raw
        )

    detail = str(first_error) if first_error is not None else "invalid JSON"
    return JsonObjectRepair(value=None, raw_text=_bounded_raw(raw), error=detail)


def _repair_candidates(text: str) -> tuple[str, ...]:
    candidates: list[str] = [text]
    fenced = tuple(_FENCE_RE.finditer(text))
    if len(fenced) == 1:
        outside_fence = text[: fenced[0].start()] + text[fenced[0].end() :]
        if not any(marker in outside_fence for marker in "{}[]"):
            candidates.append(fenced[0].group(1).strip())
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
                outside = text[:start] + text[index + 1 :]
                if any(marker in outside for marker in "{}[]"):
                    return None
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


__all__ = ["JsonObjectRepair", "repair_json_object"]
