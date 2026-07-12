from __future__ import annotations

import re

from vibe.core.lsp._types import Position, Range, utf16_column

_LSP_LINE_ENDING = re.compile(r"\r\n|\r|\n")
_UTF16_SURROGATE_THRESHOLD = 0xFFFF


def split_lsp_lines(text: str) -> list[str]:
    return _LSP_LINE_ENDING.split(text)


def codepoint_column_to_utf16(line_text: str, character: int) -> int:
    return utf16_column(line_text, character)


def utf16_column_to_codepoint(line_text: str, character: int) -> int:
    remaining = max(0, character)
    for index, value in enumerate(line_text):
        width = 2 if ord(value) > _UTF16_SURROGATE_THRESHOLD else 1
        if remaining < width:
            return index
        remaining -= width
        if remaining == 0:
            return index + 1
    return len(line_text)


def codepoint_position_to_utf16(text: str, position: Position) -> Position:
    line_text = _line_at(text, position.line)
    return Position(
        line=position.line,
        character=codepoint_column_to_utf16(line_text, position.character),
    )


def utf16_position_to_codepoint(text: str, position: Position) -> Position:
    line_text = _line_at(text, position.line)
    return Position(
        line=position.line,
        character=utf16_column_to_codepoint(line_text, position.character),
    )


def codepoint_range_to_utf16(text: str, range_: Range) -> Range:
    return Range(
        start=codepoint_position_to_utf16(text, range_.start),
        end=codepoint_position_to_utf16(text, range_.end),
    )


def utf16_range_to_codepoint(text: str, range_: Range) -> Range:
    return Range(
        start=utf16_position_to_codepoint(text, range_.start),
        end=utf16_position_to_codepoint(text, range_.end),
    )


def _line_at(text: str, line: int) -> str:
    lines = split_lsp_lines(text)
    if line < 0 or line >= len(lines):
        raise ValueError(f"line {line} is outside document range 0..{len(lines) - 1}")
    return lines[line]


__all__ = [
    "codepoint_column_to_utf16",
    "codepoint_position_to_utf16",
    "codepoint_range_to_utf16",
    "split_lsp_lines",
    "utf16_column_to_codepoint",
    "utf16_position_to_codepoint",
    "utf16_range_to_codepoint",
]
