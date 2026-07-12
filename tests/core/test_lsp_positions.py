from __future__ import annotations

import pytest

from vibe.core.lsp._positions import (
    codepoint_column_to_utf16,
    codepoint_position_to_utf16,
    codepoint_range_to_utf16,
    split_lsp_lines,
    utf16_column_to_codepoint,
    utf16_position_to_codepoint,
    utf16_range_to_codepoint,
)
from vibe.core.lsp._types import Position, Range


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", [""]),
        ("x\n", ["x", ""]),
        ("x\r\ny", ["x", "y"]),
        ("x\ry", ["x", "y"]),
        ("x\u2028y", ["x\u2028y"]),
    ],
)
def test_split_lsp_lines_uses_protocol_line_endings(
    text: str, expected: list[str]
) -> None:
    assert split_lsp_lines(text) == expected


@pytest.mark.parametrize(
    ("codepoint", "utf16"), [(0, 0), (1, 1), (2, 3), (3, 4), (99, 4)]
)
def test_codepoint_column_to_utf16_counts_surrogate_pairs(
    codepoint: int, utf16: int
) -> None:
    assert codepoint_column_to_utf16("a😀b", codepoint) == utf16


@pytest.mark.parametrize(
    ("utf16", "codepoint"), [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (99, 3)]
)
def test_utf16_column_to_codepoint_clamps_to_codepoint_boundaries(
    utf16: int, codepoint: int
) -> None:
    assert utf16_column_to_codepoint("a😀b", utf16) == codepoint


def test_position_conversion_round_trips_astral_text() -> None:
    text = "header\n😀target\n"
    codepoint = Position(line=1, character=1)

    protocol = codepoint_position_to_utf16(text, codepoint)

    assert protocol == Position(line=1, character=2)
    assert utf16_position_to_codepoint(text, protocol) == codepoint


def test_position_conversion_accepts_empty_trailing_line() -> None:
    text = "value\n"
    position = Position(line=1, character=0)

    assert codepoint_position_to_utf16(text, position) == position


def test_position_conversion_rejects_line_outside_document() -> None:
    with pytest.raises(ValueError, match="outside document range"):
        codepoint_position_to_utf16("value", Position(line=1, character=0))


def test_range_conversion_round_trips() -> None:
    text = "😀name"
    codepoint = Range(
        start=Position(line=0, character=1), end=Position(line=0, character=5)
    )

    protocol = codepoint_range_to_utf16(text, codepoint)

    assert protocol == Range(
        start=Position(line=0, character=2), end=Position(line=0, character=6)
    )
    assert utf16_range_to_codepoint(text, protocol) == codepoint
