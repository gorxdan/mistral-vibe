from __future__ import annotations

from rich.cells import cell_len

from vibe.cli.textual_ui.widgets.chat_input.completion_popup import CompletionPopup


def _rendered_text_length(label: str, description: str) -> int:
    # Mirrors how the popup sizes a row: the displayed (prefix-stripped) label
    # plus the description, each measured in terminal cells, with a 2-cell
    # separator when a description is present.
    length = cell_len(CompletionPopup._display_label(label)) + cell_len(description)
    if description:
        length += 2
    return length


def test_rendered_text_length_uses_terminal_cell_width() -> None:
    # "你" and "🙂" both occupy 2 terminal cells in Rich (+2 for separator).
    assert _rendered_text_length("@你", "🙂") == 6


def test_rendered_text_length_keeps_description_separator() -> None:
    assert _rendered_text_length("@abc", "def") == 8
