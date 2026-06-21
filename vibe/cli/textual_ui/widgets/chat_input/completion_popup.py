from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from vibe.core.autocompletion.menu import MenuRow, RowKind

COMPLETION_POPUP_MAX_HEIGHT = 12
COMPLETION_POPUP_MAX_WIDTH = 80
COMPLETION_POPUP_PADDING_X = 1

# Section labels ("COMMANDS"/"SKILLS") and the "+N more" footer are decoration:
# muted so the selectable rows stand out, and never given a reverse highlight so
# selection-detection (which keys off the reverse style) ignores them.
_HEADER_STYLE = "dim bold"
_HINT_STYLE = "dim italic"


class _CompletionItem(Static):
    pass


class CompletionPopup(VerticalScroll):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="completion-popup", **kwargs)
        self.styles.display = "none"
        self.styles.max_height = COMPLETION_POPUP_MAX_HEIGHT
        self.styles.max_width = COMPLETION_POPUP_MAX_WIDTH
        self.styles.padding = (0, COMPLETION_POPUP_PADDING_X)
        self.can_focus = False

    def update_suggestions(
        self, suggestions: list[tuple[str, str]], selected: int
    ) -> None:
        if not suggestions:
            self.hide()
            return

        self.remove_children()

        items: list[_CompletionItem] = []
        for idx, (label, description) in enumerate(suggestions):
            text = Text()
            label_style = "bold reverse" if idx == selected else "bold"
            description_style = "italic" if idx == selected else "dim"

            text.append(self._display_label(label), style=label_style)
            if description:
                text.append("  ")
                text.append(description, style=description_style)

            item = _CompletionItem(text)
            items.append(item)

        self.mount_all(items)
        self.styles.display = "block"

        if 0 <= selected < len(items):
            items[selected].scroll_visible(animate=False)

    def update_menu(self, rows: list[MenuRow], selected: int) -> None:
        if not rows:
            self.hide()
            return

        self.remove_children()

        items: list[_CompletionItem] = []
        for idx, row in enumerate(rows):
            items.append(
                _CompletionItem(self._render_row(row, selected=idx == selected))
            )

        self.mount_all(items)
        self.styles.display = "block"

        if 0 <= selected < len(items):
            items[selected].scroll_visible(animate=False)

    @classmethod
    def _render_row(cls, row: MenuRow, *, selected: bool) -> Text:
        text = Text()
        if row.kind is RowKind.HEADER:
            text.append(row.text, style=_HEADER_STYLE)
            return text
        if row.kind is RowKind.HINT:
            text.append(row.text, style=_HINT_STYLE)
            return text

        label_style = "bold reverse" if selected else "bold"
        description_style = "italic" if selected else "dim"
        text.append(cls._display_label(row.text), style=label_style)
        if row.description:
            text.append("  ")
            text.append(row.description, style=description_style)
        return text

    def hide(self) -> None:
        self.remove_children()
        self.styles.display = "none"

    @property
    def content_text(self) -> str:
        return "\n".join(str(child.render()) for child in self.query(_CompletionItem))

    @staticmethod
    def _display_label(label: str) -> str:
        if label.startswith("@"):
            return label[1:]
        return label

    @classmethod
    def rendered_text_length(cls, label: str, description: str) -> int:
        text_length = cell_len(cls._display_label(label)) + cell_len(description)
        if description:
            text_length += 2
        return text_length
