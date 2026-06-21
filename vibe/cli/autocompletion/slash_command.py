from __future__ import annotations

from textual import events

from vibe.cli.autocompletion.base import CompletionResult, CompletionView
from vibe.core.autocompletion.completers import CommandCompleter
from vibe.core.autocompletion.menu import (
    DEFAULT_SKILL_PREVIEW_CAP,
    MenuRow,
    build_menu_rows,
    first_selectable_index,
)


class SlashCommandController:
    def __init__(
        self,
        completer: CommandCompleter,
        view: CompletionView,
        *,
        skill_cap: int = DEFAULT_SKILL_PREVIEW_CAP,
    ) -> None:
        self._completer = completer
        self._view = view
        self._rows: list[MenuRow] = []
        self._selected_index = 0
        self._skill_cap = skill_cap

    def can_handle(self, text: str, cursor_index: int) -> bool:
        return text.startswith("/")

    def reset(self) -> None:
        if self._rows:
            self._rows = []
            self._selected_index = 0
            self._view.clear_completion_suggestions()

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        if cursor_index < 0 or cursor_index > len(text):
            self.reset()
            return

        if not self.can_handle(text, cursor_index):
            self.reset()
            return

        entries = self._completer.get_menu_entries(text, cursor_index)
        if not entries:
            self.reset()
            return

        query_empty = self._completer.head_query(text, cursor_index) == ""
        rows = build_menu_rows(
            entries, query_empty=query_empty, skill_cap=self._skill_cap
        )
        first = first_selectable_index(rows)
        if first is None:
            self.reset()
            return

        self._rows = rows
        self._selected_index = first
        self._view.render_slash_menu(rows, first)

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        if not self._rows:
            return CompletionResult.IGNORED

        match event.key:
            case "tab":
                if self._apply_selected_completion(text, cursor_index):
                    result = CompletionResult.HANDLED
                else:
                    result = CompletionResult.IGNORED
            case "enter":
                if self._apply_selected_completion(text, cursor_index):
                    result = CompletionResult.SUBMIT
                else:
                    result = CompletionResult.HANDLED
            case "down":
                self._move_selection(1)
                result = CompletionResult.HANDLED
            case "up":
                self._move_selection(-1)
                result = CompletionResult.HANDLED
            case _:
                result = CompletionResult.IGNORED

        return result

    def _selectable_indices(self) -> list[int]:
        return [i for i, row in enumerate(self._rows) if row.selectable]

    def _move_selection(self, delta: int) -> None:
        indices = self._selectable_indices()
        if not indices:
            return

        try:
            position = indices.index(self._selected_index)
        except ValueError:
            position = 0
        position = (position + delta) % len(indices)
        self._selected_index = indices[position]
        self._view.render_slash_menu(self._rows, self._selected_index)

    def _apply_selected_completion(self, text: str, cursor_index: int) -> bool:
        if not self._rows:
            return False

        row = self._rows[self._selected_index]
        if not row.selectable:
            return False

        replacement_range = self._completer.get_replacement_range(text, cursor_index)
        if replacement_range is None:
            self.reset()
            return False

        start, end = replacement_range
        self._view.replace_completion_range(start, end, row.text)
        self.reset()
        return True
