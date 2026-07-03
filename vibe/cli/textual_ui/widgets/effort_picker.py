from __future__ import annotations

from typing import Any, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.config import EffortLevel

_EFFORT_DESCRIPTIONS: dict[str, str] = {
    "normal": "Work turn-by-turn",
    "le-chaton": "Max thinking + auto-workflow planning",
}


def _build_option_text(level: str, is_current: bool) -> Text:
    text = Text(no_wrap=True)
    marker = "› " if is_current else "  "
    style = "bold" if is_current else ""
    text.append(marker, style="green" if is_current else "")
    label = level.replace("-", " ").title() if level == "le-chaton" else level.title()
    text.append(label, style=style)
    desc = _EFFORT_DESCRIPTIONS.get(level)
    if desc:
        text.append(f"  ({desc})", style="dim" if not is_current else "")
    return text


class EffortPickerApp(Container):
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False)
    ]

    class EffortSelected(Message):
        level: EffortLevel

        def __init__(self, level: EffortLevel) -> None:
            self.level = level
            super().__init__()

    class Cancelled(Message):
        pass

    def __init__(
        self, effort_levels: list[str], current_effort: str, **kwargs: Any
    ) -> None:
        super().__init__(id="effortpicker-app", **kwargs)
        self._effort_levels = effort_levels
        self._current_effort = current_effort

    def compose(self) -> ComposeResult:
        options = [
            Option(_build_option_text(level, level == self._current_effort), id=level)
            for level in self._effort_levels
        ]
        with Vertical(id="effortpicker-content"):
            yield NoMarkupStatic("Select Effort Mode", classes="effortpicker-title")
            yield NavigableOptionList(*options, id="effortpicker-options")
            yield NoMarkupStatic(
                "↑↓ Navigate  Enter Select  Esc Cancel", classes="effortpicker-help"
            )

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        for i, level in enumerate(self._effort_levels):
            if level == self._current_effort:
                option_list.highlighted = i
                break
        option_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.post_message(self.EffortSelected(cast(EffortLevel, event.option.id)))

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())
