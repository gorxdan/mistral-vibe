from __future__ import annotations

from typing import Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from vibe.cli.textual_ui.shortcut_hints import shortcut, shortcut_hint
from vibe.cli.textual_ui.widgets.navigable_option_list import NavigableOptionList
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput

_FILTER_INPUT_ID = "modelpicker-filter"


def _build_option_text(label: str, alias: str, is_current: bool) -> Text:
    text = Text(no_wrap=True)
    marker = "› " if is_current else "  "
    style = "bold" if is_current else ""
    text.append(marker, style="green" if is_current else "")
    # Primary label is the provider's API model name; the friendly alias (the
    # value persisted as active_model) is shown dim alongside when it differs.
    text.append(label, style=style)
    if alias != label:
        text.append(f"  · {alias}", style="dim")
    return text


def _build_provider_text(provider: str) -> Text:
    text = Text(no_wrap=True)
    text.append(f"  Provider: {provider}", style="dim bold")
    return text


class ModelPickerApp(Container):
    """Model picker bottom app for selecting the active model.

    Large providers (OpenRouter, OpenAI, ...) live-discover hundreds of models,
    so the list is filterable: type to narrow by model name, alias, or provider.
    ``up``/``down``/``enter`` keep working because the nav/select bindings are
    priority-matched ahead of the filter input's own key handling.
    """

    can_focus_children = True

    # Priority bindings fire before the focused filter input's key handler, so
    # arrow / enter / escape drive the option list even while typing a filter.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("up", "mp_up", show=False, priority=True),
        Binding("down", "mp_down", show=False, priority=True),
        Binding("pageup", "mp_page_up", show=False, priority=True),
        Binding("pagedown", "mp_page_down", show=False, priority=True),
        Binding("enter", "mp_select", show=False, priority=True),
    ]

    class ModelSelected(Message):
        def __init__(self, alias: str) -> None:
            self.alias = alias
            super().__init__()

    class Cancelled(Message):
        pass

    def __init__(
        self,
        model_aliases: list[str],
        current_model: str,
        *,
        display_names: dict[str, str] | None = None,
        footer_hint: str | None = None,
        providers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="modelpicker-app", **kwargs)
        self._model_aliases = model_aliases
        self._current_model = current_model
        # alias -> API model name. Aliases without a mapping fall back to showing
        # the alias itself (so the widget stays usable with bare alias lists).
        self._display_names = display_names or {}
        self._footer_hint = footer_hint
        self._providers = providers or {}
        # Live filter substring; empty shows every model (the default view).
        self._filter = ""

    def _matches_filter(self, alias: str) -> bool:
        needle = self._filter.strip().lower()
        if not needle:
            return True
        label = self._display_names.get(alias, alias)
        provider = self._providers.get(alias, "")
        return needle in f"{label} {alias} {provider}".lower()

    def _build_options(self) -> list[Option]:
        options: list[Option] = []
        last_provider: str | None = None
        for alias in self._model_aliases:
            if not self._matches_filter(alias):
                continue
            provider = self._providers.get(alias, "Other")
            if self._providers and provider != last_provider:
                options.append(Option(_build_provider_text(provider), disabled=True))
                last_provider = provider
            options.append(
                Option(
                    _build_option_text(
                        self._display_names.get(alias, alias),
                        alias,
                        alias == self._current_model,
                    ),
                    id=alias,
                )
            )
        return options

    def _restore_highlight(self) -> None:
        option_list = self.query_one(OptionList)
        for i, option in enumerate(option_list.options):
            if option.id == self._current_model:
                option_list.highlighted = i
                return
        # Current model filtered out (or none) — land on the first selectable row.
        for i, option in enumerate(option_list.options):
            if not option.disabled:
                option_list.highlighted = i
                return

    def _render_options(self) -> None:
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_options(self._build_options())
        self._restore_highlight()

    def compose(self) -> ComposeResult:
        with Vertical(id="modelpicker-content"):
            yield NoMarkupStatic("Select Model", classes="modelpicker-title")
            yield VscodeCompatInput(placeholder="Filter models…", id=_FILTER_INPUT_ID)
            yield NavigableOptionList(*self._build_options(), id="modelpicker-options")
            if self._footer_hint:
                yield NoMarkupStatic(self._footer_hint, classes="modelpicker-hint")
            yield NoMarkupStatic(
                shortcut_hint(
                    f"Type to filter  {shortcut('↑↓')} Navigate  "
                    f"{shortcut('Enter')} Select  {shortcut('Esc')} Cancel"
                ),
                classes="modelpicker-help",
            )

    def on_mount(self) -> None:
        self._restore_highlight()
        self.query_one(f"#{_FILTER_INPUT_ID}", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != _FILTER_INPUT_ID:
            return
        self._filter = event.value
        self._render_options()

    def action_mp_up(self) -> None:
        self.query_one(OptionList).action_cursor_up()

    def action_mp_down(self) -> None:
        self.query_one(OptionList).action_cursor_down()

    def action_mp_page_up(self) -> None:
        self.query_one(OptionList).action_page_up()

    def action_mp_page_down(self) -> None:
        self.query_one(OptionList).action_page_down()

    def action_mp_select(self) -> None:
        option_list = self.query_one(OptionList)
        option = option_list.highlighted_option
        if option is not None and option.id:
            self.post_message(self.ModelSelected(option.id))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # Reached when the option list itself has focus and handles Enter (e.g.
        # click-to-focus then Enter). The priority binding above covers the
        # keyboard path while the filter input is focused.
        if event.option.id:
            self.post_message(self.ModelSelected(event.option.id))

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())
