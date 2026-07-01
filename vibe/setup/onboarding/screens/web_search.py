from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Horizontal, Vertical

from vibe.cli.textual_ui.widgets.banner.petit_chat import PetitChat
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.config import MissingAPIKeyError, VibeConfig
from vibe.core.logger import logger
from vibe.core.search import DEFAULT_PORT, default_url, detect_engine
from vibe.setup.onboarding.base import OnboardingScreen

NAV_HINT = "Use arrows to navigate - Enter Select - Esc Cancel"

MISTRAL_KEY = "mistral"
SEARXNG_KEY = "searxng"


@dataclass(frozen=True)
class _Option:
    key: str
    label: str
    description: str


_OPTIONS: tuple[_Option, ...] = (
    _Option(
        key=MISTRAL_KEY,
        label="Mistral web search",
        description="Default. Searches the web through Mistral. No setup required.",
    ),
    _Option(
        key=SEARXNG_KEY,
        label="Local SearXNG (private)",
        description="Self-hosted metasearch. vibe can start/stop the container for you.",
    ),
)


class WebSearchScreen(OnboardingScreen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    NEXT_SCREEN = "provider_selection"

    def __init__(self, next_screen: str = "provider_selection") -> None:
        super().__init__()
        self.NEXT_SCREEN = next_screen
        self._selected_index = 0
        self._option_markers: list[NoMarkupStatic] = []
        self._option_widgets: list[NoMarkupStatic] = []
        self._status_widget: NoMarkupStatic
        self._help_widget: NoMarkupStatic

    def compose(self) -> ComposeResult:
        with Vertical(id="web-search-content", classes="onboarding-content"):
            with Center():
                with Vertical(id="web-search-panel", classes="onboarding-panel"):
                    yield PetitChat(id="web-search-chat", classes="onboarding-chat")
                    yield NoMarkupStatic(
                        "Choose a web search backend",
                        id="web-search-title",
                        classes="onboarding-heading",
                    )
                    yield NoMarkupStatic(
                        "How should Mistral Vibe search the web? You can change this later "
                        "in config.",
                        id="web-search-subtitle",
                    )
                    with Vertical(id="web-search-options"):
                        yield from self._compose_option_rows()
                    self._status_widget = NoMarkupStatic(
                        "", id="web-search-status", classes="provider-status"
                    )
                    yield self._status_widget
                    self._help_widget = NoMarkupStatic(
                        "", id="web-search-help", classes="onboarding-hint-row"
                    )
                    yield self._help_widget

    def _compose_option_rows(self) -> ComposeResult:
        self._option_markers = []
        self._option_widgets = []
        for _ in _OPTIONS:
            with Horizontal(classes="provider-option-row onboarding-option-row"):
                marker = NoMarkupStatic("", classes="provider-option-marker")
                option = NoMarkupStatic("", classes="provider-option onboarding-card")
                self._option_markers.append(marker)
                self._option_widgets.append(option)
                yield marker
                yield option

    def on_mount(self) -> None:
        self._update_display()
        self.focus()

    def action_move_up(self) -> None:
        self._selected_index = (self._selected_index - 1) % len(_OPTIONS)
        self._update_display()

    def action_move_down(self) -> None:
        self._selected_index = (self._selected_index + 1) % len(_OPTIONS)
        self._update_display()

    def action_select(self) -> None:
        option = _OPTIONS[self._selected_index]
        if option.key == SEARXNG_KEY:
            self._persist_searxng_choice()
        super().action_next()

    def _persist_searxng_choice(self) -> None:
        try:
            VibeConfig.save_updates({
                "tools": {
                    "web_search": {
                        "searxng_url": default_url(DEFAULT_PORT),
                        "searxng_manage": True,
                        "searxng_port": DEFAULT_PORT,
                        "searxng_autostart": True,
                        "searxng_stop_on_exit": True,
                    }
                }
            })
        except (OSError, MissingAPIKeyError) as err:
            logger.warning("Failed to persist SearXNG web-search choice: %s", err)

    def _update_display(self) -> None:
        for index, (marker, widget, option) in enumerate(
            zip(self._option_markers, self._option_widgets, _OPTIONS, strict=True)
        ):
            is_selected = index == self._selected_index
            content = Text()
            content.append(option.label, style="bold")
            content.append("\n")
            content.append(option.description, style="dim")
            marker.update(">" if is_selected else "")
            marker.remove_class("selected")
            widget.update(content)
            widget.remove_class("selected")
            if is_selected:
                marker.add_class("selected")
                widget.add_class("selected")

        self._update_status()
        self._help_widget.update(NAV_HINT)

    def _update_status(self) -> None:
        if _OPTIONS[self._selected_index].key != SEARXNG_KEY:
            self._status_widget.remove_class("error")
            self._status_widget.update("")
            return
        if detect_engine() is None:
            self._status_widget.add_class("error")
            self._status_widget.update(
                "No docker/podman found. Install one to let vibe run SearXNG; "
                "the setting will still be saved."
            )
        else:
            self._status_widget.remove_class("error")
            self._status_widget.update(
                f"SearXNG will start on next launch at {default_url(DEFAULT_PORT)}."
            )
