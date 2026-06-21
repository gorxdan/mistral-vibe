from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import httpx
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Horizontal, Vertical

from vibe.cli.textual_ui.widgets.banner.petit_chat import PetitChat
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.config import ModelConfig
from vibe.core.llm.model_discovery import candidate_local_providers
from vibe.core.logger import logger
from vibe.setup.onboarding.base import OnboardingScreen
from vibe.setup.onboarding.provider_presets import (
    PRESETS,
    ProviderPreset,
    apply_provider_config,
)

if TYPE_CHECKING:
    from vibe.setup.onboarding import OnboardingApp

OLLAMA_PROBE_TIMEOUT = 2.0
OLLAMA_ALIAS = "ollama"
OLLAMA_NOT_FOUND_HINT = (
    "No Ollama server found. Start Ollama (ollama serve) and try again, or pick "
    "another provider."
)
NAV_HINT = "Use arrows to navigate - Enter Select - Esc Cancel"


def _default_index_for_provider(provider_name: str) -> int:
    for index, preset in enumerate(PRESETS):
        if preset.provider is not None and preset.provider.name == provider_name:
            return index
    return 0


class ProviderSelectionScreen(OnboardingScreen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, resolved_provider_name: str | None = None) -> None:
        super().__init__()
        self._selected_index = _default_index_for_provider(resolved_provider_name or "")
        self._option_markers: list[NoMarkupStatic] = []
        self._option_widgets: list[NoMarkupStatic] = []
        self._help_widget: NoMarkupStatic
        self._status_widget: NoMarkupStatic
        self._probing = False

    def compose(self) -> ComposeResult:
        with Vertical(id="provider-selection-content", classes="onboarding-content"):
            with Center():
                with Vertical(
                    id="provider-selection-panel", classes="onboarding-panel"
                ):
                    yield PetitChat(
                        id="provider-selection-chat", classes="onboarding-chat"
                    )
                    yield NoMarkupStatic(
                        "Choose your provider",
                        id="provider-selection-title",
                        classes="onboarding-heading",
                    )
                    yield NoMarkupStatic(
                        "Pick the model provider you want to use with Chaton.",
                        id="provider-selection-subtitle",
                    )
                    with Vertical(id="provider-selection-options"):
                        yield from self._compose_option_rows()
                    self._status_widget = NoMarkupStatic(
                        "", id="provider-selection-status", classes="provider-status"
                    )
                    yield self._status_widget
                    self._help_widget = NoMarkupStatic(
                        "", id="provider-selection-help", classes="onboarding-hint-row"
                    )
                    yield self._help_widget

    def _compose_option_rows(self) -> ComposeResult:
        self._option_markers = []
        self._option_widgets = []
        for _ in PRESETS:
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
        if self._probing:
            return
        self._selected_index = (self._selected_index - 1) % len(self._option_widgets)
        self._update_display()

    def action_move_down(self) -> None:
        if self._probing:
            return
        self._selected_index = (self._selected_index + 1) % len(self._option_widgets)
        self._update_display()

    def action_select(self) -> None:
        if self._probing:
            return
        preset = PRESETS[self._selected_index]
        match preset.key:
            case "custom":
                self.app.switch_screen("custom_provider")
            case "ollama":
                self._start_ollama_probe()
            case "mistral":
                cast("OnboardingApp", self.app).install_mistral_screens()
                target = (
                    "auth_method"
                    if cast("OnboardingApp", self.app).supports_browser_sign_in
                    else "api_key"
                )
                self.app.switch_screen(target)
            case _:
                self._install_keyed_preset(preset)
                self.app.switch_screen("api_key")

    def _install_keyed_preset(self, preset: ProviderPreset) -> None:
        if preset.provider is None or preset.model is None:
            return
        cast("OnboardingApp", self.app).install_api_key_screen(
            preset.provider, help_url=preset.help_url, pending_model=preset.model
        )

    def _start_ollama_probe(self) -> None:
        self._probing = True
        self._status_widget.remove_class("error")
        self._status_widget.update("Looking for a local Ollama server...")
        self.run_worker(self._probe_ollama(), group="ollama-probe", exclusive=True)

    async def _probe_ollama(self) -> None:
        provider = candidate_local_providers()[0]
        url = f"{provider.api_base.rstrip('/')}/models"
        model_name: str | None = None
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_PROBE_TIMEOUT) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
            items = data.get("data") if isinstance(data, dict) else None
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        model_name = item["id"]
                        break
        except (httpx.HTTPError, ValueError) as err:
            logger.debug("Ollama probe at %s failed: %s", url, err)

        if model_name is None:
            self._probing = False
            self._status_widget.add_class("error")
            self._status_widget.update(OLLAMA_NOT_FOUND_HINT)
            return

        model = ModelConfig(name=model_name, provider=provider.name, alias=OLLAMA_ALIAS)
        try:
            apply_provider_config(provider, model)
        except (OSError, ValueError) as err:
            self._probing = False
            self._status_widget.add_class("error")
            self._status_widget.update(f"Could not save Ollama provider: {err}")
            return

        self.app.exit("completed")

    def _update_display(self) -> None:
        for index, (marker, widget, preset) in enumerate(
            zip(self._option_markers, self._option_widgets, PRESETS, strict=True)
        ):
            is_selected = index == self._selected_index
            content = Text()
            content.append(preset.label, style="bold")
            content.append("\n")
            content.append(preset.description, style="dim")
            marker.update(">" if is_selected else "")
            marker.remove_class("selected")
            widget.border_title = preset.badge or ""
            widget.update(content)
            widget.remove_class("selected")
            if is_selected:
                marker.add_class("selected")
                widget.add_class("selected")

        self._help_widget.update(NAV_HINT)
