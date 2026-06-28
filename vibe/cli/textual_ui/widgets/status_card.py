from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from vibe.cli.textual_ui.widgets._status_render import (
    StatusCardData,
    render_status_card,
)


class StatusCard(Static):
    """Renders the rich /status card (Codex-style) into the chat stream."""

    def __init__(self, data: StatusCardData) -> None:
        super().__init__()
        self.add_class("status-card")
        self._data = data

    def render(self) -> Text:
        return render_status_card(self._data)
