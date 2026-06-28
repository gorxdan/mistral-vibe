from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.widgets import Static

from vibe.cli.textual_ui.widgets._status_render import render_status_card
from vibe.core.types import AgentStats
from vibe.core.usage import RateLimitSnapshot, UsageSummary


class StatusCard(Static):
    """Renders the rich /status card (Codex-style) into the chat stream."""

    def __init__(
        self,
        *,
        stats: AgentStats,
        summary: UsageSummary,
        version: str,
        model_name: str,
        provider_name: str,
        workdir: Path,
        session_id: str,
        context_window: int | None = None,
        rate_limits: dict[str, RateLimitSnapshot] | None = None,
        width: int = 72,
    ) -> None:
        super().__init__()
        self.add_class("status-card")
        self._stats = stats
        self._summary = summary
        self._version = version
        self._model_name = model_name
        self._provider_name = provider_name
        self._workdir = workdir
        self._session_id = session_id
        self._context_window = context_window
        self._rate_limits = rate_limits
        self._width = width

    def render(self) -> Text:
        return render_status_card(
            stats=self._stats,
            summary=self._summary,
            version=self._version,
            model_name=self._model_name,
            provider_name=self._provider_name,
            workdir=self._workdir,
            session_id=self._session_id,
            context_window=self._context_window,
            rate_limits=self._rate_limits,
            width=self._width,
        )
