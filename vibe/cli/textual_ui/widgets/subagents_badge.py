from __future__ import annotations

from collections import Counter
from typing import Any

from textual.reactive import reactive
from textual.timer import Timer

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.cli.textual_ui.widgets.spinner import create_spinner

_MAX_NAMES = 3
_MAX_BODY = 26
_FRAME_INTERVAL_S = 0.1


class SubagentsBadge(NoMarkupStatic):
    running: reactive[tuple[str, ...]] = reactive(())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._spinner = create_spinner()
        self._timer: Timer | None = None
        self._body = ""

    def watch_running(self, names: tuple[str, ...]) -> None:
        if not names:
            self._body = ""
            self._stop_timer()
            self.update("")
            return
        self._body = self._format_body(names)
        self._start_timer()
        self._render_frame()

    def on_mount(self) -> None:
        if self._body:
            self._start_timer()

    def on_unmount(self) -> None:
        self._stop_timer()

    def _format_body(self, names: tuple[str, ...]) -> str:
        counts = Counter(names)
        if len(counts) > _MAX_NAMES:
            return f"{len(names)} agents"
        parts = [name if n == 1 else f"{name} ×{n}" for name, n in counts.items()]
        body = ", ".join(parts)
        if len(body) > _MAX_BODY:
            return f"{len(names)} agents"
        return body

    def _render_frame(self) -> None:
        if not self._body:
            return
        self.update(f"{self._spinner.next_frame()} {self._body}")

    def _start_timer(self) -> None:
        if self._timer is None and self.is_mounted:
            self._timer = self.set_interval(_FRAME_INTERVAL_S, self._render_frame)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
