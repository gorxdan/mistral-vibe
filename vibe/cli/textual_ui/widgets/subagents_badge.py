from __future__ import annotations

from textual.reactive import reactive

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic

_MAX_NAMES = 3


class SubagentsBadge(NoMarkupStatic):
    running: reactive[tuple[str, ...]] = reactive(())

    def watch_running(self, names: tuple[str, ...]) -> None:
        if not names:
            self.update("")
            return
        shown = ", ".join(names[:_MAX_NAMES])
        extra = len(names) - _MAX_NAMES
        if extra > 0:
            shown += f" +{extra}"
        self.update(f"⟦⧖ {shown}⟧")
