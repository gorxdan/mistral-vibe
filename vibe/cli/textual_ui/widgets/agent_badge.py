from __future__ import annotations

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic

_ELIDE_CAP = 28


def _elide(value: str, cap: int = _ELIDE_CAP) -> str:
    if len(value) <= cap:
        return value
    return value[: cap - 1] + "…"


class ModelStatusBadge(NoMarkupStatic):
    def set_model(self, active_model: str) -> None:
        if not active_model:
            self.update("")
            return
        self.update(f"· {_elide(active_model)}")


class SubModelBadge(NoMarkupStatic):
    def set_model(self, subagent_model: str, active_model: str = "") -> None:
        if not subagent_model or subagent_model == active_model:
            self.update("")
            return
        self.update(f"· sub {_elide(subagent_model)}")
