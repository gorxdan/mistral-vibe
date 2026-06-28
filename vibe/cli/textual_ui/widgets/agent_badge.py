from __future__ import annotations

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic


class ModelStatusBadge(NoMarkupStatic):
    def set_models(self, active_model: str, subagent_model: str) -> None:
        if not active_model and not subagent_model:
            self.update("")
            return
        model_label = active_model or "unknown"
        subagent_label = subagent_model or "unknown"
        self.update(f"⟦MODEL {model_label} / SUB MODEL {subagent_label}⟧")
