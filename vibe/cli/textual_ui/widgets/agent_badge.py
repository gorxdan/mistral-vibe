from __future__ import annotations

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.agents import AgentSafety

_SAFETY_CLASSES: dict[AgentSafety, str] = {
    AgentSafety.SAFE: "badge-safe",
    AgentSafety.DESTRUCTIVE: "badge-warning",
    AgentSafety.YOLO: "badge-error",
}


class AgentProfileBadge(NoMarkupStatic):
    def set_profile(self, name: str, safety: AgentSafety) -> None:
        for badge_class in _SAFETY_CLASSES.values():
            self.remove_class(badge_class)
        badge_class = _SAFETY_CLASSES.get(safety)
        if badge_class:
            self.add_class(badge_class)
        self.update(f"⟦● {name}⟧" if name else "")
