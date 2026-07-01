from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalGroup
from textual.events import Resize
from textual.reactive import reactive
from textual.widgets import Static

from vibe import __version__
from vibe.cli.textual_ui.widgets.banner.petit_chat import PetitChat
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.config import VibeConfig
from vibe.core.skills.manager import SkillManager


def _pluralize(count: int, singular: str) -> str:
    return f"{count} {singular}{'s' if count != 1 else ''}"


_FEATURES_PREFIX = "Features:  "


@dataclass
class BannerState:
    active_model: str = ""
    models_count: int = 0
    mcp_servers_enabled: int = 0
    mcp_servers_total: int = 0
    connectors_connected: int = 0
    connectors_total: int = 0
    skills_count: int = 0
    hooks_count: int = 0
    plan_description: str | None = None
    safety_judge: str | None = None
    # Optional feature toggles surfaced as a startup checklist ([x]/[ ] per flag).
    features: list[tuple[str, bool]] = field(default_factory=list)


class Banner(Static):
    state = reactive(BannerState(), init=False)

    def __init__(
        self,
        config: VibeConfig,
        skill_manager: SkillManager,
        connectors_connected: int = 0,
        connectors_total: int = 0,
        hooks_count: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.can_focus = False
        self._initial_state = self._build_state(
            config=config,
            skill_manager=skill_manager,
            connectors_connected=connectors_connected,
            connectors_total=connectors_total,
            hooks_count=hooks_count,
            plan_description=None,
        )
        self._animated = not config.disable_welcome_banner_animation

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="banner-container"):
            yield PetitChat(animate=self._animated)

            with Vertical(id="banner-info"):
                with Horizontal(classes="banner-line"):
                    yield NoMarkupStatic("Mistral Vibe", id="banner-brand")
                    yield NoMarkupStatic(" ", classes="banner-spacer")
                    yield NoMarkupStatic(f"v{__version__} · ", classes="banner-meta")
                    yield NoMarkupStatic("", id="banner-model")
                    yield NoMarkupStatic("", id="banner-user-plan")
                with Horizontal(classes="banner-line"):
                    yield NoMarkupStatic("", id="banner-meta-counts")
                with Horizontal(classes="banner-line"):
                    yield NoMarkupStatic(
                        self._features_text(self._initial_state), id="banner-features"
                    )
                with Horizontal(classes="banner-line"):
                    yield NoMarkupStatic("Type ", classes="banner-meta")
                    yield NoMarkupStatic("/help", classes="banner-cmd")
                    yield NoMarkupStatic(" for more information", classes="banner-meta")

    def on_mount(self) -> None:
        self.state = self._initial_state

    def on_resize(self, event: Resize) -> None:
        if not self.is_attached:
            return
        self.query_one("#banner-features", NoMarkupStatic).update(
            self._format_features()
        )

    def watch_state(self) -> None:
        if not self.is_attached:
            return
        self.query_one("#banner-model", NoMarkupStatic).update(self.state.active_model)
        self.query_one("#banner-meta-counts", NoMarkupStatic).update(
            self._format_meta_counts()
        )
        self.query_one("#banner-features", NoMarkupStatic).update(
            self._format_features()
        )
        self.query_one("#banner-user-plan", NoMarkupStatic).update(self._format_plan())

    def freeze_animation(self) -> None:
        if self._animated:
            self.query_one(PetitChat).freeze_animation()

    def set_state(
        self,
        config: VibeConfig,
        skill_manager: SkillManager,
        connectors_connected: int = 0,
        connectors_total: int = 0,
        hooks_count: int = 0,
        plan_description: str | None = None,
    ) -> None:
        self.state = self._build_state(
            config,
            skill_manager,
            connectors_connected,
            connectors_total,
            hooks_count,
            plan_description,
        )

    @staticmethod
    def _build_state(
        config: VibeConfig,
        skill_manager: SkillManager,
        connectors_connected: int = 0,
        connectors_total: int = 0,
        hooks_count: int = 0,
        plan_description: str | None = None,
    ) -> BannerState:
        all_servers = config.mcp_servers
        enabled_servers = [s for s in all_servers if not s.disabled]

        active_model = config.get_active_model()
        judge = getattr(config, "safety_judge", None)
        safety_judge = (
            (judge.model or "on") if judge and judge.enabled and judge.model else None
        )
        return BannerState(
            active_model=f"{active_model.alias}[{active_model.thinking}]",
            models_count=len(config.models),
            mcp_servers_enabled=len(enabled_servers),
            mcp_servers_total=len(all_servers),
            connectors_connected=connectors_connected,
            connectors_total=connectors_total,
            skills_count=skill_manager.custom_skills_count,
            hooks_count=hooks_count,
            plan_description=plan_description,
            safety_judge=safety_judge,
            features=Banner._feature_flags(config),
        )

    @staticmethod
    def _feature_flags(config: VibeConfig) -> list[tuple[str, bool]]:
        """Curated optional-feature toggles for the startup checklist.

        Defensive against partial/Mock configs: every access falls back to a
        safe default, and provider-derived flags are only computed when
        ``providers`` is a real list (Mock configs return a non-iterable Mock).
        """
        shaping = getattr(config, "context_shaping", None)

        def _on(obj: object, attr: str) -> bool:
            return bool(getattr(obj, attr, False))

        providers = getattr(config, "providers", None)
        discover = cache = False
        if isinstance(providers, list):
            discover = any(getattr(p, "discover_models", False) for p in providers)
            cache = any(
                getattr(getattr(p, "cache", None), "mode", "off") == "explicit"
                for p in providers
            )
        return [
            ("snip", _on(getattr(shaping, "snip", None), "enabled")),
            ("microcompact", _on(getattr(shaping, "microcompact", None), "enabled")),
            ("context-warnings", _on(config, "context_warnings")),
            ("file-watcher", _on(config, "file_watcher_for_autocomplete")),
            ("commit-signature", _on(config, "include_commit_signature")),
            ("memory", _on(getattr(config, "memory", None), "enabled")),
            ("safety-judge", _on(getattr(config, "safety_judge", None), "enabled")),
            (
                "output-escalation",
                _on(getattr(config, "max_output_escalation", None), "enabled"),
            ),
            ("model-discovery", bool(discover)),
            ("provider-cache", bool(cache)),
        ]

    def _format_meta_counts(self) -> str:
        parts = [_pluralize(self.state.models_count, "model")]
        # Format as x/y for MCP servers and connectors (only when enabled != total)
        if self.state.connectors_total > 0:
            if self.state.connectors_connected != self.state.connectors_total:
                connector_str = (
                    f"{self.state.connectors_connected}/{self.state.connectors_total} connector"
                    + ("s" if self.state.connectors_total != 1 else "")
                )
            else:
                connector_str = _pluralize(self.state.connectors_connected, "connector")
            parts.append(connector_str)
        # Always show MCP servers count (even if 0/0)
        if self.state.mcp_servers_enabled != self.state.mcp_servers_total:
            mcp_str = (
                f"{self.state.mcp_servers_enabled}/{self.state.mcp_servers_total} MCP server"
                + ("s" if self.state.mcp_servers_total != 1 else "")
            )
        else:
            mcp_str = _pluralize(self.state.mcp_servers_enabled, "MCP server")
        parts.append(mcp_str)
        parts.append(_pluralize(self.state.skills_count, "skill"))
        if self.state.hooks_count > 0:
            parts.append(_pluralize(self.state.hooks_count, "hook"))
        if self.state.safety_judge:
            parts.append(f"🛡 judge:{self.state.safety_judge}")
        return " · ".join(parts)

    def _format_features(self) -> str:
        return self._features_text(self.state, max_width=self._features_max_width())

    def _features_max_width(self) -> int | None:
        # Terminal width is the real constraint: the banner is laid out with
        # width:auto, so the widget's own size is content-driven and unreliable.
        # #banner-container reserves one column of right padding (app.tcss).
        if not self.is_attached:
            return None
        width = self.app.size.width
        if width <= 0:
            return None
        return max(1, width - 1)

    @staticmethod
    def _features_text(state: BannerState, max_width: int | None = None) -> str:
        # Startup checklist of optional feature toggles. Empty (e.g. partial
        # config in unit tests) renders as a blank line to keep the layout.
        # Seeded at compose time from _initial_state (not just the reactive
        # update) so the line is present in the very first rendered frame.
        if not state.features:
            return ""
        tags = [f"[{'x' if on else ' '}] {label}" for label, on in state.features]
        # Narrow terminals / unattached (unit-test) widgets: one line, let the
        # container clip rather than guess. The first tag can't be dropped, so
        # we always place it even if the prefix alone already overflows.
        if max_width is None or max_width <= len(_FEATURES_PREFIX) + len(tags[0]):
            return _FEATURES_PREFIX + "  ".join(tags)
        indent = " " * len(_FEATURES_PREFIX)
        sep = "  "
        lines: list[str] = []
        current = _FEATURES_PREFIX
        for tag in tags:
            is_first_on_line = current == _FEATURES_PREFIX and not lines
            addition = tag if is_first_on_line else sep + tag
            if len(current) + len(addition) <= max_width or is_first_on_line:
                current += addition
            else:
                lines.append(current)
                current = indent + tag
        lines.append(current)
        return "\n".join(lines)

    def _format_plan(self) -> str:
        return (
            ""
            if self.state.plan_description is None
            else f" · {self.state.plan_description}"
        )
