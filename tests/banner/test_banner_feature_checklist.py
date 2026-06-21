from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult

from tests.conftest import build_test_vibe_config
from vibe.cli.textual_ui.widgets.banner.banner import Banner, BannerState
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.config import ContextShapingConfig, VibeConfig
from vibe.core.config._settings import (
    MemoryConfig,
    MicrocompactConfig,
    ModelConfig,
    SafetyJudgeConfig,
)
from vibe.core.skills.manager import SkillManager


def _banner() -> Banner:
    config = Mock(spec=VibeConfig)
    config.active_model = "m"
    config.models = ["m"]
    config.mcp_servers = []
    config.disable_welcome_banner_animation = False
    config.get_active_model.return_value = ModelConfig(
        name="m", provider="mistral", alias="m", thinking="off"
    )
    sm = Mock(spec=SkillManager)
    sm.custom_skills_count = 0
    return Banner(config=config, skill_manager=sm)


def _labels(flags: list[tuple[str, bool]]) -> dict[str, bool]:
    return dict(flags)


def test_feature_flags_default_config_has_expected_labels() -> None:
    flags = Banner._feature_flags(build_test_vibe_config())
    assert list(label for label, _ in flags) == [
        "snip",
        "microcompact",
        "context-warnings",
        "file-watcher",
        "commit-signature",
        "memory",
        "safety-judge",
        "output-escalation",
        "model-discovery",
        "provider-cache",
    ]


def test_feature_flags_reflect_default_on_toggles() -> None:
    # Defaults flipped to on: shaping stages, context-warnings, file-watcher,
    # commit-signature, memory, safety-judge, output-escalation.
    on = _labels(Banner._feature_flags(build_test_vibe_config()))
    for key in (
        "snip",
        "microcompact",
        "context-warnings",
        "file-watcher",
        "commit-signature",
        "memory",
        "safety-judge",
        "output-escalation",
    ):
        assert on[key] is True, f"{key} should default on"


def test_feature_flags_reflect_disabled_toggles() -> None:
    cfg = build_test_vibe_config(
        context_shaping=ContextShapingConfig(
            microcompact=MicrocompactConfig(enabled=False)
        ),
        context_warnings=False,
        memory=MemoryConfig(enabled=False),
        safety_judge=SafetyJudgeConfig(enabled=False),
    )
    off = _labels(Banner._feature_flags(cfg))
    assert off["microcompact"] is False
    assert off["context-warnings"] is False
    assert off["memory"] is False
    assert off["safety-judge"] is False
    # Untouched toggles stay on.
    assert off["snip"] is True


def test_format_features_renders_checkmark_tags() -> None:
    banner = _banner()
    banner.state = BannerState(features=[("snip", True), ("memory", False)])
    out = banner._format_features()
    assert out.startswith("Features:")
    assert "[x] snip" in out
    assert "[ ] memory" in out


def test_format_features_empty_when_no_flags() -> None:
    banner = _banner()
    banner.state = BannerState()
    assert banner._format_features() == ""


def _all_flags() -> list[tuple[str, bool]]:
    return Banner._feature_flags(build_test_vibe_config())


def test_features_text_single_line_when_width_unbounded() -> None:
    out = Banner._features_text(BannerState(features=_all_flags()), max_width=None)
    assert out.count("\n") == 0
    assert out.startswith("Features:  ")
    assert "[x] snip" in out


def test_features_text_wraps_to_multiple_lines_under_narrow_width() -> None:
    out = Banner._features_text(BannerState(features=_all_flags()), max_width=50)
    lines = out.split("\n")
    assert len(lines) >= 2
    assert lines[0].startswith("Features:  ")
    # No wrapped line exceeds the budget.
    assert all(len(line) <= 50 for line in lines)
    # Continuation lines indent under the feature list (len("Features:  ") == 11).
    for line in lines[1:]:
        assert line.startswith(" " * 11)
        assert not line[11:].startswith(" ")  # tag text, not extra spaces


def test_features_text_keeps_each_tag_atomic_when_wrapping() -> None:
    flags = [("snip", True), ("memory", False)]
    out = Banner._features_text(BannerState(features=flags), max_width=20)
    # Both tags present whole; neither split across the wrap.
    assert "[x] snip" in out
    assert "[ ] memory" in out


class _BannerApp(App[None]):
    def __init__(self, banner: Banner) -> None:
        super().__init__()
        self._banner = banner

    def compose(self) -> ComposeResult:
        yield self._banner


@pytest.mark.asyncio
async def test_mounted_banner_reflows_features_on_terminal_resize() -> None:
    banner = _banner()
    banner.state = BannerState(features=_all_flags())

    async with _BannerApp(banner).run_test(size=(220, 24)) as pilot:
        wide = str(banner.query_one("#banner-features", NoMarkupStatic).render())
        assert "\n" not in wide

        await pilot.resize_terminal(45, 24)
        await pilot.pause()
        narrow = str(banner.query_one("#banner-features", NoMarkupStatic).render())
        assert "\n" in narrow
        for line in narrow.split("\n"):
            assert len(line) <= 45
