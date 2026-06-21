from __future__ import annotations

from unittest.mock import Mock

from tests.conftest import build_test_vibe_config
from vibe.cli.textual_ui.widgets.banner.banner import Banner, BannerState
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
