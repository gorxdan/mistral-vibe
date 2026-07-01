from __future__ import annotations

from typing import Any

from tests.conftest import build_test_vibe_config, make_test_models
from vibe.core.baseline_scaling import (
    BaselineTier,
    baseline_tier_for,
    scaled_guard_tokens,
    section_enabled,
    trim_tool_descriptions,
)
from vibe.core.config._settings import ModelConfig


def _cfg(models, **kw):
    return build_test_vibe_config(models=models, active_model=models[0].alias, **kw)


def _model(**kw) -> ModelConfig:
    base: dict[str, Any] = {"name": "m", "provider": "mistral", "alias": "m"}
    base.update(kw)
    return ModelConfig(**base)


# ---- tier classification (opt-in) ----


def test_no_window_is_large():
    cfg = _cfg(make_test_models(auto_compact_threshold=27_000))
    assert baseline_tier_for(cfg.models[0], cfg) is BaselineTier.LARGE


def test_small_window_is_small():
    cfg = _cfg([_model(context_window=32_768)])
    assert baseline_tier_for(cfg.models[0], cfg) is BaselineTier.SMALL


def test_medium_window_is_medium():
    cfg = _cfg([_model(context_window=131_072)])
    assert baseline_tier_for(cfg.models[0], cfg) is BaselineTier.MEDIUM


def test_large_window_is_large():
    cfg = _cfg([_model(context_window=400_000)])
    assert baseline_tier_for(cfg.models[0], cfg) is BaselineTier.LARGE


def test_disabled_forces_large_even_with_window():
    cfg = _cfg([_model(context_window=32_768)])
    cfg.baseline_scaling.enabled = False
    assert baseline_tier_for(cfg.models[0], cfg) is BaselineTier.LARGE


# ---- section gating ----


def test_large_emits_every_section():
    for s in ("config_reference", "le_chaton_long", "model_routing_list", "unknown"):
        assert section_enabled(BaselineTier.LARGE, s) is True


def test_medium_sheds_only_largest_blocks():
    assert section_enabled(BaselineTier.MEDIUM, "config_reference") is False
    assert section_enabled(BaselineTier.MEDIUM, "le_chaton_long") is False
    assert section_enabled(BaselineTier.MEDIUM, "model_routing_list") is True


def test_small_drops_all_gated_but_keeps_unknown():
    for s in (
        "config_reference",
        "model_routing_list",
        "humanizer",
        "skills_summaries",
    ):
        assert section_enabled(BaselineTier.SMALL, s) is False
    assert section_enabled(BaselineTier.SMALL, "core_instructions") is True


# ---- guard scaling ----


def test_guard_large_is_raw():
    cfg = _cfg(make_test_models(auto_compact_threshold=880_000))
    assert scaled_guard_tokens(cfg, cfg.models[0], BaselineTier.LARGE) == 4000


def test_guard_small_scales_to_window():
    cfg = _cfg([_model(context_window=32_768)])
    # min(4000, max(512, int(32768*0.05)=1638)) = 1638
    assert scaled_guard_tokens(cfg, cfg.models[0], BaselineTier.SMALL) == 1638


def test_guard_floor_applies_on_tiny_window():
    cfg = _cfg([_model(context_window=8_000)])
    # int(8000*0.05)=400 < guard_floor 512 -> 512
    assert scaled_guard_tokens(cfg, cfg.models[0], BaselineTier.SMALL) == 512


def test_trim_tool_descriptions_only_small():
    cfg = _cfg([_model(context_window=32_768)])
    assert trim_tool_descriptions(BaselineTier.SMALL, cfg) is True
    assert trim_tool_descriptions(BaselineTier.MEDIUM, cfg) is False
    assert trim_tool_descriptions(BaselineTier.LARGE, cfg) is False
