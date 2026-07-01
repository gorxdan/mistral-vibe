from __future__ import annotations

from typing import Any

from tests.conftest import build_test_vibe_config, make_test_models
from vibe.core.config._settings import ModelConfig


def _cfg_with_models(models):
    return build_test_vibe_config(models=models, active_model=models[0].alias)


def _model(**kw) -> ModelConfig:
    base: dict[str, Any] = {"name": "m", "provider": "mistral", "alias": "m"}
    base.update(kw)
    return ModelConfig(**base)


def test_no_window_keeps_explicit_threshold_unchanged():
    cfg = _cfg_with_models(make_test_models(auto_compact_threshold=27_000))
    assert cfg.models[0].context_window is None
    assert cfg.models[0].auto_compact_threshold == 27_000
    assert cfg.models[0].effective_context_window == 27_000


def test_declared_window_no_explicit_threshold_derives_85pct():
    cfg = _cfg_with_models([_model(context_window=32_768)])
    assert cfg.models[0].auto_compact_threshold == int(32_768 * 0.85)
    assert cfg.models[0].effective_context_window == 32_768


def test_explicit_threshold_clamped_to_window():
    cfg = _cfg_with_models([
        _model(context_window=32_768, auto_compact_threshold=60_000)
    ])
    assert cfg.models[0].auto_compact_threshold == int(32_768 * 0.95)


def test_explicit_threshold_below_cap_respected():
    cfg = _cfg_with_models([
        _model(context_window=32_768, auto_compact_threshold=27_000)
    ])
    # 27000 < 0.95*32768=31129 -> respected verbatim.
    assert cfg.models[0].auto_compact_threshold == 27_000


def test_baseline_scaling_defaults():
    cfg = _cfg_with_models(make_test_models(auto_compact_threshold=200_000))
    assert cfg.baseline_scaling.enabled is True
    assert cfg.baseline_scaling.small_max == 48_000
    assert cfg.baseline_scaling.medium_max == 200_000
