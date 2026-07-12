from __future__ import annotations

from vibe.core.config import DEFAULT_MODELS


def test_default_mistral_models_include_cache_read_prices() -> None:
    by_alias = {model.alias: model for model in DEFAULT_MODELS}

    assert by_alias["mistral-medium-3.5"].cached_input_price == 0.15
    assert by_alias["devstral-small"].cached_input_price == 0.01


def test_default_local_model_is_explicitly_free() -> None:
    local = next(model for model in DEFAULT_MODELS if model.alias == "local")

    assert local.pricing_mode == "free"
