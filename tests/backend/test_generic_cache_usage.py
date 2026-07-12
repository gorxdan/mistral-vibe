from __future__ import annotations

import pytest

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.generic import OpenAIAdapter
from vibe.core.types import LLMUsage


def _parse_usage(usage: dict[str, object]) -> LLMUsage:
    chunk = OpenAIAdapter().parse_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": usage,
        },
        ProviderConfig(name="test", api_base="https://example.test/v1"),
    )
    assert chunk.usage is not None
    return chunk.usage


def _parse_cache_write_usage(usage: dict[str, object]) -> int:
    return _parse_usage(usage).cache_write_tokens


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {
                "prompt_tokens_details": {"cache_write_tokens": 20},
                "cache_write_tokens": 5,
            },
            20,
        ),
        ({"cache_write_tokens": 7}, 7),
    ],
)
def test_generic_adapter_maps_cache_write_tokens(
    usage: dict[str, object], expected: int
) -> None:
    assert _parse_cache_write_usage(usage) == expected


def test_generic_adapter_nested_zero_cache_metrics_take_precedence() -> None:
    usage = _parse_usage({
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "cached_tokens": 7,
        "cache_write_tokens": 9,
    })

    assert usage.cached_tokens == 0
    assert usage.cache_write_tokens == 0


def test_generic_adapter_maps_cache_alias_and_authoritative_cost() -> None:
    usage = _parse_usage({"prompt_cache_hit_tokens": 17, "cost": 0.0042})

    assert usage.cached_tokens == 17
    assert usage.reported_cost_usd == pytest.approx(0.0042)
