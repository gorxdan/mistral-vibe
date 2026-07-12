from __future__ import annotations

import pytest

from vibe.core.config import ProviderConfig
from vibe.core.llm.backend.anthropic import AnthropicAdapter, AnthropicMapper
from vibe.core.llm.backend.reasoning_adapter import ReasoningAdapter
from vibe.core.types import Backend


def _anthropic_usage() -> dict[str, int]:
    return {
        "input_tokens": 10,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 70,
        "output_tokens": 5,
    }


def test_anthropic_response_maps_cache_creation_to_cache_write() -> None:
    chunk = AnthropicMapper().parse_response({
        "content": [{"type": "text", "text": "ok"}],
        "usage": _anthropic_usage(),
    })

    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 100
    assert chunk.usage.cached_tokens == 70
    assert chunk.usage.cache_write_tokens == 20


def test_anthropic_mapper_stream_maps_cache_creation_to_cache_write() -> None:
    chunk, _ = AnthropicMapper().parse_streaming_event(
        "message_start", {"message": {"usage": _anthropic_usage()}}, 0
    )

    assert chunk is not None
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 100
    assert chunk.usage.cached_tokens == 70
    assert chunk.usage.cache_write_tokens == 20


def test_anthropic_adapter_stream_maps_cache_creation_to_cache_write() -> None:
    chunk = AnthropicAdapter().parse_response({
        "type": "message_start",
        "message": {"usage": _anthropic_usage()},
    })

    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 100
    assert chunk.usage.cached_tokens == 70
    assert chunk.usage.cache_write_tokens == 20


def test_reasoning_adapter_maps_cache_and_reasoning_usage() -> None:
    chunk = ReasoningAdapter().parse_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "prompt_tokens_details": {
                    "cached_tokens": 60,
                    "cache_write_tokens": 20,
                },
                "completion_tokens_details": {"reasoning_tokens": 25},
            },
        },
        ProviderConfig(
            name="reasoning-test",
            api_base="https://example.invalid/v1",
            backend=Backend.GENERIC,
            api_style="reasoning",
        ),
    )

    assert chunk.usage is not None
    assert chunk.usage.cached_tokens == 60
    assert chunk.usage.cache_write_tokens == 20
    assert chunk.usage.reasoning_tokens == 25


def test_reasoning_adapter_accepts_flat_cache_usage() -> None:
    chunk = ReasoningAdapter().parse_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "cached_tokens": 60,
                "cache_write_tokens": 20,
                "reasoning_tokens": 25,
            },
        },
        ProviderConfig(
            name="reasoning-test",
            api_base="https://example.invalid/v1",
            backend=Backend.GENERIC,
            api_style="reasoning",
        ),
    )

    assert chunk.usage is not None
    assert chunk.usage.cached_tokens == 60
    assert chunk.usage.cache_write_tokens == 20
    assert chunk.usage.reasoning_tokens == 25


def test_reasoning_adapter_nested_zero_metrics_take_precedence() -> None:
    chunk = ReasoningAdapter().parse_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
                "cached_tokens": 7,
                "cache_write_tokens": 9,
                "reasoning_tokens": 11,
            },
        },
        ProviderConfig(
            name="reasoning-test",
            api_base="https://example.invalid/v1",
            backend=Backend.GENERIC,
            api_style="reasoning",
        ),
    )

    assert chunk.usage is not None
    assert chunk.usage.cached_tokens == 0
    assert chunk.usage.cache_write_tokens == 0
    assert chunk.usage.reasoning_tokens == 0


def test_reasoning_adapter_maps_cache_alias_and_authoritative_cost() -> None:
    chunk = ReasoningAdapter().parse_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_cache_hit_tokens": 17, "cost": 0.0042},
        },
        ProviderConfig(
            name="reasoning-test",
            api_base="https://example.invalid/v1",
            backend=Backend.GENERIC,
            api_style="reasoning",
        ),
    )

    assert chunk.usage is not None
    assert chunk.usage.cached_tokens == 17
    assert chunk.usage.reported_cost_usd == pytest.approx(0.0042)
