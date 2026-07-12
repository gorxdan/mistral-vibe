from __future__ import annotations

from typing import Any, cast

from vibe.core import tracing
from vibe.core.types import AgentStats, LLMUsage


class _Span:
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


def test_initialized_accumulated_cost_can_be_exact_zero() -> None:
    stats = AgentStats(
        session_prompt_tokens=1_000_000,
        input_price_per_million=2.0,
        accumulated_cost_usd=0.0,
        accumulated_cost_initialized=True,
    )

    assert stats.session_cost == 0.0


def test_legacy_positive_accumulated_cost_still_overrides_repricing() -> None:
    stats = AgentStats(
        session_prompt_tokens=1_000_000,
        input_price_per_million=2.0,
        accumulated_cost_usd=1.25,
    )

    assert stats.session_cost == 1.25


def test_reset_context_state_clears_turn_cache_write_and_reasoning() -> None:
    stats = AgentStats(last_turn_cache_write_tokens=12, last_turn_reasoning_tokens=9)

    stats.reset_context_state()

    assert stats.last_turn_cache_write_tokens == 0
    assert stats.last_turn_reasoning_tokens == 0


def test_set_usage_emits_cache_write_tokens() -> None:
    span = _Span()

    tracing.set_usage(
        cast(Any, span),
        LLMUsage(prompt_tokens=100, cached_tokens=60, cache_write_tokens=20),
    )

    assert span.attrs["gen_ai.usage.cache_write_input_tokens"] == 20


def test_set_agent_usage_emits_cache_write_and_reasoning_tokens() -> None:
    span = _Span()

    tracing.set_agent_usage(
        cast(Any, span),
        input_tokens=100,
        output_tokens=50,
        cached_tokens=60,
        cache_write_tokens=20,
        reasoning_tokens=30,
    )

    assert span.attrs["gen_ai.usage.cache_write_input_tokens"] == 20
    assert span.attrs["gen_ai.usage.reasoning_tokens"] == 30
