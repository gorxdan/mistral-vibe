from __future__ import annotations

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.types import (
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    UnclassifiedBackendError,
)
from vibe.core.usage import CallKind


def _config(model: ModelConfig):
    return build_test_vibe_config(
        active_model=model.alias,
        models=[model],
        providers=[
            ProviderConfig(name=model.provider, api_base="https://example.test/v1")
        ],
        enabled_tools=[],
    )


async def _run(model: ModelConfig, usage: LLMUsage):
    backend = FakeBackend(
        LLMChunk(message=LLMMessage(role=Role.ASSISTANT, content="done"), usage=usage)
    )
    agent = build_test_agent_loop(config=_config(model), backend=backend)
    async for _ in agent.act("hello"):
        pass
    return agent.stats


@pytest.mark.asyncio
async def test_agent_loop_uses_cache_read_and_write_prices() -> None:
    model = ModelConfig(
        name="custom-priced-model",
        provider="provider",
        alias="model",
        pricing_mode="api",
        input_price=1.0,
        cached_input_price=0.2,
        cache_write_input_price=1.25,
        output_price=2.0,
    )
    usage = LLMUsage(
        prompt_tokens=1_000_000,
        cached_tokens=600_000,
        cache_write_tokens=200_000,
        completion_tokens=100_000,
    )

    stats = await _run(model, usage)

    assert stats.session_cost == pytest.approx(0.77)
    assert stats.session_cached_tokens == 600_000
    assert stats.session_cache_write_tokens == 200_000


@pytest.mark.asyncio
async def test_subscription_model_is_not_repriced_from_api_table() -> None:
    model = ModelConfig(
        name="glm-5.2", provider="zai", alias="glm", pricing_mode="subscription"
    )

    stats = await _run(
        model,
        LLMUsage(
            prompt_tokens=1_000_000, cached_tokens=800_000, completion_tokens=100_000
        ),
    )

    assert stats.session_cost == 0.0
    assert stats.accumulated_cost_initialized is True
    assert stats.cost_is_estimated is False


@pytest.mark.asyncio
async def test_agent_loop_normalizes_overlapping_provider_cache_usage() -> None:
    model = ModelConfig(
        name="custom-priced-model",
        provider="provider",
        alias="model",
        pricing_mode="api",
        input_price=1.0,
        cached_input_price=0.2,
        cache_write_input_price=1.25,
        output_price=2.0,
    )

    stats = await _run(
        model,
        LLMUsage(
            prompt_tokens=10,
            cached_tokens=8,
            cache_write_tokens=8,
            completion_tokens=2,
            reasoning_tokens=3,
        ),
    )

    assert stats.session_cached_tokens == 8
    assert stats.session_cache_write_tokens == 2
    assert stats.session_reasoning_tokens == 2


@pytest.mark.asyncio
async def test_agent_loop_uses_authoritative_provider_cost() -> None:
    model = ModelConfig(
        name="routed-model",
        provider="openrouter",
        alias="model",
        pricing_mode="unknown",
    )

    stats = await _run(
        model,
        LLMUsage(prompt_tokens=100, completion_tokens=20, reported_cost_usd=0.0042),
    )

    assert stats.session_cost == pytest.approx(0.0042)
    assert stats.cost_is_estimated is False


def test_auxiliary_meter_contributes_to_agent_session_totals() -> None:
    model = ModelConfig(
        name="aux-model",
        provider="provider",
        alias="model",
        pricing_mode="api",
        input_price=1.0,
        output_price=2.0,
    )
    agent = build_test_agent_loop(config=_config(model), backend=FakeBackend())
    reservation = agent._usage_meter.try_reserve(20, estimated_cost_usd=0.01)
    assert reservation is not None

    agent._usage_meter.reconcile(
        reservation,
        usage=LLMUsage(prompt_tokens=10, completion_tokens=2),
        model=model,
        provider=agent.config.get_active_provider(),
        call_kind=CallKind.MEMORY_SELECT,
        duration_s=0.1,
    )

    assert agent.stats.session_prompt_tokens == 10
    assert agent.stats.session_completion_tokens == 2
    assert agent.stats.session_cost == pytest.approx(14 / 1_000_000)


@pytest.mark.asyncio
async def test_missing_provider_usage_records_conservative_settlement() -> None:
    model = ModelConfig(
        name="unknown-model", provider="provider", alias="model", pricing_mode="unknown"
    )
    backend = FakeBackend(
        LLMChunk(message=LLMMessage(role=Role.ASSISTANT, content="done"), usage=None)
    )
    agent = build_test_agent_loop(config=_config(model), backend=backend)

    with pytest.raises(UnclassifiedBackendError):
        await agent._chat()

    assert agent.stats.session_prompt_tokens > 0
    assert agent.stats.session_completion_tokens > 0
    assert agent.stats.accumulated_cost_initialized is True
    assert agent.stats.cost_is_estimated is True
    assert agent.stats.session_cost > 0.0
