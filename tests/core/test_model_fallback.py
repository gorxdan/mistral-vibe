from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.types import BaseEvent, RateLimitError

_PROVIDER = ProviderConfig(
    name="local",
    api_base="http://127.0.0.1:8080/v1",
    api_key_env_var="",  # keyless → always "available"
    api_style="openai",
    backend="generic",
)


def _model(alias: str) -> ModelConfig:
    return ModelConfig(
        name=alias, provider="local", alias=alias, temperature=0.2, thinking="off"
    )


def _loop(fallbacks: list[str]):
    config = build_test_vibe_config(
        providers=[_PROVIDER],
        models=[_model("primary"), _model("backup")],
        active_model="primary",
        fallback_models=fallbacks,
    )
    return build_test_agent_loop(config=config)


@pytest.mark.asyncio
async def test_rate_limit_fails_over_to_fallback_and_retries() -> None:
    loop = _loop(["backup"])
    calls = {"turn": 0}

    async def fake_turn() -> AsyncGenerator[BaseEvent, None]:
        calls["turn"] += 1
        if calls["turn"] == 1:
            raise RateLimitError("local", "primary")
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]

    events = [e async for e in loop._conversation_loop("hi")]

    assert calls["turn"] == 2, "turn retried on the fallback model"
    assert loop._fallback_model_override is not None
    assert loop._fallback_model_override.alias == "backup"
    assert events


@pytest.mark.asyncio
async def test_rate_limit_with_no_fallback_surfaces_error() -> None:
    loop = _loop([])  # no fallbacks configured

    async def always_rate_limited() -> AsyncGenerator[BaseEvent, None]:
        raise RateLimitError("local", "primary")
        yield  # pragma: no cover

    loop._perform_llm_turn = always_rate_limited  # type: ignore[method-assign]

    with pytest.raises(RateLimitError):
        _ = [e async for e in loop._conversation_loop("hi")]
    assert loop._fallback_model_override is None
