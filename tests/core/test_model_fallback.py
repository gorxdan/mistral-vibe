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
async def test_rate_limit_with_no_fallback_surfaces_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop = _loop([])  # no fallbacks configured

    async def always_rate_limited() -> AsyncGenerator[BaseEvent, None]:
        raise RateLimitError("local", "primary")
        yield  # pragma: no cover

    loop._perform_llm_turn = always_rate_limited  # type: ignore[method-assign]

    with caplog.at_level("WARNING"):
        with pytest.raises(RateLimitError):
            _ = [e async for e in loop._conversation_loop("hi")]
    assert loop._fallback_model_override is None
    # The silent no-op is now diagnosable: an actionable hint is logged.
    assert "no fallback_models configured" in caplog.text


@pytest.mark.asyncio
async def test_rate_limit_prompts_model_switch_and_retries() -> None:
    # No automatic fallback, but a rate_limit_callback is wired: a 429 should pop
    # the model-switch dialog, switch to the chosen model, and retry the turn.
    loop = _loop([])
    seen: dict[str, object] = {}

    async def pick(provider: str, model: str, candidates: list[str]) -> str | None:
        seen["provider"] = provider
        seen["model"] = model
        seen["candidates"] = list(candidates)
        return "backup"

    loop.rate_limit_callback = pick

    calls = {"turn": 0}

    async def fake_turn() -> AsyncGenerator[BaseEvent, None]:
        calls["turn"] += 1
        if calls["turn"] == 1:
            raise RateLimitError("local", "primary")
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]

    events = [e async for e in loop._conversation_loop("hi")]

    assert calls["turn"] == 2, "turn retried on the user-chosen model"
    assert loop._fallback_model_override is not None
    assert loop._fallback_model_override.alias == "backup"
    assert seen["model"] == "primary"
    # Candidates offered exclude the rate-limited (now tried) current model.
    assert "backup" in seen["candidates"]  # type: ignore[operator]
    assert "primary" not in seen["candidates"]  # type: ignore[operator]
    assert events


@pytest.mark.asyncio
async def test_rate_limit_dialog_declined_surfaces_error() -> None:
    # User cancels the dialog (callback returns None) → surface the error.
    loop = _loop([])

    async def decline(provider: str, model: str, candidates: list[str]) -> str | None:
        return None

    loop.rate_limit_callback = decline

    async def always_rate_limited() -> AsyncGenerator[BaseEvent, None]:
        raise RateLimitError("local", "primary")
        yield  # pragma: no cover

    loop._perform_llm_turn = always_rate_limited  # type: ignore[method-assign]

    with pytest.raises(RateLimitError):
        _ = [e async for e in loop._conversation_loop("hi")]
    assert loop._fallback_model_override is None
