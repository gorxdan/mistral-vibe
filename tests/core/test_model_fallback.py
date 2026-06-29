from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.types import Backend, BaseEvent, RateLimitError

_PROVIDER = ProviderConfig(
    name="local",
    api_base="http://127.0.0.1:8080/v1",
    api_key_env_var="",  # keyless → always "available"
    api_style="openai",
    backend=Backend.GENERIC,
)


def _model(alias: str) -> ModelConfig:
    return ModelConfig(
        name=alias, provider="local", alias=alias, temperature=0.2, thinking="off"
    )


def _loop(fallbacks: list[str], *, models: list[ModelConfig] | None = None):
    config = build_test_vibe_config(
        providers=[_PROVIDER],
        models=models or [_model("primary"), _model("backup")],
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
    # Single configured model: no fallback_models AND no alternatives for
    # headless auto-recovery, so the error surfaces with the actionable hint.
    loop = _loop([], models=[_model("primary")])

    async def always_rate_limited() -> AsyncGenerator[BaseEvent, None]:
        raise RateLimitError("local", "primary")
        yield  # pragma: no cover

    loop._perform_llm_turn = always_rate_limited  # type: ignore[method-assign]

    raised: RateLimitError | None = None
    with caplog.at_level("WARNING"):
        with pytest.raises(RateLimitError) as exc_info:
            _ = [e async for e in loop._conversation_loop("hi")]
        raised = exc_info.value
    assert loop._fallback_model_override is None
    # The silent no-op is now diagnosable: an actionable hint is logged AND
    # attached to the terminal error so it reaches the user-visible message
    # rather than only the log file.
    assert "no fallback_models configured" in caplog.text
    assert raised is not None
    assert raised.failover_hint is not None
    assert "no fallback_models configured" in raised.failover_hint


@pytest.mark.asyncio
async def test_rate_limit_headless_auto_recovers_to_available_model() -> None:
    # Headless (no rate_limit_callback) with no fallback_models configured but a
    # second available model present: a 429 should auto-recover to it instead of
    # dead-ending. This closes the gap where ACP/workflows/forked sessions failed
    # ~69% of rate-limit events while the TUI recovered interactively.
    loop = _loop([])  # primary + backup, no fallback_models, no callback
    assert loop.rate_limit_callback is None

    calls = {"turn": 0}

    async def fake_turn() -> AsyncGenerator[BaseEvent, None]:
        calls["turn"] += 1
        if calls["turn"] == 1:
            raise RateLimitError("local", "primary")
        return
        yield  # pragma: no cover

    loop._perform_llm_turn = fake_turn  # type: ignore[method-assign]

    events = [e async for e in loop._conversation_loop("hi")]

    assert calls["turn"] == 2, "turn retried on the auto-recovered model"
    assert loop._fallback_model_override is not None
    assert loop._fallback_model_override.alias == "backup"
    assert events


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


@pytest.mark.asyncio
async def test_resolve_active_model_prefers_fallback_override() -> None:
    loop = _loop([])
    backup = next(m for m in loop.config.models if m.alias == "backup")
    loop._fallback_model_override = backup
    model, provider = loop._resolve_active_model()
    assert model.alias == "backup"
    assert provider.name == loop.config.get_provider_for_model(backup).name
    # An explicit per-call override beats the fallback override.
    primary = next(m for m in loop.config.models if m.alias == "primary")
    m2, _ = loop._resolve_active_model(model_override=primary)
    assert m2.alias == "primary"


@pytest.mark.asyncio
async def test_streaming_honors_fallback_model_override() -> None:
    # Regression: _chat_streaming must use _fallback_model_override (set by a
    # model switch) like _chat does — not config.get_active_model(). Otherwise
    # the rebuilt backend gets the OLD model name (e.g. gpt-5.5 sent to zai ->
    # "Unknown Model"), which stalled live sessions. The prior fallback tests
    # faked _perform_llm_turn, so the streaming path was never exercised.
    from tests.mock.utils import mock_llm_chunk
    from tests.stubs.fake_backend import FakeBackend
    from vibe.core.types import LLMMessage, Role

    config = build_test_vibe_config(
        providers=[_PROVIDER],
        models=[_model("primary"), _model("backup")],
        active_model="primary",
    )
    backend = FakeBackend([[mock_llm_chunk(content="ok")]])
    loop = build_test_agent_loop(config=config, backend=backend)
    backup = next(m for m in loop.config.models if m.alias == "backup")
    loop._fallback_model_override = backup
    loop.messages.append(LLMMessage(role=Role.USER, content="hi", message_id="u1"))

    _ = [chunk async for chunk in loop._chat_streaming()]

    assert backend.requests_models, "streaming should have reached the backend"
    assert backend.requests_models[-1].alias == "backup", (
        "streaming must send the switched-to model, not the configured one"
    )


@pytest.mark.asyncio
async def test_chat_model_override_uses_matching_backend_after_failover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: after a failover rebuilds self.backend for provider B, a
    # compaction-style _chat(model_override=<model on provider A>) must NOT reuse
    # the B backend — otherwise the A model name + temperature reach the B
    # endpoint (gpt-5.5 -> kimi -> "invalid temperature"). _chat must build a
    # backend for the override's provider when it differs from effective_model()'s.
    from tests.mock.utils import mock_llm_chunk
    from tests.stubs.fake_backend import FakeBackend
    from vibe.core.types import LLMMessage, Role

    alpha = ProviderConfig(
        name="alpha",
        api_base="http://alpha/v1",
        api_key_env_var="",
        api_style="openai",
        backend=Backend.GENERIC,
    )
    beta = ProviderConfig(
        name="beta",
        api_base="http://beta/v1",
        api_key_env_var="",
        api_style="openai",
        backend=Backend.GENERIC,
    )
    primary = ModelConfig(
        name="primary", provider="alpha", alias="primary", temperature=0.2
    )
    backup = ModelConfig(
        name="backup", provider="beta", alias="backup", temperature=1.0
    )
    config = build_test_vibe_config(
        providers=[alpha, beta], models=[primary, backup], active_model="primary"
    )
    # Simulate the post-failover state: backend rebuilt for beta (backup).
    beta_backend = FakeBackend([[mock_llm_chunk(content="from-beta")]])
    loop = build_test_agent_loop(config=config, backend=beta_backend)
    loop._fallback_model_override = backup
    loop.messages.append(
        LLMMessage(role=Role.USER, content="summarize", message_id="u1")
    )

    alpha_backend = FakeBackend([[mock_llm_chunk(content="from-alpha")]])

    def fake_create_backend(
        *, provider: ProviderConfig, timeout: float = 720.0
    ) -> object:
        return alpha_backend if provider.name == "alpha" else beta_backend

    monkeypatch.setattr("vibe.core.agent_loop._loop.create_backend", fake_create_backend)

    await loop._chat(model_override=primary)

    # The override model lives on alpha, so the request must reach the alpha
    # backend — not the post-failover beta backend that self.backend still is.
    assert alpha_backend.requests_models, (
        "model_override must reach a backend for its own provider"
    )
    assert alpha_backend.requests_models[-1].alias == "primary"
    assert not beta_backend.requests_models, (
        "the failover (beta) backend must not serve a provider-alpha model_override"
    )


@pytest.mark.asyncio
async def test_reload_preserves_override_when_active_model_unchanged() -> None:
    # A rate-limit switch sets _fallback_model_override transiently (config still
    # points at the rate-limited model). A reload that does NOT change the active
    # model (config edit, agent switch, LSP toggle) must PRESERVE the switch —
    # otherwise the loop reverts to the rate-limited model and re-prompts every
    # turn (the f38d32d over-clear).
    loop = _loop([])
    backup = next(m for m in loop.config.models if m.alias == "backup")
    loop._fallback_model_override = backup
    loop._tried_fallback_aliases.add("primary")

    await loop.reload_with_initial_messages()  # base_config=None -> no model change

    assert loop._fallback_model_override is not None
    assert loop._fallback_model_override.alias == "backup"
    model, _ = loop._resolve_active_model()
    assert model.alias == "backup", "the rate-limit switch survives a no-op reload"


@pytest.mark.asyncio
async def test_reload_clears_override_when_active_model_changes() -> None:
    # When the reload makes a DIFFERENT model authoritative (e.g. /model picker
    # writes active_model, then _reload_config), the stale override is dropped so
    # it can't force the old model onto the new backend (glm reaching a kimi
    # backend -> "invalid temperature / unknown model").
    loop = _loop([])
    backup = next(m for m in loop.config.models if m.alias == "backup")
    loop._fallback_model_override = backup
    loop._tried_fallback_aliases.add("primary")

    changed = build_test_vibe_config(
        providers=[_PROVIDER],
        models=[_model("primary"), _model("backup")],
        active_model="backup",  # config now authoritative on a different model
    )
    await loop.reload_with_initial_messages(base_config=changed)

    assert loop._fallback_model_override is None
    assert loop._tried_fallback_aliases == set()
    model, _ = loop._resolve_active_model()
    assert model.alias == "backup", "resolution follows the new config"
