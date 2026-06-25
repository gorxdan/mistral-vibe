from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from tests.conftest import build_test_vibe_config
from vibe.core.auth import openai_oauth as oauth
from vibe.core.config import ModelConfig, ProviderConfig, VibeConfig
from vibe.core.llm.model_discovery import (
    DiscoveredModel,
    RawModel,
    _is_chat_model,
    build_persisted_updates,
    candidate_local_providers,
    discover_extra_models,
    fetch_model_ids,
    fetch_models,
)
from vibe.core.types import Backend

MODELS_URL = "http://ollama-test/v1/models"


def _provider(**kw: object) -> ProviderConfig:
    defaults: dict[str, object] = {
        "name": "ollama",
        "api_base": "http://ollama-test/v1",
        "api_key_env_var": "",
        "backend": "generic",
        "discover_models": True,
    }
    defaults.update(kw)
    return ProviderConfig(**defaults)  # type: ignore[arg-type]


def _mistral_only_config() -> VibeConfig:
    """A config that does not cover ollama, so auto-detection probes it."""
    return build_test_vibe_config(
        providers=[
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                backend=Backend.MISTRAL,
            )
        ],
        models=[ModelConfig(name="m", provider="mistral", alias="m")],
    )


# --- chat-model filter -----------------------------------------------------


@pytest.mark.parametrize(
    "model_id,is_chat",
    [
        ("gpt-4o", True),
        ("gpt-4o-mini", True),
        ("o3", True),
        ("gpt-4o-audio-preview", True),  # audio chat-completions model: kept
        ("gpt-4o-search-preview", True),  # web-search chat model: kept
        ("mistral-large-latest", True),
        ("text-embedding-3-small", False),
        ("whisper-1", False),
        ("tts-1-hd", False),
        ("dall-e-3", False),
        ("gpt-image-1", False),
        ("omni-moderation-latest", False),
        ("gpt-4o-realtime-preview", False),
        ("davinci-002", False),
        ("babbage-002", False),
        ("nomic-embed-text", False),
    ],
)
def test_is_chat_model(model_id: str, is_chat: bool) -> None:
    assert _is_chat_model(model_id) is is_chat


@pytest.mark.asyncio
async def test_discover_drops_non_chat_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # A provider's /v1/models lists embeddings/audio/image models alongside chat
    # models; discovery must surface only the chat-completions ones.
    config = build_test_vibe_config(
        providers=[
            ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="",
                discover_models=True,
            )
        ],
        models=[ModelConfig(name="seed", provider="openai", alias="seed")],
    )

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        if provider.name != "openai":
            return []
        return [
            RawModel("gpt-4o"),
            RawModel("o3"),
            RawModel("text-embedding-3-small"),
            RawModel("whisper-1"),
            RawModel("tts-1"),
            RawModel("dall-e-3"),
            RawModel("gpt-4o-realtime-preview"),
        ]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)
    assert {dm.model.alias for dm in out} == {"gpt-4o", "o3"}


# --- fetch_model_ids -------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_model_ids_parses_data_ids() -> None:
    respx.get(MODELS_URL).mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "gemma4:12b"}, {"id": "qwen36-32k:latest"}]}
        )
    )
    assert await fetch_model_ids(_provider()) == ["gemma4:12b", "qwen36-32k:latest"]


@pytest.mark.asyncio
@respx.mock
async def test_fetch_model_ids_empty_on_http_error() -> None:
    respx.get(MODELS_URL).mock(return_value=httpx.Response(500))
    assert await fetch_model_ids(_provider()) == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_model_ids_empty_on_connection_error() -> None:
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("server down"))
    assert await fetch_model_ids(_provider()) == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_model_ids_empty_on_invalid_json() -> None:
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, content=b"not json"))
    assert await fetch_model_ids(_provider()) == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_model_ids_empty_on_unexpected_shape() -> None:
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"object": "x"}))
    assert await fetch_model_ids(_provider()) == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_model_ids_sends_auth_header_when_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCOVERY_KEY", "sk-test")
    route = respx.get(MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
    )
    await fetch_model_ids(_provider(api_key_env_var="DISCOVERY_KEY"))
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"


# --- fetch_models: context-window detection --------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_enriches_ollama_context_from_api_tags() -> None:
    # ollama's /v1/models carries no context info; /api/tags supplies it.
    respx.get(MODELS_URL).mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "gemma4:12b"}, {"id": "qwen:7b"}]}
        )
    )
    respx.get("http://ollama-test/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"name": "gemma4:12b", "details": {"context_length": 131072}},
                    {"name": "qwen:7b", "details": {"context_length": 32768}},
                ]
            },
        )
    )
    out = await fetch_models(_provider())
    assert {m.id: m.context_length for m in out} == {
        "gemma4:12b": 131072,
        "qwen:7b": 32768,
    }


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_reads_context_from_v1_models_for_generic() -> None:
    # Non-ollama (vLLM/llama.cpp) advertise context on /v1/models; /api/tags is
    # never touched (respx would error on an unmocked request if it were).
    prov = _provider(name="vllm", api_base="http://vllm-test/v1")
    respx.get("http://vllm-test/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "qwen", "max_model_len": 32768},
                    {"id": "llama", "meta": {"n_ctx_train": 8192}},
                    {"id": "noctx"},
                ]
            },
        )
    )
    out = await fetch_models(prov)
    assert {m.id: m.context_length for m in out} == {
        "qwen": 32768,
        "llama": 8192,
        "noctx": None,
    }


@pytest.mark.asyncio
@respx.mock
async def test_fetch_model_ids_ignores_api_tags() -> None:
    # The id-only shim must not enrich (so its respx tests need no /api/tags mock).
    respx.get(MODELS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
    )
    assert await fetch_model_ids(_provider()) == ["m"]


# --- discover_extra_models: context budget ---------------------------------


@pytest.mark.asyncio
async def test_discover_budget_capped_by_default_num_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)
    config = _mistral_only_config()

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return [RawModel("gemma4:12b", 131072)] if provider.name == "ollama" else []

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    # served window defaults to 4096 -> floor(0.85 * 4096) = 3481
    assert out[0].model.auto_compact_threshold == 3481


@pytest.mark.asyncio
async def test_discover_budget_honors_ollama_context_length_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "131072")
    config = _mistral_only_config()

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return [RawModel("gemma4:12b", 131072)] if provider.name == "ollama" else []

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    # min(131072, 131072) -> floor(0.85 * 131072) = 111411
    assert out[0].model.auto_compact_threshold == 111411


@pytest.mark.asyncio
async def test_discover_generic_budget_uncapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider(name="vllm", api_base="http://vllm-test/v1")],
        models=[ModelConfig(name="x", provider="vllm", alias="x")],
    )

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return [RawModel("big", 32768)] if provider.name == "vllm" else []

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    # non-ollama: no num_ctx cap -> floor(0.85 * 32768) = 27852
    assert out[0].model.auto_compact_threshold == 27852


@pytest.mark.asyncio
async def test_discover_no_context_leaves_default_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _mistral_only_config()

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return [RawModel("noctx", None)] if provider.name == "ollama" else []

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    # No detection -> field left unset so the global default still applies.
    assert "auto_compact_threshold" not in out[0].model.model_fields_set
    assert out[0].model.auto_compact_threshold == 200_000


# --- candidate_local_providers (auto-detect targets) -----------------------


def test_ollama_candidate_default_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    prov = candidate_local_providers()[0]
    assert prov.name == "ollama"
    assert prov.api_base == "http://127.0.0.1:11434/v1"
    assert prov.reasoning_field_name == "reasoning"
    assert prov.discover_models is True


def test_ollama_candidate_honors_ollama_host_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.5:11434")
    assert candidate_local_providers()[0].api_base == "http://192.168.1.5:11434/v1"


def test_ollama_candidate_honors_full_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://my-box:9999")
    assert candidate_local_providers()[0].api_base == "http://my-box:9999/v1"


# --- discover_extra_models: auto-detection ---------------------------------


@pytest.mark.asyncio
async def test_autodetects_ollama_with_zero_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _mistral_only_config()

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return (
            [RawModel("gemma:1b"), RawModel("qwen:7b")]
            if provider.name == "ollama"
            else []
        )

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    assert {dm.model.name for dm in out} == {"gemma:1b", "qwen:7b"}
    assert all(dm.ephemeral for dm in out)
    assert all(dm.provider.name == "ollama" for dm in out)
    assert all(dm.provider.reasoning_field_name == "reasoning" for dm in out)


@pytest.mark.asyncio
async def test_autodetect_suppressed_when_config_defines_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # User defined an ollama provider with discovery off -> their config wins,
    # auto-detect must not override it (and explicit discovery is off, so the
    # user's provider is not probed either).
    config = build_test_vibe_config(
        providers=[
            ProviderConfig(
                name="ollama",
                api_base="http://127.0.0.1:11434/v1",
                discover_models=False,
            )
        ],
        models=[ModelConfig(name="m", provider="ollama", alias="m")],
    )

    async def _fake(_provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return [RawModel("should-not-appear")]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    assert await discover_extra_models(config) == []


@pytest.mark.asyncio
async def test_explicit_discover_models_is_not_ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider(name="lmstudio", api_base="http://127.0.0.1:1234/v1")],
        models=[ModelConfig(name="x", provider="lmstudio", alias="x")],
    )

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return (
            [RawModel("model-a")] if provider.name == "lmstudio" else []
        )  # ollama down

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    assert len(out) == 1
    assert out[0].model.name == "model-a"
    assert out[0].ephemeral is False


@pytest.mark.asyncio
async def test_discover_excludes_already_configured_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider(name="ollama")],  # explicit, suppresses candidate
        models=[ModelConfig(name="gemma4:12b", provider="ollama", alias="gemma")],
    )

    async def _fake(_provider: ProviderConfig, **_k: object) -> list[RawModel]:
        return [
            RawModel("gemma4:12b"),
            RawModel("qwen36-32k:latest"),
            RawModel("north:latest"),
        ]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    assert {dm.model.name for dm in out} == {"qwen36-32k:latest", "north:latest"}
    for dm in out:
        assert dm.model.alias == dm.model.name  # no collision -> raw id alias
        assert dm.model.thinking == "off"


@pytest.mark.asyncio
async def test_discover_dedups_across_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[
            _provider(name="ollama", api_base="http://a/v1"),  # suppresses candidate
            _provider(name="lmstudio", api_base="http://b/v1"),
        ],
        models=[ModelConfig(name="x", provider="ollama", alias="x")],
    )

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        ids = ["shared", "onlyA"] if provider.name == "ollama" else ["shared", "onlyB"]
        return [RawModel(i) for i in ids]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    by_alias = {dm.model.alias: dm.model.provider for dm in out}
    assert by_alias["shared"] == "ollama"
    assert by_alias["lmstudio/shared"] == "lmstudio"
    assert by_alias["onlyA"] == "ollama"
    assert by_alias["onlyB"] == "lmstudio"


@pytest.mark.asyncio
async def test_discover_queries_providers_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[
            _provider(name="ollama", api_base="http://a/v1"),  # suppresses candidate
            _provider(name="lmstudio", api_base="http://b/v1"),
        ],
        models=[ModelConfig(name="x", provider="ollama", alias="x")],
    )

    in_flight = 0
    max_in_flight = 0

    async def _fake(provider: ProviderConfig, **_k: object) -> list[RawModel]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return [RawModel(f"m-{provider.name}")]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_models", _fake)
    out = await discover_extra_models(config)

    assert max_in_flight == 2  # both providers queried at once, not serially
    assert {dm.model.name for dm in out} == {"m-ollama", "m-lmstudio"}


# --- build_persisted_updates -----------------------------------------------


def test_persist_ephemeral_writes_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _mistral_only_config()
    monkeypatch.setattr(
        VibeConfig,
        "get_persisted_config",
        classmethod(
            lambda _cls: {
                "providers": [
                    {"name": "mistral", "api_base": "x", "backend": "mistral"}
                ],
                "models": [{"name": "m", "provider": "mistral", "alias": "m"}],
            }
        ),
    )
    oll = candidate_local_providers()[0]
    dm = DiscoveredModel(
        model=ModelConfig(name="gemma:1b", provider="ollama", alias="gemma:1b"),
        provider=oll,
        ephemeral=True,
    )

    upd = build_persisted_updates(config, dm)

    assert [m["alias"] for m in upd["models"]] == ["m", "gemma:1b"]
    assert [p["name"] for p in upd["providers"]] == ["mistral", "ollama"]
    persisted_ollama = next(p for p in upd["providers"] if p["name"] == "ollama")
    assert persisted_ollama["discover_models"] is True
    assert persisted_ollama["reasoning_field_name"] == "reasoning"


def test_persist_non_ephemeral_writes_only_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider(name="lmstudio", api_base="http://127.0.0.1:1234/v1")],
        models=[ModelConfig(name="x", provider="lmstudio", alias="x")],
    )
    monkeypatch.setattr(
        VibeConfig,
        "get_persisted_config",
        classmethod(
            lambda _cls: {
                "models": [{"name": "x", "provider": "lmstudio", "alias": "x"}]
            }
        ),
    )
    dm = DiscoveredModel(
        model=ModelConfig(name="y", provider="lmstudio", alias="y"),
        provider=config.providers[0],
        ephemeral=False,
    )

    upd = build_persisted_updates(config, dm)

    assert [m["alias"] for m in upd["models"]] == ["x", "y"]
    assert "providers" not in upd


def test_persist_does_not_duplicate_existing_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _mistral_only_config()
    monkeypatch.setattr(
        VibeConfig,
        "get_persisted_config",
        classmethod(
            lambda _cls: {
                "providers": [
                    {"name": "ollama", "api_base": "http://127.0.0.1:11434/v1"}
                ],
                "models": [],
            }
        ),
    )
    oll = candidate_local_providers()[0]
    dm = DiscoveredModel(
        model=ModelConfig(name="g", provider="ollama", alias="g"),
        provider=oll,
        ephemeral=True,
    )

    upd = build_persisted_updates(config, dm)

    # provider already persisted -> not appended again
    assert "providers" not in upd
    assert [m["alias"] for m in upd["models"]] == ["g"]


# --- ChatGPT (codex) discovery ---------------------------------------------

CHATGPT_MODELS_URL = f"{oauth.OPENAI_CHATGPT_API_BASE}/models"


def _chatgpt_provider(**kw: object) -> ProviderConfig:
    defaults: dict[str, object] = {
        "name": "openai-chatgpt",
        "api_base": oauth.OPENAI_CHATGPT_API_BASE,
        "api_style": "openai-chatgpt",
        "api_key_env_var": "",
        "backend": "generic",
        "discover_models": True,
    }
    defaults.update(kw)
    return ProviderConfig(**defaults)  # type: ignore[arg-type]


def _seed_chatgpt_session() -> None:
    """Seed the OAuth token store so resolve_chatgpt_credentials() succeeds."""
    oauth.save_tokens(
        oauth.OpenAIOAuthTokens(
            access_token="access-1",
            refresh_token="refresh-1",
            account_id="acct_123",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_parses_chatgpt_models_endpoint() -> None:
    _seed_chatgpt_session()
    respx.get(CHATGPT_MODELS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-5.5", "visibility": "list", "context_window": 272000},
                    {"slug": "gpt-5.4", "visibility": "list"},
                    {"slug": "codex-auto-review", "visibility": "hide"},
                    {"slug": "ghost", "visibility": "none"},
                ]
            },
        )
    )
    out = await fetch_models(_chatgpt_provider())
    assert out == [RawModel("gpt-5.5", 272000), RawModel("gpt-5.4", None)]


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_chatgpt_sends_oauth_bearer() -> None:
    _seed_chatgpt_session()
    route = respx.get(CHATGPT_MODELS_URL).mock(
        return_value=httpx.Response(200, json={"models": []})
    )
    await fetch_models(_chatgpt_provider())
    headers = route.calls.last.request.headers
    assert headers["Authorization"] == "Bearer access-1"
    assert headers["ChatGPT-Account-ID"] == "acct_123"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_models_chatgpt_empty_when_not_authenticated() -> None:
    # No token store -> resolve raises OpenAINotAuthenticatedError -> skip request.
    route = respx.get(CHATGPT_MODELS_URL).mock(
        return_value=httpx.Response(
            200, json={"models": [{"slug": "gpt-5.5", "visibility": "list"}]}
        )
    )
    assert await fetch_models(_chatgpt_provider()) == []
    assert not route.called


@pytest.mark.asyncio
@respx.mock
async def test_discover_extra_models_includes_chatgpt_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Suppress the auto-detected ollama candidate so only chatgpt is probed.
    monkeypatch.setattr(
        "vibe.core.llm.model_discovery.candidate_local_providers", lambda: []
    )
    _seed_chatgpt_session()
    config = build_test_vibe_config(
        providers=[_chatgpt_provider()],
        models=[
            ModelConfig(name="gpt-5.5", provider="openai-chatgpt", alias="gpt-5.5")
        ],
    )
    respx.get(CHATGPT_MODELS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-5.5", "visibility": "list"},
                    {"slug": "gpt-5.4", "visibility": "list"},
                    {"slug": "codex-auto-review", "visibility": "hide"},
                ]
            },
        )
    )
    out = await discover_extra_models(config)

    assert {dm.model.name for dm in out} == {"gpt-5.4"}
    assert all(dm.provider.name == "openai-chatgpt" for dm in out)
    assert all(dm.ephemeral is False for dm in out)
