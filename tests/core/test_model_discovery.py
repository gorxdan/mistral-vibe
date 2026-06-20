from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from tests.conftest import build_test_vibe_config
from vibe.core.config import ModelConfig, ProviderConfig, VibeConfig
from vibe.core.llm.model_discovery import (
    DiscoveredModel,
    build_persisted_updates,
    candidate_local_providers,
    discover_extra_models,
    fetch_model_ids,
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

    async def _fake(provider: ProviderConfig, **_k: object) -> list[str]:
        return ["gemma:1b", "qwen:7b"] if provider.name == "ollama" else []

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
    out = await discover_extra_models(config)

    assert {dm.model.name for dm in out} == {"gemma:1b", "qwen:7b"}
    assert all(dm.ephemeral for dm in out)
    assert all(dm.provider.name == "ollama" for dm in out)
    assert all(dm.provider.reasoning_field_name == "reasoning" for dm in out)


@pytest.mark.asyncio
async def test_autodetect_suppressed_when_config_defines_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # User defined an ollama provider WITHOUT discover_models -> their config
    # wins, auto-detect must not override it.
    config = build_test_vibe_config(
        providers=[ProviderConfig(name="ollama", api_base="http://127.0.0.1:11434/v1")],
        models=[ModelConfig(name="m", provider="ollama", alias="m")],
    )

    async def _fake(_provider: ProviderConfig, **_k: object) -> list[str]:
        return ["should-not-appear"]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
    assert await discover_extra_models(config) == []


@pytest.mark.asyncio
async def test_explicit_discover_models_is_not_ephemeral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider(name="lmstudio", api_base="http://127.0.0.1:1234/v1")],
        models=[ModelConfig(name="x", provider="lmstudio", alias="x")],
    )

    async def _fake(provider: ProviderConfig, **_k: object) -> list[str]:
        return ["model-a"] if provider.name == "lmstudio" else []  # ollama down

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
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

    async def _fake(_provider: ProviderConfig, **_k: object) -> list[str]:
        return ["gemma4:12b", "qwen36-32k:latest", "north:latest"]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
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

    async def _fake(provider: ProviderConfig, **_k: object) -> list[str]:
        return ["shared", "onlyA"] if provider.name == "ollama" else ["shared", "onlyB"]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
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

    async def _fake(provider: ProviderConfig, **_k: object) -> list[str]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return [f"m-{provider.name}"]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
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
