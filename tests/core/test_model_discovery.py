from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from tests.conftest import build_test_vibe_config
from vibe.core.config import ModelConfig, ProviderConfig, VibeConfig
from vibe.core.llm.model_discovery import (
    build_persisted_models_update,
    discover_extra_models,
    fetch_model_ids,
)

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


# --- discover_extra_models -------------------------------------------------


@pytest.mark.asyncio
async def test_discover_extra_models_no_network_when_not_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[ProviderConfig(name="ollama", api_base="http://x/v1")],
        models=[ModelConfig(name="m", provider="ollama", alias="m")],
    )

    called = False

    async def _fake(*_a: object, **_k: object) -> list[str]:
        nonlocal called
        called = True
        return ["should-not-appear"]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
    assert await discover_extra_models(config) == []
    assert called is False


@pytest.mark.asyncio
async def test_discover_extra_models_synthesizes_and_dedups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider()],
        # gemma4:12b is already configured -> must be excluded from discovery.
        models=[ModelConfig(name="gemma4:12b", provider="ollama", alias="gemma")],
    )

    async def _fake(_provider_arg: object, **_k: object) -> list[str]:
        return ["gemma4:12b", "qwen36-32k:latest", "north:latest"]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
    out = await discover_extra_models(config)

    assert {m.name for m in out} == {"qwen36-32k:latest", "north:latest"}
    for m in out:
        assert m.provider == "ollama"
        assert m.input_price == 0.0
        assert m.output_price == 0.0
        assert m.thinking == "off"
        # alias defaults to the raw id when there is no collision
        assert m.alias == m.name


@pytest.mark.asyncio
async def test_discover_extra_models_namespaces_alias_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider()],
        models=[ModelConfig(name="something", provider="ollama", alias="mymodel")],
    )

    async def _fake(_provider_arg: object, **_k: object) -> list[str]:
        return ["mymodel"]  # id collides with an existing model's alias

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
    out = await discover_extra_models(config)

    assert len(out) == 1
    assert out[0].name == "mymodel"
    assert out[0].alias == "ollama/mymodel"


@pytest.mark.asyncio
async def test_discover_extra_models_dedups_across_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[
            _provider(name="ollama", api_base="http://a/v1"),
            _provider(name="lmstudio", api_base="http://b/v1"),
        ],
        models=[ModelConfig(name="x", provider="ollama", alias="x")],
    )

    async def _fake(provider: ProviderConfig, **_k: object) -> list[str]:
        return ["shared", "onlyA"] if provider.name == "ollama" else ["shared", "onlyB"]

    monkeypatch.setattr("vibe.core.llm.model_discovery.fetch_model_ids", _fake)
    out = await discover_extra_models(config)

    by_alias = {m.alias: m.provider for m in out}
    # First provider (ollama) keeps the raw id; the colliding id from the
    # second provider is namespaced. Order is deterministic.
    assert by_alias["shared"] == "ollama"
    assert by_alias["lmstudio/shared"] == "lmstudio"
    assert by_alias["onlyA"] == "ollama"
    assert by_alias["onlyB"] == "lmstudio"


@pytest.mark.asyncio
async def test_discover_extra_models_queries_providers_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[
            _provider(name="ollama", api_base="http://a/v1"),
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
    assert {m.name for m in out} == {"m-ollama", "m-lmstudio"}


# --- build_persisted_models_update -----------------------------------------


def test_build_persisted_models_update_appends_to_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider()],
        models=[ModelConfig(name="a", provider="ollama", alias="a")],
    )
    monkeypatch.setattr(
        VibeConfig,
        "get_persisted_config",
        classmethod(
            lambda _cls: {"models": [{"name": "x", "provider": "ollama", "alias": "x"}]}
        ),
    )
    new = ModelConfig(name="b", provider="ollama", alias="b")

    update = build_persisted_models_update(config, new)

    # Appends to the ON-DISK list, not the effective (defaults-merged) list.
    assert [m["alias"] for m in update["models"]] == ["x", "b"]


def test_build_persisted_models_update_falls_back_to_effective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_vibe_config(
        providers=[_provider()],
        models=[ModelConfig(name="a", provider="ollama", alias="a")],
    )
    monkeypatch.setattr(
        VibeConfig, "get_persisted_config", classmethod(lambda _cls: {})
    )
    new = ModelConfig(name="b", provider="ollama", alias="b")

    update = build_persisted_models_update(config, new)

    aliases = [m["alias"] for m in update["models"]]
    assert aliases[-1] == "b"
    assert "a" in aliases
