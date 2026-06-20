"""Live model discovery for OpenAI-compatible providers.

Two ways a local model reaches the /model picker without a per-model config
block:

1. Auto-detection (zero config): well-known local runtimes (currently ollama)
   are probed implicitly whenever config does not already define a provider for
   them. If the server is up, its models are listed; if not, nothing happens.
   This is what makes local models "just appear" on a fresh machine.
2. Opt-in (``discover_models = true`` on a provider): for custom or remote
   OpenAI-compatible servers the user configures explicitly (llama.cpp, vLLM,
   LM Studio, a remote gateway, ...).

Discovery is best effort: any failure (server down, non-2xx, malformed body)
yields no models, so the picker never blocks or breaks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.types import Backend
from vibe.core.utils.http import build_ssl_context

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

logger = logging.getLogger(__name__)

DEFAULT_DISCOVERY_TIMEOUT = 2.0
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class DiscoveredModel:
    """A live-discovered model plus the provider it came from.

    ``ephemeral`` is True when the provider was auto-detected and is NOT in the
    user's config — selecting such a model must persist the provider too so it
    stays resolvable after reload.
    """

    model: ModelConfig
    provider: ProviderConfig
    ephemeral: bool


def _ollama_base_url() -> str:
    """Resolve the ollama base URL, honoring the standard OLLAMA_HOST env var."""
    raw = os.getenv("OLLAMA_HOST", "").strip()
    if not raw:
        return DEFAULT_OLLAMA_HOST
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def candidate_local_providers() -> list[ProviderConfig]:
    """Well-known local runtimes to auto-detect when config doesn't define them.

    Only servers on an unambiguous, runtime-specific port are probed implicitly.
    ollama's 11434 qualifies; llama.cpp's 8080 does not (it collides with common
    dev servers) and stays behind the explicit ``discover_models`` flag.
    """
    return [
        ProviderConfig(
            name="ollama",
            api_base=f"{_ollama_base_url()}/v1",
            api_key_env_var="",
            backend=Backend.GENERIC,
            api_style="openai",
            reasoning_field_name="reasoning",
            discover_models=True,
        )
    ]


async def fetch_model_ids(
    provider: ProviderConfig,
    *,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Return model ids from a provider's ``/models`` endpoint.

    Never raises: returns ``[]`` on any failure. ``api_base`` already carries
    the API version prefix (e.g. ``.../v1``), so the endpoint is
    ``{api_base}/models``.
    """
    url = f"{provider.api_base.rstrip('/')}/models"
    headers: dict[str, str] = {}
    if provider.api_key_env_var and (key := os.getenv(provider.api_key_env_var)):
        headers["Authorization"] = f"Bearer {key}"

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), verify=build_ssl_context()
        )
    try:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data: Any = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Model discovery failed for provider %r: %s", provider.name, exc)
        return []
    finally:
        if owns_client:
            await client.aclose()

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [
        item["id"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def _synth_model(provider_name: str, model_id: str, alias: str) -> ModelConfig:
    return ModelConfig(
        name=model_id,
        provider=provider_name,
        alias=alias,
        input_price=0.0,
        output_price=0.0,
        thinking="off",
    )


def _providers_to_probe(config: VibeConfig) -> list[tuple[ProviderConfig, bool]]:
    """(provider, ephemeral) pairs to probe: explicit opt-ins + auto-detected.

    Auto-detected candidates are dropped when config already defines a provider
    with the same name or base URL, so a configured server is never probed twice.
    """
    explicit = [(p, False) for p in config.providers if p.discover_models]

    configured_names = {p.name for p in config.providers}
    configured_bases = {p.api_base.rstrip("/") for p in config.providers}
    auto = [
        (c, True)
        for c in candidate_local_providers()
        if c.name not in configured_names
        and c.api_base.rstrip("/") not in configured_bases
    ]
    return explicit + auto


async def discover_extra_models(
    config: VibeConfig, *, timeout: float = DEFAULT_DISCOVERY_TIMEOUT
) -> list[DiscoveredModel]:
    """Live-discover models not already present in config.

    Probes explicit ``discover_models`` providers plus auto-detected local
    runtimes, all concurrently. Deduped against config by ``(provider, name)``
    and by ``alias`` (a colliding alias is namespaced ``{provider}/{id}``, then
    skipped if that also collides).
    """
    probe = _providers_to_probe(config)
    if not probe:
        return []

    results = await asyncio.gather(
        *(fetch_model_ids(provider, timeout=timeout) for provider, _ in probe)
    )

    existing_keys = {(m.provider, m.name) for m in config.models}
    seen_aliases = {m.alias for m in config.models}
    discovered: list[DiscoveredModel] = []

    for (provider, ephemeral), model_ids in zip(probe, results, strict=True):
        for model_id in model_ids:
            if (provider.name, model_id) in existing_keys:
                continue
            alias = model_id
            if alias in seen_aliases:
                alias = f"{provider.name}/{model_id}"
                if alias in seen_aliases:
                    continue
            seen_aliases.add(alias)
            discovered.append(
                DiscoveredModel(
                    model=_synth_model(provider.name, model_id, alias),
                    provider=provider,
                    ephemeral=ephemeral,
                )
            )

    return discovered


def build_persisted_updates(config: VibeConfig, dm: DiscoveredModel) -> dict[str, Any]:
    """Build a ``save_updates`` payload that persists a picked discovered model.

    Appends the model to the on-disk models list (falling back to the effective
    list when the file omits one). When the model came from an auto-detected
    (ephemeral) provider, the provider is appended too so the model stays
    resolvable after reload and offline.
    """
    persisted = config.get_persisted_config()
    updates: dict[str, Any] = {}

    base_models = persisted.get("models")
    if not isinstance(base_models, list):
        base_models = [m.model_dump(exclude_none=True) for m in config.models]
    updates["models"] = [*base_models, dm.model.model_dump(exclude_none=True)]

    if dm.ephemeral:
        base_providers = persisted.get("providers")
        if not isinstance(base_providers, list):
            base_providers = [p.model_dump(exclude_none=True) for p in config.providers]
        if not any(p.get("name") == dm.provider.name for p in base_providers):
            updates["providers"] = [
                *base_providers,
                dm.provider.model_dump(exclude_none=True),
            ]

    return updates
