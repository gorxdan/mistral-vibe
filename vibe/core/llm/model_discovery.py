"""Live model discovery for OpenAI-compatible providers.

Lets a provider opt in (``discover_models = true``) to having its served
models listed in the model picker automatically — typically a local
ollama/llama.cpp server — instead of requiring a hand-written ``[[models]]``
block per model. Discovery is best effort: any failure (server down, non-2xx,
malformed body) yields an empty list so the picker never breaks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

from vibe.core.config import ModelConfig
from vibe.core.utils.http import build_ssl_context

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig, VibeConfig

logger = logging.getLogger(__name__)

DEFAULT_DISCOVERY_TIMEOUT = 2.0


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


async def discover_extra_models(
    config: VibeConfig, *, timeout: float = DEFAULT_DISCOVERY_TIMEOUT
) -> list[ModelConfig]:
    """Synthesize ModelConfigs for live-discovered models not already in config.

    Only providers with ``discover_models = true`` are queried, so there is
    zero network overhead unless a provider opts in. Deduped against config by
    ``(provider, name)`` and by ``alias`` (a colliding alias is namespaced as
    ``{provider}/{id}``, then skipped if that also collides).
    """
    providers = [p for p in config.providers if p.discover_models]
    if not providers:
        return []

    existing_keys = {(m.provider, m.name) for m in config.models}
    seen_aliases = {m.alias for m in config.models}
    discovered: list[ModelConfig] = []

    # Query all opted-in providers concurrently so a slow/down server does not
    # serialize the picker-open latency (fetch_model_ids never raises). Results
    # are consumed in provider order to keep alias deduplication deterministic.
    results = await asyncio.gather(
        *(fetch_model_ids(provider, timeout=timeout) for provider in providers)
    )
    for provider, model_ids in zip(providers, results, strict=True):
        for model_id in model_ids:
            if (provider.name, model_id) in existing_keys:
                continue
            alias = model_id
            if alias in seen_aliases:
                alias = f"{provider.name}/{model_id}"
                if alias in seen_aliases:
                    continue
            seen_aliases.add(alias)
            discovered.append(_synth_model(provider.name, model_id, alias))

    return discovered


def build_persisted_models_update(
    config: VibeConfig, model: ModelConfig
) -> dict[str, Any]:
    """Build a ``save_updates`` payload that appends ``model`` to the models list.

    Reads the on-disk models list (falling back to the effective list when the
    file omits one) so a freshly picked discovered model is written to config
    and stays resolvable even if the discovery server is later offline.
    """
    persisted = config.get_persisted_config()
    base = persisted.get("models")
    if not isinstance(base, list):
        base = [m.model_dump(exclude_none=True) for m in config.models]
    models = [*base, model.model_dump(exclude_none=True)]
    return {"models": models}
