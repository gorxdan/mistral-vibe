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
import math
import os
from typing import TYPE_CHECKING, Any

import httpx

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.logger import logger
from vibe.core.types import Backend
from vibe.core.utils.http import build_ssl_context

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

DEFAULT_DISCOVERY_TIMEOUT = 2.0
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# ollama's default served context window when OLLAMA_CONTEXT_LENGTH is unset. A
# model's trained context_length can be far larger, but ollama only serves this
# many tokens unless told otherwise, silently truncating beyond it.
DEFAULT_OLLAMA_NUM_CTX = 4096
# Fraction of the effective context window used as the token budget, so context
# shaping/compaction fires before the server's real limit is hit.
CONTEXT_BUDGET_SAFETY = 0.85

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

@dataclass(frozen=True)
class RawModel:
    """A discovered model id plus the context window the server advertises.

    ``context_length`` is the model's own maximum (ollama's
    ``details.context_length`` or vLLM's ``max_model_len`` and friends); it is
    None when the server advertises nothing. The token budget is derived from it
    later (capped by ollama's served window) — see :func:`_budget_from_context`.
    """

    id: str
    context_length: int | None = None

# Field names different OpenAI-compatible runtimes use to advertise a model's
# context window on /v1/models (vLLM: max_model_len; llama.cpp: meta.n_ctx*;
# LM Studio: context_length). ollama's /v1/models carries none of these, so it
# is enriched separately from its native /api/tags endpoint.
_CTX_ITEM_KEYS = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
    "n_ctx",
    "n_ctx_train",
)
_CTX_META_KEYS = ("n_ctx", "n_ctx_train", "context_length", "max_model_len")

def _ctx_value(d: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None

def _ctx_from_models_item(item: dict[str, Any]) -> int | None:
    """Best-effort context-window length from one /v1/models entry, or None."""
    direct = _ctx_value(item, _CTX_ITEM_KEYS)
    if direct is not None:
        return direct
    meta = item.get("meta")
    if isinstance(meta, dict):
        return _ctx_value(meta, _CTX_META_KEYS)
    return None

def _auth_headers(provider: ProviderConfig) -> dict[str, str]:
    if provider.api_key_env_var and (key := os.getenv(provider.api_key_env_var)):
        return {"Authorization": f"Bearer {key}"}
    return {}

async def _get_json(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], provider_name: str
) -> Any | None:
    """GET + parse JSON, returning None on any failure (discovery is best effort)."""
    try:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Discovery request to %s (%s) failed: %s", url, provider_name, exc)
        return None

async def _fetch_v1_models(
    provider: ProviderConfig, client: httpx.AsyncClient
) -> list[RawModel]:
    """Parse a provider's OpenAI-compatible ``/models`` endpoint into RawModels.

    ``api_base`` already carries the API version prefix (e.g. ``.../v1``), so the
    endpoint is ``{api_base}/models``.
    """
    url = f"{provider.api_base.rstrip('/')}/models"
    data = await _get_json(client, url, _auth_headers(provider), provider.name)
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [
        RawModel(item["id"], _ctx_from_models_item(item))
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]

def _is_ollama_provider(provider: ProviderConfig) -> bool:
    """Whether to enrich context windows from ollama's native /api/tags."""
    return provider.name == "ollama" or ":11434" in provider.api_base

async def _fetch_ollama_context_lengths(
    provider: ProviderConfig, client: httpx.AsyncClient
) -> dict[str, int]:
    """Map model name -> trained context_length from ollama's ``/api/tags``.

    ollama's /v1/models omits context info, but the native /api/tags lists every
    model with ``details.context_length``. The native API lives at the base host,
    not under the ``/v1`` OpenAI prefix.
    """
    base = provider.api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    data = await _get_json(
        client, f"{base}/api/tags", _auth_headers(provider), provider.name
    )
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return {}
    out: dict[str, int] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        details = m.get("details")
        ctx = details.get("context_length") if isinstance(details, dict) else None
        if (
            isinstance(name, str)
            and isinstance(ctx, int)
            and not isinstance(ctx, bool)
            and ctx > 0
        ):
            out[name] = ctx
    return out

async def fetch_models(
    provider: ProviderConfig,
    *,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    client: httpx.AsyncClient | None = None,
) -> list[RawModel]:
    """Discover a provider's models with their advertised context windows.

    Never raises: any failure yields ``[]`` (or models without context info), so
    the picker never blocks. ollama models are enriched from /api/tags.
    """
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), verify=build_ssl_context()
        )
    try:
        models = await _fetch_v1_models(provider, client)
        if models and _is_ollama_provider(provider):
            tag_ctx = await _fetch_ollama_context_lengths(provider, client)
            if tag_ctx:
                models = [
                    RawModel(m.id, tag_ctx.get(m.id, m.context_length)) for m in models
                ]
        return models
    finally:
        if owns_client:
            await client.aclose()

async def fetch_model_ids(
    provider: ProviderConfig,
    *,
    timeout: float = DEFAULT_DISCOVERY_TIMEOUT,
    client: httpx.AsyncClient | None = None,
) -> list[str]:
    """Return just the model ids from a provider's ``/models`` endpoint.

    Thin wrapper over the OpenAI-compatible /models parse; does not perform
    ollama context enrichment. Never raises (returns ``[]`` on failure).
    """
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout), verify=build_ssl_context()
        )
    try:
        return [m.id for m in await _fetch_v1_models(provider, client)]
    finally:
        if owns_client:
            await client.aclose()

def _ollama_num_ctx_cap() -> int:
    """ollama's served context window (``OLLAMA_CONTEXT_LENGTH``), default 4096.

    A model's trained context_length (e.g. 131072) is only served in full when
    ollama is configured for it; otherwise ollama serves this many tokens and
    silently truncates history. The budget is capped here to avoid that.
    """
    raw = os.getenv("OLLAMA_CONTEXT_LENGTH", "").strip()
    if raw:
        try:
            v = int(raw)
        except ValueError:
            v = 0
        if v > 0:
            return v
    return DEFAULT_OLLAMA_NUM_CTX

def _budget_from_context(context_length: int, *, num_ctx_cap: int | None) -> int:
    """Token budget (``auto_compact_threshold``) from a model's context window.

    Sized to :data:`CONTEXT_BUDGET_SAFETY` of the effective window. For ollama
    the effective window is capped by the served ``num_ctx``.
    """
    effective = (
        min(context_length, num_ctx_cap) if num_ctx_cap is not None else context_length
    )
    return max(1, math.floor(CONTEXT_BUDGET_SAFETY * effective))

def _synth_model(
    provider_name: str,
    model_id: str,
    alias: str,
    *,
    auto_compact_threshold: int | None = None,
) -> ModelConfig:
    kwargs: dict[str, Any] = dict(
        name=model_id,
        provider=provider_name,
        alias=alias,
        input_price=0.0,
        output_price=0.0,
        thinking="off",
    )
    if auto_compact_threshold is not None:
        kwargs["auto_compact_threshold"] = auto_compact_threshold
    return ModelConfig(**kwargs)

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
        *(fetch_models(provider, timeout=timeout) for provider, _ in probe)
    )

    existing_keys = {(m.provider, m.name) for m in config.models}
    seen_aliases = {m.alias for m in config.models}
    discovered: list[DiscoveredModel] = []

    for (provider, ephemeral), raw_models in zip(probe, results, strict=True):
        num_ctx_cap = _ollama_num_ctx_cap() if _is_ollama_provider(provider) else None
        for rm in raw_models:
            if (provider.name, rm.id) in existing_keys:
                continue
            alias = rm.id
            if alias in seen_aliases:
                alias = f"{provider.name}/{rm.id}"
                if alias in seen_aliases:
                    continue
            seen_aliases.add(alias)
            budget = (
                _budget_from_context(rm.context_length, num_ctx_cap=num_ctx_cap)
                if rm.context_length
                else None
            )
            discovered.append(
                DiscoveredModel(
                    model=_synth_model(
                        provider.name, rm.id, alias, auto_compact_threshold=budget
                    ),
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
