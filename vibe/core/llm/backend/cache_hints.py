"""Prompt-cache hints for the generic / OpenAI-compatible path.

Generic providers get no routing key by default. Providers opt in with
``provider.cache.session_keyed`` and choose the documented request field; the
canonical OpenAI and ChatGPT endpoints retain their established default routing.
``build_cache_routing_hint`` is shared by chat completions, Responses, and the
native Mistral backend.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig

_EPHEMERAL = {"type": "ephemeral"}


def build_cache_hint(
    provider: ProviderConfig,
    converted_messages: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    skip_trailing: int = 0,
) -> dict[str, Any] | None:
    """Return a request-body fragment to merge, or None for no hint.

    For ``anthropic-compat`` the messages are tagged in place (the caller
    serializes this exact list) and any routing fragment is returned.

    ``session_id`` is the stable per-conversation routing pin; when given it is
    preferred over the content-hash fallback for cache-routed providers.

    ``skip_trailing`` excludes that many trailing messages (the ephemeral
    late-memory tail) from ``anthropic-compat`` breakpoint placement, so cache
    entries end on the last persisted message and stay prefix-matchable.
    """
    cache = getattr(provider, "cache", None)
    if cache is None or cache.mode != "explicit" or cache.style == "off":
        return None

    routing = build_cache_routing_hint(provider, converted_messages, session_id)

    if cache.style == "passthrough":
        fragment = copy.deepcopy(cache.extra_body)
        for key, value in routing.items():
            fragment.setdefault(key, value)
        return fragment

    if cache.style == "anthropic-compat":
        _tag_anthropic_compat(converted_messages, skip_trailing)
        return routing

    return None


def build_cache_routing_hint(
    provider: ProviderConfig,
    converted_messages: list[dict[str, Any]],
    session_id: str | None = None,
) -> dict[str, str]:
    """Build the provider-documented session-routing body fragment."""
    cache = getattr(provider, "cache", None)
    if cache is None or cache.mode != "explicit" or cache.style == "off":
        return {}
    key = cache.cache_key or _auto_cache_key(
        converted_messages,
        session_id,
        session_keyed=cache.session_keyed or _is_canonical_openai_provider(provider),
    )
    return {cache.session_key_field: key} if key else {}


def _is_canonical_openai_provider(provider: ProviderConfig) -> bool:
    base = (getattr(provider, "api_base", "") or "").lower()
    return (
        getattr(provider, "name", "") == "openai" and "api.openai.com" in base
    ) or getattr(provider, "api_style", "") == "openai-chatgpt"


def _auto_cache_key(
    converted_messages: list[dict[str, Any]],
    session_id: str | None = None,
    *,
    session_keyed: bool = False,
) -> str | None:
    """Stable per-conversation cache key for a load-balanced prefix cache.

    Such a cache load-balances requests across machines and misses unless the
    provider-specific field carrying this key pins a conversation to one
    partition. It is enabled for canonical OpenAI/ChatGPT compatibility and for
    providers that explicitly opt in through ``provider.cache.session_keyed``.

    Prefer the conversation's ``session_id`` (codex keys on its thread_id, a
    per-session UUID): unique per conversation so concurrent sessions spread
    over partitions instead of colliding, and immune to prefix rewrites. Only
    when no session id is threaded through (one-shot callers: memory, summary)
    fall back to a content hash of the stable prefix (system + first user turn),
    which is identical across a conversation's turns and distinct across openings.
    """
    if not session_keyed:
        return None
    return session_id or prefix_cache_key(converted_messages)


def prefix_cache_key(messages: list[dict[str, Any]]) -> str | None:
    """Stable per-conversation cache key from the prefix (system + first user).

    Identical across a conversation's turns (the prefix doesn't change as history
    grows) and distinct across conversations. Shared by the generic and Responses
    OpenAI paths so the routing key is derived the same way. ``messages`` are
    role/content dicts (chat or Responses input items).
    """
    sys_txt = _first_content(messages, "system")
    usr_txt = _first_content(messages, "user")
    if sys_txt is None and usr_txt is None:
        return None
    digest = hashlib.sha256(f"{sys_txt}\x00{usr_txt}".encode()).hexdigest()
    return f"vibe-{digest[:40]}"


def _first_content(messages: list[dict[str, Any]], role: str) -> str | None:
    for m in messages:
        if m.get("role") == role:
            content = m.get("content")
            return (
                content
                if isinstance(content, str)
                else json.dumps(content, sort_keys=True)
            )
    return None


def _tag_anthropic_compat(
    messages: list[dict[str, Any]], skip_trailing: int = 0
) -> None:
    """Tag the last system + last user message with an ephemeral cache breakpoint
    (<=2 breakpoints, mirroring the native Anthropic adapter), handling both
    string and already-converted list content. ``skip_trailing`` messages at the
    end (the ephemeral late-memory tail) are excluded from placement.
    """
    end = len(messages) - 1 - skip_trailing
    sys_idx = next(
        (i for i in range(end, -1, -1) if messages[i].get("role") == "system"), None
    )
    usr_idx = next(
        (i for i in range(end, -1, -1) if messages[i].get("role") == "user"), None
    )
    for idx in {sys_idx, usr_idx}:
        if idx is None:
            continue
        msg = messages[idx]
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": _EPHEMERAL}
            ]
        elif isinstance(content, list) and content:
            last = dict(content[-1])
            last["cache_control"] = _EPHEMERAL
            content[-1] = last
