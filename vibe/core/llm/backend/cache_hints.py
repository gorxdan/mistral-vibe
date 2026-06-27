"""Prompt-cache hints for the generic / OpenAI-compatible path.

Non-OpenAI generic providers (DeepSeek, GLM/zai, Together, Groq) auto-cache
prefixes reliably and get no hint — ``build_cache_hint`` returns an inert empty
fragment unless a provider sets an explicit ``provider.cache`` knob. OpenAI is
the exception: its prefix cache load-balances across machines and misses without
a ``prompt_cache_key`` to pin a conversation to one partition, so OpenAI
providers auto-get a stable per-conversation key (see ``_auto_openai_cache_key``
/ ``prefix_cache_key``; the Responses backend uses the same derivation).
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
    provider: ProviderConfig, converted_messages: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return a request-body fragment to merge, or None for no hint.

    For ``anthropic-compat`` the messages are tagged in place (the caller
    serializes this exact list) and an empty fragment is returned.
    """
    cache = getattr(provider, "cache", None)
    if cache is None or cache.mode != "explicit" or cache.style == "off":
        return None

    if cache.style == "passthrough":
        fragment = copy.deepcopy(cache.extra_body)
        key = cache.cache_key or _auto_openai_cache_key(provider, converted_messages)
        if key:
            fragment.setdefault("prompt_cache_key", key)
        return fragment

    if cache.style == "anthropic-compat":
        _tag_anthropic_compat(converted_messages)
        return {}

    return None


def _is_openai_provider(provider: ProviderConfig) -> bool:
    base = (getattr(provider, "api_base", "") or "").lower()
    return (
        getattr(provider, "name", "") == "openai"
        or "api.openai.com" in base
        or "api.sakana.ai" in base
    )


def _auto_openai_cache_key(
    provider: ProviderConfig, converted_messages: list[dict[str, Any]]
) -> str | None:
    """Stable per-conversation ``prompt_cache_key`` for OpenAI.

    OpenAI's prefix auto-cache load-balances requests across machines and misses
    unless ``prompt_cache_key`` pins a conversation to one cache partition — the
    codex reference client sends one (its thread id) for exactly this reason.
    Other generic providers (GLM/zai, DeepSeek, Groq, Together) cache reliably
    without it, so this is OpenAI-only to avoid perturbing their working path.

    Key on the stable prefix — the system prompt plus the first user turn:
    identical across every turn of a conversation (the prefix doesn't change as
    history grows), and distinct across conversations so concurrent sessions
    spread over partitions instead of contending on one machine's cache.
    """
    if not _is_openai_provider(provider):
        return None
    return prefix_cache_key(converted_messages)


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


def _tag_anthropic_compat(messages: list[dict[str, Any]]) -> None:
    """Tag the last system + last user message with an ephemeral cache breakpoint
    (<=2 breakpoints, mirroring the native Anthropic adapter), handling both
    string and already-converted list content.
    """
    sys_idx = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "system"
        ),
        None,
    )
    usr_idx = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
        ),
        None,
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
