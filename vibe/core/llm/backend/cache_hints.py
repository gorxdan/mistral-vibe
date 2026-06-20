"""Opt-in prompt-cache hints for the generic / OpenAI-compatible path.

Most generic-path providers (OpenAI, DeepSeek, GLM, Together, Groq) auto-cache
prefixes and need no hints — ``build_cache_hint`` returns None for them. This is
a thin escape hatch for the minority of OpenAI-compatible gateways that expose
an explicit cache knob, gated entirely behind ``provider.cache``.
"""

from __future__ import annotations

import copy
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
        if cache.cache_key:
            fragment.setdefault("prompt_cache_key", cache.cache_key)
        return fragment

    if cache.style == "anthropic-compat":
        _tag_anthropic_compat(converted_messages)
        return {}

    return None


def _tag_anthropic_compat(messages: list[dict[str, Any]]) -> None:
    """Tag the last system + last user message with an ephemeral cache breakpoint
    (<=2 breakpoints, mirroring the native Anthropic adapter), handling both
    string and already-converted list content.
    """
    sys_idx = next(
        (i for i, m in enumerate(messages) if m.get("role") == "system"), None
    )
    usr_idx = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
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
