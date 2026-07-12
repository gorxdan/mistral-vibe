from __future__ import annotations

import math
from typing import Any


def cache_read_tokens(usage: dict[str, Any]) -> Any:
    details = _details(usage, "prompt_tokens_details")
    return _first_present(
        details,
        ("cached_tokens", "prompt_cache_hit_tokens"),
        usage,
        ("cached_tokens", "prompt_cache_hit_tokens"),
    )


def cache_write_tokens(usage: dict[str, Any]) -> Any:
    details = _details(usage, "prompt_tokens_details")
    return _first_present(
        details, ("cache_write_tokens",), usage, ("cache_write_tokens",)
    )


def reasoning_tokens(usage: dict[str, Any]) -> Any:
    details = _details(usage, "completion_tokens_details")
    return _first_present(details, ("reasoning_tokens",), usage, ("reasoning_tokens",))


def reported_cost_usd(usage: dict[str, Any]) -> float | None:
    value = usage.get("cost")
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    cost = float(value)
    return cost if math.isfinite(cost) and cost >= 0.0 else None


def _details(usage: dict[str, Any], key: str) -> dict[str, Any]:
    value = usage.get(key)
    return value if isinstance(value, dict) else {}


def _first_present(
    nested: dict[str, Any],
    nested_keys: tuple[str, ...],
    flat: dict[str, Any],
    flat_keys: tuple[str, ...],
) -> Any:
    for source, keys in ((nested, nested_keys), (flat, flat_keys)):
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return 0
