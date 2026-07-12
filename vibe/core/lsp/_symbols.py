from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vibe.core.lsp._types import Range


@dataclass(frozen=True)
class NormalizedSymbol:
    name: str
    kind: int | None
    detail: str | None
    uri: str
    selection_range: Range | None
    position_encoding: str | None
    depth: int
    container_path: tuple[str, ...]
    container_name: str | None
    hierarchical: bool


def normalize_document_symbols(raw: Any, document_uri: str) -> list[NormalizedSymbol]:
    if not isinstance(raw, list):
        return []
    normalized: list[NormalizedSymbol] = []
    for value in raw:
        if not isinstance(value, dict):
            continue
        location = value.get("location")
        if isinstance(location, dict):
            symbol = _normalize_symbol_information(value, location)
            if symbol is not None:
                normalized.append(symbol)
            continue
        normalized.extend(_flatten_document_symbol(value, document_uri, (), 0))
    return normalized


def _normalize_symbol_information(
    value: dict[str, Any], location: dict[str, Any]
) -> NormalizedSymbol | None:
    name = _name(value)
    selection_range = _range(location.get("range"))
    if name is None:
        return None
    return NormalizedSymbol(
        name=name,
        kind=_kind(value),
        detail=_optional_text(value.get("detail")),
        uri=str(location.get("uri", "")),
        selection_range=selection_range,
        position_encoding="utf-16" if selection_range is not None else None,
        depth=0,
        container_path=(),
        container_name=_optional_text(value.get("containerName")),
        hierarchical=False,
    )


def _flatten_document_symbol(
    value: dict[str, Any],
    document_uri: str,
    container_path: tuple[str, ...],
    depth: int,
) -> list[NormalizedSymbol]:
    name = _name(value)
    selection_range = _range(value.get("selectionRange"))
    if name is None or selection_range is None:
        return []
    normalized = [
        NormalizedSymbol(
            name=name,
            kind=_kind(value),
            detail=_optional_text(value.get("detail")),
            uri=document_uri,
            selection_range=selection_range,
            position_encoding="utf-16",
            depth=depth,
            container_path=container_path,
            container_name=None,
            hierarchical=True,
        )
    ]
    children = value.get("children")
    if not isinstance(children, list):
        return normalized
    child_path = (*container_path, name)
    for child in children:
        if isinstance(child, dict):
            normalized.extend(
                _flatten_document_symbol(child, document_uri, child_path, depth + 1)
            )
    return normalized


def _range(value: Any) -> Range | None:
    if not isinstance(value, dict):
        return None
    start = value.get("start")
    end = value.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    if "line" not in start or "character" not in start:
        return None
    if "line" not in end or "character" not in end:
        return None
    try:
        return Range.from_lsp(value)
    except (TypeError, ValueError):
        return None


def _name(value: dict[str, Any]) -> str | None:
    name = str(value.get("name", "")).strip()
    return name or None


def _kind(value: dict[str, Any]) -> int | None:
    raw_kind = value.get("kind")
    try:
        return int(raw_kind) if raw_kind is not None else None
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["NormalizedSymbol", "normalize_document_symbols"]
