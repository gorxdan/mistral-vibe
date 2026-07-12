from __future__ import annotations

from typing import Any

from vibe.core.lsp._symbols import normalize_document_symbols
from vibe.core.lsp._types import Position, Range


def _range(line: int, start: int, end: int) -> dict[str, Any]:
    return {
        "start": {"line": line, "character": start},
        "end": {"line": line, "character": end},
    }


def _document_symbol(
    name: str, line: int, *, children: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    symbol: dict[str, Any] = {
        "name": name,
        "kind": 12,
        "detail": f"{name}()",
        "range": _range(line, 0, len(name) + 4),
        "selectionRange": _range(line, 4, len(name) + 4),
    }
    if children is not None:
        symbol["children"] = children
    return symbol


def test_normalize_symbol_information_remains_flat() -> None:
    raw = [
        {
            "name": "method",
            "kind": 6,
            "containerName": "Widget",
            "location": {
                "uri": "file:///workspace/widget.py",
                "range": _range(4, 8, 14),
            },
        }
    ]

    symbols = normalize_document_symbols(raw, "file:///ignored.py")

    assert len(symbols) == 1
    symbol = symbols[0]
    assert symbol.name == "method"
    assert symbol.uri == "file:///workspace/widget.py"
    assert symbol.selection_range == Range(
        start=Position(line=4, character=8), end=Position(line=4, character=14)
    )
    assert symbol.depth == 0
    assert symbol.container_path == ()
    assert symbol.container_name == "Widget"
    assert not symbol.hierarchical


def test_normalize_document_symbols_flattens_preorder_with_ancestry() -> None:
    leaf = _document_symbol("leaf", 2)
    inner = _document_symbol("Inner", 1, children=[leaf])
    outer = _document_symbol("Outer", 0, children=[inner])

    symbols = normalize_document_symbols([outer], "file:///workspace/module.py")

    assert [symbol.name for symbol in symbols] == ["Outer", "Inner", "leaf"]
    assert [symbol.depth for symbol in symbols] == [0, 1, 2]
    assert [symbol.container_path for symbol in symbols] == [
        (),
        ("Outer",),
        ("Outer", "Inner"),
    ]
    assert all(symbol.uri == "file:///workspace/module.py" for symbol in symbols)
    assert all(symbol.hierarchical for symbol in symbols)


def test_normalize_document_symbol_preserves_detail_kind_and_selection_range() -> None:
    raw = [_document_symbol("run", 7)]

    [symbol] = normalize_document_symbols(raw, "file:///workspace/module.py")

    assert symbol.kind == 12
    assert symbol.detail == "run()"
    assert symbol.selection_range == Range(
        start=Position(line=7, character=4), end=Position(line=7, character=7)
    )


def test_normalize_document_symbols_handles_mixed_union_members() -> None:
    flat = {
        "name": "flat",
        "kind": 13,
        "location": {"uri": "file:///workspace/flat.py", "range": _range(0, 0, 4)},
    }
    nested = _document_symbol("nested", 1)

    symbols = normalize_document_symbols(
        [flat, nested], "file:///workspace/document.py"
    )

    assert [symbol.name for symbol in symbols] == ["flat", "nested"]
    assert [symbol.hierarchical for symbol in symbols] == [False, True]


def test_normalize_document_symbols_skips_malformed_entries() -> None:
    raw = [
        None,
        {"name": "missing-range", "kind": 12},
        {"name": "missing-end", "selectionRange": {"start": {"line": 0}}},
        _document_symbol("valid", 3, children=[{"name": "broken-child"}]),
    ]

    symbols = normalize_document_symbols(raw, "file:///workspace/module.py")

    assert [symbol.name for symbol in symbols] == ["valid"]


def test_normalize_document_symbols_returns_empty_for_non_list() -> None:
    assert normalize_document_symbols(None, "file:///workspace/module.py") == []
    assert normalize_document_symbols({}, "file:///workspace/module.py") == []


def test_workspace_symbol_without_range_preserves_unknown_position() -> None:
    [symbol] = normalize_document_symbols(
        [{"name": "target", "location": {"uri": "file:///workspace/module.py"}}], ""
    )

    assert symbol.selection_range is None
    assert symbol.position_encoding is None
