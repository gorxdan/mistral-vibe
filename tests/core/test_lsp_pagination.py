from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest

from vibe.core.lsp._pagination import (
    LspContinuationError,
    LspContinuationReloadRequired,
    LspContinuationSerializationError,
    LspContinuationStore,
    LspQueryBinding,
)


def _binding(**changes: Any) -> LspQueryBinding:
    binding = LspQueryBinding(
        operation="find_references",
        file_path="/workspace/src/main.py",
        line=10,
        character=4,
        query=None,
        session_id="session-1",
        task_brief_hash="brief-1",
        lsp_generation=7,
        workspace_root="/workspace",
    )
    return replace(binding, **changes)


def _items(count: int) -> list[dict[str, Any]]:
    return [
        {"name": f"item-{index}", "location": {"line": index}} for index in range(count)
    ]


def test_pages_snapshot_without_losing_items() -> None:
    store = LspContinuationStore()
    binding = _binding()
    expected = _items(7)

    first = store.first_page(binding, expected, page_size=3)
    second = store.get_page(first.continuation_token or "", binding)
    third = store.get_page(second.continuation_token or "", binding)

    assert first.offset == 0
    assert second.offset == 3
    assert third.offset == 6
    assert first.returned_count == 3
    assert second.returned_count == 3
    assert third.returned_count == 1
    assert first.total_count == second.total_count == third.total_count == 7
    assert third.continuation_token is None
    assert [*first.items, *second.items, *third.items] == expected


def test_exact_boundary_does_not_create_snapshot() -> None:
    store = LspContinuationStore()

    page = store.first_page(_binding(), _items(3), page_size=3)

    assert page.continuation_token is None
    assert not page.has_more
    assert store.snapshot_count == 0


def test_tokens_are_opaque_and_do_not_embed_binding_values() -> None:
    store = LspContinuationStore()
    binding = _binding(query="SensitiveSymbol")

    page = store.first_page(binding, _items(2), page_size=1)
    token = page.continuation_token or ""

    assert binding.file_path is not None
    assert binding.query is not None
    assert binding.session_id is not None
    assert binding.task_brief_hash is not None
    assert token.startswith("lspc1.")
    assert binding.file_path not in token
    assert binding.query not in token
    assert binding.session_id not in token
    assert binding.task_brief_hash not in token


def test_replaying_token_returns_independent_identical_page() -> None:
    store = LspContinuationStore()
    binding = _binding()
    first = store.first_page(binding, _items(5), page_size=2)
    token = first.continuation_token or ""

    replayed = store.get_page(token, binding)
    mutated = replayed.items
    mutated[0]["name"] = "changed"

    again = store.get_page(token, binding)

    assert again == replayed
    assert again.items[0]["name"] == "item-2"
    assert again.continuation_token == replayed.continuation_token


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("operation", "go_to_definition"),
        ("file_path", "/workspace/src/other.py"),
        ("line", 11),
        ("character", 5),
        ("query", "OtherSymbol"),
        ("session_id", "session-2"),
        ("task_brief_hash", "brief-2"),
        ("lsp_generation", 8),
        ("workspace_root", "/other-workspace"),
    ],
)
def test_token_is_bound_to_exact_query_and_context(field_name: str, value: Any) -> None:
    store = LspContinuationStore()
    binding = _binding()
    first = store.first_page(binding, _items(2), page_size=1)

    with pytest.raises(LspContinuationError, match="Invalid or expired"):
        store.get_page(
            first.continuation_token or "", replace(binding, **{field_name: value})
        )


@pytest.mark.parametrize("token", ["", "invalid", "lspc1.not-base64!", "x" * 300])
def test_malformed_tokens_use_generic_error(token: str) -> None:
    store = LspContinuationStore()

    with pytest.raises(LspContinuationError) as exc_info:
        store.get_page(token, _binding())

    assert str(exc_info.value) == (
        "Invalid or expired LSP continuation token; rerun the original query."
    )


def test_tampered_and_cross_store_tokens_use_generic_error() -> None:
    first_store = LspContinuationStore()
    second_store = LspContinuationStore()
    binding = _binding()
    page = first_store.first_page(binding, _items(2), page_size=1)
    token = page.continuation_token or ""
    tampered = f"{token[:-1]}{'A' if token[-1] != 'A' else 'B'}"

    for candidate_store, candidate_token in (
        (first_store, tampered),
        (second_store, token),
    ):
        with pytest.raises(LspContinuationError, match="Invalid or expired"):
            candidate_store.get_page(candidate_token, binding)


def test_ttl_is_absolute_and_replay_does_not_extend_it() -> None:
    now = [10.0]
    store = LspContinuationStore(ttl_seconds=5, clock=lambda: now[0])
    binding = _binding()
    first = store.first_page(binding, _items(3), page_size=1)
    token = first.continuation_token or ""
    now[0] = 14.0

    store.get_page(token, binding)
    now[0] = 15.0

    with pytest.raises(LspContinuationError, match="Invalid or expired"):
        store.get_page(token, binding)
    assert store.snapshot_count == 0


def test_lru_eviction_prefers_least_recently_used_snapshot() -> None:
    store = LspContinuationStore(max_snapshots=2)
    one = _binding(query="one")
    two = _binding(query="two")
    three = _binding(query="three")
    one_token = store.first_page(one, _items(3), page_size=1).continuation_token or ""
    two_token = store.first_page(two, _items(3), page_size=1).continuation_token or ""
    store.get_page(one_token, one)

    store.first_page(three, _items(3), page_size=1)

    store.get_page(one_token, one)
    with pytest.raises(LspContinuationError, match="Invalid or expired"):
        store.get_page(two_token, two)
    assert store.snapshot_count == 2


def test_oversized_snapshot_reloads_and_validates_digest() -> None:
    store = LspContinuationStore(max_retained_bytes=128, max_snapshot_bytes=128)
    binding = _binding()
    items = [{"name": f"item-{index}", "body": "x" * 200} for index in range(4)]
    first = store.first_page(binding, items, page_size=2)
    token = first.continuation_token or ""

    assert store.retained_bytes == 0
    with pytest.raises(LspContinuationReloadRequired):
        store.get_page(token, binding)

    second = store.get_page(token, binding, reloaded_items=items)

    assert [item["name"] for item in second.items] == ["item-2", "item-3"]
    assert second.continuation_token is None


def test_oversized_snapshot_rejects_changed_reload_and_invalidates_token() -> None:
    store = LspContinuationStore(max_snapshot_bytes=1)
    binding = _binding()
    items = _items(3)
    first = store.first_page(binding, items, page_size=1)
    token = first.continuation_token or ""
    changed = [*items]
    changed[2] = {"name": "different"}

    with pytest.raises(LspContinuationError, match="Invalid or expired"):
        store.get_page(token, binding, reloaded_items=changed)
    with pytest.raises(LspContinuationError, match="Invalid or expired"):
        store.get_page(token, binding, reloaded_items=items)


def test_reload_digest_ignores_mapping_key_order() -> None:
    store = LspContinuationStore(max_snapshot_bytes=1)
    binding = _binding()
    original = [{"name": "one", "line": 1}, {"name": "two", "line": 2}]
    reordered_keys = [{"line": 1, "name": "one"}, {"line": 2, "name": "two"}]
    first = store.first_page(binding, original, page_size=1)

    page = store.get_page(
        first.continuation_token or "", binding, reloaded_items=reordered_keys
    )

    assert page.items == ({"line": 2, "name": "two"},)


def test_retained_memory_never_exceeds_cap() -> None:
    store = LspContinuationStore(
        max_snapshots=4, max_retained_bytes=500, max_snapshot_bytes=500
    )

    for index in range(8):
        store.first_page(_binding(query=f"query-{index}"), _items(3), page_size=1)
        assert store.retained_bytes <= 500
        assert store.snapshot_count <= 4


def test_concurrent_replay_is_stable() -> None:
    store = LspContinuationStore()
    binding = _binding()
    first = store.first_page(binding, _items(8), page_size=2)
    token = first.continuation_token or ""

    with ThreadPoolExecutor(max_workers=8) as executor:
        pages = list(executor.map(lambda _: store.get_page(token, binding), range(32)))

    assert all(page == pages[0] for page in pages)


def test_clear_invalidates_all_tokens() -> None:
    store = LspContinuationStore()
    binding = _binding()
    first = store.first_page(binding, _items(2), page_size=1)
    store.clear()

    with pytest.raises(LspContinuationError, match="Invalid or expired"):
        store.get_page(first.continuation_token or "", binding)
    assert store.snapshot_count == 0
    assert store.retained_bytes == 0


def test_rejects_non_json_items_without_exposing_item() -> None:
    store = LspContinuationStore()

    with pytest.raises(
        LspContinuationSerializationError, match="must be JSON-serializable"
    ) as exc_info:
        store.first_page(_binding(), [object()], page_size=1)

    assert "object at" not in str(exc_info.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl_seconds": 0},
        {"max_snapshots": 0},
        {"max_retained_bytes": -1},
        {"max_snapshot_bytes": -1},
        {"max_page_size": 0},
    ],
)
def test_rejects_invalid_store_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LspContinuationStore(**kwargs)


@pytest.mark.parametrize("page_size", [0, 3])
def test_rejects_invalid_page_size(page_size: int) -> None:
    store = LspContinuationStore(max_page_size=2)

    with pytest.raises(ValueError, match="page_size"):
        store.first_page(_binding(), _items(2), page_size=page_size)
