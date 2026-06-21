from __future__ import annotations

from pathlib import Path

from vibe.core.tools.tool_result_store import ToolResultStore, truncate_middle_chars


def test_small_result_unchanged(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    assert store.shape("c1", "hello world", preview_chars=100, hard_cap=100) == (
        "hello world"
    )


def test_result_at_limit_unchanged(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    text = "x" * 100
    assert store.shape("c1", text, preview_chars=12, hard_cap=100) == text


def test_oversized_result_persisted_and_previewed(tmp_path: Path) -> None:
    text = "A" * 80_000 + "B" * 80_000
    store = ToolResultStore(lambda: tmp_path)

    out = store.shape("call-1", text, preview_chars=12_000, hard_cap=100_000)

    # Smaller than the cap: the oversized result no longer floods context.
    assert len(out) < 100_000 + 400
    assert out.startswith("A")  # head preserved
    # Tail is preserved inside the preview, before the persisted-output marker.
    assert "B" * 100 in out
    assert out.rstrip().endswith("retrieve it.]…")
    # Full content persisted to disk and recoverable byte-for-byte.
    assert "persisted to" in out
    assert store.read("call-1") == text


def test_preview_names_a_readable_path(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    out = store.shape("c1", "z" * 1000, preview_chars=100, hard_cap=500)
    persisted = store.path_for("c1")
    assert persisted is not None
    assert str(persisted) in out
    assert persisted.is_file()


def test_unsafe_call_id_sanitized_to_filename(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    store.shape("call/with..bad chars", "z" * 1000, preview_chars=100, hard_cap=500)
    persisted = store.path_for("call/with..bad chars")
    assert persisted is not None
    assert persisted.is_file()


def test_falls_back_to_truncation_when_no_session_dir() -> None:
    store = ToolResultStore(lambda: None)
    text = "A" * 80_000 + "B" * 80_000
    out = store.shape("c1", text, preview_chars=12_000, hard_cap=100_000)
    # No persistence possible: permanent middle-truncation at the hard cap.
    assert len(out) < 100_000 + 200
    assert out.startswith("A")
    assert out.endswith("B")
    assert "elided" in out
    assert store.read("c1") is None


def test_truncate_middle_chars_helper() -> None:
    assert truncate_middle_chars("short", 100) == "short"
    text = "A" * 80 + "B" * 80
    out = truncate_middle_chars(text, 40)
    assert out.startswith("A")
    assert out.endswith("B")
    assert "120 characters elided" in out  # 160 total - 40 kept
