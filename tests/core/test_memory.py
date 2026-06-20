from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import build_test_agent_loop
from vibe.core.memory.models import MemoryEntry, MemoryMetadata
from vibe.core.memory.selector import MemorySelector
from vibe.core.memory.store import MemoryStore


def _entry(mid: str, body: str = "b", desc: str = "d") -> MemoryEntry:
    return MemoryEntry(
        metadata=MemoryMetadata(id=mid, title=mid, description=desc), body=body
    )


# --------------------------------------------------------------------------- #
# MemoryStore                                                                  #
# --------------------------------------------------------------------------- #


def test_upsert_list_get_delete_roundtrip(tmp_path) -> None:
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("git-norms", body="commit often", desc="git rules"))
    assert store.get("git-norms").body == "commit often"
    assert any("git-norms" in line for line in store.index())
    assert store.ids() == ["git-norms"]
    assert store.delete("git-norms") is True
    assert store.get("git-norms") is None


def test_malformed_file_is_skipped_and_recorded(tmp_path) -> None:
    (tmp_path / "bad.md").write_text("no frontmatter here")
    (tmp_path / "good.md").write_text(
        "---\nid: good\ntitle: Good\ndescription: ok\n---\nbody"
    )
    store = MemoryStore(user_dir=tmp_path)
    assert store.ids() == ["good"]
    assert any("bad.md" in i for i in store.issues)


def test_default_id_from_filename(tmp_path) -> None:
    (tmp_path / "from-name.md").write_text("---\ntitle: T\n---\nbody")
    store = MemoryStore(user_dir=tmp_path)
    assert store.get("from-name") is not None


def test_bodies_respects_char_cap_whole_entry_drop(tmp_path) -> None:
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("a", body="x" * 100))
    store.upsert(_entry("b", body="y" * 100))
    out = store.bodies(["a", "b"], max_chars=130)  # only first fits
    assert "x" * 100 in out
    assert "y" * 100 not in out


def test_mtime_cache_invalidation(tmp_path) -> None:
    store = MemoryStore(user_dir=tmp_path)
    assert store.ids() == []
    store.upsert(_entry("new"))
    assert store.ids() == ["new"]  # picked up after write (cache invalidated)


# --------------------------------------------------------------------------- #
# MemoryMetadata validation                                                    #
# --------------------------------------------------------------------------- #


def test_slug_pattern_enforced() -> None:
    with pytest.raises(Exception):
        MemoryMetadata(id="Not A Slug", title="t")
    MemoryMetadata(id="ok-slug-1", title="t")  # valid


def test_description_max_length() -> None:
    with pytest.raises(Exception):
        MemoryMetadata(id="x", title="t", description="z" * 301)


# --------------------------------------------------------------------------- #
# MemorySelector                                                               #
# --------------------------------------------------------------------------- #


def _selector() -> MemorySelector:
    from vibe.core.config import ModelConfig, ProviderConfig

    return MemorySelector(
        model=ModelConfig(name="m", provider="p", alias="m"),
        provider=ProviderConfig(name="p", api_base="x", backend="generic"),
        max_selected=2,
    )


def test_selector_parse_filters_and_clamps() -> None:
    sel = _selector()
    valid = {"a", "b", "c"}
    assert sel._parse('{"ids": ["a", "x", "b", "a", "c"]}', valid) == ["a", "b"]
    assert sel._parse("garbage", valid) == []
    assert sel._parse('{"ids": "notalist"}', valid) == []


@pytest.mark.asyncio
async def test_selector_fails_to_empty_on_backend_error(monkeypatch) -> None:
    class _Boom:
        def __init__(self, **k: Any) -> None:
            pass

        async def __aenter__(self) -> _Boom:
            return self

        async def __aexit__(self, *e: Any) -> None:
            return None

        async def complete(self, **k: Any) -> Any:
            raise RuntimeError("down")

    monkeypatch.setattr(
        "vibe.core.memory.selector.BACKEND_FACTORY", {"generic": _Boom}
    )
    ids = await _selector().select(["- [a] A"], "user msg", {"a"})
    assert ids == []


@pytest.mark.asyncio
async def test_selector_empty_index_skips_call() -> None:
    assert await _selector().select([], "msg", set()) == []


# --------------------------------------------------------------------------- #
# Injection into the system prompt                                            #
# --------------------------------------------------------------------------- #


def test_set_memory_section_appends_strips_and_replaces() -> None:
    loop = build_test_agent_loop()
    base = loop.messages[0].content or ""

    loop._set_memory_section("### A\nMEMTOKEN_ONE")
    after = loop.messages[0].content or ""
    assert "<memories>" in after and "MEMTOKEN_ONE" in after
    assert after.startswith(base)

    # Replacing (not accumulating): second call strips the prior block.
    loop._set_memory_section("### B\nMEMTOKEN_TWO")
    after2 = loop.messages[0].content or ""
    assert "MEMTOKEN_TWO" in after2 and "MEMTOKEN_ONE" not in after2
    assert after2.count("<memories>") == 1

    # Empty clears the block, restoring the base prompt.
    loop._set_memory_section("")
    assert loop.messages[0].content == base
