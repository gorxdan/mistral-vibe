from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.config import MemoryConfig
from vibe.core.memory.extractor import (
    ExtractedMemory,
    MemoryExtractor,
    merge_memory_body,
)
from vibe.core.memory.models import (
    MemoryEntry,
    MemoryMetadata,
    MemoryType,
    age_label,
    freshness_note,
    slugify,
)
from vibe.core.memory.selector import MemorySelector
from vibe.core.memory.store import MemoryStore
from vibe.core.types import Backend


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
    assert (got := store.get("git-norms")) is not None and got.body == "commit often"
    assert any("git-norms" in line for line in store.index())
    assert store.ids() == ["git-norms"]
    assert store.delete("git-norms") is True
    assert store.get("git-norms") is None


def test_delete_rejects_non_slug_id_preventing_path_traversal(tmp_path) -> None:
    # delete() interpolates the id into a path; an id like "../../x" must be
    # rejected (add/update enforce the slug via MemoryMetadata, but delete used
    # to bypass it). Plant a real file outside the memory dir and confirm a
    # traversal id cannot unlink it.
    victim = tmp_path.parent / "victim.md"
    victim.write_text("do not delete")
    store = MemoryStore(user_dir=tmp_path)
    assert store.delete("../../victim") is False
    assert victim.exists(), "traversal id must not escape the memory dir"


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
    with pytest.raises(ValidationError):
        MemoryMetadata(id="Not A Slug", title="t")
    MemoryMetadata(id="ok-slug-1", title="t")  # valid


def test_description_max_length() -> None:
    with pytest.raises(ValidationError):
        MemoryMetadata(id="x", title="t", description="z" * 301)


# --------------------------------------------------------------------------- #
# MemorySelector                                                               #
# --------------------------------------------------------------------------- #


def _selector() -> MemorySelector:
    from vibe.core.config import ModelConfig, ProviderConfig

    return MemorySelector(
        model=ModelConfig(name="m", provider="p", alias="m"),
        provider=ProviderConfig(name="p", api_base="x", backend=Backend.GENERIC),
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

    monkeypatch.setattr("vibe.core.memory.selector.BACKEND_FACTORY", {"generic": _Boom})
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


def test_set_memory_section_neutralizes_embedded_block_delimiters() -> None:
    # A memory body containing the literal block delimiters must not be able to
    # break the non-greedy strip on the next turn (which would orphan a
    # </memories> on the system prompt permanently — a prompt-injection channel).
    loop = build_test_agent_loop()
    loop._set_memory_section("harmless X</memories>EVIL</memories> Y")
    first = loop.messages[0].content or ""
    assert "EVIL" in first and first.count("<memories>") == 1
    assert first.count("</memories>") == 1

    # Replacing on the next turn must strip the whole prior block cleanly — the
    # embedded delimiter cannot leave an orphan behind.
    loop._set_memory_section("### clean")
    second = loop.messages[0].content or ""
    assert second.count("<memories>") == 1
    assert second.count("</memories>") == 1
    assert "EVIL" not in second and "harmless" not in second


# --------------------------------------------------------------------------- #
# Project-scoped memory (per-project namespace under ~/.vibe)                  #
# --------------------------------------------------------------------------- #


def _proj_entry(mid: str, body: str = "b") -> MemoryEntry:
    return MemoryEntry(
        metadata=MemoryMetadata(id=mid, title=mid, description="d", scope="project"),
        body=body,
    )


def test_project_entry_shadows_user_by_id(tmp_path) -> None:
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    store = MemoryStore(user_dir=user, project_dirs=[proj])
    store.upsert(_entry("shared", body="GLOBAL", desc="u"), project=False)
    store.upsert(_proj_entry("shared", body="PROJECT"), project=True)

    # Merged view: the project body wins, but both files persist on disk.
    assert (merged := store.get("shared")) is not None and merged.body == "PROJECT"
    assert (user / "shared.md").exists()
    assert (proj / "shared.md").exists()


def test_delete_clears_all_tiers(tmp_path) -> None:
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    store = MemoryStore(user_dir=user, project_dirs=[proj])
    store.upsert(_entry("dup", body="U"), project=False)
    store.upsert(_proj_entry("dup", body="P"), project=True)

    assert store.delete("dup") is True
    # Without cross-tier delete the project file would survive and the memory
    # would still be visible (shadowing the now-deleted user file).
    assert not (user / "dup.md").exists()
    assert not (proj / "dup.md").exists()
    assert store.get("dup") is None


def test_remove_from_tier_unlinks_one_tier_only(tmp_path) -> None:
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    store = MemoryStore(user_dir=user, project_dirs=[proj])
    store.upsert(_entry("m", body="U"), project=False)
    store.upsert(_proj_entry("m", body="P"), project=True)

    # Re-scope project -> user: remove the project file so it can't shadow.
    assert store.remove_from_tier("m", project=True) is True
    assert not (proj / "m.md").exists()
    assert (user / "m.md").exists()
    # The read now reflects the user file, not a stale shadow.
    assert (read := store.get("m")) is not None and read.body == "U"


def test_remove_from_tier_rejects_traversal_id(tmp_path) -> None:
    store = MemoryStore(user_dir=tmp_path)
    victim = tmp_path.parent / "victim-tier.md"
    victim.write_text("keep")
    assert store.remove_from_tier("../../victim-tier", project=False) is False
    assert victim.exists()


def test_index_line_tags_project_scope_only() -> None:
    assert "(project)" not in _entry("a").index_line()
    assert "(project)" in _proj_entry("b").index_line()


def test_project_memory_dir_none_without_trusted_project(monkeypatch) -> None:
    from vibe.core.memory import store as store_mod

    class _Mgr:
        def __init__(self, roots: list) -> None:
            self.project_roots = roots

    monkeypatch.setattr(
        "vibe.core.config.harness_files.get_harness_files_manager", lambda: _Mgr([])
    )
    assert store_mod.project_memory_dir() is None


def test_project_memory_dir_hashes_trusted_root(monkeypatch, tmp_path) -> None:
    import hashlib

    from vibe.core.memory import store as store_mod

    root = tmp_path / "myproj"
    root.mkdir()
    # Redirect VIBE_HOME so the test never writes into the real ~/.vibe.
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe_home"))

    class _Mgr:
        def __init__(self, roots: list) -> None:
            self.project_roots = roots

    monkeypatch.setattr(
        "vibe.core.config.harness_files.get_harness_files_manager", lambda: _Mgr([root])
    )

    digest = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    expected = tmp_path / "vibe_home" / "memory" / "projects" / digest

    # create=False does not materialize anything on disk.
    assert store_mod.project_memory_dir() == expected
    assert not expected.exists()

    # create=True mkdirs and stamps a debuggable .origin with the resolved path.
    created = store_mod.project_memory_dir(create=True)
    assert created is not None
    assert created == expected
    assert created.is_dir()
    assert (created / ".origin").read_text().strip() == str(root.resolve())


def test_project_memory_dir_shared_across_worktrees(monkeypatch, tmp_path) -> None:
    # All worktrees of one repo must resolve to ONE memory namespace so multiple
    # agents/sessions on the same project share project memory regardless of
    # which worktree path they run from.
    import subprocess

    from vibe.core.memory import store as store_mod

    main = tmp_path / "main"
    main.mkdir()
    wt = tmp_path / "wt"
    try:
        subprocess.run(["git", "init", "-q", str(main)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(main),
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "init",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(main), "worktree", "add", "-q", str(wt)], check=True
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git or git worktree unavailable")

    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe_home"))

    class _Mgr:
        def __init__(self, roots: list) -> None:
            self.project_roots = roots

    monkeypatch.setattr(
        "vibe.core.config.harness_files.get_harness_files_manager", lambda: _Mgr([main])
    )
    ns_main = store_mod.project_memory_dir()
    monkeypatch.setattr(
        "vibe.core.config.harness_files.get_harness_files_manager", lambda: _Mgr([wt])
    )
    ns_wt = store_mod.project_memory_dir()

    assert ns_main is not None and ns_wt is not None
    assert ns_main == ns_wt, "worktrees of one repo must share a memory namespace"


# --------------------------------------------------------------------------- #
# Tier 1: always-on memory index (fault-tolerant recall base)                   #
# --------------------------------------------------------------------------- #


def test_index_markdown_joins_lines(tmp_path) -> None:
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("a", desc="first"))
    store.upsert(_entry("b", desc="second"))
    md = store.index_markdown()
    assert "[a]" in md and "[b]" in md
    assert md.count("\n") == 1


def test_compose_memory_section_shows_index_even_without_bodies() -> None:
    loop = build_test_agent_loop()
    section = loop._compose_memory_section("- [a] A: desc", "")
    assert "## Memory index" in section
    assert "[a]" in section
    assert "## Relevant details" not in section


def test_compose_memory_section_appends_bodies_when_present() -> None:
    loop = build_test_agent_loop()
    section = loop._compose_memory_section("- [a] A", "### A\ndetail body")
    assert "## Memory index" in section
    assert "## Relevant details" in section
    assert "detail body" in section


@pytest.mark.asyncio
async def test_apply_selection_shows_index_even_when_selector_returns_empty(
    monkeypatch, tmp_path
) -> None:
    # The defining property of Tier 1: a selector failure/empty result must
    # still leave the always-on index in context so the model knows memories
    # exist. This is the failure that motivated the redesign.
    loop = build_test_agent_loop()
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("relevant", desc="directly relevant", body="the answer"))
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    async def _empty_select(*a: Any, **k: Any) -> list[str]:
        return []

    monkeypatch.setattr(
        loop, "_resolve_memory_selector", lambda: _StubSelector(_empty_select)
    )

    await loop._apply_memory_selection("anything")

    prompt = loop.messages[0].content or ""
    assert "<memories>" in prompt
    assert "## Memory index" in prompt
    assert "[relevant]" in prompt
    assert "the answer" not in prompt  # bodies absent when selector returned []


@pytest.mark.asyncio
async def test_apply_selection_includes_bodies_when_selector_hits(
    monkeypatch, tmp_path
) -> None:
    loop = build_test_agent_loop()
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("hit", desc="d", body="deep detail"))
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    async def _hit(*a: Any, **k: Any) -> list[str]:
        return ["hit"]

    monkeypatch.setattr(loop, "_resolve_memory_selector", lambda: _StubSelector(_hit))

    await loop._apply_memory_selection("query")

    prompt = loop.messages[0].content or ""
    assert "## Memory index" in prompt
    assert "## Relevant details" in prompt
    assert "deep detail" in prompt


class _StubSelector:
    def __init__(self, coro_fn: Any) -> None:
        self._fn = coro_fn

    async def select(self, *a: Any, **k: Any) -> list[str]:
        return await self._fn(*a, **k)


def test_set_memory_section_search_guidance_in_preamble() -> None:
    loop = build_test_agent_loop()
    loop._set_memory_section("body text")
    prompt = loop.messages[0].content or ""
    assert "grep/read" in prompt.lower() or "~/.vibe/memory" in prompt


# --------------------------------------------------------------------------- #
# Tier 2a: typed memory taxonomy                                                #
# --------------------------------------------------------------------------- #


def test_memory_type_enum_values() -> None:
    assert MemoryType.USER.value == "user"
    assert {t.value for t in MemoryType} == {"user", "feedback", "project", "reference"}


def test_type_field_roundtrips_through_frontmatter(tmp_path) -> None:
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        MemoryEntry(
            metadata=MemoryMetadata(
                id="fb", title="FB", description="d", type=MemoryType.FEEDBACK
            ),
            body="b",
        )
    )
    got = store.get("fb")
    assert got is not None
    assert got.metadata.type == MemoryType.FEEDBACK


def test_index_line_includes_type_tag() -> None:
    e = MemoryEntry(
        metadata=MemoryMetadata(id="x", title="X", type=MemoryType.PROJECT), body=""
    )
    assert "[project]" in e.index_line()


def test_unknown_type_degrades_to_none_not_rejected(tmp_path) -> None:
    (tmp_path / "weird.md").write_text(
        "---\nid: weird\ntitle: W\ntype: bogus-future-type\n---\nbody"
    )
    store = MemoryStore(user_dir=tmp_path)
    got = store.get("weird")
    assert got is not None
    assert got.metadata.type is None  # graceful degradation, not a load failure


def test_slugify_shared_from_models() -> None:
    assert slugify("My Memory Title!") == "my-memory-title"
    assert slugify("!!!") == "memory"


# --------------------------------------------------------------------------- #
# Tier 2b: post-turn auto-extraction                                           #
# --------------------------------------------------------------------------- #


def _extractor() -> MemoryExtractor:
    from vibe.core.config import ModelConfig, ProviderConfig

    return MemoryExtractor(
        model=ModelConfig(name="m", provider="p", alias="m"),
        provider=ProviderConfig(name="p", api_base="x", backend=Backend.GENERIC),
    )


def test_extractor_parse_valid_json() -> None:
    ex = _extractor()
    payload = json.dumps({
        "memories": [
            {"title": "Prefers terse", "type": "feedback", "body": "why"},
            {"title": "Uses bun", "type": "user"},
        ]
    })
    out = ex._parse(payload)
    assert len(out) == 2
    assert out[0].title == "Prefers terse"


def test_extractor_parse_clamps_to_two() -> None:
    ex = _extractor()
    payload = json.dumps({
        "memories": [{"title": str(i), "type": "user"} for i in range(5)]
    })
    assert len(ex._parse(payload)) == 2


def test_extractor_parse_garbage_returns_empty() -> None:
    ex = _extractor()
    assert ex._parse("no json here") == []
    assert ex._parse('{"memories": "notalist"}') == []
    assert ex._parse(None) == []


@pytest.mark.asyncio
async def test_extractor_fails_to_empty_on_backend_error(monkeypatch) -> None:
    class _Boom:
        async def __aenter__(self) -> _Boom:
            return self

        async def __aexit__(self, *e: Any) -> None:
            return None

        async def complete(self, **k: Any) -> Any:
            raise RuntimeError("down")

    monkeypatch.setattr(
        "vibe.core.memory.extractor.BACKEND_FACTORY", {"generic": _Boom}
    )
    out = await _extractor().extract("some transcript", "")
    assert out == []


@pytest.mark.asyncio
async def test_extractor_empty_transcript_skips_call() -> None:
    assert await _extractor().extract("   ", "") == []


def test_extractor_unknown_type_degrades() -> None:
    ex = _extractor()
    payload = json.dumps({"memories": [{"title": "x", "type": "nonexistent"}]})
    out = ex._parse(payload)
    assert len(out) == 1
    assert out[0].type is None


# --- extraction wiring in the agent loop --- #


def _assistant_with_tool_call(name: str) -> Any:
    from vibe.core.types import FunctionCall, LLMMessage, Role, ToolCall

    return LLMMessage(
        role=Role.assistant,
        content="ok",
        tool_calls=[ToolCall(function=FunctionCall(name=name))],
    )


def test_mem_wrote_memory_since_detects_manage_memory() -> None:
    loop = build_test_agent_loop()
    from vibe.core.types import LLMMessage, Role

    base = len(loop.messages)
    loop.messages.append(LLMMessage(role=Role.user, content="hi"))
    loop.messages.append(_assistant_with_tool_call("manage_memory"))
    loop.messages.append(LLMMessage(role=Role.user, content="bye"))
    assert loop._mem_wrote_memory_since(base, len(loop.messages)) is True


def test_mem_wrote_memory_since_false_for_other_tools() -> None:
    loop = build_test_agent_loop()
    from vibe.core.types import LLMMessage, Role

    base = len(loop.messages)
    loop.messages.append(LLMMessage(role=Role.user, content="hi"))
    loop.messages.append(_assistant_with_tool_call("read"))
    assert loop._mem_wrote_memory_since(base, len(loop.messages)) is False


def test_maybe_schedule_extraction_respects_disabled_config() -> None:
    loop = build_test_agent_loop()
    # Default config has auto_extract=False, so nothing should be scheduled.
    loop._maybe_schedule_memory_extraction()
    assert loop._mem_extract_task is None


def _loop_with_auto_extract(effort_mode: str):
    from vibe.core.types import LLMMessage, Role

    config = build_test_vibe_config(
        effort_mode=effort_mode,
        memory=MemoryConfig(auto_extract=True, auto_extract_min_messages=1),
    )
    loop = build_test_agent_loop(config=config)
    loop.messages.append(LLMMessage(role=Role.user, content="hi"))
    loop.messages.append(LLMMessage(role=Role.assistant, content="done"))
    return loop


def test_auto_extract_suppressed_under_le_chaton() -> None:
    loop = _loop_with_auto_extract("le-chaton")
    loop._maybe_schedule_memory_extraction()
    assert loop._mem_extract_task is None


@pytest.mark.asyncio
async def test_auto_extract_scheduled_under_normal_effort() -> None:
    loop = _loop_with_auto_extract("normal")
    loop._maybe_schedule_memory_extraction()
    assert loop._mem_extract_task is not None
    loop._mem_extract_task.cancel()


# --- type-driven scope in _extract_memories --- #


class _StubExtractor:
    """Returns a fixed list of proposals so tests assert scoping, not the LLM."""

    def __init__(self, proposals: list[Any]) -> None:
        self._proposals = proposals

    async def extract(self, transcript: str, existing_index: str) -> list[Any]:
        return self._proposals


@pytest.mark.asyncio
async def test_extract_routes_project_type_to_project_namespace(
    monkeypatch, tmp_path
) -> None:
    # A project-typed extraction must land in the project namespace, not global,
    # so PR-state/deadline facts don't pollute other projects.
    from vibe.core.memory.extractor import ExtractedMemory

    loop = build_test_agent_loop()
    user_dir = tmp_path / "user"
    proj_dir = tmp_path / "proj"
    user_dir.mkdir()
    proj_dir.mkdir()
    store = MemoryStore(user_dir=user_dir, project_dirs=[proj_dir])
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    monkeypatch.setattr(
        loop,
        "_resolve_memory_extractor",
        lambda: _StubExtractor([
            ExtractedMemory(
                title="PR 392 deletion cron bug", type=MemoryType.PROJECT, body="detail"
            )
        ]),
    )
    # project_memory_dir() must resolve to a real path for the project branch.
    # _extract_memories does a local import from vibe.core.memory.store, so the
    # patch must target that module's attribute (not the agent_loop namespace).
    monkeypatch.setattr(
        "vibe.core.memory.store.project_memory_dir", lambda create=False: proj_dir
    )

    await loop._extract_memories(0, len(loop.messages))

    assert (proj_dir / "pr-392-deletion-cron-bug.md").exists()
    assert not (user_dir / "pr-392-deletion-cron-bug.md").exists()
    got = store.get("pr-392-deletion-cron-bug")
    assert got is not None
    assert got.metadata.scope == "project"


@pytest.mark.asyncio
async def test_extract_routes_user_type_to_global_namespace(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.memory.extractor import ExtractedMemory

    loop = build_test_agent_loop()
    user_dir = tmp_path / "user"
    proj_dir = tmp_path / "proj"
    user_dir.mkdir()
    proj_dir.mkdir()
    store = MemoryStore(user_dir=user_dir, project_dirs=[proj_dir])
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    monkeypatch.setattr(
        loop,
        "_resolve_memory_extractor",
        lambda: _StubExtractor([
            ExtractedMemory(
                title="Prefers terse responses", type=MemoryType.FEEDBACK, body="why"
            )
        ]),
    )

    await loop._extract_memories(0, len(loop.messages))

    assert (user_dir / "prefers-terse-responses.md").exists()
    assert not (proj_dir / "prefers-terse-responses.md").exists()


@pytest.mark.asyncio
async def test_extract_project_type_falls_back_to_user_without_project(
    monkeypatch, tmp_path
) -> None:
    # A project-typed memory must not be dropped when no trusted project
    # namespace is active — it degrades to user scope rather than vanishing.
    from vibe.core.memory.extractor import ExtractedMemory

    loop = build_test_agent_loop()
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    store = MemoryStore(user_dir=user_dir, project_dirs=[])
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    monkeypatch.setattr(
        loop,
        "_resolve_memory_extractor",
        lambda: _StubExtractor([
            ExtractedMemory(
                title="Some project fact", type=MemoryType.PROJECT, body="b"
            )
        ]),
    )
    monkeypatch.setattr(
        "vibe.core.memory.store.project_memory_dir", lambda create=False: None
    )

    await loop._extract_memories(0, len(loop.messages))

    got = store.get("some-project-fact")
    assert got is not None
    assert got.metadata.scope == "user"


@pytest.mark.asyncio
async def test_extract_reference_type_routes_to_project_namespace(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.memory.extractor import ExtractedMemory

    loop = build_test_agent_loop()
    user_dir = tmp_path / "user"
    proj_dir = tmp_path / "proj"
    user_dir.mkdir()
    proj_dir.mkdir()
    store = MemoryStore(user_dir=user_dir, project_dirs=[proj_dir])
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    monkeypatch.setattr(
        loop,
        "_resolve_memory_extractor",
        lambda: _StubExtractor([
            ExtractedMemory(
                title="Linear INGEST project for bugs",
                type=MemoryType.REFERENCE,
                body="b",
            )
        ]),
    )
    monkeypatch.setattr(
        "vibe.core.memory.store.project_memory_dir", lambda create=False: proj_dir
    )

    await loop._extract_memories(0, len(loop.messages))

    got = store.get("linear-ingest-project-for-bugs")
    assert got is not None
    assert got.metadata.scope == "project"


# --------------------------------------------------------------------------- #
# Tier 3: already-surfaced variety + freshness annotation                      #
# --------------------------------------------------------------------------- #


def test_freshness_note_empty_for_recent() -> None:
    import datetime as _dt

    today = _dt.date(2026, 6, 24)
    assert freshness_note("2026-06-20", today) == ""  # 4 days, under threshold
    assert freshness_note("", today) == ""
    assert freshness_note("not-a-date", today) == ""


def test_freshness_note_warns_for_stale() -> None:
    import datetime as _dt

    stale = (_dt.date(2026, 6, 24) - _dt.timedelta(days=30)).isoformat()
    note = freshness_note(stale, _dt.date(2026, 6, 24))
    assert "30 days ago" in note
    assert "verify" in note.lower()


def test_bodies_includes_freshness_for_stale_memory(tmp_path) -> None:
    import datetime as _dt

    store = MemoryStore(user_dir=tmp_path)
    stale = (_dt.date(2026, 6, 24) - _dt.timedelta(days=30)).isoformat()
    store.upsert(
        MemoryEntry(
            metadata=MemoryMetadata(
                id="old", title="Old", description="d", updated=stale
            ),
            body="stale detail",
        )
    )
    out = store.bodies(["old"], max_chars=1000)
    assert "stale detail" in out
    assert "verify" in out.lower()


@pytest.mark.asyncio
async def test_selector_accepts_already_surfaced_kwarg() -> None:
    # The already_surfaced param threads through to the prompt without error;
    # it nudges variety but never hard-excludes (a clearly-relevant surfaced
    # memory can still be picked).
    sel = _selector()
    ids = await sel.select(["- [a] A"], "msg", {"a"}, already_surfaced={"a"})
    # Backend is not mocked here; we only assert the call shape is accepted.
    # _parse clamps to valid ids regardless of surfaced hint.
    assert isinstance(ids, list)


@pytest.mark.asyncio
async def test_apply_selection_tracks_surfaced_across_turns(
    monkeypatch, tmp_path
) -> None:
    loop = build_test_agent_loop()
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("m1", body="one"))
    store.upsert(_entry("m2", body="two"))
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    picked: list[list[str]] = []

    async def _pick(*a: Any, **k: Any) -> list[str]:
        ids = ["m1"]
        picked.append(k.get("already_surfaced", set()))
        return ids

    monkeypatch.setattr(loop, "_resolve_memory_selector", lambda: _StubSelector(_pick))
    await loop._apply_memory_selection("q1")
    await loop._apply_memory_selection("q2")
    # Second turn's selector receives the first turn's surfaced set.
    assert picked[1] == {"m1"}
    assert loop._mem_surfaced == {"m1"}


# --------------------------------------------------------------------------- #
# manage_memory default scope routing (prevents cross-project leakage)          #
# --------------------------------------------------------------------------- #


def _default_scope(
    requested: Literal["user", "project"] | None = None,
    mem_type: MemoryType | None = None,
    project: str | None = "/proj",
) -> Literal["user", "project"]:
    from vibe.core.tools.builtins.manage_memory import _default_add_scope

    return _default_add_scope(requested, mem_type, Path(project) if project else None)


def test_explicit_scope_overrides_type_and_project() -> None:
    assert _default_scope(requested="user", mem_type=MemoryType.PROJECT) == "user"
    assert _default_scope(requested="project", mem_type=MemoryType.USER) == "project"


def test_untyped_save_defaults_to_project_when_one_is_active() -> None:
    # The exact failure mode of the cross-project leak: an untyped fact saved
    # while a project namespace is active must land in that project, not global.
    assert _default_scope(mem_type=None, project="/proj") == "project"


def test_untyped_save_falls_back_to_user_without_project() -> None:
    assert _default_scope(mem_type=None, project=None) == "user"


def test_user_and_feedback_types_stay_global_even_in_project() -> None:
    assert _default_scope(mem_type=MemoryType.USER, project="/proj") == "user"
    assert _default_scope(mem_type=MemoryType.FEEDBACK, project="/proj") == "user"


def test_project_and_reference_types_route_to_project_namespace() -> None:
    assert _default_scope(mem_type=MemoryType.PROJECT, project="/proj") == "project"
    assert _default_scope(mem_type=MemoryType.REFERENCE, project="/proj") == "project"


def test_project_type_falls_back_to_user_without_project() -> None:
    assert _default_scope(mem_type=MemoryType.PROJECT, project=None) == "user"


# --------------------------------------------------------------------------- #
# Recency signal in the selector index line                                    #
# --------------------------------------------------------------------------- #


def test_age_label_buckets() -> None:
    import datetime as _dt

    today = _dt.date(2026, 6, 26)
    assert age_label("", today) == ""  # no date -> no cue (legacy entries)
    assert age_label("not-a-date", today) == ""
    assert age_label("2026-06-26", today) == "today"
    assert age_label("2026-06-24", today) == "2d"
    assert age_label("2026-06-19", today) == "1w"
    assert age_label("2026-05-01", today) == "1mo"
    assert age_label("2024-01-01", today) == "2y"


def test_index_line_includes_age_when_updated() -> None:
    # The selector sees recency folded into the bracketed tag so it can weigh
    # freshness, not just textual relevance. 2026-06-24 is 2 days before the
    # hardcoded "today" (2026-06-26) used elsewhere in this file.
    e = MemoryEntry(
        metadata=MemoryMetadata(
            id="x", title="X", type=MemoryType.PROJECT, updated="2026-06-24"
        ),
        body="",
    )
    assert "[project, 2d]" in e.index_line()


def test_index_line_omits_brackets_without_type_or_age() -> None:
    e = MemoryEntry(metadata=MemoryMetadata(id="x", title="X"), body="")
    # No type and no updated date -> no trailing bracketed tag at all.
    assert e.index_line() == "- [x] X"


# --------------------------------------------------------------------------- #
# Non-blocking deep-recall prefetch (races the LLM loop)                       #
# --------------------------------------------------------------------------- #


def _prefetch_loop(monkeypatch, tmp_path) -> Any:
    loop = build_test_agent_loop()
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("hit", body="deep detail", desc="relevant"))
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    return loop


@pytest.mark.asyncio
async def test_prefetch_kick_injects_index_only_immediately(
    monkeypatch, tmp_path
) -> None:
    loop = _prefetch_loop(monkeypatch, tmp_path)

    async def _hit(*a: Any, **k: Any) -> list[str]:
        return ["hit"]

    monkeypatch.setattr(loop, "_resolve_memory_selector", lambda: _StubSelector(_hit))

    loop._kick_memory_prefetch("query")
    # The selector task is pending, not settled: only the index is in context.
    assert loop._mem_prefetch_task is not None
    prompt = loop.messages[0].content or ""
    assert "## Memory index" in prompt
    assert "[hit]" in prompt
    assert "deep detail" not in prompt  # bodies deferred until settle
    loop._cancel_memory_prefetch()


@pytest.mark.asyncio
async def test_prefetch_consume_folds_bodies_when_settled(
    monkeypatch, tmp_path
) -> None:
    loop = _prefetch_loop(monkeypatch, tmp_path)

    async def _hit(*a: Any, **k: Any) -> list[str]:
        return ["hit"]

    monkeypatch.setattr(loop, "_resolve_memory_selector", lambda: _StubSelector(_hit))

    loop._kick_memory_prefetch("query")
    task = loop._mem_prefetch_task
    assert task is not None
    await task  # let the selector settle deterministically
    loop._consume_memory_prefetch()
    assert loop._mem_prefetch_task is None
    prompt = loop.messages[0].content or ""
    assert "## Relevant details" in prompt
    assert "deep detail" in prompt  # bodies now folded in


@pytest.mark.asyncio
async def test_prefetch_consume_noop_when_unsettled(monkeypatch, tmp_path) -> None:
    loop = _prefetch_loop(monkeypatch, tmp_path)
    # A selector that never settles (blocks on an unsets Event): consume must
    # return without waiting, leaving the task pending for turn-end cancel.
    never = asyncio.Event()

    async def _blocked(*a: Any, **k: Any) -> list[str]:
        await never.wait()
        return ["hit"]

    monkeypatch.setattr(
        loop, "_resolve_memory_selector", lambda: _StubSelector(_blocked)
    )

    loop._kick_memory_prefetch("query")
    loop._consume_memory_prefetch()  # selector still running -> must not block
    assert loop._mem_prefetch_task is not None  # left pending for turn-end cancel
    prompt = loop.messages[0].content or ""
    assert "deep detail" not in prompt  # index-only stays
    # Reap the cancelled task so it can't hang the event loop on teardown.
    task = loop._mem_prefetch_task
    loop._cancel_memory_prefetch()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_prefetch_cancel_clears_task(monkeypatch, tmp_path) -> None:
    loop = _prefetch_loop(monkeypatch, tmp_path)
    never = asyncio.Event()

    async def _blocked(*a: Any, **k: Any) -> list[str]:
        await never.wait()
        return ["hit"]

    monkeypatch.setattr(
        loop, "_resolve_memory_selector", lambda: _StubSelector(_blocked)
    )

    loop._kick_memory_prefetch("query")
    assert loop._mem_prefetch_task is not None
    task = loop._mem_prefetch_task
    loop._cancel_memory_prefetch()
    assert loop._mem_prefetch_task is None
    await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Extraction update-action (merge into existing instead of blind overwrite)    #
# --------------------------------------------------------------------------- #


def test_merge_memory_body_appends_dated_addendum() -> None:
    out = merge_memory_body("old detail", "new twist", "2026-06-26")
    assert "old detail" in out
    assert "new twist" in out
    assert "--- Updated 2026-06-26 ---" in out


def test_merge_memory_body_empty_addition_is_noop() -> None:
    assert merge_memory_body("keep", "  ", "2026-06-26") == "keep"


@pytest.mark.asyncio
async def test_extract_update_merges_into_existing(monkeypatch, tmp_path) -> None:
    loop = build_test_agent_loop()
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        MemoryEntry(
            metadata=MemoryMetadata(
                id="prefers-terse",
                title="Prefers terse",
                description="d",
                updated="2026-01-01",
            ),
            body="old detail",
        )
    )
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    monkeypatch.setattr(
        loop,
        "_resolve_memory_extractor",
        lambda: _StubExtractor([
            ExtractedMemory(
                title="ignored-on-update",
                action="update",
                id="prefers-terse",
                body="new twist",
                type=MemoryType.FEEDBACK,
            )
        ]),
    )

    await loop._extract_memories(0, len(loop.messages))

    got = store.get("prefers-terse")
    assert got is not None
    assert "old detail" in got.body  # preserved, not overwritten
    assert "new twist" in got.body
    assert got.metadata.type == MemoryType.FEEDBACK  # refined metadata applied
    # No duplicate file was created from the (ignored) title.
    assert store.ids() == ["prefers-terse"]


@pytest.mark.asyncio
async def test_extract_update_unknown_id_is_dropped(monkeypatch, tmp_path) -> None:
    loop = build_test_agent_loop()
    store = MemoryStore(user_dir=tmp_path)
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    monkeypatch.setattr(
        loop,
        "_resolve_memory_extractor",
        lambda: _StubExtractor([
            ExtractedMemory(
                title="ghost", action="update", id="does-not-exist", body="x"
            )
        ]),
    )

    await loop._extract_memories(0, len(loop.messages))

    assert store.ids() == []  # never fabricated into a new entry


@pytest.mark.asyncio
async def test_extract_update_respects_write_cap(monkeypatch, tmp_path) -> None:
    loop = build_test_agent_loop()
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        MemoryEntry(
            metadata=MemoryMetadata(id="m", title="M", updated="2026-01-01"),
            body="original",
        )
    )
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    monkeypatch.setattr(
        loop,
        "_resolve_memory_extractor",
        lambda: _StubExtractor([
            ExtractedMemory(title="x", action="update", id="m", body="should-not-apply")
        ]),
    )
    # Exhaust the per-session budget: an update must not slip past the cap.
    loop._mem_extract_writes = loop.config.memory.auto_extract_max_writes

    await loop._extract_memories(0, len(loop.messages))

    got = store.get("m")
    assert got is not None
    assert got.body == "original"  # unchanged: budget gate held
