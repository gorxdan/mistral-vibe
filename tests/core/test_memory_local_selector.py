from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.config import MemoryConfig
from vibe.core.memory.local_selector import LocalMemorySelector, _select_cached
from vibe.core.memory.models import MemoryEntry, MemoryMetadata, MemoryType
from vibe.core.memory.store import MemoryStore


def _entry(
    memory_id: str,
    *,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    body: str = "body",
    memory_type: MemoryType | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        metadata=MemoryMetadata(
            id=memory_id,
            title=title,
            description=description,
            tags=tags or [],
            type=memory_type,
        ),
        body=body,
    )


def _selector(*, max_selected: int = 2, min_score: float = 3.0):
    return LocalMemorySelector(
        max_selected=max_selected, min_score=min_score, ambiguity_margin=0.15
    )


def test_memory_config_defaults_to_hybrid_local_first() -> None:
    config = MemoryConfig()

    assert config.selector_mode == "hybrid"
    assert config.local_min_score == 3.0
    assert config.local_ambiguity_margin == 0.15
    assert config.max_selected == 2
    assert config.max_inject_chars == 4000


@pytest.mark.parametrize(
    ("entry", "query"),
    [
        (_entry("title-hit", title="Python tooling"), "python tooling"),
        (
            _entry(
                "description-hit",
                title="Release notes",
                description="Use hatch vcs for version tags",
            ),
            "hatch vcs",
        ),
        (
            _entry("tag-hit", title="Deployments", tags=["release-pipeline"]),
            "release pipeline",
        ),
    ],
)
def test_local_selector_scores_metadata_fields(entry, query) -> None:
    result = _selector(max_selected=1).select(
        [entry, _entry("unrelated", title="Terminal colors")], query
    )

    assert result.ids == (entry.id,)
    assert result.ambiguous is False


def test_local_selector_scores_full_index_line() -> None:
    entry = _entry("correction", title="Correction", memory_type=MemoryType.FEEDBACK)

    result = _selector(max_selected=1, min_score=1.0).select([entry], "feedback")

    assert result.ids == ("correction",)


def test_local_selector_no_overlap_is_confident_empty() -> None:
    result = _selector().select(
        [_entry("python", title="Python tooling")], "database migration"
    )

    assert result.ids == ()
    assert result.ambiguous is False


def test_local_selector_marks_tied_cutoff_ambiguous() -> None:
    entries = [
        _entry("alpha", title="Alpha", description="database migration"),
        _entry("beta", title="Beta", description="database migration"),
    ]

    result = _selector(max_selected=1).select(entries, "database migration")

    assert result.ids == ("alpha",)
    assert result.ambiguous is True


def test_local_selector_prefers_unsurfaced_tie() -> None:
    entries = [
        _entry("alpha", title="Alpha", description="database migration"),
        _entry("beta", title="Beta", description="database migration"),
    ]

    result = _selector(max_selected=1).select(
        entries, "database migration", already_surfaced={"alpha"}
    )

    assert result.ids == ("beta",)


def test_local_selector_cache_uses_query_and_index_fingerprint() -> None:
    _select_cached.cache_clear()
    selector = _selector(max_selected=1)
    entries = [_entry("uv", title="UV commands")]

    first = selector.select(entries, "uv commands")
    before_repeat = _select_cached.cache_info()
    repeated = selector.select(entries, "uv commands")
    after_repeat = _select_cached.cache_info()
    changed = selector.select(
        [_entry("uv-revised", title="UV commands revised")], "uv commands"
    )
    after_change = _select_cached.cache_info()

    assert repeated == first
    assert after_repeat.hits == before_repeat.hits + 1
    assert changed.fingerprint != first.fingerprint
    assert after_change.misses == after_repeat.misses + 1


def test_memory_llm_resolvers_share_host_usage_meter() -> None:
    loop = build_test_agent_loop()

    clients = [
        loop._resolve_memory_selector(),
        loop._resolve_memory_extractor(),
        loop._resolve_memory_consolidator(),
        loop._resolve_memory_verifier(),
    ]

    assert all(client is not None for client in clients)
    assert all(client._usage_meter is loop._usage_meter for client in clients if client)


@pytest.mark.asyncio
async def test_hybrid_confident_match_skips_llm_selector(monkeypatch, tmp_path) -> None:
    config = build_test_vibe_config(
        memory=MemoryConfig(inject_mode="system", selector_mode="hybrid")
    )
    loop = build_test_agent_loop(config=config)
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        _entry(
            "uv-commands",
            title="UV commands",
            description="Run Python tools through uv",
            body="USE_UV_BODY",
        )
    )
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    def _unexpected_selector() -> None:
        raise AssertionError("confident local recall must not call the LLM selector")

    monkeypatch.setattr(loop, "_resolve_memory_selector", _unexpected_selector)

    await loop._apply_memory_selection("use uv commands")

    prompt = loop.messages[0].content or ""
    assert "USE_UV_BODY" in prompt
    assert "## Memory index" in prompt


@pytest.mark.asyncio
async def test_hybrid_ambiguous_match_uses_llm_selector(monkeypatch, tmp_path) -> None:
    config = build_test_vibe_config(
        memory=MemoryConfig(
            inject_mode="system", selector_mode="hybrid", max_selected=1
        )
    )
    loop = build_test_agent_loop(config=config)
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        _entry(
            "alpha", title="Alpha", description="database migration", body="ALPHA_BODY"
        )
    )
    store.upsert(
        _entry("beta", title="Beta", description="database migration", body="BETA_BODY")
    )
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    class _Selector:
        async def select(self, *args: Any, **kwargs: Any) -> list[str]:
            return ["beta"]

    monkeypatch.setattr(loop, "_resolve_memory_selector", _Selector)

    await loop._apply_memory_selection("database migration")

    prompt = loop.messages[0].content or ""
    assert "BETA_BODY" in prompt
    assert "ALPHA_BODY" not in prompt


def test_prefetch_injects_confident_local_body_without_task(
    monkeypatch, tmp_path
) -> None:
    config = build_test_vibe_config(
        memory=MemoryConfig(inject_mode="system", selector_mode="hybrid")
    )
    loop = build_test_agent_loop(config=config)
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        _entry(
            "uv-commands",
            title="UV commands",
            description="Run Python tools through uv",
            body="USE_UV_BODY",
        )
    )
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    loop._kick_memory_prefetch("use uv commands")

    assert loop._mem_prefetch_task is None
    assert "USE_UV_BODY" in (loop.messages[0].content or "")


@pytest.mark.asyncio
async def test_late_prefetch_completion_is_consumed_after_first_poll(
    monkeypatch, tmp_path
) -> None:
    config = build_test_vibe_config(
        memory=MemoryConfig(inject_mode="system", selector_mode="llm")
    )
    loop = build_test_agent_loop(config=config)
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("late", title="Late result", body="LATE_BODY"))
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)
    release = asyncio.Event()

    class _Selector:
        async def select(self, *args: Any, **kwargs: Any) -> list[str]:
            await release.wait()
            return ["late"]

    monkeypatch.setattr(loop, "_resolve_memory_selector", _Selector)

    loop._kick_memory_prefetch("late result")
    task = loop._mem_prefetch_task
    assert task is not None
    loop._consume_memory_prefetch()
    assert loop._mem_prefetch_task is task

    release.set()
    await task
    await asyncio.sleep(0)

    assert loop._mem_prefetch_task is None
    assert "LATE_BODY" in (loop.messages[0].content or "")


@pytest.mark.asyncio
async def test_local_recall_keeps_pinned_index_under_inject_budget(
    monkeypatch, tmp_path
) -> None:
    config = build_test_vibe_config(
        memory=MemoryConfig(
            inject_mode="system",
            selector_mode="local",
            max_selected=1,
            index_max_chars=45,
        )
    )
    loop = build_test_agent_loop(config=config)
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(
        _entry(
            "pinned", title="Shell", description="Use fish", memory_type=MemoryType.USER
        )
    )
    store.upsert(
        _entry(
            "atlas-parser",
            title="Atlas parser",
            description="Parser implementation details",
            body="ATLAS_BODY",
            memory_type=MemoryType.PROJECT,
        )
    )
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    await loop._apply_memory_selection("atlas parser")

    prompt = loop.messages[0].content or ""
    index = prompt.split("## Relevant details", maxsplit=1)[0]
    assert "[pinned]" in index
    assert "[atlas-parser]" not in index
    assert "ATLAS_BODY" in prompt


@pytest.mark.asyncio
async def test_local_recall_preserves_per_session_selection(
    monkeypatch, tmp_path
) -> None:
    config = build_test_vibe_config(
        memory=MemoryConfig(
            inject_mode="system",
            selector_mode="local",
            select_mode="per-session",
            max_selected=1,
        )
    )
    loop = build_test_agent_loop(config=config)
    store = MemoryStore(user_dir=tmp_path)
    store.upsert(_entry("alpha", title="Alpha guide", body="ALPHA_BODY"))
    store.upsert(_entry("beta", title="Beta guide", body="BETA_BODY"))
    monkeypatch.setattr(loop, "_get_memory_store", lambda: store)

    await loop._apply_memory_selection("alpha guide")
    await loop._apply_memory_selection("beta guide")

    prompt = loop.messages[0].content or ""
    assert "ALPHA_BODY" in prompt
    assert "BETA_BODY" not in prompt
