from __future__ import annotations

from typing import Any

import pytest

from vibe.core.types import AssistantEvent
from vibe.core.workflows.runtime import (
    WorkflowRuntime,
    _coerce_json_safe,
    _dedup_by,
    _flatten,
    _merge_by,
)

# --- unit: the pure synthesis helpers behave as documented ---


def test_flatten_one_level_with_string_and_dict_atoms() -> None:
    # strings, bytes, dicts are atoms (not iterated); scalars pass through.
    assert _flatten([[1, 2], [3], ["ab"], {"k": 1}, 4]) == [1, 2, 3, "ab", {"k": 1}, 4]


def test_flatten_keeps_none_and_scalars() -> None:
    # flatten never silently drops data; None is kept for the script to filter.
    assert _flatten([[1], None, 2]) == [1, None, 2]


def test_flatten_empty() -> None:
    assert _flatten([]) == []


def test_dedup_by_keeps_first_occurrence() -> None:
    items = [{"id": 1, "v": "a"}, {"id": 1, "v": "dup"}, {"id": 2, "v": "b"}]
    out = _dedup_by(items, lambda x: x["id"])
    assert out == [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]


def test_dedup_by_compound_key() -> None:
    findings = [
        {"file": "a.py", "line": 10, "title": "x"},
        {"file": "a.py", "line": 10, "title": "dup"},
        {"file": "a.py", "line": 11, "title": "y"},
    ]
    out = _dedup_by(findings, lambda f: f"{f['file']}:{f['line']}")
    assert len(out) == 2
    assert out[0]["title"] == "x"
    assert out[1]["title"] == "y"


def test_dedup_by_unhashable_key_falls_back_to_id() -> None:
    # A key that returns an unhashable value must not crash; it's kept unique.
    items = [{"k": [1]}, {"k": [1]}]
    out = _dedup_by(items, lambda x: x["k"])  # list key is unhashable
    assert len(out) == 2  # both kept (treated as unique by id())


def test_merge_by_sums_counts_per_group() -> None:
    items = [{"k": "a", "n": 1}, {"k": "a", "n": 2}, {"k": "b", "n": 5}]
    out = _merge_by(items, lambda x: x["k"], lambda a, b: {**a, "n": a["n"] + b["n"]})
    assert out == [{"k": "a", "n": 3}, {"k": "b", "n": 5}]


def test_merge_by_preserves_first_seen_order() -> None:
    items = [{"k": "b"}, {"k": "a"}, {"k": "b"}]
    out = _merge_by(items, lambda x: x["k"], lambda a, b: a)
    assert [x["k"] for x in out] == ["b", "a"]


# --- _coerce_json_safe: snapshot return_value must survive a non-serializable
# value (set/dataclass) by degrading to str, and pass clean JSON through. ---


def test_coerce_json_safe_passes_clean_dict() -> None:
    assert _coerce_json_safe({"a": [1, 2]}) == {"a": [1, 2]}


def test_coerce_json_safe_degrades_set_to_str() -> None:
    # A set is not JSON-serializable; default=str turns it into its repr string
    # rather than crashing the snapshot dump.
    out = _coerce_json_safe({1, 2, 3})
    assert isinstance(out, str)
    assert "1" in out


def test_coerce_json_safe_none_is_noop() -> None:
    assert _coerce_json_safe(None) is None


# --- end-to-end: the helpers are injected into the workflow namespace and
# usable inside a script. ---


class _Loop:
    async def act(self, prompt: str, *, response_format: Any = None) -> Any:
        yield AssistantEvent(content="ok", message_id="a1")

    class stats:  # type: ignore[no-redef]
        session_prompt_tokens = 10
        session_completion_tokens = 5


@pytest.mark.asyncio
async def test_synthesis_helpers_callable_inside_workflow_script() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=lambda prompt, *, agent, parent_context=None: _Loop(),
        max_agents=10,
    )
    script = """
async def main():
    grouped = [["a", "a", "b"], ["b", "c"]]
    flat = flatten(grouped)
    uniq = dedup_by(flat, lambda x: x)
    counts = merge_by(uniq, lambda x: x, lambda a, b: a)
    return {"flat": flat, "unique": len(uniq), "n": len(counts)}
"""
    result = await rt.run(script)
    assert result.return_value == {
        "flat": ["a", "a", "b", "b", "c"],
        "unique": 3,
        "n": 3,
    }


@pytest.mark.asyncio
async def test_snapshot_round_trips_return_value_for_finished_run() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=lambda prompt, *, agent, parent_context=None: _Loop(),
        max_agents=10,
    )
    script = """
async def main():
    return {"report": "all good", "counts": [1, 2, 3]}
"""
    result = await rt.run(script)
    snap = rt.snapshot("wf-1", script, return_value=result.return_value)
    assert snap.return_value == {"report": "all good", "counts": [1, 2, 3]}
    # Re-validates cleanly (the resume read-back path uses model_validate).
    from vibe.core.workflows.models import WorkflowRunSnapshot

    reloaded = WorkflowRunSnapshot.model_validate(snap.model_dump(mode="json"))
    assert reloaded.return_value == {"report": "all good", "counts": [1, 2, 3]}
