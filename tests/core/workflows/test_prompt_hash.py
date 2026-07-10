from __future__ import annotations

from vibe.core.workflows.citations import CitationSpec
from vibe.core.workflows.runtime import _prompt_hash


def test_prompt_hash_stable_for_identical_inputs() -> None:
    a = _prompt_hash("p", "explore", "phase1")
    b = _prompt_hash("p", "explore", "phase1")
    assert a == b
    assert len(a) == 64


def test_prompt_hash_differs_by_model() -> None:
    base = _prompt_hash("p", "explore", model=None)
    other = _prompt_hash("p", "explore", model="fast")
    assert base != other


def test_prompt_hash_differs_by_schema() -> None:
    s1 = {"type": "object", "properties": {"a": {"type": "string"}}}
    s2 = {"type": "object", "properties": {"a": {"type": "integer"}}}
    assert _prompt_hash("p", "explore", schema=s1) != _prompt_hash(
        "p", "explore", schema=s2
    )


def test_prompt_hash_differs_by_contract() -> None:
    c1 = {"outputs": [{"path": "a.py"}]}
    c2 = {"outputs": [{"path": "b.py"}]}
    assert _prompt_hash("p", "worker", contract=c1) != _prompt_hash(
        "p", "worker", contract=c2
    )


def test_prompt_hash_schema_key_order_insensitive() -> None:
    s1 = {
        "type": "object",
        "properties": {"b": {"type": "string"}, "a": {"type": "string"}},
    }
    s2 = {
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "type": "object",
    }
    assert _prompt_hash("p", "explore", schema=s1) == _prompt_hash(
        "p", "explore", schema=s2
    )


def test_prompt_hash_binds_full_result_policy() -> None:
    first_citations = CitationSpec(
        items_path="findings", path_field="path", strict=False
    )
    second_citations = CitationSpec(
        items_path="findings", path_field="path", strict=True
    )

    assert _prompt_hash("p", "explore", citation_spec=first_citations) != _prompt_hash(
        "p", "explore", citation_spec=second_citations
    )
    assert _prompt_hash("p", "explore", strip_unknown=True) != _prompt_hash(
        "p", "explore", strip_unknown=False
    )
    assert _prompt_hash("p", "explore", then=None) != _prompt_hash(
        "p", "explore", then="verifier"
    )
