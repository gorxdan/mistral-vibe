from __future__ import annotations

import pytest

from vibe.core.repair import repair_json_object


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"answer": "ok"}\n```', {"answer": "ok"}),
        ('prefix {"answer": "ok"} suffix', {"answer": "ok"}),
        ('{"answer": "ok",}', {"answer": "ok"}),
        ('{"items": [1, 2,],}', {"items": [1, 2]}),
    ],
)
def test_repair_json_object_applies_only_conservative_repairs(
    raw: str, expected: dict[str, object]
) -> None:
    result = repair_json_object(raw)

    assert result.value == expected
    assert result.repaired is True
    assert result.raw_text == raw


def test_repair_json_object_refuses_ambiguous_multiple_objects() -> None:
    raw = '{"first": 1} {"second": 2}'

    result = repair_json_object(raw)

    assert result.value is None
    assert result.error is not None
    assert result.raw_text == raw


def test_repair_json_object_refuses_object_outside_single_fence() -> None:
    raw = '{"first": 1}\n```json\n{"second": 2}\n```'

    result = repair_json_object(raw)

    assert result.value is None
    assert result.raw_text == raw


def test_repair_json_object_does_not_invent_missing_values() -> None:
    result = repair_json_object('{"answer":')

    assert result.value is None
    assert result.repaired is False


def test_repair_json_object_reports_non_object_shape() -> None:
    result = repair_json_object('["answer"]')

    assert result.value is None
    assert result.actual_type == "list"
