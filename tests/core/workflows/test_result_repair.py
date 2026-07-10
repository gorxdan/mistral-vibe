from __future__ import annotations

from vibe.core.repair import RepairController
from vibe.core.workflows._result_repair import (
    WorkflowRepairRoute,
    repair_progress_snapshot,
    repair_workflow_result,
)

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def test_invalid_json_exposes_formatter_handoff() -> None:
    repaired = repair_workflow_result("not json", _SCHEMA, strip_unknown=True)
    assert repaired.diagnostic is not None
    decision = RepairController.with_finite_defaults().observe_failure(
        repaired.diagnostic,
        repair_progress_snapshot(repaired.diagnostic, repaired.errors),
        caller_budget_remaining=True,
    )

    handoff = repaired.handoff(_SCHEMA, decision)

    assert handoff.route is WorkflowRepairRoute.FORMATTER
    assert handoff.raw_response == "not json"
    assert handoff.errors[0].startswith("JSON parse error")


def test_valid_json_schema_failure_exposes_semantic_handoff() -> None:
    repaired = repair_workflow_result('{"answer": 42}', _SCHEMA, strip_unknown=True)

    assert repaired.route is WorkflowRepairRoute.SEMANTIC
    assert repaired.diagnostic is not None
    assert repaired.diagnostic.field == "$.answer"
