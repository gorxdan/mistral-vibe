from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any

from vibe.core.failure_diagnostic import (
    FailureCategory,
    FailureDiagnostic,
    build_failure_diagnostic,
)
from vibe.core.repair import ProgressSnapshot, RepairDecision, repair_json_object
from vibe.core.workflows.schema import strip_unknown_properties, validate_against_schema


@dataclass(frozen=True, slots=True)
class WorkflowResultRepair:
    value: dict[str, Any] | None
    errors: tuple[str, ...]
    diagnostic: FailureDiagnostic | None
    route: WorkflowRepairRoute | None = None
    repaired: bool = False

    def handoff(
        self, schema: dict[str, Any], decision: RepairDecision
    ) -> WorkflowRepairHandoff:
        if self.diagnostic is None or self.route is None:
            raise ValueError("successful workflow result has no repair handoff")
        return WorkflowRepairHandoff(
            route=self.route,
            raw_response=self.diagnostic.actual or "",
            schema=schema,
            errors=self.errors,
            diagnostic=self.diagnostic,
            decision=decision,
        )


class WorkflowRepairRoute(StrEnum):
    FORMATTER = auto()
    SEMANTIC = auto()


@dataclass(frozen=True, slots=True)
class WorkflowRepairHandoff:
    route: WorkflowRepairRoute
    raw_response: str
    schema: dict[str, Any]
    errors: tuple[str, ...]
    diagnostic: FailureDiagnostic
    decision: RepairDecision


def repair_workflow_result(
    raw: str, schema: dict[str, Any], *, strip_unknown: bool
) -> WorkflowResultRepair:
    parsed = repair_json_object(raw)
    if parsed.value is None:
        if parsed.actual_type is not None:
            message = "Workflow result must be one JSON object"
            expected = "object matching the result schema"
            error = f"expected object, got {parsed.actual_type}"
        else:
            message = (
                f"Workflow result is not valid JSON: {parsed.error or 'invalid JSON'}"
            )
            expected = "one valid JSON object matching the result schema"
            error = f"JSON parse error: {parsed.error or 'invalid JSON'}"
        diagnostic = build_failure_diagnostic(
            category=FailureCategory.RESULT_SCHEMA,
            message=message,
            field="$",
            expected=expected,
            actual=parsed.raw_text,
            evidence_pointer="workflow.agent.response",
            suggested_action="return only the corrected JSON object without repeating the task",
        )
        return WorkflowResultRepair(
            None, (error,), diagnostic, route=WorkflowRepairRoute.FORMATTER
        )

    value = parsed.value
    if strip_unknown:
        value = strip_unknown_properties(value, schema)
    validation_errors = validate_against_schema(value, schema)
    if not validation_errors:
        return WorkflowResultRepair(value, (), None, repaired=parsed.repaired)

    errors = tuple(str(error) for error in validation_errors)
    first = validation_errors[0]
    diagnostic = build_failure_diagnostic(
        category=FailureCategory.RESULT_SCHEMA,
        message=f"Workflow result schema validation failed: {'; '.join(errors)}",
        field=first.path,
        expected=first.message,
        actual=parsed.raw_text,
        evidence_pointer=f"workflow.agent.response{first.path.removeprefix('$')}",
        suggested_action=(
            f"correct only {first.path} and return the JSON object without repeating "
            "successful investigation"
        ),
    )
    return WorkflowResultRepair(
        None,
        errors,
        diagnostic,
        route=WorkflowRepairRoute.SEMANTIC,
        repaired=parsed.repaired,
    )


def repair_progress_snapshot(
    diagnostic: FailureDiagnostic, errors: tuple[str, ...]
) -> ProgressSnapshot:
    from vibe.core._workspace_verification import workspace_fingerprint

    return ProgressSnapshot.from_state(
        diff_state=workspace_fingerprint() or "workspace-unavailable",
        error_fingerprint=diagnostic.fingerprint,
        acceptance_state=list(errors),
        newly_read_files=(),
        tool_effect={"failure": diagnostic.fingerprint},
    )


def build_repair_prompt(decision: RepairDecision) -> str:
    diagnostic = decision.diagnostic
    if diagnostic is None:
        raise ValueError("repair decision must retain its failure diagnostic")
    warning = (
        f" Repair controller: {decision.reason}." if not decision.made_progress else ""
    )
    return (
        "Correct your previous structured result in this existing conversation. "
        f"{diagnostic.for_model()}{warning} Return only one corrected JSON object. "
        "Do not repeat successful repository exploration or unrelated tool calls."
    )


__all__ = [
    "WorkflowRepairHandoff",
    "WorkflowRepairRoute",
    "WorkflowResultRepair",
    "build_repair_prompt",
    "repair_progress_snapshot",
    "repair_workflow_result",
]
