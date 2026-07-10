from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from vibe.core.tasking import (
    TaskBrief,
    TaskOutcome,
    TaskOutcomeStatus,
    resolve_task_outcome,
)
from vibe.core.tasking._policy import BoundTaskContract
from vibe.core.teams._task_checks import (
    TaskCheckEvidence,
    run_guarded_task_checks,
    task_check_diagnostics,
)
from vibe.core.usage._session import (
    SpendAdmissionBlockedError,
    SpendBudgetExceededError,
)

StructuredRepair = Callable[[str], Awaitable[str | None]]


def _checked_outcome(
    brief: TaskBrief, evidence: tuple[TaskCheckEvidence, ...], mutation: str | None
) -> TaskOutcome:
    summaries = [
        f"{item.name}: exit {item.exit_code} ({item.duration_ms} ms)"
        for item in evidence
    ]
    if mutation is not None:
        return TaskOutcome(
            status=TaskOutcomeStatus.BLOCKED,
            summary="Trusted checks violated the candidate boundary",
            evidence=summaries,
            diagnostics=[mutation],
            manifest=brief.manifest,
        )
    if evidence and all(item.passed for item in evidence):
        return TaskOutcome(
            status=TaskOutcomeStatus.SUCCEEDED,
            summary="Structured task and trusted checks succeeded",
            evidence=summaries,
            manifest=brief.manifest,
        )
    return TaskOutcome(
        status=TaskOutcomeStatus.RETRYABLE,
        summary="Trusted acceptance checks failed",
        evidence=summaries,
        diagnostics=list(task_check_diagnostics(evidence)),
        manifest=brief.manifest,
    )


def _repair_prompt(outcome: TaskOutcome) -> str:
    diagnostics = "\n\n".join(outcome.diagnostics)
    return (
        "Trusted acceptance checks failed. Continue in this same task conversation "
        "and repair only the exact failures below. Do not repeat repository "
        "exploration. Finish with exactly one TASK_OUTCOME terminal line.\n\n"
        f"{diagnostics}"
    )


async def evaluate_structured_attempt(
    brief: TaskBrief,
    contract: BoundTaskContract,
    summary: str | None,
    *,
    repair: StructuredRepair | None = None,
) -> TaskOutcome:
    response = (summary or "").strip()
    failed_checks: TaskOutcome | None = None
    for attempt in range(2):
        reported = resolve_task_outcome(brief, response, completed=True)
        if not reported.succeeded:
            if failed_checks is None:
                return reported
            return reported.model_copy(
                update={
                    "evidence": failed_checks.evidence,
                    "diagnostics": [*failed_checks.diagnostics, *reported.diagnostics],
                }
            )
        evidence, mutation = await asyncio.to_thread(
            run_guarded_task_checks, contract.trusted_checks, contract.workspace_root
        )
        checked = _checked_outcome(brief, evidence, mutation)
        if not checked.retryable or repair is None or attempt == 1:
            return checked
        failed_checks = checked
        try:
            repaired = await repair(_repair_prompt(checked))
        except (SpendAdmissionBlockedError, SpendBudgetExceededError) as exc:
            return TaskOutcome(
                status=TaskOutcomeStatus.BLOCKED,
                summary="Same-worker repair exhausted its bound spend envelope",
                evidence=checked.evidence,
                diagnostics=[*checked.diagnostics, str(exc)],
                manifest=brief.manifest,
            )
        except Exception as exc:
            return checked.model_copy(
                update={
                    "diagnostics": [
                        *checked.diagnostics,
                        f"same-worker repair failed: {type(exc).__name__}: {exc}",
                    ]
                }
            )
        response = (repaired or "").strip()
        if not response:
            return checked
    raise RuntimeError("bounded structured repair loop did not terminate")


__all__ = ["StructuredRepair", "evaluate_structured_attempt"]
