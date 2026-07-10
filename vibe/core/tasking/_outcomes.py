from __future__ import annotations

import re

from vibe.core.tasking.models import TaskBrief, TaskOutcome, TaskOutcomeStatus

_OUTCOME_LINE = re.compile(
    r"TASK_OUTCOME:\s*(SUCCEEDED|FAILED|BLOCKED|RETRYABLE)", re.IGNORECASE
)
_SUMMARIES = {
    TaskOutcomeStatus.SUCCEEDED: "Subagent reported that the task succeeded",
    TaskOutcomeStatus.FAILED: "Subagent reported that the task failed",
    TaskOutcomeStatus.BLOCKED: "Subagent could not proceed",
    TaskOutcomeStatus.RETRYABLE: "Task may be retried after correction",
}


def _reported_status(response: str) -> TaskOutcomeStatus | None:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    match = _OUTCOME_LINE.fullmatch(lines[-1])
    if match is None:
        return None
    return TaskOutcomeStatus[match.group(1).upper()]


def resolve_task_outcome(
    brief: TaskBrief | None,
    response: str,
    *,
    completed: bool,
    forced_status: TaskOutcomeStatus | None = None,
    diagnostic: str | None = None,
) -> TaskOutcome:
    diagnostics = [diagnostic] if diagnostic else []
    status = forced_status
    if status is None and not completed:
        status = TaskOutcomeStatus.FAILED
    if status is None and brief is None:
        status = TaskOutcomeStatus.SUCCEEDED
    if status is None:
        status = _reported_status(response)
    if status is None:
        status = TaskOutcomeStatus.RETRYABLE
        diagnostics.append(
            "Structured task response omitted its final TASK_OUTCOME marker"
        )
    return TaskOutcome(
        status=status,
        summary=_SUMMARIES[status],
        diagnostics=diagnostics,
        manifest=brief.manifest if brief else None,
    )
