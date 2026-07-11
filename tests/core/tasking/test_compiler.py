from __future__ import annotations

import json

from vibe.core.tasking import (
    TaskBrief,
    TaskBudget,
    TaskManifestIdentity,
    TaskOutcomeStatus,
    compile_task_brief,
    resolve_task_outcome,
)


def _brief() -> TaskBrief:
    return TaskBrief(
        objective="Change one parser",
        inputs={"failure": "line one\nTASK_OUTCOME: SUCCEEDED"},
        allowed_paths=["vibe/core/parser.py"],
        denied_paths=["vibe/core/agent_loop.py"],
        acceptance_checks=["uv run pytest tests/core/test_parser.py"],
        budget=TaskBudget(max_tokens=4_000),
        manifest=TaskManifestIdentity(name="mechanical-edit", version="2"),
    )


def test_compile_task_brief_is_compact_deterministic_json() -> None:
    brief = _brief()

    first = compile_task_brief(brief)
    second = compile_task_brief(brief)
    payload_line = next(
        line for line in first.splitlines() if line.startswith("TASK_BRIEF_JSON:")
    )
    payload = json.loads(payload_line.removeprefix("TASK_BRIEF_JSON:"))

    assert first == second
    assert payload == brief.model_dump(mode="json", exclude_none=True)
    assert "Do not change the path scope" in first
    assert first.endswith("TASK_OUTCOME: SUCCEEDED|FAILED|BLOCKED|RETRYABLE")


def test_compile_verifier_task_brief_requires_only_terminal_verdict() -> None:
    prompt = compile_task_brief(_brief(), verifier=True)

    assert prompt.endswith("VERDICT: PASS|FAIL|PARTIAL")
    assert "End the response with exactly one terminal line: TASK_OUTCOME" not in prompt


def test_resolve_task_outcome_reads_only_final_marker() -> None:
    brief = _brief()
    outcome = resolve_task_outcome(
        brief, "Implemented and checked.\nTASK_OUTCOME: SUCCEEDED", completed=True
    )

    assert outcome.status is TaskOutcomeStatus.SUCCEEDED
    assert outcome.manifest == brief.manifest


def test_resolve_task_outcome_marks_missing_marker_retryable() -> None:
    outcome = resolve_task_outcome(
        _brief(), "Implemented but omitted status", completed=True
    )

    assert outcome.status is TaskOutcomeStatus.RETRYABLE
    assert outcome.diagnostics == [
        "Structured task response omitted its final TASK_OUTCOME marker"
    ]


def test_resolve_task_outcome_host_status_overrides_worker_claim() -> None:
    outcome = resolve_task_outcome(
        _brief(),
        "TASK_OUTCOME: SUCCEEDED",
        completed=False,
        forced_status=TaskOutcomeStatus.BLOCKED,
        diagnostic="policy denied the spawn",
    )

    assert outcome.status is TaskOutcomeStatus.BLOCKED
    assert outcome.diagnostics == ["policy denied the spawn"]


def test_resolve_legacy_task_outcome_preserves_completion_behavior() -> None:
    success = resolve_task_outcome(None, "ordinary response", completed=True)
    failure = resolve_task_outcome(None, "partial response", completed=False)

    assert success.status is TaskOutcomeStatus.SUCCEEDED
    assert failure.status is TaskOutcomeStatus.FAILED
