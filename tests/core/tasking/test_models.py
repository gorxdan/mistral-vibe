from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError
import pytest

from vibe.core.tasking import (
    TaskBrief,
    TaskBudget,
    TaskManifestIdentity,
    TaskOutcome,
    TaskOutcomeStatus,
)


def _brief(**overrides: object) -> TaskBrief:
    values: dict[str, object] = {
        "objective": "Implement the task contract",
        "inputs": {"target": "vibe/core/tools/builtins/task.py:154"},
        "allowed_paths": ["vibe/core/tasking/**", "vibe/core/tasking/**"],
        "denied_paths": ["vibe/core/agent_loop.py"],
        "acceptance_checks": ["uv run pytest tests/core/tasking"],
        "budget": {"max_tokens": 8_000, "max_cost_usd": 0.25, "max_calls": 4},
        "deadline": "2030-01-02T03:04:05Z",
        "manifest": {"name": "implement-verify", "version": "1", "digest": "abc"},
    }
    values.update(overrides)
    return TaskBrief.model_validate(values)


def test_task_brief_round_trips_structured_contract() -> None:
    brief = _brief()

    assert brief.allowed_paths == ["vibe/core/tasking/**"]
    assert brief.deadline == datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert brief.budget == TaskBudget(max_tokens=8_000, max_cost_usd=0.25, max_calls=4)
    assert brief.manifest.identity == "implement-verify@1#abc"
    assert TaskBrief.model_validate_json(brief.model_dump_json()) == brief


@pytest.mark.parametrize("field", ["allowed_paths", "acceptance_checks"])
def test_task_brief_requires_nonempty_contract_lists(field: str) -> None:
    with pytest.raises(ValidationError):
        _brief(**{field: []})


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "vibe\\core"])
def test_task_brief_rejects_paths_outside_portable_workspace_scope(path: str) -> None:
    with pytest.raises(ValidationError):
        _brief(allowed_paths=[path])


def test_task_brief_rejects_conflicting_scope() -> None:
    with pytest.raises(ValidationError, match="both allowed and denied"):
        _brief(
            allowed_paths=["vibe/core/tasking/**"],
            denied_paths=["vibe/core/tasking/**"],
        )


def test_task_brief_rejects_naive_deadline() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _brief(deadline=datetime(2030, 1, 2, 3, 4, 5))


def test_task_budget_requires_at_least_one_limit() -> None:
    with pytest.raises(ValidationError, match="at least one finite limit"):
        TaskBudget()


def test_task_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _brief(unexpected=True)
    with pytest.raises(ValidationError):
        TaskOutcome.model_validate({
            "status": "failed",
            "summary": "failed",
            "unexpected": True,
        })


def test_task_outcome_exposes_all_terminal_states() -> None:
    assert set(TaskOutcomeStatus) == {
        TaskOutcomeStatus.SUCCEEDED,
        TaskOutcomeStatus.FAILED,
        TaskOutcomeStatus.BLOCKED,
        TaskOutcomeStatus.RETRYABLE,
    }

    outcome = TaskOutcome(
        status=TaskOutcomeStatus.RETRYABLE,
        summary="Provider timed out",
        diagnostics=["timeout"],
        changed_paths=["vibe/core/tasking/models.py"],
        receipt_id="receipt-1",
        remaining_work=["Run focused tests"],
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )

    assert outcome.succeeded is False
    assert outcome.retryable is True
