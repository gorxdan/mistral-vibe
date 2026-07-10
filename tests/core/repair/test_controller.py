from __future__ import annotations

import math

import pytest

from vibe.core.failure_diagnostic import (
    FailureCategory,
    FailureDiagnostic,
    build_failure_diagnostic,
)
from vibe.core.repair import (
    FailureRetryBudget,
    ProgressSnapshot,
    RepairAction,
    RepairController,
    RetryBudgetSet,
)


def _diagnostic(
    category: FailureCategory, message: str, *, retryable: bool = True
) -> FailureDiagnostic:
    return build_failure_diagnostic(
        category=category,
        message=message,
        retryable=retryable,
        suggested_action="apply the smallest targeted correction",
    )


@pytest.fixture
def malformed_failure() -> FailureDiagnostic:
    return _diagnostic(
        FailureCategory.TOOL_ARGUMENT_PARSE, "Malformed JSON at line 1 column 8"
    )


@pytest.fixture
def schema_failure() -> FailureDiagnostic:
    return _diagnostic(FailureCategory.RESULT_SCHEMA, "Unknown field $.answer")


@pytest.fixture
def tool_schema_failure() -> FailureDiagnostic:
    return _diagnostic(
        FailureCategory.TOOL_ARGUMENT_SCHEMA, "Unknown tool argument mode"
    )


@pytest.fixture
def test_failure() -> FailureDiagnostic:
    return _diagnostic(
        FailureCategory.ACCEPTANCE_CHECK, "test_parser still fails at assertion 4"
    )


@pytest.fixture
def transport_failure() -> FailureDiagnostic:
    return _diagnostic(FailureCategory.PROVIDER_TRANSPORT, "provider connection reset")


@pytest.fixture
def budget_failure() -> FailureDiagnostic:
    return _diagnostic(
        FailureCategory.BUDGET, "repair spend limit reached", retryable=False
    )


def _snapshot(
    diagnostic: FailureDiagnostic,
    *,
    diff: str = "diff-a",
    acceptance: str = "failing",
    reads: tuple[str, ...] = ("a.py",),
    tool_effect: str = "exit-1",
) -> ProgressSnapshot:
    return ProgressSnapshot.from_state(
        diff_state=diff,
        error_fingerprint=diagnostic.fingerprint,
        acceptance_state=acceptance,
        newly_read_files=reads,
        tool_effect=tool_effect,
    )


def _controller(category: FailureCategory, maximum: int = 5) -> RepairController:
    return RepairController(
        RetryBudgetSet(
            budgets=(FailureRetryBudget(category=category, max_attempts=maximum),)
        )
    )


@pytest.fixture
def unchanged_progress(test_failure: FailureDiagnostic) -> ProgressSnapshot:
    return _snapshot(test_failure)


@pytest.fixture
def oscillating_progress(
    test_failure: FailureDiagnostic,
) -> tuple[ProgressSnapshot, ProgressSnapshot]:
    return (
        _snapshot(test_failure, diff="diff-a", tool_effect="edit-a"),
        _snapshot(test_failure, diff="diff-b", tool_effect="edit-b"),
    )


def test_malformed_failure_uses_its_finite_category_budget(
    malformed_failure: FailureDiagnostic,
) -> None:
    controller = RepairController.with_finite_defaults()

    decision = controller.observe_failure(
        malformed_failure, _snapshot(malformed_failure), caller_budget_remaining=True
    )

    assert decision.action is RepairAction.CONTINUE
    assert decision.attempt == 1
    assert decision.remaining_attempts == 3


@pytest.mark.parametrize("fixture_name", ["schema_failure", "tool_schema_failure"])
def test_schema_failures_accept_targeted_repair(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    diagnostic: FailureDiagnostic = request.getfixturevalue(fixture_name)
    decision = RepairController.with_finite_defaults().observe_failure(
        diagnostic, _snapshot(diagnostic), caller_budget_remaining=True
    )

    assert decision.action is RepairAction.CONTINUE
    assert decision.category is diagnostic.category


def test_unchanged_failure_warns_then_escalates_when_eligible(
    test_failure: FailureDiagnostic, unchanged_progress: ProgressSnapshot
) -> None:
    controller = _controller(test_failure.category)
    snapshot = unchanged_progress

    first = controller.observe_failure(
        test_failure, snapshot, caller_budget_remaining=True
    )
    warning = controller.observe_failure(
        test_failure, snapshot, caller_budget_remaining=True
    )
    escalation = controller.observe_failure(
        test_failure, snapshot, caller_budget_remaining=True
    )

    assert first.action is RepairAction.CONTINUE
    assert warning.action is RepairAction.WARN
    assert warning.no_progress_strikes == 1
    assert escalation.action is RepairAction.ESCALATE
    assert escalation.no_progress_strikes == 2
    assert escalation.escalation_reason is not None
    assert escalation.metrics.escalation_reason == escalation.escalation_reason


def test_second_repeat_stops_when_caller_budget_is_exhausted(
    test_failure: FailureDiagnostic,
) -> None:
    controller = _controller(test_failure.category)
    snapshot = _snapshot(test_failure)

    controller.observe_failure(test_failure, snapshot, caller_budget_remaining=False)
    controller.observe_failure(test_failure, snapshot, caller_budget_remaining=False)
    stopped = controller.observe_failure(
        test_failure, snapshot, caller_budget_remaining=False
    )

    assert stopped.action is RepairAction.STOP
    assert "caller budget is exhausted" in stopped.reason
    assert stopped.escalation_reason is None


def test_oscillation_counts_repeated_prior_states(
    test_failure: FailureDiagnostic,
    oscillating_progress: tuple[ProgressSnapshot, ProgressSnapshot],
) -> None:
    controller = _controller(test_failure.category, maximum=6)
    state_a, state_b = oscillating_progress

    decisions = [
        controller.observe_failure(test_failure, snapshot, caller_budget_remaining=True)
        for snapshot in (state_a, state_b, state_a, state_b)
    ]

    assert [decision.action for decision in decisions] == [
        RepairAction.CONTINUE,
        RepairAction.CONTINUE,
        RepairAction.WARN,
        RepairAction.ESCALATE,
    ]


def test_new_semantic_state_resets_no_progress_strike(
    test_failure: FailureDiagnostic,
) -> None:
    controller = _controller(test_failure.category)
    unchanged = _snapshot(test_failure)
    changed = _snapshot(
        test_failure,
        diff="diff-b",
        acceptance="different failure",
        reads=("a.py", "b.py"),
        tool_effect="exit-2",
    )

    controller.observe_failure(test_failure, unchanged, caller_budget_remaining=True)
    warning = controller.observe_failure(
        test_failure, unchanged, caller_budget_remaining=True
    )
    progress = controller.observe_failure(
        test_failure, changed, caller_budget_remaining=True
    )

    assert warning.action is RepairAction.WARN
    assert progress.action is RepairAction.CONTINUE
    assert progress.made_progress is True
    assert progress.no_progress_strikes == 0


def test_retry_budgets_are_isolated_by_failure_class(
    schema_failure: FailureDiagnostic, transport_failure: FailureDiagnostic
) -> None:
    controller = RepairController(
        RetryBudgetSet(
            budgets=(
                FailureRetryBudget(category=schema_failure.category, max_attempts=2),
                FailureRetryBudget(category=transport_failure.category, max_attempts=4),
            )
        )
    )

    controller.observe_failure(
        schema_failure, _snapshot(schema_failure), caller_budget_remaining=True
    )
    schema_stop = controller.observe_failure(
        schema_failure,
        _snapshot(schema_failure, diff="diff-b"),
        caller_budget_remaining=True,
    )
    transport_start = controller.observe_failure(
        transport_failure, _snapshot(transport_failure), caller_budget_remaining=True
    )

    assert schema_stop.action is RepairAction.STOP
    assert "Retry budget exhausted" in schema_stop.reason
    assert transport_start.action is RepairAction.CONTINUE
    assert transport_start.attempt == 1
    assert transport_start.remaining_attempts == 3


def test_transport_repeat_can_escalate_to_another_provider(
    transport_failure: FailureDiagnostic,
) -> None:
    controller = RepairController.with_finite_defaults()
    snapshot = _snapshot(transport_failure)

    controller.observe_failure(
        transport_failure, snapshot, caller_budget_remaining=True
    )
    controller.observe_failure(
        transport_failure, snapshot, caller_budget_remaining=True
    )
    escalation = controller.observe_failure(
        transport_failure, snapshot, caller_budget_remaining=True
    )

    assert escalation.action is RepairAction.ESCALATE
    assert escalation.remaining_attempts == 1


def test_noneligible_failure_class_stops_on_second_repeat() -> None:
    diagnostic = _diagnostic(
        FailureCategory.NO_PROGRESS, "repair state repeated without progress"
    )
    controller = _controller(diagnostic.category, maximum=5)
    snapshot = _snapshot(diagnostic)

    controller.observe_failure(diagnostic, snapshot, caller_budget_remaining=True)
    controller.observe_failure(diagnostic, snapshot, caller_budget_remaining=True)
    stopped = controller.observe_failure(
        diagnostic, snapshot, caller_budget_remaining=True
    )

    assert stopped.action is RepairAction.STOP
    assert stopped.escalation_reason is None


def test_budget_failure_stops_without_escalation(
    budget_failure: FailureDiagnostic,
) -> None:
    controller = RepairController.with_finite_defaults()

    decision = controller.observe_failure(
        budget_failure, _snapshot(budget_failure), caller_budget_remaining=True
    )

    assert decision.action is RepairAction.STOP
    assert decision.metrics.finished is True
    assert decision.metrics.recovered is False
    assert decision.escalation_reason is None


def test_recovery_metrics_capture_attempts_tokens_and_cost(
    schema_failure: FailureDiagnostic,
) -> None:
    controller = _controller(schema_failure.category)
    controller.observe_failure(
        schema_failure,
        _snapshot(schema_failure),
        caller_budget_remaining=True,
        added_tokens=100,
        added_cost_usd=0.01,
    )

    recovered = controller.record_recovered(
        schema_failure.category, added_tokens=25, added_cost_usd=0.002
    )

    assert recovered.action is RepairAction.RECOVERED
    assert recovered.metrics.recovered is True
    assert recovered.metrics.finished is True
    assert recovered.metrics.attempts == 1
    assert recovered.metrics.added_tokens == 125
    assert recovered.metrics.added_cost_usd == pytest.approx(0.012)
    assert recovered.metrics.escalation_reason is None


@pytest.mark.parametrize("cost", [math.nan, math.inf, -math.inf])
def test_controller_rejects_nonfinite_cost_before_mutating_state(
    schema_failure: FailureDiagnostic, cost: float
) -> None:
    controller = _controller(schema_failure.category)

    with pytest.raises(ValueError, match="finite"):
        controller.observe_failure(
            schema_failure,
            _snapshot(schema_failure),
            caller_budget_remaining=True,
            added_cost_usd=cost,
        )

    assert controller.metrics(schema_failure.category).attempts == 0
