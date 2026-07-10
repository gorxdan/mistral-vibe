from __future__ import annotations

import math

from pydantic import ValidationError
import pytest

from vibe.core.failure_diagnostic import FailureCategory
from vibe.core.repair import (
    FailureRetryBudget,
    ProgressSnapshot,
    RepairEpisodeMetrics,
    RepairEpisodeOutcome,
    RetryBudgetSet,
)

_ERROR_FINGERPRINT = "0123456789abcdef"


def test_progress_snapshot_canonicalizes_mapping_and_file_order() -> None:
    first = ProgressSnapshot.from_state(
        diff_state={"b": [2, 1], "a": {"changed": True}},
        error_fingerprint=_ERROR_FINGERPRINT,
        acceptance_state={"test_b": "pass", "test_a": "fail"},
        newly_read_files=["b.py", "a.py", "b.py"],
        tool_effect={"stderr": "failed", "exit_code": 1},
    )
    second = ProgressSnapshot.from_state(
        diff_state={"a": {"changed": True}, "b": [2, 1]},
        error_fingerprint=_ERROR_FINGERPRINT,
        acceptance_state={"test_a": "fail", "test_b": "pass"},
        newly_read_files=["a.py", "b.py"],
        tool_effect={"exit_code": 1, "stderr": "failed"},
    )

    assert first == second
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert all(
        len(digest) == 64
        for digest in (
            first.diff_hash,
            first.acceptance_state_hash,
            first.newly_read_files_hash,
            first.tool_effect_hash,
        )
    )


def test_progress_snapshot_changes_when_any_semantic_component_changes() -> None:
    base = ProgressSnapshot.from_state(
        diff_state="diff-a",
        error_fingerprint=_ERROR_FINGERPRINT,
        acceptance_state={"tests": "failed"},
        newly_read_files=["a.py"],
        tool_effect={"exit_code": 1},
    )
    changed = ProgressSnapshot.from_state(
        diff_state="diff-a",
        error_fingerprint=_ERROR_FINGERPRINT,
        acceptance_state={"tests": "passed"},
        newly_read_files=["a.py"],
        tool_effect={"exit_code": 1},
    )

    assert base.semantic_fingerprint != changed.semantic_fingerprint


def test_progress_snapshot_rejects_invalid_digest_shapes() -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        ProgressSnapshot(
            diff_hash="short",
            error_fingerprint=_ERROR_FINGERPRINT,
            acceptance_state_hash="0" * 64,
            newly_read_files_hash="1" * 64,
            tool_effect_hash="2" * 64,
        )
    with pytest.raises(ValidationError, match="16 lowercase hex"):
        ProgressSnapshot.from_state(
            diff_state="diff",
            error_fingerprint="BAD",
            acceptance_state="failed",
            newly_read_files=[],
            tool_effect="none",
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_progress_snapshot_rejects_nonfinite_state(value: float) -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        ProgressSnapshot.from_state(
            diff_state={"cost": value},
            error_fingerprint=_ERROR_FINGERPRINT,
            acceptance_state="failed",
            newly_read_files=[],
            tool_effect="none",
        )


def test_repair_models_are_frozen_and_forbid_extra_fields() -> None:
    snapshot = ProgressSnapshot.from_state(
        diff_state="diff",
        error_fingerprint=_ERROR_FINGERPRINT,
        acceptance_state="failed",
        newly_read_files=[],
        tool_effect="none",
    )

    with pytest.raises(ValidationError):
        snapshot.__setattr__("diff_hash", "0" * 64)
    with pytest.raises(ValidationError):
        FailureRetryBudget.model_validate({
            "category": FailureCategory.RESULT_SCHEMA,
            "max_attempts": 2,
            "unexpected": True,
        })


def test_retry_budget_set_rejects_duplicate_categories() -> None:
    repeated = FailureRetryBudget(
        category=FailureCategory.RESULT_SCHEMA, max_attempts=2
    )
    with pytest.raises(ValidationError, match="each failure category once"):
        RetryBudgetSet(budgets=(repeated, repeated))


def test_finite_defaults_cover_every_failure_category() -> None:
    budgets = RetryBudgetSet.finite_defaults()

    assert {budget.category for budget in budgets.budgets} == set(FailureCategory)
    assert all(
        isinstance(budget.max_attempts, int) and budget.max_attempts >= 0
        for budget in budgets.budgets
    )


@pytest.mark.parametrize("cost", [math.nan, math.inf, -math.inf])
def test_episode_metrics_reject_nonfinite_cost(cost: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        RepairEpisodeMetrics(
            category=FailureCategory.RESULT_SCHEMA,
            outcome=RepairEpisodeOutcome.NOT_RECOVERED,
            finished=False,
            attempts=1,
            added_tokens=10,
            added_cost_usd=cost,
        )
