from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest

from evals.models import (
    ConfidenceInterval,
    DistributionSummary,
    EvalRun,
    EvaluationDataset,
    RunMetrics,
)
from tests.evals._factories import identity, make_dataset, make_run


def test_run_schema_is_strict_versioned_and_artifact_complete() -> None:
    run = make_run(1, harness_revision="candidate")
    payload: dict[str, Any] = run.model_dump()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs"):
        EvalRun.model_validate(payload)

    payload = run.model_dump()
    payload["schema_version"] = 2
    with pytest.raises(ValidationError, match="Input should be 1"):
        EvalRun.model_validate(payload)

    payload = run.model_dump()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["calls"] = "10"
    with pytest.raises(ValidationError, match="valid integer"):
        EvalRun.model_validate(payload)

    assert run.artifacts.terminal_diff.digest
    assert run.artifacts.check_output.digest
    assert run.artifacts.verification_receipt is not None
    assert run.artifacts.verification_receipt.trusted_valid
    assert run.artifacts.verification_receipt.receipt.digest


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"total_tokens": 999}, "prompt plus completion"),
        ({"reasoning_tokens": 201}, "reasoning tokens"),
        ({"harness_tokens": 99}, "harness prompt plus completion"),
        ({"auxiliary_calls": 1, "auxiliary_results_used": 2}, "used auxiliary"),
        ({"attributed_calls": 11}, "attributed calls"),
        ({"false_done": True, "claimed_success": False}, "false_done"),
        (
            {"false_done": False, "claimed_success": True, "verified_pass": False},
            "false_done",
        ),
    ],
)
def test_run_metrics_reject_inconsistent_accounting(
    updates: dict[str, object], error: str
) -> None:
    payload = make_run(1, harness_revision="candidate").metrics.model_dump()
    payload.update(updates)

    with pytest.raises(ValidationError, match=error):
        RunMetrics.model_validate(payload)


def test_dataset_rejects_duplicate_run_and_trial_identity() -> None:
    first = make_run(1, harness_revision="candidate")

    with pytest.raises(ValidationError, match="duplicate run_id"):
        EvaluationDataset(dataset=make_dataset((first,)).dataset, runs=(first, first))

    duplicate_trial = first.model_copy(update={"run_id": "different-run"})
    with pytest.raises(ValidationError, match="duplicate trial"):
        EvaluationDataset(
            dataset=make_dataset((first,)).dataset, runs=(first, duplicate_trial)
        )


def test_failed_run_explicitly_has_no_receipt_artifact() -> None:
    run = make_run(1, harness_revision="candidate", verified=False)

    assert run.artifacts.verification_receipt is None


def test_verified_pass_cannot_omit_receipt_artifact() -> None:
    run = make_run(1, harness_revision="candidate")
    payload: dict[str, Any] = run.model_dump()
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["verification_receipt"] = None

    with pytest.raises(ValidationError, match="requires trusted verification receipt"):
        EvalRun.model_validate(payload)


def test_verified_pass_requires_trusted_receipt_evidence() -> None:
    payload: dict[str, Any] = make_run(1, harness_revision="candidate").model_dump()
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    evidence = artifacts["verification_receipt"]
    assert isinstance(evidence, dict)
    evidence["trusted_valid"] = False

    with pytest.raises(ValidationError, match="trusted receipt validity"):
        EvalRun.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("task_brief", "task brief"),
        ("repository_fixture", "repository fixture"),
        ("terminal_diff", "terminal diff"),
        ("check_output", "check output"),
        ("harness_config", "harness config"),
    ],
)
def test_verified_pass_requires_matching_receipt_bindings(
    field: str, error: str
) -> None:
    payload: dict[str, Any] = make_run(1, harness_revision="candidate").model_dump()
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    evidence = artifacts["verification_receipt"]
    assert isinstance(evidence, dict)
    evidence[field] = identity(f"stale-{field}").model_dump()

    with pytest.raises(ValidationError, match=error):
        EvalRun.model_validate(payload)


def test_generic_receipt_identity_is_not_sufficient_evidence() -> None:
    payload: dict[str, Any] = make_run(1, harness_revision="candidate").model_dump()
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["verification_receipt"] = identity("unbound-receipt").model_dump()

    with pytest.raises(ValidationError, match="Field required"):
        EvalRun.model_validate(payload)


def test_report_intervals_and_distributions_validate_ordering() -> None:
    with pytest.raises(ValidationError, match="contain its estimate"):
        ConfidenceInterval(estimate=2.0, lower=0.0, upper=1.0, method="test")

    interval = ConfidenceInterval(estimate=2.0, lower=1.0, upper=3.0, method="test")
    with pytest.raises(ValidationError, match="unordered"):
        DistributionSummary(
            count=5, minimum=3.0, median=2.0, maximum=1.0, confidence_interval=interval
        )
