from __future__ import annotations

from typing import Any

import pytest

from evals.gates import compare_datasets
from evals.models import ComparisonReport, EvalRun, TaskCategory
from tests.evals._factories import identity, make_dataset, make_run, make_trials


def _gate(report: ComparisonReport, name: str):
    return next(gate for gate in report.gates if gate.name == name)


def _replace_comparable_artifact(run: EvalRun, field: str) -> EvalRun:
    payload: dict[str, Any] = run.model_dump()
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    replacement = identity(f"different-{field}").model_dump()
    artifacts[field] = replacement
    evidence = artifacts["verification_receipt"]
    assert isinstance(evidence, dict)
    if field in evidence:
        evidence[field] = replacement
    return EvalRun.model_validate(payload)


def _release_trials(harness_revision: str, **overrides: Any) -> tuple[EvalRun, ...]:
    return (
        *make_trials(harness_revision, task_name="core", **overrides),
        *make_trials(
            harness_revision,
            task_name="policy",
            task_category=TaskCategory.POLICY,
            **overrides,
        ),
        *make_trials(
            harness_revision,
            task_name="security",
            task_category=TaskCategory.SECURITY,
            **overrides,
        ),
    )


def _passing_report(*, release_gate: bool = True) -> ComparisonReport:
    baseline = make_dataset(
        _release_trials(
            "baseline",
            total_cost_usd=10.0,
            harness_share=0.20,
            maintenance_share=0.05,
            calls=100,
            attributed_calls=99,
        )
    )
    candidate = make_dataset(
        _release_trials(
            "candidate",
            total_cost_usd=7.0,
            harness_share=0.20,
            maintenance_share=0.05,
            calls=100,
            attributed_calls=99,
        )
    )
    return compare_datasets(baseline, candidate, release_gate=release_gate)


def test_release_gates_pass_at_required_thresholds() -> None:
    report = _passing_report()

    assert report.passed
    assert _gate(report, "minimum_trials_per_group").actual == 5.0
    assert _gate(report, "comparable_artifacts").passed
    assert _gate(report, "false_done_rate").passed
    assert _gate(report, "policy_security_false_done").passed
    assert _gate(report, "policy_security_unsafe_mutation").passed
    cost_gate = _gate(report, "weakest_profile_cost_per_verified_pass_improvement")
    assert cost_gate.actual == 0.3
    assert cost_gate.passed
    assert cost_gate.confidence_interval is not None
    assert _gate(report, "weakest_profile_pass_at_1_drop").actual == 0.0
    assert _gate(report, "harness_utilization").actual == 0.2
    assert _gate(report, "maintenance_utilization").actual == 0.05
    assert _gate(report, "spend_attribution_completeness").actual == 0.99


def test_release_mode_requires_five_trials_per_group() -> None:
    baseline = make_dataset(make_trials("baseline", count=4, total_cost_usd=10.0))
    candidate = make_dataset(make_trials("candidate", count=4, total_cost_usd=6.0))

    report = compare_datasets(baseline, candidate, release_gate=True)

    gate = _gate(report, "minimum_trials_per_group")
    assert not report.passed
    assert not gate.passed
    assert gate.actual == 4.0


def test_release_mode_fails_without_policy_or_security_fixtures() -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate = make_dataset(make_trials("candidate", total_cost_usd=6.0))

    report = compare_datasets(baseline, candidate, release_gate=True)

    assert not report.passed
    for name in ("policy_security_false_done", "policy_security_unsafe_mutation"):
        gate = _gate(report, name)
        assert not gate.passed
        assert gate.actual is None
        assert "no policy/security fixture" in gate.detail


def test_candidate_cannot_skew_comparison_with_extra_trials() -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate = make_dataset((
        *make_trials("candidate", total_cost_usd=6.0),
        make_run(6, harness_revision="candidate", total_cost_usd=6.0),
    ))

    report = compare_datasets(baseline, candidate, release_gate=True)

    assert not _gate(report, "trial_alignment").passed


def test_candidate_trials_must_use_matching_random_seeds() -> None:
    baseline_runs = tuple(
        run.model_copy(update={"random_seed": run.trial_index})
        for run in make_trials("baseline", total_cost_usd=10.0)
    )
    candidate_runs = tuple(
        run.model_copy(update={"random_seed": run.trial_index + 100})
        for run in make_trials("candidate", total_cost_usd=6.0)
    )

    report = compare_datasets(
        make_dataset(baseline_runs), make_dataset(candidate_runs), release_gate=True
    )

    assert not _gate(report, "trial_alignment").passed


@pytest.mark.parametrize(
    "field", ["repository_fixture", "task_brief", "recipe", "pricing_table", "policy"]
)
def test_comparison_requires_matching_evaluation_artifacts(field: str) -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate_runs = list(make_trials("candidate", total_cost_usd=6.0))
    candidate_runs[0] = _replace_comparable_artifact(candidate_runs[0], field)

    report = compare_datasets(
        baseline, make_dataset(tuple(candidate_runs)), release_gate=True
    )

    gate = _gate(report, "comparable_artifacts")
    assert not gate.passed
    assert field in gate.detail


@pytest.mark.parametrize(
    ("gate_name", "candidate_kwargs"),
    [
        ("weakest_profile_cost_per_verified_pass_improvement", {"total_cost_usd": 7.1}),
        ("harness_utilization", {"total_cost_usd": 6.0, "harness_share": 0.21}),
        ("maintenance_utilization", {"total_cost_usd": 6.0, "maintenance_share": 0.06}),
    ],
)
def test_release_threshold_regressions_fail(
    gate_name: str, candidate_kwargs: dict[str, Any]
) -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate = make_dataset(make_trials("candidate", **candidate_kwargs))

    report = compare_datasets(baseline, candidate, release_gate=True)

    assert not _gate(report, gate_name).passed


def test_single_favorable_cost_trial_cannot_carry_cost_gate() -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate = make_dataset(
        tuple(
            make_run(trial, harness_revision="candidate", total_cost_usd=cost)
            for trial, cost in enumerate((1.0, 8.5, 8.5, 8.5, 8.5), start=1)
        )
    )

    report = compare_datasets(baseline, candidate, release_gate=True)

    gate = _gate(report, "weakest_profile_cost_per_verified_pass_improvement")
    assert gate.confidence_interval is not None
    assert gate.confidence_interval.estimate == 0.3
    assert gate.confidence_interval.lower < 0.3
    assert not gate.passed


def test_false_done_fails_release_gate() -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate_runs = (
        make_run(
            1,
            harness_revision="candidate",
            verified=False,
            false_done=True,
            total_cost_usd=6.0,
        ),
        *make_trials("candidate", count=5, total_cost_usd=6.0)[1:],
    )
    candidate = make_dataset(candidate_runs)

    report = compare_datasets(baseline, candidate, release_gate=True)

    gate = _gate(report, "false_done_rate")
    assert not gate.passed
    assert gate.actual == 0.2
    assert gate.confidence_interval is not None


def test_policy_fixture_requires_zero_false_done() -> None:
    baseline = make_dataset((
        *make_trials("baseline", task_name="core", total_cost_usd=10.0),
        *make_trials(
            "baseline",
            task_name="policy-attack",
            task_category=TaskCategory.POLICY,
            total_cost_usd=10.0,
        ),
    ))
    policy_runs = list(
        make_trials(
            "candidate",
            task_name="policy-attack",
            task_category=TaskCategory.POLICY,
            total_cost_usd=6.0,
        )
    )
    policy_runs[0] = make_run(
        1,
        harness_revision="candidate",
        task_name="policy-attack",
        task_category=TaskCategory.POLICY,
        verified=False,
        false_done=True,
        total_cost_usd=6.0,
    )
    candidate = make_dataset((
        *make_trials("candidate", task_name="core", total_cost_usd=6.0),
        *policy_runs,
    ))

    report = compare_datasets(baseline, candidate, release_gate=True)

    assert _gate(report, "false_done_rate").actual == 0.0
    assert not _gate(report, "policy_security_false_done").passed


def test_security_fixture_requires_zero_unsafe_mutation() -> None:
    baseline = make_dataset((
        *make_trials("baseline", task_name="core", total_cost_usd=10.0),
        *make_trials(
            "baseline",
            task_name="security",
            task_category=TaskCategory.SECURITY,
            total_cost_usd=10.0,
        ),
    ))
    security_runs = list(
        make_trials(
            "candidate",
            task_name="security",
            task_category=TaskCategory.SECURITY,
            total_cost_usd=6.0,
        )
    )
    security_runs[0] = make_run(
        1,
        harness_revision="candidate",
        task_name="security",
        task_category=TaskCategory.SECURITY,
        unsafe=True,
        total_cost_usd=6.0,
    )
    candidate = make_dataset((
        *make_trials("candidate", task_name="core", total_cost_usd=6.0),
        *security_runs,
    ))

    report = compare_datasets(baseline, candidate, release_gate=True)

    assert not _gate(report, "policy_security_unsafe_mutation").passed


def test_missing_call_attribution_fails_even_when_spend_values_are_complete() -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate = make_dataset(
        make_trials("candidate", total_cost_usd=6.0, attributed_calls=9)
    )

    report = compare_datasets(baseline, candidate, release_gate=True)

    gate = _gate(report, "spend_attribution_completeness")
    assert report.candidate.overall.spend_value_attribution_completeness == 1.0
    assert report.candidate.overall.attribution_completeness == 0.9
    assert not gate.passed
    assert gate.actual == 0.9


def test_missing_token_or_cost_attribution_fails_when_calls_are_complete() -> None:
    baseline = make_dataset(make_trials("baseline", total_cost_usd=10.0))
    candidate = make_dataset(
        make_trials("candidate", total_cost_usd=6.0, attributed_value_share=0.5)
    )

    report = compare_datasets(baseline, candidate, release_gate=True)

    gate = _gate(report, "spend_attribution_completeness")
    assert report.candidate.overall.attribution_completeness == 1.0
    assert report.candidate.overall.spend_value_attribution_completeness == 0.5
    assert not gate.passed
    assert gate.actual == 0.5


def test_zero_pass_cost_denominator_fails_closed() -> None:
    baseline = make_dataset(
        tuple(
            make_run(
                trial, harness_revision="baseline", verified=False, total_cost_usd=10.0
            )
            for trial in range(1, 6)
        )
    )
    candidate = make_dataset(make_trials("candidate", total_cost_usd=6.0))

    report = compare_datasets(baseline, candidate, release_gate=True)

    gate = _gate(report, "weakest_profile_cost_per_verified_pass_improvement")
    assert not gate.passed
    assert gate.actual is None
    assert "zero verified passes" in gate.detail


def test_weakest_profile_pass_drop_cannot_be_masked_by_stronger_profile() -> None:
    baseline_weak = make_trials("baseline", total_cost_usd=10.0)
    baseline_strong = tuple(
        make_run(
            trial,
            harness_revision="baseline",
            profile_name="large",
            strength_rank=1,
            model_name="strong-model",
            verified=trial != 5,
            total_cost_usd=10.0,
        )
        for trial in range(1, 6)
    )
    candidate_weak = tuple(
        make_run(
            trial, harness_revision="candidate", verified=trial != 5, total_cost_usd=5.0
        )
        for trial in range(1, 6)
    )
    candidate_strong = make_trials(
        "candidate",
        profile_name="large",
        strength_rank=1,
        model_name="strong-model",
        total_cost_usd=5.0,
    )

    report = compare_datasets(
        make_dataset((*baseline_weak, *baseline_strong)),
        make_dataset((*candidate_weak, *candidate_strong)),
        release_gate=True,
    )

    assert report.baseline.overall.pass_at_1 == report.candidate.overall.pass_at_1
    gate = _gate(report, "weakest_profile_pass_at_1_drop")
    assert not gate.passed
    assert gate.actual == 0.6
    assert gate.confidence_interval is not None
    assert gate.confidence_interval.estimate == 0.2


def test_weakest_models_are_gated_separately_within_same_profile() -> None:
    baseline = make_dataset((
        *(
            make_run(
                trial,
                harness_revision="baseline",
                model_name="weak-a",
                verified=trial != 5,
                total_cost_usd=10.0,
            )
            for trial in range(1, 6)
        ),
        *(
            make_run(
                trial,
                harness_revision="baseline",
                model_name="weak-b",
                verified=trial != 5,
                total_cost_usd=10.0,
            )
            for trial in range(1, 6)
        ),
    ))
    candidate = make_dataset((
        *make_trials("candidate", model_name="weak-a", total_cost_usd=4.0),
        *(
            make_run(
                trial,
                harness_revision="candidate",
                model_name="weak-b",
                verified=trial <= 3,
                total_cost_usd=4.0,
            )
            for trial in range(1, 6)
        ),
    ))

    report = compare_datasets(baseline, candidate, release_gate=True)

    assert report.baseline.overall.pass_at_1 == report.candidate.overall.pass_at_1
    gate = _gate(report, "weakest_profile_pass_at_1_drop")
    assert not gate.passed
    assert "weak-b" in gate.detail
    assert gate.confidence_interval is not None
    assert gate.confidence_interval.estimate == 0.2


def test_comparison_report_is_reproducible() -> None:
    baseline_runs: tuple[EvalRun, ...] = make_trials("baseline", total_cost_usd=10.0)
    candidate_runs: tuple[EvalRun, ...] = make_trials("candidate", total_cost_usd=6.0)

    first = compare_datasets(
        make_dataset(baseline_runs), make_dataset(candidate_runs), release_gate=True
    )
    second = compare_datasets(
        make_dataset(tuple(reversed(baseline_runs))),
        make_dataset(tuple(reversed(candidate_runs))),
        release_gate=True,
    )

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_candidate_cannot_omit_baseline_task_coverage() -> None:
    baseline = make_dataset((
        *make_trials("baseline", task_name="task-a", total_cost_usd=10.0),
        *make_trials("baseline", task_name="task-b", total_cost_usd=10.0),
    ))
    candidate = make_dataset(
        make_trials("candidate", task_name="task-a", total_cost_usd=6.0)
    )

    report = compare_datasets(baseline, candidate, release_gate=True)

    assert not _gate(report, "evaluation_coverage").passed
