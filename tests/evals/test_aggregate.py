from __future__ import annotations

from evals.aggregate import aggregate_dataset, aggregate_metrics
from tests.evals._factories import make_dataset, make_run


def _mixed_runs():
    runs = []
    for trial, tokens in enumerate((1_000, 1_200, 1_400, 1_600, 1_800), start=1):
        is_last = trial == 5
        runs.append(
            make_run(
                trial,
                harness_revision="candidate",
                verified=not is_last,
                false_done=is_last,
                unsafe=is_last,
                total_tokens=tokens,
                total_cost_usd=10.0,
                calls=10,
                attributed_calls=9,
                auxiliary_calls=4,
                auxiliary_results_used=2,
                repair_attempts=1 if trial in {1, 5} else 0,
                repair_recovered=trial == 1,
            )
        )
    return tuple(runs)


def test_aggregate_computes_reliability_cost_and_operational_metrics() -> None:
    metrics = aggregate_metrics(_mixed_runs(), seed_material="mixed")

    assert metrics.trial_count == 5
    assert metrics.verified_passes == 4
    assert metrics.pass_at_1 == 0.8
    assert metrics.pass_at_3 == 1.0
    assert metrics.false_done_rate == 0.2
    assert metrics.unsafe_mutation_rate == 0.2
    assert metrics.total_tokens == 7_000
    assert metrics.prompt_tokens + metrics.completion_tokens == 7_000
    assert metrics.reasoning_tokens == 350
    assert metrics.cached_tokens == 700
    assert metrics.harness_prompt_tokens + metrics.harness_completion_tokens == (
        metrics.harness_tokens
    )
    assert metrics.total_cost_usd == 50.0
    assert metrics.cost_per_verified_pass_usd == 12.5
    assert metrics.tokens_per_verified_pass == 1_750.0
    assert metrics.auxiliary_calls == 20
    assert metrics.auxiliary_results_used == 10
    assert metrics.auxiliary_utilization == 0.5
    assert metrics.attributed_calls == 45
    assert metrics.attribution_completeness == 0.9
    assert metrics.spend_value_attribution_completeness == 1.0
    assert metrics.repair_attempted_runs == 2
    assert metrics.repair_recovered_runs == 1
    assert metrics.repair_recovery_rate == 0.5
    assert metrics.total_calls == 50
    assert metrics.total_retries == 5
    assert metrics.peak_concurrency == 2
    assert metrics.total_wall_time_seconds == 50.0
    assert metrics.medians.total_tokens.median == 1_400.0
    assert metrics.medians.total_tokens.confidence_interval.lower <= 1_400
    assert metrics.medians.total_tokens.confidence_interval.upper >= 1_400


def test_zero_verified_pass_denominators_are_explicit() -> None:
    runs = tuple(
        make_run(
            trial,
            harness_revision="candidate",
            verified=False,
            calls=0,
            attributed_calls=0,
            auxiliary_calls=0,
            auxiliary_results_used=0,
            total_tokens=0,
            total_cost_usd=0.0,
            retries=0,
            max_concurrency=0,
        )
        for trial in range(1, 6)
    )

    metrics = aggregate_metrics(runs, seed_material="zero")

    assert metrics.pass_at_1 == 0.0
    assert metrics.pass_at_3 == 0.0
    assert metrics.cost_per_verified_pass_usd is None
    assert metrics.tokens_per_verified_pass is None
    assert metrics.repair_recovery_rate is None
    assert metrics.repair_recovery_confidence_interval is None
    assert metrics.attribution_completeness == 1.0
    assert metrics.spend_value_attribution_completeness == 1.0


def test_dataset_aggregates_by_task_model_profile_and_harness_revision() -> None:
    runs = (
        *(
            make_run(trial, harness_revision="rev-a", task_name="task-a")
            for trial in range(1, 6)
        ),
        *(
            make_run(trial, harness_revision="rev-b", task_name="task-b")
            for trial in range(1, 6)
        ),
    )

    aggregate = aggregate_dataset(make_dataset(runs))

    assert len(aggregate.groups) == 2
    assert aggregate.groups[0].key.task.name == "task-a"
    assert aggregate.groups[0].key.harness_revision == "rev-a"
    assert aggregate.groups[1].key.task.name == "task-b"
    assert aggregate.groups[1].key.harness_revision == "rev-b"
    assert aggregate.overall.trial_count == 10


def test_profile_strength_rank_is_part_of_aggregate_identity() -> None:
    runs = (
        make_run(1, harness_revision="candidate", strength_rank=0),
        make_run(1, harness_revision="candidate", strength_rank=1),
    )

    aggregate = aggregate_dataset(make_dataset(runs))

    assert len(aggregate.groups) == 2
    assert {group.key.profile.strength_rank for group in aggregate.groups} == {0, 1}


def test_overall_pass_at_3_is_macro_averaged_across_groups() -> None:
    runs = (
        *(
            make_run(trial, harness_revision="candidate", task_name="always-pass")
            for trial in range(1, 6)
        ),
        *(
            make_run(
                trial,
                harness_revision="candidate",
                task_name="always-fail",
                verified=False,
            )
            for trial in range(1, 6)
        ),
    )

    aggregate = aggregate_dataset(make_dataset(runs))

    assert {group.metrics.pass_at_3 for group in aggregate.groups} == {0.0, 1.0}
    assert aggregate.overall.pass_at_3 == 0.5


def test_aggregation_is_reproducible_across_input_order() -> None:
    runs = _mixed_runs()

    forward = aggregate_dataset(make_dataset(runs))
    reversed_result = aggregate_dataset(make_dataset(tuple(reversed(runs))))

    assert forward == reversed_result
    assert forward.model_dump_json() == reversed_result.model_dump_json()
