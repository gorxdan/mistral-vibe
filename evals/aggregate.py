from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import hashlib
import math
import random
import statistics

import orjson

from evals.models import (
    AggregateKey,
    AggregateMetrics,
    ConfidenceInterval,
    DistributionSummary,
    EvalRun,
    EvaluationAggregate,
    EvaluationDataset,
    GroupAggregate,
    MedianMetrics,
)

_WILSON_Z = 1.959963984540054
_BOOTSTRAP_SAMPLES = 4_096
_ROUND_DIGITS = 12


def aggregate_dataset(dataset: EvaluationDataset) -> EvaluationAggregate:
    digest = dataset_digest(dataset)
    grouped: dict[tuple[str, str, str, str, str], list[EvalRun]] = defaultdict(list)
    for run in dataset.runs:
        grouped[run.group_identity].append(run)

    groups: list[GroupAggregate] = []
    for identity in sorted(grouped):
        runs = sorted(grouped[identity], key=lambda run: (run.trial_index, run.run_id))
        first = runs[0]
        groups.append(
            GroupAggregate(
                key=AggregateKey(
                    task=first.task,
                    task_category=first.task_category,
                    model=first.model,
                    profile=first.profile,
                    harness_revision=first.harness_revision,
                ),
                metrics=aggregate_metrics(runs, seed_material="\0".join(identity)),
            )
        )

    all_runs = sorted(
        dataset.runs, key=lambda run: (*run.group_identity, run.trial_index, run.run_id)
    )
    overall = aggregate_metrics(
        all_runs, seed_material=f"overall:{dataset.dataset.key}"
    )
    overall = overall.model_copy(
        update={
            "pass_at_3": _rounded(
                statistics.fmean(group.metrics.pass_at_3 for group in groups)
            )
        }
    )
    return EvaluationAggregate(
        dataset=dataset.dataset,
        input_digest=digest,
        groups=tuple(groups),
        overall=overall,
    )


def aggregate_metrics(
    runs: Sequence[EvalRun], *, seed_material: str
) -> AggregateMetrics:
    if not runs:
        raise ValueError("cannot aggregate an empty run set")

    trial_count = len(runs)
    completed_count = sum(run.metrics.completed for run in runs)
    verified_passes = sum(run.metrics.verified_pass for run in runs)
    false_done_count = sum(run.metrics.false_done for run in runs)
    unsafe_mutation_count = sum(run.metrics.unsafe_mutation for run in runs)
    totals = _spend_totals(runs)
    repair_runs = [run for run in runs if run.metrics.repair_attempts > 0]
    repair_recovered = sum(run.metrics.repair_recovered for run in repair_runs)

    return AggregateMetrics(
        trial_count=trial_count,
        completed_count=completed_count,
        completed_rate=_rate(completed_count, trial_count),
        completed_confidence_interval=wilson_interval(completed_count, trial_count),
        verified_passes=verified_passes,
        pass_at_1=_rate(verified_passes, trial_count),
        pass_at_1_confidence_interval=wilson_interval(verified_passes, trial_count),
        pass_at_3=_pass_at_k(trial_count, verified_passes, 3),
        false_done_count=false_done_count,
        false_done_rate=_rate(false_done_count, trial_count),
        false_done_confidence_interval=wilson_interval(false_done_count, trial_count),
        unsafe_mutation_count=unsafe_mutation_count,
        unsafe_mutation_rate=_rate(unsafe_mutation_count, trial_count),
        unsafe_mutation_confidence_interval=wilson_interval(
            unsafe_mutation_count, trial_count
        ),
        total_tokens=totals.total_tokens,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        reasoning_tokens=totals.reasoning_tokens,
        cached_tokens=totals.cached_tokens,
        harness_tokens=totals.harness_tokens,
        harness_prompt_tokens=totals.harness_prompt_tokens,
        harness_completion_tokens=totals.harness_completion_tokens,
        harness_reasoning_tokens=totals.harness_reasoning_tokens,
        harness_cached_tokens=totals.harness_cached_tokens,
        auxiliary_tokens=totals.auxiliary_tokens,
        maintenance_tokens=totals.maintenance_tokens,
        attributed_tokens=totals.attributed_tokens,
        total_cost_usd=totals.total_cost_usd,
        harness_cost_usd=totals.harness_cost_usd,
        auxiliary_cost_usd=totals.auxiliary_cost_usd,
        maintenance_cost_usd=totals.maintenance_cost_usd,
        attributed_cost_usd=totals.attributed_cost_usd,
        cost_per_verified_pass_usd=_per_pass(totals.total_cost_usd, verified_passes),
        tokens_per_verified_pass=_per_pass(totals.total_tokens, verified_passes),
        harness_utilization=_utilization(
            totals.harness_tokens,
            totals.total_tokens,
            totals.harness_cost_usd,
            totals.total_cost_usd,
        ),
        auxiliary_utilization=_rounded(
            _share(totals.auxiliary_results_used, totals.auxiliary_calls)
        ),
        maintenance_utilization=_utilization(
            totals.maintenance_tokens,
            totals.total_tokens,
            totals.maintenance_cost_usd,
            totals.total_cost_usd,
        ),
        attribution_completeness=_rounded(
            _coverage(totals.attributed_calls, totals.total_calls)
        ),
        spend_value_attribution_completeness=_spend_value_attribution(totals),
        repair_attempted_runs=len(repair_runs),
        repair_recovered_runs=repair_recovered,
        repair_recovery_rate=(
            _rate(repair_recovered, len(repair_runs)) if repair_runs else None
        ),
        repair_recovery_confidence_interval=(
            wilson_interval(repair_recovered, len(repair_runs)) if repair_runs else None
        ),
        total_calls=totals.total_calls,
        auxiliary_calls=totals.auxiliary_calls,
        auxiliary_results_used=totals.auxiliary_results_used,
        attributed_calls=totals.attributed_calls,
        total_retries=sum(run.metrics.retries for run in runs),
        peak_concurrency=max(run.metrics.max_concurrency for run in runs),
        total_wall_time_seconds=_rounded(
            math.fsum(run.metrics.wall_time_seconds for run in runs)
        ),
        total_interventions=sum(run.metrics.interventions for run in runs),
        medians=_median_metrics(runs, seed_material),
    )


class _SpendTotals:
    def __init__(self, runs: Sequence[EvalRun]) -> None:
        self.total_tokens = sum(run.metrics.total_tokens for run in runs)
        self.prompt_tokens = sum(run.metrics.prompt_tokens for run in runs)
        self.completion_tokens = sum(run.metrics.completion_tokens for run in runs)
        self.reasoning_tokens = sum(run.metrics.reasoning_tokens for run in runs)
        self.cached_tokens = sum(run.metrics.cached_tokens for run in runs)
        self.harness_tokens = sum(run.metrics.harness_tokens for run in runs)
        self.harness_prompt_tokens = sum(
            run.metrics.harness_prompt_tokens for run in runs
        )
        self.harness_completion_tokens = sum(
            run.metrics.harness_completion_tokens for run in runs
        )
        self.harness_reasoning_tokens = sum(
            run.metrics.harness_reasoning_tokens for run in runs
        )
        self.harness_cached_tokens = sum(
            run.metrics.harness_cached_tokens for run in runs
        )
        self.auxiliary_tokens = sum(run.metrics.auxiliary_tokens for run in runs)
        self.maintenance_tokens = sum(run.metrics.maintenance_tokens for run in runs)
        self.attributed_tokens = sum(run.metrics.attributed_tokens for run in runs)
        self.total_cost_usd = _rounded(
            math.fsum(run.metrics.total_cost_usd for run in runs)
        )
        self.harness_cost_usd = _rounded(
            math.fsum(run.metrics.harness_cost_usd for run in runs)
        )
        self.auxiliary_cost_usd = _rounded(
            math.fsum(run.metrics.auxiliary_cost_usd for run in runs)
        )
        self.maintenance_cost_usd = _rounded(
            math.fsum(run.metrics.maintenance_cost_usd for run in runs)
        )
        self.attributed_cost_usd = _rounded(
            math.fsum(run.metrics.attributed_cost_usd for run in runs)
        )
        self.total_calls = sum(run.metrics.calls for run in runs)
        self.auxiliary_calls = sum(run.metrics.auxiliary_calls for run in runs)
        self.auxiliary_results_used = sum(
            run.metrics.auxiliary_results_used for run in runs
        )
        self.attributed_calls = sum(run.metrics.attributed_calls for run in runs)


def _spend_totals(runs: Sequence[EvalRun]) -> _SpendTotals:
    return _SpendTotals(runs)


def _median_metrics(runs: Sequence[EvalRun], seed_material: str) -> MedianMetrics:
    return MedianMetrics(
        total_tokens=_distribution(
            [run.metrics.total_tokens for run in runs], seed_material, "total_tokens"
        ),
        harness_tokens=_distribution(
            [run.metrics.harness_tokens for run in runs],
            seed_material,
            "harness_tokens",
        ),
        total_cost_usd=_distribution(
            [run.metrics.total_cost_usd for run in runs],
            seed_material,
            "total_cost_usd",
        ),
        harness_cost_usd=_distribution(
            [run.metrics.harness_cost_usd for run in runs],
            seed_material,
            "harness_cost_usd",
        ),
        auxiliary_utilization=_distribution(
            [
                _share(run.metrics.auxiliary_results_used, run.metrics.auxiliary_calls)
                for run in runs
            ],
            seed_material,
            "auxiliary_utilization",
        ),
        calls=_distribution(
            [run.metrics.calls for run in runs], seed_material, "calls"
        ),
        retries=_distribution(
            [run.metrics.retries for run in runs], seed_material, "retries"
        ),
        max_concurrency=_distribution(
            [run.metrics.max_concurrency for run in runs],
            seed_material,
            "max_concurrency",
        ),
        wall_time_seconds=_distribution(
            [run.metrics.wall_time_seconds for run in runs],
            seed_material,
            "wall_time_seconds",
        ),
        interventions=_distribution(
            [run.metrics.interventions for run in runs], seed_material, "interventions"
        ),
    )


def _distribution(
    values: Sequence[int | float], seed_material: str, metric: str
) -> DistributionSummary:
    ordered = sorted(float(value) for value in values)
    estimate = float(statistics.median(ordered))
    if ordered[0] == ordered[-1]:
        interval = ConfidenceInterval(
            estimate=_rounded(estimate),
            lower=_rounded(estimate),
            upper=_rounded(estimate),
            method="degenerate-median",
        )
        return DistributionSummary(
            count=len(ordered),
            minimum=_rounded(estimate),
            median=_rounded(estimate),
            maximum=_rounded(estimate),
            confidence_interval=interval,
        )

    rng = random.Random(_stable_seed(f"{seed_material}\0{metric}"))
    bootstrap = sorted(
        float(statistics.median(rng.choices(ordered, k=len(ordered))))
        for _ in range(_BOOTSTRAP_SAMPLES)
    )
    interval = ConfidenceInterval(
        estimate=_rounded(estimate),
        lower=_rounded(_percentile(bootstrap, 0.025)),
        upper=_rounded(_percentile(bootstrap, 0.975)),
        method=f"deterministic-bootstrap-median-{_BOOTSTRAP_SAMPLES}",
    )
    return DistributionSummary(
        count=len(ordered),
        minimum=_rounded(ordered[0]),
        median=_rounded(estimate),
        maximum=_rounded(ordered[-1]),
        confidence_interval=interval,
    )


def wilson_interval(successes: int, trials: int) -> ConfidenceInterval:
    estimate = successes / trials
    denominator = 1 + (_WILSON_Z**2 / trials)
    center = (estimate + (_WILSON_Z**2 / (2 * trials))) / denominator
    margin = (
        _WILSON_Z
        * math.sqrt(
            (estimate * (1 - estimate) / trials) + (_WILSON_Z**2 / (4 * trials**2))
        )
        / denominator
    )
    return ConfidenceInterval(
        estimate=_rounded(estimate),
        lower=_rounded(max(0.0, center - margin)),
        upper=_rounded(min(1.0, center + margin)),
        method="wilson-score",
    )


def _pass_at_k(trials: int, successes: int, k: int) -> float:
    sample = min(k, trials)
    failures = trials - successes
    if failures < sample:
        return 1.0
    return _rounded(1 - (math.comb(failures, sample) / math.comb(trials, sample)))


def _utilization(
    part_tokens: int, total_tokens: int, part_cost: float, total_cost: float
) -> float:
    return _rounded(
        max(_share(part_tokens, total_tokens), _share(part_cost, total_cost))
    )


def _spend_value_attribution(totals: _SpendTotals) -> float:
    return _rounded(
        min(
            _coverage(totals.attributed_tokens, totals.total_tokens),
            _coverage(totals.attributed_cost_usd, totals.total_cost_usd),
        )
    )


def _share(part: int | float, total: int | float) -> float:
    return 0.0 if total == 0 else float(part / total)


def _coverage(part: int | float, total: int | float) -> float:
    return 1.0 if total == 0 else float(part / total)


def _per_pass(value: int | float, verified_passes: int) -> float | None:
    if verified_passes == 0:
        return None
    return _rounded(value / verified_passes)


def _rate(successes: int, trials: int) -> float:
    return _rounded(successes / trials)


def _percentile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + ((values[upper] - values[lower]) * fraction)


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _rounded(value: int | float) -> float:
    rounded = round(float(value), _ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def dataset_digest(dataset: EvaluationDataset) -> str:
    runs = sorted(
        (run.model_dump(mode="json") for run in dataset.runs),
        key=lambda run: (
            run["task"]["name"],
            run["task"]["version"],
            run["task"]["digest"],
            run["task_category"],
            run["model"]["provider"],
            run["model"]["name"],
            run["model"]["revision"],
            run["profile"]["strength_rank"],
            run["profile"]["name"],
            run["profile"]["version"],
            run["profile"]["digest"],
            run["harness_revision"],
            run["trial_index"],
            run["run_id"],
        ),
    )
    payload = orjson.dumps(
        {
            "schema_version": dataset.schema_version,
            "dataset": dataset.dataset.model_dump(mode="json"),
            "runs": runs,
        },
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "aggregate_dataset",
    "aggregate_metrics",
    "dataset_digest",
    "wilson_interval",
]
