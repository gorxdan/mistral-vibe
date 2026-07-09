from __future__ import annotations

import hashlib
from typing import Any

from evals.models import (
    ArtifactIdentity,
    CapabilityProfileIdentity,
    EvalRun,
    EvaluationDataset,
    ModelIdentity,
    RunArtifacts,
    RunMetrics,
    TaskCategory,
    VerificationReceiptEvidence,
)


def identity(name: str, version: str = "1") -> ArtifactIdentity:
    return ArtifactIdentity(
        name=name,
        version=version,
        digest=hashlib.sha256(f"{name}@{version}".encode()).hexdigest(),
    )


def make_run(
    trial: int,
    *,
    harness_revision: str,
    verified: bool = True,
    false_done: bool = False,
    unsafe: bool = False,
    total_tokens: int = 1_000,
    total_cost_usd: float = 1.0,
    harness_share: float = 0.10,
    maintenance_share: float = 0.0,
    calls: int = 10,
    attributed_calls: int | None = None,
    auxiliary_calls: int = 2,
    auxiliary_results_used: int = 2,
    attributed_value_share: float = 1.0,
    repair_attempts: int = 0,
    repair_recovered: bool = False,
    retries: int = 1,
    max_concurrency: int = 2,
    wall_time_seconds: float = 10.0,
    interventions: int = 0,
    task_name: str = "narrow-fix",
    task_category: TaskCategory = TaskCategory.CORE,
    model_name: str = "weak-model",
    profile_name: str = "small",
    strength_rank: int = 0,
) -> EvalRun:
    run_id = (
        f"{harness_revision}-{task_name}-{model_name}-{profile_name}-"
        f"r{strength_rank}-{trial}"
    )
    prompt_tokens = (total_tokens * 4) // 5
    completion_tokens = total_tokens - prompt_tokens
    reasoning_tokens = completion_tokens // 4
    cached_tokens = prompt_tokens // 8
    harness_prompt_tokens = round(prompt_tokens * harness_share)
    harness_completion_tokens = round(completion_tokens * harness_share)
    harness_tokens = harness_prompt_tokens + harness_completion_tokens
    harness_reasoning_tokens = min(reasoning_tokens, harness_completion_tokens // 4)
    harness_cached_tokens = min(cached_tokens, harness_prompt_tokens // 8)
    maintenance_tokens = round(total_tokens * maintenance_share)
    auxiliary_tokens = max(
        maintenance_tokens, round(total_tokens * 0.10) if auxiliary_calls else 0
    )
    maintenance_cost = total_cost_usd * maintenance_share
    auxiliary_cost = max(
        maintenance_cost, total_cost_usd * 0.10 if auxiliary_calls else 0.0
    )
    metrics = RunMetrics(
        completed=True,
        claimed_success=verified or false_done,
        verified_pass=verified,
        false_done=false_done,
        unsafe_mutation=unsafe,
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        harness_tokens=harness_tokens,
        harness_prompt_tokens=harness_prompt_tokens,
        harness_completion_tokens=harness_completion_tokens,
        harness_reasoning_tokens=harness_reasoning_tokens,
        harness_cached_tokens=harness_cached_tokens,
        auxiliary_tokens=auxiliary_tokens,
        maintenance_tokens=maintenance_tokens,
        attributed_tokens=round(total_tokens * attributed_value_share),
        total_cost_usd=total_cost_usd,
        harness_cost_usd=total_cost_usd * harness_share,
        auxiliary_cost_usd=auxiliary_cost,
        maintenance_cost_usd=maintenance_cost,
        attributed_cost_usd=total_cost_usd * attributed_value_share,
        repair_attempts=repair_attempts,
        repair_recovered=repair_recovered,
        calls=calls,
        auxiliary_calls=auxiliary_calls,
        auxiliary_results_used=auxiliary_results_used,
        attributed_calls=calls if attributed_calls is None else attributed_calls,
        retries=retries,
        max_concurrency=max_concurrency,
        wall_time_seconds=wall_time_seconds,
        interventions=interventions,
    )
    repository_fixture = identity("fixture")
    task_brief = identity(f"brief-{task_name}")
    harness_config = identity(f"config-{harness_revision}")
    terminal_diff = identity(f"diff-{run_id}")
    check_output = identity(f"checks-{run_id}")
    verification_receipt = (
        VerificationReceiptEvidence(
            receipt=identity(f"receipt-{run_id}"),
            task_brief=task_brief,
            repository_fixture=repository_fixture,
            terminal_diff=terminal_diff,
            check_output=check_output,
            harness_config=harness_config,
            trusted_valid=True,
        )
        if verified
        else None
    )
    return EvalRun(
        run_id=run_id,
        task=identity(task_name),
        task_category=task_category,
        model=ModelIdentity(provider="test", name=model_name, revision="2026-01"),
        profile=CapabilityProfileIdentity(
            name=profile_name,
            version="1",
            digest=identity(f"profile-{profile_name}").digest,
            strength_rank=strength_rank,
        ),
        harness_revision=harness_revision,
        trial_index=trial,
        random_seed=trial,
        artifacts=RunArtifacts(
            repository_fixture=repository_fixture,
            task_brief=task_brief,
            recipe=identity("recipe"),
            pricing_table=identity("pricing"),
            policy=identity("policy"),
            harness_config=harness_config,
            raw_events=identity(f"events-{run_id}"),
            terminal_diff=terminal_diff,
            check_output=check_output,
            verification_receipt=verification_receipt,
        ),
        metrics=metrics,
    )


def make_trials(
    harness_revision: str, *, count: int = 5, **overrides: Any
) -> tuple[EvalRun, ...]:
    return tuple(
        make_run(trial, harness_revision=harness_revision, **overrides)
        for trial in range(1, count + 1)
    )


def make_dataset(runs: tuple[EvalRun, ...]) -> EvaluationDataset:
    return EvaluationDataset(dataset=identity("benchmark", "7"), runs=runs)


__all__ = ["identity", "make_dataset", "make_run", "make_trials"]
