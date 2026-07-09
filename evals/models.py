from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVAL_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )


def _identifier(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be nonempty without surrounding whitespace")
    return value


class ArtifactIdentity(StrictModel):
    name: str
    version: str
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("name", "version")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        return _identifier(value, "artifact identity")

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}#{self.digest}"


class ModelIdentity(StrictModel):
    provider: str
    name: str
    revision: str

    @field_validator("provider", "name", "revision")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        return _identifier(value, "model identity")

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.name}@{self.revision}"


class CapabilityProfileIdentity(ArtifactIdentity):
    strength_rank: int = Field(ge=0)

    @property
    def key(self) -> str:
        return f"{super().key}:rank={self.strength_rank}"


class TaskCategory(StrEnum):
    CORE = auto()
    POLICY = auto()
    SECURITY = auto()


class VerificationReceiptEvidence(StrictModel):
    receipt: ArtifactIdentity
    task_brief: ArtifactIdentity
    repository_fixture: ArtifactIdentity
    terminal_diff: ArtifactIdentity
    check_output: ArtifactIdentity
    harness_config: ArtifactIdentity
    trusted_valid: bool


class RunArtifacts(StrictModel):
    repository_fixture: ArtifactIdentity
    task_brief: ArtifactIdentity
    recipe: ArtifactIdentity
    pricing_table: ArtifactIdentity
    policy: ArtifactIdentity
    harness_config: ArtifactIdentity
    raw_events: ArtifactIdentity
    terminal_diff: ArtifactIdentity
    check_output: ArtifactIdentity
    verification_receipt: VerificationReceiptEvidence | None


class RunMetrics(StrictModel):
    completed: bool
    claimed_success: bool
    verified_pass: bool
    false_done: bool
    unsafe_mutation: bool
    total_tokens: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    harness_tokens: int = Field(ge=0)
    harness_prompt_tokens: int = Field(ge=0)
    harness_completion_tokens: int = Field(ge=0)
    harness_reasoning_tokens: int = Field(ge=0)
    harness_cached_tokens: int = Field(ge=0)
    auxiliary_tokens: int = Field(ge=0)
    maintenance_tokens: int = Field(ge=0)
    attributed_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    harness_cost_usd: float = Field(ge=0)
    auxiliary_cost_usd: float = Field(ge=0)
    maintenance_cost_usd: float = Field(ge=0)
    attributed_cost_usd: float = Field(ge=0)
    repair_attempts: int = Field(ge=0)
    repair_recovered: bool
    calls: int = Field(ge=0)
    auxiliary_calls: int = Field(ge=0)
    auxiliary_results_used: int = Field(ge=0)
    attributed_calls: int = Field(ge=0)
    retries: int = Field(ge=0)
    max_concurrency: int = Field(ge=0)
    wall_time_seconds: float = Field(ge=0)
    interventions: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_consistency(self) -> RunMetrics:
        if self.verified_pass and not self.completed:
            raise ValueError("a verified pass must be a completed run")
        if self.claimed_success and not self.completed:
            raise ValueError("a success claim must come from a completed run")
        expected_false_done = self.claimed_success and not self.verified_pass
        if self.false_done != expected_false_done:
            raise ValueError(
                "false_done must equal claimed_success and not verified_pass"
            )
        if self.repair_recovered and (
            self.repair_attempts == 0 or not self.verified_pass
        ):
            raise ValueError(
                "repair recovery requires an attempted repair and verified pass"
            )
        self._validate_token_breakdown()
        self._validate_spend_bounds()
        if self.retries > self.calls:
            raise ValueError("retries cannot exceed calls")
        if self.max_concurrency > self.calls:
            raise ValueError("max concurrency cannot exceed calls")
        if self.auxiliary_calls > self.calls:
            raise ValueError("auxiliary calls cannot exceed calls")
        if self.auxiliary_results_used > self.auxiliary_calls:
            raise ValueError("used auxiliary results cannot exceed auxiliary calls")
        if self.auxiliary_calls == 0 and (
            self.auxiliary_tokens > 0 or self.auxiliary_cost_usd > 0
        ):
            raise ValueError("auxiliary spend requires at least one auxiliary call")
        if self.attributed_calls > self.calls:
            raise ValueError("attributed calls cannot exceed calls")
        if self.calls == 0 and (self.total_tokens > 0 or self.total_cost_usd > 0):
            raise ValueError("nonzero spend requires at least one call")
        return self

    def _validate_token_breakdown(self) -> None:
        if self.prompt_tokens + self.completion_tokens != self.total_tokens:
            raise ValueError("prompt plus completion tokens must equal total tokens")
        if self.reasoning_tokens > self.completion_tokens:
            raise ValueError("reasoning tokens cannot exceed completion tokens")
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached tokens cannot exceed prompt tokens")
        if (
            self.harness_prompt_tokens + self.harness_completion_tokens
            != self.harness_tokens
        ):
            raise ValueError(
                "harness prompt plus completion tokens must equal harness tokens"
            )
        component_bounds = (
            (self.harness_prompt_tokens, self.prompt_tokens, "prompt"),
            (self.harness_completion_tokens, self.completion_tokens, "completion"),
            (self.harness_reasoning_tokens, self.reasoning_tokens, "reasoning"),
            (self.harness_cached_tokens, self.cached_tokens, "cached"),
        )
        for harness_value, total_value, label in component_bounds:
            if harness_value > total_value:
                raise ValueError(f"harness {label} tokens cannot exceed {label} tokens")
        if self.harness_reasoning_tokens > self.harness_completion_tokens:
            raise ValueError(
                "harness reasoning tokens cannot exceed harness completion tokens"
            )
        if self.harness_cached_tokens > self.harness_prompt_tokens:
            raise ValueError(
                "harness cached tokens cannot exceed harness prompt tokens"
            )

    def _validate_spend_bounds(self) -> None:
        token_parts = {
            "harness": self.harness_tokens,
            "auxiliary": self.auxiliary_tokens,
            "attributed": self.attributed_tokens,
        }
        for label, value in token_parts.items():
            if value > self.total_tokens:
                raise ValueError(f"{label} tokens cannot exceed total tokens")
        if self.maintenance_tokens > self.auxiliary_tokens:
            raise ValueError("maintenance tokens cannot exceed auxiliary tokens")

        cost_parts = {
            "harness": self.harness_cost_usd,
            "auxiliary": self.auxiliary_cost_usd,
            "attributed": self.attributed_cost_usd,
        }
        for label, value in cost_parts.items():
            if value > self.total_cost_usd:
                raise ValueError(f"{label} cost cannot exceed total cost")
        if self.maintenance_cost_usd > self.auxiliary_cost_usd:
            raise ValueError("maintenance cost cannot exceed auxiliary cost")


class EvalRun(StrictModel):
    schema_version: Literal[1] = EVAL_SCHEMA_VERSION
    run_id: str
    task: ArtifactIdentity
    task_category: TaskCategory
    model: ModelIdentity
    profile: CapabilityProfileIdentity
    harness_revision: str
    trial_index: int = Field(ge=1)
    random_seed: int | None
    artifacts: RunArtifacts
    metrics: RunMetrics

    @field_validator("run_id", "harness_revision")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        return _identifier(value, "run identity")

    @model_validator(mode="after")
    def _validate_verification_receipt(self) -> EvalRun:
        evidence = self.artifacts.verification_receipt
        if evidence is None:
            if self.metrics.verified_pass:
                raise ValueError(
                    "a verified pass requires trusted verification receipt evidence"
                )
            return self

        if evidence.trusted_valid != self.metrics.verified_pass:
            raise ValueError(
                "trusted receipt validity must equal the verified-pass outcome"
            )
        if not evidence.trusted_valid:
            return self

        bindings = (
            ("task brief", evidence.task_brief, self.artifacts.task_brief),
            (
                "repository fixture",
                evidence.repository_fixture,
                self.artifacts.repository_fixture,
            ),
            ("terminal diff", evidence.terminal_diff, self.artifacts.terminal_diff),
            ("check output", evidence.check_output, self.artifacts.check_output),
            ("harness config", evidence.harness_config, self.artifacts.harness_config),
        )
        for label, bound, current in bindings:
            if bound != current:
                raise ValueError(
                    f"verification receipt {label} binding does not match the run"
                )
        return self

    @property
    def group_identity(self) -> tuple[str, str, str, str, str]:
        return (
            self.task.key,
            self.task_category.value,
            self.model.key,
            self.profile.key,
            self.harness_revision,
        )

    @property
    def comparison_identity(self) -> tuple[str, str, str, str]:
        return self.group_identity[:4]


class EvaluationDataset(StrictModel):
    schema_version: Literal[1] = EVAL_SCHEMA_VERSION
    dataset: ArtifactIdentity
    runs: tuple[EvalRun, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_trials(self) -> EvaluationDataset:
        run_ids: set[str] = set()
        trials: set[tuple[tuple[str, str, str, str, str], int]] = set()
        for run in self.runs:
            if run.run_id in run_ids:
                raise ValueError(f"duplicate run_id: {run.run_id}")
            run_ids.add(run.run_id)
            trial = (run.group_identity, run.trial_index)
            if trial in trials:
                raise ValueError(
                    f"duplicate trial {run.trial_index} for {run.group_identity}"
                )
            trials.add(trial)
        return self


class ConfidenceInterval(StrictModel):
    estimate: float
    lower: float
    upper: float
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    method: str

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        return _identifier(value, "confidence interval method")

    @model_validator(mode="after")
    def _validate_order(self) -> ConfidenceInterval:
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("confidence interval must contain its estimate")
        return self


class DistributionSummary(StrictModel):
    count: int = Field(ge=1)
    minimum: float
    median: float
    maximum: float
    confidence_interval: ConfidenceInterval

    @model_validator(mode="after")
    def _validate_order(self) -> DistributionSummary:
        if not self.minimum <= self.median <= self.maximum:
            raise ValueError("distribution minimum, median, and maximum are unordered")
        if self.confidence_interval.estimate != self.median:
            raise ValueError("distribution interval estimate must equal its median")
        return self


class MedianMetrics(StrictModel):
    total_tokens: DistributionSummary
    harness_tokens: DistributionSummary
    total_cost_usd: DistributionSummary
    harness_cost_usd: DistributionSummary
    auxiliary_utilization: DistributionSummary
    calls: DistributionSummary
    retries: DistributionSummary
    max_concurrency: DistributionSummary
    wall_time_seconds: DistributionSummary
    interventions: DistributionSummary


class AggregateMetrics(StrictModel):
    trial_count: int = Field(ge=1)
    completed_count: int = Field(ge=0)
    completed_rate: float = Field(ge=0, le=1)
    completed_confidence_interval: ConfidenceInterval
    verified_passes: int = Field(ge=0)
    pass_at_1: float = Field(ge=0, le=1)
    pass_at_1_confidence_interval: ConfidenceInterval
    pass_at_3: float = Field(ge=0, le=1)
    false_done_count: int = Field(ge=0)
    false_done_rate: float = Field(ge=0, le=1)
    false_done_confidence_interval: ConfidenceInterval
    unsafe_mutation_count: int = Field(ge=0)
    unsafe_mutation_rate: float = Field(ge=0, le=1)
    unsafe_mutation_confidence_interval: ConfidenceInterval
    total_tokens: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    harness_tokens: int = Field(ge=0)
    harness_prompt_tokens: int = Field(ge=0)
    harness_completion_tokens: int = Field(ge=0)
    harness_reasoning_tokens: int = Field(ge=0)
    harness_cached_tokens: int = Field(ge=0)
    auxiliary_tokens: int = Field(ge=0)
    maintenance_tokens: int = Field(ge=0)
    attributed_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    harness_cost_usd: float = Field(ge=0)
    auxiliary_cost_usd: float = Field(ge=0)
    maintenance_cost_usd: float = Field(ge=0)
    attributed_cost_usd: float = Field(ge=0)
    cost_per_verified_pass_usd: float | None
    tokens_per_verified_pass: float | None
    harness_utilization: float = Field(ge=0, le=1)
    auxiliary_utilization: float = Field(ge=0, le=1)
    maintenance_utilization: float = Field(ge=0, le=1)
    attribution_completeness: float = Field(ge=0, le=1)
    spend_value_attribution_completeness: float = Field(ge=0, le=1)
    repair_attempted_runs: int = Field(ge=0)
    repair_recovered_runs: int = Field(ge=0)
    repair_recovery_rate: float | None = Field(ge=0, le=1)
    repair_recovery_confidence_interval: ConfidenceInterval | None
    total_calls: int = Field(ge=0)
    auxiliary_calls: int = Field(ge=0)
    auxiliary_results_used: int = Field(ge=0)
    attributed_calls: int = Field(ge=0)
    total_retries: int = Field(ge=0)
    peak_concurrency: int = Field(ge=0)
    total_wall_time_seconds: float = Field(ge=0)
    total_interventions: int = Field(ge=0)
    medians: MedianMetrics

    @model_validator(mode="after")
    def _validate_aggregate_consistency(self) -> AggregateMetrics:
        outcome_counts = (
            self.completed_count,
            self.verified_passes,
            self.false_done_count,
            self.unsafe_mutation_count,
        )
        if any(count > self.trial_count for count in outcome_counts):
            raise ValueError("aggregate outcome counts cannot exceed trial count")
        if self.prompt_tokens + self.completion_tokens != self.total_tokens:
            raise ValueError("aggregate token components do not equal total tokens")
        if (
            self.harness_prompt_tokens + self.harness_completion_tokens
            != self.harness_tokens
        ):
            raise ValueError("aggregate harness components do not equal harness tokens")
        if self.auxiliary_results_used > self.auxiliary_calls:
            raise ValueError("aggregate auxiliary results exceed auxiliary calls")
        if self.attributed_calls > self.total_calls:
            raise ValueError("aggregate attributed calls exceed total calls")
        if self.repair_recovered_runs > self.repair_attempted_runs:
            raise ValueError("aggregate repair recoveries exceed attempted repairs")
        has_repair_rate = self.repair_recovery_rate is not None
        has_repair_interval = self.repair_recovery_confidence_interval is not None
        if self.repair_attempted_runs == 0 and (has_repair_rate or has_repair_interval):
            raise ValueError("repair metrics require attempted repair runs")
        if self.repair_attempted_runs > 0 and not (
            has_repair_rate and has_repair_interval
        ):
            raise ValueError("attempted repair runs require recovery metrics")
        return self


class AggregateKey(StrictModel):
    task: ArtifactIdentity
    task_category: TaskCategory
    model: ModelIdentity
    profile: CapabilityProfileIdentity
    harness_revision: str


class GroupAggregate(StrictModel):
    key: AggregateKey
    metrics: AggregateMetrics


class EvaluationAggregate(StrictModel):
    schema_version: Literal[1] = REPORT_SCHEMA_VERSION
    dataset: ArtifactIdentity
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    groups: tuple[GroupAggregate, ...] = Field(min_length=1)
    overall: AggregateMetrics


class GateResult(StrictModel):
    name: str
    passed: bool
    actual: float | None
    threshold: float | None
    detail: str
    confidence_interval: ConfidenceInterval | None = None

    @field_validator("name", "detail")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _identifier(value, "gate result")


class ComparisonReport(StrictModel):
    schema_version: Literal[1] = REPORT_SCHEMA_VERSION
    release_gate: bool
    baseline: EvaluationAggregate
    candidate: EvaluationAggregate
    weakest_profile_rank: int | None
    gates: tuple[GateResult, ...] = Field(min_length=1)
    passed: bool

    @model_validator(mode="after")
    def _validate_outcome(self) -> ComparisonReport:
        if self.passed != all(gate.passed for gate in self.gates):
            raise ValueError("comparison outcome must equal the conjunction of gates")
        return self


__all__ = [
    "EVAL_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "AggregateKey",
    "AggregateMetrics",
    "ArtifactIdentity",
    "CapabilityProfileIdentity",
    "ComparisonReport",
    "ConfidenceInterval",
    "DistributionSummary",
    "EvalRun",
    "EvaluationAggregate",
    "EvaluationDataset",
    "GateResult",
    "GroupAggregate",
    "MedianMetrics",
    "ModelIdentity",
    "RunArtifacts",
    "RunMetrics",
    "TaskCategory",
    "VerificationReceiptEvidence",
]
