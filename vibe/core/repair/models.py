from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum, auto
import hashlib
import json
import math
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from vibe.core.failure_diagnostic import FailureCategory, FailureDiagnostic

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_FAILURE_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16}")


class RepairAction(StrEnum):
    CONTINUE = auto()
    WARN = auto()
    STOP = auto()
    ESCALATE = auto()
    RECOVERED = auto()


class RepairEpisodeOutcome(StrEnum):
    RECOVERED = auto()
    NOT_RECOVERED = auto()


class ProgressSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    diff_hash: str = Field(min_length=1)
    error_fingerprint: str = Field(min_length=1)
    acceptance_state_hash: str = Field(min_length=1)
    newly_read_files_hash: str = Field(min_length=1)
    tool_effect_hash: str = Field(min_length=1)

    @field_validator(
        "diff_hash",
        "acceptance_state_hash",
        "newly_read_files_hash",
        "tool_effect_hash",
    )
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _DIGEST_RE.fullmatch(value) is None:
            raise ValueError("progress component hash must be a SHA-256 hex digest")
        return value

    @field_validator("error_fingerprint")
    @classmethod
    def validate_error_fingerprint(cls, value: str) -> str:
        if _FAILURE_FINGERPRINT_RE.fullmatch(value) is None:
            raise ValueError("error fingerprint must be 16 lowercase hex characters")
        return value

    @classmethod
    def from_state(
        cls,
        *,
        diff_state: JsonValue,
        error_fingerprint: str,
        acceptance_state: JsonValue,
        newly_read_files: Iterable[str],
        tool_effect: JsonValue,
    ) -> ProgressSnapshot:
        return cls(
            diff_hash=_canonical_digest(diff_state),
            error_fingerprint=error_fingerprint,
            acceptance_state_hash=_canonical_digest(acceptance_state),
            newly_read_files_hash=_canonical_digest(sorted(set(newly_read_files))),
            tool_effect_hash=_canonical_digest(tool_effect),
        )

    @property
    def semantic_fingerprint(self) -> str:
        identity = "\0".join((
            self.diff_hash,
            self.error_fingerprint,
            self.acceptance_state_hash,
            self.newly_read_files_hash,
            self.tool_effect_hash,
        ))
        return hashlib.sha256(identity.encode()).hexdigest()


class FailureRetryBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    category: FailureCategory
    max_attempts: int = Field(ge=0)


class RetryBudgetSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    budgets: tuple[FailureRetryBudget, ...]
    default_max_attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def reject_duplicate_categories(self) -> RetryBudgetSet:
        categories = [budget.category for budget in self.budgets]
        if len(categories) != len(set(categories)):
            raise ValueError("retry budgets must contain each failure category once")
        return self

    def max_attempts_for(self, category: FailureCategory) -> int:
        for budget in self.budgets:
            if budget.category is category:
                return budget.max_attempts
        return self.default_max_attempts

    @classmethod
    def finite_defaults(cls) -> RetryBudgetSet:
        attempts = {
            FailureCategory.TOOL_ARGUMENT_PARSE: 4,
            FailureCategory.TOOL_ARGUMENT_SCHEMA: 4,
            FailureCategory.RESULT_SCHEMA: 4,
            FailureCategory.ACCEPTANCE_CHECK: 4,
            FailureCategory.PROVIDER_TRANSPORT: 4,
            FailureCategory.POLICY: 0,
            FailureCategory.BUDGET: 0,
            FailureCategory.NO_PROGRESS: 3,
        }
        return cls(
            budgets=tuple(
                FailureRetryBudget(category=category, max_attempts=maximum)
                for category, maximum in attempts.items()
            ),
            default_max_attempts=0,
        )


class RepairEpisodeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    category: FailureCategory
    outcome: RepairEpisodeOutcome
    finished: bool
    attempts: int = Field(ge=0)
    added_tokens: int = Field(ge=0)
    added_cost_usd: float = Field(ge=0)
    escalation_reason: str | None = None
    terminal_reason: str | None = None

    @field_validator("added_cost_usd", mode="before")
    @classmethod
    def validate_added_cost(cls, value: object) -> object:
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("added repair cost must be finite")
        return value

    @property
    def recovered(self) -> bool:
        return self.outcome is RepairEpisodeOutcome.RECOVERED


class RepairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: RepairAction
    category: FailureCategory
    attempt: int = Field(ge=0)
    remaining_attempts: int = Field(ge=0)
    made_progress: bool
    no_progress_strikes: int = Field(ge=0)
    reason: str
    escalation_reason: str | None = None
    diagnostic: FailureDiagnostic | None = None
    snapshot_fingerprint: str | None = None
    metrics: RepairEpisodeMetrics


def _canonical_digest(value: object) -> str:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("progress state must be canonical JSON data") from exc
    return hashlib.sha256(canonical.encode()).hexdigest()
