from __future__ import annotations

from datetime import datetime
from enum import StrEnum, auto
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.candidate_delivery import CandidateDelivery


def _nonempty(value: str, field: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must not be empty")
    return cleaned


def _scope_path(value: str) -> str:
    cleaned = _nonempty(value, "path")
    if "\\" in cleaned:
        raise ValueError("scope paths must use forward slashes")
    path = PurePosixPath(cleaned)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("scope paths must stay relative to the workspace")
    return path.as_posix()


class TaskOutcomeStatus(StrEnum):
    SUCCEEDED = auto()
    FAILED = auto()
    BLOCKED = auto()
    RETRYABLE = auto()


class TaskBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_calls: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def require_limit(self) -> TaskBudget:
        if (
            self.max_tokens is None
            and self.max_cost_usd is None
            and self.max_calls is None
        ):
            raise ValueError("task budget must define at least one finite limit")
        return self


class TaskManifestIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    digest: str | None = None

    @field_validator("name", "version", "digest")
    @classmethod
    def validate_identity_part(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonempty(value, "manifest identity")

    @property
    def identity(self) -> str:
        base = f"{self.name}@{self.version}"
        return f"{base}#{self.digest}" if self.digest else base


class TaskBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: str
    inputs: dict[str, str] = Field(default_factory=dict)
    allowed_paths: list[str] = Field(min_length=1)
    denied_paths: list[str] = Field(default_factory=list)
    acceptance_checks: list[str] = Field(min_length=1, max_length=8)
    budget: TaskBudget | None = None
    deadline: datetime | None = None
    manifest: TaskManifestIdentity

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        return _nonempty(value, "objective")

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for key, input_value in value.items():
            clean_key = _nonempty(key, "input name")
            if not input_value.strip():
                raise ValueError(f"input {clean_key!r} must not be empty")
            validated[clean_key] = input_value
        return validated

    @field_validator("allowed_paths", "denied_paths")
    @classmethod
    def validate_scope_paths(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(_scope_path(path) for path in value))

    @field_validator("acceptance_checks")
    @classmethod
    def validate_acceptance_checks(cls, value: list[str]) -> list[str]:
        return list(
            dict.fromkeys(_nonempty(check, "acceptance check") for check in value)
        )

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("deadline must include a timezone")
        return value

    @model_validator(mode="after")
    def reject_conflicting_paths(self) -> TaskBrief:
        conflict = set(self.allowed_paths) & set(self.denied_paths)
        if conflict:
            paths = ", ".join(sorted(conflict))
            raise ValueError(f"paths cannot be both allowed and denied: {paths}")
        return self


class TaskOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: TaskOutcomeStatus
    summary: str
    evidence: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)
    receipt_id: str | None = None
    remaining_work: list[str] = Field(default_factory=list)
    manifest: TaskManifestIdentity | None = None
    candidate_delivery: CandidateDelivery | None = None

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _nonempty(value, "outcome summary")

    @field_validator("evidence", "diagnostics", "remaining_work")
    @classmethod
    def validate_text_items(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(_nonempty(item, "outcome item") for item in value))

    @field_validator("changed_paths")
    @classmethod
    def validate_changed_paths(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(_scope_path(path) for path in value))

    @field_validator("receipt_id")
    @classmethod
    def validate_receipt_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonempty(value, "receipt id")

    @property
    def succeeded(self) -> bool:
        return self.status is TaskOutcomeStatus.SUCCEEDED

    @property
    def retryable(self) -> bool:
        return self.status is TaskOutcomeStatus.RETRYABLE
