from __future__ import annotations

from enum import StrEnum, auto
import re

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_FULL_SHA = re.compile(r"[0-9a-f]{40}")


class CandidateDeliveryStatus(StrEnum):
    NOT_REQUESTED = auto()
    NO_CHANGES = auto()
    LANDED = auto()
    PRESERVED = auto()


class CandidateIntegrationMethod(StrEnum):
    FAST_FORWARD = auto()
    MERGE = auto()
    ALREADY_CONTAINED = auto()


class CandidateDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CandidateDeliveryStatus
    base_sha: str | None = None
    candidate_sha: str | None = None
    parent_sha_before: str | None = None
    parent_sha_after: str | None = None
    branch: str | None = None
    worktree_path: str | None = None
    integration_method: CandidateIntegrationMethod | None = None
    diagnostic: str | None = None

    @field_validator(
        "base_sha", "candidate_sha", "parent_sha_before", "parent_sha_after"
    )
    @classmethod
    def validate_sha(cls, value: str | None) -> str | None:
        if value is not None and _FULL_SHA.fullmatch(value) is None:
            raise ValueError("candidate delivery SHAs must be full lowercase Git SHAs")
        return value

    @model_validator(mode="after")
    def validate_status_fields(self) -> CandidateDelivery:
        if self.status is CandidateDeliveryStatus.LANDED:
            required = (
                self.base_sha,
                self.candidate_sha,
                self.parent_sha_before,
                self.parent_sha_after,
                self.integration_method,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "landed delivery requires base, candidate, parent, and "
                    "integration authority"
                )
            if (
                self.integration_method is CandidateIntegrationMethod.FAST_FORWARD
                and self.parent_sha_after != self.candidate_sha
            ):
                raise ValueError(
                    "fast-forward delivery requires parent HEAD to equal candidate"
                )
            if (
                self.integration_method is CandidateIntegrationMethod.ALREADY_CONTAINED
                and self.parent_sha_before != self.parent_sha_after
            ):
                raise ValueError(
                    "already-contained delivery must not change parent HEAD"
                )
        elif self.integration_method is not None:
            raise ValueError("only landed delivery may declare an integration method")
        if self.status is CandidateDeliveryStatus.NO_CHANGES:
            if self.base_sha is None or self.candidate_sha != self.base_sha:
                raise ValueError("no-changes delivery requires candidate to equal base")
            if self.parent_sha_before is None or self.parent_sha_after is None:
                raise ValueError("no-changes delivery requires observed parent SHAs")
            if self.parent_sha_before != self.parent_sha_after:
                raise ValueError("no-changes delivery must not change parent HEAD")
        return self

    @property
    def accepted(self) -> bool:
        return self.status in {
            CandidateDeliveryStatus.LANDED,
            CandidateDeliveryStatus.NO_CHANGES,
        }

    @property
    def preserved(self) -> bool:
        return self.status is CandidateDeliveryStatus.PRESERVED


__all__ = ["CandidateDelivery", "CandidateDeliveryStatus", "CandidateIntegrationMethod"]
