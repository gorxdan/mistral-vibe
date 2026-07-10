from __future__ import annotations

from enum import StrEnum, auto
import hashlib
import re

from pydantic import BaseModel, ConfigDict, field_validator

_FINGERPRINT_RE = re.compile(r"[0-9a-f]{16}")


class FailureCategory(StrEnum):
    TOOL_ARGUMENT_PARSE = auto()
    TOOL_ARGUMENT_SCHEMA = auto()
    RESULT_SCHEMA = auto()
    ACCEPTANCE_CHECK = auto()
    PROVIDER_TRANSPORT = auto()
    POLICY = auto()
    BUDGET = auto()
    NO_PROGRESS = auto()


class FailureDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    category: FailureCategory
    fingerprint: str
    message: str
    field: str | None = None
    expected: str | None = None
    actual: str | None = None
    retryable: bool = True
    evidence_pointer: str | None = None
    suggested_action: str

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if _FINGERPRINT_RE.fullmatch(value) is None:
            raise ValueError("failure fingerprint must be 16 lowercase hex characters")
        return value

    def for_model(self) -> str:
        location = f" Field: {self.field}." if self.field else ""
        expected = f" Expected: {self.expected}." if self.expected else ""
        evidence = f" Actual: {self.actual}." if self.actual else ""
        return (
            f"{self.message}{location}{expected}{evidence} "
            f"Next action: {self.suggested_action} "
            f"[failure={self.fingerprint}]"
        )


def build_failure_diagnostic(
    *,
    category: FailureCategory,
    message: str,
    field: str | None = None,
    expected: str | None = None,
    actual: str | None = None,
    retryable: bool = True,
    evidence_pointer: str | None = None,
    suggested_action: str,
) -> FailureDiagnostic:
    identity = "\0".join((category.value, field or "", expected or "", message))
    return FailureDiagnostic(
        category=category,
        fingerprint=hashlib.sha256(identity.encode()).hexdigest()[:16],
        message=message,
        field=field,
        expected=expected,
        actual=actual,
        retryable=retryable,
        evidence_pointer=evidence_pointer,
        suggested_action=suggested_action,
    )
