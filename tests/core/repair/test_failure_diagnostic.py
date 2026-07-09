from __future__ import annotations

from pydantic import ValidationError
import pytest

from vibe.core.failure_diagnostic import (
    FailureCategory,
    FailureDiagnostic,
    build_failure_diagnostic,
)


def test_failure_categories_cover_repair_boundaries() -> None:
    assert {
        FailureCategory.RESULT_SCHEMA,
        FailureCategory.ACCEPTANCE_CHECK,
        FailureCategory.PROVIDER_TRANSPORT,
        FailureCategory.POLICY,
        FailureCategory.BUDGET,
    }.issubset(set(FailureCategory))


def test_failure_fingerprint_is_stable_and_hex() -> None:
    first = build_failure_diagnostic(
        category=FailureCategory.RESULT_SCHEMA,
        message="Unknown result field",
        field="$.answer",
        expected="string",
        actual="42",
        suggested_action="return a string answer",
    )
    second = build_failure_diagnostic(
        category=FailureCategory.RESULT_SCHEMA,
        message="Unknown result field",
        field="$.answer",
        expected="string",
        actual="different invalid output",
        suggested_action="return only the corrected field",
    )

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 16
    assert int(first.fingerprint, 16) >= 0


def test_failure_diagnostic_rejects_malformed_fingerprint() -> None:
    with pytest.raises(ValidationError, match="16 lowercase hex"):
        FailureDiagnostic(
            category=FailureCategory.RESULT_SCHEMA,
            fingerprint="not-a-digest",
            message="invalid",
            retryable=True,
            suggested_action="correct it",
        )
