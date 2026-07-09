"""Session state for durable verification receipts and legacy pass observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING

from vibe.core._verification_receipt import (
    ReceiptValidation,
    VerificationReceipt,
    VerificationReceiptStore,
    validate_receipt_id,
)
from vibe.core.verification_contract import VerificationReport

if TYPE_CHECKING:
    from vibe.core._verification_runner import TrustedCheck


def workspace_fingerprint() -> str | None:
    from vibe.core._workspace_verification import workspace_fingerprint as calculate

    return calculate()


@dataclass
class VerificationPass:
    source: str
    summary: str
    workspace_fingerprint: str | None
    report: VerificationReport | None = None
    recorded_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class VerificationReceiptReference:
    receipt_id: str
    repository_identity: str
    base_sha: str
    candidate_head: str
    task_brief_hash: str
    contract_hash: str
    configuration_hash: str
    checks_hash: str
    recipe_version: str
    recorded_at: float = field(default_factory=time.monotonic)


@dataclass
class VerificationState:
    receipt_store: VerificationReceiptStore = field(
        default_factory=VerificationReceiptStore, repr=False
    )
    receipt_reference: VerificationReceiptReference | None = None
    last_receipt_validation: ReceiptValidation | None = None
    last_contract_pass: VerificationPass | None = None
    last_verifier_pass: VerificationPass | None = None

    def record_contract_pass(self, summary: str) -> None:
        self.last_contract_pass = VerificationPass(
            source="workflow-contract",
            summary=summary,
            workspace_fingerprint=workspace_fingerprint(),
        )

    def record_verifier_pass(self, report: VerificationReport) -> None:
        if not report.passed:
            raise ValueError("only a passing verifier report can satisfy the gate")
        self.last_verifier_pass = VerificationPass(
            source="verifier-subagent",
            summary=report.summary(),
            workspace_fingerprint=workspace_fingerprint(),
            report=report,
        )

    def record_receipt(
        self,
        receipt: VerificationReceipt,
        *,
        store: VerificationReceiptStore | None = None,
    ) -> None:
        if not receipt.passed:
            raise ValueError("only a passing verification receipt can satisfy the gate")
        if store is not None:
            self.receipt_store = store
        self.receipt_store.persist_receipt(receipt)
        self.receipt_reference = _receipt_reference(receipt)
        self.last_receipt_validation = None

    def run_trusted_checks(
        self,
        checks: Sequence[TrustedCheck],
        *,
        repository_path: Path,
        base_sha: str,
        task_brief_hash: str,
        recipe_version: str,
        contract_hash: str,
        configuration_hash: str,
        allowed_paths: Sequence[str],
    ) -> VerificationReceipt:
        from vibe.core._verification_runner import run_trusted_verification

        receipt = run_trusted_verification(
            checks,
            repository_path=repository_path,
            base_sha=base_sha,
            task_brief_hash=task_brief_hash,
            recipe_version=recipe_version,
            contract_hash=contract_hash,
            configuration_hash=configuration_hash,
            allowed_paths=allowed_paths,
            store=self.receipt_store,
        )
        if receipt.passed:
            self.record_receipt(receipt)
        return receipt

    def has_valid_receipt(
        self,
        *,
        repository_path: Path,
        expected_base_sha: str,
        expected_candidate_head: str | None = None,
        receipt_id: str | None = None,
    ) -> bool:
        reference = self.receipt_reference
        selected_id = receipt_id or (reference.receipt_id if reference else None)
        if selected_id is None:
            self.last_receipt_validation = ReceiptValidation(
                receipt_id=None,
                valid=False,
                reasons=("no verification receipt was recorded",),
            )
            return False

        if reference is None or reference.receipt_id != selected_id:
            self.last_receipt_validation = ReceiptValidation(
                receipt_id=selected_id,
                valid=False,
                reasons=(
                    "receipt ID is not bound to the current trusted verification state",
                ),
            )
            return False

        validation = validate_receipt_id(
            selected_id,
            store=self.receipt_store,
            repository_path=repository_path,
            expected_base_sha=expected_base_sha,
            expected_candidate_head=expected_candidate_head,
            expected_task_brief_hash=reference.task_brief_hash,
            expected_contract_hash=reference.contract_hash,
            expected_configuration_hash=reference.configuration_hash,
            expected_checks_hash=reference.checks_hash,
            expected_recipe_version=reference.recipe_version,
        )
        self.last_receipt_validation = validation
        if validation.valid and validation.receipt is not None:
            self.receipt_reference = _receipt_reference(validation.receipt)
        return validation.valid

    def has_pass(self) -> bool:
        """Return the legacy in-session observation; never use this to authorize."""
        current = workspace_fingerprint()
        if current is None:
            return False
        return any(
            recorded is not None and recorded.workspace_fingerprint == current
            for recorded in (self.last_contract_pass, self.last_verifier_pass)
        )

    def latest(self) -> VerificationPass | None:
        c = self.last_contract_pass
        v = self.last_verifier_pass
        if c is None:
            return v
        if v is None:
            return c
        return c if c.recorded_at >= v.recorded_at else v

    def clear(self) -> None:
        self.receipt_reference = None
        self.last_receipt_validation = None
        self.last_contract_pass = None
        self.last_verifier_pass = None


def _receipt_reference(receipt: VerificationReceipt) -> VerificationReceiptReference:
    return VerificationReceiptReference(
        receipt_id=receipt.receipt_id,
        repository_identity=receipt.repository.repository_identity,
        base_sha=receipt.repository.base_sha,
        candidate_head=receipt.repository.candidate_head,
        task_brief_hash=receipt.task_brief_hash,
        contract_hash=receipt.contract_hash,
        configuration_hash=receipt.configuration_hash,
        checks_hash=receipt.checks_hash,
        recipe_version=receipt.recipe_version,
    )


__all__ = [
    "VerificationPass",
    "VerificationReceiptReference",
    "VerificationState",
    "workspace_fingerprint",
]
