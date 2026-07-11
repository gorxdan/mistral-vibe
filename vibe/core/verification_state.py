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
    hash_payload,
    validate_receipt_id,
)
from vibe.core.verification_contract import VerificationReport

if TYPE_CHECKING:
    from vibe.core._verification_runner import TrustedCheck
    from vibe.core.config import TrustedVerificationRecipeConfig, VibeConfig


def workspace_fingerprint() -> str | None:
    from vibe.core._workspace_verification import workspace_fingerprint as calculate

    return calculate()


def landing_base_sha() -> str | None:
    from git import Repo
    from git.exc import GitError

    from vibe.core.worktree.manager import worktree_manager

    handle = worktree_manager.active
    if handle is None:
        return None
    try:
        return Repo(str(handle.original_repo_root)).head.commit.hexsha
    except (GitError, OSError, ValueError):
        return None


@dataclass
class VerificationPass:
    source: str
    summary: str
    workspace_fingerprint: str | None
    base_sha: str | None
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


@dataclass(frozen=True, slots=True)
class BoundVerificationRecipe:
    config: TrustedVerificationRecipeConfig
    checks: tuple[TrustedCheck, ...]
    recipe_version: str
    task_brief_hash: str
    contract_hash: str
    configuration_hash: str
    allowed_paths: tuple[str, ...]

    @classmethod
    def from_config(
        cls, recipe: TrustedVerificationRecipeConfig
    ) -> BoundVerificationRecipe:
        from vibe.core._verification_runner import TrustedCheck

        return cls(
            config=recipe.model_copy(deep=True),
            checks=tuple(
                TrustedCheck(
                    name=check.name,
                    argv=check.argv,
                    cwd=check.cwd,
                    timeout_seconds=check.timeout_seconds,
                )
                for check in recipe.checks
            ),
            recipe_version=recipe.recipe_version,
            task_brief_hash=hash_payload(recipe.task_brief),
            contract_hash=hash_payload(recipe.acceptance_contract),
            configuration_hash=hash_payload(recipe),
            allowed_paths=recipe.allowed_paths,
        )


@dataclass
class VerificationState:
    receipt_store: VerificationReceiptStore = field(
        default_factory=VerificationReceiptStore, repr=False
    )
    receipt_reference: VerificationReceiptReference | None = None
    last_receipt_validation: ReceiptValidation | None = None
    last_verifier_pass: VerificationPass | None = None
    trusted_recipe: BoundVerificationRecipe | None = None
    verifier_attempt_generation: int = 0

    @classmethod
    def from_recipe(
        cls, recipe: TrustedVerificationRecipeConfig | None
    ) -> VerificationState:
        return cls(
            trusted_recipe=(
                BoundVerificationRecipe.from_config(recipe)
                if recipe is not None
                else None
            )
        )

    def begin_verifier_attempt(self) -> int:
        self.verifier_attempt_generation += 1
        self.receipt_reference = None
        self.last_receipt_validation = None
        self.last_verifier_pass = None
        return self.verifier_attempt_generation

    def is_current_verifier_attempt(self, generation: int) -> bool:
        return generation == self.verifier_attempt_generation

    def preserve_recipe_in_config(self, config: VibeConfig) -> VibeConfig:
        recipe = self.trusted_recipe
        return config.model_copy(
            update={
                "trusted_verification_recipe": (
                    recipe.config if recipe is not None else None
                )
            }
        )

    def record_verifier_pass(
        self,
        report: VerificationReport,
        *,
        verified_workspace_fingerprint: str | None = None,
        verified_base_sha: str | None = None,
    ) -> None:
        if not report.passed:
            raise ValueError("only a passing verifier report can satisfy the gate")
        self.last_verifier_pass = VerificationPass(
            source="verifier-subagent",
            summary=report.summary(),
            workspace_fingerprint=(
                verified_workspace_fingerprint or workspace_fingerprint()
            ),
            base_sha=verified_base_sha or landing_base_sha(),
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

    def run_bound_recipe(
        self, *, repository_path: Path, base_sha: str
    ) -> VerificationReceipt:
        recipe = self.trusted_recipe
        if recipe is None:
            raise ValueError("no trusted verification recipe is bound to this session")
        self.receipt_reference = None
        self.last_receipt_validation = None
        return self.run_trusted_checks(
            recipe.checks,
            repository_path=repository_path,
            base_sha=base_sha,
            task_brief_hash=recipe.task_brief_hash,
            recipe_version=recipe.recipe_version,
            contract_hash=recipe.contract_hash,
            configuration_hash=recipe.configuration_hash,
            allowed_paths=recipe.allowed_paths,
        )

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

    def has_pass(self, *, expected_base_sha: str | None = None) -> bool:
        """Return whether a legacy pass still matches the current workspace."""
        current = workspace_fingerprint()
        if current is None:
            return False
        return _pass_matches(self.last_verifier_pass, current, expected_base_sha)

    def has_verifier_pass(self, *, expected_base_sha: str | None = None) -> bool:
        current = workspace_fingerprint()
        return current is not None and _pass_matches(
            self.last_verifier_pass, current, expected_base_sha
        )

    def latest(
        self, *, expected_base_sha: str | None = None
    ) -> VerificationPass | None:
        current = workspace_fingerprint()
        if current is None:
            return None
        v = (
            self.last_verifier_pass
            if _pass_matches(self.last_verifier_pass, current, expected_base_sha)
            else None
        )
        return v

    def clear(self) -> None:
        self.verifier_attempt_generation += 1
        self.receipt_reference = None
        self.last_receipt_validation = None
        self.last_verifier_pass = None


def _pass_matches(
    recorded: VerificationPass | None,
    current_fingerprint: str,
    expected_base_sha: str | None,
) -> bool:
    return bool(
        recorded is not None
        and recorded.workspace_fingerprint == current_fingerprint
        and (expected_base_sha is None or recorded.base_sha == expected_base_sha)
    )


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
    "BoundVerificationRecipe",
    "VerificationPass",
    "VerificationReceiptReference",
    "VerificationState",
    "landing_base_sha",
    "workspace_fingerprint",
]
