"""Session state for durable verification receipts and legacy pass observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
import threading
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
    from vibe.core.worktree._trusted_git import TrustedGitError, TrustedGitWorktree
    from vibe.core.worktree.manager import worktree_manager

    handle = worktree_manager.active
    if handle is None:
        return None
    try:
        return TrustedGitWorktree.open(handle.original_repo_root).head_sha()
    except (OSError, TrustedGitError, ValueError):
        return None


@dataclass
class VerificationPass:
    source: str
    summary: str
    workspace_fingerprint: str | None
    base_sha: str | None
    verifier_attempt_generation: int
    report: VerificationReport | None = None
    recorded_at: float = field(default_factory=time.monotonic)


class VerifierAttemptDisposition(StrEnum):
    PENDING = auto()
    PASS = auto()
    FAIL = auto()
    PARTIAL = auto()
    INVALID = auto()


class VerificationCompletionStatus(StrEnum):
    IN_PROGRESS = auto()
    UNVERIFIED = auto()
    PARTIAL = auto()
    BLOCKED = auto()


_MAX_DISPLAY_TODOS = 20
_MAX_DISPLAY_TODO_LENGTH = 80


def _format_open_todos(todo_ids: tuple[str, ...]) -> str:
    visible = todo_ids[:_MAX_DISPLAY_TODOS]
    rendered = []
    for identifier in visible:
        shortened = identifier[:_MAX_DISPLAY_TODO_LENGTH]
        if len(identifier) > _MAX_DISPLAY_TODO_LENGTH:
            shortened += "..."
        rendered.append(ascii(shortened))
    if len(todo_ids) > len(visible):
        rendered.append(f"(+{len(todo_ids) - len(visible)} more)")
    return ", ".join(rendered)


@dataclass(frozen=True, slots=True)
class VerifierAttemptResult:
    generation: int
    disposition: VerifierAttemptDisposition
    diagnostic: str
    recorded_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class VerificationCompletionConstraint:
    generation: int
    status: VerificationCompletionStatus
    disposition: VerifierAttemptDisposition
    diagnostic: str

    def render(self) -> str:
        status = self.status.value.upper()
        return (
            f"HOST VERIFICATION STATUS: {status}\n\n"
            f"{self.diagnostic}\n\n"
            "The host did not record current authorization for completed work. "
            "This result cannot be described as verified, complete, ready for "
            "acceptance, or safe to land."
        )


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
    verifier_attempt_generation: int
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

    def checks_hash_for(self, repository_path: Path) -> str:
        repository_root = repository_path.resolve()
        return hash_payload([
            {
                "name": check.name,
                "argv": check.argv,
                "cwd": str(
                    (
                        Path(check.cwd)
                        if Path(check.cwd).is_absolute()
                        else repository_root / check.cwd
                    ).resolve()
                ),
                "timeout_seconds": check.timeout_seconds,
            }
            for check in self.checks
        ])

    @classmethod
    def from_config(
        cls, recipe: TrustedVerificationRecipeConfig
    ) -> BoundVerificationRecipe:
        from vibe.core._verification_runner import TrustedCheck

        checks = tuple(
            TrustedCheck(
                name=check.name,
                argv=check.argv,
                cwd=check.cwd,
                timeout_seconds=check.timeout_seconds,
                executable_sha256=check.executable_sha256,
                environment_attestation_path=check.environment_attestation_path,
                environment_attestation_sha256=check.environment_attestation_sha256,
                required_output_patterns=check.required_output_patterns,
                forbidden_output_patterns=check.forbidden_output_patterns,
                test_count_pattern=check.test_count_pattern,
                minimum_test_count=check.minimum_test_count,
                custom_runner=check.custom_runner,
            )
            for check in recipe.checks
        )
        return cls(
            config=recipe.model_copy(deep=True),
            checks=checks,
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
    latest_verifier_attempt: VerifierAttemptResult | None = None
    workspace_baseline_fingerprint: str | None = None
    workspace_baseline_observed: bool = False
    verification_required: bool = False
    open_todo_ids: tuple[str, ...] = ()
    _authorization_generation: int | None = field(default=None, init=False, repr=False)
    _authorization_reservation_receipt_id: str | None = field(
        default=None, init=False, repr=False
    )
    _authorization_invalidation_pending: bool = field(
        default=False, init=False, repr=False
    )
    _authorization_clear_pending: bool | None = field(
        default=None, init=False, repr=False
    )
    _authorization_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

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
        with self._authorization_lock:
            if self._authorization_generation is not None:
                raise RuntimeError(
                    "cannot start a verifier while a verification authorization "
                    "transaction is in progress"
                )
            self.verifier_attempt_generation += 1
            self.receipt_reference = None
            self.last_receipt_validation = None
            self.last_verifier_pass = None
            self.latest_verifier_attempt = VerifierAttemptResult(
                generation=self.verifier_attempt_generation,
                disposition=VerifierAttemptDisposition.PENDING,
                diagnostic="Independent verification is still running.",
            )
            return self.verifier_attempt_generation

    def reserve_verifier_delivery(self, generation: int) -> bool:
        with self._authorization_lock:
            attempt = self.latest_verifier_attempt
            if (
                self._authorization_generation is not None
                or generation != self.verifier_attempt_generation
                or attempt is None
                or attempt.generation != generation
                or attempt.disposition is not VerifierAttemptDisposition.PENDING
            ):
                return False
            self._authorization_generation = generation
            return True

    def reserve_landing_authorization(self, generation: int, receipt_id: str) -> bool:
        with self._authorization_lock:
            reference = self.receipt_reference
            if (
                self._authorization_generation is not None
                or self._current_verifier_pass_generation_unlocked() != generation
                or reference is None
                or reference.receipt_id != receipt_id
                or reference.verifier_attempt_generation != generation
            ):
                return False
            self._authorization_generation = generation
            self._authorization_reservation_receipt_id = receipt_id
            return True

    def release_authorization(
        self, generation: int, *, receipt_id: str | None = None
    ) -> bool:
        with self._authorization_lock:
            if (
                self._authorization_generation != generation
                or self._authorization_reservation_receipt_id != receipt_id
            ):
                raise RuntimeError(
                    "verification authorization reservation does not match"
                )
            self._authorization_generation = None
            self._authorization_reservation_receipt_id = None
            clear_pending = self._authorization_clear_pending
            invalidation_pending = self._authorization_invalidation_pending
            self._authorization_clear_pending = None
            self._authorization_invalidation_pending = False
            if clear_pending is not None:
                self._clear_unlocked(preserve_requirement=clear_pending)
                return False
            if invalidation_pending:
                self._invalidate_authorization_unlocked()
                return False
            return self.current_verifier_pass_generation() == generation

    def observe_workspace_baseline(self) -> None:
        if self.workspace_baseline_observed:
            return
        self.workspace_baseline_fingerprint = workspace_fingerprint()
        self.workspace_baseline_observed = True

    def observe_workspace_change(self) -> None:
        current = workspace_fingerprint()
        if not self.workspace_baseline_observed:
            self.workspace_baseline_fingerprint = current
            self.workspace_baseline_observed = True
            return
        if current != self.workspace_baseline_fingerprint:
            self.verification_required = True

    def record_open_todos(self, todo_ids: Sequence[str]) -> None:
        self.open_todo_ids = tuple(dict.fromkeys(todo_ids))

    def record_candidate_mutation(self, *, invalidate_authorization: bool) -> None:
        self.verification_required = True
        if not invalidate_authorization:
            return
        with self._authorization_lock:
            if self._authorization_generation is not None:
                self._authorization_invalidation_pending = True
                return
            self._invalidate_authorization_unlocked()

    def _invalidate_authorization_unlocked(self) -> None:
        self.verifier_attempt_generation += 1
        self.receipt_reference = None
        self.last_receipt_validation = None
        self.last_verifier_pass = None
        self.latest_verifier_attempt = None

    def is_current_verifier_attempt(self, generation: int) -> bool:
        with self._authorization_lock:
            return generation == self.verifier_attempt_generation

    def is_pending_verifier_attempt(self, generation: int) -> bool:
        with self._authorization_lock:
            attempt = self.latest_verifier_attempt
            return bool(
                generation == self.verifier_attempt_generation
                and attempt is not None
                and attempt.generation == generation
                and attempt.disposition is VerifierAttemptDisposition.PENDING
            )

    def record_verifier_result(
        self,
        generation: int | None,
        disposition: VerifierAttemptDisposition,
        diagnostic: str,
    ) -> bool:
        with self._authorization_lock:
            selected_generation = (
                self.verifier_attempt_generation if generation is None else generation
            )
            reserved_generation = self._authorization_generation
            if (
                reserved_generation is not None
                and selected_generation != reserved_generation
            ):
                if selected_generation > reserved_generation:
                    self._authorization_invalidation_pending = True
                return False
            attempt = self.latest_verifier_attempt
            if (
                reserved_generation is not None
                and attempt is not None
                and attempt.disposition is VerifierAttemptDisposition.PASS
                and disposition is not VerifierAttemptDisposition.PASS
            ):
                self._authorization_invalidation_pending = True
                return False
            if (
                selected_generation != self.verifier_attempt_generation
                or disposition is VerifierAttemptDisposition.PENDING
                or attempt is None
                or attempt.generation != selected_generation
            ):
                return False
            if attempt.disposition is not VerifierAttemptDisposition.PENDING:
                if (
                    attempt.disposition is VerifierAttemptDisposition.PASS
                    and disposition is not VerifierAttemptDisposition.PASS
                ):
                    self.verifier_attempt_generation += 1
                    self.receipt_reference = None
                    self.last_receipt_validation = None
                    self.last_verifier_pass = None
                    self.latest_verifier_attempt = None
                    self.verification_required = True
                return False
            if disposition is not VerifierAttemptDisposition.PASS:
                self.receipt_reference = None
                self.last_receipt_validation = None
                self.last_verifier_pass = None
            self.latest_verifier_attempt = VerifierAttemptResult(
                generation=selected_generation,
                disposition=disposition,
                diagnostic=diagnostic,
            )
            return True

    def completion_constraint(
        self, *, receipt_valid: bool
    ) -> VerificationCompletionConstraint | None:
        with self._authorization_lock:
            if self._authorization_generation is not None:
                return VerificationCompletionConstraint(
                    generation=self._authorization_generation,
                    status=VerificationCompletionStatus.IN_PROGRESS,
                    disposition=VerifierAttemptDisposition.PENDING,
                    diagnostic=(
                        "A verification authorization transaction is still in progress."
                    ),
                )
            attempt = self.latest_verifier_attempt
            if attempt is not None:
                attempt_constraint = self._attempt_completion_constraint(
                    attempt, receipt_valid=receipt_valid
                )
                if attempt_constraint is not None:
                    return attempt_constraint
            if self.open_todo_ids:
                identifiers = _format_open_todos(self.open_todo_ids)
                return VerificationCompletionConstraint(
                    generation=self.verifier_attempt_generation,
                    status=VerificationCompletionStatus.PARTIAL,
                    disposition=VerifierAttemptDisposition.INVALID,
                    diagnostic=(
                        "The task ledger still has unfinished items: "
                        f"{identifiers}. Complete or explicitly cancel them before "
                        "claiming the work is complete."
                    ),
                )
            if attempt is None:
                if self.verification_required:
                    return VerificationCompletionConstraint(
                        generation=self.verifier_attempt_generation,
                        status=VerificationCompletionStatus.UNVERIFIED,
                        disposition=VerifierAttemptDisposition.INVALID,
                        diagnostic=(
                            "The candidate requires verification, but no current "
                            "independent verifier attempt was recorded."
                        ),
                    )
            return None

    def _attempt_completion_constraint(
        self, attempt: VerifierAttemptResult, *, receipt_valid: bool
    ) -> VerificationCompletionConstraint | None:
        if attempt.disposition is VerifierAttemptDisposition.PASS:
            return self._pass_completion_constraint(attempt, receipt_valid)
        status = VerificationCompletionStatus.BLOCKED
        if attempt.disposition is VerifierAttemptDisposition.PENDING:
            status = VerificationCompletionStatus.IN_PROGRESS
        elif attempt.disposition is VerifierAttemptDisposition.PARTIAL:
            status = VerificationCompletionStatus.PARTIAL
        return VerificationCompletionConstraint(
            generation=attempt.generation,
            status=status,
            disposition=attempt.disposition,
            diagnostic=attempt.diagnostic,
        )

    def completion_claim_is_authorized(self, *, receipt_valid: bool) -> bool:
        with self._authorization_lock:
            if self.current_verifier_pass_generation() is None:
                return False
            if self.trusted_recipe is not None:
                return receipt_valid
            return self.has_verifier_pass()

    def _pass_completion_constraint(
        self, attempt: VerifierAttemptResult, receipt_valid: bool
    ) -> VerificationCompletionConstraint | None:
        if self.trusted_recipe is not None:
            if receipt_valid:
                return None
            return VerificationCompletionConstraint(
                generation=attempt.generation,
                status=VerificationCompletionStatus.PARTIAL,
                disposition=attempt.disposition,
                diagnostic=(
                    "The verifier reported PASS, but the host-configured trusted "
                    "verification receipt is missing, stale, or invalid."
                ),
            )
        if self.has_verifier_pass():
            return None
        return VerificationCompletionConstraint(
            generation=attempt.generation,
            status=VerificationCompletionStatus.BLOCKED,
            disposition=VerifierAttemptDisposition.INVALID,
            diagnostic=(
                "The recorded verifier PASS no longer matches the current "
                "workspace or landing base."
            ),
        )

    def preserve_recipe_in_config(self, config: VibeConfig) -> VibeConfig:
        recipe = self.trusted_recipe
        return config.model_copy(
            update={
                "trusted_verification_recipe": (
                    recipe.config if recipe is not None else None
                ),
                "verification_subsystem": (
                    True if recipe is not None else config.verification_subsystem
                ),
            }
        )

    def record_verifier_pass(
        self,
        report: VerificationReport,
        *,
        verifier_attempt_generation: int | None = None,
        verified_workspace_fingerprint: str | None = None,
        verified_base_sha: str | None = None,
    ) -> None:
        if not report.passed:
            raise ValueError("only a passing verifier report can satisfy the gate")
        fingerprint = verified_workspace_fingerprint or workspace_fingerprint()
        base_sha = verified_base_sha or landing_base_sha()
        with self._authorization_lock:
            selected_generation = (
                self.verifier_attempt_generation
                if verifier_attempt_generation is None
                else verifier_attempt_generation
            )
            if selected_generation != self.verifier_attempt_generation:
                raise ValueError("verifier PASS belongs to a superseded attempt")
            attempt = self.latest_verifier_attempt
            if attempt is not None and (
                attempt.generation != selected_generation
                or attempt.disposition is not VerifierAttemptDisposition.PASS
            ):
                raise ValueError("verifier PASS requires the current terminal PASS")
            self.last_verifier_pass = VerificationPass(
                source="verifier-subagent",
                summary=report.summary(),
                workspace_fingerprint=fingerprint,
                base_sha=base_sha,
                verifier_attempt_generation=selected_generation,
                report=report,
            )

    def record_receipt(
        self,
        receipt: VerificationReceipt,
        *,
        verifier_attempt_generation: int,
        store: VerificationReceiptStore | None = None,
    ) -> None:
        if not receipt.passed:
            raise ValueError("only a passing verification receipt can satisfy the gate")
        with self._authorization_lock:
            expected_store = self.receipt_store
            target_store = store or expected_store
        self._publish_receipt(
            receipt,
            verifier_attempt_generation=verifier_attempt_generation,
            expected_store=expected_store,
            target_store=target_store,
        )

    def _publish_receipt(
        self,
        receipt: VerificationReceipt,
        *,
        verifier_attempt_generation: int,
        expected_store: VerificationReceiptStore,
        target_store: VerificationReceiptStore,
    ) -> None:
        with self._authorization_lock:
            self._require_current_pass_generation(verifier_attempt_generation)
            if self.receipt_store is not expected_store:
                raise ValueError(
                    "verification receipt store changed before publication"
                )
            expected_reference = self.receipt_reference
        target_store.persist_receipt(receipt)
        with self._authorization_lock:
            self._require_current_pass_generation(verifier_attempt_generation)
            if self.receipt_store is not expected_store:
                raise ValueError(
                    "verification receipt store changed during publication"
                )
            if self.receipt_reference is not expected_reference:
                raise ValueError(
                    "verification receipt reference changed during publication"
                )
            self.receipt_store = target_store
            self.receipt_reference = _receipt_reference(
                receipt, verifier_attempt_generation=verifier_attempt_generation
            )
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
        verifier_attempt_generation: int | None = None,
        publish: bool = True,
    ) -> VerificationReceipt:
        from vibe.core._verification_runner import run_trusted_verification

        with self._authorization_lock:
            selected_generation = (
                self.verifier_attempt_generation
                if verifier_attempt_generation is None
                else verifier_attempt_generation
            )
            selected_store = self.receipt_store
        receipt = run_trusted_verification(
            checks,
            repository_path=repository_path,
            base_sha=base_sha,
            task_brief_hash=task_brief_hash,
            recipe_version=recipe_version,
            contract_hash=contract_hash,
            configuration_hash=configuration_hash,
            allowed_paths=allowed_paths,
            store=selected_store,
        )
        if receipt.passed and publish:
            self._publish_receipt(
                receipt,
                verifier_attempt_generation=selected_generation,
                expected_store=selected_store,
                target_store=selected_store,
            )
        return receipt

    def run_bound_recipe(
        self,
        *,
        repository_path: Path,
        base_sha: str,
        verifier_attempt_generation: int | None = None,
        publish: bool = True,
    ) -> VerificationReceipt:
        with self._authorization_lock:
            recipe = self.trusted_recipe
            if recipe is None:
                raise ValueError(
                    "no trusted verification recipe is bound to this session"
                )
            selected_generation = (
                self.verifier_attempt_generation
                if verifier_attempt_generation is None
                else verifier_attempt_generation
            )
            self._require_current_pass_generation(selected_generation)
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
            verifier_attempt_generation=selected_generation,
            publish=publish,
        )

    def has_valid_receipt(
        self,
        *,
        repository_path: Path,
        expected_base_sha: str,
        expected_candidate_head: str | None = None,
        receipt_id: str | None = None,
    ) -> bool:
        with self._authorization_lock:
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
                        "receipt ID is not bound to the current trusted "
                        "verification state",
                    ),
                )
                return False
            current_generation = self.current_verifier_pass_generation()
            if (
                current_generation is None
                or reference.verifier_attempt_generation != current_generation
            ):
                self.last_receipt_validation = ReceiptValidation(
                    receipt_id=selected_id,
                    valid=False,
                    reasons=(
                        "receipt is not bound to the current verifier PASS generation",
                    ),
                )
                return False
            recipe = self.trusted_recipe
            if recipe is not None and not _reference_matches_recipe(
                reference, recipe, repository_path
            ):
                self.last_receipt_validation = ReceiptValidation(
                    receipt_id=selected_id,
                    valid=False,
                    reasons=(
                        "receipt reference does not match the frozen trusted recipe",
                    ),
                )
                return False
            store = self.receipt_store
        validation = validate_receipt_id(
            selected_id,
            store=store,
            repository_path=repository_path,
            expected_base_sha=expected_base_sha,
            expected_candidate_head=expected_candidate_head,
            expected_task_brief_hash=reference.task_brief_hash,
            expected_contract_hash=reference.contract_hash,
            expected_configuration_hash=reference.configuration_hash,
            expected_checks_hash=reference.checks_hash,
            expected_recipe_version=reference.recipe_version,
        )
        with self._authorization_lock:
            if (
                self.receipt_reference is not reference
                or self.receipt_store is not store
                or self.trusted_recipe is not recipe
                or self.current_verifier_pass_generation() != current_generation
            ):
                self.last_receipt_validation = ReceiptValidation(
                    receipt_id=selected_id,
                    valid=False,
                    reasons=(
                        "verification authority changed during receipt validation",
                    ),
                )
                return False
            self.last_receipt_validation = validation
            if validation.valid and validation.receipt is not None:
                self.receipt_reference = _receipt_reference(
                    validation.receipt, verifier_attempt_generation=current_generation
                )
            return validation.valid

    def has_pass(self, *, expected_base_sha: str | None = None) -> bool:
        """Return whether a legacy pass still matches the current workspace."""
        current = workspace_fingerprint()
        if current is None:
            return False
        with self._authorization_lock:
            if self._authorization_generation is not None:
                return False
            return _pass_matches(self.last_verifier_pass, current, expected_base_sha)

    def has_verifier_pass(self, *, expected_base_sha: str | None = None) -> bool:
        current = workspace_fingerprint()
        if current is None:
            return False
        with self._authorization_lock:
            current_generation = self.current_verifier_pass_generation()
            recorded = self.last_verifier_pass
            return (
                current_generation is not None
                and recorded is not None
                and recorded.verifier_attempt_generation == current_generation
                and _pass_matches(recorded, current, expected_base_sha)
            )

    def current_verifier_pass_generation(self) -> int | None:
        with self._authorization_lock:
            if self._authorization_generation is not None:
                return None
            return self._current_verifier_pass_generation_unlocked()

    def _current_verifier_pass_generation_unlocked(self) -> int | None:
        attempt = self.latest_verifier_attempt
        if (
            attempt is None
            or attempt.generation != self.verifier_attempt_generation
            or attempt.disposition is not VerifierAttemptDisposition.PASS
        ):
            return None
        return attempt.generation

    def _require_current_pass_generation(self, generation: int) -> None:
        if self.current_verifier_pass_generation() != generation:
            raise ValueError(
                "verification receipt requires the current verifier PASS generation"
            )

    def latest(
        self, *, expected_base_sha: str | None = None
    ) -> VerificationPass | None:
        current = workspace_fingerprint()
        if current is None:
            return None
        with self._authorization_lock:
            if self._authorization_generation is not None:
                return None
            return (
                self.last_verifier_pass
                if _pass_matches(self.last_verifier_pass, current, expected_base_sha)
                else None
            )

    def clear(self, *, preserve_requirement: bool = False) -> None:
        with self._authorization_lock:
            if self._authorization_generation is not None:
                pending = self._authorization_clear_pending
                self._authorization_clear_pending = (
                    preserve_requirement
                    if pending is None
                    else pending and preserve_requirement
                )
                return
            self._clear_unlocked(preserve_requirement=preserve_requirement)

    def _clear_unlocked(self, *, preserve_requirement: bool) -> None:
        self.verifier_attempt_generation += 1
        self.receipt_reference = None
        self.last_receipt_validation = None
        self.last_verifier_pass = None
        self.latest_verifier_attempt = None
        if not preserve_requirement:
            self.open_todo_ids = ()
            self.workspace_baseline_fingerprint = None
            self.workspace_baseline_observed = False
            self.verification_required = False


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


def _reference_matches_recipe(
    reference: VerificationReceiptReference,
    recipe: BoundVerificationRecipe,
    repository_path: Path,
) -> bool:
    observed = (
        reference.task_brief_hash,
        reference.contract_hash,
        reference.configuration_hash,
        reference.checks_hash,
        reference.recipe_version,
    )
    expected = (
        recipe.task_brief_hash,
        recipe.contract_hash,
        recipe.configuration_hash,
        recipe.checks_hash_for(repository_path),
        recipe.recipe_version,
    )
    return observed == expected


def _receipt_reference(
    receipt: VerificationReceipt, *, verifier_attempt_generation: int
) -> VerificationReceiptReference:
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
        verifier_attempt_generation=verifier_attempt_generation,
    )


__all__ = [
    "BoundVerificationRecipe",
    "VerificationCompletionConstraint",
    "VerificationCompletionStatus",
    "VerificationPass",
    "VerificationReceiptReference",
    "VerificationState",
    "VerifierAttemptDisposition",
    "VerifierAttemptResult",
    "landing_base_sha",
    "workspace_fingerprint",
]
