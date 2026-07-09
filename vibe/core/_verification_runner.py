from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import time

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vibe.core._verification_receipt import (
    OUTPUT_EXCERPT_CHARS,
    CheckEvidence,
    ReceiptBuildContext,
    ReceiptOutcome,
    VerificationReceipt,
    VerificationReceiptError,
    VerificationReceiptStore,
    allowed_paths_match,
    build_receipt,
    capture_repository_state,
    check_evidence_hash,
    validate_binding_hash,
)
from vibe.core.utils.io import decode_safe


class TrustedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float = Field(default=300.0, gt=0, le=3_600)

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument for argument in value):
            raise ValueError("check argv entries must be nonempty")
        return value


def run_trusted_verification(
    checks: Sequence[TrustedCheck],
    *,
    repository_path: Path,
    base_sha: str,
    task_brief_hash: str,
    recipe_version: str,
    contract_hash: str,
    configuration_hash: str,
    allowed_paths: Sequence[str],
    store: VerificationReceiptStore | None = None,
) -> VerificationReceipt:
    validate_binding_hash(task_brief_hash, "task brief hash", allow_empty=False)
    validate_binding_hash(contract_hash, "contract hash", allow_empty=False)
    validate_binding_hash(configuration_hash, "configuration hash")
    if not recipe_version.strip():
        raise VerificationReceiptError("recipe version must not be empty")
    started_at = datetime.now(UTC)
    receipt_store = store or VerificationReceiptStore()
    before = capture_repository_state(repository_path, base_sha)
    allowed_paths_passed = allowed_paths_match(before.changed_paths, allowed_paths)
    evidence = tuple(
        _run_check(
            check, repository_path.resolve(), receipt_store, before.repository_identity
        )
        for check in checks
    )
    after = capture_repository_state(repository_path, base_sha)
    state_unchanged = before == after
    outcome = (
        ReceiptOutcome.PASS
        if evidence
        and all(item.passed for item in evidence)
        and allowed_paths_passed
        and state_unchanged
        and not before.dirty
        else ReceiptOutcome.FAIL
    )
    completed_at = datetime.now(UTC)
    receipt = build_receipt(
        context=ReceiptBuildContext(
            task_brief_hash=task_brief_hash,
            recipe_version=recipe_version,
            repository=before,
            contract_hash=contract_hash,
            configuration_hash=configuration_hash,
            allowed_paths=tuple(allowed_paths),
            started_at=started_at,
            completed_at=completed_at,
        ),
        checks_hash=check_evidence_hash(evidence),
        allowed_paths_passed=allowed_paths_passed,
        evidence=evidence,
        outcome=outcome,
    )
    receipt_store.persist_receipt(receipt)
    return receipt


def _run_check(
    check: TrustedCheck,
    repository_root: Path,
    store: VerificationReceiptStore,
    repository_id: str,
) -> CheckEvidence:
    started = time.monotonic()
    stdout = b""
    stderr = b""
    exit_code: int | None = None
    timed_out = False
    requested_cwd = Path(check.cwd)
    cwd = (
        requested_cwd.resolve()
        if requested_cwd.is_absolute()
        else (repository_root / requested_cwd).resolve()
    )
    try:
        cwd = _resolve_cwd(repository_root, check.cwd)
        completed = subprocess.run(
            list(check.argv),
            cwd=str(cwd),
            shell=False,
            capture_output=True,
            timeout=check.timeout_seconds,
            check=False,
        )
        stdout = _to_bytes(completed.stdout)
        stderr = _to_bytes(completed.stderr)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _to_bytes(exc.stdout)
        stderr = _to_bytes(exc.stderr)
        timed_out = True
    except (OSError, ValueError, VerificationReceiptError) as exc:
        stderr = str(exc).encode("utf-8", errors="replace")

    duration_ms = max(0, round((time.monotonic() - started) * 1_000))
    artifact_hash, artifact_path, artifact_size = store.persist_artifact(
        repository_id, stdout, stderr
    )
    return CheckEvidence(
        name=check.name,
        argv=check.argv,
        cwd=str(cwd),
        timeout_seconds=check.timeout_seconds,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=duration_ms,
        stdout_excerpt=_bounded_excerpt(stdout),
        stderr_excerpt=_bounded_excerpt(stderr),
        output_artifact_hash=artifact_hash,
        output_artifact_path=artifact_path,
        output_artifact_size=artifact_size,
    )


def _resolve_cwd(repository_root: Path, requested: str) -> Path:
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = repository_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise VerificationReceiptError(
            f"check cwd escapes the repository: {requested!r}"
        ) from exc
    if not resolved.is_dir():
        raise VerificationReceiptError(f"check cwd is not a directory: {requested!r}")
    return resolved


def _bounded_excerpt(raw: bytes) -> str:
    text = decode_safe(raw, from_subprocess=True).text
    if len(text) <= OUTPUT_EXCERPT_CHARS:
        return text
    omitted = len(text) - OUTPUT_EXCERPT_CHARS
    marker = f"\n... [{omitted} chars omitted] ...\n"
    remaining = OUTPUT_EXCERPT_CHARS - len(marker)
    head = remaining // 2
    tail = remaining - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def _to_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


__all__ = ["TrustedCheck", "run_trusted_verification"]
