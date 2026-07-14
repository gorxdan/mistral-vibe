from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core._trusted_command import (
    TRUSTED_GIT_CONFIG_ARGS,
    minimal_trusted_git_environment,
    resolve_trusted_system_executable,
    validate_trusted_command_argv,
)
from vibe.core._trusted_host_runner import (
    FrozenSourceSnapshot,
    TrustedEnvironmentAttestation,
    TrustedExecutable,
    cleanup_frozen_source_snapshot,
    cleanup_trusted_executable,
    create_frozen_source_snapshot,
    minimal_check_environment,
    resolve_environment_attestation,
    resolve_trusted_executable,
    run_bounded_process,
    validate_environment_attestation,
    validate_trusted_executable,
    verify_frozen_source_snapshot,
)
from vibe.core._verification_output import (
    output_regex_diagnostics,
    validate_custom_runner_contract,
    validate_output_patterns,
)
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
from vibe.core.tools.sandbox import (
    SandboxSpec,
    build_sandbox_command,
    resolve_backend,
    strict_read_hidden_roots,
)
from vibe.core.utils.io import decode_safe
from vibe.core.verification_contract import verification_command_output_diagnostics


@dataclass(frozen=True, slots=True)
class _SandboxInvocation:
    argv: list[str]
    env: dict[str, str]
    writable_root: Path
    executable: TrustedExecutable
    environment_attestation: TrustedEnvironmentAttestation | None


class TrustedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float = Field(default=300.0, gt=0, le=3_600)
    executable_sha256: str | None = None
    environment_attestation_path: str | None = None
    environment_attestation_sha256: str | None = None
    required_output_patterns: tuple[str, ...] = ()
    forbidden_output_patterns: tuple[str, ...] = ()
    test_count_pattern: str | None = None
    minimum_test_count: int | None = Field(default=None, ge=1)
    custom_runner: bool = False

    @field_validator("argv")
    @classmethod
    def _validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not argument for argument in value):
            raise ValueError("check argv entries must be nonempty")
        validate_trusted_command_argv(value)
        return value

    @field_validator("executable_sha256", "environment_attestation_sha256")
    @classmethod
    def _validate_executable_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("executable_sha256 must be a full lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _validate_environment_attestation(self) -> TrustedCheck:
        if (self.environment_attestation_path is None) != (
            self.environment_attestation_sha256 is None
        ):
            raise ValueError(
                "environment attestation path and SHA-256 must be configured together"
            )
        if self.environment_attestation_path is not None:
            path = Path(self.environment_attestation_path)
            if not path.is_absolute() or "\0" in self.environment_attestation_path:
                raise ValueError("environment attestation path must be absolute")
        return self

    @field_validator(
        "required_output_patterns", "forbidden_output_patterns", mode="after"
    )
    @classmethod
    def _validate_output_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validate_output_patterns(value)
        for pattern in value:
            _compile_output_pattern(pattern)
        return value

    @model_validator(mode="after")
    def _validate_test_count_evidence(self) -> TrustedCheck:
        if (self.test_count_pattern is None) != (self.minimum_test_count is None):
            raise ValueError(
                "test_count_pattern and minimum_test_count must be configured together"
            )
        if self.test_count_pattern is not None:
            validate_output_patterns((self.test_count_pattern,))
            pattern = _compile_output_pattern(self.test_count_pattern)
            if "count" not in pattern.groupindex:
                raise ValueError("test_count_pattern must define a named 'count' group")
        validate_custom_runner_contract(
            custom_runner=self.custom_runner,
            executable_sha256=self.executable_sha256,
            required_output_patterns=self.required_output_patterns,
            test_count_pattern=self.test_count_pattern,
            minimum_test_count=self.minimum_test_count,
        )
        return self


def _compile_output_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid verification output pattern: {exc}") from exc


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
    repository_root = repository_path.resolve()
    before = capture_repository_state(repository_root, base_sha)
    allowed_paths_passed = allowed_paths_match(before.changed_paths, allowed_paths)
    snapshot: FrozenSourceSnapshot | None = None
    snapshot_unchanged = False
    try:
        snapshot = create_frozen_source_snapshot(
            repository_root,
            candidate_head=before.candidate_head,
            candidate_tree=before.candidate_tree,
            git_common_root=_git_common_root(repository_root),
        )
        evidence = tuple(
            _run_check(
                check,
                repository_root,
                snapshot,
                receipt_store,
                before.repository_identity,
            )
            for check in checks
        )
        try:
            verify_frozen_source_snapshot(snapshot)
            snapshot_unchanged = True
        except VerificationReceiptError as exc:
            evidence = _attach_snapshot_failure(evidence, str(exc))
    except (OSError, ValueError, VerificationReceiptError) as exc:
        evidence = tuple(
            _failed_check_evidence(
                check,
                repository_root,
                receipt_store,
                before.repository_identity,
                str(exc),
            )
            for check in checks
        )
    finally:
        if snapshot is not None:
            cleanup_frozen_source_snapshot(snapshot)
    after = capture_repository_state(repository_root, base_sha)
    state_unchanged = before == after
    outcome = (
        ReceiptOutcome.PASS
        if evidence
        and all(item.passed for item in evidence)
        and all(item.executable_sha256 is not None for item in evidence)
        and all(item.environment_attestation_sha256 is not None for item in evidence)
        and allowed_paths_passed
        and state_unchanged
        and snapshot_unchanged
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
    snapshot: FrozenSourceSnapshot,
    store: VerificationReceiptStore,
    repository_id: str,
) -> CheckEvidence:
    started = time.monotonic()
    stdout = b""
    stderr = b""
    exit_code: int | None = None
    timed_out = False
    invocation: _SandboxInvocation | None = None
    logical_cwd = _unchecked_cwd(repository_root, check.cwd)
    try:
        logical_cwd = _resolve_cwd(repository_root, check.cwd)
        snapshot_cwd = (
            snapshot.source_root / logical_cwd.relative_to(repository_root)
        ).resolve()
        if not snapshot_cwd.is_relative_to(snapshot.source_root):
            raise VerificationReceiptError(
                f"check cwd escapes the frozen source snapshot: {check.cwd!r}"
            )
        if not snapshot_cwd.is_dir():
            raise VerificationReceiptError(
                f"check cwd is absent from the frozen source snapshot: {check.cwd!r}"
            )
        invocation = _prepare_sandbox_invocation(
            check, repository_root, snapshot, snapshot_cwd
        )
        validate_trusted_executable(invocation.executable)
        result = run_bounded_process(
            invocation.argv,
            cwd=snapshot_cwd,
            env=invocation.env,
            timeout_seconds=check.timeout_seconds,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.exit_code
        timed_out = result.timed_out
        if result.collector_error is not None:
            exit_code = None
            diagnostic = (
                "trusted verification output collection was incomplete: "
                f"{result.collector_error}"
            ).encode()
            if diagnostic not in stderr:
                stderr = b"\n".join(part for part in (stderr, diagnostic) if part)
        validate_trusted_executable(invocation.executable)
        validate_environment_attestation(invocation.environment_attestation)
    except (OSError, ValueError, VerificationReceiptError) as exc:
        diagnostic = str(exc).encode("utf-8", errors="replace")
        stderr = b"\n".join(part for part in (stderr, diagnostic) if part)
        exit_code = None
    finally:
        if invocation is not None:
            _cleanup_sandbox_invocation(invocation)

    duration_ms = max(0, round((time.monotonic() - started) * 1_000))
    artifact_hash, artifact_path, artifact_size = store.persist_artifact(
        repository_id, stdout, stderr
    )
    assertion_diagnostics = _output_assertion_diagnostics(check, stdout, stderr)
    return CheckEvidence(
        name=check.name,
        argv=check.argv,
        cwd=str(logical_cwd),
        timeout_seconds=check.timeout_seconds,
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=duration_ms,
        stdout_excerpt=_bounded_excerpt(stdout),
        stderr_excerpt=_bounded_excerpt(stderr),
        output_artifact_hash=artifact_hash,
        output_artifact_path=artifact_path,
        output_artifact_size=artifact_size,
        executable_sha256=(
            invocation.executable.sha256 if invocation is not None else None
        ),
        environment_attestation_sha256=(
            invocation.environment_attestation.sha256
            if invocation is not None and invocation.environment_attestation is not None
            else None
        ),
        assertions_passed=not assertion_diagnostics,
        assertion_diagnostics=assertion_diagnostics,
    )


def _failed_check_evidence(
    check: TrustedCheck,
    repository_root: Path,
    store: VerificationReceiptStore,
    repository_id: str,
    diagnostic: str,
) -> CheckEvidence:
    stderr = diagnostic.encode("utf-8", errors="replace")
    artifact_hash, artifact_path, artifact_size = store.persist_artifact(
        repository_id, b"", stderr
    )
    return CheckEvidence(
        name=check.name,
        argv=check.argv,
        cwd=str(_unchecked_cwd(repository_root, check.cwd)),
        timeout_seconds=check.timeout_seconds,
        exit_code=None,
        timed_out=False,
        duration_ms=0,
        stdout_excerpt="",
        stderr_excerpt=_bounded_excerpt(stderr),
        output_artifact_hash=artifact_hash,
        output_artifact_path=artifact_path,
        output_artifact_size=artifact_size,
    )


def _attach_snapshot_failure(
    evidence: tuple[CheckEvidence, ...], diagnostic: str
) -> tuple[CheckEvidence, ...]:
    if not evidence:
        return evidence
    last = evidence[-1]
    updated = last.model_copy(
        update={
            "assertions_passed": False,
            "assertion_diagnostics": (*last.assertion_diagnostics, diagnostic),
        }
    )
    return (*evidence[:-1], updated)


def _output_assertion_diagnostics(
    check: TrustedCheck, stdout: bytes, stderr: bytes
) -> tuple[str, ...]:
    output = "\n".join(
        part
        for raw in (stdout, stderr)
        if (part := decode_safe(raw, from_subprocess=True).text)
    )
    diagnostics = list(
        verification_command_output_diagnostics(
            check.argv,
            output,
            custom_runner=check.custom_runner,
            has_test_count_contract=(
                check.test_count_pattern is not None
                and check.minimum_test_count is not None
            ),
        )
    )
    diagnostics.extend(
        output_regex_diagnostics(
            required_patterns=check.required_output_patterns,
            forbidden_patterns=check.forbidden_output_patterns,
            test_count_pattern=check.test_count_pattern,
            minimum_test_count=check.minimum_test_count,
            output=output,
        )
    )
    return tuple(diagnostics)


def _prepare_sandbox_invocation(
    check: TrustedCheck,
    repository_root: Path,
    snapshot: FrozenSourceSnapshot,
    cwd: Path,
) -> _SandboxInvocation:
    backend = resolve_backend("auto")
    if not sys.platform.startswith("linux") or backend.name != "bwrap":
        raise VerificationReceiptError(
            "authority-bearing trusted checks require Linux bubblewrap; "
            f"observed sandbox backend {backend.name!r}"
        )
    writable_parent = snapshot.run_root / "checks"
    writable_parent.mkdir(exist_ok=True)
    writable_root = (writable_parent / f"check-{uuid.uuid4().hex}").resolve()
    writable_root.mkdir(mode=0o700)
    executable: TrustedExecutable | None = None
    try:
        git_common = _git_common_root(repository_root)
        executable_parent = snapshot.run_root / "executables"
        executable_parent.mkdir(mode=0o700, exist_ok=True)
        executable = resolve_trusted_executable(
            check.argv[0],
            forbidden_roots=(repository_root, git_common, snapshot.run_root),
            expected_sha256=check.executable_sha256,
            materialization_root=(executable_parent / f"executable-{uuid.uuid4().hex}"),
        )
        environment_attestation = resolve_environment_attestation(
            check.environment_attestation_path,
            check.environment_attestation_sha256,
            forbidden_roots=(repository_root, git_common, snapshot.run_root),
        )
        env = minimal_check_environment(writable_root)
        prefix, resolved_backend, profile = build_sandbox_command(
            SandboxSpec(
                write_roots=[writable_root],
                read_roots=[snapshot.source_root, *executable.read_roots],
                hidden_roots=_trusted_hidden_roots(repository_root, git_common),
                protected_roots=[snapshot.source_root],
                protect_git_metadata=True,
                allow_network=False,
                env=env,
                cwd=cwd,
            ),
            backend,
        )
        if prefix is None or resolved_backend != "bwrap" or profile is not None:
            raise VerificationReceiptError(
                "trusted check Linux bubblewrap sandbox could not be constructed"
            )
        prefix[-1:-1] = ["--argv0", str(executable.lexical_path)]
        return _SandboxInvocation(
            argv=[*prefix, str(executable.materialized_path), *check.argv[1:]],
            env=env,
            writable_root=writable_root,
            executable=executable,
            environment_attestation=environment_attestation,
        )
    except Exception:
        if executable is not None:
            cleanup_trusted_executable(executable)
        shutil.rmtree(writable_root, ignore_errors=True)
        raise


def _cleanup_sandbox_invocation(invocation: _SandboxInvocation) -> None:
    cleanup_trusted_executable(invocation.executable)
    shutil.rmtree(invocation.writable_root, ignore_errors=True)


def _trusted_hidden_roots(repository_root: Path, git_common: Path) -> list[Path]:
    hidden = strict_read_hidden_roots()
    for candidate in (repository_root.resolve(), git_common.resolve()):
        if any(candidate.is_relative_to(root) for root in hidden):
            continue
        hidden = [root for root in hidden if not root.is_relative_to(candidate)]
        hidden.append(candidate)
    return sorted(hidden)


def _git_common_root(repository_root: Path) -> Path:
    try:
        git = resolve_trusted_system_executable("git")
        result = subprocess.run(
            (
                str(git),
                *TRUSTED_GIT_CONFIG_ARGS,
                "-C",
                str(repository_root),
                "rev-parse",
                "--git-common-dir",
            ),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=minimal_trusted_git_environment(Path("/")),
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            diagnostic = result.stderr.strip() or result.stdout.strip() or "no output"
            raise VerificationReceiptError(
                f"could not resolve repository Git metadata: {diagnostic}"
            )
        common = Path(result.stdout.strip())
        if not common.is_absolute():
            common = repository_root / common
        return common.resolve()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise VerificationReceiptError(
            f"could not resolve repository Git metadata: {exc}"
        ) from exc


def _unchecked_cwd(repository_root: Path, requested: str) -> Path:
    candidate = Path(requested)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repository_root / candidate).resolve()


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


__all__ = ["TrustedCheck", "run_trusted_verification"]
