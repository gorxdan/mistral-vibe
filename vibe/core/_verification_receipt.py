from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum, auto
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Literal

import orjson
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from vibe import __version__
from vibe.core._immutable_store import ImmutableFileStore, ImmutableStoreError
from vibe.core._trusted_command import (
    TRUSTED_GIT_CONFIG_ARGS,
    TrustedCommandError,
    minimal_trusted_git_environment,
    resolve_trusted_system_executable,
)
from vibe.core.paths import VIBE_HOME
from vibe.core.tasking._path_scope import path_matches_scope

if TYPE_CHECKING:
    from git import Repo

RECEIPT_VERSION = 1
OUTPUT_EXCERPT_CHARS = 4_000
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_GIT_TEXT_BYTES = 8 * 1024 * 1024
_MAX_GIT_DIFF_BYTES = 512 * 1024 * 1024
_MAX_REPOSITORY_PATHS = 100_000
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_CONTENT_HASHES = frozenset(
    hash_payload
    for hash_payload in (
        hashlib.sha256(payload).hexdigest() for payload in (b"", b"null", b"{}", b"[]")
    )
)


class VerificationReceiptError(ValueError):
    pass


class ReceiptOutcome(StrEnum):
    PASS = auto()
    FAIL = auto()


class RepositoryState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_identity: str
    worktree_root: str
    base_sha: str
    candidate_head: str
    candidate_tree: str
    branch: str | None
    index_tree: str
    index_diff_hash: str
    worktree_hash: str
    diff_hash: str
    workspace_hash: str
    dirty: bool
    changed_paths: tuple[str, ...]

    @field_validator(
        "repository_identity",
        "index_diff_hash",
        "worktree_hash",
        "diff_hash",
        "workspace_hash",
    )
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if not _HASH_PATTERN.fullmatch(value):
            raise ValueError("expected a lowercase SHA-256 digest")
        return value


class CheckEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str
    timeout_seconds: float = Field(gt=0)
    exit_code: int | None
    timed_out: bool
    duration_ms: int = Field(ge=0)
    stdout_excerpt: str
    stderr_excerpt: str
    output_artifact_hash: str
    output_artifact_path: str
    output_artifact_size: int = Field(ge=0)
    executable_sha256: str | None = None
    environment_attestation_sha256: str | None = None
    assertions_passed: bool = True
    assertion_diagnostics: tuple[str, ...] = ()

    @field_validator("output_artifact_hash")
    @classmethod
    def _validate_artifact_hash(cls, value: str) -> str:
        if not _HASH_PATTERN.fullmatch(value):
            raise ValueError("expected a lowercase SHA-256 digest")
        return value

    @field_validator("executable_sha256", "environment_attestation_sha256")
    @classmethod
    def _validate_optional_hash(cls, value: str | None) -> str | None:
        if value is not None and not _HASH_PATTERN.fullmatch(value):
            raise ValueError("expected a lowercase SHA-256 digest")
        return value

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0 and self.assertions_passed


class VerificationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_version: Literal[1] = RECEIPT_VERSION
    receipt_id: str
    task_brief_hash: str
    recipe_version: str = Field(min_length=1)
    repository: RepositoryState
    contract_hash: str
    configuration_hash: str
    checks_hash: str
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    allowed_paths_passed: bool
    evidence: tuple[CheckEvidence, ...]
    outcome: ReceiptOutcome
    started_at: datetime
    completed_at: datetime
    created_at: datetime
    harness_version: str = Field(min_length=1)

    @field_validator(
        "receipt_id",
        "task_brief_hash",
        "contract_hash",
        "configuration_hash",
        "checks_hash",
    )
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if not _HASH_PATTERN.fullmatch(value):
            raise ValueError("expected a lowercase SHA-256 digest")
        return value

    @field_validator("task_brief_hash", "contract_hash")
    @classmethod
    def _validate_nonempty_content_hash(cls, value: str) -> str:
        if value in _EMPTY_CONTENT_HASHES:
            raise ValueError("task brief and contract must not be empty")
        return value

    @field_validator("allowed_paths")
    @classmethod
    def _validate_allowed_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            _normalize_allowed_pattern(pattern)
        return value

    @field_validator("started_at", "completed_at", "created_at")
    @classmethod
    def _validate_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("receipt timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def _validate_timestamps(self) -> VerificationReceipt:
        if self.completed_at < self.started_at:
            raise ValueError("receipt completion precedes its start")
        if self.created_at < self.completed_at:
            raise ValueError("receipt creation precedes check completion")
        return self

    @model_validator(mode="after")
    def _validate_pass_authority(self) -> VerificationReceipt:
        if self.outcome != ReceiptOutcome.PASS:
            return self
        if not self.evidence:
            raise ValueError("passing receipt must contain trusted check evidence")
        if any(not item.passed for item in self.evidence):
            raise ValueError("passing receipt contains failed trusted check evidence")
        if self.repository.dirty or not self.allowed_paths_passed:
            raise ValueError("passing receipt does not describe an allowed clean tree")
        if self.checks_hash != check_evidence_hash(self.evidence):
            raise ValueError("passing receipt has an inconsistent check-command hash")
        if any(item.executable_sha256 is None for item in self.evidence):
            raise ValueError("passing receipt check is missing executable identity")
        if any(item.environment_attestation_sha256 is None for item in self.evidence):
            raise ValueError("passing receipt check is missing environment attestation")
        return self

    @property
    def passed(self) -> bool:
        return self.outcome == ReceiptOutcome.PASS


@dataclass(frozen=True, slots=True)
class ReceiptValidation:
    receipt_id: str | None
    valid: bool
    reasons: tuple[str, ...]
    receipt: VerificationReceipt | None = None

    def summary(self) -> str:
        if self.valid:
            return f"verification receipt {self.receipt_id} is current"
        return "; ".join(self.reasons) or "verification receipt is invalid"


@dataclass(frozen=True, slots=True)
class _RawRepositoryState:
    base_sha: str
    candidate_head: str
    candidate_tree: str
    branch: str | None
    index_tree: str
    staged_diff_hash: str
    staged_diff_present: bool
    working_diff_hash: str
    working_diff_present: bool
    committed_diff_hash: str
    untracked: tuple[tuple[str, str], ...]
    changed_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GitProbeResult:
    stdout: bytes
    stdout_sha256: str
    stdout_present: bool
    stderr: bytes
    returncode: int


@dataclass(frozen=True, slots=True)
class ReceiptBuildContext:
    task_brief_hash: str
    recipe_version: str
    repository: RepositoryState
    contract_hash: str
    configuration_hash: str
    allowed_paths: tuple[str, ...]
    started_at: datetime
    completed_at: datetime


def hash_payload(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, BaseModel):
        payload = orjson.dumps(
            value.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS
        )
    else:
        payload = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


def validate_binding_hash(value: str, label: str, *, allow_empty: bool = True) -> str:
    _require_hash(value, label)
    if not allow_empty and value in _EMPTY_CONTENT_HASHES:
        raise VerificationReceiptError(f"{label} must not be empty")
    return value


def repository_identity(path: Path | None = None) -> str:
    repo = _open_repo(path)
    if repo.working_tree_dir is None:
        raise VerificationReceiptError("verification requires a non-bare repository")
    root = Path(repo.working_tree_dir)
    roots = sorted(
        _trusted_git(root, "rev-list", "--max-parents=0", "--all").splitlines()
    )
    remote_config = _trusted_git(
        root,
        "config",
        "--local",
        "--no-includes",
        "--get-regexp",
        r"^remote\..*\.url$",
        allowed_exit_codes=(0, 1),
    )
    remote_urls = sorted({
        value
        for line in remote_config.splitlines()
        if (value := line.partition(" ")[2].strip())
    })
    return hash_payload({"root_commits": roots, "remote_urls": remote_urls})


def capture_repository_state(path: Path | None, base_sha: str) -> RepositoryState:
    repo = _open_repo(path)
    if repo.working_tree_dir is None:
        raise VerificationReceiptError("verification requires a non-bare repository")
    root = Path(repo.working_tree_dir).resolve()
    raw = _read_repository_state(repo, base_sha)

    untracked_payload = [
        {"path": name, "blob_hash": blob_hash} for name, blob_hash in raw.untracked
    ]
    index_diff_hash = raw.staged_diff_hash
    worktree_hash = hash_payload({
        "working_diff_hash": raw.working_diff_hash,
        "untracked": untracked_payload,
    })
    diff_hash = hash_payload({
        "committed_diff_hash": raw.committed_diff_hash,
        "staged_diff_hash": raw.staged_diff_hash,
        "working_diff_hash": raw.working_diff_hash,
        "untracked": untracked_payload,
    })
    workspace_hash = hash_payload({
        "head": raw.candidate_head,
        "tree": raw.candidate_tree,
        "index_tree": raw.index_tree,
        "index_diff_hash": index_diff_hash,
        "worktree_hash": worktree_hash,
        "diff_hash": diff_hash,
    })
    return RepositoryState(
        repository_identity=repository_identity(root),
        worktree_root=str(root),
        base_sha=raw.base_sha,
        candidate_head=raw.candidate_head,
        candidate_tree=raw.candidate_tree,
        branch=raw.branch,
        index_tree=raw.index_tree,
        index_diff_hash=index_diff_hash,
        worktree_hash=worktree_hash,
        diff_hash=diff_hash,
        workspace_hash=workspace_hash,
        dirty=bool(
            raw.staged_diff_present or raw.working_diff_present or raw.untracked
        ),
        changed_paths=raw.changed_paths,
    )


def allowed_paths_match(changed_paths: Sequence[str], patterns: Sequence[str]) -> bool:
    if not changed_paths or not patterns:
        return False
    normalized = tuple(_normalize_allowed_pattern(pattern) for pattern in patterns)
    if any(
        PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
        for path in changed_paths
    ):
        return False
    return all(
        any(path_matches_scope(path, pattern) for pattern in normalized)
        for path in changed_paths
    )


def check_evidence_hash(evidence: Sequence[CheckEvidence]) -> str:
    return hash_payload([
        {
            "name": item.name,
            "argv": item.argv,
            "cwd": item.cwd,
            "timeout_seconds": item.timeout_seconds,
        }
        for item in evidence
    ])


def build_receipt(
    *,
    context: ReceiptBuildContext,
    checks_hash: str,
    allowed_paths_passed: bool,
    evidence: Sequence[CheckEvidence],
    outcome: ReceiptOutcome,
) -> VerificationReceipt:
    created_at = datetime.now(UTC)
    draft = VerificationReceipt(
        receipt_id="0" * 64,
        task_brief_hash=context.task_brief_hash,
        recipe_version=context.recipe_version,
        repository=context.repository,
        contract_hash=context.contract_hash,
        configuration_hash=context.configuration_hash,
        checks_hash=checks_hash,
        allowed_paths=context.allowed_paths,
        allowed_paths_passed=allowed_paths_passed,
        evidence=tuple(evidence),
        outcome=outcome,
        started_at=context.started_at,
        completed_at=context.completed_at,
        created_at=created_at,
        harness_version=__version__,
    )
    return draft.model_copy(update={"receipt_id": receipt_content_hash(draft)})


def receipt_content_hash(receipt: VerificationReceipt) -> str:
    return hash_payload(receipt.model_dump(mode="json", exclude={"receipt_id"}))


class VerificationReceiptStore:
    def __init__(self, root: Path | None = None) -> None:
        requested = (root or VIBE_HOME.path / "verification").expanduser()
        self.root = requested if requested.is_absolute() else requested.absolute()
        try:
            self._files = ImmutableFileStore(self.root)
        except ImmutableStoreError as exc:
            raise VerificationReceiptError(str(exc)) from exc

    def persist_artifact(
        self, repository_id: str, stdout: bytes, stderr: bytes
    ) -> tuple[str, str, int]:
        _require_hash(repository_id, "repository ID")
        payload = orjson.dumps(
            {
                "version": 1,
                "stdout_base64": base64.b64encode(stdout).decode("ascii"),
                "stderr_base64": base64.b64encode(stderr).decode("ascii"),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(payload) > _MAX_ARTIFACT_BYTES:
            raise VerificationReceiptError(
                "verification output artifact exceeds the store size limit"
            )
        digest = hashlib.sha256(payload).hexdigest()
        relative = PurePosixPath("artifacts") / repository_id / f"{digest}.json"
        self._persist_immutable(relative, payload)
        return digest, str(relative), len(payload)

    def persist_receipt(self, receipt: VerificationReceipt) -> Path:
        if receipt.receipt_id != receipt_content_hash(receipt):
            raise VerificationReceiptError("receipt content hash does not match its ID")
        relative = self._receipt_relative(
            receipt.repository.repository_identity, receipt.receipt_id
        )
        payload = (receipt.model_dump_json(indent=2) + "\n").encode("utf-8")
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise VerificationReceiptError(
                "verification receipt exceeds the store size limit"
            )
        self._persist_immutable(relative, payload)
        return self.root.joinpath(*relative.parts)

    def receipt_path(self, repository_id: str, receipt_id: str) -> Path:
        relative = self._receipt_relative(repository_id, receipt_id)
        return self.root.joinpath(*relative.parts)

    @staticmethod
    def _receipt_relative(repository_id: str, receipt_id: str) -> PurePosixPath:
        _require_hash(repository_id, "repository ID")
        _require_hash(receipt_id, "receipt ID")
        return PurePosixPath("receipts") / repository_id / f"{receipt_id}.json"

    def load(self, repository_id: str, receipt_id: str) -> VerificationReceipt:
        relative = self._receipt_relative(repository_id, receipt_id)
        return self._load_path(relative, receipt_id)

    def load_any(self, receipt_id: str) -> VerificationReceipt:
        _require_hash(receipt_id, "receipt ID")
        try:
            repository_ids = self._files.list_directory(PurePosixPath("receipts"))
        except FileNotFoundError:
            raise VerificationReceiptError(
                f"verification receipt {receipt_id} was not found"
            ) from None
        except (ImmutableStoreError, OSError) as exc:
            raise VerificationReceiptError(
                f"verification receipt store could not be inspected: {exc}"
            ) from exc
        matches: list[PurePosixPath] = []
        for repository_id in repository_ids:
            if _HASH_PATTERN.fullmatch(repository_id) is None:
                continue
            relative = self._receipt_relative(repository_id, receipt_id)
            try:
                self._files.read(relative, max_bytes=_MAX_RECEIPT_BYTES)
            except FileNotFoundError:
                continue
            except (ImmutableStoreError, OSError) as exc:
                raise VerificationReceiptError(
                    f"verification receipt {receipt_id} could not be read: {exc}"
                ) from exc
            matches.append(relative)
        if len(matches) != 1:
            raise VerificationReceiptError(
                f"expected one stored verification receipt {receipt_id}, found {len(matches)}"
            )
        return self._load_path(matches[0], receipt_id)

    def validate_artifact(self, evidence: CheckEvidence) -> str | None:
        try:
            self._validate_artifact(evidence)
        except VerificationReceiptError as exc:
            return str(exc)
        return None

    def _validate_artifact(self, evidence: CheckEvidence) -> None:
        relative = PurePosixPath(evidence.output_artifact_path)
        expected_name = f"{evidence.output_artifact_hash}.json"
        if not _valid_artifact_path(relative, expected_name):
            raise VerificationReceiptError(
                "output artifact path escapes the receipt store"
            )
        try:
            payload = self._files.read(relative, max_bytes=_MAX_ARTIFACT_BYTES)
        except FileNotFoundError as exc:
            raise VerificationReceiptError(
                f"output artifact is missing: {evidence.output_artifact_path}"
            ) from exc
        except (ImmutableStoreError, OSError) as exc:
            raise VerificationReceiptError(
                f"output artifact could not be read: {exc}"
            ) from exc
        if len(payload) != evidence.output_artifact_size:
            raise VerificationReceiptError(
                f"output artifact size changed: {evidence.output_artifact_path}"
            )
        if hashlib.sha256(payload).hexdigest() != evidence.output_artifact_hash:
            raise VerificationReceiptError(
                f"output artifact hash changed: {evidence.output_artifact_path}"
            )
        try:
            parsed = orjson.loads(payload)
            if set(parsed) != {"version", "stdout_base64", "stderr_base64"}:
                raise ValueError("unexpected artifact fields")
            if parsed["version"] != 1:
                raise ValueError("unsupported artifact version")
            base64.b64decode(parsed["stdout_base64"], validate=True)
            base64.b64decode(parsed["stderr_base64"], validate=True)
        except (KeyError, TypeError, ValueError, orjson.JSONDecodeError) as exc:
            raise VerificationReceiptError(
                f"output artifact is malformed: {evidence.output_artifact_path}"
            ) from exc

    def _load_path(
        self, relative: PurePosixPath, receipt_id: str
    ) -> VerificationReceipt:
        try:
            payload = self._files.read(relative, max_bytes=_MAX_RECEIPT_BYTES)
        except FileNotFoundError as exc:
            raise VerificationReceiptError(
                f"verification receipt {receipt_id} was not found"
            ) from exc
        except (ImmutableStoreError, OSError) as exc:
            raise VerificationReceiptError(
                f"verification receipt {receipt_id} could not be read: {exc}"
            ) from exc
        try:
            receipt = VerificationReceipt.model_validate_json(payload)
        except ValidationError as exc:
            raise VerificationReceiptError(
                f"verification receipt {receipt_id} is malformed: {exc}"
            ) from exc
        if (
            receipt.receipt_id != receipt_id
            or receipt_content_hash(receipt) != receipt_id
        ):
            raise VerificationReceiptError(
                f"verification receipt {receipt_id} failed its content hash"
            )
        return receipt

    def _persist_immutable(self, relative: PurePosixPath, payload: bytes) -> None:
        try:
            self._files.write(relative, payload)
        except (ImmutableStoreError, OSError) as exc:
            raise VerificationReceiptError(
                f"immutable receipt artifact could not be persisted: {exc}"
            ) from exc


def validate_receipt(
    receipt: VerificationReceipt,
    *,
    store: VerificationReceiptStore,
    repository_path: Path,
    expected_base_sha: str,
    expected_candidate_head: str | None = None,
    expected_task_brief_hash: str | None = None,
    expected_contract_hash: str | None = None,
    expected_configuration_hash: str | None = None,
    expected_checks_hash: str | None = None,
    expected_recipe_version: str | None = None,
) -> ReceiptValidation:
    reasons = _receipt_structure_errors(receipt)
    reasons.extend(
        _binding_errors(
            receipt,
            expected_task_brief_hash=expected_task_brief_hash,
            expected_contract_hash=expected_contract_hash,
            expected_configuration_hash=expected_configuration_hash,
            expected_checks_hash=expected_checks_hash,
            expected_recipe_version=expected_recipe_version,
        )
    )
    reasons.extend(
        _repository_errors(
            receipt,
            repository_path=repository_path,
            expected_base_sha=expected_base_sha,
            expected_candidate_head=expected_candidate_head,
        )
    )
    reasons.extend(
        artifact_error
        for evidence in receipt.evidence
        if (artifact_error := store.validate_artifact(evidence)) is not None
    )
    return ReceiptValidation(
        receipt_id=receipt.receipt_id,
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        receipt=receipt,
    )


def _receipt_structure_errors(receipt: VerificationReceipt) -> list[str]:
    reasons: list[str] = []
    if receipt.receipt_id != receipt_content_hash(receipt):
        reasons.append("receipt content hash does not match its ID")
    if not receipt.passed:
        reasons.append("receipt outcome is not PASS")
    if receipt.repository.dirty:
        reasons.append("receipt candidate tree is dirty")
    if not receipt.evidence:
        reasons.append("receipt contains no trusted checks")
    elif any(not evidence.passed for evidence in receipt.evidence):
        reasons.append("receipt contains a failed or timed-out check")
    elif any(evidence.executable_sha256 is None for evidence in receipt.evidence):
        reasons.append("receipt check is missing its executable identity")
    elif any(
        evidence.environment_attestation_sha256 is None for evidence in receipt.evidence
    ):
        reasons.append("receipt check is missing its environment attestation")
    if receipt.checks_hash != check_evidence_hash(receipt.evidence):
        reasons.append("receipt check-command hash is inconsistent")
    if not receipt.allowed_paths_passed or not allowed_paths_match(
        receipt.repository.changed_paths, receipt.allowed_paths
    ):
        reasons.append("receipt allowed-path check did not pass")
    return reasons


def _binding_errors(
    receipt: VerificationReceipt,
    *,
    expected_task_brief_hash: str | None,
    expected_contract_hash: str | None,
    expected_configuration_hash: str | None,
    expected_checks_hash: str | None,
    expected_recipe_version: str | None,
) -> list[str]:
    reasons: list[str] = []
    comparisons = (
        (expected_task_brief_hash, receipt.task_brief_hash, "task brief"),
        (expected_contract_hash, receipt.contract_hash, "contract"),
        (
            expected_configuration_hash,
            receipt.configuration_hash,
            "verification configuration",
        ),
        (expected_checks_hash, receipt.checks_hash, "check commands"),
        (expected_recipe_version, receipt.recipe_version, "recipe version"),
    )
    for expected, actual, label in comparisons:
        if expected is not None and expected != actual:
            reasons.append(f"{label} changed after verification")
    return reasons


def _repository_errors(
    receipt: VerificationReceipt,
    *,
    repository_path: Path,
    expected_base_sha: str,
    expected_candidate_head: str | None,
) -> list[str]:
    reasons: list[str] = []
    try:
        current = capture_repository_state(repository_path, expected_base_sha)
    except VerificationReceiptError as exc:
        reasons.append(str(exc))
    else:
        if receipt.repository.base_sha != current.base_sha:
            reasons.append("base commit changed after verification")
        if receipt.repository != current:
            reasons.append("candidate repository state changed after verification")
        if (
            expected_candidate_head is not None
            and current.candidate_head != expected_candidate_head
        ):
            reasons.append("receipt does not describe the candidate branch HEAD")
    return reasons


def validate_receipt_id(
    receipt_id: str,
    *,
    store: VerificationReceiptStore,
    repository_path: Path,
    expected_base_sha: str,
    expected_candidate_head: str | None = None,
    expected_task_brief_hash: str | None = None,
    expected_contract_hash: str | None = None,
    expected_configuration_hash: str | None = None,
    expected_checks_hash: str | None = None,
    expected_recipe_version: str | None = None,
) -> ReceiptValidation:
    try:
        receipt = store.load_any(receipt_id)
    except VerificationReceiptError as exc:
        return ReceiptValidation(
            receipt_id=receipt_id, valid=False, reasons=(str(exc),)
        )
    return validate_receipt(
        receipt,
        store=store,
        repository_path=repository_path,
        expected_base_sha=expected_base_sha,
        expected_candidate_head=expected_candidate_head,
        expected_task_brief_hash=expected_task_brief_hash,
        expected_contract_hash=expected_contract_hash,
        expected_configuration_hash=expected_configuration_hash,
        expected_checks_hash=expected_checks_hash,
        expected_recipe_version=expected_recipe_version,
    )


def _open_repo(path: Path | None) -> Repo:
    from git import Repo
    from git.exc import GitError

    try:
        requested = (path or Path.cwd()).expanduser().resolve()
        root = Path(_trusted_git(requested, "rev-parse", "--show-toplevel")).resolve()
        repo = Repo(root, search_parent_directories=False)
    except (GitError, OSError, ValueError) as exc:
        raise VerificationReceiptError(f"could not open repository: {exc}") from exc
    if repo.working_tree_dir is None:
        raise VerificationReceiptError("verification requires a non-bare repository")
    return repo


def _read_repository_state(repo: Repo, base_sha: str) -> _RawRepositoryState:
    from git.exc import GitError

    try:
        if repo.working_tree_dir is None:
            raise VerificationReceiptError(
                "verification requires a non-bare repository"
            )
        root = Path(repo.working_tree_dir)
        base = _trusted_git(root, "rev-parse", f"{base_sha}^{{commit}}")
        candidate_head = _trusted_git(root, "rev-parse", "HEAD")
        candidate_tree = _trusted_git(root, "rev-parse", "HEAD^{tree}")
        diff_guards = ("--no-ext-diff", "--no-textconv")
        staged_diff_hash, staged_diff_present = _trusted_git_digest(
            root, "diff", *diff_guards, "--binary", "--cached", "HEAD", "--"
        )
        working_diff_hash, working_diff_present = _trusted_git_digest(
            root, "diff", *diff_guards, "--binary", "HEAD", "--"
        )
        committed_diff_hash, _ = _trusted_git_digest(
            root, "diff", *diff_guards, "--binary", base, candidate_head, "--"
        )
        index_tree = _trusted_git(root, "write-tree")
        untracked_names = _trusted_git(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).split("\0")
        if len(untracked_names) > _MAX_REPOSITORY_PATHS:
            raise VerificationReceiptError(
                "repository contains too many untracked paths to verify safely"
            )
        untracked = tuple(
            (name, _trusted_git(root, "hash-object", "--no-filters", "--", name))
            for name in sorted(name for name in untracked_names if name)
        )
        branch = (
            _trusted_git(
                root,
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
                allowed_exit_codes=(0, 1),
            )
            or None
        )
    except (GitError, OSError, ValueError, VerificationReceiptError) as exc:
        raise VerificationReceiptError(
            f"could not inspect repository state: {exc}"
        ) from exc
    return _RawRepositoryState(
        base_sha=base,
        candidate_head=candidate_head,
        candidate_tree=candidate_tree,
        branch=branch,
        index_tree=index_tree,
        staged_diff_hash=staged_diff_hash,
        staged_diff_present=staged_diff_present,
        working_diff_hash=working_diff_hash,
        working_diff_present=working_diff_present,
        committed_diff_hash=committed_diff_hash,
        untracked=untracked,
        changed_paths=_changed_paths(root, base, candidate_head, untracked),
    )


def _changed_paths(
    root: Path, base_sha: str, candidate_sha: str, untracked: Sequence[tuple[str, str]]
) -> tuple[str, ...]:
    guards = ("--no-ext-diff", "--no-textconv", "--no-renames", "--name-only")
    outputs = (
        _trusted_git(root, "diff", *guards, base_sha, candidate_sha, "--"),
        _trusted_git(root, "diff", *guards, "--cached", "HEAD", "--"),
        _trusted_git(root, "diff", *guards, "HEAD", "--"),
    )
    paths = {line for output in outputs for line in output.splitlines() if line}
    paths.update(name for name, _ in untracked)
    if len(paths) > _MAX_REPOSITORY_PATHS:
        raise VerificationReceiptError(
            "repository contains too many changed paths to verify safely"
        )
    return tuple(sorted(paths))


def _trusted_git_environment() -> dict[str, str]:
    return minimal_trusted_git_environment(Path("/"))


def _trusted_git(
    root: Path, *arguments: str, allowed_exit_codes: tuple[int, ...] = (0,)
) -> str:
    result = _run_trusted_git(root, arguments, capture_stdout=True)
    if result.returncode not in allowed_exit_codes:
        diagnostic = _decode_git_diagnostic(result.stderr, result.stdout)
        raise VerificationReceiptError(
            f"trusted Git probe failed ({' '.join(arguments)}): {diagnostic}"
        )
    return result.stdout.decode("utf-8", errors="surrogateescape").strip()


def _trusted_git_digest(root: Path, *arguments: str) -> tuple[str, bool]:
    result = _run_trusted_git(root, arguments, capture_stdout=False)
    if result.returncode != 0:
        diagnostic = _decode_git_diagnostic(result.stderr, result.stdout)
        raise VerificationReceiptError(
            f"trusted Git probe failed ({' '.join(arguments)}): {diagnostic}"
        )
    return result.stdout_sha256, result.stdout_present


def _run_trusted_git(
    root: Path, arguments: tuple[str, ...], *, capture_stdout: bool
) -> _GitProbeResult:
    try:
        git = resolve_trusted_system_executable("git")
        with (
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            result = subprocess.run(
                [str(git), *TRUSTED_GIT_CONFIG_ARGS, "-C", str(root), *arguments],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                env=_trusted_git_environment(),
                text=False,
                timeout=30,
            )
            stdout_size = os.fstat(stdout_file.fileno()).st_size
            output_limit = (
                _MAX_GIT_TEXT_BYTES if capture_stdout else _MAX_GIT_DIFF_BYTES
            )
            if stdout_size > output_limit:
                raise VerificationReceiptError(
                    f"trusted Git probe output exceeded the {output_limit}-byte limit"
                )
            stdout_file.seek(0)
            stderr_file.seek(0)
            digest = hashlib.sha256()
            while chunk := stdout_file.read(64 * 1024):
                digest.update(chunk)
            stdout_file.seek(0)
            stdout = stdout_file.read() if capture_stdout else b""
            stderr = stderr_file.read(_MAX_GIT_TEXT_BYTES + 1)
    except (OSError, subprocess.SubprocessError, TrustedCommandError) as exc:
        raise VerificationReceiptError(f"trusted Git probe failed: {exc}") from exc
    if len(stderr) > _MAX_GIT_TEXT_BYTES:
        raise VerificationReceiptError(
            "trusted Git diagnostic exceeded the bounded text limit"
        )
    return _GitProbeResult(
        stdout=stdout,
        stdout_sha256=digest.hexdigest(),
        stdout_present=stdout_size > 0,
        stderr=stderr,
        returncode=result.returncode,
    )


def _decode_git_diagnostic(stderr: bytes, stdout: bytes) -> str:
    raw = stderr or stdout[:_MAX_GIT_TEXT_BYTES]
    return raw.decode("utf-8", errors="replace").strip() or "no output"


def _normalize_allowed_pattern(pattern: str) -> str:
    normalized = pattern.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise VerificationReceiptError(f"invalid allowed-path pattern: {pattern!r}")
    return normalized


def _valid_artifact_path(relative: PurePosixPath, expected_name: str) -> bool:
    artifact_path_parts = 3
    if relative.is_absolute() or ".." in relative.parts:
        return False
    if len(relative.parts) != artifact_path_parts:
        return False
    return (
        relative.parts[0] == "artifacts"
        and _HASH_PATTERN.fullmatch(relative.parts[1]) is not None
        and relative.name == expected_name
    )


def _require_hash(value: str, label: str) -> None:
    if not _HASH_PATTERN.fullmatch(value):
        raise VerificationReceiptError(f"invalid {label}: expected a SHA-256 digest")


__all__ = [
    "OUTPUT_EXCERPT_CHARS",
    "RECEIPT_VERSION",
    "CheckEvidence",
    "ReceiptBuildContext",
    "ReceiptOutcome",
    "ReceiptValidation",
    "RepositoryState",
    "VerificationReceipt",
    "VerificationReceiptError",
    "VerificationReceiptStore",
    "allowed_paths_match",
    "build_receipt",
    "capture_repository_state",
    "check_evidence_hash",
    "hash_payload",
    "receipt_content_hash",
    "repository_identity",
    "validate_binding_hash",
    "validate_receipt",
    "validate_receipt_id",
]
