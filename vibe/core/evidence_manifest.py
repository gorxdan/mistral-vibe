from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from vibe.core._trusted_command import validate_trusted_command_argv

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_SHA_PATTERN = r"^[0-9a-f]{40}$"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_BYTES = 1024 * 1024 * 1024
_MAX_ARTIFACTS = 10_000
_MAX_TREE_ENTRIES = 20_000
_MAX_TREE_DEPTH = 16
_MAX_RESULT_SCHEMA_BYTES = 64 * 1024
_MAX_RESULT_BYTES = 8 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_LOCK_TIMEOUT_SECONDS = 2.0
_LOCK_RETRY_SECONDS = 0.02
_ARTIFACT_TYPE = re.compile(r"^[a-z][a-z0-9_-]*$")


class EvidenceManifestError(ValueError):
    pass


class _Environment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    uv_lock_sha256: str = Field(pattern=_HASH_PATTERN)
    runner: str = Field(min_length=1)


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HASH_PATTERN)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if _ARTIFACT_TYPE.fullmatch(value) is None:
            raise ValueError("artifact type must be a lowercase identifier")
        return value

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("artifact path must be a normalized relative POSIX path")
        return value


class _RecordedEnvironmentContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: Literal["exact"]
    values: dict[str, str]

    @field_validator("values")
    @classmethod
    def _validate_values(cls, value: dict[str, str]) -> dict[str, str]:
        for name, recorded_value in value.items():
            if not name or "\0" in name or "=" in name:
                raise ValueError("recorded environment names must be nonempty names")
            if "\0" in recorded_value:
                raise ValueError("recorded environment values must not contain NUL")
        return value


class EvidenceScenarioContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    surface: Literal["ui", "non_ui", "mixed"]
    command: tuple[str, ...] = Field(min_length=1)
    recorded_environment: _RecordedEnvironmentContract
    required_artifact_types: tuple[str, ...] = Field(min_length=1)
    result_schema: dict[str, JsonValue]
    expected_status: Literal["pass", "fail"]
    allowed_notes: tuple[str, ...]
    allowed_gap_notes: tuple[str, ...]

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_scenario_id(value)

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_direct_command(value)

    @field_validator("required_artifact_types")
    @classmethod
    def _validate_required_artifact_types(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if any(_ARTIFACT_TYPE.fullmatch(item) is None for item in value):
            raise ValueError("required artifact types must be lowercase identifiers")
        if value != tuple(sorted(set(value))):
            raise ValueError("required artifact types must be sorted and unique")
        if "result" not in value:
            raise ValueError("required artifact types must include result")
        return value

    @field_validator("result_schema")
    @classmethod
    def _validate_result_schema(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        if len(encoded) > _MAX_RESULT_SCHEMA_BYTES:
            raise ValueError("result schema exceeds the size limit")
        if _contains_schema_reference(value):
            raise ValueError("result schema cannot contain references")
        try:
            Draft202012Validator.check_schema(value)
        except SchemaError as exc:
            raise ValueError(f"invalid result schema: {exc.message}") from exc
        return value

    @field_validator("allowed_notes", "allowed_gap_notes")
    @classmethod
    def _validate_notes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not note or "\0" in note for note in value):
            raise ValueError("allowed evidence notes must be nonempty")
        return value

    @model_validator(mode="after")
    def _validate_pass_contract(self) -> EvidenceScenarioContract:
        if self.expected_status == "pass" and (
            self.allowed_notes or self.allowed_gap_notes
        ):
            raise ValueError("passing scenarios cannot authorize failure or gap notes")
        if self.expected_status == "fail" and not (
            self.allowed_notes or self.allowed_gap_notes
        ):
            raise ValueError("failing scenarios must authorize an exact failure note")
        return self


class _Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    surface: Literal["ui", "non_ui", "mixed"]
    status: Literal["pass", "fail"]
    command: tuple[str, ...] = Field(min_length=1)
    recorded_environment: dict[str, str]
    exit_code: int | None
    artifacts: tuple[_Artifact, ...]
    metrics: dict[str, JsonValue]
    notes: tuple[str, ...]
    started_at: AwareDatetime
    finished_at: AwareDatetime
    result_path: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_scenario_id(value)

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _validate_direct_command(value)

    @field_validator("result_path")
    @classmethod
    def _validate_result_path(cls, value: str) -> str:
        return _Artifact._validate_path(value)

    @model_validator(mode="after")
    def _validate_scenario_identity(self) -> _Scenario:
        if self.finished_at < self.started_at:
            raise ValueError("scenario finish time precedes its start time")
        if self.status == "pass" and self.exit_code != 0:
            raise ValueError("passing scenario must have exit_code 0")
        paths = [artifact.path for artifact in self.artifacts]
        if self.result_path not in paths:
            raise ValueError("scenario result_path must name a declared artifact")
        identities = [(artifact.type, artifact.path) for artifact in self.artifacts]
        if identities != sorted(identities):
            raise ValueError("scenario artifacts must be sorted by type and path")
        return self


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    baseline_sha: str = Field(pattern=_SHA_PATTERN)
    candidate_sha: str = Field(pattern=_SHA_PATTERN)
    upstream_sha: str = Field(pattern=_SHA_PATTERN)
    environment: _Environment
    scenarios: tuple[_Scenario, ...] = Field(min_length=1)


def _validate_scenario_id(value: str) -> str:
    if value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise ValueError("scenario ID must be one path segment")
    return value


def _validate_direct_command(value: tuple[str, ...]) -> tuple[str, ...]:
    if any(not argument or "\0" in argument for argument in value):
        raise ValueError("scenario command entries must be nonempty")
    validate_trusted_command_argv(value)
    return value


def _contains_schema_reference(value: JsonValue) -> bool:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            if "$ref" in current or "$dynamicRef" in current:
                return True
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return False


@dataclass(frozen=True, slots=True)
class EvidenceManifestSnapshot:
    manifest_path: Path
    manifest_sha256: str
    scenario_ids: tuple[str, ...]
    artifact_count: int
    artifact_bytes: int
    tree_identity: tuple[
        tuple[
            str, Literal["directory", "file"], tuple[int, int, int, int, int, int, int]
        ],
        ...,
    ]


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    kind: Literal["directory", "file"]
    identity: tuple[int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _TreeInventory:
    entries: dict[str, _TreeEntry]

    @property
    def files(self) -> set[str]:
        return {path for path, entry in self.entries.items() if entry.kind == "file"}

    @property
    def directories(self) -> set[str]:
        return {
            path
            for path, entry in self.entries.items()
            if entry.kind == "directory" and path != "."
        }


def validate_evidence_manifest(
    evidence_workspace: Path,
    *,
    run_id: str,
    runner_id: str,
    baseline_sha: str,
    candidate_sha: str,
    upstream_sha: str,
    expected_uv_lock_sha256: str,
    expected_manifest_sha256: str,
    expected_scenarios: tuple[EvidenceScenarioContract, ...],
) -> EvidenceManifestSnapshot:
    if not hasattr(os, "O_NOFOLLOW"):
        raise EvidenceManifestError(
            "evidence validation requires descriptor-safe no-follow support"
        )
    relative_root = (".ai", "runs", run_id, "test-evidence", "latest")
    manifest_path = evidence_workspace.joinpath(*relative_root, "manifest.json")
    try:
        workspace_fd = _open_directory_path(evidence_workspace)
        try:
            root_fd = _open_directory_chain(workspace_fd, relative_root)
        finally:
            os.close(workspace_fd)
        try:
            with _manifest_lock(root_fd) as lock_identity:
                before_inventory = _inventory_tree(root_fd)
                _require_inventory_identity(
                    before_inventory, ".manifest.lock", "file", lock_identity
                )
                _validate_root_inventory(before_inventory, expected_scenarios)
                manifest_bytes, _ = _read_regular_file(
                    root_fd,
                    ("manifest.json",),
                    max_bytes=_MAX_MANIFEST_BYTES,
                    inventory=before_inventory,
                )
                observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                if observed_manifest_sha256 != expected_manifest_sha256:
                    raise EvidenceManifestError(
                        "evidence manifest digest mismatch: expected "
                        f"{expected_manifest_sha256}, observed "
                        f"{observed_manifest_sha256}"
                    )
                manifest = _parse_manifest(manifest_bytes)
                _validate_identity(
                    manifest,
                    runner_id=runner_id,
                    baseline_sha=baseline_sha,
                    candidate_sha=candidate_sha,
                    upstream_sha=upstream_sha,
                    expected_uv_lock_sha256=expected_uv_lock_sha256,
                    expected_scenarios=expected_scenarios,
                )
                artifact_count, artifact_bytes = _validate_artifacts(
                    root_fd,
                    manifest,
                    expected_scenarios=expected_scenarios,
                    inventory=before_inventory,
                )
                after_inventory = _inventory_tree(root_fd)
                if after_inventory != before_inventory:
                    raise EvidenceManifestError(
                        "evidence tree changed while it was validated"
                    )
                _require_directory_path_identity(manifest_path.parent, root_fd)
        finally:
            os.close(root_fd)
    except EvidenceManifestError:
        raise
    except OSError as exc:
        raise EvidenceManifestError(
            f"could not inspect evidence manifest {manifest_path}: {exc}"
        ) from exc

    return EvidenceManifestSnapshot(
        manifest_path=manifest_path,
        manifest_sha256=observed_manifest_sha256,
        scenario_ids=tuple(scenario.id for scenario in manifest.scenarios),
        artifact_count=artifact_count,
        artifact_bytes=artifact_bytes,
        tree_identity=_frozen_tree_identity(after_inventory),
    )


def revalidate_evidence_snapshot(snapshot: EvidenceManifestSnapshot) -> None:
    try:
        root_fd = _open_directory_path(snapshot.manifest_path.parent)
        try:
            with _manifest_lock(root_fd) as lock_identity:
                inventory = _inventory_tree(root_fd)
                _require_inventory_identity(
                    inventory, ".manifest.lock", "file", lock_identity
                )
                observed = _frozen_tree_identity(inventory)
                _require_directory_path_identity(snapshot.manifest_path.parent, root_fd)
        finally:
            os.close(root_fd)
    except EvidenceManifestError:
        raise
    except OSError as exc:
        raise EvidenceManifestError(
            f"could not revalidate evidence tree {snapshot.manifest_path.parent}: {exc}"
        ) from exc
    if observed != snapshot.tree_identity:
        raise EvidenceManifestError(
            "evidence tree identity changed after authority validation"
        )


def _frozen_tree_identity(
    inventory: _TreeInventory,
) -> tuple[
    tuple[str, Literal["directory", "file"], tuple[int, int, int, int, int, int, int]],
    ...,
]:
    return tuple(
        (path, entry.kind, entry.identity)
        for path, entry in sorted(inventory.entries.items())
    )


def _open_directory_path(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = absolute.absolute()
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _require_directory_path_identity(path: Path, expected_fd: int) -> None:
    current_fd = _open_directory_path(path)
    try:
        expected = os.fstat(expected_fd)
        current = os.fstat(current_fd)
        if (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
            raise EvidenceManifestError(
                "evidence directory ancestry changed during validation"
            )
    finally:
        os.close(current_fd)


def _open_directory_chain(parent_fd: int, parts: tuple[str, ...]) -> int:
    current_fd = os.dup(parent_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
    except BaseException:
        os.close(current_fd)
        raise
    return current_fd


@contextmanager
def _bound_directory_chain(
    root_fd: int, parts: tuple[str, ...], inventory: _TreeInventory
) -> Iterator[int]:
    descriptors = [os.dup(root_fd)]
    try:
        _require_descriptor_identity(
            descriptors[0], _require_inventory_entry(inventory, ".", "directory"), "."
        )
        for index, part in enumerate(parts):
            relative = "/".join(parts[: index + 1])
            expected = _require_inventory_entry(inventory, relative, "directory")
            parent_fd = descriptors[-1]
            _require_named_identity(parent_fd, part, "directory", expected.identity)
            child_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            descriptors.append(child_fd)
            _require_descriptor_identity(child_fd, expected, relative)
            _require_named_identity(parent_fd, part, "directory", expected.identity)
        try:
            yield descriptors[-1]
        finally:
            _recheck_directory_chain(descriptors, parts, inventory)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _recheck_directory_chain(
    descriptors: list[int], parts: tuple[str, ...], inventory: _TreeInventory
) -> None:
    _require_descriptor_identity(
        descriptors[0], _require_inventory_entry(inventory, ".", "directory"), "."
    )
    for index, part in enumerate(parts):
        relative = "/".join(parts[: index + 1])
        expected = _require_inventory_entry(inventory, relative, "directory")
        _require_descriptor_identity(
            descriptors[index],
            _require_inventory_entry(
                inventory, "/".join(parts[:index]) or ".", "directory"
            ),
            "/".join(parts[:index]) or ".",
        )
        _require_named_identity(
            descriptors[index], part, "directory", expected.identity
        )
        _require_descriptor_identity(descriptors[index + 1], expected, relative)


def _require_inventory_entry(
    inventory: _TreeInventory, path: str, kind: Literal["directory", "file"]
) -> _TreeEntry:
    entry = inventory.entries.get(path)
    if entry is None or entry.kind != kind:
        raise EvidenceManifestError(
            f"evidence inventory is missing expected {kind}: {path}"
        )
    return entry


def _require_inventory_identity(
    inventory: _TreeInventory,
    path: str,
    kind: Literal["directory", "file"],
    identity: tuple[int, int, int, int, int, int, int],
) -> None:
    if _require_inventory_entry(inventory, path, kind).identity != identity:
        raise EvidenceManifestError(
            f"evidence {path} does not match its locked identity"
        )


def _require_descriptor_identity(
    descriptor: int, expected: _TreeEntry, display_path: str
) -> None:
    if _tree_identity(os.fstat(descriptor)) != expected.identity:
        raise EvidenceManifestError(
            f"evidence directory identity does not match frozen inventory: {display_path}"
        )


def _require_named_identity(
    parent_fd: int,
    name: str,
    kind: Literal["directory", "file"],
    identity: tuple[int, int, int, int, int, int, int],
) -> None:
    observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    is_expected_kind = (
        stat.S_ISDIR(observed.st_mode)
        if kind == "directory"
        else stat.S_ISREG(observed.st_mode)
    )
    if not is_expected_kind or _tree_identity(observed) != identity:
        raise EvidenceManifestError(
            f"evidence {kind} path changed during validation: {name}"
        )


@contextmanager
def _manifest_lock(root_fd: int) -> Iterator[tuple[int, int, int, int, int, int, int]]:
    try:
        import fcntl
    except ImportError as exc:
        raise EvidenceManifestError(
            "evidence validation requires POSIX manifest locking"
        ) from exc

    root_identity = _tree_identity(os.fstat(root_fd))
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(".manifest.lock", flags, 0o600, dir_fd=root_fd)
    except OSError as exc:
        raise EvidenceManifestError(
            f"could not open the evidence manifest lock: {exc}"
        ) from exc
    try:
        opened = _require_regular_single_link(
            descriptor, ".manifest.lock", _MAX_MANIFEST_BYTES
        )
        lock_identity = _tree_identity(opened)
        _require_named_identity(root_fd, ".manifest.lock", "file", lock_identity)
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise EvidenceManifestError(
                        "evidence manifest is busy; its writer has not finalized"
                    ) from None
                time.sleep(_LOCK_RETRY_SECONDS)
        _require_named_identity(root_fd, ".manifest.lock", "file", lock_identity)
        try:
            yield lock_identity
        finally:
            try:
                if _tree_identity(os.fstat(root_fd)) != root_identity:
                    raise EvidenceManifestError(
                        "evidence root changed while its manifest lock was held"
                    )
                if _tree_identity(os.fstat(descriptor)) != lock_identity:
                    raise EvidenceManifestError(
                        "evidence manifest lock inode changed while held"
                    )
                _require_named_identity(
                    root_fd, ".manifest.lock", "file", lock_identity
                )
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _read_regular_file(
    root_fd: int, parts: tuple[str, ...], *, max_bytes: int, inventory: _TreeInventory
) -> tuple[bytes, os.stat_result]:
    with _bound_regular_file(
        root_fd, parts, max_bytes=max_bytes, inventory=inventory
    ) as (descriptor, before):
        payload = bytearray()
        while chunk := os.read(descriptor, min(_READ_CHUNK_BYTES, max_bytes + 1)):
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise EvidenceManifestError(
                    f"evidence artifact exceeds the size limit: {'/'.join(parts)}"
                )
        return bytes(payload), _require_unchanged(descriptor, before, parts)


def _hash_regular_file(
    root_fd: int, parts: tuple[str, ...], *, max_bytes: int, inventory: _TreeInventory
) -> tuple[str, os.stat_result]:
    with _bound_regular_file(
        root_fd, parts, max_bytes=max_bytes, inventory=inventory
    ) as (descriptor, before):
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            digest.update(chunk)
        return digest.hexdigest(), _require_unchanged(descriptor, before, parts)


@contextmanager
def _bound_regular_file(
    root_fd: int, parts: tuple[str, ...], *, max_bytes: int, inventory: _TreeInventory
) -> Iterator[tuple[int, os.stat_result]]:
    display_path = "/".join(parts)
    expected = _require_inventory_entry(inventory, display_path, "file")
    with _bound_directory_chain(root_fd, parts[:-1], inventory) as directory_fd:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(parts[-1], flags, dir_fd=directory_fd)
        try:
            before = _require_regular_single_link(descriptor, display_path, max_bytes)
            if _tree_identity(before) != expected.identity:
                raise EvidenceManifestError(
                    f"evidence artifact identity does not match frozen inventory: {display_path}"
                )
            _require_named_identity(directory_fd, parts[-1], "file", expected.identity)
            try:
                yield descriptor, before
            finally:
                if _tree_identity(os.fstat(descriptor)) != expected.identity:
                    raise EvidenceManifestError(
                        f"evidence artifact changed during validation: {display_path}"
                    )
                _require_named_identity(
                    directory_fd, parts[-1], "file", expected.identity
                )
        finally:
            os.close(descriptor)


def _require_unchanged(
    descriptor: int, before: os.stat_result, parts: tuple[str, ...]
) -> os.stat_result:
    after = os.fstat(descriptor)
    if _file_identity(before) != _file_identity(after):
        raise EvidenceManifestError(
            f"evidence artifact changed while it was read: {'/'.join(parts)}"
        )
    return after


def _require_regular_single_link(
    descriptor: int, display_path: str, max_bytes: int
) -> os.stat_result:
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        raise EvidenceManifestError(
            f"evidence artifact is not a regular file: {display_path}"
        )
    if observed.st_nlink != 1:
        raise EvidenceManifestError(
            f"evidence artifact must not be hard-linked: {display_path}"
        )
    if observed.st_size > max_bytes:
        raise EvidenceManifestError(
            f"evidence artifact exceeds the size limit: {display_path}"
        )
    return observed


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _tree_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _inventory_tree(root_fd: int) -> _TreeInventory:
    entries: dict[str, _TreeEntry] = {
        ".": _TreeEntry(kind="directory", identity=_tree_identity(os.fstat(root_fd)))
    }
    pending: list[tuple[str, ...]] = [()]
    while pending:
        parts = pending.pop()
        if len(parts) > _MAX_TREE_DEPTH:
            raise EvidenceManifestError(
                f"evidence tree exceeds the maximum depth near {'/'.join(parts)}"
            )
        directory_fd = _open_directory_chain(root_fd, parts)
        try:
            directory_path = "/".join(parts) or "."
            expected_directory = entries[directory_path]
            before = os.fstat(directory_fd)
            if _tree_identity(before) != expected_directory.identity:
                raise EvidenceManifestError(
                    f"evidence directory changed while it was inspected: {directory_path}"
                )
            for name in sorted(os.listdir(directory_fd)):
                observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                relative_parts = (*parts, name)
                relative = "/".join(relative_parts)
                if len(entries) >= _MAX_TREE_ENTRIES:
                    raise EvidenceManifestError(
                        f"evidence tree has too many entries near {relative}"
                    )
                if stat.S_ISLNK(observed.st_mode):
                    raise EvidenceManifestError(
                        f"evidence tree contains a symlink: {relative}"
                    )
                if stat.S_ISREG(observed.st_mode):
                    if observed.st_nlink != 1:
                        raise EvidenceManifestError(
                            f"evidence artifact must not be hard-linked: {relative}"
                        )
                    entries[relative] = _TreeEntry(
                        kind="file", identity=_tree_identity(observed)
                    )
                    continue
                if not stat.S_ISDIR(observed.st_mode):
                    raise EvidenceManifestError(
                        f"evidence tree contains a non-regular entry: {relative}"
                    )
                if len(relative_parts) > _MAX_TREE_DEPTH:
                    raise EvidenceManifestError(
                        f"evidence tree exceeds the maximum depth near {relative}"
                    )
                child_fd = _open_directory_chain(directory_fd, (name,))
                try:
                    opened = os.fstat(child_fd)
                finally:
                    os.close(child_fd)
                if _tree_identity(opened) != _tree_identity(observed):
                    raise EvidenceManifestError(
                        f"evidence directory changed while it was inspected: {relative}"
                    )
                entries[relative] = _TreeEntry(
                    kind="directory", identity=_tree_identity(opened)
                )
                pending.append(relative_parts)
            after = os.fstat(directory_fd)
            if _tree_identity(after) != _tree_identity(before):
                raise EvidenceManifestError(
                    f"evidence directory changed while it was inspected: {directory_path}"
                )
        finally:
            os.close(directory_fd)
    return _TreeInventory(entries=entries)


def _parse_manifest(payload: bytes) -> _Manifest:
    try:
        decoded = payload.decode("utf-8")
        raw = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        canonical = (
            json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        if payload != canonical:
            raise EvidenceManifestError(
                "evidence manifest is not canonical sorted, indented JSON"
            )
        return _Manifest.model_validate(raw)
    except EvidenceManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise EvidenceManifestError(f"invalid evidence manifest: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceManifestError(
                f"evidence manifest contains a duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise EvidenceManifestError(
        f"evidence manifest contains a non-JSON numeric value: {value}"
    )


def _validate_identity(
    manifest: _Manifest,
    *,
    runner_id: str,
    baseline_sha: str,
    candidate_sha: str,
    upstream_sha: str,
    expected_uv_lock_sha256: str,
    expected_scenarios: tuple[EvidenceScenarioContract, ...],
) -> None:
    expected = {
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "upstream_sha": upstream_sha,
    }
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise EvidenceManifestError(
                f"evidence manifest identity mismatch for {field}: expected {value}"
            )
    if manifest.environment.runner != runner_id:
        raise EvidenceManifestError(
            f"evidence manifest identity mismatch for runner: expected {runner_id!r}"
        )
    if manifest.environment.uv_lock_sha256 != expected_uv_lock_sha256:
        raise EvidenceManifestError(
            "evidence manifest identity mismatch for uv_lock_sha256: expected "
            f"{expected_uv_lock_sha256}"
        )
    observed_scenarios = tuple(scenario.id for scenario in manifest.scenarios)
    if len(observed_scenarios) > _MAX_TREE_ENTRIES:
        raise EvidenceManifestError("evidence manifest has too many scenarios")
    expected_scenario_ids = tuple(scenario.id for scenario in expected_scenarios)
    if observed_scenarios != expected_scenario_ids:
        raise EvidenceManifestError(
            "evidence manifest scenarios do not match frozen control metadata: "
            f"expected {expected_scenario_ids}, observed {observed_scenarios}"
        )
    for scenario, contract in zip(manifest.scenarios, expected_scenarios, strict=True):
        _validate_scenario_contract(scenario, contract)


def _validate_scenario_contract(
    scenario: _Scenario, contract: EvidenceScenarioContract
) -> None:
    checks = {
        "surface": (scenario.surface, contract.surface),
        "command": (scenario.command, contract.command),
        "recorded_environment": (
            scenario.recorded_environment,
            contract.recorded_environment.values,
        ),
        "status": (scenario.status, contract.expected_status),
        "notes": (scenario.notes, contract.allowed_notes),
    }
    for field, (observed, expected) in checks.items():
        if observed != expected:
            raise EvidenceManifestError(
                f"scenario {scenario.id} {field} does not match frozen control "
                f"metadata: expected {expected!r}, observed {observed!r}"
            )


def _validate_artifacts(
    root_fd: int,
    manifest: _Manifest,
    *,
    expected_scenarios: tuple[EvidenceScenarioContract, ...],
    inventory: _TreeInventory,
) -> tuple[int, int]:
    scenario_ids = [scenario.id for scenario in manifest.scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise EvidenceManifestError("evidence manifest has duplicate scenario IDs")

    declared_paths: set[str] = set()
    artifact_bytes = 0
    contracts = {contract.id: contract for contract in expected_scenarios}
    for scenario in manifest.scenarios:
        scenario_bytes = _validate_scenario_artifacts(
            root_fd,
            scenario,
            contracts[scenario.id],
            inventory=inventory,
            declared_paths=declared_paths,
        )
        artifact_bytes += scenario_bytes
        if artifact_bytes > _MAX_TOTAL_ARTIFACT_BYTES:
            raise EvidenceManifestError(
                "evidence artifacts exceed the aggregate size limit"
            )
    return len(declared_paths), artifact_bytes


def _validate_scenario_artifacts(
    root_fd: int,
    scenario: _Scenario,
    contract: EvidenceScenarioContract,
    *,
    inventory: _TreeInventory,
    declared_paths: set[str],
) -> int:
    scenario_paths: set[str] = set()
    observed_types: set[str] = set()
    artifact_bytes = 0
    result_payload: bytes | None = None
    for artifact in scenario.artifacts:
        parts = PurePosixPath(artifact.path).parts
        if parts[0] != scenario.id:
            raise EvidenceManifestError(
                f"scenario {scenario.id} declares an artifact outside its directory"
            )
        if artifact.path in declared_paths:
            raise EvidenceManifestError(
                f"duplicate evidence artifact path: {artifact.path}"
            )
        declared_paths.add(artifact.path)
        scenario_paths.add(artifact.path)
        observed_types.add(artifact.type)
        if len(declared_paths) > _MAX_ARTIFACTS:
            raise EvidenceManifestError("evidence manifest has too many artifacts")
        if artifact.path == scenario.result_path:
            result_payload, observed = _read_regular_file(
                root_fd, parts, max_bytes=_MAX_RESULT_BYTES, inventory=inventory
            )
            observed_digest = hashlib.sha256(result_payload).hexdigest()
        else:
            observed_digest, observed = _hash_regular_file(
                root_fd, parts, max_bytes=_MAX_ARTIFACT_BYTES, inventory=inventory
            )
        if observed_digest != artifact.sha256:
            raise EvidenceManifestError(
                f"evidence artifact digest mismatch: {artifact.path}"
            )
        artifact_bytes += observed.st_size
    missing_types = sorted(set(contract.required_artifact_types) - observed_types)
    if missing_types:
        raise EvidenceManifestError(
            f"scenario {scenario.id} is missing required artifact types: "
            f"{', '.join(missing_types)}"
        )
    _validate_result_artifact(scenario, contract, result_payload)
    _validate_scenario_inventory(scenario.id, scenario_paths, inventory)
    return artifact_bytes


def _validate_scenario_inventory(
    scenario_id: str, scenario_paths: set[str], inventory: _TreeInventory
) -> None:
    inventory_files = {
        path for path in inventory.files if PurePosixPath(path).parts[0] == scenario_id
    }
    inventory_directories = {
        path
        for path in inventory.directories
        if PurePosixPath(path).parts[0] == scenario_id and path != scenario_id
    }
    expected_directories = {
        PurePosixPath(*PurePosixPath(path).parts[:index]).as_posix()
        for path in scenario_paths
        for index in range(2, len(PurePosixPath(path).parts))
    }
    if (
        inventory_files == scenario_paths
        and inventory_directories == expected_directories
    ):
        return
    details = []
    if undeclared_files := sorted(inventory_files - scenario_paths):
        details.append(f"undeclared files: {', '.join(undeclared_files)}")
    if absent_files := sorted(scenario_paths - inventory_files):
        details.append(f"missing files: {', '.join(absent_files)}")
    if undeclared_directories := sorted(inventory_directories - expected_directories):
        details.append(f"undeclared directories: {', '.join(undeclared_directories)}")
    raise EvidenceManifestError(
        f"scenario {scenario_id} artifact inventory mismatch ({'; '.join(details)})"
    )


def _validate_result_artifact(
    scenario: _Scenario, contract: EvidenceScenarioContract, payload: bytes | None
) -> None:
    result_artifact = next(
        (
            artifact
            for artifact in scenario.artifacts
            if artifact.path == scenario.result_path
        ),
        None,
    )
    if result_artifact is None or result_artifact.type != "result" or payload is None:
        raise EvidenceManifestError(
            f"scenario {scenario.id} result_path must identify a result artifact"
        )
    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except EvidenceManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceManifestError(
            f"scenario {scenario.id} result artifact is invalid JSON: {exc}"
        ) from exc
    errors = sorted(
        Draft202012Validator(contract.result_schema).iter_errors(result),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        raise EvidenceManifestError(
            f"scenario {scenario.id} result artifact violates its frozen schema: "
            f"{errors[0].message}"
        )
    if not isinstance(result, dict):
        raise EvidenceManifestError(
            f"scenario {scenario.id} result artifact must be an object"
        )
    result_checks = {
        "status": scenario.status,
        "notes": list(scenario.notes),
        "gap_notes": list(contract.allowed_gap_notes),
    }
    for field, expected in result_checks.items():
        if result.get(field) != expected:
            raise EvidenceManifestError(
                f"scenario {scenario.id} result {field} does not match frozen "
                f"control metadata: expected {expected!r}"
            )


def _validate_root_inventory(
    inventory: _TreeInventory, expected_scenarios: tuple[EvidenceScenarioContract, ...]
) -> None:
    expected_scenario_ids = tuple(scenario.id for scenario in expected_scenarios)
    expected = {
        "manifest.json",
        ".manifest.lock",
        ".reservations",
        *expected_scenario_ids,
    }
    observed_names = {
        path for path in inventory.entries if path != "." and "/" not in path
    }
    if observed_names != expected:
        unexpected = sorted(observed_names - expected)
        missing = sorted(expected - observed_names)
        details = []
        if unexpected:
            details.append(f"unexpected entries: {', '.join(unexpected)}")
        if missing:
            details.append(f"missing entries: {', '.join(missing)}")
        raise EvidenceManifestError(
            f"evidence root inventory mismatch ({'; '.join(details)})"
        )

    for filename in ("manifest.json", ".manifest.lock"):
        entry = inventory.entries[filename]
        if entry.kind != "file":
            raise EvidenceManifestError(
                f"evidence run metadata is not a single-link regular file: {filename}"
            )
    for directory in (".reservations", *expected_scenario_ids):
        entry = inventory.entries[directory]
        if entry.kind != "directory":
            raise EvidenceManifestError(
                f"evidence run entry is not a physical directory: {directory}"
            )
    reservations = [
        path for path in inventory.entries if path.startswith(".reservations/")
    ]
    if reservations:
        raise EvidenceManifestError(
            "evidence run still has active or stale reservations: "
            f"{', '.join(sorted(reservations))}"
        )


__all__ = [
    "EvidenceManifestError",
    "EvidenceManifestSnapshot",
    "EvidenceScenarioContract",
    "revalidate_evidence_snapshot",
    "validate_evidence_manifest",
]
