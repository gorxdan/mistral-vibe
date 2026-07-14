from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from pydantic import ValidationError
import yaml

from vibe.core._trusted_command import (
    TRUSTED_GIT_CONFIG_ARGS,
    TrustedCommandError,
    minimal_trusted_git_environment,
    resolve_trusted_system_executable,
)
from vibe.core.config._verification_config import TrustedExecutionTopologyConfig
from vibe.core.evidence_manifest import (
    EvidenceManifestError,
    EvidenceManifestSnapshot,
    EvidenceScenarioContract,
    revalidate_evidence_snapshot,
    validate_evidence_manifest,
)
from vibe.core.utils.io import decode_safe, read_safe

_FRONTMATTER_PARTS = 3
_LS_TREE_FIELDS = 3
_MOUNTINFO_MIN_SEPARATOR_INDEX = 6
_MOUNTINFO_PATH = Path("/proc/self/mountinfo")
_VOLATILE_FILESYSTEMS = {"ramfs", "tmpfs"}


class ExecutionTopologyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutionTopologySnapshot:
    control_worktree: Path
    candidate_worktree: Path
    evidence_workspace: Path
    git_common_directory: Path
    registered_worktrees: tuple[Path, ...]
    candidate_head: str
    evidence_manifest_sha256: str | None
    evidence_snapshot: EvidenceManifestSnapshot | None


def validate_execution_topology(
    topology: TrustedExecutionTopologyConfig, *, current_directory: Path
) -> ExecutionTopologySnapshot:
    control = _require_directory(topology.control_worktree, "control worktree")
    candidate = _require_directory(topology.candidate_worktree, "candidate worktree")
    evidence = _require_directory(topology.evidence_workspace, "evidence workspace")
    if control == candidate:
        raise ExecutionTopologyError(
            "control and candidate must be distinct physical worktrees"
        )

    observed_root = _canonical_git_path(
        candidate, _git(candidate, "rev-parse", "--show-toplevel")
    )
    current_root = _canonical_git_path(
        current_directory, _git(current_directory, "rev-parse", "--show-toplevel")
    )
    if observed_root != candidate or current_root != candidate:
        raise ExecutionTopologyError(
            "session must start inside the assigned candidate worktree"
        )

    worktrees = _registered_worktrees(candidate)
    if control not in worktrees:
        raise ExecutionTopologyError(
            "control worktree is not a registered physical Git worktree"
        )
    if candidate not in worktrees:
        raise ExecutionTopologyError(
            "candidate worktree is not a registered physical Git worktree"
        )

    common_directory = _canonical_git_path(
        candidate, _git(candidate, "rev-parse", "--git-common-dir")
    )
    _validate_external_evidence(evidence, worktrees, common_directory)
    _validate_durable_filesystem(evidence)

    _require_clean(control, "control worktree")
    _require_clean(candidate, "candidate worktree")
    _require_exact_commit(control, topology.control_sha, "control SHA")
    _require_resolvable_commit(candidate, topology.baseline_sha, "baseline SHA")
    _require_resolvable_commit(candidate, topology.upstream_sha, "upstream SHA")

    candidate_head = _git(candidate, "rev-parse", "HEAD")
    expected_head = topology.candidate_sha or topology.baseline_sha
    if candidate_head != expected_head:
        raise ExecutionTopologyError(
            f"candidate HEAD mismatch: expected {expected_head}, observed {candidate_head}"
        )
    branch = _git(candidate, "branch", "--show-current")
    if branch != topology.candidate_branch:
        raise ExecutionTopologyError(
            f"candidate branch mismatch: expected {topology.candidate_branch!r}, "
            f"observed {branch!r}"
        )

    expected_scenarios = _validate_control_metadata(topology, control)
    if topology.state == "active":
        _probe_active_durable_workspace(evidence)
    evidence_snapshot = _validate_verification_evidence(
        topology,
        candidate=candidate,
        evidence=evidence,
        expected_scenarios=expected_scenarios,
    )
    return ExecutionTopologySnapshot(
        control_worktree=control,
        candidate_worktree=candidate,
        evidence_workspace=evidence,
        git_common_directory=common_directory,
        registered_worktrees=worktrees,
        candidate_head=candidate_head,
        evidence_manifest_sha256=(
            evidence_snapshot.manifest_sha256 if evidence_snapshot else None
        ),
        evidence_snapshot=evidence_snapshot,
    )


def revalidate_execution_topology_snapshot(
    topology: TrustedExecutionTopologyConfig,
    snapshot: ExecutionTopologySnapshot,
    *,
    current_directory: Path,
) -> None:
    control = _require_directory(topology.control_worktree, "control worktree")
    candidate = _require_directory(topology.candidate_worktree, "candidate worktree")
    evidence = _require_directory(topology.evidence_workspace, "evidence workspace")
    if (control, candidate, evidence) != (
        snapshot.control_worktree,
        snapshot.candidate_worktree,
        snapshot.evidence_workspace,
    ):
        raise ExecutionTopologyError("execution topology paths changed after startup")

    observed_root = _canonical_git_path(
        candidate, _git(candidate, "rev-parse", "--show-toplevel")
    )
    current_root = _canonical_git_path(
        current_directory, _git(current_directory, "rev-parse", "--show-toplevel")
    )
    if observed_root != candidate or current_root != candidate:
        raise ExecutionTopologyError(
            "session moved outside the assigned candidate worktree"
        )

    worktrees = _registered_worktrees(candidate)
    if control not in worktrees or candidate not in worktrees:
        raise ExecutionTopologyError("assigned physical worktree registration changed")
    common_directory = _canonical_git_path(
        candidate, _git(candidate, "rev-parse", "--git-common-dir")
    )
    if common_directory != snapshot.git_common_directory:
        raise ExecutionTopologyError("Git common directory changed after startup")
    _validate_external_evidence(evidence, worktrees, common_directory)
    _validate_durable_filesystem(evidence)

    _require_clean(control, "control worktree")
    _require_clean(candidate, "candidate worktree")
    _require_exact_commit(control, topology.control_sha, "control SHA")
    _require_resolvable_commit(candidate, topology.baseline_sha, "baseline SHA")
    _require_resolvable_commit(candidate, topology.upstream_sha, "upstream SHA")
    candidate_head = _git(candidate, "rev-parse", "HEAD")
    expected_head = topology.candidate_sha or topology.baseline_sha
    if candidate_head != expected_head or candidate_head != snapshot.candidate_head:
        raise ExecutionTopologyError("candidate HEAD changed after startup")
    if _git(candidate, "branch", "--show-current") != topology.candidate_branch:
        raise ExecutionTopologyError("candidate branch changed after startup")

    evidence_snapshot = snapshot.evidence_snapshot
    if topology.state == "verification":
        if (
            evidence_snapshot is None
            or snapshot.evidence_manifest_sha256 != topology.evidence_manifest_sha256
        ):
            raise ExecutionTopologyError(
                "frozen evidence identity changed after startup"
            )
        try:
            revalidate_evidence_snapshot(evidence_snapshot)
        except EvidenceManifestError as exc:
            raise ExecutionTopologyError(str(exc)) from exc
    elif evidence_snapshot is not None:
        raise ExecutionTopologyError("active topology acquired verification evidence")


def _validate_verification_evidence(
    topology: TrustedExecutionTopologyConfig,
    *,
    candidate: Path,
    evidence: Path,
    expected_scenarios: tuple[EvidenceScenarioContract, ...],
) -> EvidenceManifestSnapshot | None:
    if topology.state != "verification":
        return None
    candidate_sha = topology.candidate_sha
    expected_manifest_sha256 = topology.evidence_manifest_sha256
    if candidate_sha is None or expected_manifest_sha256 is None:
        raise ExecutionTopologyError(
            "verification topology is missing frozen evidence identity"
        )
    try:
        return validate_evidence_manifest(
            evidence,
            run_id=topology.run_id,
            runner_id=topology.runner_id,
            baseline_sha=topology.baseline_sha,
            candidate_sha=candidate_sha,
            upstream_sha=topology.upstream_sha,
            expected_uv_lock_sha256=hashlib.sha256(
                _read_git_blob(
                    candidate, candidate_sha, "uv.lock", "candidate lockfile"
                )
            ).hexdigest(),
            expected_manifest_sha256=expected_manifest_sha256,
            expected_scenarios=expected_scenarios,
        )
    except EvidenceManifestError as exc:
        raise ExecutionTopologyError(str(exc)) from exc


def _require_directory(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ExecutionTopologyError(f"{label} is not a directory: {path}")
    return path


def _git(directory: Path, *arguments: str) -> str:
    try:
        git = resolve_trusted_system_executable("git")
        result = subprocess.run(
            [str(git), *TRUSTED_GIT_CONFIG_ARGS, "-C", str(directory), *arguments],
            check=False,
            capture_output=True,
            env=_git_environment(),
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError, TrustedCommandError) as exc:
        raise ExecutionTopologyError(
            f"Git topology probe failed for {directory}: {exc}"
        ) from exc
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ExecutionTopologyError(
            f"Git topology probe failed ({' '.join(arguments)}): {diagnostic}"
        )
    return result.stdout.strip()


def _git_bytes(directory: Path, *arguments: str) -> bytes:
    try:
        git = resolve_trusted_system_executable("git")
        result = subprocess.run(
            [str(git), *TRUSTED_GIT_CONFIG_ARGS, "-C", str(directory), *arguments],
            check=False,
            capture_output=True,
            env=_git_environment(),
            text=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError, TrustedCommandError) as exc:
        raise ExecutionTopologyError(
            f"Git topology probe failed for {directory}: {exc}"
        ) from exc
    if result.returncode != 0:
        raw_diagnostic = result.stderr.strip() or result.stdout.strip()
        diagnostic = (
            decode_safe(raw_diagnostic, from_subprocess=True).text
            if raw_diagnostic
            else "no output"
        )
        raise ExecutionTopologyError(
            f"Git topology probe failed ({' '.join(arguments)}): {diagnostic}"
        )
    return result.stdout


def _git_environment() -> dict[str, str]:
    return minimal_trusted_git_environment(Path("/"))


def _canonical_git_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _registered_worktrees(repository: Path) -> tuple[Path, ...]:
    worktrees: list[Path] = []
    for line in _git(repository, "worktree", "list", "--porcelain").splitlines():
        if not line.startswith("worktree "):
            continue
        worktrees.append(Path(line.removeprefix("worktree ")).resolve())
    return tuple(worktrees)


def _validate_external_evidence(
    evidence: Path, worktrees: tuple[Path, ...], common_directory: Path
) -> None:
    for volatile_root in _known_volatile_roots():
        if evidence == volatile_root or evidence.is_relative_to(volatile_root):
            raise ExecutionTopologyError(
                "evidence workspace must be durable and cannot be inside a known "
                f"volatile directory: {volatile_root}"
            )
    excluded = (*worktrees, common_directory)
    for root in excluded:
        if (
            evidence == root
            or evidence.is_relative_to(root)
            or root.is_relative_to(evidence)
        ):
            raise ExecutionTopologyError(
                f"evidence workspace overlaps repository control state: {root}"
            )


def _known_volatile_roots() -> tuple[Path, ...]:
    roots = {
        Path("/tmp").resolve(),
        Path("/run").resolve(),
        Path("/dev/shm").resolve(),
        Path(tempfile.gettempdir()).resolve(),
    }
    return tuple(sorted(roots, key=str))


def _validate_durable_filesystem(evidence: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        mountinfo = read_safe(_MOUNTINFO_PATH, raise_on_error=True).text
    except (OSError, UnicodeError) as exc:
        raise ExecutionTopologyError(
            f"could not inspect evidence filesystem durability: {_MOUNTINFO_PATH}: {exc}"
        ) from exc
    matched_mount: Path | None = None
    matched_filesystem: str | None = None
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < _MOUNTINFO_MIN_SEPARATOR_INDEX or separator + 1 >= len(fields):
            continue
        mount = Path(_decode_mountinfo_path(fields[4])).resolve()
        if evidence != mount and not evidence.is_relative_to(mount):
            continue
        if matched_mount is None or len(mount.parts) > len(matched_mount.parts):
            matched_mount = mount
            matched_filesystem = fields[separator + 1]
    if matched_mount is None or matched_filesystem is None:
        raise ExecutionTopologyError(
            f"could not identify the evidence workspace mount: {evidence}"
        )
    if matched_filesystem.casefold() in _VOLATILE_FILESYSTEMS:
        raise ExecutionTopologyError(
            "evidence workspace must use durable storage; mount "
            f"{matched_mount} uses {matched_filesystem}"
        )


def _decode_mountinfo_path(value: str) -> str:
    decoded = value
    for escaped, replacement in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        decoded = decoded.replace(escaped, replacement)
    return decoded


def _probe_active_durable_workspace(evidence: Path) -> None:
    name = f".vibe-topology-probe-{uuid.uuid4().hex}"
    payload = uuid.uuid4().bytes
    directory_fd: int | None = None
    created = False
    try:
        directory_fd = os.open(
            evidence, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        descriptor = os.open(
            name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        created = True
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise ExecutionTopologyError(
                    "evidence workspace failed the durable write probe"
                )
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
        descriptor = os.open(
            name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd
        )
        try:
            observed = os.read(descriptor, len(payload) + 1)
        finally:
            os.close(descriptor)
        if observed != payload:
            raise ExecutionTopologyError(
                "evidence workspace failed the durable write/read probe"
            )
    except OSError as exc:
        raise ExecutionTopologyError(
            f"evidence workspace is not durably writable: {evidence}: {exc}"
        ) from exc
    finally:
        if directory_fd is not None:
            try:
                if created:
                    os.unlink(name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except OSError as exc:
                raise ExecutionTopologyError(
                    "evidence workspace could not durably remove its active "
                    f"probe: {evidence}: {exc}"
                ) from exc
            finally:
                os.close(directory_fd)


def _require_clean(repository: Path, label: str) -> None:
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ExecutionTopologyError(f"{label} is dirty")


def _require_resolvable_commit(repository: Path, sha: str, label: str) -> None:
    observed = _git(repository, "rev-parse", f"{sha}^{{commit}}")
    if observed != sha:
        raise ExecutionTopologyError(
            f"{label} did not resolve exactly: expected {sha}, observed {observed}"
        )


def _require_exact_commit(repository: Path, sha: str, label: str) -> None:
    _require_resolvable_commit(repository, sha, label)
    observed = _git(repository, "rev-parse", "HEAD")
    if observed != sha:
        raise ExecutionTopologyError(
            f"{label} mismatch: expected {sha}, observed {observed}"
        )


def _validate_control_metadata(
    topology: TrustedExecutionTopologyConfig, control: Path
) -> tuple[EvidenceScenarioContract, ...]:
    packet, status = _load_control_documents(topology, control)
    entries = status.get("packets")
    if not isinstance(entries, list):
        raise ExecutionTopologyError("campaign status has no packets list")
    states = _status_states(status)
    entry = next(
        (
            item
            for item in entries
            if isinstance(item, dict) and item.get("id") == topology.packet_id
        ),
        None,
    )
    if entry is None:
        raise ExecutionTopologyError(
            f"packet {topology.packet_id!r} is absent from campaign status"
        )
    _validate_control_assignment(topology, packet, entry)
    scenario_contracts = _validate_control_evidence(topology, packet, entry)
    _validate_control_roles(packet, entry)
    _validate_dependencies(packet, entry, states)
    return scenario_contracts


def _load_control_documents(
    topology: TrustedExecutionTopologyConfig, control: Path
) -> tuple[dict[str, object], dict[str, object]]:
    packet_source = f"{topology.control_sha}:{topology.packet_path}"
    packet = _load_packet(
        _read_control_blob(
            control, topology.control_sha, topology.packet_path, "packet"
        ),
        packet_source,
    )
    status_source = f"{topology.control_sha}:{topology.status_path}"
    status = _load_yaml_mapping(
        _read_control_blob(
            control, topology.control_sha, topology.status_path, "campaign status"
        ),
        "campaign status",
        status_source,
    )
    return packet, status


def _validate_control_assignment(
    topology: TrustedExecutionTopologyConfig,
    packet: dict[str, object],
    entry: dict[str, object],
) -> None:
    if packet.get("id") != topology.packet_id:
        raise ExecutionTopologyError(
            "packet frontmatter identity does not match assignment"
        )
    if packet.get("state") != topology.state or entry.get("state") != topology.state:
        raise ExecutionTopologyError(
            f"packet state must be {topology.state!r} in both control documents"
        )

    expected_fields: dict[str, object] = {
        "baseline_sha": topology.baseline_sha,
        "candidate_sha": topology.candidate_sha,
        "upstream_sha": topology.upstream_sha,
        "worktree": topology.candidate_worktree,
        "branch": topology.candidate_branch,
    }
    for field, expected in expected_fields.items():
        if packet.get(field) != expected or entry.get(field) != expected:
            raise ExecutionTopologyError(
                f"control metadata mismatch for {field}: expected {expected!r}"
            )


def _validate_control_evidence(
    topology: TrustedExecutionTopologyConfig,
    packet: dict[str, object],
    entry: dict[str, object],
) -> tuple[EvidenceScenarioContract, ...]:
    expected_evidence = {
        "workspace": topology.evidence_workspace,
        "run_id": topology.run_id,
        "runner_id": topology.runner_id,
    }
    if topology.evidence_manifest_sha256 is not None:
        expected_evidence["manifest_sha256"] = topology.evidence_manifest_sha256
    for document, label in ((packet, "packet"), (entry, "status")):
        evidence = document.get("evidence")
        if not isinstance(evidence, dict):
            raise ExecutionTopologyError(f"{label} evidence metadata is missing")
        for field, expected in expected_evidence.items():
            if evidence.get(field) != expected:
                raise ExecutionTopologyError(
                    f"{label} evidence mismatch for {field}: expected {expected!r}"
                )

    packet_evidence = packet.get("evidence")
    if not isinstance(packet_evidence, dict):
        raise ExecutionTopologyError("packet evidence metadata is missing")
    status_evidence = entry.get("evidence")
    if not isinstance(status_evidence, dict):
        raise ExecutionTopologyError("status evidence metadata is missing")
    packet_scenarios = packet_evidence.get("scenarios")
    status_scenarios = entry.get("required_scenarios")
    if (
        not isinstance(packet_scenarios, list)
        or not packet_scenarios
        or status_scenarios != packet_scenarios
        or any(
            not isinstance(scenario, str)
            or not scenario
            or scenario in {".", ".."}
            or "/" in scenario
            or "\\" in scenario
            for scenario in packet_scenarios
        )
        or packet_scenarios != sorted(set(packet_scenarios))
    ):
        raise ExecutionTopologyError(
            "control evidence scenario assignment is invalid or inconsistent"
        )
    packet_contracts = packet_evidence.get("scenario_contracts")
    status_contracts = status_evidence.get("scenario_contracts")
    if packet_contracts != status_contracts:
        raise ExecutionTopologyError(
            "packet and status scenario contracts must be exactly identical"
        )
    scenario_contracts = _load_scenario_contracts(packet_contracts)
    contract_ids = [contract.id for contract in scenario_contracts]
    if contract_ids != packet_scenarios:
        raise ExecutionTopologyError(
            "frozen scenario contract identities do not match the scenario assignment"
        )
    return scenario_contracts


def _validate_control_roles(
    packet: dict[str, object], entry: dict[str, object]
) -> None:
    for field in ("owner", "reviewer", "verifier", "evidence_operator"):
        packet_value = packet.get(field)
        if (
            not isinstance(packet_value, str)
            or not packet_value.strip()
            or entry.get(field) != packet_value
        ):
            raise ExecutionTopologyError(f"control role assignment is invalid: {field}")
    execution_profile = packet.get("execution_profile")
    if (
        not isinstance(execution_profile, str)
        or not execution_profile.strip()
        or entry.get("execution_profile") != execution_profile
    ):
        raise ExecutionTopologyError("control execution_profile assignment is invalid")


def _load_scenario_contracts(value: object) -> tuple[EvidenceScenarioContract, ...]:
    if not isinstance(value, list) or not value:
        raise ExecutionTopologyError("control scenario contracts are missing")
    try:
        contracts = tuple(
            EvidenceScenarioContract.model_validate(contract) for contract in value
        )
    except ValidationError as exc:
        raise ExecutionTopologyError(
            f"control scenario contract is invalid: {exc}"
        ) from exc
    ids = tuple(contract.id for contract in contracts)
    if ids != tuple(sorted(set(ids))):
        raise ExecutionTopologyError(
            "control scenario contracts must be sorted by unique scenario ID"
        )
    return contracts


def _validate_dependencies(
    packet: dict[str, object], entry: dict[str, object], states: dict[str, object]
) -> None:
    dependencies = packet.get("depends_on")
    if not isinstance(dependencies, list) or entry.get("depends_on") != dependencies:
        raise ExecutionTopologyError("control dependency metadata is invalid")
    dependency_ids: list[str] = []
    for dependency in dependencies:
        if not isinstance(dependency, str) or not dependency.strip():
            raise ExecutionTopologyError("control dependency metadata is invalid")
        dependency_ids.append(dependency)
    duplicate_dependencies = _duplicate_ids(dependency_ids)
    if duplicate_dependencies:
        raise ExecutionTopologyError(
            f"duplicate dependency ID in packet: {', '.join(duplicate_dependencies)}"
        )
    incomplete = [
        dependency
        for dependency in dependency_ids
        if states.get(dependency) != "complete"
    ]
    if incomplete:
        raise ExecutionTopologyError(
            f"packet dependencies are incomplete: {', '.join(incomplete)}"
        )


def _read_control_blob(control: Path, control_sha: str, path: str, label: str) -> str:
    raw = _read_git_blob(control, control_sha, path, label)
    try:
        return decode_safe(raw, raise_on_error=True).text
    except UnicodeError as exc:
        raise ExecutionTopologyError(
            f"could not decode {label} blob at {control_sha}:{path}: {exc}"
        ) from exc


def _read_git_blob(repository: Path, commit_sha: str, path: str, label: str) -> bytes:
    records = [
        record
        for record in _git_bytes(
            repository, "ls-tree", "-z", commit_sha, "--", path
        ).split(b"\0")
        if record
    ]
    expected_path = os.fsencode(path)
    if len(records) != 1:
        raise ExecutionTopologyError(
            f"{label} must be a regular tracked Git blob at {commit_sha}: {path}"
        )
    metadata, separator, observed_path = records[0].partition(b"\t")
    fields = metadata.split(b" ")
    if (
        separator != b"\t"
        or observed_path != expected_path
        or len(fields) != _LS_TREE_FIELDS
        or fields[0] not in {b"100644", b"100755"}
        or fields[1] != b"blob"
    ):
        raise ExecutionTopologyError(
            f"{label} must be a regular tracked Git blob at {commit_sha}: {path}"
        )
    object_id = fields[2].decode("ascii")
    return _git_bytes(repository, "cat-file", "blob", object_id)


def _load_packet(contents: str, source: str) -> dict[str, object]:
    parts = contents.split("---", 2)
    if len(parts) != _FRONTMATTER_PARTS or parts[0].strip():
        raise ExecutionTopologyError(f"invalid packet frontmatter: {source}")
    try:
        payload = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        raise ExecutionTopologyError(
            f"could not parse packet frontmatter: {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExecutionTopologyError(f"packet frontmatter is not a mapping: {source}")
    return payload


def _load_yaml_mapping(contents: str, label: str, source: str) -> dict[str, object]:
    try:
        payload = yaml.safe_load(contents)
    except yaml.YAMLError as exc:
        raise ExecutionTopologyError(
            f"could not parse {label}: {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExecutionTopologyError(f"{label} is not a mapping: {source}")
    return payload


def _status_states(status: dict[str, object]) -> dict[str, object]:
    states: dict[str, object] = {}
    for section in ("packets", "required_future_packets"):
        entries = status.get(section, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                packet_id = entry["id"]
                if packet_id in states:
                    raise ExecutionTopologyError(
                        f"duplicate campaign packet ID: {packet_id}"
                    )
                states[packet_id] = entry.get("state")
    return states


def _duplicate_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


__all__ = [
    "ExecutionTopologyError",
    "ExecutionTopologySnapshot",
    "revalidate_execution_topology_snapshot",
    "validate_execution_topology",
]
