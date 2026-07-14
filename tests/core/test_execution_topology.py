from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Literal, cast

from git import Repo
from pydantic import ValidationError
import pytest
import yaml

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core import evidence_manifest as evidence_manifest_module
from vibe.core._verification_receipt import VerificationReceiptStore
from vibe.core.agents.manager import AgentManager
from vibe.core.config import (
    TrustedExecutionTopologyConfig,
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.evidence_manifest import revalidate_evidence_snapshot
from vibe.core.execution_topology import (
    ExecutionTopologyError,
    validate_execution_topology,
)
from vibe.core.tools.base import BaseToolState, InvokeContext
from vibe.core.tools.builtins.verify_work import (
    VerifyWork,
    VerifyWorkArgs,
    VerifyWorkConfig,
)
from vibe.core.utils.io import read_safe, write_safe
from vibe.core.verification_contract import (
    CommandEvidence,
    VerificationReport,
    VerificationVerdict,
)
from vibe.core.verification_state import VerificationState


@dataclass(frozen=True, slots=True)
class _TopologyFixture:
    topology: TrustedExecutionTopologyConfig
    candidate: Path
    control: Path
    evidence: Path


def _topology_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: Literal["active", "verification"] = "active",
) -> _TopologyFixture:
    candidate = tmp_path / "candidate"
    control = tmp_path / "control"
    evidence = tmp_path / "durable" / "evidence"
    evidence.mkdir(parents=True)
    (tmp_path / "system-temporary").mkdir()
    mountinfo = tmp_path / "mountinfo"
    write_safe(mountinfo, "1 0 0:1 / / rw - ext4 /dev/root rw\n")
    monkeypatch.setattr(
        "vibe.core.execution_topology.tempfile.gettempdir",
        lambda: str(tmp_path / "system-temporary"),
    )
    monkeypatch.setattr(
        "vibe.core.execution_topology._known_volatile_roots",
        lambda: (tmp_path / "system-temporary", Path("/run"), Path("/dev/shm")),
    )
    monkeypatch.setattr("vibe.core.execution_topology._MOUNTINFO_PATH", mountinfo)

    candidate.mkdir()
    repo = Repo.init(candidate, initial_branch="candidate")
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")
    write_safe(candidate / "tracked.txt", "baseline\n")
    write_safe(candidate / "uv.lock", "test-lock\n")
    repo.index.add(["tracked.txt", "uv.lock"])
    baseline_sha = repo.index.commit("baseline").hexsha
    repo.git.worktree("add", "-b", "control", str(control), baseline_sha)
    candidate_sha = None
    manifest_sha256 = None
    if state == "verification":
        write_safe(candidate / "tracked.txt", "candidate\n")
        repo.index.add(["tracked.txt"])
        candidate_sha = repo.index.commit("candidate").hexsha
        artifact_path = (
            evidence / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
        )
        artifact_payload = b'{"gap_notes": [], "notes": [], "status": "pass"}\n'
        artifact_path.parent.mkdir(parents=True)
        (artifact_path.parents[1] / ".reservations").mkdir()
        write_safe(artifact_path.parents[1] / ".manifest.lock", "")
        write_safe(artifact_path, artifact_payload.decode())
        manifest_payload = (
            json.dumps(
                {
                    "version": 1,
                    "baseline_sha": baseline_sha,
                    "candidate_sha": candidate_sha,
                    "upstream_sha": baseline_sha,
                    "environment": {
                        "python": "3.12.11",
                        "platform": "linux-x86_64",
                        "uv_lock_sha256": hashlib.sha256(b"test-lock\n").hexdigest(),
                        "runner": "local-test",
                    },
                    "scenarios": [
                        {
                            "id": "IT-13",
                            "surface": "non_ui",
                            "status": "pass",
                            "command": ["/usr/bin/true"],
                            "recorded_environment": {},
                            "exit_code": 0,
                            "artifacts": [
                                {
                                    "type": "result",
                                    "path": "IT-13/result.json",
                                    "sha256": hashlib.sha256(
                                        artifact_payload
                                    ).hexdigest(),
                                }
                            ],
                            "metrics": {},
                            "notes": [],
                            "started_at": "2026-07-13T12:00:00+00:00",
                            "finished_at": "2026-07-13T12:00:01+00:00",
                            "result_path": "IT-13/result.json",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        write_safe(
            artifact_path.parents[1] / "manifest.json", manifest_payload.decode()
        )
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()

    scenario_contracts = [
        {
            "id": "IT-13",
            "surface": "non_ui",
            "command": ["/usr/bin/true"],
            "recorded_environment": {"policy": "exact", "values": {}},
            "required_artifact_types": ["result"],
            "result_schema": {
                "additionalProperties": False,
                "properties": {
                    "gap_notes": {"items": {"type": "string"}, "type": "array"},
                    "notes": {"items": {"type": "string"}, "type": "array"},
                    "status": {"enum": ["pass", "fail"]},
                },
                "required": ["gap_notes", "notes", "status"],
                "type": "object",
            },
            "expected_status": "pass",
            "allowed_notes": [],
            "allowed_gap_notes": [],
        }
    ]
    packet_path = "docs/design/fork-maintenance/packets/I00-P01-test.md"
    status_path = "docs/design/fork-maintenance/status.yaml"
    packet = {
        "id": "I00-P01",
        "state": state,
        "owner": "worker",
        "reviewer": "reviewer",
        "verifier": "verifier",
        "evidence_operator": "operator",
        "depends_on": [],
        "baseline_sha": baseline_sha,
        "candidate_sha": candidate_sha,
        "upstream_sha": baseline_sha,
        "worktree": str(candidate),
        "branch": "candidate",
        "execution_profile": "managed",
        "evidence": {
            "workspace": str(evidence),
            "run_id": "i00-p01-test",
            "runner_id": "local-test",
            "scenarios": ["IT-13"],
            "scenario_contracts": scenario_contracts,
        },
        "required_scenarios": ["IT-13"],
    }
    if manifest_sha256 is not None:
        evidence_metadata = packet["evidence"]
        assert isinstance(evidence_metadata, dict)
        evidence_metadata["manifest_sha256"] = manifest_sha256
    packet_file = control / packet_path
    packet_file.parent.mkdir(parents=True)
    write_safe(packet_file, f"---\n{yaml.safe_dump(packet)}---\n\n# Test packet\n")
    status_file = control / status_path
    status_file.parent.mkdir(parents=True, exist_ok=True)
    write_safe(status_file, yaml.safe_dump({"packets": [packet]}))
    control_repo = Repo(control)
    control_repo.index.add([packet_path, status_path])
    control_sha = control_repo.index.commit("activate packet").hexsha

    topology = TrustedExecutionTopologyConfig(
        packet_id="I00-P01",
        packet_path=packet_path,
        status_path=status_path,
        state=state,
        control_worktree=str(control),
        control_sha=control_sha,
        candidate_worktree=str(candidate),
        candidate_branch="candidate",
        baseline_sha=baseline_sha,
        candidate_sha=candidate_sha,
        upstream_sha=baseline_sha,
        evidence_workspace=str(evidence),
        run_id="i00-p01-test",
        runner_id="local-test",
        evidence_manifest_sha256=manifest_sha256,
    )
    return _TopologyFixture(topology, candidate, control, evidence)


def _recipe(
    topology: TrustedExecutionTopologyConfig,
) -> TrustedVerificationRecipeConfig:
    return TrustedVerificationRecipeConfig(
        recipe_version="test-v1",
        task_brief="Test the managed candidate",
        acceptance_contract="The topology and focused check must pass",
        allowed_paths=("tracked.txt",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=(sys.executable, "-c", "print('ok')"),
                executable_sha256=hashlib.sha256(
                    Path(sys.executable).read_bytes()
                ).hexdigest(),
                environment_attestation_path="/usr/bin/true",
                environment_attestation_sha256=hashlib.sha256(
                    Path("/usr/bin/true").read_bytes()
                ).hexdigest(),
            ),
        ),
        execution_topology=topology,
    )


def _commit_control_paths(
    fixture: _TopologyFixture, *paths: str
) -> TrustedExecutionTopologyConfig:
    repository = Repo(fixture.control)
    repository.index.add(list(paths))
    control_sha = repository.index.commit("update control metadata").hexsha
    return fixture.topology.model_copy(update={"control_sha": control_sha})


def _load_mapping(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(read_safe(path, raise_on_error=True).text)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _load_packet_mapping(path: Path) -> dict[str, object]:
    parts = read_safe(path, raise_on_error=True).text.split("---", 2)
    assert len(parts) == 3
    payload = yaml.safe_load(parts[1])
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _write_packet_mapping(path: Path, payload: dict[str, object]) -> None:
    write_safe(path, f"---\n{yaml.safe_dump(payload)}---\n\n# Test packet\n")


def _manifest_path(fixture: _TopologyFixture) -> Path:
    return fixture.evidence / ".ai/runs/i00-p01-test/test-evidence/latest/manifest.json"


def _rebind_manifest(
    fixture: _TopologyFixture, payload: bytes
) -> TrustedExecutionTopologyConfig:
    write_safe(_manifest_path(fixture), payload.decode())
    digest = hashlib.sha256(payload).hexdigest()

    packet_path = fixture.control / fixture.topology.packet_path
    packet = _load_packet_mapping(packet_path)
    packet_evidence = packet["evidence"]
    assert isinstance(packet_evidence, dict)
    packet_evidence["manifest_sha256"] = digest
    _write_packet_mapping(packet_path, packet)

    status_path = fixture.control / fixture.topology.status_path
    status = _load_mapping(status_path)
    packets = status["packets"]
    assert isinstance(packets, list)
    entry = packets[0]
    assert isinstance(entry, dict)
    status_evidence = entry["evidence"]
    assert isinstance(status_evidence, dict)
    status_evidence["manifest_sha256"] = digest
    write_safe(status_path, yaml.safe_dump(status))

    topology = _commit_control_paths(
        fixture, fixture.topology.packet_path, fixture.topology.status_path
    )
    return topology.model_copy(update={"evidence_manifest_sha256": digest})


def _update_scenario_contract(fixture: _TopologyFixture, **updates: object) -> None:
    packet_path = fixture.control / fixture.topology.packet_path
    packet = _load_packet_mapping(packet_path)
    packet_evidence = packet["evidence"]
    assert isinstance(packet_evidence, dict)
    packet_contracts = packet_evidence["scenario_contracts"]
    assert isinstance(packet_contracts, list)
    packet_contract = packet_contracts[0]
    assert isinstance(packet_contract, dict)
    packet_contract.update(updates)
    _write_packet_mapping(packet_path, packet)

    status_path = fixture.control / fixture.topology.status_path
    status = _load_mapping(status_path)
    packets = status["packets"]
    assert isinstance(packets, list)
    entry = packets[0]
    assert isinstance(entry, dict)
    status_evidence = entry["evidence"]
    assert isinstance(status_evidence, dict)
    status_contracts = status_evidence["scenario_contracts"]
    assert isinstance(status_contracts, list)
    status_contract = status_contracts[0]
    assert isinstance(status_contract, dict)
    status_contract.update(updates)
    write_safe(status_path, yaml.safe_dump(status))


def test_valid_physical_topology_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)

    snapshot = validate_execution_topology(
        fixture.topology, current_directory=fixture.candidate
    )

    assert snapshot.control_worktree == fixture.control
    assert snapshot.candidate_worktree == fixture.candidate
    assert snapshot.evidence_workspace == fixture.evidence
    assert snapshot.candidate_head == fixture.topology.baseline_sha
    assert snapshot.evidence_manifest_sha256 is None
    assert not list(fixture.evidence.glob(".vibe-topology-probe-*"))


def test_verification_topology_binds_manifest_and_artifact_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")

    snapshot = validate_execution_topology(
        fixture.topology, current_directory=fixture.candidate
    )

    assert snapshot.evidence_manifest_sha256 == (
        fixture.topology.evidence_manifest_sha256
    )
    assert snapshot.evidence_snapshot is not None
    revalidate_evidence_snapshot(snapshot.evidence_snapshot)


def test_cheap_evidence_revalidation_rejects_tree_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    snapshot = validate_execution_topology(
        fixture.topology, current_directory=fixture.candidate
    )
    assert snapshot.evidence_snapshot is not None
    artifact = (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
    )
    write_safe(artifact, '{"gap_notes": [], "notes": [], "status": "fail"}\n')

    with pytest.raises(ValueError, match="tree identity changed"):
        revalidate_evidence_snapshot(snapshot.evidence_snapshot)


def test_cheap_evidence_revalidation_does_not_rehash_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    snapshot = validate_execution_topology(
        fixture.topology, current_directory=fixture.candidate
    )
    assert snapshot.evidence_snapshot is not None

    def reject_hash(*args, **kwargs):
        raise AssertionError("cheap revalidation must not hash artifact contents")

    monkeypatch.setattr(evidence_manifest_module, "_hash_regular_file", reject_hash)

    revalidate_evidence_snapshot(snapshot.evidence_snapshot)


def test_evidence_directory_open_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    run = real / "run"
    run.mkdir(parents=True)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        evidence_manifest_module._open_directory_path(linked / "run")


def test_evidence_directory_identity_detects_path_replacement(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    descriptor = evidence_manifest_module._open_directory_path(evidence)
    moved = tmp_path / "moved-evidence"
    try:
        evidence.rename(moved)
        evidence.mkdir()

        with pytest.raises(ValueError, match="ancestry changed"):
            evidence_manifest_module._require_directory_path_identity(
                evidence, descriptor
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("surface", "mixed", "surface does not match"),
        ("command", ["/usr/bin/false"], "command does not match"),
        (
            "recorded_environment",
            {"LANG": "C.UTF-8"},
            "recorded_environment does not match",
        ),
    ],
)
def test_scenario_id_cannot_substitute_for_frozen_execution_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    raw = json.loads(read_safe(_manifest_path(fixture)).text)
    raw["scenarios"][0][field] = replacement
    payload = (
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    topology = _rebind_manifest(fixture, payload)

    with pytest.raises(ExecutionTopologyError, match=message):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_scenario_must_include_every_control_required_artifact_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    _update_scenario_contract(fixture, required_artifact_types=["log", "result"])
    payload = read_safe(_manifest_path(fixture)).text.encode()
    topology = _rebind_manifest(fixture, payload)

    with pytest.raises(ExecutionTopologyError, match="missing required.*log"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_result_artifact_must_match_frozen_control_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    schema = {
        "properties": {"proof": {"const": "host-observed"}},
        "required": ["proof"],
        "type": "object",
    }
    _update_scenario_contract(fixture, result_schema=schema)
    topology = _rebind_manifest(
        fixture, read_safe(_manifest_path(fixture)).text.encode()
    )

    with pytest.raises(ExecutionTopologyError, match="violates its frozen schema"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_packet_and_status_scenario_contracts_must_match_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    status_path = fixture.control / fixture.topology.status_path
    status = _load_mapping(status_path)
    packets = status["packets"]
    assert isinstance(packets, list)
    entry = packets[0]
    assert isinstance(entry, dict)
    status_evidence = entry["evidence"]
    assert isinstance(status_evidence, dict)
    contracts = status_evidence["scenario_contracts"]
    assert isinstance(contracts, list)
    contract = contracts[0]
    assert isinstance(contract, dict)
    contract["command"] = ["/usr/bin/false"]
    write_safe(status_path, yaml.safe_dump(status))
    topology = _commit_control_paths(fixture, fixture.topology.status_path)

    with pytest.raises(ExecutionTopologyError, match="exactly identical"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_verification_topology_rejects_artifact_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    artifact = (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
    )
    write_safe(artifact, '{"status":"substituted"}\n')

    with pytest.raises(ExecutionTopologyError, match="artifact digest mismatch"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_undeclared_scenario_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    write_safe(
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/omitted.log",
        "omitted\n",
    )

    with pytest.raises(ExecutionTopologyError, match="undeclared files"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_undeclared_nested_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/empty-extra"
    ).mkdir()

    with pytest.raises(ExecutionTopologyError, match="undeclared directories"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_tree_mutation_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    scenario_directory = _manifest_path(fixture).parent / "IT-13"
    original_read = evidence_manifest_module._read_regular_file

    def mutate_tree(root_fd, parts, **kwargs):
        result = original_read(root_fd, parts, **kwargs)
        if parts != ("IT-13", "result.json"):
            return result
        observed = scenario_directory.stat()
        os.utime(
            scenario_directory,
            ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000),
        )
        return result

    monkeypatch.setattr(evidence_manifest_module, "_read_regular_file", mutate_tree)

    with pytest.raises(ExecutionTopologyError, match="tree changed"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_binds_opened_artifact_to_prevalidation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    artifact = _manifest_path(fixture).parent / "IT-13/result.json"
    valid_payload = read_safe(artifact, raise_on_error=True).text
    write_safe(artifact, '{ "gap_notes": [], "notes": [], "status": "pass" }\n')
    substitute = tmp_path / "valid-substitute.json"
    displaced = tmp_path / "displaced-result.json"
    write_safe(substitute, valid_payload)
    original_read = evidence_manifest_module._read_regular_file

    def substitute_during_read(root_fd, parts, **kwargs):
        if parts != ("IT-13", "result.json"):
            return original_read(root_fd, parts, **kwargs)
        os.replace(artifact, displaced)
        os.replace(substitute, artifact)
        try:
            return original_read(root_fd, parts, **kwargs)
        finally:
            os.replace(artifact, substitute)
            os.replace(displaced, artifact)

    monkeypatch.setattr(
        evidence_manifest_module, "_read_regular_file", substitute_during_read
    )

    with pytest.raises(ExecutionTopologyError, match="identity|changed"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_result_artifact_is_not_reopened_after_bound_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    original_hash = evidence_manifest_module._hash_regular_file

    def reject_result_rehash(root_fd, parts, **kwargs):
        if parts == ("IT-13", "result.json"):
            raise AssertionError("result artifact must not be reopened for hashing")
        return original_hash(root_fd, parts, **kwargs)

    monkeypatch.setattr(
        evidence_manifest_module, "_hash_regular_file", reject_result_rehash
    )

    validate_execution_topology(fixture.topology, current_directory=fixture.candidate)


def test_manifest_lock_path_cannot_be_replaced_and_restored_while_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    lock_path = _manifest_path(fixture).parent / ".manifest.lock"
    displaced = tmp_path / "displaced-manifest.lock"
    original_flock = fcntl.flock
    attacked = False

    def replace_lock(descriptor: int, operation: int) -> None:
        nonlocal attacked
        original_flock(descriptor, operation)
        if attacked or not operation & fcntl.LOCK_EX:
            return
        attacked = True
        os.replace(lock_path, displaced)
        write_safe(lock_path, "replacement\n")
        lock_path.unlink()
        os.replace(displaced, lock_path)

    monkeypatch.setattr(fcntl, "flock", replace_lock)

    with pytest.raises(ExecutionTopologyError, match="changed.*manifest.lock"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_excessive_evidence_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    directory = _manifest_path(fixture).parent / "IT-13"
    for index in range(20):
        directory /= f"level-{index}"
        directory.mkdir()

    with pytest.raises(ExecutionTopologyError, match="maximum depth"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_symlinked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    artifact = (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
    )
    target = tmp_path / "outside-result.json"
    write_safe(target, '{"status":"pass"}\n')
    artifact.unlink()
    artifact.symlink_to(target)

    with pytest.raises(ExecutionTopologyError, match="contains a symlink"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_hardlinked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    artifact = (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
    )
    target = tmp_path / "outside-result.json"
    write_safe(target, '{"status":"pass"}\n')
    artifact.unlink()
    os.link(target, artifact)

    with pytest.raises(ExecutionTopologyError, match="must not be hard-linked"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_unfinished_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    write_safe(
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/.reservations/IT-13.json",
        "{}\n",
    )

    with pytest.raises(ExecutionTopologyError, match="reservations"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_busy_manifest_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    lock_path = _manifest_path(fixture).parent / ".manifest.lock"
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr("vibe.core.evidence_manifest._LOCK_TIMEOUT_SECONDS", 0.0)
        with pytest.raises(ExecutionTopologyError, match="writer has not finalized"):
            validate_execution_topology(
                fixture.topology, current_directory=fixture.candidate
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_verification_topology_rejects_undeclared_top_level_scenario_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    extra = _manifest_path(fixture).parent / "IT-99"
    extra.mkdir()
    write_safe(extra / "result.json", "{}\n")

    with pytest.raises(ExecutionTopologyError, match="unexpected entries: IT-99"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_rejects_noncanonical_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    raw = json.loads(read_safe(_manifest_path(fixture)).text)
    compact = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    topology = _rebind_manifest(fixture, compact)

    with pytest.raises(ExecutionTopologyError, match="not canonical"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_verification_topology_rejects_duplicate_manifest_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    payload = (
        read_safe(_manifest_path(fixture))
        .text.encode()
        .replace(b'  "version": 1\n', b'  "version": 1,\n  "version": 1\n')
    )
    topology = _rebind_manifest(fixture, payload)

    with pytest.raises(ExecutionTopologyError, match="duplicate JSON key"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_verification_topology_rejects_wrong_lockfile_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    raw = json.loads(read_safe(_manifest_path(fixture)).text)
    raw["environment"]["uv_lock_sha256"] = "f" * 64
    payload = (
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    topology = _rebind_manifest(fixture, payload)

    with pytest.raises(ExecutionTopologyError, match="uv_lock_sha256"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_passing_evidence_scenario_requires_zero_exit_code() -> None:
    with pytest.raises(ValidationError, match="passing scenario must have exit_code 0"):
        evidence_manifest_module._Scenario.model_validate({
            "id": "IT-13",
            "surface": "non_ui",
            "status": "pass",
            "command": ["pytest"],
            "recorded_environment": {},
            "exit_code": 137,
            "artifacts": [
                {"type": "result", "path": "IT-13/result.json", "sha256": "a" * 64}
            ],
            "metrics": {},
            "notes": [],
            "started_at": "2026-07-13T20:00:00Z",
            "finished_at": "2026-07-13T20:01:00Z",
            "result_path": "IT-13/result.json",
        })


def test_unapproved_failure_scenario_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    raw = json.loads(read_safe(_manifest_path(fixture)).text)
    raw["scenarios"][0]["status"] = "fail"
    raw["scenarios"][0]["exit_code"] = 1
    raw["scenarios"][0]["notes"] = ["documented expected gap"]
    artifact = (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
    )
    result_payload = (
        json.dumps(
            {
                "gap_notes": ["documented expected gap"],
                "notes": ["documented expected gap"],
                "status": "fail",
            },
            sort_keys=True,
        )
        + "\n"
    )
    write_safe(artifact, result_payload)
    raw["scenarios"][0]["artifacts"][0]["sha256"] = hashlib.sha256(
        result_payload.encode()
    ).hexdigest()
    payload = (
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    topology = _rebind_manifest(fixture, payload)

    with pytest.raises(ExecutionTopologyError, match="status does not match"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_control_authorized_failure_and_exact_gap_notes_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    note = "documented expected gap"
    _update_scenario_contract(
        fixture, expected_status="fail", allowed_notes=[note], allowed_gap_notes=[note]
    )
    artifact = (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
    )
    result_payload = (
        json.dumps(
            {"gap_notes": [note], "notes": [note], "status": "fail"}, sort_keys=True
        )
        + "\n"
    )
    write_safe(artifact, result_payload)
    raw = json.loads(read_safe(_manifest_path(fixture)).text)
    raw["scenarios"][0]["status"] = "fail"
    raw["scenarios"][0]["exit_code"] = 1
    raw["scenarios"][0]["notes"] = [note]
    raw["scenarios"][0]["artifacts"][0]["sha256"] = hashlib.sha256(
        result_payload.encode()
    ).hexdigest()
    payload = (
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    topology = _rebind_manifest(fixture, payload)

    snapshot = validate_execution_topology(
        topology, current_directory=fixture.candidate
    )

    assert snapshot.evidence_manifest_sha256 == hashlib.sha256(payload).hexdigest()


def test_failure_gap_notes_must_match_frozen_control_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    note = "documented expected gap"
    _update_scenario_contract(
        fixture, expected_status="fail", allowed_notes=[note], allowed_gap_notes=[note]
    )
    artifact = (
        fixture.evidence
        / ".ai/runs/i00-p01-test/test-evidence/latest/IT-13/result.json"
    )
    result_payload = (
        json.dumps(
            {
                "gap_notes": [note, "unapproved substitute"],
                "notes": [note],
                "status": "fail",
            },
            sort_keys=True,
        )
        + "\n"
    )
    write_safe(artifact, result_payload)
    raw = json.loads(read_safe(_manifest_path(fixture)).text)
    raw["scenarios"][0]["status"] = "fail"
    raw["scenarios"][0]["exit_code"] = 1
    raw["scenarios"][0]["notes"] = [note]
    raw["scenarios"][0]["artifacts"][0]["sha256"] = hashlib.sha256(
        result_payload.encode()
    ).hexdigest()
    payload = (
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    topology = _rebind_manifest(fixture, payload)

    with pytest.raises(ExecutionTopologyError, match="result gap_notes"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_git_probes_strip_all_ambient_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    original_run = subprocess.run
    observed_environments: list[dict[str, str]] = []
    expected_git_environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for name, value in {
        "GIT_DIR": str(tmp_path / "redirected.git"),
        "GIT_WORK_TREE": str(tmp_path / "redirected-worktree"),
        "GIT_INDEX_FILE": str(tmp_path / "redirected-index"),
        "GIT_OBJECT_DIRECTORY": str(tmp_path / "redirected-objects"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(tmp_path / "hooks"),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", str(tmp_path / "hostile-bin"))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "hostile.so"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "hostile-python"))

    def inspect_run(*args, **kwargs):
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        git_environment = {
            key: value
            for key, value in environment.items()
            if key.upper().startswith("GIT_")
        }
        observed_environments.append(git_environment)
        assert git_environment == expected_git_environment
        assert environment["PATH"].startswith("/usr/local/sbin:")
        assert "LD_PRELOAD" not in environment
        assert "PYTHONPATH" not in environment
        command = args[0]
        assert Path(command[0]).is_absolute()
        return original_run(*args, **kwargs)

    monkeypatch.setattr("vibe.core.execution_topology.subprocess.run", inspect_run)

    validate_execution_topology(fixture.topology, current_directory=fixture.candidate)

    assert observed_environments


@pytest.mark.parametrize("path_field", ["packet_path", "status_path"])
def test_control_metadata_is_read_from_control_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_field: str
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    relative_path = cast(str, getattr(fixture.topology, path_field))
    repository = Repo(fixture.control)
    repository.git.update_index("--skip-worktree", relative_path)
    write_safe(fixture.control / relative_path, "not: [valid YAML")
    assert not repository.git.status("--porcelain", "--untracked-files=all")

    snapshot = validate_execution_topology(
        fixture.topology, current_directory=fixture.candidate
    )

    assert snapshot.candidate_head == fixture.topology.baseline_sha


@pytest.mark.parametrize("path_field", ["packet_path", "status_path"])
def test_control_metadata_symlink_is_not_a_regular_control_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_field: str
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    relative_path = cast(str, getattr(fixture.topology, path_field))
    path = fixture.control / relative_path
    target = tmp_path / f"{path.name}.target"
    write_safe(target, read_safe(path, raise_on_error=True).text)
    path.unlink()
    path.symlink_to(target)
    topology = _commit_control_paths(fixture, relative_path)

    with pytest.raises(ExecutionTopologyError, match="regular tracked Git blob"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


@pytest.mark.parametrize("path_field", ["packet_path", "status_path"])
def test_ignored_untracked_control_metadata_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_field: str
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    source_path = cast(str, getattr(fixture.topology, path_field))
    ignored_path = f"docs/design/fork-maintenance/ignored-{Path(source_path).name}"
    write_safe(
        fixture.control / ignored_path,
        read_safe(fixture.control / source_path, raise_on_error=True).text,
    )
    exclude_file = fixture.candidate / ".git" / "info" / "exclude"
    existing_excludes = read_safe(exclude_file, raise_on_error=True).text
    write_safe(exclude_file, f"{existing_excludes}\n/{ignored_path}\n")
    topology = fixture.topology.model_copy(update={path_field: ignored_path})
    assert not Repo(fixture.control).git.status("--porcelain", "--untracked-files=all")

    with pytest.raises(ExecutionTopologyError, match="regular tracked Git blob"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


@pytest.mark.parametrize("path_field", ["packet_path", "status_path"])
def test_missing_control_metadata_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_field: str
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    topology = fixture.topology.model_copy(
        update={path_field: f"docs/design/fork-maintenance/missing-{path_field}"}
    )

    with pytest.raises(ExecutionTopologyError, match="regular tracked Git blob"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


@pytest.mark.parametrize("relation", ["ancestor", "descendant"])
def test_evidence_workspace_cannot_overlap_repository_control_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relation: str
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    if relation == "ancestor":
        evidence = tmp_path
    else:
        evidence = fixture.candidate / "evidence"
        evidence.mkdir()
    topology = fixture.topology.model_copy(update={"evidence_workspace": str(evidence)})

    with pytest.raises(ExecutionTopologyError, match="overlaps repository"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_duplicate_campaign_packet_ids_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    status_path = fixture.control / fixture.topology.status_path
    status = _load_mapping(status_path)
    packets = status["packets"]
    assert isinstance(packets, list)
    assert isinstance(packets[0], dict)
    packets.append(dict(packets[0]))
    write_safe(status_path, yaml.safe_dump(status))
    topology = _commit_control_paths(fixture, fixture.topology.status_path)

    with pytest.raises(ExecutionTopologyError, match="duplicate campaign packet ID"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_duplicate_dependency_ids_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    packet_path = fixture.control / fixture.topology.packet_path
    packet = _load_packet_mapping(packet_path)
    packet["depends_on"] = ["I00-P00", "I00-P00"]
    _write_packet_mapping(packet_path, packet)
    status_path = fixture.control / fixture.topology.status_path
    status = _load_mapping(status_path)
    packets = status["packets"]
    assert isinstance(packets, list)
    assert isinstance(packets[0], dict)
    packets[0]["depends_on"] = ["I00-P00", "I00-P00"]
    status["required_future_packets"] = [{"id": "I00-P00", "state": "complete"}]
    write_safe(status_path, yaml.safe_dump(status))
    topology = _commit_control_paths(
        fixture, fixture.topology.packet_path, fixture.topology.status_path
    )

    with pytest.raises(ExecutionTopologyError, match="duplicate dependency ID"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_unregistered_control_directory_cannot_substitute_for_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    fake_control = tmp_path / "control-copy"
    shutil.copytree(fixture.control, fake_control)
    topology = fixture.topology.model_copy(
        update={"control_worktree": str(fake_control)}
    )

    with pytest.raises(ExecutionTopologyError, match="registered physical"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_system_temporary_directory_cannot_hold_campaign_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "vibe.core.execution_topology._known_volatile_roots", lambda: (tmp_path,)
    )

    with pytest.raises(ExecutionTopologyError, match="must be durable"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_tmpfs_mount_cannot_hold_campaign_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    mountinfo = tmp_path / "tmpfs-mountinfo"
    escaped = str(fixture.evidence).replace(" ", "\\040")
    write_safe(
        mountinfo,
        f"1 0 0:1 / / rw - ext4 /dev/root rw\n"
        f"2 1 0:2 / {escaped} rw - tmpfs tmpfs rw\n",
    )
    monkeypatch.setattr("vibe.core.execution_topology._MOUNTINFO_PATH", mountinfo)

    with pytest.raises(ExecutionTopologyError, match="uses tmpfs"):
        validate_execution_topology(
            fixture.topology, current_directory=fixture.candidate
        )


def test_verification_topology_does_not_run_mutating_durability_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")

    def reject_probe(*args, **kwargs):
        raise AssertionError("verification validation must be read-only")

    monkeypatch.setattr(
        "vibe.core.execution_topology._probe_active_durable_workspace", reject_probe
    )

    validate_execution_topology(fixture.topology, current_directory=fixture.candidate)


def test_active_durability_probe_fsyncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    original_fsync = os.fsync
    fsync_modes: list[int] = []

    def observe_fsync(descriptor: int) -> None:
        fsync_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr("vibe.core.execution_topology.os.fsync", observe_fsync)

    validate_execution_topology(fixture.topology, current_directory=fixture.candidate)

    assert sum(stat.S_ISDIR(mode) for mode in fsync_modes) == 2
    assert sum(stat.S_ISREG(mode) for mode in fsync_modes) == 1


def test_control_metadata_mismatch_stops_before_candidate_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    topology = fixture.topology.model_copy(update={"run_id": "substituted-run"})
    probe_called = False

    def observe_probe(*args, **kwargs):
        nonlocal probe_called
        probe_called = True

    monkeypatch.setattr(
        "vibe.core.execution_topology._probe_active_durable_workspace", observe_probe
    )

    with pytest.raises(ExecutionTopologyError, match="evidence mismatch for run_id"):
        validate_execution_topology(topology, current_directory=fixture.candidate)
    assert not probe_called


def test_control_scenario_assignment_must_match_packet_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    status_path = fixture.control / fixture.topology.status_path
    status = _load_mapping(status_path)
    packets = status["packets"]
    assert isinstance(packets, list)
    entry = packets[0]
    assert isinstance(entry, dict)
    entry["required_scenarios"] = ["IT-99"]
    write_safe(status_path, yaml.safe_dump(status))
    topology = _commit_control_paths(fixture, fixture.topology.status_path)

    with pytest.raises(ExecutionTopologyError, match="scenario assignment"):
        validate_execution_topology(topology, current_directory=fixture.candidate)


def test_verification_topology_requires_frozen_candidate_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValidationError, match="requires candidate_sha"):
        TrustedExecutionTopologyConfig.model_validate(
            fixture.topology.model_dump() | {"state": "verification"}
        )


def test_verification_topology_requires_frozen_manifest_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValidationError, match="requires evidence_manifest_sha256"):
        TrustedExecutionTopologyConfig.model_validate(
            fixture.topology.model_dump()
            | {"state": "verification", "candidate_sha": fixture.topology.baseline_sha}
        )


def test_active_topology_cannot_predeclare_manifest_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValidationError, match="must not predeclare"):
        TrustedExecutionTopologyConfig.model_validate(
            fixture.topology.model_dump() | {"evidence_manifest_sha256": "5" * 64}
        )


def test_manifest_digest_is_bound_into_verification_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    original = VerificationState.from_recipe(_recipe(fixture.topology)).trusted_recipe
    replacement_digest = (
        "0" * 64 if fixture.topology.evidence_manifest_sha256 != "0" * 64 else "1" * 64
    )
    changed_topology = TrustedExecutionTopologyConfig.model_validate(
        fixture.topology.model_dump() | {"evidence_manifest_sha256": replacement_digest}
    )
    changed = VerificationState.from_recipe(_recipe(changed_topology)).trusted_recipe

    assert original is not None
    assert changed is not None
    assert original.configuration_hash != changed.configuration_hash


def test_agent_loop_fails_closed_when_host_topology_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    config = build_test_vibe_config(
        trusted_verification_recipe=_recipe(fixture.topology)
    )

    def fail_validation(*args, **kwargs):
        raise ExecutionTopologyError("physical control worktree is unavailable")

    monkeypatch.setattr(
        "vibe.core.execution_topology.validate_execution_topology", fail_validation
    )
    with pytest.raises(ExecutionTopologyError, match="physical control"):
        build_test_agent_loop(config=config)


@pytest.mark.parametrize(
    "argv",
    [
        ("bash", "-c", "false | tail; echo PASS"),
        ("/bin/sh", "-c", "set +e; false; exit 0"),
        ("uv", "run", "bash", "-c", "pytest | tail"),
        ("env", "bash", "-c", "echo masked"),
    ],
)
def test_trusted_recipe_rejects_shell_wrappers(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="cannot invoke"):
        TrustedVerificationCheckConfig(name="masked", argv=argv)


def test_verification_check_requires_paired_environment_attestation() -> None:
    with pytest.raises(ValidationError, match="configured together"):
        TrustedVerificationCheckConfig(
            name="unpaired",
            argv=(sys.executable, "-c", "print('ok')"),
            environment_attestation_path="/opt/vibe/environment.json",
        )


def test_topology_managed_recipe_requires_environment_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch)
    check = TrustedVerificationCheckConfig(
        name="unattested",
        argv=(sys.executable, "-c", "print('ok')"),
        executable_sha256="a" * 64,
    )

    with pytest.raises(ValidationError, match="require an environment attestation"):
        TrustedVerificationRecipeConfig(
            recipe_version="test-v1",
            task_brief="Test",
            acceptance_contract="Check",
            allowed_paths=("tracked.txt",),
            checks=(check,),
            execution_topology=fixture.topology,
        )


class _VerificationConfig:
    verification_subsystem = True


class _VerificationAgentManager:
    config = _VerificationConfig()


@pytest.mark.asyncio
async def test_verify_work_uses_bound_verification_topology_without_active_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _topology_fixture(tmp_path, monkeypatch, state="verification")
    recipe = _recipe(fixture.topology)
    state = VerificationState.from_recipe(recipe)
    state.receipt_store = VerificationReceiptStore(tmp_path / "receipt-store")
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )
    state.record_verifier_pass(
        VerificationReport(
            verdict=VerificationVerdict.PASS,
            evidence=(
                CommandEvidence(
                    check="independent",
                    command="uv run pytest -q",
                    output="1 passed",
                    result=VerificationVerdict.PASS,
                ),
            ),
        ),
        verified_workspace_fingerprint="candidate",
        verified_base_sha=fixture.topology.baseline_sha,
    )
    config = build_test_vibe_config(trusted_verification_recipe=recipe)
    assert VerifyWork.is_available(config)
    tool = VerifyWork(config_getter=lambda: VerifyWorkConfig(), state=BaseToolState())
    ctx = InvokeContext(
        tool_call_id="verify",
        agent_manager=cast(AgentManager, _VerificationAgentManager()),
        verification_state=state,
    )
    monkeypatch.chdir(fixture.candidate)

    results = [result async for result in tool.run(VerifyWorkArgs(), ctx)]

    assert len(results) == 1
    assert results[0].passed
    assert results[0].candidate_head == fixture.topology.candidate_sha
    assert state.receipt_reference is not None
