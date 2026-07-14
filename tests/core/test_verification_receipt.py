from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import time
from typing import cast

from git import Repo
import psutil
from pydantic import ValidationError
import pytest

from vibe.core._trusted_command import resolve_trusted_system_executable
from vibe.core._trusted_host_runner import BoundedProcessResult, stable_file_sha256
from vibe.core._verification_receipt import (
    OUTPUT_EXCERPT_CHARS,
    ReceiptOutcome,
    VerificationReceiptError,
    VerificationReceiptStore,
    allowed_paths_match,
    capture_repository_state,
    hash_payload,
    validate_receipt,
)
from vibe.core._verification_runner import (
    TrustedCheck,
    _output_assertion_diagnostics,
    run_trusted_verification,
)
from vibe.core.agents.manager import AgentManager
from vibe.core.config._verification_config import (
    TrustedExecutionTopologyConfig,
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.tools.base import InvokeContext, ToolError
from vibe.core.tools.builtins.land_work import LandWorkArgs, _require_verification_note
from vibe.core.tools.sandbox import SandboxSpec, detect_backend
from vibe.core.utils.io import read_safe, write_durable, write_safe
from vibe.core.verification_state import VerificationState, VerifierAttemptDisposition


class _FakeConfig:
    verification_subsystem = True


class _FakeAgentManager:
    config = _FakeConfig()


_CUSTOM_CHECK_SENTINEL = "VIBE_CUSTOM_CHECKS"
_CUSTOM_EXECUTABLE = resolve_trusted_system_executable("python3")
_CUSTOM_EXECUTABLE_SHA256 = stable_file_sha256(_CUSTOM_EXECUTABLE)
_TEST_ENVIRONMENT_ATTESTATION = _CUSTOM_EXECUTABLE
_TEST_ENVIRONMENT_ATTESTATION_SHA256 = _CUSTOM_EXECUTABLE_SHA256


def _custom_python_check(
    name: str, source: str, *arguments: str, timeout_seconds: float = 300
) -> TrustedCheck:
    instrumented = (
        f"{source}\n"
        "import sys as _vibe_custom_sys\n"
        f"print('{_CUSTOM_CHECK_SENTINEL}: 1', file=_vibe_custom_sys.stderr)\n"
    )
    return TrustedCheck(
        name=name,
        argv=(str(_CUSTOM_EXECUTABLE), "-c", instrumented, *arguments),
        timeout_seconds=timeout_seconds,
        executable_sha256=_CUSTOM_EXECUTABLE_SHA256,
        environment_attestation_path=str(_TEST_ENVIRONMENT_ATTESTATION),
        environment_attestation_sha256=_TEST_ENVIRONMENT_ATTESTATION_SHA256,
        required_output_patterns=(_CUSTOM_CHECK_SENTINEL,),
        test_count_pattern=rf"{_CUSTOM_CHECK_SENTINEL}:\s*(?P<count>\d+)",
        minimum_test_count=1,
        custom_runner=True,
    )


@pytest.fixture(autouse=True)
def _receipt_sandbox_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    if detect_backend("auto") == "bwrap":
        return
    monkeypatch.setattr(
        "vibe.core._verification_runner.detect_backend", lambda _override: "bwrap"
    )
    monkeypatch.setattr(
        "vibe.core._verification_runner.build_sandbox_command",
        lambda _spec, _backend: ([], "bwrap", None),
    )


def _ctx(state: VerificationState) -> InvokeContext:
    return InvokeContext(
        tool_call_id="receipt-test",
        agent_manager=cast(AgentManager, _FakeAgentManager()),
        verification_state=state,
    )


def _repo(path: Path) -> tuple[Repo, str]:
    path.mkdir()
    repo = Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")
    write_safe(path / "tracked.txt", "base\n")
    repo.index.add(["tracked.txt"])
    base_sha = repo.index.commit("base").hexsha
    write_safe(path / "tracked.txt", "candidate\n")
    repo.index.add(["tracked.txt"])
    repo.index.commit("candidate")
    return repo, base_sha


def _run(
    repository: Path,
    base_sha: str,
    store: VerificationReceiptStore,
    checks: tuple[TrustedCheck, ...] | None = None,
    allowed_paths: tuple[str, ...] = ("tracked.txt",),
):
    selected_checks = (
        checks
        if checks is not None
        else (
            _custom_python_check(
                name="focused", source="print('all checks passed')", timeout_seconds=10
            ),
        )
    )
    attested_checks = tuple(
        check
        if check.environment_attestation_path is not None
        else check.model_copy(
            update={
                "environment_attestation_path": str(_TEST_ENVIRONMENT_ATTESTATION),
                "environment_attestation_sha256": _TEST_ENVIRONMENT_ATTESTATION_SHA256,
            }
        )
        for check in selected_checks
    )
    return run_trusted_verification(
        attested_checks,
        repository_path=repository,
        base_sha=base_sha,
        task_brief_hash=hash_payload("task brief"),
        recipe_version="test-v1",
        contract_hash=hash_payload("contract"),
        configuration_hash=hash_payload("config"),
        allowed_paths=allowed_paths,
        store=store,
    )


def _validate(receipt, repository: Path, base_sha: str, store):
    return validate_receipt(
        receipt,
        store=store,
        repository_path=repository,
        expected_base_sha=base_sha,
        expected_candidate_head=receipt.repository.candidate_head,
        expected_task_brief_hash=hash_payload("task brief"),
        expected_contract_hash=hash_payload("contract"),
        expected_configuration_hash=hash_payload("config"),
        expected_checks_hash=receipt.checks_hash,
        expected_recipe_version="test-v1",
    )


def _managed_topology() -> TrustedExecutionTopologyConfig:
    return TrustedExecutionTopologyConfig(
        packet_id="I00-P01",
        packet_path="docs/packet.md",
        state="verification",
        control_worktree="/control",
        control_sha="1" * 40,
        candidate_worktree="/candidate",
        candidate_branch="candidate",
        baseline_sha="2" * 40,
        candidate_sha="3" * 40,
        upstream_sha="4" * 40,
        evidence_workspace="/evidence",
        evidence_manifest_sha256="5" * 64,
        run_id="run-1",
        runner_id="runner-1",
    )


def _managed_recipe(
    check: TrustedVerificationCheckConfig,
) -> TrustedVerificationRecipeConfig:
    return TrustedVerificationRecipeConfig(
        recipe_version="managed-v1",
        task_brief="Verify the frozen candidate",
        acceptance_contract="The configured check passes",
        allowed_paths=("tracked.txt",),
        checks=(check,),
        execution_topology=_managed_topology(),
    )


def test_allowed_path_single_segment_glob_does_not_cross_directories() -> None:
    assert allowed_paths_match(["src/feature.py"], ["src/*.py"])
    assert not allowed_paths_match(["src/private/secret.py"], ["src/*.py"])


def test_allowed_file_path_does_not_authorize_a_same_named_directory() -> None:
    assert allowed_paths_match(["scripts/runner.py"], ["scripts/runner.py"])
    assert not allowed_paths_match(
        ["scripts/runner.py/payload.py"], ["scripts/runner.py"]
    )


def test_repository_capture_ignores_ambient_git_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    Repo.init(decoy)
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))
    monkeypatch.setenv("PATH", str(tmp_path / "hostile-bin"))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "hostile.so"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "hostile-python"))

    state = capture_repository_state(repository, base_sha)

    assert state.worktree_root == str(repository.resolve())
    assert state.changed_paths == ("tracked.txt",)


def test_repository_capture_disables_external_diff_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repo, base_sha = _repo(repository)
    marker = tmp_path / "external-diff-ran"
    helper = tmp_path / "external-diff"
    write_safe(helper, f"#!/bin/sh\ntouch {marker}\n")
    helper.chmod(0o755)
    repo.git.config("diff.external", str(helper))
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", str(helper))

    state = capture_repository_state(repository, base_sha)

    assert state.changed_paths == ("tracked.txt",)
    assert not marker.exists()


def test_runner_persists_current_receipt_and_full_output(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")

    receipt = _run(repository, base_sha, store)

    assert receipt.outcome == ReceiptOutcome.PASS
    assert receipt.repository.base_sha == base_sha
    assert receipt.repository.dirty is False
    assert receipt.repository.changed_paths == ("tracked.txt",)
    assert receipt.allowed_paths_passed is True
    assert receipt.started_at <= receipt.completed_at <= receipt.created_at
    assert len(receipt.evidence) == 1
    evidence = receipt.evidence[0]
    assert evidence.argv[0] == str(_CUSTOM_EXECUTABLE)
    assert evidence.cwd == str(repository)
    assert evidence.timeout_seconds == 10
    assert evidence.exit_code == 0
    assert evidence.timed_out is False
    assert evidence.duration_ms >= 0
    assert evidence.stdout_excerpt == "all checks passed\n"
    assert (
        store.load(receipt.repository.repository_identity, receipt.receipt_id)
        == receipt
    )
    assert _validate(receipt, repository, base_sha, store).valid


def test_output_excerpt_is_bounded_while_artifact_stays_complete(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    check = _custom_python_check(
        name="large-output", source="print('x' * 10000)", timeout_seconds=10
    )

    receipt = _run(repository, base_sha, store, (check,))
    evidence = receipt.evidence[0]

    assert len(evidence.stdout_excerpt) <= OUTPUT_EXCERPT_CHARS
    assert "chars omitted" in evidence.stdout_excerpt
    assert store.validate_artifact(evidence) is None


def test_receipt_model_rejects_reversed_timestamps(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    receipt = _run(repository, base_sha, store)
    payload = receipt.model_dump()
    payload["completed_at"] = receipt.started_at - timedelta(seconds=1)

    with pytest.raises(ValidationError, match="completion precedes"):
        type(receipt).model_validate(payload)

    payload = receipt.model_dump()
    payload["created_at"] = receipt.completed_at - timedelta(seconds=1)
    with pytest.raises(ValidationError, match="creation precedes"):
        type(receipt).model_validate(payload)

    payload = receipt.model_dump()
    payload["started_at"] = receipt.started_at.replace(tzinfo=None)
    with pytest.raises(ValidationError, match="include a timezone"):
        type(receipt).model_validate(payload)


@pytest.mark.parametrize("mutation", ["working", "index", "head"])
def test_receipt_invalidates_for_every_candidate_state_change(
    tmp_path: Path, mutation: str
) -> None:
    repository = tmp_path / "repo"
    repo, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    receipt = _run(repository, base_sha, store)
    write_safe(repository / "tracked.txt", f"{mutation}\n")
    if mutation in {"index", "head"}:
        repo.index.add(["tracked.txt"])
    if mutation == "head":
        repo.index.commit("post-verification")

    validation = _validate(receipt, repository, base_sha, store)

    assert not validation.valid
    assert "candidate repository state changed" in validation.summary()


@pytest.mark.parametrize(
    ("field", "expected", "message"),
    [
        ("expected_task_brief_hash", hash_payload("other task"), "task brief"),
        ("expected_contract_hash", hash_payload("other contract"), "contract"),
        (
            "expected_configuration_hash",
            hash_payload("other config"),
            "verification configuration",
        ),
        ("expected_checks_hash", hash_payload("other checks"), "check commands"),
        ("expected_recipe_version", "test-v2", "recipe version"),
    ],
)
def test_receipt_invalidates_when_trusted_inputs_change(
    tmp_path: Path, field: str, expected: str, message: str
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    receipt = _run(repository, base_sha, store)
    kwargs = {
        "store": store,
        "repository_path": repository,
        "expected_base_sha": base_sha,
        field: expected,
    }

    validation = validate_receipt(receipt, **kwargs)

    assert not validation.valid
    assert f"{message} changed" in validation.summary()


def test_receipt_invalidates_when_base_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repo, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    receipt = _run(repository, base_sha, store)

    validation = validate_receipt(
        receipt,
        store=store,
        repository_path=repository,
        expected_base_sha=repo.head.commit.hexsha,
    )

    assert not validation.valid
    assert "base commit changed" in validation.summary()


def test_receipt_scope_includes_both_rename_endpoints(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repo, base_sha = _repo(repository)
    write_safe(repository / "outside.py", "value = 1\n")
    repo.index.add(["outside.py"])
    base_sha = repo.index.commit("add source").hexsha
    (repository / "docs").mkdir()
    repo.git.mv("outside.py", "docs/inside.md")
    repo.index.commit("rename into allowed path")
    store = VerificationReceiptStore(tmp_path / "store")

    receipt = _run(repository, base_sha, store, allowed_paths=("docs/**",))

    assert receipt.repository.changed_paths == ("docs/inside.md", "outside.py")
    assert not receipt.allowed_paths_passed
    assert not receipt.passed


def test_empty_or_failed_check_set_never_produces_pass(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    empty = _run(repository, base_sha, store, ())
    failed = _run(
        repository,
        base_sha,
        store,
        (
            _custom_python_check(
                name="failure", source="raise SystemExit(7)", timeout_seconds=10
            ),
        ),
    )

    assert empty.outcome == ReceiptOutcome.FAIL
    assert failed.outcome == ReceiptOutcome.FAIL
    assert failed.evidence[0].exit_code == 7
    assert not _validate(empty, repository, base_sha, store).valid
    assert not _validate(failed, repository, base_sha, store).valid


def test_trusted_check_requires_configured_output_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    check = TrustedCheck(
        name="dotnet-tests",
        argv=(str(_CUSTOM_EXECUTABLE), "-c", "print('Build succeeded. Total: 0')"),
        required_output_patterns=(r"Build succeeded",),
        forbidden_output_patterns=(r"FAILED",),
        test_count_pattern=r"Total:\s*(?P<count>\d+)",
        minimum_test_count=1,
        executable_sha256=_CUSTOM_EXECUTABLE_SHA256,
        custom_runner=True,
    )

    receipt = _run(repository, base_sha, store, (check,))

    assert not receipt.passed
    assert not receipt.evidence[0].passed
    assert receipt.evidence[0].assertion_diagnostics == (
        "observed test count 0 is below required minimum 1",
    )


def test_trusted_check_accepts_matching_output_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    check = TrustedCheck(
        name="dotnet-tests",
        argv=(str(_CUSTOM_EXECUTABLE), "-c", "print('Passed! Total: 151')"),
        required_output_patterns=(r"Passed!",),
        test_count_pattern=r"Total:\s*(?P<count>\d+)",
        minimum_test_count=1,
        executable_sha256=_CUSTOM_EXECUTABLE_SHA256,
        custom_runner=True,
    )

    receipt = _run(repository, base_sha, store, (check,))

    assert receipt.passed
    assert receipt.evidence[0].assertions_passed


@pytest.mark.parametrize(
    ("output", "passed"),
    [
        ("Build succeeded.\n", False),
        ("Passed! - Failed: 0, Passed: 0, Skipped: 0, Total: 0\n", False),
        (
            "candidate says Total: 151\n"
            "Passed! - Failed: 0, Passed: 0, Skipped: 0, Total: 0\n",
            False,
        ),
        ("Passed! - Failed: 0, Passed: 3, Skipped: 0, Total: 3\n", True),
    ],
)
def test_dotnet_test_has_builtin_nonzero_test_count_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str, passed: bool
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    monkeypatch.setattr(
        "vibe.core._verification_runner.run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            stdout=output.encode(),
            stderr=b"",
            exit_code=0,
            timed_out=False,
            output_limited=False,
        ),
    )

    receipt = _run(
        repository,
        base_sha,
        store,
        (TrustedCheck(name="dotnet-tests", argv=("dotnet", "test")),),
    )

    assert receipt.passed is passed
    assert receipt.evidence[0].assertions_passed is passed


def test_trusted_check_rejects_nonexecuting_mode_even_with_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    monkeypatch.setattr(
        "vibe.core._verification_runner.run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            stdout=b"TestFeature\n",
            stderr=b"",
            exit_code=0,
            timed_out=False,
            output_limited=False,
        ),
    )

    receipt = _run(
        repository,
        base_sha,
        store,
        (TrustedCheck(name="dotnet-list", argv=("dotnet", "test", "--list-tests")),),
    )

    assert not receipt.passed
    assert receipt.evidence[0].assertion_diagnostics == (
        "verification command does not execute checks: dotnet test --list-tests",
    )


def test_trusted_check_rejects_unrecognized_runner_even_with_exit_zero(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")

    receipt = _run(
        repository,
        base_sha,
        store,
        (TrustedCheck(name="no-op", argv=("/usr/bin/true",)),),
    )

    assert not receipt.passed
    assert receipt.evidence[0].exit_code == 0
    assert receipt.evidence[0].assertion_diagnostics == (
        "verification command is not a recognized check runner: /usr/bin/true",
    )


@pytest.mark.parametrize("model", [TrustedCheck, TrustedVerificationCheckConfig])
def test_custom_runner_requires_a_strong_contract(model) -> None:
    with pytest.raises(ValidationError, match="custom runner"):
        model(
            name="custom",
            argv=("/opt/trusted/check",),
            custom_runner=True,
            executable_sha256="a" * 64,
            required_output_patterns=(r"passed",),
        )


@pytest.mark.parametrize("model", [TrustedCheck, TrustedVerificationCheckConfig])
def test_custom_runner_accepts_a_pinned_output_and_count_contract(model) -> None:
    check = model(
        name="custom",
        argv=("/opt/trusted/check",),
        custom_runner=True,
        executable_sha256="a" * 64,
        required_output_patterns=(r"passed",),
        test_count_pattern=r"Total:\s*(?P<count>\d+)",
        minimum_test_count=1,
    )

    assert check.custom_runner


def test_count_assertions_reject_conflicts_and_any_below_minimum() -> None:
    check = TrustedCheck(
        name="counts",
        argv=("pytest",),
        test_count_pattern=r"Total:\s*(?P<count>\d+)",
        minimum_test_count=5,
    )

    diagnostics = _output_assertion_diagnostics(check, b"Total: 8\nTotal: 3\n", b"")

    assert "conflicting test counts observed: 3, 8" in diagnostics
    assert "observed test count 3 is below required minimum 5" in diagnostics


def test_indirect_test_runner_accepts_explicit_positive_count_contract() -> None:
    check = TrustedCheck(
        name="project-tests",
        argv=("npm", "test"),
        test_count_pattern=r"Tests:\s*(?P<count>\d+)",
        minimum_test_count=1,
    )

    diagnostics = _output_assertion_diagnostics(check, b"Tests: 4\n", b"")

    assert diagnostics == ()


def test_output_regex_evaluation_is_killably_bounded() -> None:
    check = TrustedCheck(
        name="catastrophic-regex",
        argv=("pytest",),
        required_output_patterns=(r"(a+)+$",),
    )
    started = time.monotonic()

    diagnostics = _output_assertion_diagnostics(check, b"a" * 32 + b"X", b"")

    assert time.monotonic() - started < 3
    assert diagnostics == ("verification output pattern evaluation timed out",)


@pytest.mark.parametrize("model", [TrustedCheck, TrustedVerificationCheckConfig])
def test_verification_models_report_invalid_output_regex(model) -> None:
    with pytest.raises(ValidationError, match="invalid verification output pattern"):
        model(name="invalid-regex", argv=("pytest",), required_output_patterns=("[",))


def test_non_numeric_test_count_is_failed_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    check = TrustedCheck(
        name="malformed-count",
        argv=("/usr/bin/printf", "Total: many\n"),
        executable_sha256=stable_file_sha256(Path("/usr/bin/printf")),
        required_output_patterns=(r"Total:",),
        test_count_pattern=r"Total:\s*(?P<count>\w+)",
        minimum_test_count=1,
        custom_runner=True,
    )

    receipt = _run(repository, base_sha, store, (check,))

    assert not receipt.passed
    assert receipt.evidence[0].assertion_diagnostics == (
        "test count pattern produced a non-integer count",
    )


@pytest.mark.process_e2e
def test_timeout_is_captured_as_failed_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    timed_out = _run(
        repository,
        base_sha,
        store,
        (
            _custom_python_check(
                name="timeout",
                source="import time; time.sleep(1)",
                timeout_seconds=0.01,
            ),
        ),
    )

    assert not timed_out.passed
    assert timed_out.evidence[0].timed_out
    assert timed_out.evidence[0].exit_code is None
    assert timed_out.evidence[0].duration_ms >= 0


def test_dirty_candidate_never_produces_pass(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    write_safe(repository / "tracked.txt", "dirty candidate\n")

    receipt = _run(repository, base_sha, store)

    assert receipt.repository.dirty
    assert not receipt.passed
    assert "dirty" in _validate(receipt, repository, base_sha, store).summary()


def test_dirty_candidate_check_reads_frozen_head_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    write_safe(repository / "tracked.txt", "dirty candidate\n")
    check = _custom_python_check(
        name="snapshot-read",
        source=(
            "from pathlib import Path; print(Path('tracked.txt').read_text(), end='')"
        ),
    )

    receipt = _run(repository, base_sha, store, (check,))

    assert not receipt.passed
    assert receipt.evidence[0].exit_code == 0
    assert receipt.evidence[0].stdout_excerpt == "candidate\n"


def test_empty_contract_is_rejected_before_command_execution(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    marker = repository / "should-not-run"
    check = _custom_python_check(
        name="marker", source=f"from pathlib import Path; Path({str(marker)!r}).touch()"
    )

    with pytest.raises(
        VerificationReceiptError, match="contract hash must not be empty"
    ):
        run_trusted_verification(
            (check,),
            repository_path=repository,
            base_sha=base_sha,
            task_brief_hash=hash_payload("task"),
            recipe_version="test-v1",
            contract_hash=hash_payload(""),
            configuration_hash=hash_payload("config"),
            allowed_paths=("tracked.txt",),
            store=store,
        )

    assert not marker.exists()


def test_runner_does_not_interpret_shell_metacharacters(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    marker = repository / "injected"
    check = _custom_python_check(
        "argv-only",
        "import sys; print(sys.argv[1])",
        f"safe; touch {marker}",
        timeout_seconds=10,
    )

    receipt = _run(repository, base_sha, store, (check,))

    assert receipt.passed
    assert not marker.exists()
    assert f"safe; touch {marker}" in receipt.evidence[0].stdout_excerpt


@pytest.mark.parametrize(
    "argv",
    [
        ("bash", "-c", "false | tail; echo PASS"),
        ("/bin/sh", "-c", "set +e; false; exit 0"),
        ("uv", "run", "bash", "-c", "pytest | tail"),
        ("env", "-S", "sh -c 'false | tail; echo PASS'"),
        ("uv", "run", "env", "-S", "sh -c 'false | tail; echo PASS'"),
    ],
)
def test_runner_model_rejects_shell_wrapped_checks(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError, match="cannot invoke"):
        TrustedCheck(name="masked", argv=argv)


def test_runner_model_allows_shell_named_data_argument() -> None:
    check = TrustedCheck(name="safe-data", argv=("pytest", "tests/bash"))

    assert check.argv == ("pytest", "tests/bash")


def test_managed_recipe_requires_bound_executable_digest() -> None:
    check = TrustedVerificationCheckConfig(name="tests", argv=("pytest", "-q"))

    with pytest.raises(ValidationError, match="executable_sha256"):
        _managed_recipe(check)


def test_managed_recipe_requires_environment_attestation() -> None:
    check = TrustedVerificationCheckConfig(
        name="tests", argv=("pytest", "-q"), executable_sha256="a" * 64
    )

    with pytest.raises(ValidationError, match="environment attestation"):
        _managed_recipe(check)


@pytest.mark.parametrize("model", [TrustedCheck, TrustedVerificationCheckConfig])
def test_check_rejects_partial_environment_attestation(model) -> None:
    with pytest.raises(ValidationError, match="configured together"):
        model(
            name="tests",
            argv=("pytest", "-q"),
            environment_attestation_path="/opt/vibe/environment.json",
        )


@pytest.mark.parametrize(
    "argv",
    [
        ("uv", "run", "pytest", "-q"),
        ("pre-commit", "run", "--all-files"),
        ("python3", "-m", "pre_commit", "run", "--all-files"),
    ],
)
def test_managed_recipe_rejects_offline_bootstrap_commands(
    argv: tuple[str, ...],
) -> None:
    check = TrustedVerificationCheckConfig(
        name="bootstrap",
        argv=argv,
        executable_sha256="a" * 64,
        environment_attestation_path="/opt/vibe/environment.json",
        environment_attestation_sha256="b" * 64,
    )

    with pytest.raises(ValidationError, match="bootstrap"):
        _managed_recipe(check)


@pytest.mark.parametrize("backend", ["none", "unshare", "sandbox-exec"])
def test_trusted_runner_requires_linux_bubblewrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    marker = repository / "must-not-run"
    monkeypatch.setattr(
        "vibe.core._verification_runner.detect_backend", lambda _override: backend
    )

    receipt = _run(
        repository,
        base_sha,
        store,
        (
            _custom_python_check(
                name="uncontained",
                source=f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ),
        ),
    )

    assert not receipt.passed
    assert receipt.evidence[0].exit_code is None
    assert "Linux bubblewrap" in receipt.evidence[0].stderr_excerpt
    assert not marker.exists()


def test_trusted_runner_rejects_bubblewrap_on_non_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    monkeypatch.setattr(
        "vibe.core._verification_runner.detect_backend", lambda _override: "bwrap"
    )
    monkeypatch.setattr("vibe.core._verification_runner.sys.platform", "darwin")

    receipt = _run(repository, base_sha, store)

    assert not receipt.passed
    assert receipt.evidence[0].exit_code is None
    assert "Linux bubblewrap" in receipt.evidence[0].stderr_excerpt


def test_trusted_runner_rejects_candidate_owned_executable(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repo, base_sha = _repo(repository)
    executable = repository / "candidate-check"
    write_safe(executable, "#!/bin/sh\nprintf 'must not run\\n'\n")
    executable.chmod(0o755)
    repo.index.add([executable.name])
    repo.index.commit("candidate executable")
    store = VerificationReceiptStore(tmp_path / "store")

    receipt = _run(
        repository,
        base_sha,
        store,
        (TrustedCheck(name="candidate-executable", argv=(str(executable),)),),
        allowed_paths=("**",),
    )

    assert not receipt.passed
    assert receipt.evidence[0].exit_code is None
    assert "cannot come from candidate" in receipt.evidence[0].stderr_excerpt
    assert "must not run" not in receipt.evidence[0].stdout_excerpt


def test_trusted_runner_rejects_executable_digest_mismatch(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    check = TrustedCheck(
        name="wrong-executable",
        argv=(str(_CUSTOM_EXECUTABLE), "-c", "print('must not run')"),
        executable_sha256="0" * 64,
    )

    receipt = _run(repository, base_sha, store, (check,))

    assert not receipt.passed
    assert receipt.evidence[0].exit_code is None
    assert "executable SHA-256 mismatch" in receipt.evidence[0].stderr_excerpt
    assert "must not run" not in receipt.evidence[0].stdout_excerpt


@pytest.mark.process_e2e
def test_trusted_runner_bounds_combined_output_and_terminates_child(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    payload_size = 2 * 1024 * 1024
    source = (
        "import os\n"
        f"payload = b'x' * {payload_size}\n"
        "os.write(1, payload)\n"
        "os.write(2, payload)\n"
    )
    check = _custom_python_check(
        name="bounded-output", source=source, timeout_seconds=10
    )

    receipt = _run(repository, base_sha, store, (check,))
    evidence = receipt.evidence[0]

    assert not receipt.passed
    assert not evidence.timed_out
    assert evidence.exit_code is None
    assert "combined output exceeded" in evidence.stderr_excerpt
    assert evidence.output_artifact_size < payload_size * 2


def test_trusted_runner_builds_networkless_read_only_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    captured: list[SandboxSpec] = []
    monkeypatch.setenv("GH_TOKEN", "host-token")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/host-agent.sock")

    def fake_build(spec: SandboxSpec, _backend: str):
        captured.append(spec)
        return [], "bwrap", None

    monkeypatch.setattr(
        "vibe.core._verification_runner.detect_backend", lambda _override: "bwrap"
    )
    monkeypatch.setattr(
        "vibe.core._verification_runner.build_sandbox_command", fake_build
    )

    receipt = _run(repository, base_sha, store)

    assert receipt.passed
    [spec] = captured
    assert not spec.allow_network
    assert spec.protect_git_metadata
    assert repository not in spec.read_roots
    assert repository not in spec.write_roots
    assert spec.cwd in spec.read_roots
    assert spec.cwd in spec.protected_roots
    assert spec.cwd != repository
    assert repository / ".git" / "objects" not in spec.read_roots
    assert not (spec.cwd / ".git").exists()
    assert any(repository.is_relative_to(root) for root in spec.hidden_roots)
    assert any(Path("/run").is_relative_to(root) for root in spec.hidden_roots)
    assert len(spec.write_roots) == 1
    assert spec.write_roots[0] != spec.cwd
    assert Path(spec.env["HOME"]).parent == spec.write_roots[0]
    assert "GH_TOKEN" not in spec.env
    assert "SSH_AUTH_SOCK" not in spec.env


def test_trusted_runner_rejects_temporary_directory_inside_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    marker = repository / "must-not-run"

    def unsafe_run_root() -> Path:
        path = repository / "vibe-trusted-verification-unsafe"
        path.mkdir()
        return path

    monkeypatch.setattr(
        "vibe.core._verification_runner.detect_backend", lambda _override: "bwrap"
    )
    monkeypatch.setattr(
        "vibe.core._trusted_host_runner._create_run_root", unsafe_run_root
    )

    receipt = _run(
        repository,
        base_sha,
        store,
        (
            _custom_python_check(
                name="unsafe-temporary-root",
                source=f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ),
        ),
    )

    assert not receipt.passed
    assert "overlaps candidate or Git metadata" in receipt.evidence[0].stderr_excerpt
    assert not marker.exists()


def test_trusted_runner_blocks_host_candidate_and_git_mutation(tmp_path: Path) -> None:
    if detect_backend("auto") != "bwrap":
        pytest.skip("Linux bubblewrap is unavailable")
    repository = tmp_path / "repo"
    repo, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    host_marker = Path("/var/tmp") / (
        f"vibe-verification-host-{os.getpid()}-{tmp_path.name}"
    )
    git_marker = Path(repo.git_dir) / "verification-mutated"
    host_targets = (host_marker, repository / "tracked.txt", git_marker)
    source = (
        "from pathlib import Path\n"
        "for target in ('tracked.txt', '.git/verification-mutated'):\n"
        "    try:\n"
        "        Path(target).write_text('mutated')\n"
        "    except OSError:\n"
        "        pass\n"
        "    else:\n"
        "        raise SystemExit(2)\n"
        f"for target in {tuple(str(path) for path in host_targets)!r}:\n"
        "    try:\n"
        "        Path(target).write_text('mutated')\n"
        "    except OSError:\n"
        "        pass\n"
    )
    host_marker.unlink(missing_ok=True)

    try:
        receipt = _run(
            repository,
            base_sha,
            store,
            (_custom_python_check(name="read-only-host", source=source),),
        )
        host_mutated = host_marker.exists()
    finally:
        host_marker.unlink(missing_ok=True)

    assert receipt.passed
    assert not host_mutated
    assert not git_marker.exists()
    assert read_safe(repository / "tracked.txt").text == "candidate\n"


def test_trusted_runner_scrubs_credentials_and_uses_disposable_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if detect_backend("auto") != "bwrap":
        pytest.skip("Linux bubblewrap is unavailable")
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    secret = "trusted-runner-must-not-inherit"
    monkeypatch.setenv("TRUSTED_RUNNER_SECRET", secret)
    source = (
        "import os\n"
        "from pathlib import Path\n"
        "assert os.getenv('TRUSTED_RUNNER_SECRET') is None\n"
        "names = ('HOME', 'TMPDIR', 'XDG_CACHE_HOME', 'UV_CACHE_DIR', "
        "'PRE_COMMIT_HOME', 'PIP_CACHE_DIR')\n"
        "assert len({Path(os.environ[name]).parent for name in names}) == 1\n"
        "assert os.environ['GIT_CONFIG_NOSYSTEM'] == '1'\n"
        "assert os.environ['GIT_TERMINAL_PROMPT'] == '0'\n"
        "print('isolated environment')\n"
    )

    receipt = _run(
        repository,
        base_sha,
        store,
        (_custom_python_check(name="clean-environment", source=source),),
    )

    assert receipt.passed
    assert receipt.evidence[0].stdout_excerpt == "isolated environment\n"
    assert secret not in receipt.evidence[0].stdout_excerpt
    assert secret not in receipt.evidence[0].stderr_excerpt


@pytest.mark.process_e2e
def test_trusted_runner_terminates_detached_descendants(tmp_path: Path) -> None:
    if detect_backend("auto") != "bwrap":
        pytest.skip("Linux bubblewrap is unavailable")
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    token = f"vibe-verification-descendant-{os.getpid()}-{tmp_path.name}"
    child = "import time; time.sleep(30)"
    source = (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}, {token!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, start_new_session=True)\n"
    )

    try:
        receipt = _run(
            repository,
            base_sha,
            store,
            (
                _custom_python_check(
                    name="detached-descendant", source=source, timeout_seconds=2
                ),
            ),
        )
        time.sleep(0.1)
        survivors = _processes_with_argument(token)
    finally:
        for process in _processes_with_argument(token):
            try:
                process.kill()
            except psutil.Error:
                pass

    assert receipt.passed
    assert not survivors


def _processes_with_argument(argument: str) -> list[psutil.Process]:
    found: list[psutil.Process] = []
    for process in psutil.process_iter(["cmdline"]):
        try:
            if argument in (process.info["cmdline"] or []):
                found.append(process)
        except (psutil.Error, OSError):
            continue
    return found


def test_escaping_check_cwd_is_recorded_as_failure(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")

    receipt = _run(
        repository,
        base_sha,
        store,
        (TrustedCheck(name="escape", argv=("true",), cwd=".."),),
    )

    assert not receipt.passed
    assert receipt.evidence[0].exit_code is None
    assert "escapes the repository" in receipt.evidence[0].stderr_excerpt


def test_tampered_artifact_and_receipt_fail_validation(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    receipt = _run(repository, base_sha, store)
    artifact = store.root / receipt.evidence[0].output_artifact_path
    write_durable(artifact, b"{}")

    validation = _validate(receipt, repository, base_sha, store)

    assert not validation.valid
    assert "artifact" in validation.summary()

    receipt_path = store.receipt_path(
        receipt.repository.repository_identity, receipt.receipt_id
    )
    write_durable(receipt_path, b"{}")
    with pytest.raises(VerificationReceiptError, match="malformed"):
        store.load(receipt.repository.repository_identity, receipt.receipt_id)


def test_land_work_accepts_current_receipt_and_rejects_unbound_or_stale(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repo, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    receipt = _run(repository, base_sha, store)
    state = VerificationState(receipt_store=store)
    generation = state.begin_verifier_attempt()
    assert state.record_verifier_result(
        generation,
        VerifierAttemptDisposition.PASS,
        "Verifier PASS was recorded for the current candidate.",
    )
    state.record_receipt(receipt, verifier_attempt_generation=generation)
    kwargs = {
        "changed_paths": ["tracked.txt"],
        "repository_path": repository,
        "expected_base_sha": base_sha,
        "expected_candidate_head": repo.head.commit.hexsha,
    }

    assert _require_verification_note(LandWorkArgs(), _ctx(state), **kwargs) == (
        receipt.receipt_id
    )

    resumed = VerificationState(receipt_store=store)
    args = LandWorkArgs(verification_receipt_id=receipt.receipt_id)
    with pytest.raises(ToolError, match="trusted verification state"):
        _require_verification_note(args, _ctx(resumed), **kwargs)

    write_safe(repository / "tracked.txt", "post-verification\n")
    with pytest.raises(ToolError, match="stale or invalid"):
        _require_verification_note(args, _ctx(state), **kwargs)
