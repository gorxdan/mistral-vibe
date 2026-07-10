from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
from typing import cast

from git import Repo
from pydantic import ValidationError
import pytest

from vibe.core._verification_receipt import (
    OUTPUT_EXCERPT_CHARS,
    ReceiptOutcome,
    VerificationReceiptError,
    VerificationReceiptStore,
    allowed_paths_match,
    hash_payload,
    validate_receipt,
)
from vibe.core._verification_runner import TrustedCheck, run_trusted_verification
from vibe.core.agents.manager import AgentManager
from vibe.core.tools.base import InvokeContext, ToolError
from vibe.core.tools.builtins.land_work import LandWorkArgs, _require_verification_note
from vibe.core.utils.io import write_durable, write_safe
from vibe.core.verification_state import VerificationState


class _FakeConfig:
    verification_subsystem = True


class _FakeAgentManager:
    config = _FakeConfig()


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
    return run_trusted_verification(
        checks
        if checks is not None
        else (
            TrustedCheck(
                name="focused",
                argv=(sys.executable, "-c", "print('all checks passed')"),
                timeout_seconds=10,
            ),
        ),
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


def test_allowed_path_single_segment_glob_does_not_cross_directories() -> None:
    assert allowed_paths_match(["src/feature.py"], ["src/*.py"])
    assert not allowed_paths_match(["src/private/secret.py"], ["src/*.py"])


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
    assert evidence.argv[0] == sys.executable
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
    check = TrustedCheck(
        name="large-output",
        argv=(sys.executable, "-c", "print('x' * 10000)"),
        timeout_seconds=10,
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
            TrustedCheck(
                name="failure",
                argv=(sys.executable, "-c", "raise SystemExit(7)"),
                timeout_seconds=10,
            ),
        ),
    )

    assert empty.outcome == ReceiptOutcome.FAIL
    assert failed.outcome == ReceiptOutcome.FAIL
    assert failed.evidence[0].exit_code == 7
    assert not _validate(empty, repository, base_sha, store).valid
    assert not _validate(failed, repository, base_sha, store).valid


def test_timeout_is_captured_as_failed_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    timed_out = _run(
        repository,
        base_sha,
        store,
        (
            TrustedCheck(
                name="timeout",
                argv=(sys.executable, "-c", "import time; time.sleep(1)"),
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


def test_empty_contract_is_rejected_before_command_execution(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _, base_sha = _repo(repository)
    store = VerificationReceiptStore(tmp_path / "store")
    marker = repository / "should-not-run"
    check = TrustedCheck(
        name="marker",
        argv=(
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ),
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
    check = TrustedCheck(
        name="argv-only",
        argv=(
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1])",
            f"safe; touch {marker}",
        ),
        timeout_seconds=10,
    )

    receipt = _run(repository, base_sha, store, (check,))

    assert receipt.passed
    assert not marker.exists()
    assert f"safe; touch {marker}" in receipt.evidence[0].stdout_excerpt


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
    state.record_receipt(receipt)
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
