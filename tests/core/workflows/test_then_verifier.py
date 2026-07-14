from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from git import Repo
import orjson
import pytest

from vibe.core._workspace_verification import workspace_fingerprint
from vibe.core.candidate_delivery import (
    CandidateDelivery,
    CandidateDeliveryStatus,
    CandidateIntegrationMethod,
)
from vibe.core.tools.base import InvokeContext
from vibe.core.utils.io import write_safe
from vibe.core.verification_contract import (
    parse_verification_report,
    verification_observation_hashes,
)
from vibe.core.verification_state import VerificationState, VerifierAttemptDisposition
from vibe.core.workflows._verified_delivery import VerifiedCandidate
from vibe.core.workflows.runtime import (
    IsolatedResult,
    WorkflowError,
    WorkflowRuntime,
    _record_verifier_pass,
    _run_verifier_in_worktree,
    _VerifierResult,
)
from vibe.core.worktree.ephemeral import create_ephemeral_worktree

pytestmark = pytest.mark.asyncio

_PASS_RESPONSE = (
    "Verification notes.\n\n"
    "### Check: focused tests\n"
    "**Command run:**\n"
    "  uv run pytest -q\n"
    "**Output observed:**\n"
    "  3 passed\n"
    "**Result: PASS**\n\n"
    "VERDICT: PASS"
)

_FAIL_RESPONSE = (
    "Verification notes.\n\n"
    "### Check: focused tests\n"
    "**Command run:**\n"
    "  uv run pytest -q\n"
    "**Output observed:**\n"
    "  1 failed\n"
    "**Result: FAIL**\n\n"
    "VERDICT: FAIL"
)

_PARTIAL_RESPONSE = (
    "Verification notes.\n\n"
    "### Check: focused tests\n"
    "**Command run:**\n"
    "  uv run pytest -q\n"
    "**Output observed:**\n"
    "  2 passed, 1 failed\n"
    "**Result: FAIL**\n\n"
    "VERDICT: PARTIAL"
)

_PASS_EVIDENCE_HASHES = verification_observation_hashes(
    "uv run pytest -q", "3 passed\n", ""
)


def _make_runtime(state: VerificationState | None = None) -> WorkflowRuntime:
    from tests.core.workflows.test_runtime import make_factory

    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    rt.parent_context = InvokeContext(tool_call_id="t1", verification_state=state)
    return rt


class _FakeResult:
    def __init__(self, output: str = "worker output") -> None:
        self.output = output
        self.stats: dict[str, int | list[str]] | None = (
            {"verification_evidence_hashes": list(_PASS_EVIDENCE_HASHES)}
            if output == _PASS_RESPONSE
            else None
        )
        self.worktree_path: str | None = None
        self.branch: str | None = None
        self.wt: Any = None


class _FakeWT:
    def __init__(self, path: str = "/tmp/wt") -> None:
        self.path = Path(path)
        self.branch = "wt-branch"
        self.base = Path("/tmp/base")


def _real_candidate(tmp_path: Path) -> tuple[Path, Repo, Any]:
    repo_root = tmp_path / "repo"
    repo = Repo.init(repo_root)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")
    write_safe(repo_root / "base.txt", "base\n")
    repo.index.add(["base.txt"])
    repo.index.commit("initial")
    wt = create_ephemeral_worktree(repo_root, "worker", base_dir=tmp_path / "worktrees")
    write_safe(wt.path / "worker.txt", "worker\n")
    return repo_root, repo, wt


def _fake_candidate_delivery(
    wt: Any,
    kwargs: dict[str, Any],
    *,
    status: CandidateDeliveryStatus = CandidateDeliveryStatus.LANDED,
) -> CandidateDelivery:
    return CandidateDelivery(
        status=status,
        base_sha=kwargs["expected_parent_sha"],
        candidate_sha=kwargs["expected_candidate_sha"],
        parent_sha_before=kwargs["expected_parent_sha"],
        parent_sha_after=(
            kwargs["expected_candidate_sha"]
            if status is CandidateDeliveryStatus.LANDED
            else kwargs["expected_parent_sha"]
        ),
        branch=getattr(wt, "branch", None),
        worktree_path=str(wt.path),
        integration_method=(
            CandidateIntegrationMethod.FAST_FORWARD
            if status is CandidateDeliveryStatus.LANDED
            else None
        ),
    )


@pytest.fixture
def fake_candidate_binding(monkeypatch: pytest.MonkeyPatch) -> VerifiedCandidate:
    import vibe.core.workflows._verified_delivery as delivery
    import vibe.core.worktree.ephemeral as ephemeral

    candidate = VerifiedCandidate(
        parent_path=Path("/tmp/parent"),
        parent_head="a" * 40,
        parent_workspace_fingerprint="parent-fingerprint",
        candidate_path=Path("/tmp/iso-wt"),
        candidate_head="b" * 40,
        candidate_workspace_fingerprint="candidate-fingerprint",
    )
    monkeypatch.setattr(delivery, "prepare_verified_candidate", lambda wt: candidate)
    monkeypatch.setattr(
        delivery,
        "verified_candidate_diagnostic",
        lambda candidate, *, delivered=False: None,
    )
    monkeypatch.setattr(
        ephemeral,
        "deliver_verified_ephemeral_worktree_result",
        lambda wt, **kwargs: _fake_candidate_delivery(wt, kwargs),
    )
    return candidate


async def test_then_verifier_requires_worktree_isolation() -> None:
    rt = _make_runtime()
    with pytest.raises(WorkflowError, match="requires isolation='worktree'"):
        await rt.spawn_agent("task", agent="worker", then="verifier")


async def test_then_rejects_unknown_value() -> None:
    rt = _make_runtime()
    with pytest.raises(WorkflowError, match="supports only 'verifier'"):
        await rt.spawn_agent(
            "task", agent="worker", isolation="worktree", then="reviewer"
        )


async def test_run_verifier_pass_returns_without_recording_parent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    rt = _make_runtime(state)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output=_PASS_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)
    result = _FakeResult("did the work")
    wt = _FakeWT()

    verifier_result = await _run_verifier_in_worktree(rt, wt, result, None, None)

    assert verifier_result.passed is True
    assert state.last_verifier_pass is None


async def test_delivered_verifier_pass_binds_terminal_generation(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    state = VerificationState()
    runtime = _make_runtime(state)
    generation = state.begin_verifier_attempt()
    monkeypatch.setattr(
        "vibe.core.workflows.runtime.landing_base_sha", lambda: "base-sha"
    )
    result = _VerifierResult(
        report=parse_verification_report(_PASS_RESPONSE),
        base_sha="base-sha",
        generation=generation,
        candidate=fake_candidate_binding,
    )

    assert _record_verifier_pass(runtime, result) is None
    assert state.latest_verifier_attempt is not None
    assert state.latest_verifier_attempt.disposition is VerifierAttemptDisposition.PASS
    assert state.last_verifier_pass is not None
    assert state.last_verifier_pass.verifier_attempt_generation == generation


async def test_run_verifier_fail_blocks_and_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_PASS_RESPONSE))
    rt = _make_runtime(state)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output=_FAIL_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)

    verifier_result = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )

    assert verifier_result.passed is False
    assert verifier_result.error == "verifier returned VERDICT: FAIL"
    assert state.last_verifier_pass is None


async def test_superseded_workflow_verifier_pass_cannot_restore_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    rt = _make_runtime(state)
    responses = iter([_PASS_RESPONSE, _FAIL_RESPONSE])

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output=next(responses))

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)
    older = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )
    newer = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )

    assert older.passed
    assert not newer.passed
    _record_verifier_pass(rt, older)
    assert state.last_verifier_pass is None


async def test_run_verifier_spawn_failure_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    rt = _make_runtime(state)

    async def boom(wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any) -> Any:
        raise RuntimeError("subprocess exploded")

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", boom)

    verifier_result = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )

    assert verifier_result.passed is False
    assert verifier_result.error == "verifier subprocess failed: subprocess exploded"
    assert state.last_verifier_pass is None


async def test_run_verifier_noop_without_state(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = _make_runtime(state=None)

    async def fake_spawn(
        wt: Any, prompt: str, agent: str, max_turns: int, **kw: Any
    ) -> Any:
        return _FakeResult(output=_PASS_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", fake_spawn)

    verifier_result = await _run_verifier_in_worktree(
        rt, _FakeWT(), _FakeResult("work"), None, None
    )

    assert verifier_result.passed is True


async def test_run_verifier_partial_blocks() -> None:
    from unittest.mock import AsyncMock

    state = VerificationState()
    rt = _make_runtime(state)
    rt.parent_context = InvokeContext(tool_call_id="t1", verification_state=state)

    fake_result = _FakeResult(output=_PARTIAL_RESPONSE)
    mock_spawn = AsyncMock(return_value=fake_result)

    import vibe.core.workflows.runtime as rt_mod

    original = rt_mod._spawn_isolated
    rt_mod._spawn_isolated = mock_spawn
    try:
        verifier_result = await _run_verifier_in_worktree(
            rt, _FakeWT(), _FakeResult("work"), None, None
        )
    finally:
        rt_mod._spawn_isolated = original

    assert verifier_result.passed is False
    assert state.last_verifier_pass is None


class _VerdictProc:
    pid = 4243
    returncode = 0

    def __init__(self, out: bytes) -> None:
        self._out = out

    async def communicate(self) -> tuple[bytes, bytes]:
        evidence = orjson.dumps({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "verification_evidence_hashes": list(_PASS_EVIDENCE_HASHES),
        })
        return (self._out, b"__VIBE_WORKFLOW_STATS__" + evidence + b"\n")


class _FakeProc:
    pid = 4242
    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b"worker output", b"")


async def test_default_isolated_executor_standalone_then_verifier_pass_delivers_and_records(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    delivered: list[Any] = []

    def deliver_verified(wt: Any, **kwargs: Any) -> CandidateDelivery:
        delivered.append(wt)
        return _fake_candidate_delivery(wt, kwargs)

    monkeypatch.setattr(
        eph, "deliver_verified_ephemeral_worktree_result", deliver_verified
    )
    original_record = state.record_verifier_pass

    def record_after_delivery(report: Any, **kwargs: Any) -> None:
        assert delivered == [fake_wt]
        original_record(report, **kwargs)

    monkeypatch.setattr(state, "record_verifier_pass", record_after_delivery)

    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(_PASS_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    out, stats, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert out == "worker output"
    assert spawn_count == 2
    assert delivered == [fake_wt]
    assert report is not None
    assert report.passed is True
    assert report.delivered is True
    assert state.last_verifier_pass is not None
    assert state.last_verifier_pass.summary.startswith("VERDICT: PASS")
    assert (
        state.last_verifier_pass.workspace_fingerprint
        == fake_candidate_binding.candidate_workspace_fingerprint
    )
    assert state.latest_verifier_attempt is not None
    assert state.latest_verifier_attempt.disposition is VerifierAttemptDisposition.PASS


async def test_default_isolated_executor_rejects_pass_when_landing_base_moves(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)
    current_base = "base-a"
    monkeypatch.setattr(
        "vibe.core.workflows.runtime.landing_base_sha", lambda: current_base
    )

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    delivered: list[Any] = []
    monkeypatch.setattr(
        eph, "deliver_ephemeral_worktree", lambda wt: delivered.append(wt) or True
    )

    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal current_base, spawn_count
        spawn_count += 1
        if spawn_count == 2:
            current_base = "base-b"
            return _VerdictProc(_PASS_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    _, _, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert delivered == []
    assert report is not None
    assert report.passed is False
    assert report.violations[0].message == (
        "landing base changed before verifier authorization was recorded"
    )
    assert state.last_verifier_pass is None
    assert state.latest_verifier_attempt is not None
    assert (
        state.latest_verifier_attempt.disposition is VerifierAttemptDisposition.INVALID
    )


async def test_then_verifier_rejects_new_attempt_during_delivery(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)
    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)

    rejected: list[str] = []

    def deliver_and_try_to_supersede(wt: Any, **kwargs: Any) -> CandidateDelivery:
        with pytest.raises(RuntimeError) as exc_info:
            state.begin_verifier_attempt()
        rejected.append(str(exc_info.value))
        return _fake_candidate_delivery(wt, kwargs)

    monkeypatch.setattr(
        eph, "deliver_verified_ephemeral_worktree_result", deliver_and_try_to_supersede
    )
    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(_PASS_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    _, _, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert report.passed
    assert report.delivered
    assert rejected == [
        "cannot start a verifier while a verification authorization transaction "
        "is in progress"
    ]
    assert state.last_verifier_pass is not None


async def test_then_verifier_finishes_delivery_before_propagating_cancellation(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)
    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()
    delivered: list[CandidateDelivery] = []

    def controlled_delivery(wt: Any, **kwargs: Any) -> CandidateDelivery:
        result = _fake_candidate_delivery(wt, kwargs)
        delivered.append(result)
        return result

    monkeypatch.setattr(
        eph, "deliver_verified_ephemeral_worktree_result", controlled_delivery
    )

    async def controlled_to_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
        if function is controlled_delivery:
            delivery_started.set()
            await release_delivery.wait()
            return function(*args, **kwargs)
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", controlled_to_thread)
    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(_PASS_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    execution = asyncio.create_task(
        rt._default_isolated_executor("do it", "worker", "lbl", 40, then="verifier")
    )
    await delivery_started.wait()
    execution.cancel()
    await asyncio.sleep(0)
    assert not execution.done()
    release_delivery.set()

    with pytest.raises(asyncio.CancelledError):
        await execution

    assert len(delivered) == 1
    assert state.last_verifier_pass is None
    assert state.latest_verifier_attempt is not None
    assert (
        state.latest_verifier_attempt.disposition is VerifierAttemptDisposition.INVALID
    )


async def test_default_isolated_executor_standalone_then_verifier_fail_blocks(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    removed: list[Any] = []
    monkeypatch.setattr(
        eph,
        "remove_ephemeral_worktree",
        lambda wt, **k: removed.append(k.get("keep_if_changed")),
    )
    delivered: list[Any] = []
    monkeypatch.setattr(
        eph, "deliver_ephemeral_worktree", lambda wt: delivered.append(wt) or True
    )

    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(_FAIL_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    out, stats, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert out == "worker output"
    assert spawn_count == 2
    assert delivered == []
    assert report is None or not report.passed
    assert state.last_verifier_pass is None
    assert state.latest_verifier_attempt is not None
    assert state.latest_verifier_attempt.disposition is VerifierAttemptDisposition.FAIL
    assert removed and removed[0] is True


async def test_then_verifier_surfaces_report_parse_error(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)
    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(b"VERDICT: PASS")
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    _out, _stats, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert not report.passed
    assert report.violations[0].message == (
        "verifier report rejected: verification report has no command evidence"
    )
    assert state.latest_verifier_attempt is not None
    assert (
        state.latest_verifier_attempt.disposition is VerifierAttemptDisposition.INVALID
    )


async def test_then_verifier_delivery_failure_does_not_record_pass(
    monkeypatch: pytest.MonkeyPatch, fake_candidate_binding: VerifiedCandidate
) -> None:
    import vibe.core.worktree.ephemeral as eph

    state = VerificationState()
    rt = _make_runtime(state)
    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    monkeypatch.setattr(
        eph,
        "deliver_verified_ephemeral_worktree_result",
        lambda wt, **kwargs: _fake_candidate_delivery(
            wt, kwargs, status=CandidateDeliveryStatus.PRESERVED
        ),
    )
    spawn_count = 0

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_count
        spawn_count += 1
        if spawn_count == 2:
            return _VerdictProc(_PASS_RESPONSE.encode())
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    _out, _stats, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert report.passed
    assert not report.delivered
    assert state.last_verifier_pass is None
    assert state.latest_verifier_attempt is not None
    assert (
        state.latest_verifier_attempt.disposition is VerifierAttemptDisposition.INVALID
    )


async def test_then_verifier_rejects_unrelated_parent_mutation_during_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _, wt = _real_candidate(tmp_path)

    async def fake_worker(*args: Any, **kwargs: Any) -> IsolatedResult:
        return IsolatedResult(output="worker output", wt=wt)

    async def mutate_parent_then_pass(
        wt: Any, prompt: str, agent: str, max_turns: int, **kwargs: Any
    ) -> _FakeResult:
        write_safe(repo_root / "unrelated.txt", "parent mutation\n")
        return _FakeResult(output=_PASS_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime.run_isolated_agent", fake_worker)
    monkeypatch.setattr(
        "vibe.core.workflows.runtime._spawn_isolated", mutate_parent_then_pass
    )
    monkeypatch.setattr(
        "vibe.core.workflows.runtime.landing_base_sha", lambda: "outer-base"
    )
    state = VerificationState()
    rt = _make_runtime(state)

    _, _, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert not report.passed
    assert not report.delivered
    assert (
        report.violations[0].message == "parent workspace changed during verification"
    )
    assert state.last_verifier_pass is None


async def test_then_verifier_records_exact_delivered_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _, wt = _real_candidate(tmp_path)

    async def fake_worker(*args: Any, **kwargs: Any) -> IsolatedResult:
        return IsolatedResult(output="worker output", wt=wt)

    async def verify_prepared_candidate(
        wt: Any, prompt: str, agent: str, max_turns: int, **kwargs: Any
    ) -> _FakeResult:
        candidate = Repo(wt.path)
        assert not candidate.is_dirty(untracked_files=True)
        assert candidate.head.commit.hexsha != wt.base_sha
        return _FakeResult(output=_PASS_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime.run_isolated_agent", fake_worker)
    monkeypatch.setattr(
        "vibe.core.workflows.runtime._spawn_isolated", verify_prepared_candidate
    )
    monkeypatch.setattr(
        "vibe.core.workflows.runtime.landing_base_sha", lambda: "outer-base"
    )
    state = VerificationState()
    rt = _make_runtime(state)

    _, _, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert report.passed
    assert report.delivered
    assert state.last_verifier_pass is not None
    assert state.last_verifier_pass.workspace_fingerprint == workspace_fingerprint(
        repo_root
    )


async def test_then_verifier_rejects_candidate_mutation_during_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, wt = _real_candidate(tmp_path)

    async def fake_worker(*args: Any, **kwargs: Any) -> IsolatedResult:
        return IsolatedResult(output="worker output", wt=wt)

    async def mutate_candidate_then_pass(
        wt: Any, prompt: str, agent: str, max_turns: int, **kwargs: Any
    ) -> _FakeResult:
        write_safe(wt.path / "test-artifact.txt", "generated\n")
        return _FakeResult(output=_PASS_RESPONSE)

    monkeypatch.setattr("vibe.core.workflows.runtime.run_isolated_agent", fake_worker)
    monkeypatch.setattr(
        "vibe.core.workflows.runtime._spawn_isolated", mutate_candidate_then_pass
    )
    monkeypatch.setattr(
        "vibe.core.workflows.runtime.landing_base_sha", lambda: "outer-base"
    )
    state = VerificationState()
    rt = _make_runtime(state)

    _, _, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert not report.passed
    assert not report.delivered
    assert report.violations[0].message == (
        "candidate workspace changed during verification"
    )
    assert state.last_verifier_pass is None


async def test_then_verifier_rejects_dirty_parent_before_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _, wt = _real_candidate(tmp_path)
    write_safe(repo_root / "unrelated.txt", "already dirty\n")

    async def fake_worker(*args: Any, **kwargs: Any) -> IsolatedResult:
        return IsolatedResult(output="worker output", wt=wt)

    async def verifier_must_not_run(*args: Any, **kwargs: Any) -> _FakeResult:
        raise AssertionError("verifier must not run for an unbound parent workspace")

    monkeypatch.setattr("vibe.core.workflows.runtime.run_isolated_agent", fake_worker)
    monkeypatch.setattr(
        "vibe.core.workflows.runtime._spawn_isolated", verifier_must_not_run
    )
    state = VerificationState()
    rt = _make_runtime(state)

    _, _, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert not report.passed
    assert not report.delivered
    assert report.violations[0].message == (
        "parent workspace was dirty before verification"
    )
    assert state.last_verifier_pass is None


async def test_then_verifier_rejects_post_delivery_workspace_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe.core.worktree.ephemeral as eph

    repo_root, _, wt = _real_candidate(tmp_path)
    deliver = eph.deliver_verified_ephemeral_worktree_result

    async def fake_worker(*args: Any, **kwargs: Any) -> IsolatedResult:
        return IsolatedResult(output="worker output", wt=wt)

    async def verifier_passes(
        wt: Any, prompt: str, agent: str, max_turns: int, **kwargs: Any
    ) -> _FakeResult:
        return _FakeResult(output=_PASS_RESPONSE)

    def deliver_then_contaminate(wt: Any, **kwargs: Any) -> CandidateDelivery:
        delivered = deliver(wt, **kwargs)
        if delivered.accepted:
            write_safe(repo_root / "post-delivery.txt", "contamination\n")
        return delivered

    monkeypatch.setattr("vibe.core.workflows.runtime.run_isolated_agent", fake_worker)
    monkeypatch.setattr("vibe.core.workflows.runtime._spawn_isolated", verifier_passes)
    monkeypatch.setattr(
        eph, "deliver_verified_ephemeral_worktree_result", deliver_then_contaminate
    )
    monkeypatch.setattr(
        "vibe.core.workflows.runtime.landing_base_sha", lambda: "outer-base"
    )
    state = VerificationState()
    rt = _make_runtime(state)

    _, _, report = await rt._default_isolated_executor(
        "do it", "worker", "lbl", 40, then="verifier"
    )

    assert report is not None
    assert not report.passed
    assert report.delivered
    assert report.violations[0].message == (
        "delivered workspace does not match the verified candidate"
    )
    assert state.last_verifier_pass is None
