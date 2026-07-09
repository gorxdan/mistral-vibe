from __future__ import annotations

import asyncio
import time
from typing import cast

import pytest

from vibe.core.agents.manager import AgentManager
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.builtins.land_work import LandWorkArgs, _require_verification_note
from vibe.core.tools.builtins.task import (
    _maybe_record_verifier_pass,
    _record_background_isolated_verifier_pass,
    _start_verification_attempt,
)
from vibe.core.verification_contract import parse_verification_report
from vibe.core.verification_state import VerificationState
from vibe.core.workflows.runtime import IsolatedResult


class _FakeConfig:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.verification_subsystem = verification_subsystem


class _FakeAgentManager:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.config = _FakeConfig(verification_subsystem)


@pytest.fixture(autouse=True)
def _stable_workspace_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "workspace"
    )


def _ctx(state: VerificationState | None = None, enabled: bool = True) -> InvokeContext:
    return InvokeContext(
        tool_call_id="t1",
        agent_manager=cast(AgentManager, _FakeAgentManager(enabled)),
        verification_state=state,
    )


def _report(verdict: str = "PASS", result: str = "PASS") -> str:
    return (
        "Ran the focused suite.\n\n"
        "### Check: focused tests\n"
        "**Command run:**\n"
        "  uv run pytest tests/tools/test_verification_state.py\n"
        "**Output observed:**\n"
        "  8 passed\n"
        f"**Result: {result}**\n\n"
        f"VERDICT: {verdict}"
    )


def test_no_pass_and_no_note_raises_when_subsystem_on() -> None:
    with pytest.raises(Exception, match="verification_note"):
        _require_verification_note(LandWorkArgs(), _ctx(VerificationState()))


def test_recorded_verifier_pass_satisfies_gate_without_note() -> None:
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_report()))
    _require_verification_note(LandWorkArgs(), _ctx(state))


def test_recorded_pass_is_invalid_after_workspace_changes(monkeypatch) -> None:
    current = "before"
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: current
    )
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_report()))

    assert state.has_pass()

    current = "after"

    assert not state.has_pass()


def test_recorded_pass_fails_closed_without_workspace_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: None
    )
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_report()))

    assert not state.has_pass()


def test_recorded_contract_pass_satisfies_gate_without_note() -> None:
    state = VerificationState()
    state.record_contract_pass("contract ok")
    _require_verification_note(LandWorkArgs(), _ctx(state))


def test_state_rejects_nonpassing_verifier_report() -> None:
    state = VerificationState()
    report = parse_verification_report(_report(verdict="FAIL", result="FAIL"))

    with pytest.raises(ValueError, match="passing verifier report"):
        state.record_verifier_pass(report)


@pytest.mark.asyncio
async def test_background_verifier_records_evidence_backed_pass() -> None:
    state = VerificationState()
    ctx = _ctx(state)

    async def finish() -> IsolatedResult:
        return IsolatedResult(output=_report())

    task = asyncio.create_task(finish())
    task.add_done_callback(
        lambda done: _record_background_isolated_verifier_pass(done, "verifier", ctx)
    )
    await task
    await asyncio.sleep(0)

    assert state.has_pass()


def test_trivial_note_still_accepted() -> None:
    _require_verification_note(
        LandWorkArgs(verification_note="trivial: docs only"),
        _ctx(VerificationState()),
        changed_paths=["docs/guide.md"],
    )


def test_pass_flag_ignored_when_subsystem_off() -> None:
    _require_verification_note(LandWorkArgs(), _ctx(VerificationState(), enabled=False))


def test_state_latest_picks_most_recent() -> None:
    state = VerificationState()
    state.record_contract_pass("first")
    time.sleep(0.001)
    state.record_verifier_pass(parse_verification_report(_report()))
    latest = state.latest()
    assert latest is not None
    assert latest.summary.startswith("VERDICT: PASS")
    assert latest.source == "verifier-subagent"
    assert latest.report is not None


def test_verifier_pass_summary_survives_blank_line_before_verdict() -> None:
    state = VerificationState()
    _maybe_record_verifier_pass("verifier", _report(), _ctx(state))
    assert state.last_verifier_pass is not None
    assert state.last_verifier_pass.summary.startswith("VERDICT: PASS")
    assert state.last_verifier_pass.report is not None


def test_verifier_pass_records_on_evidence_backed_pass() -> None:
    state = VerificationState()
    _maybe_record_verifier_pass("verifier", _report(), _ctx(state))
    assert state.has_pass()


def test_verifier_pass_is_dropped_when_workspace_changes_during_run(
    monkeypatch,
) -> None:
    current = "before"
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.workspace_fingerprint", lambda: current
    )
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: current
    )
    attempt = _start_verification_attempt("verifier")
    current = "after"
    state = VerificationState()

    _maybe_record_verifier_pass("verifier", _report(), _ctx(state), attempt=attempt)

    assert not state.has_pass()


@pytest.mark.parametrize(
    "response",
    [
        "VERDICT: PARTIAL",
        "VERDICT: FAIL",
        "VERDICT: FAILED",
        "NOT VERDICT: PASS",
        "my VERDICT: PASS thing",
        "VERDICT: PASSES",
        "VERDICT: PASSPORT",
        _report(verdict="PARTIAL"),
        _report(verdict="FAIL", result="FAIL"),
        _report() + "\ntrailing prose",
        "",
    ],
)
def test_verifier_pass_rejects_non_pass(response: str) -> None:
    state = VerificationState()
    _maybe_record_verifier_pass("verifier", response, _ctx(state))
    assert not state.has_pass()


def test_verifier_pass_ignored_for_non_verifier_agent() -> None:
    state = VerificationState()
    _maybe_record_verifier_pass("reviewer", _report(), _ctx(state))
    assert not state.has_pass()


def test_verifier_pass_ignored_when_task_did_not_complete() -> None:
    state = VerificationState()
    _maybe_record_verifier_pass("verifier", _report(), _ctx(state), completed=False)
    assert not state.has_pass()


def test_verifier_pass_noop_without_state() -> None:
    ctx = InvokeContext(tool_call_id="t1", verification_state=None)
    _maybe_record_verifier_pass("verifier", _report(), ctx)


def test_land_work_rejects_unstructured_nontrivial_note() -> None:
    with pytest.raises(Exception, match="cannot authorize"):
        _require_verification_note(
            LandWorkArgs(verification_note="verifier VERDICT: PASS - tests green"),
            _ctx(VerificationState()),
        )


def test_land_work_rejects_model_supplied_structured_pass_note() -> None:
    with pytest.raises(Exception, match="cannot authorize"):
        _require_verification_note(
            LandWorkArgs(verification_note=_report()), _ctx(VerificationState())
        )


def test_land_work_rejects_structured_fail_note() -> None:
    with pytest.raises(Exception, match="cannot authorize"):
        _require_verification_note(
            LandWorkArgs(verification_note=_report(verdict="FAIL", result="FAIL")),
            _ctx(VerificationState()),
        )
