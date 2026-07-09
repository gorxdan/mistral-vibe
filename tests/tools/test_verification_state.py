from __future__ import annotations

import time

import pytest

from vibe.core.tools.base import InvokeContext
from vibe.core.tools.builtins.land_work import LandWorkArgs, _require_verification_note
from vibe.core.verification_state import VerificationState


class _FakeConfig:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.verification_subsystem = verification_subsystem


class _FakeAgentManager:
    def __init__(self, verification_subsystem: bool = True) -> None:
        self.config = _FakeConfig(verification_subsystem)


def _ctx(state: VerificationState | None = None, enabled: bool = True) -> InvokeContext:
    return InvokeContext(
        tool_call_id="t1",
        agent_manager=_FakeAgentManager(enabled),  # type: ignore[arg-type]
        verification_state=state,
    )


def test_no_pass_and_no_note_raises_when_subsystem_on() -> None:
    with pytest.raises(Exception, match="verification_note"):
        _require_verification_note(LandWorkArgs(), _ctx(VerificationState()))


def test_recorded_verifier_pass_satisfies_gate_without_note() -> None:
    state = VerificationState()
    state.record_verifier_pass("VERDICT: PASS — tests green")
    _require_verification_note(LandWorkArgs(), _ctx(state))


def test_recorded_contract_pass_satisfies_gate_without_note() -> None:
    state = VerificationState()
    state.record_contract_pass("contract ok")
    _require_verification_note(LandWorkArgs(), _ctx(state))


def test_free_text_note_still_accepted() -> None:
    _require_verification_note(
        LandWorkArgs(verification_note="trivial: docs only"), _ctx(VerificationState())
    )


def test_pass_flag_ignored_when_subsystem_off() -> None:
    _require_verification_note(LandWorkArgs(), _ctx(VerificationState(), enabled=False))


def test_state_latest_picks_most_recent() -> None:
    state = VerificationState()
    state.record_contract_pass("first")
    time.sleep(0.001)
    state.record_verifier_pass("second")
    latest = state.latest()
    assert latest is not None
    assert latest.summary == "second"
    assert latest.source == "verifier-subagent"
