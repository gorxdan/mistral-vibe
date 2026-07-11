from __future__ import annotations

from typing import cast

import pytest

from vibe.core.agents.manager import AgentManager
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.land_work import LandWorkArgs, _require_verification_note
from vibe.core.tools.builtins.task import (
    Task,
    TaskArgs,
    TaskResult,
    TaskToolConfig,
    _maybe_record_verifier_pass,
    _start_verification_attempt,
)
from vibe.core.types import AssistantEvent
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
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.workspace_fingerprint", lambda: "workspace"
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
    with pytest.raises(ToolError, match="session-recorded"):
        _require_verification_note(LandWorkArgs(), _ctx(VerificationState()))


def test_recorded_verifier_pass_authorizes_unconfigured_legacy_landing() -> None:
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


def test_recorded_pass_is_invalid_after_landing_base_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.landing_base_sha", lambda: "base-before"
    )
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_report()))

    assert state.has_pass(expected_base_sha="base-before")
    assert not state.has_pass(expected_base_sha="base-after")

    with pytest.raises(ToolError, match="session-recorded"):
        _require_verification_note(
            LandWorkArgs(), _ctx(state), expected_base_sha="base-after"
        )


def test_recorded_pass_fails_closed_without_workspace_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: None
    )
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_report()))

    assert not state.has_pass()


def test_state_rejects_nonpassing_verifier_report() -> None:
    state = VerificationState()
    report = parse_verification_report(_report(verdict="FAIL", result="FAIL"))

    with pytest.raises(ValueError, match="passing verifier report"):
        state.record_verifier_pass(report)


@pytest.mark.asyncio
async def test_background_verifier_records_evidence_backed_pass_for_landing(
    monkeypatch,
) -> None:
    monkeypatch.setattr("vibe.core.verification_state.landing_base_sha", lambda: "base")
    state = VerificationState()
    ctx = _ctx(state)

    async def finish(*args, **kwargs) -> IsolatedResult:
        return IsolatedResult(output=_report())

    monkeypatch.setattr("vibe.core.tools.builtins.task.run_isolated_agent", finish)
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    result = await tool._collect_async_isolated(
        TaskArgs(task="verify", agent="verifier"), ctx, finish(), None, None
    )

    assert result.outcome is not None
    assert result.outcome.summary == "Verifier PASS recorded for the current candidate"
    assert result.outcome.evidence[-1] == (
        "Session verification state recorded the evidence-backed PASS"
    )
    assert state.has_pass()
    _require_verification_note(LandWorkArgs(), ctx, expected_base_sha="base")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "diagnostic"),
    [
        (
            "VERDICT: PASS",
            "Verifier result was not recorded: verification report has no command evidence",
        ),
        (
            _report(verdict="FAIL", result="FAIL"),
            "Verifier did not authorize landing: VERDICT: FAIL",
        ),
    ],
)
async def test_background_verifier_surfaces_rejection_reason(
    monkeypatch, response: str, diagnostic: str
) -> None:
    state = VerificationState()
    ctx = _ctx(state)

    async def finish(*args, **kwargs) -> IsolatedResult:
        return IsolatedResult(output=response)

    monkeypatch.setattr("vibe.core.tools.builtins.task.run_isolated_agent", finish)
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    result = await tool._collect_async_isolated(
        TaskArgs(task="verify", agent="verifier"), ctx, finish(), None, None
    )

    assert result.outcome is not None
    assert result.outcome.retryable
    assert result.outcome.diagnostics == [diagnostic]
    assert not state.has_pass()


@pytest.mark.asyncio
async def test_no_registry_isolated_fallback_preserves_verifier_attempt(
    monkeypatch,
) -> None:
    state = VerificationState()
    ctx = _ctx(state)
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    attempt = _start_verification_attempt("verifier", state)
    captured = []

    async def fallback(args, invoke_ctx, *, verification_attempt=None):
        captured.append(verification_attempt)
        yield TaskResult(response="done", completed=True)

    monkeypatch.setattr(tool, "_run_isolated", fallback)

    results = [
        result
        async for result in tool._run_async_isolated(
            TaskArgs(task="verify", agent="verifier"), ctx, verification_attempt=attempt
        )
    ]

    assert isinstance(results[0], TaskResult)
    assert results[0].response == "done"
    assert captured == [attempt]


def test_trivial_note_still_accepted() -> None:
    _require_verification_note(
        LandWorkArgs(verification_note="trivial: docs only"),
        _ctx(VerificationState()),
        changed_paths=["docs/guide.md"],
    )


def test_pass_flag_ignored_when_subsystem_off() -> None:
    _require_verification_note(LandWorkArgs(), _ctx(VerificationState(), enabled=False))


def test_state_latest_returns_current_verifier_pass() -> None:
    state = VerificationState()
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


def test_new_verifier_attempt_revokes_existing_authorization() -> None:
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_report()))

    _start_verification_attempt("verifier", state)

    assert state.last_verifier_pass is None
    assert not state.has_pass()


def test_superseded_verifier_pass_cannot_restore_authorization() -> None:
    state = VerificationState()
    older = _start_verification_attempt("verifier", state)
    newer = _start_verification_attempt("verifier", state)
    assert older is not None
    assert newer is not None

    newer_diagnostic = _maybe_record_verifier_pass(
        "verifier", _report(verdict="FAIL", result="FAIL"), _ctx(state), attempt=newer
    )
    older_diagnostic = _maybe_record_verifier_pass(
        "verifier", _report(), _ctx(state), attempt=older
    )

    assert newer_diagnostic == "Verifier did not authorize landing: VERDICT: FAIL"
    assert older_diagnostic == (
        "Verifier result was not recorded: verifier attempt was superseded"
    )
    assert not state.has_pass()


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


def test_verifier_pass_is_dropped_when_landing_base_changes_during_run(
    monkeypatch,
) -> None:
    current_base = "base-a"
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.landing_base_sha", lambda: current_base
    )
    attempt = _start_verification_attempt("verifier")
    current_base = "base-b"
    state = VerificationState()

    diagnostic = _maybe_record_verifier_pass(
        "verifier", _report(), _ctx(state), attempt=attempt
    )

    assert diagnostic == (
        "Verifier result was not recorded: landing base changed during verification"
    )
    assert state.last_verifier_pass is None


def test_verifier_pass_records_the_base_captured_at_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.landing_base_sha", lambda: "base-a"
    )
    monkeypatch.setattr(
        "vibe.core.verification_state.landing_base_sha", lambda: "base-b"
    )
    attempt = _start_verification_attempt("verifier")
    state = VerificationState()

    diagnostic = _maybe_record_verifier_pass(
        "verifier", _report(), _ctx(state), attempt=attempt
    )

    assert diagnostic is None
    assert state.last_verifier_pass is not None
    assert state.last_verifier_pass.base_sha == "base-a"


@pytest.mark.asyncio
async def test_in_process_incomplete_verifier_surfaces_landing_diagnostic(
    monkeypatch,
) -> None:
    class _IncompleteVerifierLoop:
        async def act(self, task: str):
            yield AssistantEvent(content=_report(), stopped_by_middleware=True)

        async def aclose(self) -> None:
            return None

    state = VerificationState()
    ctx = _ctx(state)
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    monkeypatch.setattr(
        tool,
        "_build_subagent_loop",
        lambda args, invoke_ctx: (_IncompleteVerifierLoop(), args.prompt),
    )

    result = await tool._run_in_process_collect(
        TaskArgs(task="verify", agent="verifier"),
        ctx,
        verification_attempt=_start_verification_attempt("verifier"),
    )

    assert result.outcome is not None
    assert result.outcome.diagnostics[-1] == (
        "Verifier result was not recorded: task did not complete"
    )
    assert state.last_verifier_pass is None


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
    with pytest.raises(ToolError, match="cannot authorize"):
        _require_verification_note(
            LandWorkArgs(verification_note="verifier VERDICT: PASS - tests green"),
            _ctx(VerificationState()),
        )


def test_land_work_rejects_model_supplied_structured_pass_note() -> None:
    with pytest.raises(ToolError, match="cannot authorize"):
        _require_verification_note(
            LandWorkArgs(verification_note=_report()), _ctx(VerificationState())
        )


def test_pasted_report_is_rejected_even_with_current_legacy_pass() -> None:
    state = VerificationState()
    state.record_verifier_pass(parse_verification_report(_report()))

    with pytest.raises(ToolError, match="cannot authorize"):
        _require_verification_note(
            LandWorkArgs(verification_note=_report()), _ctx(state)
        )


def test_land_work_rejects_structured_fail_note() -> None:
    with pytest.raises(ToolError, match="cannot authorize"):
        _require_verification_note(
            LandWorkArgs(verification_note=_report(verdict="FAIL", result="FAIL")),
            _ctx(VerificationState()),
        )
