from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys
from typing import Literal
from unittest.mock import MagicMock

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import (
    TrustedExecutionTopologyConfig,
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.output_formatters import (
    JsonOutputFormatter,
    StreamingJsonOutputFormatter,
)
from vibe.core.types import (
    AssistantEvent,
    FunctionCall,
    ReasoningEvent,
    ToolCall,
    ToolResultEvent,
)
from vibe.core.utils.io import write_safe
from vibe.core.verification_state import (
    VerificationReceiptReference,
    VerificationState,
    VerifierAttemptDisposition,
)


def _read_call(path: Path) -> ToolCall:
    return ToolCall(
        id="read-remediation",
        index=0,
        function=FunctionCall(
            name="read", arguments=json.dumps({"file_path": str(path)})
        ),
    )


def _failed_verifier(loop, disposition: VerifierAttemptDisposition) -> None:
    generation = loop._verification_state.begin_verifier_attempt()
    loop._verification_state.record_verifier_result(
        generation,
        disposition,
        "Verifier result was not recorded: task did not complete",
    )


def _enable_managed_guard(
    loop, tmp_path: Path, *, state: Literal["active", "verification"] = "active"
) -> None:
    for name in ("control", "candidate", "evidence"):
        (tmp_path / name).mkdir()
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="managed-v1",
        task_brief="Implement the managed packet",
        acceptance_contract="The trusted checks pass",
        allowed_paths=("candidate.py",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=(sys.executable, "-c", "print('ok')"),
                executable_sha256="0" * 64,
                environment_attestation_path="/usr/bin/true",
                environment_attestation_sha256="1" * 64,
            ),
        ),
        execution_topology=TrustedExecutionTopologyConfig(
            packet_id="I00-P01",
            packet_path="docs/packet.md",
            state=state,
            control_worktree=str(tmp_path / "control"),
            control_sha="1" * 40,
            candidate_worktree=str(tmp_path / "candidate"),
            candidate_branch="candidate",
            baseline_sha="2" * 40,
            candidate_sha="2" * 40 if state == "verification" else None,
            upstream_sha="3" * 40,
            evidence_workspace=str(tmp_path / "evidence"),
            run_id="managed-run",
            runner_id="managed-runner",
            evidence_manifest_sha256=("4" * 64 if state == "verification" else None),
        ),
    )
    loop._verification_state = VerificationState.from_recipe(recipe)


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_failed_verifier_replaces_model_success_claim_before_emission(
    enable_streaming: bool,
) -> None:
    backend = FakeBackend([
        mock_llm_chunk(content="Everything is verified, complete, and ready.")
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=backend,
        enable_streaming=enable_streaming,
    )
    _failed_verifier(loop, VerifierAttemptDisposition.INVALID)

    events = [event async for event in loop.act("Give me the completion report.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST VERIFICATION STATUS: BLOCKED" in assistant[0].content
    assert "Everything is verified" not in assistant[0].content
    assert "task did not complete" in assistant[0].content
    assert "Everything is verified" not in (loop.messages[-1].content or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "formatter_type", [JsonOutputFormatter, StreamingJsonOutputFormatter]
)
async def test_failed_verifier_withholds_raw_claim_from_message_observers(
    formatter_type: type[JsonOutputFormatter] | type[StreamingJsonOutputFormatter],
) -> None:
    stream = StringIO()
    formatter = formatter_type(stream)
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([
            mock_llm_chunk(content="Everything is verified, complete, and ready.")
        ]),
        message_observer=formatter.on_message_added,
    )
    _failed_verifier(loop, VerifierAttemptDisposition.INVALID)

    _ = [event async for event in loop.act("Give me the completion report.")]
    formatter.finalize()

    output = stream.getvalue()
    assert "HOST VERIFICATION STATUS: BLOCKED" in output
    assert "Everything is verified" not in output


@pytest.mark.asyncio
async def test_partial_verifier_emits_host_partial_status() -> None:
    backend = FakeBackend([mock_llm_chunk(content="This work passed verification.")])
    loop = build_test_agent_loop(config=build_test_vibe_config(), backend=backend)
    _failed_verifier(loop, VerifierAttemptDisposition.PARTIAL)

    events = [event async for event in loop.act("Report status.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST VERIFICATION STATUS: PARTIAL" in assistant[0].content
    assert "passed verification" not in assistant[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_failed_verifier_blocks_repeated_completion_claims_across_user_turns(
    enable_streaming: bool,
) -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="I cannot complete this because verification failed.")],
        [mock_llm_chunk(content="The work is now complete and ready to land.")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=backend,
        enable_streaming=enable_streaming,
    )
    _failed_verifier(loop, VerifierAttemptDisposition.FAIL)

    first = [event async for event in loop.act("Report status.")]
    second = [event async for event in loop.act("Report status again.")]

    first_assistant = [event for event in first if isinstance(event, AssistantEvent)]
    second_assistant = [event for event in second if isinstance(event, AssistantEvent)]
    assert len(first_assistant) == 1
    assert "HOST VERIFICATION STATUS: BLOCKED" in first_assistant[0].content
    assert len(second_assistant) == 1
    assert "HOST VERIFICATION STATUS: BLOCKED" in second_assistant[0].content
    assert "ready to land" not in second_assistant[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_completion_constraint_allows_remediation_tool_calls(
    tmp_path: Path, enable_streaming: bool
) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("inspect me", encoding="utf-8")
    backend = FakeBackend([
        mock_llm_chunk(
            content="I will inspect the failure.", tool_calls=[_read_call(evidence)]
        ),
        mock_llm_chunk(content="The candidate is now verified."),
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read"]),
        backend=backend,
        enable_streaming=enable_streaming,
    )
    _failed_verifier(loop, VerifierAttemptDisposition.FAIL)

    events = [event async for event in loop.act("Fix or report the failure.")]

    assert any(isinstance(event, ToolResultEvent) for event in events)
    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST VERIFICATION STATUS: BLOCKED" in assistant[0].content
    assert "candidate is now verified" not in assistant[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "formatter_type", [JsonOutputFormatter, StreamingJsonOutputFormatter]
)
async def test_constraint_sanitizes_claim_attached_to_remediation_tool_call(
    tmp_path: Path,
    formatter_type: type[JsonOutputFormatter] | type[StreamingJsonOutputFormatter],
) -> None:
    evidence = tmp_path / "evidence.txt"
    write_safe(evidence, "inspect me")
    raw = "Everything is verified and ready to land; I will inspect one file."
    reasoning = "I can publish the successful result before the read."
    stream = StringIO()
    formatter = formatter_type(stream)
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read"]),
        backend=FakeBackend([
            mock_llm_chunk(
                content=raw,
                reasoning_content=reasoning,
                tool_calls=[_read_call(evidence)],
            ),
            mock_llm_chunk(content="The candidate remains verified."),
        ]),
        enable_streaming=True,
        message_observer=formatter.on_message_added,
    )
    _failed_verifier(loop, VerifierAttemptDisposition.FAIL)

    events = [event async for event in loop.act("Fix or report the failure.")]
    formatter.finalize()

    output = stream.getvalue()
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert raw not in output
    assert reasoning not in output
    assert raw not in "\n".join(
        event.content
        for event in events
        if isinstance(event, AssistantEvent | ReasoningEvent)
    )
    assert all(
        raw not in str(message.content)
        and reasoning not in str(message.reasoning_content)
        for message in loop.messages
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_workspace_change_sanitizes_claim_attached_to_tool_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enable_streaming: bool
) -> None:
    calls = 0

    def moving_fingerprint() -> str:
        nonlocal calls
        calls += 1
        return "before" if calls == 1 else "after"

    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", moving_fingerprint
    )
    evidence = tmp_path / "evidence.txt"
    write_safe(evidence, "inspect me")
    raw = "Everything is verified and complete; I will inspect one file."
    reasoning = "I can expose this successful result while the read runs."
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read"]),
        backend=FakeBackend([
            mock_llm_chunk(
                content=raw,
                reasoning_content=reasoning,
                tool_calls=[_read_call(evidence)],
            ),
            mock_llm_chunk(content="The candidate is verified."),
        ]),
        enable_streaming=enable_streaming,
    )

    events = [event async for event in loop.act("Finish the implementation.")]

    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert raw not in "\n".join(
        event.content
        for event in events
        if isinstance(event, AssistantEvent | ReasoningEvent)
    )
    assert all(
        raw not in str(message.content)
        and reasoning not in str(message.reasoning_content)
        for message in loop.messages
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_no_verifier_attempt_preserves_normal_assistant_output(
    enable_streaming: bool,
) -> None:
    backend = FakeBackend([mock_llm_chunk(content="Ordinary answer.")])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=backend,
        enable_streaming=enable_streaming,
    )

    events = [event async for event in loop.act("Answer normally.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert [event.content for event in assistant] == ["Ordinary answer."]


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_workspace_change_requires_verification_before_completion(
    monkeypatch: pytest.MonkeyPatch, enable_streaming: bool
) -> None:
    fingerprint = "before"
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: fingerprint
    )
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([
            mock_llm_chunk(content="P8 Release Acceptance Audit — Complete")
        ]),
        enable_streaming=enable_streaming,
    )
    loop._verification_state.observe_workspace_baseline()
    fingerprint = "after"

    events = [event async for event in loop.act("Finish the implementation.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST VERIFICATION STATUS: UNVERIFIED" in assistant[0].content
    assert "P8 Release Acceptance Audit" not in assistant[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_workspace_change_during_generation_withholds_raw_completion(
    monkeypatch: pytest.MonkeyPatch, enable_streaming: bool
) -> None:
    calls = 0

    def moving_fingerprint() -> str:
        nonlocal calls
        calls += 1
        return "before" if calls == 1 else "after"

    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", moving_fingerprint
    )
    raw = "Everything is verified and complete."
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([
            mock_llm_chunk(
                content=raw, reasoning_content="I can publish the successful result."
            )
        ]),
        enable_streaming=enable_streaming,
    )

    events = [event async for event in loop.act("Finish the implementation.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST VERIFICATION STATUS: UNVERIFIED" in assistant[0].content
    assert raw not in assistant[0].content
    assert not any(isinstance(event, ReasoningEvent) for event in events)
    assert raw not in (loop.messages[-1].content or "")
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "formatter_type", [JsonOutputFormatter, StreamingJsonOutputFormatter]
)
async def test_workspace_change_during_generation_withholds_observer_output(
    monkeypatch: pytest.MonkeyPatch,
    formatter_type: type[JsonOutputFormatter] | type[StreamingJsonOutputFormatter],
) -> None:
    calls = 0

    def moving_fingerprint() -> str:
        nonlocal calls
        calls += 1
        return "before" if calls == 1 else "after"

    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", moving_fingerprint
    )
    stream = StringIO()
    formatter = formatter_type(stream)
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([
            mock_llm_chunk(content="Everything is verified and complete.")
        ]),
        enable_streaming=True,
        message_observer=formatter.on_message_added,
    )

    _ = [event async for event in loop.act("Finish the implementation.")]
    formatter.finalize()

    output = stream.getvalue()
    assert "HOST VERIFICATION STATUS: UNVERIFIED" in output
    assert "Everything is verified and complete" not in output


@pytest.mark.asyncio
async def test_disabled_verification_preserves_progressive_streaming() -> None:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(verification_subsystem=False),
        backend=FakeBackend([
            mock_llm_chunk(content="", reasoning_content="First thought."),
            mock_llm_chunk(content="First answer. "),
            mock_llm_chunk(content="Second answer."),
        ]),
        enable_streaming=True,
    )

    events = [event async for event in loop.act("Answer normally.")]

    visible = [
        (type(event).__name__, event.content)
        for event in events
        if isinstance(event, AssistantEvent | ReasoningEvent)
    ]
    assert visible == [
        ("ReasoningEvent", "First thought."),
        ("AssistantEvent", "First answer. "),
        ("AssistantEvent", "Second answer."),
    ]


@pytest.mark.asyncio
async def test_open_todos_force_partial_handoff() -> None:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([mock_llm_chunk(content="Everything is complete.")]),
    )
    loop._verification_state.record_open_todos(("t8", "t11"))

    events = [event async for event in loop.act("Report status.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST VERIFICATION STATUS: PARTIAL" in assistant[0].content
    assert "t8" in assistant[0].content
    assert "t11" in assistant[0].content
    assert "Everything is complete" not in assistant[0].content


def test_todo_tool_result_updates_completion_ledger() -> None:
    loop = build_test_agent_loop(config=build_test_vibe_config())
    call = MagicMock()
    call.tool_name = "todo"

    loop._observe_verification_tool_result(
        call,
        "success",
        {
            "todos": [
                {"id": "done", "status": "completed"},
                {"id": "active", "status": "in_progress"},
                {"id": "later", "status": "pending"},
                {"id": "cancelled", "status": "cancelled"},
            ]
        },
    )

    assert loop._verification_state.open_todo_ids == ("active", "later")


def test_host_handoff_escapes_control_and_directional_characters() -> None:
    loop = build_test_agent_loop(config=build_test_vibe_config())
    content = "BLOCKED: unsafe\x7f\x9b\u202econtext"

    rendered = loop._host_handoff(
        "HOST STATUS", content, allowed_prefixes=("BLOCKED:",)
    )

    assert "\x7f" not in rendered
    assert "\x9b" not in rendered
    assert "\u202e" not in rendered
    assert r"\x7f\x9b\u202e" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_managed_candidate_cannot_claim_completion_without_verifier_attempt(
    tmp_path: Path, enable_streaming: bool
) -> None:
    backend = FakeBackend([
        mock_llm_chunk(
            content="All checks passed. The candidate is complete and ready for acceptance."
        )
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=backend,
        enable_streaming=enable_streaming,
    )
    _enable_managed_guard(loop, tmp_path)

    events = [event async for event in loop.act("Report the managed packet status.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST ACTIVE-PHASE STATUS: HANDOFF" in assistant[0].content
    assert "All checks passed" not in assistant[0].content
    assert "All checks passed" not in (loop.messages[-1].content or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_managed_candidate_uses_host_typed_handoff_even_for_safe_prose(
    tmp_path: Path, enable_streaming: bool
) -> None:
    content = (
        "READY_FOR_HOST_FREEZE: Edits are prepared; verification and host freeze "
        "remain outstanding."
    )
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([mock_llm_chunk(content=content)]),
        enable_streaming=enable_streaming,
    )
    _enable_managed_guard(loop, tmp_path)

    events = [event async for event in loop.act("Report the managed packet status.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST ACTIVE-PHASE STATUS: HANDOFF" in assistant[0].content
    assert "UNTRUSTED MODEL HANDOFF" in assistant[0].content
    assert f"> {content}" in assistant[0].content


@pytest.mark.asyncio
async def test_managed_active_blocker_reaches_the_operator(tmp_path: Path) -> None:
    content = "BLOCKED: the assigned control worktree is unavailable."
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([mock_llm_chunk(content=content)]),
    )
    _enable_managed_guard(loop, tmp_path)

    events = [event async for event in loop.act("Report the blocker.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST ACTIVE-PHASE STATUS: HANDOFF" in assistant[0].content
    assert f"> {content}" in assistant[0].content


def test_managed_receipt_without_verifier_attempt_cannot_authorize_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = build_test_agent_loop(config=build_test_vibe_config())
    _enable_managed_guard(loop, tmp_path, state="verification")
    monkeypatch.setattr(loop, "_current_trusted_receipt_is_valid", lambda: True)

    assert loop._guard_managed_completion_claims() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_managed_completion_is_revalidated_after_model_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enable_streaming: bool
) -> None:
    content = "The candidate is verified and ready for acceptance."
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([mock_llm_chunk(content=content)]),
        enable_streaming=enable_streaming,
    )
    _enable_managed_guard(loop, tmp_path, state="verification")
    generation = loop._verification_state.begin_verifier_attempt()
    loop._verification_state.record_verifier_result(
        generation,
        VerifierAttemptDisposition.PASS,
        "Verifier PASS was recorded for the current candidate.",
    )
    calls = 0

    def receipt_valid_only_before_generation() -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    monkeypatch.setattr(
        loop, "_current_trusted_receipt_is_valid", receipt_valid_only_before_generation
    )

    events = [event async for event in loop.act("Report final status.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST VERIFICATION STATUS: PARTIAL" in assistant[0].content
    assert content not in assistant[0].content
    assert content not in (loop.messages[-1].content or "")
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_streaming", [False, True])
async def test_managed_completion_emits_once_after_stable_postcheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enable_streaming: bool
) -> None:
    content = "The host-authorized candidate is ready."
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([mock_llm_chunk(content=content)]),
        enable_streaming=enable_streaming,
    )
    _enable_managed_guard(loop, tmp_path, state="verification")
    generation = loop._verification_state.begin_verifier_attempt()
    loop._verification_state.record_verifier_result(
        generation,
        VerifierAttemptDisposition.PASS,
        "Verifier PASS was recorded for the current candidate.",
    )
    monkeypatch.setattr(loop, "_current_trusted_receipt_is_valid", lambda: True)

    events = [event async for event in loop.act("Report final status.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert [event.content for event in assistant] == [content]
    assert loop.messages[-1].content == content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim",
    [
        "All acceptance criteria are satisfied; you can merge this candidate now.",
        "The candidate is good to ship.",
        "The work may be accepted.",
        "Every check is green. Please accept the handoff.",
    ],
)
async def test_managed_handoff_wording_cannot_bypass_host_status(
    tmp_path: Path, claim: str
) -> None:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),
        backend=FakeBackend([mock_llm_chunk(content=claim)]),
    )
    _enable_managed_guard(loop, tmp_path)

    events = [event async for event in loop.act("Report status.")]

    assistant = [event for event in events if isinstance(event, AssistantEvent)]
    assert len(assistant) == 1
    assert "HOST ACTIVE-PHASE STATUS: HANDOFF" in assistant[0].content
    assert claim not in assistant[0].content


def test_control_topology_transition_invalidates_managed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe.core.execution_topology import ExecutionTopologyError

    loop = build_test_agent_loop(config=build_test_vibe_config())
    _enable_managed_guard(loop, tmp_path)
    loop._verification_state.receipt_reference = VerificationReceiptReference(
        receipt_id="a" * 64,
        repository_identity="repo",
        base_sha="2" * 40,
        candidate_head="2" * 40,
        task_brief_hash="b" * 64,
        contract_hash="c" * 64,
        configuration_hash="d" * 64,
        checks_hash="e" * 64,
        recipe_version="managed-v1",
        verifier_attempt_generation=loop._verification_state.verifier_attempt_generation,
    )
    checked = False

    def reject_transition(*args, **kwargs):
        nonlocal checked
        checked = True
        raise ExecutionTopologyError("control state moved")

    monkeypatch.setattr(
        "vibe.core.execution_topology.validate_execution_topology", reject_transition
    )

    assert loop._current_trusted_receipt_is_valid() is False
    assert checked is True


def test_cached_topology_uses_cheap_revalidation_for_completion_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe.core import execution_topology

    loop = build_test_agent_loop(config=build_test_vibe_config())
    _enable_managed_guard(loop, tmp_path, state="verification")
    loop._verification_state.receipt_reference = VerificationReceiptReference(
        receipt_id="a" * 64,
        repository_identity="repo",
        base_sha="2" * 40,
        candidate_head="2" * 40,
        task_brief_hash="b" * 64,
        contract_hash="c" * 64,
        configuration_hash="d" * 64,
        checks_hash="e" * 64,
        recipe_version="managed-v1",
        verifier_attempt_generation=loop._verification_state.verifier_attempt_generation,
    )
    snapshot = MagicMock()
    snapshot.candidate_worktree = tmp_path / "candidate"
    snapshot.candidate_head = "2" * 40
    loop._execution_topology_snapshot = snapshot
    calls = 0

    def revalidate(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1

    def reject_full_validation(*args, **kwargs):
        raise AssertionError("completion guard must not repeat full evidence hashing")

    monkeypatch.setattr(
        execution_topology, "revalidate_execution_topology_snapshot", revalidate
    )
    monkeypatch.setattr(
        execution_topology, "validate_execution_topology", reject_full_validation
    )
    monkeypatch.setattr(
        loop._verification_state, "has_valid_receipt", lambda **kwargs: True
    )

    assert loop._current_trusted_receipt_is_valid() is True
    assert calls == 1
