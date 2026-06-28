from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.core.types import AssistantEvent
from vibe.core.workflows import AgentLoopFactory
from vibe.core.workflows.models import (
    AgentResult,
    PhaseReport,
    WorkflowResult,
    WorkflowRun,
    WorkflowStatus,
)
from vibe.core.workflows.runtime import WorkflowRuntime, _strip_code_fences


def _result(
    *,
    return_value: Any,
    agent_results: list[AgentResult],
    status: WorkflowStatus = WorkflowStatus.COMPLETED,
    summary: str = "Workflow completed: 3 agents",
) -> WorkflowResult:
    run = WorkflowRun(
        phases=[PhaseReport(name="default", agent_results=agent_results)], status=status
    )
    return WorkflowResult(return_value=return_value, run=run, summary=summary)


# --- P0: a run where every agent failed schema validation must still deliver
# the agents' recorded outputs. This is exactly the wf-3 failure mode: agents
# raised inside parallel() -> None -> empty return_value, but the work was on
# the run. ---
@pytest.mark.asyncio
async def test_delivery_recovers_failed_agent_outputs_when_return_value_empty() -> None:
    failed = AgentResult(
        prompt="p",
        response='{"findings": [{"severity": "high", "title": "real bug"}]}',
        completed=False,
        error="Schema validation failed after 3 attempts",
        label="core/sec",
    )
    result = _result(return_value={"total_findings": 0}, agent_results=[failed])

    payload = VibeApp._format_workflow_delivery(result)

    assert "total_findings" in payload  # original return_value still present
    assert "Recovered outputs" in payload
    assert "core/sec" in payload
    assert "real bug" in payload  # the recovered work reaches the host
    assert "Schema validation failed" in payload  # the failure reason too


@pytest.mark.asyncio
async def test_delivery_omits_recovery_when_all_agents_completed() -> None:
    ok = AgentResult(prompt="p", response="done", completed=True, label="x/corr")
    result = _result(return_value={"ok": True}, agent_results=[ok])

    payload = VibeApp._format_workflow_delivery(result)

    assert "Recovered outputs" not in payload
    assert '"ok"' in payload


@pytest.mark.asyncio
async def test_delivery_skips_failed_agents_with_no_output() -> None:
    # A cancelled-before-emitting agent recorded an empty response: nothing to
    # recover, so it must not add a noise section.
    empty = AgentResult(
        prompt="p", response="", completed=False, error="cancelled by user"
    )
    result = _result(return_value={"ok": True}, agent_results=[empty])

    payload = VibeApp._format_workflow_delivery(result)

    assert "Recovered outputs" not in payload


# --- _strip_code_fences: the most common cause of spurious schema failure was
# an agent wrapping its JSON in a markdown fence. ---
def test_strip_code_fences_json_tag() -> None:
    assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_plain_fence() -> None:
    assert _strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_passthrough_no_fence() -> None:
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


# --- Regression for the wf-1 root cause: the previous _strip_code_fences only
# stripped a fence when it was the very first token, so any leading prose made
# json.loads fail at char 0 ("Expecting value: line 1 column 1 (char 0)") and
# the run exhausted its schema retries, discarding completed work. All three
# real failures had a prose prefix before the JSON. ---
def test_strip_code_fences_prose_then_fenced_json() -> None:
    text = (
        "Based on my trace of commit 33f894d, here are the findings.\n"
        '```json\n{"a": 1}\n```'
    )
    assert _strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_prose_then_unfenced_json() -> None:
    text = 'I\'ll start by examining the code.\n{"a": 1}\nDone.'
    assert _strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_trailing_prose_after_fenced_json() -> None:
    text = '```json\n{"a": 1}\n```\nLet me know if you need more detail.'
    assert _strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_brace_inside_string_is_not_a_close() -> None:
    text = 'notes\n{"msg": "contains a } char"}'
    assert _strip_code_fences(text) == '{"msg": "contains a } char"}'


def test_strip_code_fences_nested_objects_and_arrays() -> None:
    payload = '{"findings": [{"severity": "high"}, {"severity": "low"}]}'
    assert _strip_code_fences("Preface.\n" + payload + "\nTrailer.") == payload


def test_strip_code_fences_no_json_returns_as_is() -> None:
    # No JSON anywhere: return the original so json.loads raises a real error
    # (the caller surfaces it) instead of the function guessing.
    assert (
        _strip_code_fences("just prose, nothing to parse")
        == "just prose, nothing to parse"
    )


@pytest.mark.asyncio
async def test_fenced_json_response_now_passes_schema() -> None:
    # Regression for the wf-3 root cause: a schema-tagged agent that wraps its
    # output in a code fence must now validate instead of failing all retries.
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    class _Loop:
        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            yield AssistantEvent(
                content='```json\n{"answer": "42"}\n```', message_id="a1"
            )

        class stats:  # type: ignore[no-redef]
            session_prompt_tokens = 10
            session_completion_tokens = 5

    rt = WorkflowRuntime(
        agent_loop_factory=cast(
            AgentLoopFactory, lambda prompt, *, agent, parent_context=None: _Loop()
        )
    )
    parsed = await rt.spawn_agent("test", schema=schema)
    assert parsed == {"answer": "42"}


# --- strip_unknown: agent emits an extra field (e.g. a free-form `confidence`)
# note); the default lenient behavior drops it and returns a schema-shaped dict
# instead of failing or passing the extra through to the host. ---
@pytest.mark.asyncio
async def test_strip_unknown_drops_extra_properties_by_default() -> None:
    from vibe.core.workflows.schema import strip_unknown_properties

    schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                        "title": {"type": "string"},
                    },
                },
            }
        },
    }
    value = {
        "findings": [
            {"severity": "high", "title": "x", "confidence": "confirmed"},
            {"severity": "low", "title": "y"},
        ],
        "cluster": "core",  # unknown at top level
    }
    stripped = strip_unknown_properties(value, schema)
    assert stripped == {
        "findings": [
            {"severity": "high", "title": "x"},
            {"severity": "low", "title": "y"},
        ]
    }


@pytest.mark.asyncio
async def test_spawn_agent_strips_unknown_properties_from_agent_output() -> None:
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    class _Loop:
        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            # Agent adds a free-form rationale the schema does not declare.
            yield AssistantEvent(
                content='{"answer": "42", "confidence": "high"}', message_id="a1"
            )

        class stats:  # type: ignore[no-redef]
            session_prompt_tokens = 10
            session_completion_tokens = 5

    rt = WorkflowRuntime(
        agent_loop_factory=cast(
            AgentLoopFactory, lambda prompt, *, agent, parent_context=None: _Loop()
        )
    )
    parsed = await rt.spawn_agent("test", schema=schema)
    assert parsed == {"answer": "42"}  # confidence dropped, schema-shape only


@pytest.mark.asyncio
async def test_spawn_agent_strip_unknown_can_be_disabled() -> None:
    # Opt out: with strip_unknown=False, extra properties pass through untouched
    # (the validator ignores unknowns rather than rejecting them).
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    class _Loop:
        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            yield AssistantEvent(
                content='{"answer": "42", "confidence": "high"}', message_id="a1"
            )

        class stats:  # type: ignore[no-redef]
            session_prompt_tokens = 10
            session_completion_tokens = 5

    rt = WorkflowRuntime(
        agent_loop_factory=cast(
            AgentLoopFactory, lambda prompt, *, agent, parent_context=None: _Loop()
        )
    )
    parsed = await rt.spawn_agent("test", schema=schema, strip_unknown=False)
    assert parsed == {"answer": "42", "confidence": "high"}


# --- schema_errors: the field-level reasons must be recorded on the AgentResult
# (and surfaced in delivery), not just returned on the SchemaValidationFailure
# object. Previously the launching model only ever saw "Schema validation failed
# after N attempts" with no clue which field was wrong. ---
@pytest.mark.asyncio
async def test_schema_exhaustion_records_field_level_errors_on_agent_result() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    class _Loop:
        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            # Never includes the required `answer` field -> exhausts retries.
            yield AssistantEvent(content='{"wrong": "x"}', message_id="a1")

        class stats:  # type: ignore[no-redef]
            session_prompt_tokens = 10
            session_completion_tokens = 5

    rt = WorkflowRuntime(
        agent_loop_factory=cast(
            AgentLoopFactory, lambda prompt, *, agent, parent_context=None: _Loop()
        ),
        max_agents=10,
    )
    outcome = await rt.spawn_agent("test", schema=schema, label="finder")
    # Non-strict default returns a structured failure rather than raising.
    from vibe.core.workflows.models import SchemaValidationFailure as _SVF

    assert isinstance(outcome, _SVF)
    assert outcome.schema_errors  # the returned object carries detail

    # The recorded AgentResult (what workflow_results / delivery read) must too.
    recorded = [
        ar
        for p in rt._phases.values()
        for ar in p.agent_results
        if ar.label == "finder"
    ]
    assert recorded, "agent was finalized onto a phase"
    assert recorded[0].schema_errors, "field-level errors must survive to the record"
    assert any("answer" in e for e in recorded[0].schema_errors)


@pytest.mark.asyncio
async def test_delivery_surfaces_schema_errors_for_failed_agents() -> None:
    failed = AgentResult(
        prompt="p",
        response='{"wrong": "x"}',
        completed=False,
        error="Schema validation failed after 3 attempts",
        label="finder",
        schema_errors=["$.answer: required property missing"],
    )
    result = _result(return_value={"ok": 0}, agent_results=[failed])

    payload = VibeApp._format_workflow_delivery(result)

    # The field-level reason reaches the push, not just the generic summary.
    assert "$.answer: required property missing" in payload


# --- return_value pull path: _return_value_for_tool is the recovery route when
# the one-shot completion push is missed/truncated. Structured values pass
# through; oversized values truncate unless raw=True. ---
class _FakeEntry:
    def __init__(self, result: WorkflowResult | None) -> None:
        self.result = result


def test_return_value_tool_passes_structured_value_through() -> None:
    result = _result(return_value={"findings": [1, 2, 3]}, agent_results=[])
    value = VibeApp._return_value_for_tool(_FakeEntry(result), raw=False)
    assert value == {"findings": [1, 2, 3]}


def test_return_value_tool_none_when_run_in_flight() -> None:
    # entry.result is None while the run hasn't finished.
    assert VibeApp._return_value_for_tool(_FakeEntry(None), raw=False) is None


def test_return_value_tool_truncates_oversized_unless_raw() -> None:
    big = {"blob": "x" * 20_000}
    result = _result(return_value=big, agent_results=[])

    capped = VibeApp._return_value_for_tool(_FakeEntry(result), raw=False)
    assert isinstance(capped, str)
    assert "truncated" in capped

    full = VibeApp._return_value_for_tool(_FakeEntry(result), raw=True)
    assert full == big  # raw lifts the cap, structured value intact
