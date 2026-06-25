from __future__ import annotations

from typing import Any

from pydantic import BaseModel as PydBaseModel
import pytest

from vibe.core.nuage.models import (
    CustomTaskCanceled,
    CustomTaskCanceledAttributes,
    CustomTaskCompleted,
    CustomTaskCompletedAttributes,
    CustomTaskFailed,
    CustomTaskFailedAttributes,
    CustomTaskStarted,
    CustomTaskStartedAttributes,
    CustomTaskTimedOut,
    CustomTaskTimedOutAttributes,
    Failure,
    JSONPayload,
    WorkflowExecutionCanceled,
    WorkflowExecutionCanceledAttributes,
    WorkflowExecutionCompleted,
    WorkflowExecutionCompletedAttributes,
    WorkflowExecutionFailed,
    WorkflowExecutionFailedAttributes,
    WorkflowExecutionStatus,
)
from vibe.core.nuage.remote_workflow_event_models import (
    RemoteToolArgs,
    RemoteToolResult,
)
from vibe.core.nuage.remote_workflow_event_translator import (
    RemoteWorkflowEventTranslator,
    _get_value_at_path,
    _remote_tool_class,
    _set_value_at_path,
)
from vibe.core.tools.base import ToolError
from vibe.core.types import AgentStats, AssistantEvent, ReasoningEvent, ToolResultEvent

_EXEC_ID = "session-123"


def _translator(
    available_tools: dict[str, Any] | None = None,
) -> RemoteWorkflowEventTranslator:
    return RemoteWorkflowEventTranslator(
        available_tools=available_tools or {},
        stats=AgentStats(),
        merge_message=lambda _m: None,
    )


def _started(
    task_id: str, task_type: str, payload: dict[str, Any]
) -> CustomTaskStarted:
    return CustomTaskStarted(
        event_id=f"evt-{task_id}-start",
        workflow_exec_id=_EXEC_ID,
        attributes=CustomTaskStartedAttributes(
            custom_task_id=task_id,
            custom_task_type=task_type,
            payload=JSONPayload(value=payload),
        ),
    )


def _completed(
    task_id: str, task_type: str, payload: dict[str, Any]
) -> CustomTaskCompleted:
    return CustomTaskCompleted(
        event_id=f"evt-{task_id}-done",
        workflow_exec_id=_EXEC_ID,
        attributes=CustomTaskCompletedAttributes(
            custom_task_id=task_id,
            custom_task_type=task_type,
            payload=JSONPayload(value=payload),
        ),
    )


def _failed(task_id: str, task_type: str, message: str) -> CustomTaskFailed:
    return CustomTaskFailed(
        event_id=f"evt-{task_id}-fail",
        workflow_exec_id=_EXEC_ID,
        attributes=CustomTaskFailedAttributes(
            custom_task_id=task_id,
            custom_task_type=task_type,
            failure=Failure(message=message),
        ),
    )


def _timed_out(
    task_id: str, task_type: str, timeout_type: str | None = None
) -> CustomTaskTimedOut:
    return CustomTaskTimedOut(
        event_id=f"evt-{task_id}-timeout",
        workflow_exec_id=_EXEC_ID,
        attributes=CustomTaskTimedOutAttributes(
            custom_task_id=task_id,
            custom_task_type=task_type,
            timeout_type=timeout_type,
        ),
    )


def _canceled(task_id: str, task_type: str, reason: str = "") -> CustomTaskCanceled:
    return CustomTaskCanceled(
        event_id=f"evt-{task_id}-cancel",
        workflow_exec_id=_EXEC_ID,
        attributes=CustomTaskCanceledAttributes(
            custom_task_id=task_id, custom_task_type=task_type, reason=reason
        ),
    )


def _workflow_completed() -> WorkflowExecutionCompleted:
    return WorkflowExecutionCompleted(
        event_id="wf-done",
        workflow_exec_id=_EXEC_ID,
        attributes=WorkflowExecutionCompletedAttributes(),
    )


def _workflow_failed(message: str = "boom") -> WorkflowExecutionFailed:
    return WorkflowExecutionFailed(
        event_id="wf-fail",
        workflow_exec_id=_EXEC_ID,
        attributes=WorkflowExecutionFailedAttributes(failure=Failure(message=message)),
    )


def _workflow_canceled(reason: str | None = None) -> WorkflowExecutionCanceled:
    return WorkflowExecutionCanceled(
        event_id="wf-cancel",
        workflow_exec_id=_EXEC_ID,
        attributes=WorkflowExecutionCanceledAttributes(reason=reason),
    )


# --------------------------------------------------------------------------- #
# Path helpers (pure functions)                                               #
# --------------------------------------------------------------------------- #


def test_get_value_at_path_root_and_traversal() -> None:
    obj = {"a": {"b": 1}, "l": [10, 20]}
    assert _get_value_at_path("/", obj) is obj
    assert _get_value_at_path("/a/b", obj) == 1
    assert _get_value_at_path("/l/1", obj) == 20


def test_get_value_at_path_none_mid_path_returns_none() -> None:
    assert _get_value_at_path("/a/b", {"a": None}) is None


def test_get_value_at_path_list_index_miss_and_bad_index_return_none() -> None:
    obj = {"l": [1, 2]}
    assert _get_value_at_path("/l/9", obj) is None
    assert _get_value_at_path("/l/x", obj) is None


def test_get_value_at_path_missing_key_and_scalar_return_none() -> None:
    assert _get_value_at_path("/missing", {"a": 1}) is None
    assert _get_value_at_path("/a/b", 5) is None


def test_set_value_at_path_root_is_noop() -> None:
    obj: dict[str, Any] = {"a": 1}
    _set_value_at_path("/", obj, 9)
    assert obj == {"a": 1}


def test_set_value_at_path_dict_and_list_writes() -> None:
    obj: dict[str, Any] = {"a": {"b": 1}, "l": [1, 2]}
    _set_value_at_path("/a/b", obj, 2)
    _set_value_at_path("/l/0", obj, 9)
    assert obj == {"a": {"b": 2}, "l": [9, 2]}


def test_set_value_at_path_bad_mid_path_and_bad_index_swallowed() -> None:
    obj: dict[str, Any] = {"a": 1, "l": [1]}
    _set_value_at_path("/x/b", obj, 2)  # mid-path scalar -> no-op
    _set_value_at_path("/l/9", obj, 5)  # IndexError swallowed
    assert obj == {"a": 1, "l": [1]}


# --------------------------------------------------------------------------- #
# _RemoteTool classmethods + run                                              #
# --------------------------------------------------------------------------- #


def test_remote_tool_classmethods_use_remote_name() -> None:
    cls = _remote_tool_class("my.tool")
    assert cls.get_name() == "my.tool"
    assert cls.get_status_text() == "Running my.tool"


def test_remote_tool_format_call_display_falls_back_to_name() -> None:
    cls = _remote_tool_class("ns.thing")
    assert cls.format_call_display(RemoteToolArgs()).summary == "ns.thing"
    assert cls.format_call_display(RemoteToolArgs(summary="x")).summary == "x"


def test_remote_tool_result_display_branches() -> None:
    cls = _remote_tool_class("ns.thing")
    err = ToolResultEvent(
        tool_name="ns.thing", tool_class=cls, tool_call_id="c", error="boom"
    )
    assert cls.get_result_display(err).success is False
    ok = ToolResultEvent(
        tool_name="ns.thing",
        tool_class=cls,
        tool_call_id="c",
        result=RemoteToolResult(message="done"),
    )
    disp = cls.get_result_display(ok)
    assert disp.success is True and disp.message == "done"
    foreign = ToolResultEvent(
        tool_name="ns.thing",
        tool_class=cls,
        tool_call_id="c",
        result=RemoteToolArgs(summary="not a RemoteToolResult"),
    )
    assert cls.get_result_display(foreign).message == "ns.thing"


@pytest.mark.asyncio
async def test_remote_tool_run_raises_tool_error() -> None:
    cls = _remote_tool_class("ns.thing")
    tool = cls(config_getter=lambda: None, state=None)  # type: ignore[arg-type]
    with pytest.raises(ToolError, match="cannot be invoked locally"):
        async for _ in tool.run(RemoteToolArgs(), None):
            pass


# --------------------------------------------------------------------------- #
# Workflow lifecycle + idle boundary + flush                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "event_factory, expected",
    [
        (_workflow_completed, WorkflowExecutionStatus.COMPLETED),
        (_workflow_failed, WorkflowExecutionStatus.FAILED),
        (lambda: _workflow_canceled("done"), WorkflowExecutionStatus.CANCELED),
    ],
)
def test_lifecycle_events_set_status_and_clear_pending(
    event_factory: Any, expected: WorkflowExecutionStatus
) -> None:
    t = _translator()
    t.pending_input_request = object()  # type: ignore[assignment]
    assert t.consume_workflow_event(event_factory()) == []
    assert t.last_status is expected
    assert t.pending_input_request is None


def test_is_idle_boundary_for_terminals_and_wait_for_input() -> None:
    t = _translator()
    assert t.is_idle_boundary(_workflow_completed()) is True
    assert t.is_idle_boundary(_workflow_failed()) is True
    assert t.is_idle_boundary(_workflow_canceled()) is True
    assert t.is_idle_boundary(_started("w", "wait_for_input", {"label": "x"})) is True


def test_is_idle_boundary_false_for_unrelated_started() -> None:
    t = _translator()
    assert t.is_idle_boundary(_started("w", "working", {"title": "t"})) is False


def test_is_idle_boundary_false_when_tool_call_open() -> None:
    t = _translator()
    t._open_tool_calls["c1"] = "todo"  # type: ignore[attr-defined]
    t._task_state["i"] = {}  # no input yet
    assert t.is_idle_boundary(_completed("i", "AgentInputState", {})) is False


def test_flush_open_tool_calls_emits_and_clears() -> None:
    t = _translator()
    t._open_tool_calls["c1"] = "todo"  # type: ignore[attr-defined]
    events = t.flush_open_tool_calls()
    assert len(events) == 1
    assert isinstance(events[0], ToolResultEvent)
    assert events[0].tool_call_id == "c1"
    assert t._open_tool_calls == {}


# --------------------------------------------------------------------------- #
# Normalize / json-safe helpers                                               #
# --------------------------------------------------------------------------- #


def test_normalize_mapping_parses_json_string_and_rejects_bad() -> None:
    t = _translator()
    assert t._normalize_mapping('{"a": 1}') == {"a": 1}
    assert t._normalize_mapping("not json") == {}
    assert t._normalize_mapping("[1, 2]") == {}
    assert t._normalize_mapping(123) == {}


def test_normalize_output_maps_strings_and_scalars() -> None:
    t = _translator()
    assert t._normalize_output('{"a": 1}') == {"a": 1}
    assert t._normalize_output("[1, 2]") == {"value": [1, 2]}
    assert t._normalize_output("raw") == {"value": "raw"}
    assert t._normalize_output({"k": 1}) == {"k": 1}


def test_output_preview_text_known_key_and_join() -> None:
    t = _translator()
    assert t._output_preview_text({"message": "hi"}) == "hi"
    assert t._output_preview_text({"a": 1, "b": 2}) == "a: 1\nb: 2"


def test_json_safe_value_handles_set_and_basemodel() -> None:
    class _M(PydBaseModel):
        x: int

    t = _translator()
    assert t._json_safe_value({1, 3, 2}) == [1, 2, 3]
    assert t._json_safe_value(_M(x=5)) == {"x": 5}


def test_extract_user_text_variants() -> None:
    t = _translator()
    assert t._extract_user_text("hi") == "hi"
    assert t._extract_user_text(123) is None
    assert t._extract_user_text([]) is None
    assert t._extract_user_text([{"text": "a"}, {"text": "b"}]) == "ab"


def test_extract_predefined_answers_rejects_non_dict_and_missing_message() -> None:
    t = _translator()
    assert t._extract_predefined_answers("x") is None
    assert t._extract_predefined_answers({"input_schema": {"properties": {}}}) is None


# --------------------------------------------------------------------------- #
# AgentCompletionState deltas (incl. divergence reset)                        #
# --------------------------------------------------------------------------- #


def test_completion_events_content_and_reasoning_delta() -> None:
    t = _translator()
    started = _started(
        "c1", "AgentCompletionState", {"content": "Hello", "reasoning_content": "think"}
    )
    events = t.consume_workflow_event(started)
    types = [type(e) for e in events]
    assert ReasoningEvent in types
    assert AssistantEvent in types


def test_completion_events_divergence_resets_previous() -> None:
    t = _translator()
    t.consume_workflow_event(_started("c1", "AgentCompletionState", {"content": "AB"}))
    events = t.consume_workflow_event(
        _completed("c1", "AgentCompletionState", {"content": "XY"})
    )
    assert any(isinstance(e, AssistantEvent) and e.content == "XY" for e in events)


def test_completion_events_no_delta_returns_empty() -> None:
    t = _translator()
    t.consume_workflow_event(
        _started("c1", "AgentCompletionState", {"content": "same"})
    )
    assert (
        t.consume_workflow_event(
            _completed("c1", "AgentCompletionState", {"content": "same"})
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Tool terminal error variants + missing/error output                         #
# --------------------------------------------------------------------------- #


def _seed_tool_call(t: RemoteWorkflowEventTranslator, task_id: str, name: str) -> None:
    t.consume_workflow_event(
        _started(
            task_id,
            "AgentToolCallState",
            {"name": name, "tool_call_id": task_id, "kwargs": {}},
        )
    )


def test_tool_terminal_failed_uses_failure_message() -> None:
    t = _translator()
    _seed_tool_call(t, "c1", "todo")
    events = t.consume_workflow_event(_failed("c1", "AgentToolCallState", "kaboom"))
    res = [e for e in events if isinstance(e, ToolResultEvent)]
    assert res and res[0].error and "kaboom" in res[0].error


def test_tool_terminal_timed_out_with_and_without_type() -> None:
    t = _translator()
    _seed_tool_call(t, "c1", "todo")
    events = t.consume_workflow_event(
        _timed_out("c1", "AgentToolCallState", "deadline")
    )
    res = [e for e in events if isinstance(e, ToolResultEvent)]
    assert res and res[0].error and "deadline" in res[0].error

    t2 = _translator()
    _seed_tool_call(t2, "c2", "todo")
    res2 = [
        e
        for e in t2.consume_workflow_event(_timed_out("c2", "AgentToolCallState"))
        if isinstance(e, ToolResultEvent)
    ]
    assert res2 and res2[0].error and "Timed out" in res2[0].error


def test_tool_terminal_canceled_with_reason() -> None:
    t = _translator()
    _seed_tool_call(t, "c1", "todo")
    events = t.consume_workflow_event(_canceled("c1", "AgentToolCallState", "user"))
    res = [e for e in events if isinstance(e, ToolResultEvent)]
    assert (
        res and res[0].error and "Canceled" in res[0].error and "user" in res[0].error
    )


def test_tool_terminal_non_tool_call_state_returns_empty() -> None:
    t = _translator()
    assert t.consume_workflow_event(_failed("w", "working", "x")) == []


def test_tool_output_missing_emits_error_result() -> None:
    t = _translator()
    events = t.consume_workflow_event(
        _completed(
            "c1",
            "AgentToolCallState",
            {"name": "todo", "tool_call_id": "c1", "kwargs": {}, "output": None},
        )
    )
    res = [e for e in events if isinstance(e, ToolResultEvent)]
    assert res and res[0].error and "did not produce output" in res[0].error
    assert t._stats.tool_calls_failed == 1  # type: ignore[attr-defined]


def test_tool_output_with_error_string_emits_error_result() -> None:
    t = _translator()
    events = t.consume_workflow_event(
        _completed(
            "c1",
            "AgentToolCallState",
            {
                "name": "todo",
                "tool_call_id": "c1",
                "kwargs": {},
                "output": {"error": "boom"},
            },
        )
    )
    res = [e for e in events if isinstance(e, ToolResultEvent)]
    assert res and res[0].error and "boom" in res[0].error


def test_send_user_message_output_suppressed() -> None:
    t = _translator()
    events = t.consume_workflow_event(
        _completed(
            "c1",
            "AgentToolCallState",
            {
                "name": "send_user_message",
                "tool_call_id": "c1",
                "kwargs": {},
                "output": {"message": "hi"},
            },
        )
    )
    assert events == []


def test_unknown_task_type_returns_empty() -> None:
    t = _translator()
    assert t.consume_workflow_event(_started("t", "SomeOtherType", {})) == []


# --------------------------------------------------------------------------- #
# Tool-class resolution                                                       #
# --------------------------------------------------------------------------- #


def test_resolve_tool_class_dotted_and_suffix_match() -> None:
    class _Bash:
        pass

    t = _translator(available_tools={"bash": _Bash, "ns.bash": _Bash})  # type: ignore[dict-item]
    # dotted tool name resolves via short suffix match
    assert t._resolve_tool_class("other.bash") is _Bash  # type: ignore[attr-defined]
