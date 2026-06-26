"""Differential tests proving the streaming accumulators produce results
identical to folding with ``LLMMessage.__add__`` / ``LLMChunk.__add__`` (the
O(n^2) behaviour they replace), plus the O(n) tool-argument buffering in the
OpenAI Responses parser state.
"""

from __future__ import annotations

import functools
import operator

import pytest

from vibe.core.llm.backend.openai_responses import _ResponsesToolCallState
from vibe.core.types import (
    FunctionCall,
    LLMChunk,
    LLMChunkAccumulator,
    LLMMessage,
    LLMMessageAccumulator,
    LLMUsage,
    Role,
    StopInfo,
    ToolCall,
)

_ASSISTANT = {"role": Role.assistant, "message_id": "m1"}


def _tc(index, name=None, args=None, call_id="call_0"):
    return ToolCall(
        id=call_id,
        type="function",
        index=index,
        function=FunctionCall(name=name, arguments=args),
    )


def _fold_messages(messages):
    return functools.reduce(operator.add, messages)


def _accumulate_messages(messages):
    acc = LLMMessageAccumulator()
    for m in messages:
        acc.add(m)
    return acc.build()


def _assert_same(messages, label):
    folded = _fold_messages(messages)
    built = _accumulate_messages(messages)
    assert built.model_dump() == folded.model_dump(), label


def test_content_deltas_match_fold():
    msgs = [LLMMessage(**_ASSISTANT, content=c) for c in ["Hel", "lo ", "world"]]
    _assert_same(msgs, "content deltas")


def test_reasoning_signature_and_state_match_fold():
    msgs = [
        LLMMessage(
            **_ASSISTANT,
            reasoning_content="th",
            reasoning_signature="s1",
            reasoning_state=["a"],
        ),
        LLMMessage(
            **_ASSISTANT,
            reasoning_content="ink",
            reasoning_signature="s2",
            reasoning_state=["b", "c"],
        ),
        LLMMessage(**_ASSISTANT, content="answer"),
    ]
    _assert_same(msgs, "reasoning + signature + state")


def test_single_tool_call_streamed_args_match_fold():
    msgs = [
        LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, name="bash")]),
        *[
            LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, args=part)])
            for part in ['{"cmd"', ':"ls ', '-la"}']
        ],
    ]
    _assert_same(msgs, "single tool call multi-delta args")
    built = _accumulate_messages(msgs)
    assert built.tool_calls[0].function.arguments == '{"cmd":"ls -la"}'


def test_interleaved_tool_indices_match_fold():
    msgs = [
        LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, name="read", call_id="c0")]),
        LLMMessage(**_ASSISTANT, tool_calls=[_tc(1, name="write", call_id="c1")]),
        LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, args='{"p":1}', call_id="c0")]),
        LLMMessage(**_ASSISTANT, tool_calls=[_tc(1, args='{"q":2}', call_id="c1")]),
    ]
    _assert_same(msgs, "interleaved tool indices")


def test_tool_name_arriving_late_matches_fold():
    msgs = [
        LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, name=None, args="{")]),
        LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, name="grep", args="}")]),
    ]
    _assert_same(msgs, "late tool name")


def test_single_occurrence_none_arguments_stay_none():
    msgs = [LLMMessage(**_ASSISTANT, content="x"), LLMMessage(**_ASSISTANT, content="y")]
    one = [LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, name="x", args=None)])]
    # A tool call seen once keeps its (possibly None) arguments, matching the
    # deepcopy-on-first-encounter semantics of __add__.
    built = _accumulate_messages([*msgs, *one])
    folded = _fold_messages([*msgs, *one])
    assert (
        built.tool_calls[0].function.arguments
        == folded.tool_calls[0].function.arguments
    )


@pytest.mark.parametrize(
    "messages",
    [
        [
            LLMMessage(role=Role.assistant, message_id="m1", content="a"),
            LLMMessage(role=Role.user, message_id="m2", content="b"),
        ],
        [
            LLMMessage(**_ASSISTANT, tool_calls=[_tc(None, name="x")]),
            LLMMessage(**_ASSISTANT, content="z"),
        ],
        [
            LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, name="a")]),
            LLMMessage(**_ASSISTANT, tool_calls=[_tc(0, name="b")]),
        ],
    ],
)
def test_validation_errors_match_fold(messages):
    fold_error = None
    acc_error = None
    try:
        _fold_messages(messages)
    except ValueError as exc:
        fold_error = str(exc)
    try:
        _accumulate_messages(messages)
    except ValueError as exc:
        acc_error = str(exc)
    assert fold_error == acc_error
    assert fold_error is not None


def test_chunk_accumulator_matches_chunk_fold():
    chunks = [
        LLMChunk(message=LLMMessage(**_ASSISTANT, content="a"), usage=None),
        LLMChunk(
            message=LLMMessage(**_ASSISTANT, content="b"),
            usage=LLMUsage(prompt_tokens=10, completion_tokens=1),
        ),
        LLMChunk(
            message=LLMMessage(**_ASSISTANT, content="c"),
            usage=LLMUsage(completion_tokens=2),
            stop=StopInfo(reason="refusal"),
        ),
    ]
    folded = functools.reduce(operator.add, chunks)
    acc = LLMChunkAccumulator()
    for c in chunks:
        acc.add(c)
    built = acc.build()
    assert built is not None
    assert built.message.model_dump() == folded.message.model_dump()
    assert built.usage == folded.usage
    assert built.stop == folded.stop
    # Running usage total is exposed for the caller's stats.
    assert acc.usage == LLMUsage(prompt_tokens=10, completion_tokens=3)


def test_chunk_accumulator_usage_none_when_no_chunk_reports_usage():
    acc = LLMChunkAccumulator()
    acc.add(LLMChunk(message=LLMMessage(**_ASSISTANT, content="a"), usage=None))
    built = acc.build()
    assert built is not None
    assert built.usage is None


def test_chunk_accumulator_empty_builds_none():
    assert LLMChunkAccumulator().build() is None
    assert LLMChunkAccumulator().empty is True


def test_responses_tool_call_state_buffers_arguments():
    state = _ResponsesToolCallState()
    for part in ['{"path"', ':"a.py"', ',"content":"x"}']:
        state.append_arguments(part)
    assert state.arguments == '{"path":"a.py","content":"x"}'
    # Reading is idempotent (buffer materialised once).
    assert state.arguments == '{"path":"a.py","content":"x"}'
    # A full replace clears the buffer.
    state.arguments = "reset"
    assert state.arguments == "reset"
    state.append_arguments("X")
    assert state.arguments == "resetX"
