from __future__ import annotations

from vibe.core.agent_loop import MAX_TOOL_RESULT_CHARS, AgentLoop


def test_small_result_unchanged() -> None:
    text = "hello world"
    assert AgentLoop._apply_tool_result_budget(text) == text


def test_result_at_limit_unchanged() -> None:
    text = "x" * MAX_TOOL_RESULT_CHARS
    assert AgentLoop._apply_tool_result_budget(text) == text


def test_oversized_result_truncated_head_and_tail() -> None:
    text = "A" * 80_000 + "B" * 80_000
    out = AgentLoop._apply_tool_result_budget(text)
    assert len(out) < MAX_TOOL_RESULT_CHARS + 200
    assert out.startswith("A")  # head preserved
    assert out.endswith("B")  # tail preserved
    assert "elided" in out
    assert str(len(text) - MAX_TOOL_RESULT_CHARS) in out  # elided count shown
