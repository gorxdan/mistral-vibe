from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.middleware import ConversationContext, ToolResultBudgetMiddleware
from vibe.core.tools.tool_result_store import ToolResultStore
from vibe.core.types import AgentStats, LLMMessage, MessageList, Role


def _ctx(messages: list[LLMMessage], cfg) -> ConversationContext:
    stats = AgentStats()
    return ConversationContext(messages=MessageList(messages), stats=stats, config=cfg)


def _tool_msg(call_id: str, content: str) -> LLMMessage:
    return LLMMessage(role=Role.TOOL, content=content, tool_call_id=call_id)


@pytest.mark.asyncio
async def test_group_under_budget_untouched(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    cfg = build_test_vibe_config()
    mw = ToolResultBudgetMiddleware(
        store, aggregate_chars=10_000, keep_recent_messages=1
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content="calling tools"),
        _tool_msg("c1", "x" * 2_000),
        _tool_msg("c2", "y" * 2_000),
        LLMMessage(role=Role.ASSISTANT, content="done"),
    ]
    ctx = _ctx(msgs, cfg)
    await mw.before_turn(ctx)
    assert ctx.messages[2].content == "x" * 2_000
    assert ctx.messages[3].content == "y" * 2_000


@pytest.mark.asyncio
async def test_oversized_group_largest_compressed(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    cfg = build_test_vibe_config()
    mw = ToolResultBudgetMiddleware(
        store, aggregate_chars=5_000, keep_recent_messages=1
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content="calling tools"),
        _tool_msg("c1", "A" * 4_000),
        _tool_msg("c2", "B" * 4_000),
        _tool_msg("c3", "C" * 500),
        LLMMessage(role=Role.ASSISTANT, content="done"),
    ]
    ctx = _ctx(msgs, cfg)
    await mw.before_turn(ctx)

    total = sum(len(m.content or "") for m in ctx.messages if m.role == Role.TOOL)
    assert total <= 5_000 + 500  # under budget + small result untouched
    # Largest result was persisted and compressed.
    assert "persisted to" in (ctx.messages[2].content or "")
    assert store.read("c1") == "A" * 4_000


@pytest.mark.asyncio
async def test_protects_recent_group(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    cfg = build_test_vibe_config()
    mw = ToolResultBudgetMiddleware(store, aggregate_chars=100, keep_recent_messages=4)
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content="calling tools"),
        _tool_msg("c1", "A" * 4_000),
        _tool_msg("c2", "B" * 4_000),
        LLMMessage(role=Role.ASSISTANT, content="done"),
    ]
    ctx = _ctx(msgs, cfg)
    await mw.before_turn(ctx)
    # Group falls within the last 4 messages (protected suffix).
    assert ctx.messages[2].content == "A" * 4_000
    assert ctx.messages[3].content == "B" * 4_000


@pytest.mark.asyncio
async def test_idempotent(tmp_path: Path) -> None:
    store = ToolResultStore(lambda: tmp_path)
    cfg = build_test_vibe_config()
    mw = ToolResultBudgetMiddleware(
        store, aggregate_chars=5_000, keep_recent_messages=1
    )
    msgs = [
        LLMMessage(role=Role.SYSTEM, content="sys"),
        LLMMessage(role=Role.ASSISTANT, content="calling tools"),
        _tool_msg("c1", "A" * 4_000),
        _tool_msg("c2", "B" * 4_000),
        LLMMessage(role=Role.ASSISTANT, content="done"),
    ]
    ctx = _ctx(msgs, cfg)
    await mw.before_turn(ctx)
    snapshot = [m.content for m in ctx.messages]
    await mw.before_turn(ctx)
    assert [m.content for m in ctx.messages] == snapshot
