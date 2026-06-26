from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from vibe.core.tools.mcp.pool import MCPSessionPool


class _FakeSession:
    """Records calls and CM teardown for assertions."""

    def __init__(self, name: str, sink: list[str]) -> None:
        self.name = name
        self.calls = 0
        self.sink = sink

    async def call_tool(self, *args: object) -> str:
        self.calls += 1
        return f"{self.name}-result-{self.calls}"


def _factory(sink: list[str], *, fail_once: bool = False):
    """Build a SessionFactory that enters a cleanup CM into the stack."""
    state = {"built": 0}

    async def factory(stack):
        state["built"] += 1

        # A CM that records teardown — proves the stack actually closes it.
        @asynccontextmanager
        async def transport():
            sink.append(f"open-{state['built']}")
            try:
                yield
            finally:
                sink.append(f"close-{state['built']}")

        await stack.enter_async_context(transport())
        if fail_once and state["built"] == 1:
            raise RuntimeError("connect failed")
        sink.append(f"session-built-{state['built']}")
        return _FakeSession(f"srv{state['built']}", sink)

    return factory, state


@pytest.mark.asyncio
async def test_reuses_connection_across_calls() -> None:
    sink: list[str] = []
    pool = MCPSessionPool()
    factory, state = _factory(sink)

    r1 = await pool.call("fp", factory, lambda s: s.call_tool())
    r2 = await pool.call("fp", factory, lambda s: s.call_tool())

    assert r1 == "srv1-result-1"
    assert r2 == "srv1-result-2"
    assert state["built"] == 1  # connected once, reused
    assert sink.count("session-built-1") == 1


@pytest.mark.asyncio
async def test_separate_fingerprints_get_separate_sessions() -> None:
    pool = MCPSessionPool()
    f1, s1 = _factory([])
    f2, s2 = _factory([])

    await pool.call("a", f1, lambda s: s.call_tool())
    await pool.call("b", f2, lambda s: s.call_tool())

    assert s1["built"] == 1
    assert s2["built"] == 1


@pytest.mark.asyncio
async def test_call_failure_marks_dead_and_reconnects_next_call() -> None:
    sink: list[str] = []
    pool = MCPSessionPool()
    factory, state = _factory(sink)

    async def boom(session):
        raise RuntimeError("transport broke")

    # First call: the session exists, but the call itself fails -> mark dead.
    with pytest.raises(RuntimeError, match="transport broke"):
        await pool.call("fp", factory, boom)
    assert state["built"] == 1

    # Next call rebuilds the connection (old stack closed, new session built).
    r = await pool.call("fp", factory, lambda s: s.call_tool())
    assert r == "srv2-result-1"  # srv2 = rebuilt session
    assert state["built"] == 2
    # The old transport was torn down on rebuild.
    assert "close-1" in sink


@pytest.mark.asyncio
async def test_factory_failure_tears_down_stack_and_propagates() -> None:
    sink: list[str] = []
    pool = MCPSessionPool()
    factory, _ = _factory(sink, fail_once=True)

    with pytest.raises(RuntimeError, match="connect failed"):
        await pool.call("fp", factory, lambda s: s.call_tool())

    # The half-entered transport CM must have been closed, not leaked.
    assert sink == ["open-1", "close-1"]


@pytest.mark.asyncio
async def test_concurrent_calls_on_same_fingerprint_serialize() -> None:
    """Two concurrent calls must not interleave on one session (MCP is R/R)."""
    pool = MCPSessionPool()
    factory, _ = _factory([])

    in_call = {"n": 0, "max": 0}

    async def tracked(session):
        in_call["n"] += 1
        in_call["max"] = max(in_call["max"], in_call["n"])
        await asyncio.sleep(0.01)
        in_call["n"] -= 1
        return session.call_tool()

    await asyncio.gather(
        pool.call("fp", factory, tracked),
        pool.call("fp", factory, tracked),
        pool.call("fp", factory, tracked),
    )

    assert in_call["max"] == 1  # never two calls in-flight on one session


@pytest.mark.asyncio
async def test_concurrent_first_calls_build_only_one_connection() -> None:
    pool = MCPSessionPool()
    factory, state = _factory([])

    await asyncio.gather(
        pool.call("fp", factory, lambda s: s.call_tool()),
        pool.call("fp", factory, lambda s: s.call_tool()),
        pool.call("fp", factory, lambda s: s.call_tool()),
    )

    assert state["built"] == 1  # creation lock prevented duplicate


@pytest.mark.asyncio
async def test_close_all_tears_down_every_connection() -> None:
    sink: list[str] = []
    pool = MCPSessionPool()
    fa, _ = _factory(sink)
    fb, _ = _factory(sink)
    await pool.call("a", fa, lambda s: s.call_tool())
    await pool.call("b", fb, lambda s: s.call_tool())

    await pool.close_all()

    assert "close-1" in sink  # session a's transport
    assert sink.count("close-1") >= 1
    # pool is empty after close
    assert pool._conns == {}


@pytest.mark.asyncio
async def test_close_all_is_safe_when_empty() -> None:
    pool = MCPSessionPool()
    await pool.close_all()  # must not raise


# --------------------------------------------------------------------------- #
# Proxy integration: a proxy with _pool set routes through the pool           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_proxy_with_pool_reuses_session_across_calls(monkeypatch) -> None:
    """A proxy whose _pool is set must reuse one session across two calls,
    rather than reconnecting per call. Verifies the opt-in wiring end-to-end
    without a real MCP server (the session factory is monkeypatched).
    """
    from tests.mock.utils import collect_result
    from vibe.core.tools.base import BaseToolConfig, BaseToolState
    from vibe.core.tools.mcp import (
        MCPSessionPool,
        RemoteTool,
        create_mcp_http_proxy_tool_class,
        tools as mcp_tools,
    )
    from vibe.core.tools.mcp.tools import _OpenArgs

    built = {"n": 0}

    async def fake_factory(stack, url, **kwargs):
        built["n"] += 1

        class _S:
            async def call_tool(self, name, args, read_timeout_seconds=None):
                return type("R", (), {"structuredContent": None, "content": []})()

        return _S()

    # Patch the pooled-session builder so no real transport is created.
    monkeypatch.setattr(mcp_tools, "build_pooled_http_session", fake_factory)

    pool = MCPSessionPool()
    remote = RemoteTool(name="search")
    cls = create_mcp_http_proxy_tool_class(
        url="https://mcp.example.com", remote=remote, alias="srv"
    )
    cls._pool = pool  # opt in
    tool = cls(config_getter=lambda: BaseToolConfig(), state=BaseToolState())

    await collect_result(tool.run(_OpenArgs()))
    await collect_result(tool.run(_OpenArgs()))

    # One pooled session built once, reused for the second call.
    assert built["n"] == 1
    await pool.close_all()
