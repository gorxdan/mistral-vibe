"""Persistent connection pool for MCP server sessions.

Each tool call through :func:`vibe.core.tools.mcp.tools.call_tool_http` and
:func:`call_tool_stdio` today pays a full connection lifecycle — for stdio that
is a subprocess spawn plus the MCP ``initialize()`` handshake; for http a fresh
TCP/SSE handshake. For a server called many times in a session that overhead
dominates latency.

This pool keeps one initialized ``ClientSession`` alive per server fingerprint.
The mcp SDK's transports (``streamable_http_client``, ``stdio_client``) and
``ClientSession`` are async context managers designed for ``async with``; keeping
them alive beyond a single scope is done with :class:`contextlib.AsyncExitStack`
— the CMs are entered into the stack on connect and all torn down by
``stack.aclose()`` on close. Using them any other way leaks subprocesses.

MCP is request/response per session: two concurrent calls on one pooled session
would interleave on the transport and corrupt the JSON-RPC stream, so each
pooled connection carries its own :class:`asyncio.Lock` serializing calls.

The pool self-heals: any exception during a call marks the connection dead, the
next call tears the dead stack down and reconnects rather than hanging on a
broken transport.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

from vibe.core.logger import logger

# A factory builds an initialized ClientSession by entering the transport +
# session async context managers into the provided AsyncExitStack. Returning the
# session keeps it alive for the lifetime of the stack.
SessionFactory = Callable[[AsyncExitStack], Awaitable[Any]]


class _PooledConnection:
    """One live session for a server fingerprint.

    ``stack`` owns the transport and session CMs; closing it tears them all
    down. ``lock`` serializes calls — MCP is request/response per session, so
    concurrent calls on one transport would corrupt the stream. ``dead`` is set
    on any call failure so the next acquire reconnects instead of reusing a
    broken transport.
    """

    __slots__ = ("stack", "session", "lock", "dead")

    def __init__(self, stack: AsyncExitStack, session: Any) -> None:
        self.stack = stack
        self.session = session
        self.lock = asyncio.Lock()
        self.dead = False

    async def aclose(self) -> None:
        try:
            await self.stack.aclose()
        except Exception:
            logger.debug("error closing pooled MCP connection stack", exc_info=True)


class MCPSessionPool:
    """Lazily-connected, self-healing pool of MCP sessions keyed by fingerprint.

    Callers identify a server by an opaque hashable ``fingerprint`` (the registry
    derives one from the server config) and supply a :data:`SessionFactory` that
    knows how to build that server's session. The first call to a fingerprint
    connects; subsequent calls reuse. A per-connection lock serializes calls so
    concurrent tool invocations on the same server do not interleave on its
    transport.
    """

    def __init__(self) -> None:
        self._conns: dict[str, _PooledConnection] = {}
        # Per-fingerprint creation lock so two concurrent first-calls don't race
        # to build two connections to the same server.
        self._create_locks: dict[str, asyncio.Lock] = {}

    async def call(
        self,
        fingerprint: str,
        factory: SessionFactory,
        call: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Run ``call(session)`` against the pooled session for *fingerprint*.

        Connects on first use, reconnects after a prior failure. The per-session
        lock serializes the call against other concurrent callers on the same
        server. Any exception from ``call`` marks the connection dead so the
        next call rebuilds it.
        """
        conn = await self._get_or_build(fingerprint, factory)
        async with conn.lock:
            # A prior caller may have marked this conn dead between acquire and
            # lock acquisition; rebuild once if so.
            if conn.dead:
                conn = await self._get_or_build(fingerprint, factory)
                async with conn.lock:
                    return await self._invoke(conn, fingerprint, call)
            return await self._invoke(conn, fingerprint, call)

    async def _invoke(
        self,
        conn: _PooledConnection,
        fingerprint: str,
        call: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        try:
            return await call(conn.session)
        except Exception:
            # Transport is likely broken; force a reconnect on the next call.
            conn.dead = True
            logger.warning(
                "pooled MCP call failed for %s; marking connection for reconnect",
                fingerprint,
                exc_info=True,
            )
            raise

    async def _get_or_build(
        self, fingerprint: str, factory: SessionFactory
    ) -> _PooledConnection:
        # Creation is serialized per fingerprint so only one connection is built.
        create_lock = self._create_locks.setdefault(fingerprint, asyncio.Lock())
        async with create_lock:
            conn = self._conns.get(fingerprint)
            if conn is None or conn.dead:
                if conn is not None:
                    await conn.aclose()
                conn = await self._build(fingerprint, factory)
            return conn

    async def _build(
        self, fingerprint: str, factory: SessionFactory
    ) -> _PooledConnection:
        stack = AsyncExitStack()
        try:
            session = await factory(stack)
        except BaseException:
            # Factory failed before a session was produced; tear down anything it
            # half-entered so no transport is leaked.
            await stack.aclose()
            raise
        conn = _PooledConnection(stack, session)
        self._conns[fingerprint] = conn
        return conn

    async def close_all(self) -> None:
        """Tear down every pooled connection. Safe to call at shutdown."""
        conns = list(self._conns.values())
        self._conns.clear()
        for conn in conns:
            await conn.aclose()
