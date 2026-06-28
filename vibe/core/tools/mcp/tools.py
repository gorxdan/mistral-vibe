from __future__ import annotations

from collections.abc import AsyncGenerator
import contextlib
from datetime import timedelta
import functools
import hashlib
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

import httpx
import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vibe.core.logger import logger
from vibe.core.tools._schema import dereference_refs
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.mcp.pool import MCPSessionPool
from vibe.core.tools.mcp_sampling import MCPSamplingHandler
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import ToolResultDisplay, ToolUIData
from vibe.core.types import ToolStreamEvent
from vibe.core.utils.http import build_ssl_context
from vibe.core.utils.io import decode_safe

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from vibe.core.types import ToolResultEvent


# Mirrors MCP's default Streamable HTTP timeout values while avoiding an import from
# mcp.shared._httpx_utils, which is an internal module.
_MCP_DEFAULT_TIMEOUT = 30.0
_MCP_DEFAULT_SSE_READ_TIMEOUT = 300.0

_MCP_LAZY_NAMES = (
    "ClientSession",
    "StdioServerParameters",
    "stdio_client",
    "streamable_http_client",
)


# The mcp SDK pulls ~100ms of mcp.types/pydantic model construction at import.
# It is only needed to connect to a server, so load it lazily into the module
# globals on first use. The connect-time functions below call _load_mcp() and
# then reference the names as globals, so test patches of e.g.
# `...tools.streamable_http_client` still take effect.
def _load_mcp() -> None:
    # Fill only the names not already present, so an active test patch of one
    # name (or a mock teardown that removed another) never gets clobbered.
    missing = [name for name in _MCP_LAZY_NAMES if name not in globals()]
    if not missing:
        return
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamable_http_client

    source = {
        "ClientSession": ClientSession,
        "StdioServerParameters": StdioServerParameters,
        "stdio_client": stdio_client,
        "streamable_http_client": streamable_http_client,
    }
    for name in missing:
        globals()[name] = source[name]


def __getattr__(name: str) -> Any:
    # PEP 562: external access (incl. tests patching these names) triggers the
    # lazy load so they behave like normal module attributes.
    if name in _MCP_LAZY_NAMES:
        _load_mcp()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _stderr_logger_thread(read_fd: int) -> None:
    with open(read_fd, "rb") as f:
        for line in iter(f.readline, b""):
            decoded = decode_safe(line, from_subprocess=True).text.rstrip()
            if decoded:
                logger.debug("[MCP stderr] %s", decoded)


@contextlib.asynccontextmanager
async def _mcp_stderr_capture() -> AsyncGenerator[TextIO, None]:
    r, w = os.pipe()
    errlog = None
    thread_started = False
    try:
        thread = threading.Thread(target=_stderr_logger_thread, args=(r,), daemon=True)
        thread.start()
        thread_started = True
        errlog = os.fdopen(w, "w")
        yield errlog
    finally:
        if errlog is not None:
            errlog.close()
        elif thread_started:
            os.close(w)
        else:
            os.close(r)
            os.close(w)


class _OpenArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class MCPToolResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ok: bool = True
    server: str
    tool: str
    text: str | None = None
    structured: dict[str, Any] | None = None


class MCPTool(
    BaseTool[_OpenArgs, MCPToolResult, BaseToolConfig, BaseToolState],
    ToolUIData[_OpenArgs, MCPToolResult],
):
    _server_name: ClassVar[str] = ""
    _remote_name: ClassVar[str] = ""
    _is_connector: ClassVar[bool] = False
    # Hints captured from the server's declared ToolAnnotations (None when the
    # server declares none). readOnlyHint auto-approves; destructiveHint forces
    # ASK even when the tool's config permission is ALWAYS.
    _read_only_hint: ClassVar[bool | None] = None
    _destructive_hint: ClassVar[bool | None] = None
    # When the registry injects a shared MCPSessionPool, calls reuse one live
    # session per server instead of paying a full connect handshake each call.
    # None (default) keeps the original one-shot connect-per-call behavior.
    _pool: ClassVar[MCPSessionPool | None] = None

    @classmethod
    def get_server_name(cls) -> str | None:
        return cls._server_name or None

    @classmethod
    def get_remote_name(cls) -> str:
        return cls._remote_name or cls.get_name()

    @classmethod
    def is_connector(cls) -> bool:
        return cls._is_connector

    def resolve_permission(self, args: _OpenArgs) -> PermissionContext | None:
        # A destructive tool always forces an approval prompt, even when config
        # permission is ALWAYS — the server itself declares side effects.
        if self._destructive_hint:
            return PermissionContext(
                permission=ToolPermission.ASK,
                reason=(
                    f"MCP tool {self.get_remote_name()} declares a destructive "
                    "hint; approval required."
                ),
            )
        # A read-only tool auto-approves (the server declares no side effects).
        if self._read_only_hint:
            return PermissionContext(permission=ToolPermission.ALWAYS)
        return None


class RemoteToolAnnotations(BaseModel):
    """Hints an MCP server declares about a tool's effects.

    Mirrors the MCP SDK ``ToolAnnotations``. ``readOnlyHint`` means the tool
    has no side effects and is safe to auto-approve; ``destructiveHint`` means
    it mutates state and should force an approval prompt.
    """

    model_config = ConfigDict(
        from_attributes=True, populate_by_name=True, extra="ignore"
    )

    title: str | None = None
    read_only_hint: bool | None = Field(default=None, validation_alias="readOnlyHint")
    destructive_hint: bool | None = Field(
        default=None, validation_alias="destructiveHint"
    )
    idempotent_hint: bool | None = Field(
        default=None, validation_alias="idempotentHint"
    )
    open_world_hint: bool | None = Field(default=None, validation_alias="openWorldHint")


class RemoteTool(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        validation_alias="inputSchema",
    )
    annotations: RemoteToolAnnotations | None = None

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("MCP tool missing valid 'name'")
        return v

    @field_validator("input_schema", mode="before")
    @classmethod
    def _normalize_schema(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {"type": "object", "properties": {}}
        if isinstance(v, dict):
            return v
        dump = getattr(v, "model_dump", None)
        if callable(dump):
            try:
                v = dump()
            except Exception as e:
                raise ValueError(
                    "inputSchema must be a dict or have a valid model_dump method"
                ) from e
        if not isinstance(v, dict):
            raise ValueError("inputSchema must be a dict")
        return v


class _MCPContentBlock(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")
    text: str | None = None


class _MCPResultIn(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")

    structuredContent: dict[str, Any] | None = None
    content: list[_MCPContentBlock] | None = None

    @field_validator("structuredContent", mode="before")
    @classmethod
    def _normalize_structured(cls, v: Any) -> dict[str, Any] | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        dump = getattr(v, "model_dump", None)
        if callable(dump):
            try:
                v = dump()
            except Exception:
                return None
        return v if isinstance(v, dict) else None


def _parse_call_result(server: str, tool: str, result_obj: Any) -> MCPToolResult:
    parsed = _MCPResultIn.model_validate(result_obj)
    if (structured := parsed.structuredContent) is not None:
        return MCPToolResult(server=server, tool=tool, text=None, structured=structured)

    blocks = parsed.content or []
    parts = [b.text for b in blocks if isinstance(b.text, str)]
    text = "\n".join(parts) if parts else None
    return MCPToolResult(server=server, tool=tool, text=text, structured=None)


def create_vibe_mcp_http_client(
    headers: dict[str, str] | None, *, auth: httpx.Auth | None = None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        follow_redirects=True,
        headers=headers,
        auth=auth,
        timeout=httpx.Timeout(_MCP_DEFAULT_TIMEOUT, read=_MCP_DEFAULT_SSE_READ_TIMEOUT),
        verify=build_ssl_context(),
    )


async def list_tools_http(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    startup_timeout_sec: float | None = None,
) -> list[RemoteTool]:
    _load_mcp()
    timeout = timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    async with create_vibe_mcp_http_client(headers, auth=auth) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timeout
            ) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                return [RemoteTool.model_validate(t) for t in tools_resp.tools]


async def call_tool_http(
    url: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
    sampling_callback: MCPSamplingHandler | None = None,
) -> MCPToolResult:
    _load_mcp()
    init_timeout = (
        timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    )
    call_timeout = timedelta(seconds=tool_timeout_sec) if tool_timeout_sec else None
    async with create_vibe_mcp_http_client(headers, auth=auth) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=init_timeout,
                sampling_callback=sampling_callback,
            ) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool_name, arguments, read_timeout_seconds=call_timeout
                )
                return _parse_call_result(url, tool_name, result)


async def build_pooled_http_session(
    stack: contextlib.AsyncExitStack,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    startup_timeout_sec: float | None = None,
    sampling_callback: MCPSamplingHandler | None = None,
) -> Any:
    """Build an initialized ClientSession for a pooled HTTP connection.

    Enters the transport + session async context managers into ``stack`` so they
    stay alive for the pooled connection's lifetime and are torn down when the
    pool closes the stack. Mirrors :func:`call_tool_http` but returns the live
    session instead of consuming it.
    """
    _load_mcp()
    init_timeout = (
        timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    )
    http_client = await stack.enter_async_context(
        create_vibe_mcp_http_client(headers, auth=auth)
    )
    read, write, _ = await stack.enter_async_context(
        streamable_http_client(url, http_client=http_client)
    )
    session = await stack.enter_async_context(
        ClientSession(
            read,
            write,
            read_timeout_seconds=init_timeout,
            sampling_callback=sampling_callback,
        )
    )
    await session.initialize()
    return session


def create_mcp_http_proxy_tool_class(
    *,
    url: str,
    remote: RemoteTool,
    alias: str | None = None,
    server_hint: str | None = None,
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
    sampling_enabled: bool = True,
) -> type[MCPTool]:
    from urllib.parse import urlparse

    def _alias_from_url(url: str) -> str:
        p = urlparse(url)
        host = (p.hostname or "mcp").replace(".", "_")
        port = f"_{p.port}" if p.port else ""
        return f"{host}{port}"

    computed_alias = alias or _alias_from_url(url)
    published_name = f"{computed_alias}_{remote.name}"

    class MCPHttpProxyTool(MCPTool):
        description: ClassVar[str] = (
            (f"[{computed_alias}] " if computed_alias else "")
            + (remote.description or f"MCP tool '{remote.name}' from {url}")
            + (f"\nHint: {server_hint}" if server_hint else "")
        )
        _server_name: ClassVar[str] = computed_alias
        _mcp_url: ClassVar[str] = url
        _remote_name: ClassVar[str] = remote.name
        _input_schema: ClassVar[dict[str, Any]] = remote.input_schema
        _read_only_hint: ClassVar[bool | None] = (
            remote.annotations.read_only_hint if remote.annotations else None
        )
        _destructive_hint: ClassVar[bool | None] = (
            remote.annotations.destructive_hint if remote.annotations else None
        )
        _headers: ClassVar[dict[str, str]] = dict(headers or {})
        # TODO(VIBE-3057+): concurrent refresh coordinated by per-alias
        # asyncio.Lock in MCPRegistry (PR 4 / project decision #6) — this
        # object is shared across all calls on this proxy class.
        _auth: ClassVar[httpx.Auth | None] = auth
        _startup_timeout_sec: ClassVar[float | None] = startup_timeout_sec
        _tool_timeout_sec: ClassVar[float | None] = tool_timeout_sec
        _sampling_enabled: ClassVar[bool] = sampling_enabled

        @classmethod
        @functools.cache
        def get_name(cls) -> str:
            return published_name

        @classmethod
        @functools.cache
        def _build_parameters(cls) -> dict[str, Any]:
            # Remote MCP servers may publish a $ref with sibling keywords
            # ({"$ref": "#/$defs/X", "description": ...}); strict backends
            # (Moonshot/kimi) reject that. Inline references so the wire schema
            # is flat. Titles are remote-authored and preserved. Cached because
            # get_available_tools calls this for every tool on every LLM turn.
            return dereference_refs(dict(cls._input_schema))

        @classmethod
        def get_parameters(cls) -> dict[str, Any]:
            return orjson.loads(orjson.dumps(cls._build_parameters()))

        async def run(
            self, args: _OpenArgs, ctx: InvokeContext | None = None
        ) -> AsyncGenerator[ToolStreamEvent | MCPToolResult, None]:
            try:
                sampling_callback = (
                    ctx.sampling_callback if ctx and self._sampling_enabled else None
                )
                payload = args.model_dump(exclude_none=True)
                call_timeout = (
                    timedelta(seconds=self._tool_timeout_sec)
                    if self._tool_timeout_sec
                    else None
                )
                if self._pool is not None:
                    fingerprint = f"http:{self._mcp_url}"

                    async def factory(stack: contextlib.AsyncExitStack) -> Any:
                        return await build_pooled_http_session(
                            stack,
                            self._mcp_url,
                            headers=self._headers,
                            auth=self._auth,
                            startup_timeout_sec=self._startup_timeout_sec,
                            sampling_callback=sampling_callback,
                        )

                    async def call(session: Any) -> MCPToolResult:
                        result = await session.call_tool(
                            self._remote_name,
                            payload,
                            read_timeout_seconds=call_timeout,
                        )
                        return _parse_call_result(
                            self._mcp_url, self._remote_name, result
                        )

                    yield await self._pool.call(fingerprint, factory, call)
                else:
                    yield await call_tool_http(
                        self._mcp_url,
                        self._remote_name,
                        payload,
                        headers=self._headers,
                        auth=self._auth,
                        startup_timeout_sec=self._startup_timeout_sec,
                        tool_timeout_sec=self._tool_timeout_sec,
                        sampling_callback=sampling_callback,
                    )
            except Exception as exc:
                raise ToolError(f"MCP call failed: {exc}") from exc

        @classmethod
        def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
            if not isinstance(event.result, MCPToolResult):
                return ToolResultDisplay(
                    success=False,
                    message=event.error or event.skip_reason or "No result",
                )

            message = f"MCP tool {event.result.tool} completed"
            return ToolResultDisplay(success=event.result.ok, message=message)

        @classmethod
        def get_status_text(cls) -> str:
            return f"Calling MCP tool {remote.name}"

    MCPHttpProxyTool.__name__ = f"MCP_{computed_alias}__{remote.name}"
    return MCPHttpProxyTool


async def list_tools_stdio(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    startup_timeout_sec: float | None = None,
) -> list[RemoteTool]:
    _load_mcp()
    params = StdioServerParameters(
        command=command[0], args=command[1:], env=env, cwd=cwd
    )
    timeout = timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    async with (
        _mcp_stderr_capture() as errlog,
        stdio_client(params, errlog=errlog) as (read, write),
        ClientSession(read, write, read_timeout_seconds=timeout) as session,
    ):
        await session.initialize()
        tools_resp = await session.list_tools()
        return [RemoteTool.model_validate(t) for t in tools_resp.tools]


async def call_tool_stdio(
    command: list[str],
    tool_name: str,
    arguments: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
    sampling_callback: MCPSamplingHandler | None = None,
) -> MCPToolResult:
    _load_mcp()
    params = StdioServerParameters(
        command=command[0], args=command[1:], env=env, cwd=cwd
    )
    init_timeout = (
        timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    )
    call_timeout = timedelta(seconds=tool_timeout_sec) if tool_timeout_sec else None
    async with (
        _mcp_stderr_capture() as errlog,
        stdio_client(params, errlog=errlog) as (read, write),
        ClientSession(
            read,
            write,
            read_timeout_seconds=init_timeout,
            sampling_callback=sampling_callback,
        ) as session,
    ):
        await session.initialize()
        result = await session.call_tool(
            tool_name, arguments, read_timeout_seconds=call_timeout
        )
        return _parse_call_result("stdio:" + " ".join(command), tool_name, result)


async def build_pooled_stdio_session(
    stack: contextlib.AsyncExitStack,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    startup_timeout_sec: float | None = None,
    sampling_callback: MCPSamplingHandler | None = None,
) -> Any:
    """Build an initialized ClientSession for a pooled stdio connection.

    Enters the stderr capture, stdio transport, and session async context
    managers into ``stack`` so they stay alive for the pooled connection's
    lifetime and are torn down (subprocess reaped) when the pool closes the
    stack. Mirrors :func:`call_tool_stdio` but returns the live session.
    """
    _load_mcp()
    params = StdioServerParameters(
        command=command[0], args=command[1:], env=env, cwd=cwd
    )
    init_timeout = (
        timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
    )
    await stack.enter_async_context(_mcp_stderr_capture())
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(
        ClientSession(
            read,
            write,
            read_timeout_seconds=init_timeout,
            sampling_callback=sampling_callback,
        )
    )
    await session.initialize()
    return session


def create_mcp_stdio_proxy_tool_class(
    *,
    command: list[str],
    remote: RemoteTool,
    alias: str | None = None,
    server_hint: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    startup_timeout_sec: float | None = None,
    tool_timeout_sec: float | None = None,
    sampling_enabled: bool = True,
) -> type[MCPTool]:
    def _alias_from_command(cmd: list[str]) -> str:
        prog = Path(cmd[0]).name.replace(".", "_") if cmd else "mcp"
        digest = hashlib.blake2s(
            "\0".join(cmd).encode("utf-8"), digest_size=4
        ).hexdigest()
        return f"{prog}_{digest}"

    computed_alias = alias or _alias_from_command(command)
    published_name = f"{computed_alias}_{remote.name}"

    class MCPStdioProxyTool(MCPTool):
        description: ClassVar[str] = (
            (f"[{computed_alias}] " if computed_alias else "")
            + (
                remote.description
                or f"MCP tool '{remote.name}' from stdio command: {' '.join(command)}"
            )
            + (f"\nHint: {server_hint}" if server_hint else "")
        )
        _server_name: ClassVar[str] = computed_alias
        _stdio_command: ClassVar[list[str]] = command
        _remote_name: ClassVar[str] = remote.name
        _input_schema: ClassVar[dict[str, Any]] = remote.input_schema
        _read_only_hint: ClassVar[bool | None] = (
            remote.annotations.read_only_hint if remote.annotations else None
        )
        _destructive_hint: ClassVar[bool | None] = (
            remote.annotations.destructive_hint if remote.annotations else None
        )
        _env: ClassVar[dict[str, str] | None] = env
        _cwd: ClassVar[str | None] = cwd
        _startup_timeout_sec: ClassVar[float | None] = startup_timeout_sec
        _tool_timeout_sec: ClassVar[float | None] = tool_timeout_sec
        _sampling_enabled: ClassVar[bool] = sampling_enabled

        @classmethod
        @functools.cache
        def get_name(cls) -> str:
            return published_name

        @classmethod
        @functools.cache
        def _build_parameters(cls) -> dict[str, Any]:
            # Remote MCP servers may publish a $ref with sibling keywords
            # ({"$ref": "#/$defs/X", "description": ...}); strict backends
            # (Moonshot/kimi) reject that. Inline references so the wire schema
            # is flat. Titles are remote-authored and preserved. Cached because
            # get_available_tools calls this for every tool on every LLM turn.
            return dereference_refs(dict(cls._input_schema))

        @classmethod
        def get_parameters(cls) -> dict[str, Any]:
            return orjson.loads(orjson.dumps(cls._build_parameters()))

        async def run(
            self, args: _OpenArgs, ctx: InvokeContext | None = None
        ) -> AsyncGenerator[ToolStreamEvent | MCPToolResult, None]:
            try:
                sampling_callback = (
                    ctx.sampling_callback if ctx and self._sampling_enabled else None
                )
                payload = args.model_dump(exclude_none=True)
                call_timeout = (
                    timedelta(seconds=self._tool_timeout_sec)
                    if self._tool_timeout_sec
                    else None
                )
                if self._pool is not None:
                    fingerprint = "stdio:" + "\0".join(self._stdio_command)

                    async def factory(stack: contextlib.AsyncExitStack) -> Any:
                        return await build_pooled_stdio_session(
                            stack,
                            self._stdio_command,
                            env=self._env,
                            cwd=self._cwd,
                            startup_timeout_sec=self._startup_timeout_sec,
                            sampling_callback=sampling_callback,
                        )

                    async def call(session: Any) -> MCPToolResult:
                        result = await session.call_tool(
                            self._remote_name,
                            payload,
                            read_timeout_seconds=call_timeout,
                        )
                        return _parse_call_result(
                            "stdio:" + " ".join(self._stdio_command),
                            self._remote_name,
                            result,
                        )

                    yield await self._pool.call(fingerprint, factory, call)
                else:
                    yield await call_tool_stdio(
                        self._stdio_command,
                        self._remote_name,
                        payload,
                        env=self._env,
                        cwd=self._cwd,
                        startup_timeout_sec=self._startup_timeout_sec,
                        tool_timeout_sec=self._tool_timeout_sec,
                        sampling_callback=sampling_callback,
                    )
            except Exception as exc:
                raise ToolError(f"MCP stdio call failed: {exc!r}") from exc

        @classmethod
        def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
            if not isinstance(event.result, MCPToolResult):
                return ToolResultDisplay(
                    success=False,
                    message=event.error or event.skip_reason or "No result",
                )

            message = f"MCP tool {event.result.tool} completed"
            return ToolResultDisplay(success=event.result.ok, message=message)

        @classmethod
        def get_status_text(cls) -> str:
            return f"Calling MCP tool {remote.name}"

    MCPStdioProxyTool.__name__ = f"MCP_STDIO_{computed_alias}__{remote.name}"
    return MCPStdioProxyTool
