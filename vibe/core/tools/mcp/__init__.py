from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vibe.core.tools.mcp.pool import MCPSessionPool
    from vibe.core.tools.mcp.registry import MCPRegistry
    from vibe.core.tools.mcp.tools import (
        MCPToolResult,
        RemoteTool,
        RemoteToolAnnotations,
        _mcp_stderr_capture,
        _parse_call_result,
        _stderr_logger_thread,
        call_tool_http,
        call_tool_stdio,
        create_mcp_http_proxy_tool_class,
        create_mcp_stdio_proxy_tool_class,
        create_vibe_mcp_http_client,
        list_tools_http,
        list_tools_stdio,
    )

__all__ = [
    "MCPRegistry",
    "MCPSessionPool",
    "MCPToolResult",
    "RemoteTool",
    "RemoteToolAnnotations",
    "_mcp_stderr_capture",
    "_parse_call_result",
    "_stderr_logger_thread",
    "call_tool_http",
    "call_tool_stdio",
    "create_mcp_http_proxy_tool_class",
    "create_mcp_stdio_proxy_tool_class",
    "create_vibe_mcp_http_client",
    "list_tools_http",
    "list_tools_stdio",
]


def __getattr__(name: str) -> Any:
    if name == "MCPRegistry":
        from vibe.core.tools.mcp.registry import MCPRegistry

        globals()["MCPRegistry"] = MCPRegistry
        return MCPRegistry
    if name == "MCPSessionPool":
        from vibe.core.tools.mcp.pool import MCPSessionPool

        globals()["MCPSessionPool"] = MCPSessionPool
        return MCPSessionPool
    if name in {
        "MCPToolResult",
        "RemoteTool",
        "RemoteToolAnnotations",
        "_mcp_stderr_capture",
        "_parse_call_result",
        "_stderr_logger_thread",
        "call_tool_http",
        "call_tool_stdio",
        "create_mcp_http_proxy_tool_class",
        "create_mcp_stdio_proxy_tool_class",
        "create_vibe_mcp_http_client",
        "list_tools_http",
        "list_tools_stdio",
    }:
        from vibe.core.tools.mcp import tools

        value = getattr(tools, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
