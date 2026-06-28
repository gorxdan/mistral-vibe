from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vibe.core.config import MCPOAuth, MCPStreamableHttp
from vibe.core.tools.mcp import MCPRegistry


def _uncached_oauth_server(alias: str) -> MCPStreamableHttp:
    return MCPStreamableHttp(
        name=alias,
        transport="streamable-http",
        url="https://mcp.example.com/mcp",
        auth=MCPOAuth(type="oauth", scopes=["read"]),
    )


@pytest.mark.asyncio
async def test_uncached_oauth_server_yields_no_tools_when_not_logged_in() -> None:
    # An OAuth MCP server the user has not logged into yet surfaces "needs auth"
    # by discovering no tools and staying uncached, so the next refresh after
    # `/mcp login` re-runs discovery.
    registry = MCPRegistry()
    server = _uncached_oauth_server("sentry")

    with patch("vibe.core.auth.is_logged_in", new=AsyncMock(return_value=False)):
        tools = await registry.get_tools_async([server])

    assert tools == {}
    assert registry.count_loaded([server]) == 0


@pytest.mark.asyncio
async def test_discover_http_returns_none_for_unauthenticated_oauth() -> None:
    # `_discover_http` returns None (retryable, not cached) rather than {} so a
    # later `/mcp login` will re-discover the server's tools.
    registry = MCPRegistry()
    server = _uncached_oauth_server("sentry")

    with patch("vibe.core.auth.is_logged_in", new=AsyncMock(return_value=False)):
        result = await registry._discover_http(server)

    assert result is None
