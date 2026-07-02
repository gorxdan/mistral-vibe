from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolInfo,
)
from vibe.core.types import ToolStreamEvent


class ToolSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description="Words (or the exact name) describing the hidden tool you need."
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Maximum number of matching tools to return and activate.",
    )


class ToolSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    matches: list[ToolInfo]
    activated_tools: list[str]


class ToolSearch(
    BaseTool[ToolSearchArgs, ToolSearchResult, BaseToolConfig, BaseToolState]
):
    description = (
        "Search the hidden tool catalog (deactivated builtin tools and remote "
        "MCP/connector tools) by keyword and activate the best matches for future "
        "turns. Use this when a tool you need is not in your current tool list."
    )

    @classmethod
    def is_available(cls, config: Any | None = None) -> bool:
        return bool(
            config is not None
            and config.tool_manifest.dynamic_subset_enabled
            and (
                config.mcp_servers
                or config.connectors
                or config.tool_manifest.defer_builtin_tools
            )
        )

    async def run(
        self, args: ToolSearchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ToolSearchResult, None]:
        if ctx is None or ctx.tool_manager is None:
            raise ToolError("tool_search requires an active tool manager")
        matches = ctx.tool_manager.search_tools(
            args.query, max_results=args.max_results
        )
        activated = ctx.tool_manager.pin_manifest_tools([
            match.name for match in matches
        ])
        yield ToolSearchResult(
            query=args.query, matches=matches, activated_tools=activated
        )
