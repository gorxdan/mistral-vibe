from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from tests.stubs.fake_connector_registry import FakeConnectorRegistry
from tests.stubs.fake_mcp_registry import FakeMCPRegistry
from vibe.core.config import ConnectorConfig, ToolManifestConfig
from vibe.core.llm.format import APIToolFormatHandler
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.connectors.connector_registry import RemoteTool
from vibe.core.tools.manager import ToolManager


def _remote_tool(name: str, description: str) -> RemoteTool:
    return RemoteTool.model_validate({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or record identifier",
                },
                "limit": {"type": "integer", "description": "Maximum rows"},
            },
        },
    })


def _manager(*, remote_count: int = 80) -> ToolManager:
    remotes = [
        _remote_tool(
            f"search_records_{i}",
            f"Search CRM records and customer notes for shard {i}",
        )
        for i in range(remote_count)
    ]
    registry = FakeConnectorRegistry({"heavy": remotes})
    config = build_test_vibe_config(
        enable_connectors=True,
        connectors=[ConnectorConfig(name="heavy")],
        tool_manifest=ToolManifestConfig(
            dynamic_subset_enabled=True,
            dynamic_subset_threshold=40,
            dynamic_pinned_tool_limit=6,
        ),
    )
    return ToolManager(
        lambda: config, mcp_registry=FakeMCPRegistry(), connector_registry=registry
    )


def _manifest_names(manager: ToolManager) -> list[str]:
    tools = APIToolFormatHandler().get_available_tools(manager)
    return [tool.function.name for tool in tools]


def test_dynamic_manifest_suppresses_remote_tools_and_keeps_search_tool() -> None:
    manager = _manager(remote_count=80)

    names = _manifest_names(manager)
    remote_names = [name for name in names if name.startswith("connector_heavy_")]

    assert "tool_search" in names
    assert remote_names == []
    assert len(names) < 40


def test_small_remote_catalog_stays_visible_without_search_tool() -> None:
    manager = _manager(remote_count=4)

    names = _manifest_names(manager)

    assert "tool_search" not in names
    assert "connector_heavy_search_records_0" in names
    assert "connector_heavy_search_records_3" in names


def test_explicit_enabled_tools_bypasses_dynamic_manifest_gating() -> None:
    manager = _manager(remote_count=80)
    manager._config.enabled_tools = ["connector_heavy_search_records_7"]

    names = _manifest_names(manager)

    assert names == ["connector_heavy_search_records_7"]


def test_disabled_remote_tool_is_not_searchable_or_pinnable() -> None:
    manager = _manager(remote_count=80)
    manager._config.connectors[0].disabled_tools = ["search_records_7"]
    target = "connector_heavy_search_records_7"

    assert target not in [
        match.name for match in manager.search_tools("records shard 7")
    ]
    assert manager.pin_manifest_tools([target]) == []
    assert target not in _manifest_names(manager)


def test_pinned_remote_tool_is_added_to_next_manifest() -> None:
    manager = _manager(remote_count=80)
    target = "connector_heavy_search_records_7"

    manager.pin_manifest_tools([target])
    names = _manifest_names(manager)

    assert target in names
    assert "connector_heavy_search_records_8" not in names


@pytest.mark.asyncio
async def test_tool_search_discovers_and_pins_matching_remote_tools() -> None:
    manager = _manager(remote_count=80)
    tool = manager.get("tool_search")

    result = await collect_result(
        tool.invoke(
            query="records shard 7",
            max_results=3,
            ctx=InvokeContext(tool_call_id="tool-search-call", tool_manager=manager),
        )
    )

    assert result.__class__.__name__ == "ToolSearchResult"
    assert result.matches
    assert result.matches[0].name == "connector_heavy_search_records_7"
    assert "connector_heavy_search_records_7" in result.activated_tools
    assert "connector_heavy_search_records_7" in _manifest_names(manager)
