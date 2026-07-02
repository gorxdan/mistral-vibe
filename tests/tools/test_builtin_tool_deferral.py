from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict
import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from tests.stubs.fake_connector_registry import FakeConnectorRegistry
from tests.stubs.fake_mcp_registry import FakeMCPRegistry
from vibe.core.config import ConnectorConfig, ToolManifestConfig
from vibe.core.llm.format import APIToolFormatHandler
from vibe.core.llm.models import ParsedMessage, ParsedToolCall, ResolvedMessage
from vibe.core.tools.base import BaseTool, BaseToolConfig, BaseToolState, InvokeContext
from vibe.core.tools.builtins.background import Background
from vibe.core.tools.connectors.connector_registry import RemoteTool
from vibe.core.tools.manager import ToolManager
from vibe.core.types import ToolStreamEvent

DEFERRED = (
    "manage_memory",
    "schedule",
    "team_message",
    "workflow_status",
    "workflow_stop",
)

NEVER_HIDDEN = (
    "background",
    "enter_plan_mode",
    "exit_plan_mode",
    "read",
    "edit",
    "bash",
    "grep",
    "glob",
    "task",
    "todo",
    "skill",
)


class GadgetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "x"


class GadgetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool = True


class Gadget(BaseTool[GadgetArgs, GadgetResult, BaseToolConfig, BaseToolState]):
    manifest_deferrable = True
    description = "Frobnicate the gadget. Extra detail the stub must drop."

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return "Gadget usage guide: frobnicate responsibly."

    async def run(
        self, args: GadgetArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | GadgetResult, None]:
        yield GadgetResult()


def _deferral_manager(**config_kwargs) -> ToolManager:
    config = build_test_vibe_config(
        tool_manifest=ToolManifestConfig(defer_builtin_tools=True), **config_kwargs
    )
    return ToolManager(lambda: config, mcp_registry=FakeMCPRegistry())


def _remote_tool(name: str) -> RemoteTool:
    return RemoteTool.model_validate({
        "name": name,
        "description": f"Search CRM records for {name}",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Query"}},
        },
    })


def _deferral_manager_with_remotes(remote_count: int = 12) -> ToolManager:
    remotes = [_remote_tool(f"search_records_{i}") for i in range(remote_count)]
    registry = FakeConnectorRegistry({"heavy": remotes})
    config = build_test_vibe_config(
        enable_connectors=True,
        connectors=[ConnectorConfig(name="heavy")],
        tool_manifest=ToolManifestConfig(
            defer_builtin_tools=True,
            dynamic_subset_threshold=10,
            dynamic_pinned_tool_limit=8,
        ),
    )
    return ToolManager(
        lambda: config, mcp_registry=FakeMCPRegistry(), connector_registry=registry
    )


def _resolve(manager: ToolManager, name: str, raw_args: dict) -> ResolvedMessage:
    parsed = ParsedMessage(
        tool_calls=[ParsedToolCall(tool_name=name, raw_args=raw_args, call_id="1")]
    )
    return APIToolFormatHandler().resolve_tool_calls(parsed, manager)


def test_marked_builtins_hidden_and_tool_search_present() -> None:
    manager = _deferral_manager()
    names = set(manager.manifest_tools)

    for name in DEFERRED:
        assert name in manager.available_tools
        assert name not in names
    assert "tool_search" in names


def test_pinned_core_tools_never_hidden() -> None:
    assert Background.manifest_deferrable is False

    manager = _deferral_manager()
    hidden = manager.hidden_tool_names()

    for name in NEVER_HIDDEN:
        assert name not in hidden
    assert "background" in manager.manifest_tools


def test_tool_search_remains_after_activating_all_builtins() -> None:
    manager = _deferral_manager()

    activated = manager.pin_manifest_tools(list(DEFERRED))
    names = set(manager.manifest_tools)

    assert sorted(DEFERRED) == sorted(activated)
    assert "tool_search" in names
    for name in DEFERRED:
        assert name in names


def test_search_activates_manage_memory_stickily_despite_remote_pin_pressure() -> None:
    manager = _deferral_manager_with_remotes()

    matches = manager.search_tools("memory")
    assert "manage_memory" in [match.name for match in matches]

    manager.pin_manifest_tools(["manage_memory"])
    assert "manage_memory" in manager.manifest_tools

    manager.pin_manifest_tools([
        f"connector_heavy_search_records_{i}" for i in range(8)
    ])
    assert "manage_memory" in manager.manifest_tools


def test_enabled_tools_profile_bypasses_deferral() -> None:
    manager = _deferral_manager(enabled_tools=["read", "schedule"])

    assert set(manager.manifest_tools) == {"read", "schedule"}


def test_disabled_deferrable_tool_removed_from_manifest_and_stubs() -> None:
    manager = _deferral_manager(disabled_tools=["schedule"])

    stub_names = [name for name, _ in manager.deferred_builtin_stubs()]
    assert "schedule" not in manager.manifest_tools
    assert "schedule" not in stub_names
    assert "manage_memory" in stub_names


def test_tool_search_disabled_fails_open_to_full_manifest() -> None:
    manager = _deferral_manager(disabled_tools=["tool_search"])
    names = set(manager.manifest_tools)

    for name in DEFERRED:
        assert name in names
    assert manager.deferred_builtin_stubs() == []


def test_hidden_tool_names_covers_builtin_and_remote() -> None:
    manager = _deferral_manager_with_remotes()

    hidden = manager.hidden_tool_names()

    assert set(DEFERRED) <= hidden
    assert any(name.startswith("connector_heavy_") for name in hidden)


@pytest.mark.asyncio
async def test_tool_search_result_carries_usage_prose_for_builtins() -> None:
    manager = _deferral_manager()
    manager._all_tools["gadget"] = Gadget
    tool = manager.get("tool_search")

    result = await collect_result(
        tool.invoke(
            query="gadget",
            ctx=InvokeContext(tool_call_id="call-1", tool_manager=manager),
        )
    )

    match = next(m for m in result.matches if m.name == "gadget")
    assert match.usage == "Gadget usage guide: frobnicate responsibly."
    assert "gadget" in result.activated_tools
    assert "gadget" in manager.manifest_tools


def test_search_result_usage_is_none_for_remote_tools() -> None:
    manager = _deferral_manager_with_remotes()

    matches = manager.search_tools("records shard")
    remote = next(m for m in matches if m.name.startswith("connector_heavy_"))

    assert remote.usage is None


def test_resolve_deferred_tool_call_returns_activation_hint() -> None:
    manager = _deferral_manager()

    failed = _resolve(manager, "workflow_status", {}).failed_calls[0]

    assert "Unknown tool 'workflow_status'" in failed.error
    assert "not activated" in failed.error
    assert "tool_search" in failed.error


def test_resolve_truly_unknown_tool_gets_plain_message() -> None:
    manager = _deferral_manager()

    failed = _resolve(manager, "totally_bogus", {}).failed_calls[0]

    assert failed.error == "Unknown tool 'totally_bogus'"


def test_resolve_background_call_never_hits_hint_path() -> None:
    manager = _deferral_manager()

    resolved = _resolve(manager, "background", {"action": "list"})

    assert resolved.failed_calls == []
    assert resolved.tool_calls[0].tool_name == "background"


def test_pin_manifest_tools_returns_empty_for_no_op_targets() -> None:
    manager = _deferral_manager()

    assert manager.pin_manifest_tools([]) == []
    assert manager.pin_manifest_tools(["nonexistent"]) == []

    disabled = _deferral_manager(disabled_tools=["schedule"])
    assert disabled.pin_manifest_tools(["schedule"]) == []
