from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.tools.builtins.launch_workflow import LaunchWorkflow
from vibe.core.tools.builtins.task import Task
from vibe.core.tools.builtins.team_spawn import TeamSpawn
from vibe.core.tools.manager import NoSuchToolError, ToolManager


@pytest.mark.parametrize(
    ("name", "tool_class"),
    [("task", Task), ("launch_workflow", LaunchWorkflow), ("team_spawn", TeamSpawn)],
)
def test_spawner_tools_are_host_only(name: str, tool_class: type) -> None:
    config = build_test_vibe_config()
    host = ToolManager(lambda: config)
    subagent = ToolManager(lambda: config, host=False)

    assert tool_class.host_only is True
    assert name in host.available_tools
    assert name not in subagent.available_tools
    with pytest.raises(NoSuchToolError):
        subagent.get(name)


def test_authoritative_task_allowlist_cannot_restore_spawner_for_subagent() -> None:
    config = build_test_vibe_config()
    subagent = ToolManager(
        lambda: config,
        runtime_allowlist=frozenset({"task"}),
        authoritative_runtime_allowlist=True,
        host=False,
    )

    assert "task" not in subagent.available_tools
    with pytest.raises(NoSuchToolError):
        subagent.get("task")
