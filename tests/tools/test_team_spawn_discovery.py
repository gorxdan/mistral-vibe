from __future__ import annotations

from tests.conftest import build_test_vibe_config
from vibe.core.tools.manager import ToolManager


def test_team_spawn_is_discovered_as_builtin_tool() -> None:
    manager = ToolManager(lambda: build_test_vibe_config())

    assert "team_spawn" in manager.available_tools
