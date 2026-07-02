from __future__ import annotations

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_app,
    build_test_vibe_config,
)
from vibe.cli.textual_ui.app import VibeApp
from vibe.core.config import ToolManifestConfig


def _app(*, defer: bool) -> VibeApp:
    config = build_test_vibe_config(
        tool_manifest=ToolManifestConfig(defer_builtin_tools=defer)
    )
    return build_test_vibe_app(
        config=config, agent_loop=build_test_agent_loop(config=config)
    )


def test_build_team_manager_activates_team_message_when_deferral_on() -> None:
    app = _app(defer=True)
    tool_manager = app.agent_loop.tool_manager
    assert "team_message" not in tool_manager.manifest_tools

    manager = app._build_team_manager()

    assert manager is not None
    assert "team_message" in tool_manager.manifest_tools


def test_build_team_manager_is_noop_when_deferral_off() -> None:
    app = _app(defer=False)
    tool_manager = app.agent_loop.tool_manager
    before = list(tool_manager.manifest_tools)

    assert tool_manager.pin_manifest_tools(["team_message"]) == []
    app._build_team_manager()

    assert list(tool_manager.manifest_tools) == before
