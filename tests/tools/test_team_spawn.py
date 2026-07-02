from __future__ import annotations

from pathlib import Path

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError, ToolPermission
from vibe.core.tools.builtins.team_spawn import (
    TeamSpawn,
    TeamSpawnArgs,
    TeamSpawnConfig,
)


def _make_tool(config: TeamSpawnConfig | None = None) -> TeamSpawn:
    resolved_config = config or TeamSpawnConfig()
    return TeamSpawn(config_getter=lambda: resolved_config, state=BaseToolState())


@pytest.mark.parametrize("configured", [ToolPermission.ALWAYS, ToolPermission.NEVER])
def test_resolve_permission_honors_config_override(configured: ToolPermission) -> None:
    tool = _make_tool(TeamSpawnConfig(permission=configured))

    permission = tool.resolve_permission(
        TeamSpawnArgs(name="worker", prompt="Do work.")
    )

    assert permission is not None
    assert permission.permission is configured


@pytest.mark.asyncio
async def test_spawn_uses_context_callback_and_returns_team_dir(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str, int]] = []

    async def spawn(
        name: str, prompt: str, agent: str, max_turns: int
    ) -> dict[str, str]:
        calls.append((name, prompt, agent, max_turns))
        return {
            "name": name,
            "team_dir": str(tmp_path),
            "message": f"Spawned teammate `{name}`.",
        }

    ctx = InvokeContext(tool_call_id="t1", team_spawn_callback=spawn)

    result = await collect_result(
        _make_tool().run(
            TeamSpawnArgs(
                name="reviewer",
                prompt="Review the latest performance diff.",
                agent="explore",
                max_turns=3,
            ),
            ctx=ctx,
        )
    )

    assert calls == [("reviewer", "Review the latest performance diff.", "explore", 3)]
    assert result.name == "reviewer"
    assert result.team_dir == str(tmp_path)
    assert "Spawned teammate" in result.message


@pytest.mark.asyncio
async def test_spawn_requires_context_callback() -> None:
    with pytest.raises(ToolError, match="not available"):
        await collect_result(
            _make_tool().run(
                TeamSpawnArgs(name="worker", prompt="Do work."),
                ctx=InvokeContext(tool_call_id="t1"),
            )
        )
