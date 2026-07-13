from __future__ import annotations

from pathlib import Path

import pytest

from tests.mock.utils import collect_result
from vibe.core.teams.models import TeamSafetyMode
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
    calls: list[tuple[str, str, str, int, bool, TeamSafetyMode]] = []

    async def spawn(
        name: str,
        prompt: str,
        agent: str,
        max_turns: int,
        worker: bool = False,
        safety_mode: TeamSafetyMode = TeamSafetyMode.SHARED,
    ) -> dict[str, str | bool]:
        calls.append((name, prompt, agent, max_turns, worker, safety_mode))
        return {
            "launch_id": "teamrun-1",
            "name": name,
            "team_dir": str(tmp_path),
            "message": f"Spawned teammate `{name}`.",
            "worker": worker,
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

    assert calls == [
        (
            "reviewer",
            "Review the latest performance diff.",
            "explore",
            3,
            False,
            TeamSafetyMode.SHARED,
        )
    ]
    assert result.name == "reviewer"
    assert result.launch_id == "teamrun-1"
    assert result.team_dir == str(tmp_path)
    assert "Spawned teammate" in result.message
    assert result.worker is False


@pytest.mark.asyncio
async def test_spawn_worker_flag_passed_to_callback(tmp_path: Path) -> None:
    calls: list[bool] = []

    async def spawn(
        name: str,
        prompt: str,
        agent: str,
        max_turns: int,
        worker: bool = False,
        safety_mode: TeamSafetyMode = TeamSafetyMode.SHARED,
    ) -> dict[str, str | bool]:
        del prompt, agent, max_turns, safety_mode
        calls.append(worker)
        return {
            "launch_id": "teamrun-2",
            "name": name,
            "team_dir": str(tmp_path),
            "message": f"Spawned worker `{name}`.",
            "worker": worker,
        }

    ctx = InvokeContext(tool_call_id="t1", team_spawn_callback=spawn)
    result = await collect_result(
        _make_tool().run(TeamSpawnArgs(name="w1", prompt="notes", worker=True), ctx=ctx)
    )
    assert calls == [True]
    assert result.worker is True


@pytest.mark.asyncio
async def test_spawn_passes_safety_mode_to_callback(tmp_path: Path) -> None:
    calls: list[TeamSafetyMode] = []

    async def spawn(
        name: str,
        prompt: str,
        agent: str,
        max_turns: int,
        worker: bool = False,
        safety_mode: TeamSafetyMode = TeamSafetyMode.SHARED,
    ) -> dict[str, str | bool]:
        del prompt, agent, max_turns, worker
        calls.append(safety_mode)
        return {
            "launch_id": "teamrun-3",
            "name": name,
            "team_dir": str(tmp_path),
            "message": f"Spawned teammate `{name}`.",
            "worker": False,
            "safety_mode": safety_mode.value,
        }

    ctx = InvokeContext(tool_call_id="t1", team_spawn_callback=spawn)
    result = await collect_result(
        _make_tool().run(
            TeamSpawnArgs(
                name="reviewer", prompt="review", safety_mode=TeamSafetyMode.SHARED_ASK
            ),
            ctx=ctx,
        )
    )

    assert calls == [TeamSafetyMode.SHARED_ASK]
    assert result.safety_mode is TeamSafetyMode.SHARED_ASK


@pytest.mark.asyncio
async def test_spawn_requires_context_callback() -> None:
    with pytest.raises(ToolError, match="not available"):
        await collect_result(
            _make_tool().run(
                TeamSpawnArgs(name="worker", prompt="Do work."),
                ctx=InvokeContext(tool_call_id="t1"),
            )
        )
