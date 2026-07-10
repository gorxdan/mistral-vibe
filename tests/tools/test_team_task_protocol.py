from __future__ import annotations

from pathlib import Path

import pytest

from tests.mock.utils import collect_result
from vibe.core.tasking import TaskBrief, TaskManifestIdentity, TaskOutcomeStatus
from vibe.core.teams.models import TaskStatus
from vibe.core.teams.task_store import TaskStore
from vibe.core.tools.base import ToolError
from vibe.core.tools.builtins.team import Team, TeamArgs, TeamState, TeamToolConfig


def _tool() -> Team:
    return Team(config_getter=lambda: TeamToolConfig(), state=TeamState())


def _brief() -> TaskBrief:
    return TaskBrief(
        objective="Implement the parser fix",
        inputs={"target": "vibe/core/parser.py:10"},
        allowed_paths=["vibe/core/parser.py"],
        denied_paths=["vibe/core/agent_loop.py"],
        acceptance_checks=["uv run pytest tests/core/test_parser.py"],
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )


@pytest.mark.asyncio
async def test_team_tool_cannot_self_complete_structured_task(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "alice")
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    assert store.claim_task(task.id, "alice") is not None

    with pytest.raises(ToolError, match="Could not complete task"):
        await collect_result(
            _tool().run(
                TeamArgs(
                    action="complete_task",
                    task_id=task.id,
                    description="Implemented and tests pass",
                )
            )
        )

    store.reload()
    claimed = store.get_task(task.id)
    assert claimed is not None
    assert claimed.status is TaskStatus.IN_PROGRESS
    assert claimed.assignee == "alice"
    assert claimed.outcome is None


@pytest.mark.asyncio
async def test_team_tool_preserves_legacy_completion_message(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "alice")
    store = TaskStore(tmp_path)
    task = store.add_task("Legacy task")
    assert store.claim_task(task.id, "alice") is not None

    result = await collect_result(
        _tool().run(
            TeamArgs(action="complete_task", task_id=task.id, description="done")
        )
    )

    assert result.message == f"Completed task {task.id}."
    assert result.task is not None
    assert result.task["status"] == TaskStatus.COMPLETED.value
    assert result.task["outcome"]["status"] == TaskOutcomeStatus.SUCCEEDED.value
