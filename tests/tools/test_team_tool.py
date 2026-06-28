from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.base import ToolError
from vibe.core.tools.builtins.team import Team, TeamArgs, TeamState, TeamToolConfig


def _make_tool() -> Team:
    return Team(config_getter=lambda: TeamToolConfig(), state=TeamState())


def _set_teammate_env(tmp_path: Path, name: str = "alice") -> None:
    os.environ["VIBE_TEAM_DIR"] = str(tmp_path)
    os.environ["VIBE_TEAMMATE_NAME"] = name


def _clear_teammate_env(saved: dict[str, str | None]) -> None:
    for key in ("VIBE_TEAM_DIR", "VIBE_TEAMMATE_NAME"):
        val = saved.get(key)
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


@pytest.fixture(autouse=True)
def preserve_env():
    saved = {
        "VIBE_TEAM_DIR": os.environ.get("VIBE_TEAM_DIR"),
        "VIBE_TEAMMATE_NAME": os.environ.get("VIBE_TEAMMATE_NAME"),
    }
    yield
    _clear_teammate_env(saved)


def test_is_available_only_when_teammate_env_set(tmp_path: Path) -> None:
    """teams-003: the team tool must be available only inside a teammate
    (VIBE_TEAM_DIR set), so the lead does not get a duplicate of /team.
    """
    os.environ.pop("VIBE_TEAM_DIR", None)
    assert Team.is_available(None) is False

    _set_teammate_env(tmp_path)
    assert Team.is_available(None) is True


@pytest.mark.asyncio
async def test_teammate_claims_shared_task(tmp_path: Path) -> None:
    """teams-003: a teammate binds the shared TaskStore via VIBE_TEAM_DIR and
    can claim a task the lead created. Previously VIBE_TEAM_* was set but never
    read, so teammates could not reach the shared state.
    """
    from vibe.core.teams.task_store import TaskStore

    # Lead creates a task in the shared store.
    lead_store = TaskStore(tmp_path)
    lead_store.add_task("Write the auth module")

    _set_teammate_env(tmp_path, name="alice")
    result = await collect_result(
        _make_tool().run(TeamArgs(action="claim_task", task_id="task-1"))
    )
    assert result.task is not None
    assert result.task["assignee"] == "alice"
    assert result.task["status"] == "in_progress"

    # The claim is visible to the lead's store after a reload (the teammate
    # wrote it under the lock; the lead's in-memory cache is stale until then).
    lead_store.reload()
    reloaded = lead_store.get_task("task-1")
    assert reloaded is not None
    assert reloaded.assignee == "alice"


@pytest.mark.asyncio
async def test_teammate_send_and_read_message(tmp_path: Path) -> None:
    """teams-003: teammates can message each other through the shared Mailbox."""
    _set_teammate_env(tmp_path, name="alice")

    # Alice sends to bob.
    await collect_result(
        _make_tool().run(
            TeamArgs(action="send_message", to_name="bob", content="hi bob")
        )
    )

    # Bob reads.
    os.environ["VIBE_TEAMMATE_NAME"] = "bob"
    result = await collect_result(_make_tool().run(TeamArgs(action="read_messages")))
    assert result.messages is not None
    assert len(result.messages) == 1
    assert result.messages[0]["content"] == "hi bob"
    assert result.messages[0]["from_name"] == "alice"


def test_rejects_action_outside_teammate(tmp_path: Path) -> None:
    """Without VIBE_TEAM_DIR the tool refuses to bind (defence-in-depth even
    though is_available hides it from non-teammates).
    """
    os.environ.pop("VIBE_TEAM_DIR", None)
    tool = _make_tool()
    with pytest.raises(ToolError, match="VIBE_TEAM_DIR"):
        tool._bind()
