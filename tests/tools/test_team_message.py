from __future__ import annotations

from pathlib import Path

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.team_message import (
    TeamMessage,
    TeamMessageArgs,
    TeamMessageConfig,
)


def _make_tool() -> TeamMessage:
    return TeamMessage(config_getter=lambda: TeamMessageConfig(), state=BaseToolState())


def _ctx(team_dir: str | None) -> InvokeContext:
    return InvokeContext(tool_call_id="t1", team_dir_callback=lambda: team_dir)


@pytest.mark.asyncio
async def test_send_and_read_roundtrip(tmp_path: Path) -> None:
    # The lead posts to a teammate's inbox, then reads its own ("lead") inbox.
    team_dir = str(tmp_path)
    # A teammate (simulated) writes a reply into the lead's inbox directly.
    from vibe.core.teams.mailbox import Mailbox

    mailbox = Mailbox(tmp_path)
    mailbox.send("worker", "lead", "done, found 3 issues")

    send = await collect_result(
        _make_tool().run(
            TeamMessageArgs(
                action="send_message", to_name="worker", content="please audit src/"
            ),
            ctx=_ctx(team_dir),
        )
    )
    assert "worker" in send.message

    read = await collect_result(
        _make_tool().run(TeamMessageArgs(action="read_messages"), ctx=_ctx(team_dir))
    )
    assert read.messages is not None
    assert len(read.messages) == 1
    assert read.messages[0]["content"] == "done, found 3 issues"
    assert read.messages[0]["from_name"] == "worker"
    assert read.messages[0]["to_name"] == "lead"


@pytest.mark.asyncio
async def test_send_requires_to_name_and_content(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="to_name and content"):
        await collect_result(
            _make_tool().run(
                TeamMessageArgs(action="send_message"), ctx=_ctx(str(tmp_path))
            )
        )


@pytest.mark.asyncio
async def test_unread_messages(tmp_path: Path) -> None:
    from vibe.core.teams.mailbox import Mailbox

    mailbox = Mailbox(tmp_path)
    mailbox.send("worker", "lead", "first")
    mailbox.send("worker", "lead", "second")
    result = await collect_result(
        _make_tool().run(
            TeamMessageArgs(action="unread_messages"), ctx=_ctx(str(tmp_path))
        )
    )
    assert result.messages is not None
    assert len(result.messages) == 2


@pytest.mark.asyncio
async def test_errors_when_no_active_team() -> None:
    with pytest.raises(ToolError, match="No active team"):
        await collect_result(
            _make_tool().run(TeamMessageArgs(action="read_messages"), ctx=_ctx(None))
        )


@pytest.mark.asyncio
async def test_errors_without_context() -> None:
    with pytest.raises(ToolError, match="requires context"):
        await collect_result(_make_tool().run(TeamMessageArgs(action="read_messages")))


@pytest.mark.asyncio
async def test_unknown_action_errors(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="Unknown action"):
        await collect_result(
            _make_tool().run(TeamMessageArgs(action="nope"), ctx=_ctx(str(tmp_path)))
        )
