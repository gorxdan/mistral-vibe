from __future__ import annotations

from pathlib import Path

import pytest

from tests.mock.utils import collect_result
from vibe.core.config import SandboxConfig
from vibe.core.teams._escalate import EscalationDenied
from vibe.core.teams._safety import TEAM_SAFETY_MODE_ENV
from vibe.core.tools.base import BaseToolState, ToolError
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.builtins.edit import Edit, EditArgs, EditConfig
from vibe.core.tools.builtins.write_file import (
    WriteFile,
    WriteFileArgs,
    WriteFileConfig,
)


def _enable_shared_ask(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    team_dir = tmp_path / "team"
    team_dir.mkdir()
    monkeypatch.setenv("VIBE_TEAM_DIR", str(team_dir))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "alice")
    monkeypatch.setenv(TEAM_SAFETY_MODE_ENV, "shared-ask")


@pytest.mark.asyncio
async def test_bash_shared_ask_denial_blocks_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_shared_ask(monkeypatch, tmp_path)
    calls: list[tuple[str, str]] = []
    subprocess_called = False

    async def deny(tool: str, description: str) -> None:
        calls.append((tool, description))
        raise EscalationDenied("denied by lead")

    async def subprocess(*_args, **_kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        raise AssertionError("subprocess should not start")

    monkeypatch.setattr("vibe.core.teams._safety.escalate_to_lead", deny)
    monkeypatch.setattr("asyncio.create_subprocess_shell", subprocess)
    monkeypatch.setattr("asyncio.create_subprocess_exec", subprocess)

    bash = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())
    with pytest.raises(ToolError, match="denied by lead"):
        await collect_result(bash.run(BashArgs(command="rm -rf build")))

    assert calls
    assert calls[0][0] == "bash"
    assert "rm -rf build" in calls[0][1]
    assert subprocess_called is False


@pytest.mark.asyncio
async def test_bash_shared_ask_skips_escalation_for_allowlisted_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_shared_ask(monkeypatch, tmp_path)

    async def fail_if_called(tool: str, description: str) -> None:
        raise AssertionError(f"unexpected escalation for {tool}: {description}")

    monkeypatch.setattr("vibe.core.teams._safety.escalate_to_lead", fail_if_called)
    config = BashToolConfig(sandbox=SandboxConfig(enabled=False))
    bash = Bash(config_getter=lambda: config, state=BaseToolState())

    result = await collect_result(bash.run(BashArgs(command="echo hi")))

    assert result.stdout == "hi\n"


@pytest.mark.asyncio
async def test_write_file_shared_ask_denial_leaves_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_shared_ask(monkeypatch, tmp_path)

    async def deny(_tool: str, _description: str) -> None:
        raise EscalationDenied("no write")

    monkeypatch.setattr("vibe.core.teams._safety.escalate_to_lead", deny)
    tool = WriteFile(config_getter=lambda: WriteFileConfig(), state=BaseToolState())
    target = tmp_path / "new" / "file.txt"

    with pytest.raises(ToolError, match="no write"):
        await collect_result(tool.run(WriteFileArgs(path=str(target), content="x")))

    assert not target.exists()
    assert not target.parent.exists()


@pytest.mark.asyncio
async def test_edit_shared_ask_denial_leaves_file_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable_shared_ask(monkeypatch, tmp_path)

    async def deny(_tool: str, _description: str) -> None:
        raise EscalationDenied("no edit")

    monkeypatch.setattr("vibe.core.teams._safety.escalate_to_lead", deny)
    target = tmp_path / "f.txt"
    target.write_text("hello world\n")
    tool = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())

    with pytest.raises(ToolError, match="no edit"):
        await collect_result(
            tool.run(
                EditArgs(
                    file_path=str(target), old_string="hello", new_string="goodbye"
                )
            )
        )

    assert target.read_text() == "hello world\n"
