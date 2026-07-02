from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tomli_w

from tests import TESTS_ROOT
from tests.conftest import get_base_config
from tests.mock.utils import get_mocking_env, mock_llm_chunk
from vibe.core.types import FunctionCall, ToolCall
from vibe.core.worktree.ephemeral import create_ephemeral_worktree

MOCK_CLI = TESTS_ROOT / "mock" / "mock_cli_entrypoint.py"


def _init_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
    )


def _make_vibe_home(tmp_path: Path) -> Path:
    vibe_home = tmp_path / "vibe_home"
    vibe_home.mkdir()
    cfg = get_base_config()
    cfg["worktree"] = {"mode": "on"}
    (vibe_home / "config.toml").write_text(tomli_w.dumps(cfg), encoding="utf-8")
    (vibe_home / "trusted_folders.toml").write_text(
        "trusted = []\nuntrusted = []", encoding="utf-8"
    )
    return vibe_home


def _write_call(call_id: str, path: str, content: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(
            name="write_file", arguments=json.dumps({"path": path, "content": content})
        ),
    )


def _spawn_env(vibe_home: Path, chunks_env: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(chunks_env)
    env["MISTRAL_API_KEY"] = "mock"
    env["VIBE_HOME"] = str(vibe_home)
    env["VIBE_TEST_DISABLE_KEYRING"] = "1"
    env.pop("VIBE_ISOLATED_WORKTREE_ROOT", None)
    env.pop("VIBE_ISOLATED_SCRATCHPAD_DIR", None)
    return env


def _run_cli(
    args: list[str], cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MOCK_CLI), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.timeout(240)
def test_isolated_child_skips_nested_worktree_and_honors_scratchpad_grant(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    vibe_home = _make_vibe_home(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    wt = create_ephemeral_worktree(repo, "e2e", base_dir=tmp_path / "iso")

    chunks_env = get_mocking_env([
        mock_llm_chunk(
            content="",
            tool_calls=[
                _write_call("c1", "out.txt", "spawn-e2e"),
                _write_call("c2", str(scratch / "note.txt"), "grant-ok"),
            ],
        ),
        mock_llm_chunk(content="done"),
    ])
    env = _spawn_env(vibe_home, chunks_env)
    env["VIBE_ISOLATED_WORKTREE_ROOT"] = str(wt.path)
    env["VIBE_ISOLATED_SCRATCHPAD_DIR"] = str(scratch)

    proc = _run_cli(["-p", "write the files", "--auto-approve"], cwd=wt.path, env=env)
    assert proc.returncode == 0, proc.stderr

    assert (wt.path / "out.txt").read_text() == "spawn-e2e"
    assert (scratch / "note.txt").read_text() == "grant-ok"

    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    lines = [ln for ln in worktrees.splitlines() if ln.startswith("worktree ")]
    assert len(lines) == 2, worktrees

    branches = subprocess.run(
        ["git", "branch", "--list", "vibe/cli-*", "vibe/programmatic-*"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branches == ""


@pytest.mark.timeout(240)
def test_plain_programmatic_child_enters_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    vibe_home = _make_vibe_home(tmp_path)

    chunks_env = get_mocking_env([
        mock_llm_chunk(
            content="", tool_calls=[_write_call("c1", "out.txt", "worktree-e2e")]
        ),
        mock_llm_chunk(content="done"),
    ])
    env = _spawn_env(vibe_home, chunks_env)

    proc = _run_cli(["-p", "write the file", "--auto-approve"], cwd=repo, env=env)
    assert proc.returncode == 0, proc.stderr

    assert not (repo / "out.txt").exists()

    branches = subprocess.run(
        ["git", "branch", "--list", "vibe/programmatic-*"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert branches, "programmatic child under worktree.mode=on must branch"
    branch = branches.split()[-1]
    shown = subprocess.run(
        ["git", "show", f"{branch}:out.txt"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert shown == "worktree-e2e"
