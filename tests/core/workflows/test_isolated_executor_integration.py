"""RT-007 integration tests for workflow isolation.

Test A drives the REAL _default_isolated_executor: a real ephemeral git worktree
+ a real subprocess (a fake `vibe` pointed to via VIBE_ISOLATED_EXECUTOR_CMD) +
the real __VIBE_WORKFLOW_STATS__ stderr round trip + real worktree cleanup — the
only faked part is the `vibe` binary (a real one would need an API key).

Test B proves a `vibe -p`-style AgentLoop (what the 'worker' profile runs)
surfaces a configured MCP server's tools, mirroring run_programmatic's
construction (defer_heavy_init unset -> synchronous MCP integration), and that a
restricted enabled_tools allowlist filters them out — no subprocess, no network.
"""

from __future__ import annotations

import shlex
import sys
from types import SimpleNamespace
from typing import Any

from git import Repo
import pytest

import vibe.core.worktree.ephemeral as ephemeral
from vibe.core.config import MCPStdio
from vibe.core.workflows.runtime import WorkflowRuntime
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

_FAKE_VIBE = """\
import json, os, sys
sys.stdout.write("ISO-OUTPUT-OK\\n")
sys.stdout.write("CWD=" + os.getcwd() + "\\n")
if os.environ.get("VIBE_WORKFLOW_EMIT_STATS") == "1":
    sys.stderr.write(
        "\\n__VIBE_WORKFLOW_STATS__"
        + json.dumps({"prompt_tokens": 100, "completion_tokens": 50})
        + "\\n"
    )
sys.exit(0)
"""


@pytest.mark.asyncio
async def test_default_isolated_executor_end_to_end(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real git repo (worktree add needs a commit), cwd'd into so the executor's
    # create_ephemeral_worktree(Path.cwd(), ...) resolves to it.
    repo = Repo.init(str(tmp_path))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "t@t.com")
    (tmp_path / "f.txt").write_text("base\n")
    repo.index.add(["f.txt"])
    repo.index.commit("init")
    monkeypatch.chdir(tmp_path)

    # Keep worktree artifacts out of the real ~/.vibe.
    monkeypatch.setattr(ephemeral, "VIBE_HOME", SimpleNamespace(path=tmp_path / "vh"))

    # Point the executor at a fake `vibe` that emits the real sentinel.
    fake_vibe = tmp_path / "fake_vibe.py"
    fake_vibe.write_text(_FAKE_VIBE)
    monkeypatch.setenv(
        "VIBE_ISOLATED_EXECUTOR_CMD",
        f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_vibe))}",
    )

    # Spy on cleanup to capture the worktree path before it's removed.
    captured: dict[str, Any] = {}
    real_remove = ephemeral.remove_ephemeral_worktree

    def spy_remove(wt: Any, **kw: Any) -> bool:
        captured["path"] = wt.path
        return real_remove(wt, **kw)

    monkeypatch.setattr(ephemeral, "remove_ephemeral_worktree", spy_remove)

    rt = WorkflowRuntime()  # factory unused; the executor is called directly
    output, stats, _ = await rt._default_isolated_executor(
        "hi", "auto-approve", "lbl", 40
    )

    # Real subprocess output captured; real sentinel parsed into stats.
    assert "ISO-OUTPUT-OK" in output
    assert stats == {"prompt_tokens": 100, "completion_tokens": 50}

    # The subprocess actually ran inside the ephemeral worktree.
    wt_path = captured["path"]
    cwd_line = next(ln for ln in output.splitlines() if ln.startswith("CWD="))
    from pathlib import Path

    assert Path(cwd_line[len("CWD=") :]).resolve() == wt_path.resolve()

    # Worktree was cleaned up (clean tree -> removed).
    assert not wt_path.exists()


def test_programmatic_style_agentloop_exposes_configured_mcp_tool() -> None:
    # Mirrors run_programmatic's AgentLoop construction (defer_heavy_init unset
    # -> False -> ToolManager integrates MCP synchronously at init). The 'worker'
    # profile has no enabled_tools allowlist, so the MCP tool surfaces.
    mcp_server = MCPStdio(name="srv", transport="stdio", command="echo")
    config = build_test_vibe_config(mcp_servers=[mcp_server])
    loop = build_test_agent_loop(
        config=config, agent_name="auto-approve", mcp_registry=FakeMCPRegistry()
    )
    # FakeMCPRegistry publishes "<server>_<tool>" -> "srv_fake_tool".
    assert "srv_fake_tool" in loop.tool_manager.available_tools


def test_restricted_allowlist_filters_out_mcp_tool() -> None:
    # Why restricted subagents (explore/research/reviewer) don't get MCP and the
    # full-tool 'worker' does: an enabled_tools allowlist drops MCP tools (their
    # names have no common prefix to allowlist).
    mcp_server = MCPStdio(name="srv", transport="stdio", command="echo")
    config = build_test_vibe_config(mcp_servers=[mcp_server], enabled_tools=["read"])
    loop = build_test_agent_loop(config=config, mcp_registry=FakeMCPRegistry())
    assert "srv_fake_tool" not in loop.tool_manager.available_tools
