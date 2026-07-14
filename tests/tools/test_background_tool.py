from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from tests.mock.utils import collect_result
from vibe.core.config import SandboxConfig
from vibe.core.tools.background import BackgroundRegistry, TaskCategory

if TYPE_CHECKING:
    from vibe.cli.textual_ui.workflow_runner import WorkflowRunner
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.background import (
    Background,
    BackgroundArgs,
    BackgroundToolConfig,
)
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig

pytestmark = pytest.mark.process_e2e

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bash(config: BashToolConfig | None = None) -> Bash:
    return Bash(config_getter=lambda: config or BashToolConfig(), state=BaseToolState())


def _ctx(
    registry: BackgroundRegistry | None,
    *,
    session_dir: Path | None = None,
    scratchpad_dir: Path | None = None,
) -> InvokeContext:
    return InvokeContext(
        tool_call_id="test",
        background_registry=registry,
        session_dir=session_dir,
        scratchpad_dir=scratchpad_dir,
    )


def _background_tool() -> Background:
    return Background(
        config_getter=lambda: BackgroundToolConfig(), state=BaseToolState()
    )


@pytest.fixture
async def reaping_registry():
    """Yield a BackgroundRegistry that reaps every still-running process on
    teardown, so a failing assertion can't orphan a backgrounded server.
    """
    reg = BackgroundRegistry()
    yield reg
    for rec in list(reg._procs.values()):
        if rec.status == "running":
            try:
                await reg.stop(rec.task_id)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Bash background branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_returns_immediately_with_handle(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    result = await collect_result(
        bash.run(BashArgs(command="sleep 5", background=True), ctx=ctx)
    )

    assert result.background_task_id == "proc-1"
    assert result.pid is not None and result.pid > 0
    assert result.returncode == -1  # still-running sentinel

    # The process is genuinely running and tracked by the registry.
    [entry] = registry.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "running"
    assert entry.task_id == "proc-1"

    # Clean up — stop reaps the process so it doesn't outlive the test.
    assert await registry.stop("proc-1") is True


@pytest.mark.asyncio
async def test_background_writes_output_to_log_file(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    result = await collect_result(
        bash.run(BashArgs(command="echo hello-from-bg", background=True), ctx=ctx)
    )

    # Give the shell a moment to write + flush + exit.
    await asyncio.sleep(0.3)

    assert result.background_task_id is not None
    tail = registry.read_log_tail(result.background_task_id)
    assert "hello-from-bg" in tail


@pytest.mark.asyncio
async def test_background_without_registry_raises(tmp_path):
    bash = _bash()
    ctx = _ctx(None, session_dir=tmp_path)  # no registry

    with pytest.raises(ToolError, match="background execution is not available"):
        await collect_result(
            bash.run(BashArgs(command="echo x", background=True), ctx=ctx)
        )


@pytest.mark.asyncio
async def test_background_without_session_or_scratchpad_raises():
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry)  # no session_dir, no scratchpad_dir

    with pytest.raises(ToolError, match="scratchpad or session directory"):
        await collect_result(
            bash.run(BashArgs(command="echo x", background=True), ctx=ctx)
        )


@pytest.mark.asyncio
async def test_background_false_is_byte_identical_to_foreground(tmp_path):
    bash = _bash()
    _ctx(None, session_dir=tmp_path)

    result = await collect_result(bash.run(BashArgs(command="echo fg")))

    assert result.returncode == 0
    assert result.stdout.strip() == "fg"
    assert result.background_task_id is None
    assert result.pid is None


@pytest.mark.asyncio
async def test_background_stop_reaps_process(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    result = await collect_result(
        bash.run(BashArgs(command="sleep 30", background=True), ctx=ctx)
    )

    assert result.background_task_id is not None
    stopped = await registry.stop(result.background_task_id)
    assert stopped is True
    # Give the OS a moment to reflect the signal.
    await asyncio.sleep(0.2)

    [entry] = registry.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "stopped"


# ---------------------------------------------------------------------------
# background agent tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_tool_list_empty():
    tool = _background_tool()
    registry = BackgroundRegistry()
    ctx = _ctx(registry)

    result = await collect_result(tool.run(BackgroundArgs(action="list"), ctx=ctx))

    assert "No background tasks" in result.response


@pytest.mark.asyncio
async def test_background_tool_lists_spawned_process(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    await collect_result(
        bash.run(BashArgs(command="sleep 30", background=True), ctx=ctx)
    )
    bg_ctx = _ctx(registry)
    tool = _background_tool()

    result = await collect_result(tool.run(BackgroundArgs(action="list"), ctx=bg_ctx))

    assert "proc-1" in result.response
    assert "running" in result.response
    await registry.stop("proc-1")


# ---------------------------------------------------------------------------
# Scoped list (action='list' + task_id) — single-task lookup + tail scoping
# ---------------------------------------------------------------------------


async def _spawn_marker_process(bash, ctx, marker: str) -> str:
    """Background a shell that prints a marker line then stays alive, so the
    entry remains 'running' and the marker lands in its log. Returns task_id.
    """
    result = await collect_result(
        bash.run(
            BashArgs(command=f"sh -c 'echo {marker}; sleep 30'", background=True),
            ctx=ctx,
        )
    )
    # Wait for the marker to flush into the log before returning.
    for _ in range(20):
        await asyncio.sleep(0.1)
        if marker in registry_log_tail(ctx, result.background_task_id):
            break
    return result.background_task_id


def registry_log_tail(ctx: InvokeContext, task_id: str) -> str:
    assert ctx.background_registry is not None
    return ctx.background_registry.read_log_tail(task_id, lines=10)


def _has_log_block(response: str) -> bool:
    """_format_entry wraps a rendered tail in a fenced block; its presence is
    the reliable signal that a log tail was attached (independent of whatever
    text the command label itself happens to contain).
    """
    return "```" in response


@pytest.mark.asyncio
async def test_scoped_list_returns_only_the_named_task(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    await _spawn_marker_process(bash, ctx, "marker-one")  # proc-1
    await _spawn_marker_process(bash, ctx, "marker-two")  # proc-2
    tool = _background_tool()

    result = await collect_result(
        tool.run(BackgroundArgs(action="list", task_id="proc-1"), ctx=_ctx(registry))
    )

    assert "proc-1" in result.response
    assert "proc-2" not in result.response  # exact match, not prefix
    assert _has_log_block(result.response)  # default tail kicked in (scoped)
    await registry.stop("proc-1")
    await registry.stop("proc-2")


@pytest.mark.asyncio
async def test_scoped_list_shows_recent_log_by_default(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    task_id = await _spawn_marker_process(bash, ctx, "scoped-marker")
    tool = _background_tool()

    result = await collect_result(
        tool.run(BackgroundArgs(action="list", task_id=task_id), ctx=_ctx(registry))
    )

    assert _has_log_block(result.response)
    assert "scoped-marker" in result.response
    await registry.stop(task_id)


@pytest.mark.asyncio
async def test_scoped_list_tail_zero_suppresses_log(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    task_id = await _spawn_marker_process(bash, ctx, "suppress-me")
    tool = _background_tool()

    result = await collect_result(
        tool.run(
            BackgroundArgs(action="list", task_id=task_id, tail=0), ctx=_ctx(registry)
        )
    )

    assert not _has_log_block(result.response)  # tail suppressed
    assert task_id in result.response  # entry still shown, just without log
    await registry.stop(task_id)


@pytest.mark.asyncio
async def test_scoped_list_unknown_id_lists_known_ids(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    await _spawn_marker_process(bash, ctx, "known-one")  # proc-1
    tool = _background_tool()

    result = await collect_result(
        tool.run(BackgroundArgs(action="list", task_id="proc-999"), ctx=_ctx(registry))
    )

    assert "No background task" in result.response
    assert "proc-999" in result.response
    assert "proc-1" in result.response  # known-ids hint
    await registry.stop("proc-1")


@pytest.mark.asyncio
async def test_scoped_list_explicit_tail_overrides_default(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    task_id = await _spawn_marker_process(bash, ctx, "explicit-tail")
    tool = _background_tool()

    result = await collect_result(
        tool.run(
            BackgroundArgs(action="list", task_id=task_id, tail=5), ctx=_ctx(registry)
        )
    )

    assert _has_log_block(result.response)
    assert "explicit-tail" in result.response
    await registry.stop(task_id)


# ---------------------------------------------------------------------------
# Family scoping — '/' boundary match pulls in hierarchical children
# ---------------------------------------------------------------------------


@dataclass
class _FakeLiveAgent:
    agent_id: str
    label: str = "agent"
    phase: str = "default"
    tokens_total: int = 0
    agent: str | None = "explore"
    model: str | None = None
    log_path: Path | None = None


class _FakeStatus:
    def __init__(self, value: str) -> None:
        self.value = value


@dataclass
class _FakeRunEntry:
    run_id: str
    status: _FakeStatus
    elapsed: float = 0.0
    agent_count: int = 0
    tokens_total: int = 0
    phases: list[str] = field(default_factory=list)
    live_agents: list[_FakeLiveAgent] = field(default_factory=list)


class _FakeWorkflowRunner:
    """Minimal stand-in matching the shape _workflow_entries reads: a `.runs`
    list of entries, each with live_agents. Lets us exercise family scoping
    without spinning up a real workflow run.
    """

    def __init__(self) -> None:
        self.runs: list[_FakeRunEntry] = []

    async def stop(self, run_id: str) -> bool:
        return any(r.run_id == run_id for r in self.runs)

    def cancel_agent(self, run_id: str, agent_id: str) -> bool:
        return True

    def pause(self, run_id: str) -> bool:
        return True

    def unpause(self, run_id: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_family_scoping_pulls_in_workflow_children():
    registry = BackgroundRegistry()
    wf = _FakeWorkflowRunner()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_FakeStatus("running"),
            phases=["reviews"],
            live_agents=[
                _FakeLiveAgent(agent_id="explore", label="scout"),
                _FakeLiveAgent(agent_id="reviewer", label="checker"),
            ],
        )
    )
    registry.attach_workflow_runner(lambda: cast("WorkflowRunner", wf))
    tool = _background_tool()

    result = await collect_result(
        tool.run(BackgroundArgs(action="list", task_id="wf-1"), ctx=_ctx(registry))
    )

    assert "wf-1" in result.response
    assert "wf-1/live-explore" in result.response
    assert "wf-1/live-reviewer" in result.response


@pytest.mark.asyncio
async def test_family_scoping_isolates_sibling_workflows():
    registry = BackgroundRegistry()
    wf = _FakeWorkflowRunner()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_FakeStatus("running"),
            live_agents=[_FakeLiveAgent(agent_id="explore")],
        )
    )
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-2",
            status=_FakeStatus("running"),
            live_agents=[_FakeLiveAgent(agent_id="explore")],
        )
    )
    registry.attach_workflow_runner(lambda: cast("WorkflowRunner", wf))
    tool = _background_tool()

    result = await collect_result(
        tool.run(BackgroundArgs(action="list", task_id="wf-1"), ctx=_ctx(registry))
    )

    assert "wf-1" in result.response
    assert "wf-2" not in result.response
    assert "wf-2/live-explore" not in result.response


@pytest.mark.asyncio
async def test_family_scoping_has_no_numeric_footgun(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    # Spawn 10 processes so proc-1 and proc-10 both exist. Plain sleep — no
    # marker needed, registration is synchronous so ids are assigned at once.
    for _ in range(10):
        await collect_result(
            bash.run(BashArgs(command="sleep 30", background=True), ctx=ctx)
        )
    tool = _background_tool()

    result = await collect_result(
        tool.run(BackgroundArgs(action="list", task_id="proc-1"), ctx=_ctx(registry))
    )

    assert "proc-1" in result.response
    # proc-10 must NOT appear — only proc-1 (and no proc-1/ children exist).
    assert "proc-10" not in result.response
    # Every rendered entry line that names a proc should be proc-1 only.
    proc_lines = [line for line in result.response.splitlines() if "proc-" in line]
    assert proc_lines, "expected at least one proc entry"
    for line in proc_lines:
        assert line.lstrip().startswith("- proc-1 "), (
            f"unexpected sibling leaked into proc-1 scope: {line!r}"
        )

    for rec in list(registry._procs.values()):
        if rec.status == "running":
            await registry.stop(rec.task_id)


# ---------------------------------------------------------------------------
# Agent log tailing — scoped workflow lookup shows child agent transcripts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_workflow_list_tails_child_agent_transcript(tmp_path):
    import json

    log = tmp_path / "agent-transcript.jsonl"
    log.write_text(
        json.dumps({"role": "user", "content": "audit the auth module"})
        + "\n"
        + json.dumps({"role": "assistant", "content": "grepping for login routes"})
        + "\n"
        + json.dumps({"role": "assistant", "content": "login is handled in auth.py"})
        + "\n"
    )
    registry = BackgroundRegistry()
    wf = _FakeWorkflowRunner()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_FakeStatus("running"),
            phases=["audit"],
            live_agents=[
                _FakeLiveAgent(agent_id="explore", label="auditor", log_path=log)
            ],
        )
    )
    registry.attach_workflow_runner(lambda: cast("WorkflowRunner", wf))
    tool = _background_tool()

    result = await collect_result(
        tool.run(BackgroundArgs(action="list", task_id="wf-1"), ctx=_ctx(registry))
    )

    assert "wf-1" in result.response
    assert "wf-1/live-explore" in result.response
    # The agent's transcript tail renders inline, formatted (not raw JSON).
    assert _has_log_block(result.response)
    assert "login is handled in auth.py" in result.response
    assert '"role"' not in result.response  # JSON envelope stripped


@pytest.mark.asyncio
async def test_scoped_agent_list_tails_single_agent(tmp_path):
    import json

    log = tmp_path / "single-agent.jsonl"
    log.write_text(
        json.dumps({"role": "assistant", "content": "narrow result here"}) + "\n"
    )
    registry = BackgroundRegistry()
    wf = _FakeWorkflowRunner()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_FakeStatus("running"),
            live_agents=[
                _FakeLiveAgent(agent_id="explore", log_path=log),
                _FakeLiveAgent(agent_id="reviewer", label="checker"),
            ],
        )
    )
    registry.attach_workflow_runner(lambda: cast("WorkflowRunner", wf))
    tool = _background_tool()

    result = await collect_result(
        tool.run(
            BackgroundArgs(action="list", task_id="wf-1/live-explore"),
            ctx=_ctx(registry),
        )
    )

    assert "wf-1/live-explore" in result.response
    assert "wf-1/live-reviewer" not in result.response  # only the named agent
    assert "narrow result here" in result.response


@pytest.mark.asyncio
async def test_background_tool_stop(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)
    await collect_result(
        bash.run(BashArgs(command="sleep 30", background=True), ctx=ctx)
    )

    tool = _background_tool()
    result = await collect_result(
        tool.run(BackgroundArgs(action="stop", task_id="proc-1"), ctx=_ctx(registry))
    )

    assert result.stopped is True
    assert "proc-1" in result.response


@pytest.mark.asyncio
async def test_background_tool_stop_unknown_returns_not_stopped():
    tool = _background_tool()
    ctx = _ctx(BackgroundRegistry())

    result = await collect_result(
        tool.run(BackgroundArgs(action="stop", task_id="proc-999"), ctx=ctx)
    )

    assert result.stopped is False


@pytest.mark.asyncio
async def test_background_tool_stop_without_task_id_raises():
    tool = _background_tool()
    ctx = _ctx(BackgroundRegistry())

    with pytest.raises(ToolError, match="task_id"):
        await collect_result(tool.run(BackgroundArgs(action="stop"), ctx=ctx))


@pytest.mark.asyncio
async def test_background_tool_unknown_action_raises():
    tool = _background_tool()
    ctx = _ctx(BackgroundRegistry())

    with pytest.raises(ToolError, match="Unknown background action"):
        await collect_result(tool.run(BackgroundArgs(action="frobnicate"), ctx=ctx))


@pytest.mark.asyncio
async def test_background_tool_without_registry_raises():
    tool = _background_tool()
    ctx = _ctx(None)

    with pytest.raises(ToolError, match="background registry"):
        await collect_result(tool.run(BackgroundArgs(action="list"), ctx=ctx))


def test_background_tool_is_always_allowed():
    tool = _background_tool()
    perm = tool.resolve_permission(BackgroundArgs(action="stop", task_id="proc-1"))
    from vibe.core.tools.base import ToolPermission

    assert perm is not None
    assert perm.permission == ToolPermission.ALWAYS


# ---------------------------------------------------------------------------
# Log disk cap — read_log_tail trims oversized logs in place
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_log_tail_trims_oversized_log_in_place(tmp_path, monkeypatch):
    import vibe.core.tools.background as bgmod

    # Shrink the caps so the test runs fast on a small file.
    monkeypatch.setattr(bgmod, "_LOG_DISK_CAP_BYTES", 4096)
    monkeypatch.setattr(bgmod, "_LOG_DISK_KEEP_BYTES", 1024)

    reg = bgmod.BackgroundRegistry()
    log = tmp_path / "proc-1.log"
    # Write well over the cap.
    log.write_bytes(b"x" * 10_000)

    await reg.register_process(
        cast("asyncio.subprocess.Process", _DummyProc(1)),
        command="chatty",
        cwd=tmp_path,
        log_path=log,
    )
    # First read triggers the trim.
    reg.read_log_tail("proc-1", lines=5)
    assert log.stat().st_size <= bgmod._LOG_DISK_KEEP_BYTES


# ---------------------------------------------------------------------------
# End-to-end: background a real HTTP server, verify it serves, stop frees it
# ---------------------------------------------------------------------------


class _DummyProc:
    """Stand-in for register_process in the log-cap test (no real process)."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = None

    async def wait(self) -> int:
        return 0


@pytest.mark.asyncio
async def test_end_to_end_background_server_serves_then_stops(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    result = await collect_result(
        bash.run(
            BashArgs(
                command="python -u -m http.server 0 --bind 127.0.0.1", background=True
            ),
            ctx=ctx,
        )
    )
    assert result.background_task_id is not None

    # Poll the log until the bind line appears (startup takes a moment).
    log_tail = ""
    for _ in range(20):
        await asyncio.sleep(0.2)
        log_tail = registry.read_log_tail(result.background_task_id, lines=10)
        if "Serving HTTP" in log_tail:
            break
    assert "Serving HTTP" in log_tail, f"server never reported bound: {log_tail!r}"

    # The process is live and tracked.
    [entry] = registry.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "running"

    # Stop frees the process group.
    assert await registry.stop(result.background_task_id) is True
    await asyncio.sleep(0.3)
    [entry] = registry.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "stopped"


# ---------------------------------------------------------------------------
# Grandchild reaping (start_new_session + killpg reaches the whole tree)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_reaps_grandchild_process_tree(tmp_path):
    # Sandbox disabled: bwrap --unshare-pid namespaces $$, so the reported PID
    # would be an in-namespace number that _pid_alive checks on the host (false RED).
    bash = _bash(BashToolConfig(sandbox=SandboxConfig(enabled=False)))
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)
    pid_file = tmp_path / "child.pid"
    # sh -c 'sleep 300 & echo $$ > pidfile; wait' — the inner sh is a grandchild
    # of the backgrounded shell. `wait` keeps it alive until killed.
    cmd = f"sh -c 'sleep 300 & echo $$ > {pid_file} ; wait' "

    result = await collect_result(
        bash.run(BashArgs(command=cmd, background=True), ctx=ctx)
    )

    # Wait for the grandchild PID to be written.
    grandchild_pid = None
    for _ in range(20):
        await asyncio.sleep(0.1)
        if pid_file.exists():
            try:
                grandchild_pid = int(pid_file.read_text().strip())
                break
            except ValueError:
                pass
    assert grandchild_pid is not None, "grandchild never wrote its pid"

    # Stop the task — should kill the whole process group.
    assert result.background_task_id is not None
    assert await registry.stop(result.background_task_id) is True
    await asyncio.sleep(0.5)

    # The grandchild PID must be dead (reaped by killpg).
    alive = _pid_alive(grandchild_pid)
    assert not alive, (
        f"grandchild pid {grandchild_pid} survived stop — killpg did not fan out"
    )


def _pid_alive(pid: int) -> bool:
    import errno
    import os

    try:
        os.kill(pid, 0)  # signal 0 = existence check
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


# ---------------------------------------------------------------------------
# Shutdown reaps real running processes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_reaps_real_running_process(tmp_path):
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    result = await collect_result(
        bash.run(BashArgs(command="sleep 60", background=True), ctx=ctx)
    )
    pid = result.pid
    assert pid is not None
    assert _pid_alive(pid)

    await registry.shutdown()
    await asyncio.sleep(0.3)

    assert not _pid_alive(pid), "shutdown did not reap the running process"
    [entry] = registry.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "stopped"


# ---------------------------------------------------------------------------
# Concurrent running-process cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_process_enforces_running_cap(monkeypatch, tmp_path):
    import vibe.core.tools.background as bgmod

    monkeypatch.setattr(bgmod, "_MAX_RUNNING_PROCS", 2)
    reg = bgmod.BackgroundRegistry()

    class _P:
        def __init__(self) -> None:
            self.pid = 0
            self.returncode = None

        async def wait(self) -> int:
            await asyncio.sleep(30)
            return 0

    await reg.register_process(
        cast("asyncio.subprocess.Process", _P()),
        command="a",
        cwd=tmp_path,
        log_path=tmp_path / "1",
    )
    await reg.register_process(
        cast("asyncio.subprocess.Process", _P()),
        command="b",
        cwd=tmp_path,
        log_path=tmp_path / "2",
    )
    with pytest.raises(RuntimeError, match="cap reached"):
        await reg.register_process(
            cast("asyncio.subprocess.Process", _P()),
            command="c",
            cwd=tmp_path,
            log_path=tmp_path / "3",
        )


@pytest.mark.asyncio
async def test_background_cap_rejects_and_terminates_orphan(monkeypatch, tmp_path):
    import vibe.core.tools.background as bgmod
    import vibe.core.tools.builtins.bash as bashmod

    monkeypatch.setattr(bgmod, "_MAX_RUNNING_PROCS", 1)
    registry = bgmod.BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)
    bash = _bash()

    # Fill the cap with one running process.
    first = await collect_result(
        bash.run(BashArgs(command="sleep 30", background=True), ctx=ctx)
    )
    assert first.background_task_id == "proc-1"

    # Capture whether the failure handler force-kills the spawned orphan. The
    # fix calls kill_async_subprocess(proc) with no kwargs, so the spy matches.
    real_kill = bashmod.kill_async_subprocess
    killed: list[asyncio.subprocess.Process] = []

    async def _spy_kill(proc: asyncio.subprocess.Process) -> None:
        killed.append(proc)
        await real_kill(proc)

    monkeypatch.setattr(bashmod, "kill_async_subprocess", _spy_kill)

    # Second background run hits the cap -> registration raises RuntimeError.
    with pytest.raises(RuntimeError, match="cap reached"):
        await collect_result(
            bash.run(BashArgs(command="sleep 30", background=True), ctx=ctx)
        )

    # The orphaned process was force-killed (returncode set), not left running.
    assert killed, "orphaned process was not terminated on registration failure"
    assert killed[0].returncode is not None

    # And it never entered the registry: only the first process is tracked.
    running = [
        t
        for t in registry.list_tasks(category=TaskCategory.PROCESS)
        if t.status == "running"
    ]
    assert [t.task_id for t in running] == ["proc-1"]

    assert await registry.stop("proc-1") is True
