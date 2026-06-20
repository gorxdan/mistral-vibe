from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.background import BackgroundRegistry, TaskCategory
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.background import (
    Background,
    BackgroundArgs,
    BackgroundToolConfig,
)
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bash(config: BashToolConfig | None = None) -> Bash:
    return Bash(
        config_getter=lambda: config or BashToolConfig(),
        state=BaseToolState(),
    )


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
        config_getter=lambda: BackgroundToolConfig(),
        state=BaseToolState(),
    )


@pytest.fixture
async def reaping_registry():
    """Yield a BackgroundRegistry that reaps every still-running process on
    teardown, so a failing assertion can't orphan a backgrounded server."""
    reg = BackgroundRegistry()
    yield reg
    for rec in list(reg._procs.values()):
        if rec.status == "running":
            try:
                await reg.stop(rec.task_id)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Bash background branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_returns_immediately_with_handle(tmp_path):
    """The whole point: background=True does NOT block on a long command.

    Uses `sleep 5` — if the tool blocked, collect_result would hang until the
    per-test timeout. Backgrounding returns at once with a task_id and pid.
    """
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
    """Regression: background must default to False and the foreground path
    still awaits communicate() and returns the real stdout/returncode."""
    bash = _bash()
    ctx = _ctx(None, session_dir=tmp_path)

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
    pid = result.pid

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
        tool.run(
            BackgroundArgs(action="stop", task_id="proc-1"), ctx=_ctx(registry)
        )
    )

    assert result.stopped is True
    assert "proc-1" in result.response


@pytest.mark.asyncio
async def test_background_tool_stop_unknown_returns_not_stopped():
    tool = _background_tool()
    ctx = _ctx(BackgroundRegistry())

    result = await collect_result(
        tool.run(
            BackgroundArgs(action="stop", task_id="proc-999"), ctx=ctx
        )
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
        await collect_result(
            tool.run(BackgroundArgs(action="frobnicate"), ctx=ctx)
        )


@pytest.mark.asyncio
async def test_background_tool_without_registry_raises():
    tool = _background_tool()
    ctx = _ctx(None)

    with pytest.raises(ToolError, match="background registry"):
        await collect_result(tool.run(BackgroundArgs(action="list"), ctx=ctx))


def test_background_tool_is_always_allowed():
    """resolve_permission returns ALWAYS — it only touches session-launched tasks."""
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
    """A chatty server's log must not grow unbounded: when it exceeds the disk
    cap, read_log_tail rewrites it in place to a bounded tail."""
    import vibe.core.tools.background as bgmod

    # Shrink the caps so the test runs fast on a small file.
    monkeypatch.setattr(bgmod, "_LOG_DISK_CAP_BYTES", 4096)
    monkeypatch.setattr(bgmod, "_LOG_DISK_KEEP_BYTES", 1024)

    reg = bgmod.BackgroundRegistry()
    log = tmp_path / "proc-1.log"
    # Write well over the cap.
    log.write_bytes(b"x" * 10_000)

    await reg.register_process(
        _DummyProc(1), command="chatty", cwd=tmp_path, log_path=log
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
    """Background a real python http.server, confirm it writes its bind line,
    then confirm stop() frees it. This is the core 'launch a web server' UX.

    Uses -u so the startup print flushes immediately (stdout is block-buffered
    when redirected to a file, so the bind line would otherwise sit in the
    buffer until the process exits).
    """
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    result = await collect_result(
        bash.run(
            BashArgs(
                command="python -u -m http.server 0 --bind 127.0.0.1",
                background=True,
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
    """A backgrounded command that spawns its own children must have the whole
    tree reaped, not just the shell. start_new_session=True makes the shell a
    session/pgid leader, so killpg(getpgid(shell_pid)) fans out to children.

    The command writes a grandchild's PID to a file, then waits. After stop,
    that PID must no longer be alive.
    """
    import os

    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)
    pid_file = tmp_path / "child.pid"
    # sh -c 'sleep 300 & echo $! > pidfile; wait' — the sleep is a grandchild
    # of the backgrounded shell. `wait` keeps the shell alive until killed.
    cmd = (
        f"sh -c 'sleep 300 & echo $$ > {pid_file} ; wait' "
    )

    result = await collect_result(bash.run(BashArgs(command=cmd, background=True), ctx=ctx))

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
    assert await registry.stop(result.background_task_id) is True
    await asyncio.sleep(0.5)

    # The grandchild PID must be dead (reaped by killpg).
    alive = _pid_alive(grandchild_pid)
    assert not alive, f"grandchild pid {grandchild_pid} survived stop — killpg did not fan out"


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
    """registry.shutdown() (the app-exit orphan preventer) must terminate and
    reap a genuinely running process, not just a fake."""
    bash = _bash()
    registry = BackgroundRegistry()
    ctx = _ctx(registry, session_dir=tmp_path, scratchpad_dir=tmp_path)

    result = await collect_result(
        bash.run(BashArgs(command="sleep 60", background=True), ctx=ctx)
    )
    pid = result.pid
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
    """A looping agent must not be able to spawn unbounded background shells."""
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

    await reg.register_process(_P(), command="a", cwd=tmp_path, log_path=tmp_path / "1")
    await reg.register_process(_P(), command="b", cwd=tmp_path, log_path=tmp_path / "2")
    with pytest.raises(RuntimeError, match="cap reached"):
        await reg.register_process(
            _P(), command="c", cwd=tmp_path, log_path=tmp_path / "3"
        )
