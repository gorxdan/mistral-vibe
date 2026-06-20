from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vibe.core.tools.background import (
    BackgroundRegistry,
    TaskCategory,
    _team_status,
)


# ---------------------------------------------------------------------------
# Fakes — stand in for WorkflowRunner / TeamManager / LoopManager so the
# registry's aggregation and routing can be tested in isolation.
# ---------------------------------------------------------------------------


@dataclass
class _FakeLiveAgent:
    agent_id: str
    label: str = "agent"
    phase: str = "default"
    tokens_total: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    agent: str | None = "explore"
    model: str | None = None
    error: str | None = None
    log_path: Path | None = None


class _WorkflowStatus:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


@dataclass
class _FakeRunEntry:
    run_id: str
    status: _WorkflowStatus
    elapsed: float = 0.0
    agent_count: int = 0
    tokens_total: int = 0
    phases: list[str] = field(default_factory=list)
    live_agents: list[_FakeLiveAgent] = field(default_factory=list)
    is_paused: bool = False
    result: Any = None


class _FakeWorkflowRunner:
    def __init__(self) -> None:
        self.runs: list[_FakeRunEntry] = []
        self.stopped: list[str] = []
        self.cancelled: list[tuple[str, str]] = []
        self.paused: list[str] = []
        self.unpaused: list[str] = []

    def _find_run(self, run_id: str) -> _FakeRunEntry | None:
        return next((r for r in self.runs if r.run_id == run_id), None)

    async def stop(self, run_id: str) -> bool:
        entry = self._find_run(run_id)
        if entry is None:
            return False
        self.stopped.append(run_id)
        return True

    def cancel_agent(self, run_id: str, agent_id: str) -> bool:
        entry = self._find_run(run_id)
        if entry is None:
            return False
        self.cancelled.append((run_id, agent_id))
        return True

    def pause(self, run_id: str) -> bool:
        self.paused.append(run_id)
        return True

    def unpause(self, run_id: str) -> bool:
        self.unpaused.append(run_id)
        return True


@dataclass
class _FakeTeamMember:
    name: str
    agent_type: str = "auto-approve"
    status: str = "running"
    pid: int | None = None


class _FakeTeamManager:
    def __init__(self) -> None:
        self.members: list[_FakeTeamMember] = []
        self.stopped: list[str] = []

    def get_members(self) -> list[_FakeTeamMember]:
        return list(self.members)

    async def stop_teammate(self, name: str) -> bool:
        member = next((m for m in self.members if m.name == name), None)
        if member is None:
            return False
        self.stopped.append(name)
        member.status = "stopped"
        return True


@dataclass
class _FakeLoop:
    id: str
    interval_seconds: int = 60
    prompt: str = "loop"
    next_fire_at: float = 0.0
    recurring: bool = True


class _FakeLoopManager:
    def __init__(self) -> None:
        self.loops: list[_FakeLoop] = []
        self.cancelled: list[str] = []

    async def cancel(self, target: str) -> int:
        if target == "all":
            count = len(self.loops)
            self.loops.clear()
            return count
        before = len(self.loops)
        self.loops = [l for l in self.loops if l.id != target]
        if len(self.loops) < before:
            self.cancelled.append(target)
            return 1
        return 0


# ---------------------------------------------------------------------------
# list_tasks aggregation
# ---------------------------------------------------------------------------


def _registry_with_all() -> tuple[
    BackgroundRegistry, _FakeWorkflowRunner, _FakeTeamManager, _FakeLoopManager
]:
    reg = BackgroundRegistry()
    wf = _FakeWorkflowRunner()
    team = _FakeTeamManager()
    loop = _FakeLoopManager()
    reg.attach_workflow_runner(lambda: wf)
    reg.attach_team_manager(lambda: team)
    reg.attach_loop_manager(lambda: loop)
    return reg, wf, team, loop


def test_list_tasks_empty_when_nothing_attached():
    reg = BackgroundRegistry()
    assert reg.list_tasks() == []


def test_list_tasks_aggregates_all_categories():
    reg, wf, team, loop = _registry_with_all()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit"],
            live_agents=[_FakeLiveAgent(agent_id="a7", label="explore")],
        )
    )
    team.members.append(_FakeTeamMember(name="bob"))
    loop.loops.append(
        _FakeLoop(id="l9k2", prompt="recheck CI", next_fire_at=time.time() + 240)
    )

    entries = reg.list_tasks()
    cats = [e.category for e in entries]
    assert TaskCategory.WORKFLOW in cats
    assert TaskCategory.AGENT in cats
    assert TaskCategory.TEAM in cats
    assert TaskCategory.LOOP in cats
    # No processes registered → PROCESS absent
    assert TaskCategory.PROCESS not in cats


def test_list_tasks_filters_by_category():
    reg, wf, _team, _loop = _registry_with_all()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            live_agents=[_FakeLiveAgent(agent_id="a1")],
        )
    )

    workflows = reg.list_tasks(category=TaskCategory.WORKFLOW)
    assert len(workflows) == 1
    assert workflows[0].task_id == "wf-1"
    assert workflows[0].can_pause is True
    assert workflows[0].can_save is True

    agents = reg.list_tasks(category=TaskCategory.AGENT)
    assert len(agents) == 1
    assert agents[0].task_id == "wf-1/live-a1"
    assert agents[0].parent_id == "wf-1"


def test_agent_task_id_uses_live_prefix():
    reg, wf, _team, _loop = _registry_with_all()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-2",
            status=_WorkflowStatus("running"),
            live_agents=[_FakeLiveAgent(agent_id="abc")],
        )
    )
    [agent] = reg.list_tasks(category=TaskCategory.AGENT)
    assert agent.task_id == "wf-2/live-abc"


# ---------------------------------------------------------------------------
# stop() routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_routes_to_workflow_runner():
    reg, wf, _team, _loop = _registry_with_all()
    wf.runs.append(_FakeRunEntry(run_id="wf-2", status=_WorkflowStatus("running")))

    ok = await reg.stop("wf-2")
    assert ok is True
    assert wf.stopped == ["wf-2"]


@pytest.mark.asyncio
async def test_stop_routes_to_cancel_agent():
    reg, wf, _team, _loop = _registry_with_all()
    wf.runs.append(_FakeRunEntry(run_id="wf-2", status=_WorkflowStatus("running")))

    ok = await reg.stop("wf-2/live-a7")
    assert ok is True
    assert wf.cancelled == [("wf-2", "a7")]


@pytest.mark.asyncio
async def test_stop_routes_to_team_manager():
    reg, _wf, team, _loop = _registry_with_all()
    team.members.append(_FakeTeamMember(name="bob"))

    ok = await reg.stop("team:bob")
    assert ok is True
    assert team.stopped == ["bob"]


@pytest.mark.asyncio
async def test_stop_routes_to_loop_manager():
    reg, _wf, _team, loop = _registry_with_all()
    loop.loops.append(_FakeLoop(id="l9k2"))

    ok = await reg.stop("loop-l9k2")
    assert ok is True
    assert loop.cancelled == ["l9k2"]


@pytest.mark.asyncio
async def test_stop_returns_false_for_unknown_id():
    reg, _wf, _team, _loop = _registry_with_all()
    assert await reg.stop("does-not-exist") is False


@pytest.mark.asyncio
async def test_stop_returns_false_for_missing_workflow_owner():
    # No workflow runner attached → routing returns False, not an error.
    reg = BackgroundRegistry()
    assert await reg.stop("wf-9") is False


@pytest.mark.asyncio
async def test_stop_returns_false_when_workflow_not_found():
    reg, wf, _team, _loop = _registry_with_all()
    # runner attached but no run with that id
    assert await reg.stop("wf-99") is False
    assert wf.stopped == []


# ---------------------------------------------------------------------------
# pause() routing (workflow-only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_runs_then_unpauses_on_second_call():
    reg, wf, _team, _loop = _registry_with_all()
    wf.runs.append(
        _FakeRunEntry(run_id="wf-1", status=_WorkflowStatus("running"), is_paused=False)
    )

    assert await reg.pause("wf-1") is True
    assert wf.paused == ["wf-1"]
    # Flip the entry to paused so the registry takes the unpause branch next.
    wf.runs[0].is_paused = True
    assert await reg.pause("wf-1") is True
    assert wf.unpaused == ["wf-1"]


@pytest.mark.asyncio
async def test_pause_returns_false_for_non_workflow():
    reg, _wf, _team, _loop = _registry_with_all()
    assert await reg.pause("proc-1") is False
    assert await reg.pause("team:bob") is False


# ---------------------------------------------------------------------------
# Process ownership
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal asyncio.subprocess.Process stand-in for ownership tests.

    Real process objects are created by create_subprocess_*; here we simulate
    wait() returning a code and terminate/kill being no-ops so we can test the
    registry's bookkeeping without spawning. NOTE: the registry's module-level
    _signal_proc_group calls real os.killpg/os.getpgid — tests that exercise
    termination MUST monkeypatch it (see _no_real_signals) or a fake pid would
    collide with a real OS process group and SIGTERM the test runner.
    """

    def __init__(self, pid: int, *, returncode: int | None = None) -> None:
        self.pid = pid
        self._returncode = returncode

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def wait(self) -> int:
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self._returncode = -15

    def kill(self) -> None:
        self._returncode = -9


@pytest.fixture
def _no_real_signals(monkeypatch):
    """Replace _signal_proc_group with a recorder so termination tests never
    send real OS signals (which would risk hitting the runner's own pgid)."""
    calls: list[tuple[int, int]] = []

    def _fake(proc, sig):
        calls.append((proc.pid, sig))
        # Simulate the group signal "working": mark the fake proc.
        if sig == 9:
            proc.kill()
        else:
            proc.terminate()

    monkeypatch.setattr(
        "vibe.core.tools.background._signal_proc_group", _fake
    )
    return calls


@pytest.mark.asyncio
async def test_register_process_returns_proc_id_and_lists_running():
    reg = BackgroundRegistry()
    proc = _FakeProc(pid=12345)
    log = Path("/tmp/bg/proc-1.log")

    task_id = await reg.register_process(
        proc, command="vite --port 5173", cwd=Path("/srv"), log_path=log
    )
    assert task_id == "proc-1"

    entries = reg.list_tasks(category=TaskCategory.PROCESS)
    assert len(entries) == 1
    e = entries[0]
    assert e.task_id == "proc-1"
    assert e.status == "running"
    assert e.label == "vite --port 5173"
    assert e.detail["pid"] == 12345
    assert e.detail["log_path"] == str(log)


@pytest.mark.asyncio
async def test_register_process_ids_increment():
    reg = BackgroundRegistry()
    t1 = await reg.register_process(
        _FakeProc(pid=1), command="a", cwd=Path("."), log_path=Path("/x")
    )
    t2 = await reg.register_process(
        _FakeProc(pid=2), command="b", cwd=Path("."), log_path=Path("/y")
    )
    assert t1 == "proc-1"
    assert t2 == "proc-2"


@pytest.mark.asyncio
async def test_stop_process_terminates_and_marks_stopped(_no_real_signals):
    reg = BackgroundRegistry()
    proc = _FakeProc(pid=42)
    task_id = await reg.register_process(
        proc, command="sleep 30", cwd=Path("."), log_path=Path("/x")
    )

    ok = await reg.stop(task_id)
    assert ok is True
    [entry] = reg.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "stopped"
    # SIGTERM sent to the process group (signal 15).
    assert (42, 15) in _no_real_signals


@pytest.mark.asyncio
async def test_stop_process_returns_false_if_already_finalized():
    reg = BackgroundRegistry()
    proc = _FakeProc(pid=42, returncode=0)  # already exited cleanly
    task_id = await reg.register_process(
        proc, command="true", cwd=Path("."), log_path=Path("/x")
    )
    # Let the finalizer flip the status to completed.
    await asyncio.sleep(0.01)
    [entry] = reg.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "completed"

    assert await reg.stop(task_id) is False


@pytest.mark.asyncio
async def test_finalizer_flips_status_on_process_exit():
    reg = BackgroundRegistry()
    proc = _FakeProc(pid=7, returncode=None)
    await reg.register_process(
        proc, command="serve", cwd=Path("."), log_path=Path("/x")
    )
    # Trigger the wait() that the finalizer is awaiting.
    await asyncio.sleep(0.02)

    [entry] = reg.list_tasks(category=TaskCategory.PROCESS)
    assert entry.status == "completed"
    assert entry.detail["returncode"] == 0


# ---------------------------------------------------------------------------
# read_log_tail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_log_tail_returns_last_n_lines(tmp_path):
    reg = BackgroundRegistry()
    log = tmp_path / "proc-1.log"
    log.write_text("line1\nline2\nline3\nline4\nline5\n")
    await reg.register_process(
        _FakeProc(pid=1), command="c", cwd=tmp_path, log_path=log
    )

    tail = reg.read_log_tail("proc-1", lines=2)
    assert tail == "line4\nline5"


def test_read_log_tail_returns_empty_for_unknown_id():
    reg = BackgroundRegistry()
    assert reg.read_log_tail("proc-999") == ""


@pytest.mark.asyncio
async def test_read_log_tail_returns_empty_for_missing_file(tmp_path):
    reg = BackgroundRegistry()
    await reg.register_process(
        _FakeProc(pid=1),
        command="c",
        cwd=tmp_path,
        log_path=tmp_path / "never-written.log",
    )
    assert reg.read_log_tail("proc-1") == ""


# ---------------------------------------------------------------------------
# read_agent_log_tail
# ---------------------------------------------------------------------------


def _registry_with_workflow_log(tmp_path) -> tuple[BackgroundRegistry, _FakeWorkflowRunner]:
    """Wire a registry to a fake workflow runner with one run (wf-1) carrying a
    single live agent (la-0) whose transcript lives at a real on-disk path.
    """
    import json

    reg, wf, _team, _loop = _registry_with_all()
    log = tmp_path / "messages.jsonl"
    log.write_text(
        json.dumps({"role": "user", "content": "find the auth flow"}) + "\n"
        + json.dumps({"role": "assistant", "content": "I will grep for auth"}) + "\n"
        + json.dumps({"role": "assistant", "content": "Found it in auth.py:42"}) + "\n"
    )
    run = _FakeRunEntry(
        run_id="wf-1",
        status=_WorkflowStatus("running"),
        live_agents=[_FakeLiveAgent(agent_id="la-0", log_path=log)],
    )
    wf.runs.append(run)
    return reg, wf


def test_read_agent_log_tail_returns_formatted_recent_messages(tmp_path):
    """A wf-N/live-la-M id resolves to the agent's transcript and renders the
    last lines as readable 'role: content' snippets — not raw JSON.
    """
    reg, _wf = _registry_with_workflow_log(tmp_path)

    tail = reg.read_agent_log_tail("wf-1/live-la-0", lines=2)

    assert "Found it in auth.py:42" in tail
    assert "assistant: " in tail
    assert "{" not in tail  # JSON envelope stripped, not dumped raw


def test_read_agent_log_tail_empty_for_isolated_agent(tmp_path):
    """An agent with no log_path (isolated/worktree agent) returns '' — nothing
    stable to tail, rendered as 'no output yet'.
    """
    reg, wf, _team, _loop = _registry_with_all()
    wf.runs.append(
        _FakeRunEntry(
            run_id="wf-2",
            status=_WorkflowStatus("running"),
            live_agents=[_FakeLiveAgent(agent_id="la-0", log_path=None)],
        )
    )

    assert reg.read_agent_log_tail("wf-2/live-la-0") == ""


def test_read_agent_log_tail_empty_for_unknown_id():
    """Unknown run or agent ids return '' rather than raising."""
    reg, _wf, _team, _loop = _registry_with_all()

    assert reg.read_agent_log_tail("wf-999/live-la-0") == ""
    assert reg.read_agent_log_tail("wf-1/live-la-999") == ""
    # A bare proc id is not a hierarchical agent id.
    assert reg.read_agent_log_tail("proc-1") == ""


def test_read_agent_log_tail_empty_when_no_workflow_runner():
    """With no workflow runner attached, agent tails resolve to ''."""
    reg = BackgroundRegistry()
    assert reg.read_agent_log_tail("wf-1/live-la-0") == ""


def test_parse_agent_task_id_round_trip():
    """The id parser splits hierarchical agent ids and rejects non-agent ids."""
    parse = BackgroundRegistry._parse_agent_task_id
    assert parse("wf-1/live-la-3") == ("wf-1", "la-3")
    assert parse("proc-1") == (None, None)          # not hierarchical
    assert parse("wf-1/phases") == (None, None)     # suffix is not a 'live-' child
    assert parse("team:alice") == (None, None)


def test_format_jsonl_tail_handles_malformed_trailing_line():
    """A partial trailing write (no closing brace yet) is passed through, not
    dropped, so an in-progress append never blanks the tail.
    """
    from vibe.core.tools.background import _format_jsonl_tail

    raw = '{"role": "assistant", "content": "ok"}\n{"role": "user", "content": "par'
    out = _format_jsonl_tail(raw)
    assert "assistant: ok" in out
    assert "par" in out  # malformed line kept verbatim


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_terminates_running_processes(_no_real_signals):
    reg = BackgroundRegistry()
    p1 = _FakeProc(pid=1)
    p2 = _FakeProc(pid=2)
    await reg.register_process(p1, command="a", cwd=Path("."), log_path=Path("/x"))
    await reg.register_process(p2, command="b", cwd=Path("."), log_path=Path("/y"))

    await reg.shutdown()

    entries = reg.list_tasks(category=TaskCategory.PROCESS)
    assert all(e.status == "stopped" for e in entries)


# ---------------------------------------------------------------------------
# _team_status normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", "running"),
        ("running", "running"),
        ("running:pid=123", "running"),
        ("completed", "completed"),
        ("failed:oops", "failed"),
        ("stopped", "stopped"),
        ("error:boom", "failed"),
        ("weird", "running"),
    ],
)
def test_team_status_normalization(raw: str, expected: str) -> None:
    assert _team_status(raw) == expected
