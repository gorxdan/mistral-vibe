from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from vibe.core.teams.mailbox import Mailbox
from vibe.core.teams.manager import TeamManager
from vibe.core.teams.models import TaskStatus
from vibe.core.teams.task_store import TaskStore


def test_add_and_get_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    task = store.add_task("Review auth module")
    assert task.id == "task-1"
    assert task.description == "Review auth module"
    assert task.status == TaskStatus.PENDING

    fetched = store.get_task("task-1")
    assert fetched is not None
    assert fetched.description == "Review auth module"


def test_claim_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.add_task("Review auth module")
    claimed = store.claim_task("task-1", "reviewer")
    assert claimed is not None
    assert claimed.status == TaskStatus.IN_PROGRESS
    assert claimed.assignee == "reviewer"


def test_claim_already_claimed(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.add_task("Task A")
    store.claim_task("task-1", "alice")
    result = store.claim_task("task-1", "bob")
    assert result is None


def test_complete_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.add_task("Task A")
    store.claim_task("task-1", "alice")
    completed = store.complete_task("task-1", result="Done, no issues found")
    assert completed is not None
    assert completed.status == TaskStatus.COMPLETED
    assert completed.result == "Done, no issues found"
    assert completed.completed_at is not None


def test_dependencies_block_claim(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.add_task("Setup")
    store.add_task("Build", dependencies=["task-1"])
    result = store.claim_task("task-2", "builder")
    assert result is None


def test_dependencies_unblock_after_complete(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.add_task("Setup")
    store.add_task("Build", dependencies=["task-1"])
    store.claim_task("task-1", "setup")
    store.complete_task("task-1", "done")
    result = store.claim_task("task-2", "builder")
    assert result is not None
    assert result.status == TaskStatus.IN_PROGRESS


def test_persistence_across_instances(tmp_path: Path) -> None:
    store1 = TaskStore(tmp_path)
    store1.add_task("Task A")
    store1.add_task("Task B")
    store1.claim_task("task-1", "alice")

    store2 = TaskStore(tmp_path)
    tasks = store2.get_all_tasks()
    assert len(tasks) == 2
    assert tasks[0].assignee == "alice"
    assert tasks[0].status == TaskStatus.IN_PROGRESS


def test_claim_task_no_double_claim_across_instances(tmp_path: Path) -> None:
    """Two stores backed by the same dir must not both claim the same task.

    Regression for the cross-process TOCTOU race: each store held a stale
    in-memory copy and only took the filelock for the write, so two processes
    could both observe PENDING and claim the same task. The fix re-reads
    tasks.json under the lock before validating.
    """
    store_a = TaskStore(tmp_path)
    store_b = TaskStore(tmp_path)
    store_a.add_task("Shared task")

    claimed_a = store_a.claim_task("task-1", "alice")
    claimed_b = store_b.claim_task("task-1", "bob")

    assert claimed_a is not None
    assert claimed_a.assignee == "alice"
    assert claimed_b is None, "second claim must see the updated status under the lock"

    # The on-disk record must reflect exactly one claim.
    store_c = TaskStore(tmp_path)
    task = store_c.get_task("task-1")
    assert task is not None
    assert task.status == TaskStatus.IN_PROGRESS
    assert task.assignee == "alice"


def test_get_available_tasks(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.add_task("Task A")
    store.add_task("Task B", dependencies=["task-1"])
    store.claim_task("task-1", "alice")

    available = store.get_available_tasks()
    assert len(available) == 0

    store.complete_task("task-1", "done")
    available = store.get_available_tasks()
    assert len(available) == 1
    assert available[0].id == "task-2"


def test_mailbox_send_and_read(tmp_path: Path) -> None:
    mb = Mailbox(tmp_path)
    msg = mb.send("alice", "bob", "Hello Bob!")
    assert msg.from_name == "alice"
    assert msg.to_name == "bob"
    assert msg.content == "Hello Bob!"

    messages = mb.read("bob")
    assert len(messages) == 1
    assert messages[0].content == "Hello Bob!"
    assert messages[0].read is True


def test_mailbox_unread(tmp_path: Path) -> None:
    mb = Mailbox(tmp_path)
    mb.send("alice", "bob", "Message 1")
    mb.send("alice", "bob", "Message 2")

    unread = mb.get_unread("bob")
    assert len(unread) == 2

    mb.read("bob")
    unread = mb.get_unread("bob")
    assert len(unread) == 0


def test_mailbox_multiple_recipients(tmp_path: Path) -> None:
    mb = Mailbox(tmp_path)
    mb.send("lead", "alice", "Task for Alice")
    mb.send("lead", "bob", "Task for Bob")

    alice_msgs = mb.read("alice")
    bob_msgs = mb.read("bob")
    assert len(alice_msgs) == 1
    assert len(bob_msgs) == 1
    assert alice_msgs[0].content == "Task for Alice"
    assert bob_msgs[0].content == "Task for Bob"


def test_mailbox_clear(tmp_path: Path) -> None:
    mb = Mailbox(tmp_path)
    mb.send("alice", "bob", "Message 1")
    mb.send("alice", "bob", "Message 2")
    mb.clear("bob")
    assert mb.read("bob") == []


def test_mailbox_empty_inbox(tmp_path: Path) -> None:
    mb = Mailbox(tmp_path)
    assert mb.read("nonexistent") == []
    assert mb.get_unread("nonexistent") == []


def test_mailbox_read_preserves_send_order(tmp_path: Path) -> None:
    """teams-004: messages must be returned in send order, not filename order.

    Filenames are random uuid4 strings. The old code sorted the glob lexically,
    so recipients saw messages in random order. Write files whose lexical order
    is the reverse of their timestamp order so the regression is deterministic.
    """
    from vibe.core.teams.models import Message

    inbox = tmp_path / "mailbox" / "bob"
    inbox.mkdir(parents=True)
    # Lexical filename sort -> a, b, c (contents third, first, second).
    # Timestamp sort -> b, c, a (contents first, second, third).
    fixtures = [
        ("a.json", "third", 3000.0),
        ("b.json", "first", 1000.0),
        ("c.json", "second", 2000.0),
    ]
    for fname, content, ts in fixtures:
        msg = Message(
            id=fname.removesuffix(".json"),
            from_name="alice",
            to_name="bob",
            content=content,
            timestamp=ts,
        )
        (inbox / fname).write_text(msg.model_dump_json(indent=2))

    mb = Mailbox(tmp_path)
    unread = [m.content for m in mb.get_unread("bob")]
    assert unread == ["first", "second", "third"]

    read_msgs = [m.content for m in mb.read("bob")]
    assert read_msgs == ["first", "second", "third"]


class _FakeProc:
    """Mimics asyncio.subprocess.Process: blocks on communicate() until killed."""

    def __init__(self) -> None:
        self.pid = 12345
        self._returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._waiters: list[asyncio.Future[int]] = []

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def _set_rc(self, rc: int) -> None:
        self._returncode = rc
        for w in self._waiters:
            if not w.done():
                w.set_result(rc)
        self._waiters.clear()

    def terminate(self) -> None:
        self.terminated = True
        self._set_rc(-15)

    def kill(self) -> None:
        self.killed = True
        self._set_rc(-9)

    async def wait(self) -> int:
        if self._returncode is not None:
            return self._returncode
        fut: asyncio.Future[int] = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        return await fut

    async def communicate(self) -> tuple[bytes, bytes]:
        await self.wait()
        return (b"", b"")


@pytest.mark.asyncio
async def test_stop_teammate_terminates_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """teams-005: stop_teammate must terminate and reap the spawned subprocess.

    Previously stop_teammate only cancelled the asyncio task; CancelledError
    escaped `except Exception`, so the trusted `vibe -p` child was never
    terminated while its status was recorded as "stopped".
    """
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    proc = _FakeProc()

    async def fake_exec(*args: object, **kwargs: object) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    mgr = TeamManager("lead-session", team_name="test-stop")
    try:
        await mgr.spawn_teammate("alice", "do stuff", agent="explore", max_turns=1)
        # Let the task start, create the proc, and block in communicate().
        await asyncio.sleep(0.05)
        assert mgr._teammate_procs.get("alice") is proc

        stopped = await mgr.stop_teammate("alice")

        assert stopped is True
        assert proc.terminated or proc.killed
        assert proc.returncode is not None  # reaped, not orphaned
        assert "alice" not in mgr._teammate_procs
    finally:
        mgr.cleanup()


@pytest.mark.asyncio
async def test_teammate_spawned_in_new_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """teams-007: the teammate is spawned in its own session/process group so
    stop can signal the whole tree (teammate + bash-tool grandchildren) instead
    of orphaning grandchildren to init.
    """
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    proc = _FakeProc()
    captured: dict[str, object] = {}

    async def fake_exec(*args: object, **kwargs: object) -> _FakeProc:
        captured.update(kwargs)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    mgr = TeamManager("lead-session", team_name="test-newsession")
    try:
        await mgr.spawn_teammate("bob", "work", agent="explore", max_turns=1)
        await asyncio.sleep(0.05)
        assert captured.get("start_new_session") is True
    finally:
        await mgr.stop_teammate("bob")
        mgr.cleanup()


@pytest.mark.asyncio
async def test_team_lifecycle_hooks_fire_on_task_events(tmp_path: Path) -> None:
    """teams-002: team lifecycle hooks must actually fire, not just be defined.

    A HooksManager with a task_created hook (shell command writing a marker
    file) is wired into TeamManager. add_team_task must dispatch the
    TASK_CREATED event through the pipeline, running the hook. Previously the
    events were never dispatched and _HANDLERS had no entry (KeyError).
    """
    from vibe.core.hooks.manager import HooksManager
    from vibe.core.hooks.models import HookConfig, HookSessionContext, HookType

    marker = tmp_path / "created.flag"
    hook = HookConfig(
        name="on-task-created", type=HookType.TASK_CREATED, command=f"touch {marker}"
    )
    hooks_mgr = HooksManager([hook])

    def ctx() -> HookSessionContext:
        return HookSessionContext(
            session_id="lead-session", transcript_path="", cwd=str(tmp_path)
        )

    # Use a team dir under the test's tmp_path to avoid polluting real VIBE_HOME.
    monkeypatch_dir = tmp_path / "vibehome"
    monkeypatch_dir.mkdir()
    saved = os.environ.get("VIBE_HOME")
    os.environ["VIBE_HOME"] = str(monkeypatch_dir)
    try:
        mgr = TeamManager(
            "lead-session",
            team_name="test-hooks",
            hooks_manager=hooks_mgr,
            hook_context=ctx,
        )
        try:
            await mgr.add_team_task("Write the auth module")
            # The hook runs asynchronously in a subprocess; allow it to complete.
            for _ in range(50):
                if marker.exists():
                    break
                await asyncio.sleep(0.05)
            assert marker.exists(), "TASK_CREATED hook must fire on add_team_task"
        finally:
            mgr.cleanup()
    finally:
        if saved is None:
            os.environ.pop("VIBE_HOME", None)
        else:
            os.environ["VIBE_HOME"] = saved


def test_mailbox_rejects_path_traversal_names(tmp_path: Path) -> None:
    """team-tool-001: recipient/sender names become inbox path components and
    arrive from model-controlled tool args, so traversal/absolute names must be
    rejected (no write/read outside the mailbox dir).
    """
    mb = Mailbox(tmp_path)
    for bad in ["../evil", "../../etc", "/tmp/abs", "..", "a/b", "with space"]:
        with pytest.raises(ValueError):
            mb.send("alice", bad, "x")
        with pytest.raises(ValueError):
            mb.read(bad)
        with pytest.raises(ValueError):
            mb.get_unread(bad)
    # A spoofed sender name is rejected too.
    with pytest.raises(ValueError):
        mb.send("../evil", "bob", "x")
    # Normal names still work.
    mb.send("alice", "bob", "hi")
    assert len(mb.read("bob")) == 1


def test_complete_task_enforces_ownership_for_actor(tmp_path: Path) -> None:
    """team-tool-003: a teammate (actor given) may only complete a task it
    claimed and that is in progress; the lead (actor=None) is unrestricted.
    """
    store = TaskStore(tmp_path)
    store.add_task("Task A")
    store.claim_task("task-1", "alice")

    # Another teammate cannot complete alice's task.
    assert store.complete_task("task-1", "done", actor="bob") is None
    blocked = store.get_task("task-1")
    assert blocked is not None
    assert blocked.status == TaskStatus.IN_PROGRESS

    # The owner can.
    done = store.complete_task("task-1", "done", actor="alice")
    assert done is not None
    assert done.status == TaskStatus.COMPLETED

    # Completing an already-completed task as an actor is refused.
    assert store.complete_task("task-1", "again", actor="alice") is None


def test_complete_task_lead_unrestricted(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    store.add_task("Task A")  # PENDING, unclaimed
    # actor=None (lead) completes regardless of claim/status.
    done = store.complete_task("task-1", "done")
    assert done is not None
    assert done.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_complete_team_task_fires_task_completed_hook(tmp_path: Path) -> None:
    """B-001: the lead-side /team task done path (complete_team_task) must fire
    the TASK_COMPLETED lifecycle hook. The wrapper previously had no callers.
    """
    from vibe.core.hooks.manager import HooksManager
    from vibe.core.hooks.models import HookConfig, HookSessionContext, HookType

    marker = tmp_path / "completed.flag"
    hook = HookConfig(
        name="on-task-completed",
        type=HookType.TASK_COMPLETED,
        command=f"touch {marker}",
    )
    hooks_mgr = HooksManager([hook])

    def ctx() -> HookSessionContext:
        return HookSessionContext(
            session_id="lead-session", transcript_path="", cwd=str(tmp_path)
        )

    monkeypatch_dir = tmp_path / "vibehome"
    monkeypatch_dir.mkdir()
    saved = os.environ.get("VIBE_HOME")
    os.environ["VIBE_HOME"] = str(monkeypatch_dir)
    try:
        mgr = TeamManager(
            "lead-session",
            team_name="test-complete-hook",
            hooks_manager=hooks_mgr,
            hook_context=ctx,
        )
        try:
            task = await mgr.add_team_task("Do the thing")
            done = await mgr.complete_team_task(task.id, "done")
            assert done is not None
            assert done.status == TaskStatus.COMPLETED
            for _ in range(50):
                if marker.exists():
                    break
                await asyncio.sleep(0.05)
            assert marker.exists(), (
                "TASK_COMPLETED hook must fire on complete_team_task"
            )
        finally:
            mgr.cleanup()
    finally:
        if saved is None:
            os.environ.pop("VIBE_HOME", None)
        else:
            os.environ["VIBE_HOME"] = saved


@pytest.mark.asyncio
async def test_team_command_task_verbs_route_to_manager() -> None:
    """B-001 wiring: /team task add|done route to add_team_task/complete_team_task
    with correct parsing (done splits '<id> <multi word result>').
    """
    from dataclasses import dataclass as _dc
    from typing import cast

    from vibe.cli.textual_ui.app import VibeApp

    calls: list[tuple] = []

    @_dc
    class _FakeTask:
        id: str
        description: str = "d"

    class _FakeMgr:
        async def add_team_task(self, desc: str):
            calls.append(("add", desc))
            return _FakeTask("task-1", desc)

        async def complete_team_task(self, task_id: str, result):
            calls.append(("done", task_id, result))
            return _FakeTask(task_id)

    class _Stub:
        def __init__(self) -> None:
            self._team_manager = _FakeMgr()

        def _build_team_manager(self):
            return self._team_manager

        async def _mount_and_scroll(self, _w) -> None:
            pass

        async def _team_task(self, parts, ErrorMessage, UserCommandMessage) -> None:
            await VibeApp._team_task(
                cast(VibeApp, self), parts, ErrorMessage, UserCommandMessage
            )

    stub = _Stub()
    await VibeApp._team_command(stub, "task add buy more milk")  # type: ignore[arg-type]
    await VibeApp._team_command(stub, "task done task-1 all good")  # type: ignore[arg-type]

    assert ("add", "buy more milk") in calls
    assert ("done", "task-1", "all good") in calls
