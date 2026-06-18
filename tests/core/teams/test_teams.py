from __future__ import annotations

import asyncio
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
