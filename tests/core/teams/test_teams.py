from __future__ import annotations

from pathlib import Path

from vibe.core.teams.mailbox import Mailbox
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
