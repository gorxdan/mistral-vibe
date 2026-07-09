from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.teams.mailbox import Mailbox
from vibe.core.teams.models import MessageKind, TaskStatus
from vibe.core.teams.task_store import TaskStore
from vibe.core.teams.worker_loop import run_team_worker_loop


@pytest.mark.asyncio
async def test_worker_loop_claims_and_completes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")

    store = TaskStore(tmp_path)
    store.add_task("Do thing A")
    store.add_task("Do thing B")

    seen: list[str] = []

    async def run_task(prompt: str) -> str | None:
        seen.append(prompt)
        return f"done:{len(seen)}"

    summary = await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )
    assert summary is not None
    assert len(seen) == 2
    store.reload()
    tasks = store.get_all_tasks()
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)
    assert {t.assignee for t in tasks} == {"worker1"}


@pytest.mark.asyncio
async def test_worker_loop_stops_on_shutdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")

    store = TaskStore(tmp_path)
    store.add_task("Never claimed if shutdown first")
    mb = Mailbox(tmp_path)
    mb.send("lead", "worker1", "stop", kind=MessageKind.SHUTDOWN)

    async def run_task(prompt: str) -> str | None:
        raise AssertionError("should not run tasks after shutdown")

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=5, lease_s=900.0
    )
    store.reload()
    task = store.get_task("task-1")
    assert task is not None
    assert task.status == TaskStatus.PENDING
