from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest

from vibe.core.tasking import (
    TaskBrief,
    TaskBudget,
    TaskManifestIdentity,
    TaskOutcome,
    TaskOutcomeStatus,
)
from vibe.core.teams.manager import TeamManager
from vibe.core.teams.models import (
    LEGACY_TASK_PROTOCOL_VERSION,
    STRUCTURED_TASK_PROTOCOL_VERSION,
    TaskStatus,
)
from vibe.core.teams.task_store import TaskStore
from vibe.core.teams.worker_loop import run_team_worker_loop
from vibe.core.utils.io import write_safe


def _brief(*, objective: str = "Implement the parser fix") -> TaskBrief:
    return TaskBrief(
        objective=objective,
        inputs={"target": "vibe/core/parser.py:10"},
        allowed_paths=["vibe/core/parser.py", "tests/core/test_parser.py"],
        denied_paths=["vibe/core/agent_loop.py"],
        acceptance_checks=["uv run pytest tests/core/test_parser.py"],
        budget=TaskBudget(max_tokens=5_000, max_calls=6),
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )


def test_legacy_completed_record_migrates_to_explicit_success(tmp_path: Path) -> None:
    payload = {
        "tasks": [
            {
                "id": "task-1",
                "description": "Legacy work",
                "status": "completed",
                "result": "done before the protocol upgrade",
            }
        ]
    }
    write_safe(tmp_path / "tasks.json", orjson.dumps(payload).decode())

    task = TaskStore(tmp_path).get_task("task-1")

    assert task is not None
    assert task.description == "Legacy work"
    assert task.protocol_version == LEGACY_TASK_PROTOCOL_VERSION
    assert task.outcome is not None
    assert task.outcome.status is TaskOutcomeStatus.SUCCEEDED
    assert task.outcome.summary == "done before the protocol upgrade"


def test_structured_task_persists_brief_and_legacy_description(tmp_path: Path) -> None:
    brief = _brief()
    created = TaskStore(tmp_path).add_task(brief)

    persisted = TaskStore(tmp_path).get_task(created.id)

    assert persisted is not None
    assert persisted.protocol_version == STRUCTURED_TASK_PROTOCOL_VERSION
    assert persisted.description == brief.objective
    assert persisted.brief == brief
    assert persisted.outcome is None
    assert "TASK_BRIEF_JSON:" in persisted.prompt


def test_expired_structured_task_is_blocked_before_team_claim(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    brief = _brief().model_copy(
        update={"deadline": datetime.now(UTC) - timedelta(seconds=1)}
    )
    task = store.add_task(brief)

    assert store.claim_task(task.id, "worker") is None

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.COMPLETED
    assert current.assignee is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.BLOCKED
    assert current.outcome.manifest == brief.manifest
    assert "deadline" in current.outcome.diagnostics[0]


def test_structured_plain_prose_cannot_complete_or_succeed(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    assert store.claim_task(task.id, "worker") is not None

    completion = store.complete_task(
        task.id, "Implemented and all tests look good", actor="worker"
    )

    assert completion is not None
    assert completion.status is TaskStatus.PENDING
    assert completion.outcome is not None
    assert completion.outcome.status is TaskOutcomeStatus.RETRYABLE
    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.assignee is None
    assert current.claimed_at is None
    assert current.completed_at is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE
    assert current.outcome.succeeded is False


@pytest.mark.parametrize(
    "marker,status",
    [
        ("SUCCEEDED", TaskOutcomeStatus.SUCCEEDED),
        ("FAILED", TaskOutcomeStatus.FAILED),
        ("BLOCKED", TaskOutcomeStatus.BLOCKED),
    ],
)
def test_structured_explicit_terminal_marker_completes_lifecycle(
    tmp_path: Path, marker: str, status: TaskOutcomeStatus
) -> None:
    store = TaskStore(tmp_path)
    brief = _brief()
    task = store.add_task(brief)
    assert store.claim_task(task.id, "worker") is not None

    completed = store.complete_task(
        task.id, f"Attempt finished\nTASK_OUTCOME: {marker}", actor="worker"
    )

    assert completed is not None
    assert completed.status is TaskStatus.COMPLETED
    assert completed.outcome is not None
    assert completed.outcome.status is status
    assert completed.outcome.manifest == brief.manifest


def test_structured_retryable_marker_requeues_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    assert store.claim_task(task.id, "worker") is not None

    completion = store.complete_task(
        task.id, "Provider timed out\nTASK_OUTCOME: RETRYABLE", actor="worker"
    )

    assert completion is not None
    assert completion.status is TaskStatus.PENDING
    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.assignee is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE


def test_structured_outcome_cannot_change_manifest_identity(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    brief = _brief()
    task = store.add_task(brief)
    assert store.claim_task(task.id, "worker") is not None
    forged = TaskOutcome(
        status=TaskOutcomeStatus.SUCCEEDED,
        summary="done",
        manifest=TaskManifestIdentity(name="full-access", version="99"),
    )

    completion = store.complete_task(task.id, forged, actor="worker")

    assert completion is not None
    assert completion.status is TaskStatus.PENDING
    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE
    assert current.outcome.manifest == brief.manifest


def test_failed_dependency_does_not_unblock_dependent_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    dependency = store.add_task(_brief(objective="Build dependency"))
    dependent = store.add_task(
        _brief(objective="Use dependency"), dependencies=[dependency.id]
    )
    assert store.claim_task(dependency.id, "worker") is not None
    assert (
        store.complete_task(
            dependency.id, "Could not build\nTASK_OUTCOME: FAILED", actor="worker"
        )
        is not None
    )

    assert store.claim_task(dependent.id, "worker") is None
    assert dependent.id not in {task.id for task in store.get_available_tasks()}


@pytest.mark.asyncio
async def test_manager_fires_completion_hook_only_for_terminal_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    import vibe.core.teams.manager as manager_module

    monkeypatch.setattr(manager_module, "_team_dir_for", lambda _name: tmp_path)
    manager = TeamManager("lead", team_name="task-protocol")
    task = await manager.add_team_task(_brief())
    assert manager.task_store.claim_task(task.id, "worker") is not None
    dispatch = AsyncMock()
    monkeypatch.setattr(manager, "_dispatch_hook", dispatch)

    retry = await manager.complete_team_task(task.id, "ordinary prose")

    assert retry is not None
    assert retry.status is TaskStatus.PENDING
    dispatch.assert_not_awaited()

    assert manager.task_store.claim_task(task.id, "worker") is not None
    terminal = await manager.complete_team_task(
        task.id, "Finished\nTASK_OUTCOME: FAILED"
    )

    assert terminal is not None
    assert terminal.status is TaskStatus.COMPLETED
    dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_plain_prose_does_not_autocomplete_structured_task(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    prompts: list[str] = []

    async def run_task(prompt: str) -> str:
        prompts.append(prompt)
        return "Implemented and tests pass"

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert len(prompts) == 1
    assert "TASK_BRIEF_JSON:" in prompts[0]
    assert current.status is TaskStatus.PENDING
    assert current.assignee is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE


@pytest.mark.asyncio
async def test_worker_completes_structured_task_with_terminal_marker(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())

    async def run_task(prompt: str) -> str:
        assert "TASK_OUTCOME: SUCCEEDED" in prompt
        return "Implemented and checked\nTASK_OUTCOME: SUCCEEDED"

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.COMPLETED
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.SUCCEEDED
