from __future__ import annotations

from collections.abc import Iterator
import contextlib
from pathlib import Path
import time

from filelock import FileLock, Timeout
import orjson

from vibe.core.logger import logger
from vibe.core.tasking import (
    TaskBrief,
    TaskOutcome,
    TaskOutcomeStatus,
    resolve_task_outcome,
)
from vibe.core.teams.errors import TeamStorageBusyError
from vibe.core.teams.models import (
    LEGACY_TASK_PROTOCOL_VERSION,
    STRUCTURED_TASK_PROTOCOL_VERSION,
    Task,
    TaskStatus,
)
from vibe.core.utils.io import read_safe, write_safe

# Default lease for IN_PROGRESS claims. Workers that die mid-task leave the
# claim until reclaim_stale runs (worker loop tick or lead list/available).
DEFAULT_TASK_LEASE_S = 900.0


class TaskStore:
    def __init__(self, team_dir: Path) -> None:
        self._team_dir = team_dir
        self._tasks_file = team_dir / "tasks.json"
        self._lock_file = team_dir / "tasks.lock"
        self._tasks: dict[str, Task] = {}
        self._load()

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        lock = FileLock(str(self._lock_file), timeout=5)
        try:
            with lock:
                yield
        except Timeout as e:
            raise TeamStorageBusyError(str(self._lock_file)) from e

    def _read_tasks(self) -> dict[str, Task]:
        """Read tasks from disk. Caller must already hold ``_lock``."""
        if not self._tasks_file.exists():
            return {}
        try:
            data = orjson.loads(read_safe(self._tasks_file).text)
            return {t["id"]: Task.model_validate(t) for t in data.get("tasks", [])}
        except Exception as e:
            logger.warning("Failed to load tasks from %s: %s", self._tasks_file, e)
            return {}

    def _write_tasks(self, tasks: dict[str, Task]) -> None:
        """Write tasks to disk. Caller must already hold ``_lock``."""
        self._team_dir.mkdir(parents=True, exist_ok=True)
        data = {"tasks": [t.model_dump(mode="json") for t in tasks.values()]}
        write_safe(
            self._tasks_file,
            orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8"),
        )

    def _load(self) -> None:
        with self._locked():
            self._tasks = self._read_tasks()

    def _save(self) -> None:
        with self._locked():
            self._write_tasks(self._tasks)

    def add_task(
        self,
        description: str | TaskBrief,
        *,
        dependencies: list[str] | None = None,
        task_id: str | None = None,
    ) -> Task:
        with self._locked():
            tasks = self._read_tasks()
            task_id = task_id or f"task-{len(tasks) + 1}"
            if isinstance(description, TaskBrief):
                brief = description
                description_text = description.objective
                protocol_version = STRUCTURED_TASK_PROTOCOL_VERSION
            else:
                brief = None
                description_text = description
                protocol_version = LEGACY_TASK_PROTOCOL_VERSION
            task = Task(
                id=task_id,
                description=description_text,
                protocol_version=protocol_version,
                brief=brief,
                dependencies=dependencies or [],
                created_at=time.time(),
            )
            tasks[task_id] = task
            self._write_tasks(tasks)
            self._tasks = tasks
            return task

    def claim_task(self, task_id: str, assignee: str) -> Task | None:
        # Atomic read-modify-write: re-read tasks.json under the lock so two
        # processes cannot both observe PENDING and claim the same task.
        with self._locked():
            tasks = self._read_tasks()
            task = tasks.get(task_id)
            if task is None:
                return None
            if task.status != TaskStatus.PENDING:
                return None
            if not self._dependencies_met(task, tasks):
                return None
            task.status = TaskStatus.IN_PROGRESS
            task.outcome = None
            task.assignee = assignee
            task.claimed_at = time.time()
            task.completed_at = None
            task.result = None
            self._write_tasks(tasks)
            self._tasks = tasks
            return task

    def complete_task(
        self,
        task_id: str,
        result: str | TaskOutcome | None = None,
        *,
        actor: str | None = None,
    ) -> Task | None:
        with self._locked():
            tasks = self._read_tasks()
            task = tasks.get(task_id)
            if task is None:
                return None
            # When an actor is supplied (a teammate completing via the team
            # tool), only the assignee that claimed an in-progress task may
            # complete it. actor=None (lead-side) keeps unrestricted completion.
            if actor is not None and (
                task.assignee != actor or task.status != TaskStatus.IN_PROGRESS
            ):
                return None
            outcome, result_text = self._resolve_outcome(task, result)
            task.result = result_text
            task.outcome = outcome
            if outcome.retryable:
                task.status = TaskStatus.PENDING
                task.assignee = None
                task.claimed_at = None
                task.completed_at = None
                self._write_tasks(tasks)
                self._tasks = tasks
                return task
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.claimed_at = None
            self._write_tasks(tasks)
            self._tasks = tasks
            return task

    @staticmethod
    def _resolve_outcome(
        task: Task, result: str | TaskOutcome | None
    ) -> tuple[TaskOutcome, str | None]:
        if isinstance(result, TaskOutcome):
            outcome = result
            result_text = result.summary
        else:
            result_text = result.strip() if result is not None else None
            if task.brief is not None:
                outcome = resolve_task_outcome(
                    task.brief, result_text or "", completed=True
                )
            else:
                outcome = TaskOutcome(
                    status=TaskOutcomeStatus.SUCCEEDED,
                    summary=result_text or "Legacy task completed",
                )
        if task.brief is None:
            return outcome, result_text
        if outcome.manifest is not None and outcome.manifest != task.brief.manifest:
            return (
                TaskOutcome(
                    status=TaskOutcomeStatus.RETRYABLE,
                    summary="Task outcome did not match the assigned manifest",
                    diagnostics=["Outcome manifest identity mismatch"],
                    manifest=task.brief.manifest,
                ),
                result_text,
            )
        return (
            outcome.model_copy(update={"manifest": task.brief.manifest}),
            result_text,
        )

    def reclaim_stale(
        self, lease_s: float = DEFAULT_TASK_LEASE_S, *, now: float | None = None
    ) -> list[str]:
        """Return IN_PROGRESS tasks whose claim lease expired to PENDING.

        Under lock: re-read disk, reset assignee/claimed_at for each stale claim.
        Tasks without claimed_at (pre-lease records) are treated as stale so a
        crashed pre-upgrade worker cannot hold the queue forever.
        """
        if lease_s <= 0:
            return []
        deadline = (time.time() if now is None else now) - lease_s
        reclaimed: list[str] = []
        with self._locked():
            tasks = self._read_tasks()
            for task in tasks.values():
                if task.status != TaskStatus.IN_PROGRESS:
                    continue
                claimed = task.claimed_at
                if claimed is not None and claimed > deadline:
                    continue
                task.status = TaskStatus.PENDING
                task.assignee = None
                task.claimed_at = None
                reclaimed.append(task.id)
            if reclaimed:
                self._write_tasks(tasks)
                self._tasks = tasks
        return reclaimed

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def get_pending_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]

    def get_available_tasks(self) -> list[Task]:
        return [
            t
            for t in self._tasks.values()
            if t.status == TaskStatus.PENDING and self._dependencies_met(t)
        ]

    def _dependencies_met(
        self, task: Task, tasks: dict[str, Task] | None = None
    ) -> bool:
        tasks = self._tasks if tasks is None else tasks
        if not task.dependencies:
            return True
        for dep_id in task.dependencies:
            dep = tasks.get(dep_id)
            if dep is None or dep.outcome is None or not dep.outcome.succeeded:
                return False
        return True

    def reload(self) -> None:
        self._load()
