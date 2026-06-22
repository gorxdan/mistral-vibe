from __future__ import annotations

import json
from pathlib import Path
import time

from filelock import FileLock

from vibe.core.logger import logger
from vibe.core.teams.models import Task, TaskStatus
from vibe.core.utils.io import read_safe


class TaskStore:
    def __init__(self, team_dir: Path) -> None:
        self._team_dir = team_dir
        self._tasks_file = team_dir / "tasks.json"
        self._lock_file = team_dir / "tasks.lock"
        self._tasks: dict[str, Task] = {}
        self._load()

    def _lock(self) -> FileLock:
        return FileLock(str(self._lock_file), timeout=5)

    def _read_tasks(self) -> dict[str, Task]:
        """Read tasks from disk. Caller must already hold ``_lock``."""
        if not self._tasks_file.exists():
            return {}
        try:
            data = json.loads(read_safe(self._tasks_file).text)
            return {t["id"]: Task.model_validate(t) for t in data.get("tasks", [])}
        except Exception as e:
            logger.warning("Failed to load tasks from %s: %s", self._tasks_file, e)
            return {}

    def _write_tasks(self, tasks: dict[str, Task]) -> None:
        """Write tasks to disk. Caller must already hold ``_lock``."""
        self._team_dir.mkdir(parents=True, exist_ok=True)
        data = {"tasks": [t.model_dump(mode="json") for t in tasks.values()]}
        self._tasks_file.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        with self._lock():
            self._tasks = self._read_tasks()

    def _save(self) -> None:
        with self._lock():
            self._write_tasks(self._tasks)

    def add_task(
        self,
        description: str,
        *,
        dependencies: list[str] | None = None,
        task_id: str | None = None,
    ) -> Task:
        with self._lock():
            tasks = self._read_tasks()
            task_id = task_id or f"task-{len(tasks) + 1}"
            task = Task(
                id=task_id,
                description=description,
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
        with self._lock():
            tasks = self._read_tasks()
            task = tasks.get(task_id)
            if task is None:
                return None
            if task.status != TaskStatus.PENDING:
                return None
            if not self._dependencies_met(task, tasks):
                return None
            task.status = TaskStatus.IN_PROGRESS
            task.assignee = assignee
            self._write_tasks(tasks)
            self._tasks = tasks
            return task

    def complete_task(
        self, task_id: str, result: str | None = None, *, actor: str | None = None
    ) -> Task | None:
        with self._lock():
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
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            self._write_tasks(tasks)
            self._tasks = tasks
            return task

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
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def reload(self) -> None:
        self._load()
