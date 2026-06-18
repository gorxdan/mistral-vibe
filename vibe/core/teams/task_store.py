from __future__ import annotations

import json
from pathlib import Path
import time

from filelock import FileLock

from vibe.core.logger import logger
from vibe.core.teams.models import Task, TaskStatus


class TaskStore:
    def __init__(self, team_dir: Path) -> None:
        self._team_dir = team_dir
        self._tasks_file = team_dir / "tasks.json"
        self._lock_file = team_dir / "tasks.lock"
        self._tasks: dict[str, Task] = {}
        self._load()

    def _load(self) -> None:
        if not self._tasks_file.exists():
            self._tasks = {}
            return
        lock = FileLock(str(self._lock_file), timeout=5)
        with lock:
            try:
                data = json.loads(self._tasks_file.read_text())
                self._tasks = {
                    t["id"]: Task.model_validate(t) for t in data.get("tasks", [])
                }
            except Exception as e:
                logger.warning("Failed to load tasks from %s: %s", self._tasks_file, e)
                self._tasks = {}

    def _save(self) -> None:
        self._team_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._lock_file), timeout=5)
        with lock:
            data = {"tasks": [t.model_dump(mode="json") for t in self._tasks.values()]}
            self._tasks_file.write_text(json.dumps(data, indent=2))

    def add_task(
        self,
        description: str,
        *,
        dependencies: list[str] | None = None,
        task_id: str | None = None,
    ) -> Task:
        task_id = task_id or f"task-{len(self._tasks) + 1}"
        task = Task(
            id=task_id,
            description=description,
            dependencies=dependencies or [],
            created_at=time.time(),
        )
        self._tasks[task_id] = task
        self._save()
        return task

    def claim_task(self, task_id: str, assignee: str) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if task.status != TaskStatus.PENDING:
            return None
        if not self._dependencies_met(task):
            return None
        task.status = TaskStatus.IN_PROGRESS
        task.assignee = assignee
        self._save()
        return task

    def complete_task(self, task_id: str, result: str | None = None) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result
        self._save()
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

    def _dependencies_met(self, task: Task) -> bool:
        if not task.dependencies:
            return True
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def reload(self) -> None:
        self._load()
