from __future__ import annotations


class TeamStorageBusyError(RuntimeError):
    def __init__(self, lock_path: str) -> None:
        super().__init__(f"Team storage is busy; could not acquire lock {lock_path}")
        self.lock_path = lock_path
