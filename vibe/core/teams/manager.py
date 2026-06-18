from __future__ import annotations

import asyncio
import os
from pathlib import Path
import secrets
import sys
import time

from filelock import FileLock

from vibe.core.logger import logger
from vibe.core.teams.mailbox import Mailbox
from vibe.core.teams.models import TeamConfig, TeamMember
from vibe.core.teams.task_store import TaskStore


def _team_dir_for(team_name: str) -> Path:
    base = Path(os.environ.get("VIBE_HOME", str(Path.home() / ".vibe")))
    return base / "teams" / team_name


class TeamManager:
    def __init__(self, lead_session_id: str, *, team_name: str | None = None) -> None:
        self._team_name = team_name or f"team-{secrets.token_hex(4)}"
        self._team_dir = _team_dir_for(self._team_name)
        self._team_dir.mkdir(parents=True, exist_ok=True)
        self._config_file = self._team_dir / "config.json"
        self._config_lock = self._team_dir / "config.lock"
        self._lead_session_id = lead_session_id
        self._task_store: TaskStore | None = None
        self._mailbox: Mailbox | None = None
        self._teammate_tasks: dict[str, asyncio.Task[None]] = {}
        self._init_config()

    @property
    def team_name(self) -> str:
        return self._team_name

    @property
    def team_dir(self) -> Path:
        return self._team_dir

    @property
    def task_store(self) -> TaskStore:
        if self._task_store is None:
            self._task_store = TaskStore(self._team_dir)
        return self._task_store

    @property
    def mailbox(self) -> Mailbox:
        if self._mailbox is None:
            self._mailbox = Mailbox(self._team_dir)
        return self._mailbox

    def _init_config(self) -> None:
        if self._config_file.exists():
            return
        config = TeamConfig(
            team_name=self._team_name,
            created_at=time.time(),
            team_dir=str(self._team_dir),
            lead_session_id=self._lead_session_id,
        )
        self._save_config(config)

    def _load_config(self) -> TeamConfig:
        lock = FileLock(str(self._config_lock), timeout=5)
        with lock:
            return TeamConfig.model_validate_json(self._config_file.read_text())

    def _save_config(self, config: TeamConfig) -> None:
        lock = FileLock(str(self._config_lock), timeout=5)
        with lock:
            self._config_file.write_text(config.model_dump_json(indent=2))

    def get_config(self) -> TeamConfig:
        return self._load_config()

    def get_members(self) -> list[TeamMember]:
        return self._load_config().members

    def add_member(self, member: TeamMember) -> None:
        config = self._load_config()
        config.members.append(member)
        self._save_config(config)

    def remove_member(self, name: str) -> None:
        config = self._load_config()
        config.members = [m for m in config.members if m.name != name]
        self._save_config(config)

    def update_member_status(self, name: str, status: str) -> None:
        config = self._load_config()
        for m in config.members:
            if m.name == name:
                m.status = status
                break
        self._save_config(config)

    async def spawn_teammate(
        self,
        name: str,
        prompt: str,
        *,
        agent: str = "auto-approve",
        max_turns: int = 20,
    ) -> str:
        member = TeamMember(name=name, agent_type=agent, status="running")
        self.add_member(member)

        task = asyncio.create_task(self._run_teammate(name, prompt, agent, max_turns))
        self._teammate_tasks[name] = task
        return name

    async def _run_teammate(
        self, name: str, prompt: str, agent: str, max_turns: int
    ) -> None:
        try:
            cmd = [
                sys.executable,
                "-m",
                "vibe",
                "-p",
                prompt,
                "--agent",
                agent,
                "--trust",
                "--output",
                "text",
                "--max-turns",
                str(max_turns),
            ]

            env = os.environ.copy()
            env["VIBE_TEAM_NAME"] = self._team_name
            env["VIBE_TEAM_DIR"] = str(self._team_dir)
            env["VIBE_TEAMMATE_NAME"] = name

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self.update_member_status(name, f"running:pid={proc.pid}")
            config = self._load_config()
            for m in config.members:
                if m.name == name:
                    m.pid = proc.pid
                    break
            self._save_config(config)

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                result = stdout.decode() if stdout else ""
                self.update_member_status(name, "completed")
                logger.info("Teammate %s completed: %s chars output", name, len(result))
            else:
                err = stderr.decode() if stderr else "unknown error"
                self.update_member_status(name, f"failed:{err[:200]}")
                logger.error(
                    "Teammate %s failed (rc=%s): %s", name, proc.returncode, err
                )

        except Exception as e:
            self.update_member_status(name, f"error:{e}")
            logger.error("Teammate %s error", name, exc_info=e)

    async def wait_for_teammate(self, name: str) -> None:
        task = self._teammate_tasks.get(name)
        if task is not None:
            await task

    async def wait_for_all(self) -> None:
        await asyncio.gather(
            *[t for t in self._teammate_tasks.values() if not t.done()],
            return_exceptions=True,
        )

    async def stop_teammate(self, name: str) -> bool:
        task = self._teammate_tasks.get(name)
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.update_member_status(name, "stopped")
        return True

    async def stop_all(self) -> None:
        for name in list(self._teammate_tasks):
            await self.stop_teammate(name)

    def cleanup(self) -> None:
        import shutil

        try:
            shutil.rmtree(self._team_dir)
        except Exception as e:
            logger.warning("Failed to cleanup team dir %s: %s", self._team_dir, e)
