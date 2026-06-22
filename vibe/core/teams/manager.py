from __future__ import annotations

import asyncio
from collections.abc import Callable
import os
from pathlib import Path
import secrets
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

from filelock import FileLock

from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME
from vibe.core.teams.mailbox import Mailbox
from vibe.core.teams.models import Task, TeamConfig, TeamMember
from vibe.core.teams.task_store import TaskStore
from vibe.core.utils.io import read_safe

if TYPE_CHECKING:
    from vibe.core.hooks.manager import HooksManager
    from vibe.core.hooks.models import HookSessionContext


def _team_dir_for(team_name: str) -> Path:
    return VIBE_HOME.path / "teams" / team_name


class TeamManager:
    def __init__(
        self,
        lead_session_id: str,
        *,
        team_name: str | None = None,
        hooks_manager: HooksManager | None = None,
        hook_context: Callable[[], HookSessionContext | None] | None = None,
    ) -> None:
        self._team_name = team_name or f"team-{secrets.token_hex(4)}"
        self._team_dir = _team_dir_for(self._team_name)
        self._team_dir.mkdir(parents=True, exist_ok=True)
        self._config_file = self._team_dir / "config.json"
        self._config_lock = self._team_dir / "config.lock"
        self._lead_session_id = lead_session_id
        self._task_store: TaskStore | None = None
        self._mailbox: Mailbox | None = None
        self._teammate_tasks: dict[str, asyncio.Task[None]] = {}
        self._teammate_procs: dict[str, asyncio.subprocess.Process] = {}
        self._hooks_manager = hooks_manager
        self._hook_context = hook_context
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
            return TeamConfig.model_validate_json(read_safe(self._config_file).text)

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

    async def _dispatch_hook(self, hook_type: str, **fields: Any) -> None:
        """Fire a team lifecycle hook event (no-op without a hooks manager).

        TEAMMATE_IDLE / TASK_CREATED / TASK_COMPLETED are informational; hook
        output is logged but cannot gate the lifecycle action.
        """
        if self._hooks_manager is None or self._hook_context is None:
            return
        ctx = self._hook_context()
        if ctx is None:
            return
        from vibe.core.hooks.models import HookType, build_invocation

        try:
            invocation = build_invocation(HookType(hook_type), ctx, **fields)
            async for _event in self._hooks_manager.run(invocation):
                # Lifecycle hooks are informational; swallow events. (A real
                # UI can subscribe to the hooks manager stream separately.)
                pass
        except Exception:
            logger.warning("Team lifecycle hook %s failed", hook_type, exc_info=True)

    async def add_team_task(
        self,
        description: str,
        *,
        dependencies: list[str] | None = None,
        task_id: str | None = None,
    ) -> Task:
        """Create a task and fire the TASK_CREATED lifecycle hook."""
        task = self.task_store.add_task(
            description, dependencies=dependencies, task_id=task_id
        )
        await self._dispatch_hook(
            "task_created",
            task_id=task.id,
            task_description=task.description,
            assignee=task.assignee,
        )
        return task

    async def complete_team_task(
        self, task_id: str, result: str | None = None
    ) -> Task | None:
        """Mark a task complete and fire the TASK_COMPLETED lifecycle hook."""
        task = self.task_store.complete_task(task_id, result=result)
        if task is not None:
            await self._dispatch_hook(
                "task_completed",
                task_id=task.id,
                teammate_name=task.assignee or "lead",
                result=task.result,
            )
        return task

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
        proc: asyncio.subprocess.Process | None = None
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
                # Own session/process group so we can signal the whole tree
                # (the teammate + any bash-tool grandchildren) on stop; killing
                # only the direct child would orphan grandchildren to init.
                start_new_session=True,
            )
            self._teammate_procs[name] = proc

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
        finally:
            # If the task is dying (cancel/stop) with the subprocess still
            # running, signal the whole group. stop_teammate reaps via
            # _terminate_proc.
            if proc is not None and proc.returncode is None:
                self._signal_proc_group(proc, signal.SIGTERM)
            self._teammate_procs.pop(name, None)
            # The teammate is now idle (completed, failed, or stopped). Fire
            # the TEAMMATE_IDLE lifecycle hook so observers can react.
            await self._dispatch_hook(
                "teammate_idle", teammate_name=name, teammate_session_id=None
            )

    @staticmethod
    def _signal_proc_group(proc: asyncio.subprocess.Process, sig: int) -> None:
        """Signal the teammate's whole process group (it is a session leader via
        start_new_session, so its pid is its pgid), reaping bash-tool
        grandchildren. Falls back to signaling just the direct child if the
        group lookup fails (e.g. the process already exited).
        """
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            if sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except ProcessLookupError:
            pass

    async def _terminate_proc(self, name: str) -> None:
        """Terminate and reap a teammate subprocess (and its group) if still
        running.
        """
        proc = self._teammate_procs.get(name)
        if proc is None or proc.returncode is not None:
            return
        self._signal_proc_group(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except TimeoutError:
            self._signal_proc_group(proc, signal.SIGKILL)
            try:
                await proc.wait()
            except ProcessLookupError:
                pass

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
        # Terminate the subprocess so proc.communicate() unblocks and the task
        # can finish; cancel as a backstop if the task is stuck before the proc
        # was created. Previously task.cancel() alone left the trusted `vibe -p`
        # subprocess alive: CancelledError escaped `except Exception`, so `proc`
        # was never terminated while status was recorded as "stopped".
        await self._terminate_proc(name)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
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
