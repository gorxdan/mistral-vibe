from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
import contextlib
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
from vibe.core.tasking import TaskBrief, TaskOutcome
from vibe.core.teams._safety import TEAM_SAFETY_MODE_ENV
from vibe.core.teams.mailbox import Mailbox, validate_member_name
from vibe.core.teams.models import (
    Task,
    TaskStatus,
    TeamConfig,
    TeamMember,
    TeamSafetyMode,
)
from vibe.core.teams.task_store import TaskStore
from vibe.core.utils.io import read_safe, write_safe

if TYPE_CHECKING:
    from vibe.core.hooks.manager import HooksManager
    from vibe.core.hooks.models import HookSessionContext
    from vibe.core.usage._process_context import SpendProcessContext
    from vibe.core.usage._session import SessionSpendAdapter


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
        spend_adapter: SessionSpendAdapter | None = None,
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
        self._spend_adapter = spend_adapter
        self._spend_group_id = f"team:{self._team_name}"
        self._init_config()

    def _new_process_spend_context(self, name: str) -> SpendProcessContext | None:
        if self._spend_adapter is None:
            return None
        from vibe.core.usage._context import SpendPurpose, SpendScopeKind

        adapter = self._spend_adapter.child_agent(
            group_kind=SpendScopeKind.TEAM,
            group_id=self._spend_group_id,
            agent_id=(f"agent:{self._team_name}:{name}:{secrets.token_hex(4)}"),
            purpose=SpendPurpose.TEAM,
        )
        return adapter.export_process_context()

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
            write_safe(self._config_file, config.model_dump_json(indent=2))

    @contextlib.contextmanager
    def _mutate_config(self) -> Iterator[TeamConfig]:
        lock = FileLock(str(self._config_lock), timeout=5)
        with lock:
            config = TeamConfig.model_validate_json(read_safe(self._config_file).text)
            yield config
            write_safe(self._config_file, config.model_dump_json(indent=2))

    def get_config(self) -> TeamConfig:
        return self._load_config()

    def get_members(self) -> list[TeamMember]:
        return self._load_config().members

    def add_member(self, member: TeamMember) -> None:
        with self._mutate_config() as config:
            config.members.append(member)

    def remove_member(self, name: str) -> None:
        with self._mutate_config() as config:
            config.members = [m for m in config.members if m.name != name]

    def update_member_status(self, name: str, status: str) -> None:
        with self._mutate_config() as config:
            for m in config.members:
                if m.name == name:
                    m.status = status
                    break

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
        description: str | TaskBrief,
        *,
        dependencies: list[str] | None = None,
        task_id: str | None = None,
    ) -> Task:
        task = await asyncio.to_thread(
            self.task_store.add_task,
            description,
            dependencies=dependencies,
            task_id=task_id,
        )
        await self._dispatch_hook(
            "task_created",
            task_id=task.id,
            task_description=task.description,
            assignee=task.assignee,
        )
        return task

    async def complete_team_task(
        self, task_id: str, result: str | TaskOutcome | None = None
    ) -> Task | None:
        task = await asyncio.to_thread(
            self.task_store.complete_task, task_id, result=result
        )
        if task is not None and task.status is TaskStatus.COMPLETED:
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
        worker: bool = False,
        lease_s: float | None = None,
        safety_mode: TeamSafetyMode | str = TeamSafetyMode.SHARED,
    ) -> str:
        # Validate the name at the boundary: it becomes a mailbox inbox path
        # (via _safe_name) and a teammate env var, so a path-looking value like
        # "../evil" would otherwise register a teammate that team_message later
        # refuses to address. Apply the mailbox's rule here so both agree.
        validate_member_name(name)
        from vibe.core.teams.worker_loop import worker_bootstrap_prompt

        safety = TeamSafetyMode(safety_mode)
        spawn_prompt = worker_bootstrap_prompt(prompt) if worker else prompt
        # Cap stored prompt so config.json stays small; full text still goes to -p.
        max_stored = 2000
        stored_prompt = (
            spawn_prompt
            if len(spawn_prompt) <= max_stored
            else spawn_prompt[:max_stored]
        )
        member = TeamMember(
            name=name,
            agent_type=agent,
            status="running",
            spawn_prompt=stored_prompt,
            max_turns=max_turns,
            worker=worker,
            safety_mode=safety,
        )
        await asyncio.to_thread(self.add_member, member)

        task = asyncio.create_task(
            self._run_teammate(
                name,
                spawn_prompt,
                agent,
                max_turns,
                worker=worker,
                lease_s=lease_s,
                safety_mode=safety,
            )
        )
        self._teammate_tasks[name] = task
        return name

    async def _run_teammate(
        self,
        name: str,
        prompt: str,
        agent: str,
        max_turns: int,
        *,
        worker: bool = False,
        lease_s: float | None = None,
        safety_mode: TeamSafetyMode = TeamSafetyMode.SHARED,
    ) -> None:
        proc: asyncio.subprocess.Process | None = None
        try:
            cmd = [
                sys.executable,
                "-m",
                "vibe.cli.entrypoint",
                "-p",
                prompt,
                "--agent",
                agent,
                "--trust",
                "--output",
                "text",
                "--max-turns",
                str(max_turns),
                # Teams coordinate over a SHARED working tree (TaskStore/Mailbox
                # + filelock); a private teammate worktree would strand its edits.
                "--no-worktree",
            ]

            from vibe.core.teams.task_store import DEFAULT_TASK_LEASE_S
            from vibe.core.teams.worker_loop import TEAM_LEASE_ENV, TEAM_WORKER_ENV
            from vibe.core.tools.sandbox import scrub_child_env
            from vibe.core.usage._process_context import install_spend_process_context

            # Same as isolated agents: provider keys stay, host creds stripped.
            env = scrub_child_env()
            env["VIBE_TEAM_NAME"] = self._team_name
            env["VIBE_TEAM_DIR"] = str(self._team_dir)
            env["VIBE_TEAMMATE_NAME"] = name
            env[TEAM_SAFETY_MODE_ENV] = safety_mode.value
            install_spend_process_context(env, self._new_process_spend_context(name))
            if worker:
                env[TEAM_WORKER_ENV] = "1"
                env[TEAM_LEASE_ENV] = str(
                    lease_s if lease_s is not None else DEFAULT_TASK_LEASE_S
                )

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

            await asyncio.to_thread(
                self.update_member_status, name, f"running:pid={proc.pid}"
            )

            def _stamp_pid() -> None:
                with self._mutate_config() as config:
                    for m in config.members:
                        if m.name == name:
                            m.pid = proc.pid
                            break

            await asyncio.to_thread(_stamp_pid)

            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                result = stdout.decode() if stdout else ""
                await asyncio.to_thread(self.update_member_status, name, "completed")
                logger.info("Teammate %s completed: %s chars output", name, len(result))
            else:
                err = stderr.decode() if stderr else "unknown error"
                await asyncio.to_thread(
                    self.update_member_status, name, f"failed:{err[:200]}"
                )
                logger.error(
                    "Teammate %s failed (rc=%s): %s", name, proc.returncode, err
                )

        except Exception as e:
            await asyncio.to_thread(self.update_member_status, name, f"error:{e}")
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
        await asyncio.to_thread(self.update_member_status, name, "stopped")
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
