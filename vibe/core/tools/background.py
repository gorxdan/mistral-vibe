from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum, auto
import os
from pathlib import Path
import signal
import time
from typing import TYPE_CHECKING, Any

import orjson

from vibe.core.logger import logger
from vibe.core.tasking import TaskOutcome
from vibe.core.tools._background_delivery import prepare_background_completion

if TYPE_CHECKING:
    import asyncio.subprocess

    from vibe.cli.textual_ui.workflow_runner import WorkflowRunner
    from vibe.core.loop import LoopManager
    from vibe.core.teams.manager import TeamManager


_TERMINATE_GRACE_S = 3.0
_MAX_FINALIZED_PROCS = 50
_MAX_RUNNING_PROCS = 32
_LOG_TAIL_MAX_BYTES = 1 << 20
_LOG_DISK_CAP_BYTES = 16 << 20
_LOG_DISK_KEEP_BYTES = 4 << 20
_COMPLETION_DEBOUNCE_S = 0.1


class TaskCategory(StrEnum):
    PROCESS = auto()
    WORKFLOW = auto()
    AGENT = auto()
    TEAM = auto()
    LOOP = auto()
    ASYNC_AGENT = auto()


@dataclass
class TaskEntry:
    task_id: str
    category: TaskCategory
    label: str
    status: str
    elapsed: float
    detail: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    can_pause: bool = False
    can_save: bool = False


@dataclass
class _BgProc:
    task_id: str
    proc: asyncio.subprocess.Process
    command: str
    cwd: Path
    log_path: Path
    started_at: float
    status: str = "running"
    returncode: int | None = None
    finalizer: asyncio.Task[None] | None = None
    log_handle: Any = None


@dataclass
class _AsyncAgentRec:
    task_id: str
    agent: str
    label: str
    task: asyncio.Task[Any]
    started_at: float
    status: str = "running"
    finalizer: asyncio.Task[None] | None = None
    response: str = ""
    completed: bool = False
    worktree_path: str | None = None
    branch: str | None = None
    error: str | None = None
    prompt: str = ""
    model: str | None = None
    response_so_far: str = ""
    turns_used: int = 0
    log_path: Path | None = None
    outcome: TaskOutcome | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    artifact_size_bytes: int = 0
    terminal_queued: bool = False


def _signal_proc_group(proc: asyncio.subprocess.Process, sig: int) -> None:
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


async def _terminate_proc(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    _signal_proc_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_GRACE_S)
    except TimeoutError:
        _signal_proc_group(proc, signal.SIGKILL)
        try:
            await proc.wait()
        except ProcessLookupError:
            pass


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif btype in {"tool_use", "tool_result"}:
                    parts.append(f"[{block.get('name') or btype}]")
        return " ".join(parts)
    if content is None:
        return ""
    return str(content)


def _format_jsonl_tail(
    raw: str, *, content_limit: int = 160, max_role_width: int = 10
) -> str:
    if not raw:
        return ""
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = orjson.loads(line)
        except (orjson.JSONDecodeError, ValueError):
            out.append(line)
            continue
        if not isinstance(msg, dict):
            out.append(line)
            continue
        role = str(msg.get("role") or "?")[:max_role_width]
        text = _extract_message_text(msg.get("content")).replace("\n", " ").strip()
        if len(text) > content_limit:
            text = text[: content_limit - 1] + "…"
        out.append(f"{role}: {text}" if text else f"{role}:")
    return "\n".join(out)


class BackgroundRegistry:
    def __init__(self) -> None:
        self._procs: dict[str, _BgProc] = {}
        self._next_proc_id = 1
        self._async_agents: dict[str, _AsyncAgentRec] = {}
        self._next_async_id = 1
        self._async_completions: list[_AsyncAgentRec] = []

        self._workflow_runner_ref: Callable[[], WorkflowRunner | None] = lambda: None
        self._team_manager_ref: Callable[[], TeamManager | None] = lambda: None
        self._loop_manager_ref: Callable[[], LoopManager | None] = lambda: None
        self._tui_bash_ref: Callable[[], asyncio.Task | None] = lambda: None
        self._completion_callback: Callable[[], Coroutine[Any, Any, None]] | None = None
        self._completion_generation = 0
        self._completion_notify_task: asyncio.Task[None] | None = None

    @property
    def supports_async_agent_delivery(self) -> bool:
        return self._completion_callback is not None

    @property
    def has_pending_async_agent_completions(self) -> bool:
        return bool(self._async_completions)

    def attach_workflow_runner(self, ref: Callable[[], WorkflowRunner | None]) -> None:
        self._workflow_runner_ref = ref

    def attach_team_manager(self, ref: Callable[[], TeamManager | None]) -> None:
        self._team_manager_ref = ref

    def team_manager(self) -> TeamManager | None:
        """Read-only access for surfaces that need richer team data than the
        flat TaskEntry carries (inbox depth, task counts).
        """
        return self._team_manager_ref()

    def attach_loop_manager(self, ref: Callable[[], LoopManager | None]) -> None:
        self._loop_manager_ref = ref

    def attach_tui_bash(self, ref: Callable[[], asyncio.Task | None]) -> None:
        self._tui_bash_ref = ref

    def attach_completion_callback(
        self, callback: Callable[[], Coroutine[Any, Any, None]] | None
    ) -> None:
        # Wake hook: fires when async subagent finishes so idle host auto-continues.
        self._completion_callback = callback
        if callback is not None and self._async_completions:
            self._notify_completion()

    def notify_external_completion(self) -> None:
        self._notify_completion()

    def _notify_completion(self) -> None:
        if self._completion_callback is None:
            return
        self._completion_generation += 1
        if (
            self._completion_notify_task is not None
            and not self._completion_notify_task.done()
        ):
            return
        try:
            self._completion_notify_task = asyncio.create_task(
                self._run_completion_notifications(), name="background-wake-debounce"
            )
        except RuntimeError:
            pass

    async def _run_completion_notifications(self) -> None:
        observed_generation = -1
        try:
            while observed_generation != self._completion_generation:
                observed_generation = self._completion_generation
                await asyncio.sleep(_COMPLETION_DEBOUNCE_S)
                callback = self._completion_callback
                if callback is None:
                    return
                try:
                    await callback()
                except Exception as exc:
                    logger.warning("background completion wake failed: %s", exc)
        finally:
            self._completion_notify_task = None

    def _next_id(self) -> str:
        task_id = f"proc-{self._next_proc_id}"
        self._next_proc_id += 1
        return task_id

    async def register_process(
        self,
        proc: asyncio.subprocess.Process,
        *,
        command: str,
        cwd: Path,
        log_path: Path,
        log_handle: Any = None,
    ) -> str:
        running = sum(1 for r in self._procs.values() if r.status == "running")
        if running >= _MAX_RUNNING_PROCS:
            raise RuntimeError(
                f"background process cap reached ({_MAX_RUNNING_PROCS} running); "
                "stop an existing background task before starting another"
            )
        task_id = self._next_id()
        rec = _BgProc(
            task_id=task_id,
            proc=proc,
            command=command,
            cwd=cwd,
            log_path=log_path,
            started_at=time.monotonic(),
            log_handle=log_handle,
        )
        rec.finalizer = asyncio.create_task(
            self._finalize_proc(rec), name=f"bgproc-{task_id}"
        )
        self._procs[task_id] = rec
        self._reap_finalized()
        return task_id

    async def _finalize_proc(self, rec: _BgProc) -> None:
        try:
            rc = await rec.proc.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("bg proc %s wait failed: %s", rec.task_id, exc)
            rec.status = "failed"
            rec.returncode = -1
            self._close_log_handle(rec)
            return
        rec.returncode = rc
        rec.status = "completed" if rc == 0 else "failed"
        self._close_log_handle(rec)

    @staticmethod
    def _close_log_handle(rec: _BgProc) -> None:
        if rec.log_handle is None:
            return
        try:
            rec.log_handle.close()
        except Exception:
            pass
        rec.log_handle = None

    def _next_async_task_id(self) -> str:
        task_id = f"asub-{self._next_async_id}"
        self._next_async_id += 1
        return task_id

    def register_async_agent(
        self,
        agent: str,
        task: asyncio.Task[Any],
        *,
        label: str | None = None,
        prompt: str = "",
        model: str | None = None,
        log_path: Path | None = None,
    ) -> str:
        running = sum(1 for r in self._async_agents.values() if r.status == "running")
        if running >= _MAX_RUNNING_PROCS:
            raise RuntimeError(
                f"background async-agent cap reached ({_MAX_RUNNING_PROCS} running); "
                "stop an existing background task before starting another"
            )
        task_id = self._next_async_task_id()
        rec = _AsyncAgentRec(
            task_id=task_id,
            agent=agent,
            label=label or agent,
            task=task,
            started_at=time.monotonic(),
            prompt=prompt,
            model=model,
            log_path=log_path,
        )
        rec.finalizer = asyncio.create_task(
            self._finalize_async_agent(rec), name=f"bgasub-{task_id}"
        )
        self._async_agents[task_id] = rec
        return task_id

    def update_async_progress(
        self,
        task_id: str,
        *,
        response_so_far: str | None = None,
        turns_used: int | None = None,
    ) -> None:
        rec = self._async_agents.get(task_id)
        if rec is None or rec.status != "running":
            return
        if response_so_far is not None:
            rec.response_so_far = response_so_far
        if turns_used is not None:
            rec.turns_used = turns_used

    def update_async_progress_by_task(
        self,
        task: asyncio.Task[Any],
        *,
        response_so_far: str | None = None,
        turns_used: int | None = None,
    ) -> None:
        for rec in self._async_agents.values():
            if rec.task is task:
                self.update_async_progress(
                    rec.task_id, response_so_far=response_so_far, turns_used=turns_used
                )
                return

    async def _finalize_async_agent(self, rec: _AsyncAgentRec) -> None:
        try:
            result = await rec.task
        except asyncio.CancelledError:
            rec.status = "stopped"
            return
        except Exception as exc:
            rec.status = "failed"
            rec.completed = False
            error = str(exc) or exc.__class__.__name__
            error_path = (
                rec.log_path.with_suffix(f"{rec.log_path.suffix}.error")
                if rec.log_path is not None
                else None
            )
            rec.error, artifact = prepare_background_completion(error, error_path)
            rec.artifact_path = artifact.path
            rec.artifact_sha256 = artifact.sha256
            rec.artifact_size_bytes = artifact.size_bytes
            self._queue_async_completion(rec)
            return
        response = str(getattr(result, "output", result) or "")
        rec.response, artifact = prepare_background_completion(response, rec.log_path)
        rec.artifact_path = artifact.path
        rec.artifact_sha256 = artifact.sha256
        rec.artifact_size_bytes = artifact.size_bytes
        rec.completed = bool(getattr(result, "returncode", 1) == 0)
        rec.worktree_path = getattr(result, "worktree_path", None)
        rec.branch = getattr(result, "branch", None)
        rec.outcome = getattr(result, "outcome", None)
        if rec.completed and (rec.outcome is None or rec.outcome.succeeded):
            rec.status = "completed"
        else:
            rec.status = "failed"
        self._queue_async_completion(rec)

    def _queue_async_completion(self, rec: _AsyncAgentRec) -> None:
        if rec.terminal_queued:
            return
        rec.terminal_queued = True
        self._async_completions.append(rec)
        self._notify_completion()

    def pop_async_completions(self) -> list[_AsyncAgentRec]:
        if not self._async_completions:
            return []
        drained = self._async_completions[:]
        self._async_completions.clear()
        return drained

    def _reap_finalized(self) -> None:
        finalized = [tid for tid, r in self._procs.items() if r.status != "running"]
        if len(finalized) <= _MAX_FINALIZED_PROCS:
            return
        for tid in sorted(finalized, key=lambda t: int(t.split("-")[1]))[
            : len(finalized) - _MAX_FINALIZED_PROCS
        ]:
            self._procs.pop(tid, None)

    def read_log_tail(self, task_id: str, *, lines: int = 50) -> str:
        rec = self._procs.get(task_id)
        if rec is None:
            return ""
        path = rec.log_path
        try:
            if path.stat().st_size > _LOG_DISK_CAP_BYTES:
                self._trim_log_in_place(path)
        except (FileNotFoundError, OSError):
            return ""
        return self._tail_bytes(path, lines)

    @staticmethod
    def _tail_bytes(path: Path, lines: int) -> str:
        try:
            data = path.read_bytes()[-_LOG_TAIL_MAX_BYTES:]
        except (FileNotFoundError, OSError):
            return ""
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def read_async_tail(self, task_id: str, *, lines: int = 50) -> str:
        rec = self._async_agents.get(task_id)
        if rec is None:
            return ""
        if rec.log_path is not None:
            tail = self._tail_bytes(rec.log_path, lines)
            if tail:
                return tail
        text = rec.response_so_far or rec.response
        if not text:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def read_agent_log_tail(self, task_id: str, *, lines: int = 50) -> str:
        run_id, agent_id = self._parse_agent_task_id(task_id)
        if run_id is None or agent_id is None:
            return ""
        runner = self._workflow_runner_ref()
        if runner is None:
            return ""
        for entry in runner.runs:
            if getattr(entry, "run_id", None) != run_id:
                continue
            for la in getattr(entry, "live_agents", None) or []:
                if getattr(la, "agent_id", None) != agent_id:
                    continue
                path = getattr(la, "log_path", None)
                if path is None:
                    return ""
                return _format_jsonl_tail(self._tail_bytes(Path(path), lines))
        return ""

    @staticmethod
    def _parse_agent_task_id(task_id: str) -> tuple[str | None, str | None]:
        if "/" not in task_id:
            return None, None
        run_id, _, agent_suffix = task_id.partition("/")
        prefix = "live-"
        if not agent_suffix.startswith(prefix):
            return None, None
        return run_id, agent_suffix[len(prefix) :]

    @staticmethod
    def _trim_log_in_place(path: Path) -> None:
        try:
            keep = path.read_bytes()[-_LOG_DISK_KEEP_BYTES:]
            with path.open("r+b") as fh:
                fh.seek(0)
                fh.write(keep)
                fh.truncate()
        except OSError:
            pass

    def list_tasks(self, *, category: TaskCategory | None = None) -> list[TaskEntry]:
        entries: list[TaskEntry] = []
        now = time.monotonic()

        if category in {None, TaskCategory.PROCESS}:
            for rec in sorted(
                self._procs.values(),
                key=lambda r: (r.status != "running", int(r.task_id.split("-")[1])),
            ):
                entries.append(
                    TaskEntry(
                        task_id=rec.task_id,
                        category=TaskCategory.PROCESS,
                        label=rec.command,
                        status=rec.status,
                        elapsed=now - rec.started_at,
                        detail={
                            "pid": rec.proc.pid,
                            "returncode": rec.returncode,
                            "cwd": str(rec.cwd),
                            "log_path": str(rec.log_path),
                        },
                    )
                )

        if category in {None, TaskCategory.WORKFLOW, TaskCategory.AGENT}:
            entries.extend(self._workflow_entries(category, now))

        if category in {None, TaskCategory.TEAM}:
            entries.extend(self._team_entries(now))

        if category in {None, TaskCategory.LOOP}:
            entries.extend(self._loop_entries(now))

        if category in {None, TaskCategory.ASYNC_AGENT}:
            for rec in sorted(
                self._async_agents.values(),
                key=lambda r: (r.status != "running", int(r.task_id.split("-")[1])),
            ):
                entries.append(
                    TaskEntry(
                        task_id=rec.task_id,
                        category=TaskCategory.ASYNC_AGENT,
                        label=rec.label,
                        status=rec.status,
                        elapsed=now - rec.started_at,
                        detail={
                            "agent": rec.agent,
                            "completed": rec.completed,
                            "worktree_path": rec.worktree_path,
                            "branch": rec.branch,
                            "error": rec.error,
                            "prompt": rec.prompt,
                            "model": rec.model,
                            "response_so_far": rec.response_so_far,
                            "turns_used": rec.turns_used,
                            "log_path": str(rec.log_path) if rec.log_path else None,
                            "outcome": (
                                rec.outcome.model_dump(mode="json", exclude_none=True)
                                if rec.outcome is not None
                                else None
                            ),
                        },
                    )
                )

        return entries

    def _workflow_entries(
        self, filter_cat: TaskCategory | None, now: float
    ) -> list[TaskEntry]:
        runner = self._workflow_runner_ref()
        if runner is None:
            return []
        entries: list[TaskEntry] = []
        for entry in runner.runs:
            include = filter_cat in {None, TaskCategory.WORKFLOW}
            live_agents = list(getattr(entry, "live_agents", None) or [])
            phases = list(getattr(entry, "phases", None) or [])
            label = ", ".join(phases) or "(no phases)"
            detail = {
                "agent_count": getattr(entry, "agent_count", 0),
                "tokens_total": getattr(entry, "tokens_total", 0),
                "live_agent_count": len(live_agents),
            }
            if include:
                entries.append(
                    TaskEntry(
                        task_id=entry.run_id,
                        category=TaskCategory.WORKFLOW,
                        label=label,
                        status=getattr(entry.status, "value", str(entry.status)),
                        elapsed=getattr(entry, "elapsed", 0.0),
                        detail=detail,
                        can_pause=True,
                        can_save=True,
                    )
                )
            if filter_cat in {None, TaskCategory.AGENT}:
                for la in live_agents:
                    agent_id = getattr(la, "agent_id", None) or str(id(la))
                    entries.append(
                        TaskEntry(
                            task_id=f"{entry.run_id}/live-{agent_id}",
                            category=TaskCategory.AGENT,
                            label=getattr(la, "label", None) or agent_id,
                            status="running",
                            elapsed=0.0,
                            detail={
                                "phase": getattr(la, "phase", None),
                                "tokens_total": getattr(la, "tokens_total", 0),
                                "agent": getattr(la, "agent", None),
                                "model": getattr(la, "model", None),
                                "prompt": getattr(la, "prompt", "") or "",
                                "response_preview": getattr(la, "response_so_far", "")
                                or "",
                            },
                            parent_id=entry.run_id,
                        )
                    )
        return entries

    def _team_entries(self, now: float) -> list[TaskEntry]:
        manager = self._team_manager_ref()
        if manager is None:
            return []
        entries: list[TaskEntry] = []
        try:
            members = manager.get_members()
        except Exception as exc:
            logger.debug("team get_members failed: %s", exc)
            return []
        active_tasks: dict[str, Any] = {}
        try:
            for task in manager.task_store.get_all_tasks():
                status = getattr(getattr(task, "status", None), "value", None)
                if status != "in_progress" or not getattr(task, "assignee", None):
                    continue
                active_tasks[str(task.assignee)] = task
        except Exception as exc:
            logger.debug("team task_store get_all_tasks failed: %s", exc)
        wall_now = time.time()
        for m in members:
            status = _team_status(getattr(m, "status", "") or "")
            active_task = active_tasks.get(m.name)
            claimed_at = getattr(m, "last_claimed_at", None)
            if claimed_at is None and active_task is not None:
                claimed_at = getattr(active_task, "claimed_at", None)
            lease_age_s = wall_now - claimed_at if claimed_at is not None else None
            last_task_id = getattr(m, "last_task_id", None)
            if last_task_id is None and active_task is not None:
                last_task_id = getattr(active_task, "id", None)
            safety_mode = getattr(m, "safety_mode", None)
            entries.append(
                TaskEntry(
                    task_id=f"team:{m.name}",
                    category=TaskCategory.TEAM,
                    label=getattr(m, "agent_type", "teammate"),
                    status=status,
                    elapsed=0.0,
                    detail={
                        "name": m.name,
                        "pid": getattr(m, "pid", None),
                        "raw_status": getattr(m, "status", ""),
                        "spawn_prompt": getattr(m, "spawn_prompt", None),
                        "max_turns": getattr(m, "max_turns", None),
                        "worker": getattr(m, "worker", False),
                        "safety_mode": getattr(safety_mode, "value", safety_mode),
                        "last_task_id": last_task_id,
                        "last_claimed_at": claimed_at,
                        "lease_age_s": lease_age_s,
                    },
                )
            )
        return entries

    def _loop_entries(self, now: float) -> list[TaskEntry]:
        manager = self._loop_manager_ref()
        if manager is None:
            return []
        entries: list[TaskEntry] = []
        wall_now = time.time()
        try:
            loops = list(manager.loops)
        except Exception as exc:
            logger.debug("loop list failed: %s", exc)
            return []
        for loop in loops:
            remaining = max(0.0, loop.next_fire_at - wall_now)
            entries.append(
                TaskEntry(
                    task_id=f"loop-{loop.id}",
                    category=TaskCategory.LOOP,
                    label=loop.prompt,
                    status="waiting",
                    elapsed=remaining,
                    detail={
                        "loop_id": loop.id,
                        "interval_seconds": loop.interval_seconds,
                        "next_fire_at": loop.next_fire_at,
                        "recurring": loop.recurring,
                        "remaining_seconds": remaining,
                    },
                )
            )
        return entries

    async def stop(self, task_id: str) -> bool:
        if task_id.startswith(("proc-", "asub-")):
            return await self._stop_owned(task_id)
        if task_id.startswith("team:"):
            return await self._stop_team(task_id.removeprefix("team:"))
        if task_id.startswith("loop-"):
            return await self._stop_loop(task_id.removeprefix("loop-"))
        if "/" in task_id and task_id.startswith("wf-"):
            run_id, _, agent_suffix = task_id.partition("/")
            agent_id = (
                agent_suffix.removeprefix("live-")
                if agent_suffix.startswith("live-")
                else agent_suffix
            )
            return self._cancel_workflow_agent(run_id, agent_id)
        if task_id.startswith("wf-"):
            return await self._stop_workflow(task_id)
        return False

    async def _stop_owned(self, task_id: str) -> bool:
        if task_id.startswith("proc-"):
            return await self._stop_process(task_id)
        return await self._stop_async_agent(task_id)

    async def pause(self, task_id: str) -> bool:
        runner = self._workflow_runner_ref()
        if runner is None or not task_id.startswith("wf-"):
            return False
        entry = runner.find_run(task_id)
        if entry is None or getattr(entry, "result", None) is not None:
            return False
        if getattr(entry, "is_paused", False):
            return bool(runner.unpause(task_id))
        return bool(runner.pause(task_id))

    async def _stop_process(self, task_id: str) -> bool:
        rec = self._procs.get(task_id)
        if rec is None or rec.status != "running":
            return False
        if rec.finalizer is not None and not rec.finalizer.done():
            rec.finalizer.cancel()
        await _terminate_proc(rec.proc)
        rec.status = "stopped"
        rec.returncode = rec.proc.returncode
        self._close_log_handle(rec)
        return True

    async def _stop_async_agent(self, task_id: str) -> bool:
        rec = self._async_agents.get(task_id)
        if rec is None or rec.status != "running":
            return False
        if rec.finalizer is not None and not rec.finalizer.done():
            rec.finalizer.cancel()
        rec.task.cancel()
        try:
            await rec.task
        except (asyncio.CancelledError, Exception):
            pass
        rec.status = "stopped"
        rec.completed = False
        self._queue_async_completion(rec)
        return True

    async def _stop_workflow(self, run_id: str) -> bool:
        runner = self._workflow_runner_ref()
        if runner is None:
            return False
        try:
            return await runner.stop(run_id)
        except Exception as exc:
            logger.warning("workflow stop %s failed: %s", run_id, exc)
            return False

    def _cancel_workflow_agent(self, run_id: str, agent_id: str) -> bool:
        runner = self._workflow_runner_ref()
        if runner is None:
            return False
        try:
            return bool(runner.cancel_agent(run_id, agent_id))
        except Exception as exc:
            logger.warning("cancel agent %s/%s failed: %s", run_id, agent_id, exc)
            return False

    async def _stop_team(self, name: str) -> bool:
        manager = self._team_manager_ref()
        if manager is None:
            return False
        try:
            return await manager.stop_teammate(name)
        except Exception as exc:
            logger.warning("team stop %s failed: %s", name, exc)
            return False

    async def _stop_loop(self, loop_id: str) -> bool:
        manager = self._loop_manager_ref()
        if manager is None:
            return False
        try:
            count = await manager.cancel(loop_id)
        except Exception as exc:
            logger.warning("loop cancel %s failed: %s", loop_id, exc)
            return False
        return count > 0

    async def shutdown(self) -> None:
        if (
            self._completion_notify_task is not None
            and not self._completion_notify_task.done()
        ):
            self._completion_notify_task.cancel()
            try:
                await self._completion_notify_task
            except asyncio.CancelledError:
                pass
        for rec in list(self._procs.values()):
            if rec.status != "running":
                continue
            if rec.finalizer is not None and not rec.finalizer.done():
                rec.finalizer.cancel()
            try:
                await _terminate_proc(rec.proc)
            except Exception as exc:
                logger.warning("shutdown: failed to reap %s: %s", rec.task_id, exc)
            else:
                rec.status = "stopped"
                rec.returncode = rec.proc.returncode
            self._close_log_handle(rec)

        for rec in list(self._async_agents.values()):
            if rec.status != "running":
                continue
            if rec.finalizer is not None and not rec.finalizer.done():
                rec.finalizer.cancel()
            rec.task.cancel()
            try:
                await rec.task
            except (asyncio.CancelledError, Exception):
                pass
            rec.status = "stopped"


def _team_status(raw: str) -> str:
    if not raw:
        return "running"
    head = raw.split(":", 1)[0]
    if head == "error":
        return "failed"
    if head in {"running", "completed", "failed", "stopped"}:
        return head
    return "running"
