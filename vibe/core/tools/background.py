"""Unified background-task registry.

Aggregates six categories of "background thing" into one read/cancel surface so
the TUI's Tasks pane and the model-facing `background` tool see the same state:

  - process    : agent bash spawns with background=True (OWNED here — the only
                 place a process table exists; Bash.run() otherwise drops the PID)
  - workflow   : workflow runs (read from WorkflowRunner)
  - agent      : in-flight workflow agents (read from WorkflowRuntime._live_agents)
  - team       : teammate subprocesses (read from TeamManager)
  - loop       : schedule timers (read from LoopManager)
  - async_agent: task(async_run=true) subagents (OWNED here as asyncio tasks)

The registry owns processes and async subagents outright and delegates everything
else to the subsystem that already owns it, via injected refs. Nothing duplicates
state. Cancellation routes by task-id prefix to the right owner's stop method, so
the Tasks pane, the `background` tool, and any future caller all share one path.

See docs/design/tasks.md for the full design.
"""

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

if TYPE_CHECKING:
    import asyncio.subprocess

    from vibe.cli.textual_ui.workflow_runner import WorkflowRunner
    from vibe.core.loop import LoopManager
    from vibe.core.teams.manager import TeamManager


# Process termination backoff — mirrors TeamManager._terminate_proc semantics
# (SIGTERM, wait, then SIGKILL) so a backgrounded server and its children are
# reliably reaped rather than orphaned to init.
_TERMINATE_GRACE_S = 3.0

# Soft cap on finalized-process entries retained in the pane. Without this a
# long session accumulates every command ever backgrounded.
_MAX_FINALIZED_PROCS = 50

# Hard cap on concurrently RUNNING background processes. Without it a looping
# or injected agent can spawn thousands of shells (each holding an asyncio
# finalizer task + OS fds) and exhaust the host. Finalized entries are not
# counted — only live processes.
_MAX_RUNNING_PROCS = 32

# Ceiling on how much of a background log file read_log_tail will read into
# memory before splitting into lines. Chatty servers can write a lot; the
# per-file hard cap (enforced at write time by the bash tool) is larger.
_LOG_TAIL_MAX_BYTES = 1 << 20  # 1 MiB

# Write-side disk cap: when a background log exceeds this, read_log_tail trims
# it in place to _LOG_DISK_KEEP_BYTES. The pane polls every second, so a chatty
# server's log is bounded near the cap rather than growing unbounded for the
# session lifetime.
_LOG_DISK_CAP_BYTES = 16 << 20  # 16 MiB
_LOG_DISK_KEEP_BYTES = 4 << 20  # 4 MiB retained after a trim


class TaskCategory(StrEnum):
    PROCESS = auto()
    WORKFLOW = auto()
    AGENT = auto()
    TEAM = auto()
    LOOP = auto()
    ASYNC_AGENT = auto()


@dataclass
class TaskEntry:
    """Unified view-model row for one background task, any category.

    `detail` carries category-specific fields the renderer formats (pid,
    returncode, agent_count, tokens, interval, next_fire_at, log_path…). Kept
    as a plain dict so the registry never imports TUI or rendering types.
    """

    task_id: str
    category: TaskCategory
    label: str
    status: str  # running | completed | failed | paused | stopped | waiting
    elapsed: float
    detail: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    can_pause: bool = False
    can_save: bool = False


@dataclass
class _BgProc:
    """Owned background process record.

    `finalizer` awaits proc.wait() in the background and flips status when the
    process exits on its own (server crash, self-terminate). It is cancelled on
    explicit stop / registry shutdown. `log_handle` is the open file object the
    process writes to via fd-level stdout/stderr redirection (None when the
    caller used shell-level redirection); the registry closes it once the
    process is no longer running.
    """

    task_id: str
    proc: asyncio.subprocess.Process
    command: str
    cwd: Path
    log_path: Path
    started_at: float
    status: str = "running"  # running | completed | failed | stopped
    returncode: int | None = None
    finalizer: asyncio.Task[None] | None = None
    log_handle: Any = None


@dataclass
class _AsyncAgentRec:
    """Owned async-subagent record.

    The TaskTool spawns an isolated subagent as an ``asyncio.Task`` (via
    ``run_isolated_agent``), registers it here, and returns immediately with
    the task_id. ``finalizer`` awaits the asyncio.Task and captures its
    IsolatedResult; on completion the record moves to ``completed``/``failed``
    and the result is queued for the parent agent loop to drain via
    ``pop_async_completions``. Stopped records move to ``stopped`` and do not
    queue a completion (the stop is explicit; no surprise event).
    """

    task_id: str
    agent: str
    label: str
    task: asyncio.Task[Any]
    started_at: float
    status: str = "running"  # running | completed | failed | stopped
    finalizer: asyncio.Task[None] | None = None
    response: str = ""
    completed: bool = False
    worktree_path: str | None = None
    branch: str | None = None
    error: str | None = None
    # Observability fields. ``prompt``/``model`` are set at registration so the
    # Tasks pane and `background` tool can show what the agent was asked to do
    # without re-deriving it. ``response_so_far``/``turns_used`` are updated live
    # by the in-process collector; ``log_path`` is the file the isolated
    # subprocess streams to, tailed like a PROCESS log.
    prompt: str = ""
    model: str | None = None
    response_so_far: str = ""
    turns_used: int = 0
    log_path: Path | None = None


def _signal_proc_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the process group led by `proc`.

    Backgrounded processes are spawned with start_new_session=True (set by the
    bash tool), so proc.pid is the session/pgid leader — killpg reaches the
    shell AND any grandchildren (npm/vite/python child servers) that would
    otherwise orphan. Falls back to signaling the direct child if the group
    lookup fails (already-exited, permission, etc.).
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


async def _terminate_proc(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM, wait the grace period, then SIGKILL. Reaps the process."""
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
    """Pull a plain-text snippet from an LLMMessage ``content`` field.

    ``content`` may be a string or a list of content blocks (text, tool_use,
    tool_result). Text blocks contribute their text; tool blocks contribute a
    short ``[name]`` marker so the tail shows tool activity without dumping
    whole payloads.
    """
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
    """Render raw messages.jsonl text as readable ``role: content`` snippets.

    Each line is a JSON-serialized LLMMessage. Malformed or partial trailing
    writes are passed through verbatim so an in-progress append never blanks the
    tail. Long content is truncated to ``content_limit`` chars; the role label
    is capped to ``max_role_width`` chars.
    """
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
            text = text[: content_limit - 1] + "\u2026"
        out.append(f"{role}: {text}" if text else f"{role}:")
    return "\n".join(out)


class BackgroundRegistry:
    """Owns background processes; aggregates workflows/teams/loops read-only.

    Attach the subsystem owners once at TUI startup; they are read lazily via
    the refs so a lazily-created owner (e.g. TeamManager) is picked up when it
    appears. All methods tolerate a missing owner (returns empty / False).
    """

    def __init__(self) -> None:
        self._procs: dict[str, _BgProc] = {}
        self._next_proc_id = 1
        self._async_agents: dict[str, _AsyncAgentRec] = {}
        self._next_async_id = 1
        self._async_completions: list[_AsyncAgentRec] = []

        # Refs default to "absent"; wired by the TUI app after construction.
        self._workflow_runner_ref: Callable[[], WorkflowRunner | None] = lambda: None
        self._team_manager_ref: Callable[[], TeamManager | None] = lambda: None
        self._loop_manager_ref: Callable[[], LoopManager | None] = lambda: None
        self._tui_bash_ref: Callable[[], asyncio.Task | None] = lambda: None
        self._completion_callback: Callable[[], Coroutine[Any, Any, None]] | None = None

    # --- adapter wiring ---------------------------------------------------

    def attach_workflow_runner(self, ref: Callable[[], WorkflowRunner | None]) -> None:
        self._workflow_runner_ref = ref

    def attach_team_manager(self, ref: Callable[[], TeamManager | None]) -> None:
        self._team_manager_ref = ref

    def attach_loop_manager(self, ref: Callable[[], LoopManager | None]) -> None:
        self._loop_manager_ref = ref

    def attach_tui_bash(self, ref: Callable[[], asyncio.Task | None]) -> None:
        """Surface the TUI's foreground `!cmd` slot (v2 hook; unused in v1)."""
        self._tui_bash_ref = ref

    def attach_completion_callback(
        self, callback: Callable[[], Coroutine[Any, Any, None]] | None
    ) -> None:
        # Wake hook: fired fire-and-forget when an async subagent finishes so an
        # idle host auto-continues instead of stalling until the user types.
        self._completion_callback = callback

    def _notify_completion(self) -> None:
        if self._completion_callback is None:
            return
        try:
            asyncio.create_task(self._completion_callback())
        except RuntimeError:
            pass

    # --- process ownership ------------------------------------------------

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
        """Record a backgrounded process and start its exit-watcher.

        Returns the stable task_id ("proc-N") the caller yields back to the
        model. The process is NOT awaited here — the caller (Bash.run) returns
        immediately so the agent turn unblocks.

        Raises ``RuntimeError`` if the concurrent running-process cap would be
        exceeded, so the caller (bash tool) surfaces a clear error instead of
        silently growing the table to exhaustion. ``log_handle`` is the open
        file object used for fd-level stdout/stderr redirection; the registry
        owns closing it when the process leaves the running state.
        """
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
        """Background awaiter: flip status when the process exits on its own.

        Explicit stop() sets status='stopped' itself and cancels this task, so
        reaching here means the process ended without intervention (clean exit
        or crash). Any exception is swallowed — the finalizer must never raise
        into the event loop.
        """
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
        """Close the fd-level redirection handle, idempotently. Called from the
        finalizer, _stop_process, and shutdown — safe to call multiple times.
        """
        if rec.log_handle is None:
            return
        try:
            rec.log_handle.close()
        except Exception:
            pass
        rec.log_handle = None

    # --- async subagent ownership ----------------------------------------

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
        """Record an async subagent and start its completion-watcher.

        Returns the stable task_id ("asub-N") the caller yields back to the
        model. The asyncio.Task is NOT awaited here — the TaskTool returns
        immediately so the agent turn unblocks. On completion the result is
        queued for the parent agent loop via ``pop_async_completions``.

        ``prompt``/``model``/``log_path`` are observability-only: stored on the
        record so the Tasks pane and ``background`` tool can render what the
        agent is doing. ``log_path`` is the file an isolated subprocess streams
        its stdout to (tailed live like a PROCESS log); in-process agents have
        none and are monitored via ``response_so_far`` updates instead.
        """
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
        """Live-update an in-process async subagent's progress on its record.

        Called by the TaskTool's in-process collector as events stream so the
        Tasks pane's 1s poll and the ``background`` tool see partial output
        while the agent runs, not only after it finishes. No-op for unknown or
        finalized ids — a stale update after completion must not clobber the
        captured response. Same-event-loop writes only; no locking needed.
        """
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
        """Update progress keyed by the running asyncio.Task instead of id.

        The in-process collector is started before ``register_async_agent``
        assigns its task_id, so it identifies itself via ``asyncio.current_task``
        rather than threading the id through a closure. Linear scan over the
        small in-process agent table; no-op when the task is not registered.
        """
        for rec in self._async_agents.values():
            if rec.task is task:
                self.update_async_progress(
                    rec.task_id, response_so_far=response_so_far, turns_used=turns_used
                )
                return

    async def _finalize_async_agent(self, rec: _AsyncAgentRec) -> None:
        """Background awaiter: capture the IsolatedResult when the task ends.

        Exceptions are swallowed — the finalizer must never raise into the
        event loop. A captured failure is recorded on the rec and queued like
        a success; the parent loop sees it as ``completed=False``.
        """
        try:
            result = await rec.task
        except asyncio.CancelledError:
            rec.status = "stopped"
            return
        except Exception as exc:
            rec.status = "failed"
            rec.completed = False
            rec.error = str(exc) or exc.__class__.__name__
            self._async_completions.append(rec)
            self._notify_completion()
            return
        # The TaskTool's background wrapper returns an IsolatedResult-like
        # object (see run_isolated_agent). Read its fields defensively.
        rec.response = str(getattr(result, "output", result) or "")
        rec.completed = bool(getattr(result, "returncode", 1) == 0)
        rec.worktree_path = getattr(result, "worktree_path", None)
        rec.branch = getattr(result, "branch", None)
        if rec.completed:
            rec.status = "completed"
        else:
            rec.status = "failed"
        self._async_completions.append(rec)
        self._notify_completion()

    def pop_async_completions(self) -> list[_AsyncAgentRec]:
        """Drain and return the async-subagent completions queued since the
        last call. Called by the parent agent loop at the top of each turn.
        Returns an empty list when nothing is pending (the common case).
        """
        if not self._async_completions:
            return []
        drained = self._async_completions[:]
        self._async_completions.clear()
        return drained

    def _reap_finalized(self) -> None:
        """Drop oldest finalized entries beyond the cap; never reap running."""
        finalized = [tid for tid, r in self._procs.items() if r.status != "running"]
        if len(finalized) <= _MAX_FINALIZED_PROCS:
            return
        # Finalized entries are appended in id order; drop the lowest ids.
        for tid in sorted(finalized, key=lambda t: int(t.split("-")[1]))[
            : len(finalized) - _MAX_FINALIZED_PROCS
        ]:
            self._procs.pop(tid, None)

    def read_log_tail(self, task_id: str, *, lines: int = 50) -> str:
        """Last `lines` lines of a background process's log file.

        Returns an empty string for unknown ids, missing logs, or unreadable
        files — the renderer treats all of these as "no output yet".

        As a best-effort write-side guard against unbounded disk growth from a
        chatty server, if the file exceeds ``_LOG_DISK_CAP_BYTES`` it is trimmed
        in place to its tail. The trim rewrites the same inode (seek+truncate,
        not replace) so the shell's append redirect keeps the right fd. The pane
        polls every second, so disk usage stays bounded near the cap. Errors are
        swallowed — trimming must never break a read.
        """
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
        """Read the last ``lines`` lines of a file (capped to the last
        ``_LOG_TAIL_MAX_BYTES`` of bytes). Format-agnostic: returns raw decoded
        text; callers needing structured formatting (e.g. JSONL transcripts)
        post-process the result. Returns '' on missing/unreadable files.
        """
        try:
            data = path.read_bytes()[-_LOG_TAIL_MAX_BYTES:]
        except (FileNotFoundError, OSError):
            return ""
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def read_async_tail(self, task_id: str, *, lines: int = 50) -> str:
        """Recent output from an async subagent (``asub-N``).

        Isolated subagents stream their stdout to ``log_path`` — tail that file
        the same way a PROCESS log is tailed, so live progress is visible while
        the subprocess runs. In-process subagents have no file; return the tail
        of their ``response_so_far`` (kept current by the collector). Returns ''
        for unknown ids, missing logs, or agents that have produced nothing yet.
        """
        rec = self._async_agents.get(task_id)
        if rec is None:
            return ""
        if rec.log_path is not None:
            return self._tail_bytes(rec.log_path, lines)
        text = rec.response_so_far or rec.response
        if not text:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def read_agent_log_tail(self, task_id: str, *, lines: int = 50) -> str:
        """Last ``lines`` messages from an in-process workflow agent's transcript.

        Resolves a ``wf-N/live-la-M`` task id back to the agent's messages.jsonl
        via the workflow runner and renders recent messages as readable
        'role: content' lines (not raw JSON). Returns '' for isolated (worktree)
        agents, which have no in-process log; unknown ids; or missing logs — the
        renderer treats all of these as "no output yet".
        """
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
        """Split ``wf-1/live-la-3`` into ``('wf-1', 'la-3')``.

        Returns ``(None, None)`` when the id is not a hierarchical agent id (no
        ``/``, or the suffix is not a ``live-`` child), so callers can treat the
        whole non-agent-id space uniformly.
        """
        if "/" not in task_id:
            return None, None
        run_id, _, agent_suffix = task_id.partition("/")
        prefix = "live-"
        if not agent_suffix.startswith(prefix):
            return None, None
        return run_id, agent_suffix[len(prefix) :]

    @staticmethod
    def _trim_log_in_place(path: Path) -> None:
        """Rewrite a log file to keep only its tail, preserving the inode so the
        writing shell's append fd stays valid. Best-effort.
        """
        try:
            keep = path.read_bytes()[-_LOG_DISK_KEEP_BYTES:]
            with path.open("r+b") as fh:
                fh.seek(0)
                fh.write(keep)
                fh.truncate()
        except OSError:
            pass

    # --- aggregation ------------------------------------------------------

    def list_tasks(self, *, category: TaskCategory | None = None) -> list[TaskEntry]:
        """Build the unified task list across all attached sources.

        Order: processes (running first), then workflows, in-flight agents,
        teams, loops. Renderers re-sort/filter as needed.
        """
        entries: list[TaskEntry] = []
        now = time.monotonic()

        if category in {None, TaskCategory.PROCESS}:
            # Running first, then by id, so live servers sort to the top.
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
        for m in members:
            status = _team_status(getattr(m, "status", "") or "")
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

    # --- routing ----------------------------------------------------------

    async def stop(self, task_id: str) -> bool:
        """Route a cancellation by task-id prefix. Returns False if the target
        is missing or already finalized; True if a stop action was taken.

        Id grammar:
          proc-N            -> terminate owned process (+ group)
          asub-N            -> cancel owned async subagent
          wf-N              -> WorkflowRunner.stop(run_id)
          wf-N/live-AGENT   -> WorkflowRunner.cancel_agent(run_id, agent_id)
          team:NAME         -> TeamManager.stop_teammate(name)
          loop-LOOPID       -> LoopManager.cancel(loop_id)
        """
        if task_id.startswith(("proc-", "asub-")):
            return await self._stop_owned(task_id)
        if task_id.startswith("team:"):
            return await self._stop_team(task_id.removeprefix("team:"))
        if task_id.startswith("loop-"):
            return await self._stop_loop(task_id.removeprefix("loop-"))
        if "/" in task_id and task_id.startswith("wf-"):
            run_id, _, agent_suffix = task_id.partition("/")
            # agent_suffix is "live-<agent_id>"
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
        """Cancel a registry-owned background task by id.

        Routes ``proc-N`` to process termination and ``asub-N`` to async-agent
        cancellation so the public ``stop()`` does not have to spell out each
        owned prefix.
        """
        if task_id.startswith("proc-"):
            return await self._stop_process(task_id)
        return await self._stop_async_agent(task_id)

    async def pause(self, task_id: str) -> bool:
        """Pause/resume toggle for workflow runs only. Returns False otherwise.

        Unlike stop, pause is not a universal verb — only workflows model it
        (in-flight agents finish, new agents block). The pane only offers the
        key for rows with can_pause=True.
        """
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
        # Explicit stop does not queue a completion — the parent asked for it,
        # no surprise event. The Tasks pane sees status='stopped' via list().
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

    # --- lifecycle --------------------------------------------------------

    async def shutdown(self) -> None:
        """Terminate every still-running owned process. Called from the app
        exit path so backgrounded servers don't orphan to init when vibe exits.

        Aggregated categories (workflows/teams/loops) are shut down by their
        own owners (WorkflowRunner.stop_all, TeamManager.stop_all, etc.) — this
        method only reaps what the registry owns.
        """
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
    """Normalize TeamMember.status into the unified status vocabulary.

    TeamManager writes statuses as: 'running', 'running:pid=123', 'completed',
    'failed:<err>', 'stopped', 'error:<err>'. The head token before any ':'
    is the state; 'error' maps to 'failed' (an error is a failure outcome).
    Unknown heads default to 'running' (safer than hiding an active teammate).
    """
    if not raw:
        return "running"
    head = raw.split(":", 1)[0]
    if head == "error":
        return "failed"
    if head in {"running", "completed", "failed", "stopped"}:
        return head
    return "running"
