from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
import functools
import hashlib
import inspect
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, cast

import orjson
from pydantic import BaseModel, ConfigDict

from vibe.core.llm.exceptions import BackendError
from vibe.core.logger import logger
from vibe.core.types import AssistantEvent
from vibe.core.workflows._port import AgentLoopFactory
from vibe.core.workflows.budget import (
    Budget,
    BudgetExhausted,
    ReadOnlyBudget,
    Reservation,
)
from vibe.core.workflows.contract import (
    ContractFailure,
    ContractReport,
    ContractSpec,
    verify_contract,
)
from vibe.core.workflows.models import (
    AgentResult,
    BudgetSnapshot,
    CachedAgentResult,
    PhaseReport,
    SchemaValidationFailure,
    WorkflowResult,
    WorkflowRun,
    WorkflowRunSnapshot,
    WorkflowStatus,
)
from vibe.core.workflows.schema import (
    SchemaValidationError,
    build_prompt_fallback,
    build_response_format,
    strip_unknown_properties,
    validate_against_schema,
)
from vibe.core.workflows.security import build_namespace, validate_script

if TYPE_CHECKING:
    from vibe.core.agent_loop import AgentLoop
    from vibe.core.config import VibeConfig
    from vibe.core.tools.base import InvokeContext


class _AwaitableResult:
    __slots__ = ("_coro",)

    def __init__(self, coro: Awaitable[list[Any]]) -> None:
        self._coro = coro

    def __await__(self) -> Any:
        return self._coro.__await__()

    def __iter__(self) -> Any:
        raise TypeError(
            "Cannot unpack result of parallel()/pipeline() — did you forget 'await'? "
            "Use: results = await parallel(...)"
        )

    def __getitem__(self, _idx: Any) -> Any:
        raise TypeError(
            "Cannot index result of parallel()/pipeline() — did you forget 'await'? "
            "Use: results = await parallel(...)"
        )


T = TypeVar("T")
I = TypeVar("I")

DEFAULT_MAX_CONCURRENT = 32
DEFAULT_MAX_AGENTS = 1000
DEFAULT_BUDGET_TOTAL = None
DEFAULT_SCHEMA_RETRIES = 2


_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    for match in _JSON_FENCE_RE.finditer(s):
        candidate = match.group(1).strip()
        if candidate.startswith(("{", "[")):
            return candidate
    balanced = _first_json_span(s)
    if balanced is not None:
        return balanced
    return s


def _first_json_span(s: str) -> str | None:
    start = -1
    opener = ""
    closer = ""
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opener = ch
            closer = "}" if ch == "{" else "]"
            break
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]


class _AwaitableNoop:
    __slots__ = ("_fn", "_args", "_kwargs")

    def __init__(self, fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def __await__(self) -> Any:
        self._fn(*self._args, **self._kwargs)
        return iter(())

    def __bool__(self) -> bool:
        return False


def _awaitable(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _AwaitableNoop:
        return _AwaitableNoop(fn, args, kwargs)

    return wrapper


def _eager_awaitable(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _AwaitableNoop:
        fn(*args, **kwargs)  # execute NOW so binding takes effect
        return _AwaitableNoop(lambda *a, **k: None, (), {})

    return wrapper


# Pure functions so workflow authors don't reinvent flatten/dedup/merge in
# every script. All sandbox-safe: no imports, no dunders, no I/O.


def _flatten(items: Any) -> list[Any]:
    out: list[Any] = []
    for sub in items:
        if isinstance(sub, (str, bytes, dict)):
            out.append(sub)
            continue
        try:
            iterator = iter(sub)
        except TypeError:
            out.append(sub)
            continue
        out.extend(iterator)
    return out


def _dedup_by(items: Any, key: Callable[[Any], Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        try:
            k = key(item)
            hash(k)
        except Exception:
            k = id(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _merge_by(
    items: Any, key: Callable[[Any], Any], merge: Callable[[Any, Any], Any]
) -> list[Any]:
    groups: dict[Any, Any] = {}
    order: list[Any] = []
    for item in items:
        try:
            k = key(item)
            hash(k)
        except Exception:
            k = id(item)
        if k not in groups:
            groups[k] = item
            order.append(k)
        else:
            groups[k] = merge(groups[k], item)
    return [groups[k] for k in order]


def _coerce_json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        return orjson.loads(orjson.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _loop_log_path(loop: Any) -> Path | None:
    sl = getattr(loop, "session_logger", None)
    if sl is None or not getattr(sl, "enabled", False):
        return None
    return sl.messages_filepath


DEFAULT_ISOLATED_MAX_TURNS = 300

IsolatedExecutor = Callable[
    [str, str, str | None, int], Awaitable["str | tuple[str, dict[str, int] | None]"]
]


class AgentCapExceeded(Exception):
    pass


class WorkflowError(Exception):
    pass


class _WorkerSpawnArgs(BaseModel):
    # Internal-only: built by our own code for the host approval callback, never
    # round-tripped through LLM/provider/session/MCP JSON -> extra="forbid" per
    # the per-family ConfigDict policy (constructed at a single kwargs site).
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    prompt: str
    agent: str
    label: str | None = None


@dataclass
class _LiveAgent:
    agent_id: str
    agent: str
    label: str | None = None
    phase: str | None = None
    model: str | None = None
    status: str = "running"
    tokens_in: int = 0
    tokens_out: int = 0
    started_at: float = field(default_factory=time.monotonic)
    error: str | None = None
    prompt: str = ""
    # The asyncio task running this agent, captured so cancel_agent() can abort
    # a single in-flight agent without stopping the whole run. Set inside the
    # agent coroutine via asyncio.current_task(); None for agents that run on
    # the caller's coroutine (a direct `await agent(...)` with no wrapping task).
    task: asyncio.Task[Any] | None = field(default=None, init=False)
    cancel_requested: bool = field(default=False, init=False)
    # Path to this in-process agent's transcript (messages.jsonl), set after
    # the AgentLoop is created so the background tool can tail it. None for
    # isolated (worktree) agents — their subprocess writes transiently to a
    # dir removed on exit, so there is nothing stable to tail. Refreshed per
    # retry attempt, so it always points at the current attempt's log.
    log_path: Path | None = field(default=None, init=False)
    response_so_far: str = field(default="", init=False)
    # i7: per-agent timeout watchdog task. Cancelled in _retire_live (called at
    # every exit path via _finalize_agent) so it never outlives the agent.
    watchdog: asyncio.Task[None] | None = field(default=None, init=False)

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


_LIVE_RESPONSE_CAP = 8000


class _MessageBoard:
    def __init__(self) -> None:
        self._channels: dict[str, list[Any]] = {}

    def post(self, channel: str, message: Any) -> None:
        self._channels.setdefault(channel, []).append(message)

    def fetch(self, channel: str) -> list[Any]:
        return list(self._channels.get(channel, []))

    def fetch_all(self) -> dict[str, list[Any]]:
        return {k: list(v) for k, v in self._channels.items()}

    def channels(self) -> list[str]:
        return list(self._channels.keys())


def _prompt_hash(
    prompt: str, agent: str, phase: str | None = None, isolation: str | None = None
) -> str:
    # isolation is part of the identity: an isolated (subprocess/worktree) run is
    # not interchangeable with an in-process one for the same prompt/agent/phase.
    # Only fold it in when set, so ordinary keys are unchanged.
    iso = f":iso={isolation}" if isolation else ""
    return hashlib.sha256(f"{agent}:{phase}{iso}:{prompt}".encode()).hexdigest()[:16]


# Must match programmatic.py's sentinel; the isolated subprocess writes one
# stats line to stderr when VIBE_WORKFLOW_EMIT_STATS=1.
_ISOLATED_STATS_SENTINEL = "__VIBE_WORKFLOW_STATS__"


def _parse_stats(stderr_text: str) -> dict[str, int] | None:
    for line in reversed(stderr_text.splitlines()):
        if line.startswith(_ISOLATED_STATS_SENTINEL):
            try:
                data = orjson.loads(line[len(_ISOLATED_STATS_SENTINEL) :])
                return {
                    "prompt_tokens": int(data.get("prompt_tokens", 0)),
                    "completion_tokens": int(data.get("completion_tokens", 0)),
                }
            except (orjson.JSONDecodeError, ValueError, TypeError, AttributeError):
                return None
    return None


@dataclass
class IsolatedResult:
    output: str
    stats: dict[str, int] | None = None
    delivered: bool = False
    worktree_path: str | None = None
    branch: str | None = None
    wt: Any = None


async def run_isolated_agent(
    prompt: str,
    agent: str,
    *,
    label: str | None,
    max_turns: int,
    deliver: bool = False,
    keep_worktree: bool = False,
    model: str | None = None,
    log_path: Path | None = None,
) -> IsolatedResult:
    from vibe.core.worktree.ephemeral import create_ephemeral_worktree

    wt = await asyncio.to_thread(create_ephemeral_worktree, Path.cwd(), label or agent)
    if keep_worktree:
        # Caller (workflow executor) owns verification + reap on SUCCESS; hand
        # the live worktree back so it can verify against the tree before
        # delivering. On failure (incl. cancel) we still own cleanup here, so
        # wrap the spawn and reap on any exit path the caller does not cover.
        try:
            return await _spawn_isolated(
                wt,
                prompt,
                agent,
                max_turns,
                deliver=False,
                stamp_wt=wt,
                model=model,
                log_path=log_path,
            )
        except BaseException:
            _reap_on_failure(wt)
            raise
    try:
        result = await _spawn_isolated(
            wt,
            prompt,
            agent,
            max_turns,
            deliver=deliver,
            model=model,
            log_path=log_path,
        )
        try:
            await asyncio.to_thread(
                _maybe_reap_isolated_worktree, wt, result.delivered, result
            )
        except (OSError, RuntimeError) as e:
            logger.warning("isolated worktree cleanup failed: %s", e)
        return result
    except BaseException:
        _reap_on_failure(wt)
        raise


def _reap_on_failure(wt: Any) -> None:
    from vibe.core.worktree.ephemeral import remove_ephemeral_worktree

    try:
        remove_ephemeral_worktree(wt, keep_if_changed=True)
    except (OSError, RuntimeError) as e:
        logger.warning("isolated worktree cleanup failed: %s", e)


async def _spawn_isolated(
    wt: Any,
    prompt: str,
    agent: str,
    max_turns: int,
    *,
    deliver: bool,
    stamp_wt: Any = None,
    model: str | None = None,
    log_path: Path | None = None,
) -> IsolatedResult:
    import os
    import shlex
    import signal
    import sys

    from vibe.core.worktree.ephemeral import deliver_ephemeral_worktree

    base = os.environ.get("VIBE_ISOLATED_EXECUTOR_CMD")
    prefix = shlex.split(base) if base else [sys.executable, "-m", "vibe"]
    cmd = [
        *prefix,
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
    if model:
        cmd += ["--model", model]
    env = os.environ.copy()
    env["VIBE_WORKFLOW_EMIT_STATS"] = "1"
    # Child wires an auto-yes approval callback (so write/edit/bash run instead
    # of SKIPping headless — see programmatic._isolated_auto_approve) and
    # confines its file tools to this worktree (enforce_isolated_confine).
    env["VIBE_ISOLATED_AUTO_APPROVE"] = "1"
    env["VIBE_ISOLATED_WORKTREE_ROOT"] = str(wt.path)
    stdout_target, log_fh = _open_isolated_log(log_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(wt.path),
            # The child is a trusted `vibe` instance and needs the parent's
            # credentials (e.g. the provider API key) to run; pass the env
            # explicitly (same as teams). Isolation bounds files, not env/secrets.
            env=env,
            stdout=stdout_target,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout_pipe, stderr = await proc.communicate()
        except asyncio.CancelledError:
            # Reap the whole group AND wait before the caller removes the
            # worktree — otherwise `git worktree remove` races a process that
            # still owns it.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3.0)
                except TimeoutError:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    await proc.wait()
            except (ProcessLookupError, PermissionError):
                pass
            raise
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass
    stderr_text = (stderr or b"").decode("utf-8", "replace")
    if proc.returncode != 0:
        raise WorkflowError(
            f"isolated agent subprocess failed (rc={proc.returncode}): "
            f"{stderr_text[:300]}"
        )
    if log_path is not None and log_fh is not None:
        try:
            output = log_path.read_bytes().decode("utf-8", "replace")
        except OSError:
            output = (stdout_pipe or b"").decode("utf-8", "replace")
    else:
        output = (stdout_pipe or b"").decode("utf-8", "replace")
    delivered = deliver and await asyncio.to_thread(deliver_ephemeral_worktree, wt)
    return IsolatedResult(
        output=output, stats=_parse_stats(stderr_text), delivered=delivered, wt=stamp_wt
    )


def _open_isolated_log(log_path: Path | None) -> tuple[Any, Any]:
    if log_path is None:
        return asyncio.subprocess.PIPE, None
    try:
        fh = log_path.open("wb")
    except OSError as exc:
        logger.warning("isolated agent log open failed, falling back to pipe: %s", exc)
        return asyncio.subprocess.PIPE, None
    return fh, fh


def _maybe_reap_isolated_worktree(
    wt: Any, delivered: bool, result: IsolatedResult
) -> None:
    from vibe.core.worktree.ephemeral import remove_ephemeral_worktree

    if delivered:
        remove_ephemeral_worktree(wt, keep_if_changed=False)
        return
    removed = remove_ephemeral_worktree(wt, keep_if_changed=True)
    if not removed:
        result.branch = wt.branch


@dataclass(frozen=True)
class _FinalizeArgs:
    prompt: str
    response: str | dict[str, Any]
    label: str | None
    phase: str | None
    tokens_in: int
    tokens_out: int
    reservation: Reservation
    completed: bool
    error: str | None
    cache_key: str
    agent: str
    model: str | None = None
    live: _LiveAgent | None = None
    schema_errors: list[str] | None = None


@dataclass
class WorkflowRuntime:
    parent_context: InvokeContext | None = None
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    max_agents: int = DEFAULT_MAX_AGENTS
    budget_total: int | None = DEFAULT_BUDGET_TOTAL
    schema_retries: int = DEFAULT_SCHEMA_RETRIES
    # When True, an agent that exhausts its schema-retry budget raises
    # SchemaValidationError (the legacy hard-fail behavior). When False (default),
    # the agent returns a SchemaValidationFailure carrying the raw response so the
    # workflow script never silently loses output to None via parallel._safe.
    strict_schema: bool = False
    agent_timeout_s: float | None = None
    agent_budget_ceiling: int | None = None
    agent_loop_factory: AgentLoopFactory | None = None
    workflow_source_resolver: Callable[[str], str | None] | None = None
    isolated_executor: IsolatedExecutor | None = None
    _semaphore: asyncio.Semaphore = field(init=False)
    _budget: Budget = field(init=False)
    _agent_count: int = field(default=0, init=False)
    _nesting_depth: int = field(default=0, init=False)
    _phases: dict[str, PhaseReport] = field(default_factory=dict, init=False)
    _phase_order: list[str] = field(default_factory=list, init=False)
    # Implicit phase binding (i5): set by phase(name), inherited by subsequent
    # agent() calls that don't pass an explicit phase= kwarg. None means no
    # ambient phase (agents land in "default"). Explicit phase= always wins.
    _current_phase: str | None = field(default=None, init=False)
    _event_sink: Callable[[str], None] | None = field(default=None, init=False)
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _cache: dict[str, CachedAgentResult] = field(default_factory=dict, init=False)
    _live_agents: dict[str, _LiveAgent] = field(default_factory=dict, init=False)
    _next_live_id: int = field(default=0, init=False)
    _board: _MessageBoard = field(default_factory=_MessageBoard, init=False)
    # Pause gate: an asyncio.Event that is set while running and cleared while
    # paused. spawn_agent awaits it after acquiring its semaphore slot, so a
    # pause lets in-flight agents finish but blocks new ones from starting.
    _run_gate: asyncio.Event = field(init=False)
    _paused: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._budget = Budget(total=self.budget_total)
        self._run_gate = asyncio.Event()
        self._run_gate.set()

    def budget_snapshot(self) -> BudgetSnapshot:
        return self._budget.snapshot()

    def pause(self) -> None:
        self._paused = True
        self._run_gate.clear()
        self._log("paused")

    def unpause(self) -> None:
        self._paused = False
        self._run_gate.set()
        self._log("resumed")

    @property
    def is_paused(self) -> bool:
        return self._paused

    def cancel_agent(self, agent_id: str) -> bool:
        live = self._live_agents.get(agent_id)
        if live is None or live.task is None:
            return False
        if live.task.done():
            return False
        live.cancel_requested = True
        live.task.cancel()
        self._log(f"cancelled agent {agent_id}")
        return True

    def set_event_sink(self, sink: Callable[[str], None]) -> None:
        self._event_sink = sink

    def _log(self, msg: str) -> None:
        logger.info("workflow: %s", msg)
        if self._event_sink:
            self._event_sink(msg)

    def _set_phase(self, name: str | None) -> None:
        if name is None:
            self._current_phase = None
            self._log("phase: (reset)")
            return
        if name not in self._phases:
            self._phases[name] = PhaseReport(name=name)
            self._phase_order.append(name)
        self._current_phase = name
        self._log(f"phase: {name}")

    def _record_agent_result(self, result: AgentResult) -> None:
        phase_name = result.phase or "default"
        if phase_name not in self._phases:
            self._phases[phase_name] = PhaseReport(name=phase_name)
            self._phase_order.append(phase_name)
        self._phases[phase_name].agent_results.append(result)

    def _register_live(
        self,
        agent: str,
        model: str | None,
        label: str | None,
        phase: str | None,
        prompt: str = "",
    ) -> _LiveAgent:
        live_id = f"la-{self._next_live_id}"
        self._next_live_id += 1
        live = _LiveAgent(
            agent_id=live_id,
            agent=agent,
            model=model,
            label=label,
            phase=phase,
            prompt=prompt,
        )
        self._live_agents[live_id] = live
        return live

    def _retire_live(
        self, live: _LiveAgent, *, status: str, error: str | None = None
    ) -> None:
        live.status = status
        live.error = error
        # i7: cancel the timeout watchdog so it doesn't fire after the agent
        # has already finished. Every exit path runs through _finalize_agent
        # -> _retire_live, so this covers all cases.
        if live.watchdog is not None and not live.watchdog.done():
            live.watchdog.cancel()
        self._live_agents.pop(live.agent_id, None)

    def _validate_workflow_profile(self, agent: str, isolation: str | None) -> None:
        ctx = self.parent_context
        if not ctx or not ctx.agent_manager:
            return
        try:
            profile = ctx.agent_manager.get_agent(agent)
        except ValueError:
            return
        from vibe.core.agents.models import AgentType, profile_requires_isolation

        agent_type = getattr(profile, "agent_type", AgentType.SUBAGENT)
        if agent_type != AgentType.SUBAGENT:
            raise WorkflowError(
                f"Agent '{agent}' is a {agent_type.value} agent. "
                f"Only subagents can be used in workflows."
            )
        if isolation != "worktree" and profile_requires_isolation(profile):
            raise WorkflowError(
                f"Agent '{agent}' can write or run unrestricted shell; "
                f"in a workflow it must run with isolation='worktree'."
            )

    async def _judge_isolated_spawn(
        self, prompt: str, agent: str, label: str | None
    ) -> None:
        ctx = self.parent_context
        if not ctx or not ctx.safety_judge_factory:
            return
        try:
            judge = ctx.safety_judge_factory()
        except Exception:
            logger.debug(
                "safety judge factory raised; skipping worker pre-judge", exc_info=True
            )
            return
        if judge is None:
            return

        verdict = await judge.judge(
            "launch_workflow",  # select the workflow-aware system prompt
            prompt,
            [f"isolated '{agent}' worker spawn" + (f": {label}" if label else "")],
        )
        if verdict.safe:
            self._log(
                f"safety judge approved isolated '{agent}' worker"
                + (f" ({label})" if label else "")
                + f": {verdict.reason}"
            )
            return

        self._log(
            f"safety judge deferred isolated '{agent}' worker"
            + (f" ({label})" if label else "")
            + f" to user: {verdict.reason}"
        )
        if ctx.approval_callback is None:
            raise WorkflowError(
                f"Safety judge denied isolated worker spawn and no approval "
                f"callback is available: {verdict.reason}"
            )
        from vibe.core.types import ApprovalResponse

        response, _feedback, _modified = await ctx.approval_callback(
            f"workflow_worker:{agent}",
            _WorkerSpawnArgs(prompt=prompt, agent=agent, label=label),
            f"worker-spawn-{agent}-{label or 'anon'}",
            None,
            verdict.reason,
        )
        if response != ApprovalResponse.YES:
            raise WorkflowError(
                f"Isolated worker spawn denied by user (judge: {verdict.reason})"
            )

    @staticmethod
    def _resolve_contract(
        contract: dict | None, isolation: str | None
    ) -> ContractSpec | None:
        if contract is None:
            return None
        if isolation != "worktree":
            raise WorkflowError(
                "contract= requires isolation='worktree' (it validates the "
                "files an isolated code agent wrote)"
            )
        return ContractSpec.model_validate(contract)

    async def spawn_agent(
        self,
        prompt: str,
        *,
        agent: str = "explore",
        model: str | None = None,
        label: str | None = None,
        phase: str | None = None,
        schema: dict | None = None,
        budget_estimate: int | None = None,
        isolation: str | None = None,
        strip_unknown: bool = True,
        contract: dict | None = None,
    ) -> str | dict[str, Any] | SchemaValidationFailure | ContractFailure:
        contract_spec = self._resolve_contract(contract, isolation)
        effective_phase = phase if phase is not None else self._current_phase
        cache_key = _prompt_hash(prompt, agent, effective_phase, isolation)
        if cached := self._cache.get(cache_key):
            self._log(f"cache hit: {label or agent}")
            self._record_cached_result(cached)
            return cached.response
        reservation = await self._prepare_spawn(
            prompt, agent, isolation, label, budget_estimate
        )
        async with self._semaphore:
            await self._run_gate.wait()
            if isolation == "worktree":
                return await self._run_isolated_agent(
                    prompt=prompt,
                    agent=agent,
                    model=model,
                    label=label,
                    phase=effective_phase,
                    schema=schema,
                    reservation=reservation,
                    cache_key=cache_key,
                    strip_unknown=strip_unknown,
                    contract=contract_spec,
                )
            if isolation is not None:
                raise WorkflowError(
                    f"Unknown isolation mode {isolation!r} (only 'worktree')"
                )
            return await self._run_agent(
                prompt=prompt,
                agent=agent,
                model=model,
                label=label,
                phase=effective_phase,
                schema=schema,
                reservation=reservation,
                cache_key=cache_key,
                strip_unknown=strip_unknown,
            )

    async def _prepare_spawn(
        self,
        prompt: str,
        agent: str,
        isolation: str | None,
        label: str | None,
        budget_estimate: int | None,
    ) -> Reservation:
        self._validate_workflow_profile(agent, isolation)
        # Isolated workers run auto-approved in their subprocess and can't prompt
        # the host per-tool, so judge each worker's prompt at spawn. In-process
        # subagents consult the judge per-tool, so they're not pre-judged here.
        if isolation == "worktree":
            await self._judge_isolated_spawn(prompt, agent, label)
        if self._agent_count >= self.max_agents:
            raise AgentCapExceeded(
                f"Agent cap reached: {self._agent_count}/{self.max_agents}"
            )
        reservation = self._budget.reserve(budget_estimate)
        self._agent_count += 1
        return reservation

    @staticmethod
    def _is_structured_output_rejection(
        error: BaseException, *, has_response_format: bool, can_retry: bool
    ) -> TypeGuard[BackendError]:
        return (
            has_response_format
            and can_retry
            and isinstance(error, BackendError)
            and error.is_structured_output_rejected
        )

    async def _run_agent(
        self,
        *,
        prompt: str,
        agent: str,
        model: str | None,
        label: str | None,
        phase: str | None,
        schema: dict | None,
        reservation: Reservation,
        cache_key: str,
        strip_unknown: bool = True,
    ) -> str | dict[str, Any] | SchemaValidationFailure:
        response_format = build_response_format(schema) if schema is not None else None
        effective_prompt = prompt
        if schema is not None:
            effective_prompt = prompt + build_prompt_fallback(schema)

        live = self._register_live(
            agent=agent, model=model, label=label, phase=phase, prompt=prompt
        )
        # Capture the running task so cancel_agent() can abort this single
        # agent. None when spawn_agent is awaited directly (no wrapping task);
        # in that case the agent can't be cancelled individually, only via the
        # whole-run stop.
        live.task = asyncio.current_task()

        # i7: per-agent timeout watchdog (opt-in). A separate asyncio task that
        # sleeps for agent_timeout_s then cancels the agent. Catches both slowly-
        # spending agents AND fully stuck agents (no events emitted) — the inline
        # budget-ceiling check only fires on event boundaries, so the watchdog
        # is the backstop for pathological hangs. Cancelled in _retire_live.
        if self.agent_timeout_s is not None:
            timeout_s = self.agent_timeout_s

            async def _timeout_watchdog() -> None:
                try:
                    await asyncio.sleep(timeout_s)
                    if (
                        not live.cancel_requested
                        and live.task is not None
                        and not live.task.done()
                    ):
                        live.cancel_requested = True
                        live.error = f"agent timed out after {timeout_s}s"
                        live.task.cancel()
                except asyncio.CancelledError:
                    pass

            live.watchdog = asyncio.create_task(_timeout_watchdog())

        last_errors: list[str] = []
        accumulated: list[str] = []
        tokens_in = 0
        tokens_out = 0
        completed = True
        error_msg: str | None = None

        base_config: VibeConfig | None = None
        if self.agent_loop_factory is None:
            # Load config once per spawn (off the event loop) and reuse across
            # schema-retry attempts — session_logging/model are constant for the
            # spawn, so reloading per attempt (the previous behavior) was pure
            # blocking waste on the shared loop. A raise here escapes before the
            # per-attempt try block runs _finalize_agent, so release the
            # reservation (guarded by reservation.reconciled) to avoid leaking it
            # and permanently understating Budget.remaining(), then propagate.
            try:
                base_config = await asyncio.to_thread(
                    self._resolve_agent_config, agent=agent, model=model
                )
            except Exception as e:
                if not reservation.reconciled:
                    self._finalize_agent(
                        _FinalizeArgs(
                            prompt=prompt,
                            response="",
                            label=label,
                            phase=phase,
                            tokens_in=0,
                            tokens_out=0,
                            reservation=reservation,
                            completed=False,
                            error=f"agent config load failed: {e}",
                            cache_key=cache_key,
                            agent=agent,
                            model=model,
                            live=live,
                        )
                    )
                raise
        for attempt in range(self.schema_retries + 1):
            accumulated = []
            try:
                loop = self._create_loop(
                    effective_prompt, agent=agent, model=model, base_config=base_config
                )
                # Point the background tool at this attempt's transcript so it can
                # be tailed; refreshed each retry, so it follows the live log.
                live.log_path = _loop_log_path(loop)
                async with aclosing(
                    loop.act(effective_prompt, response_format=response_format)
                ) as events:
                    async for event in events:
                        content = self._extract_content(event)
                        if content:
                            accumulated.append(content)
                            if len(live.response_so_far) < _LIVE_RESPONSE_CAP:
                                live.response_so_far += content
                        # i7: per-agent budget ceiling (opt-in). Check mid-stream
                        # so a runaway agent is stopped before exhausting the run
                        # budget, not just at reserve() time. Set completed=False
                        # and break — no task.cancel() (cancelling the current
                        # task is racy if there's no real await after the break).
                        if self.agent_budget_ceiling is not None:
                            spent = (
                                tokens_in
                                + getattr(loop.stats, "session_prompt_tokens", 0)
                                + tokens_out
                                + getattr(loop.stats, "session_completion_tokens", 0)
                            )
                            if spent > self.agent_budget_ceiling:
                                completed = False
                                error_msg = (
                                    f"agent exceeded per-agent budget ceiling "
                                    f"({spent} > {self.agent_budget_ceiling} tokens)"
                                )
                                break
                        # Live token accounting: the real AgentLoop updates
                        # stats at turn boundaries (between emitted events), so
                        # polling after each event reflects spend as each turn
                        # completes rather than only at finalize. tokens_in/out
                        # hold prior attempts' totals; loop.stats is this
                        # attempt's running total (fresh loop per attempt).
                        live.tokens_in = tokens_in + getattr(
                            loop.stats, "session_prompt_tokens", 0
                        )
                        live.tokens_out = tokens_out + getattr(
                            loop.stats, "session_completion_tokens", 0
                        )
            except asyncio.CancelledError:
                # Whole-run stop re-raises; a targeted cancel_agent() or the i7
                # timeout watchdog sets cancel_requested and we record this agent
                # as failed instead. Use live.error if the watchdog pre-set a
                # descriptive message (e.g. "agent timed out after Ns").
                if not live.cancel_requested:
                    raise
                completed = False
                error_msg = live.error or "cancelled by user"
                break
            except Exception as e:
                # Graceful structured-output degradation: if the provider rejected
                # the response_format payload itself (e.g. Responses API seeing the
                # Chat Completions schema shape), drop response_format and retry on
                # the next attempt using the prompt-level JSON fallback already
                # appended to effective_prompt. Costs ~0 tokens (the 400 fires
                # before streaming) and turns a hard run failure into a transparent
                # fallback. Only when a retry slot remains.
                if self._is_structured_output_rejection(
                    e,
                    has_response_format=response_format is not None,
                    can_retry=attempt < self.schema_retries,
                ):
                    logger.warning(
                        "Provider rejected structured-output response_format "
                        "(status=%s, detail=%s); retrying without it. "
                        "Prompt-level JSON schema fallback remains active.",
                        e.status,
                        e.parsed_error,
                    )
                    response_format = None
                    completed = True
                    error_msg = None
                    continue
                completed = False
                error_msg = str(e)
                break

            # Accumulate across attempts: a fresh loop is created per retry, so
            # its stats report only that attempt. Overwriting dropped the tokens
            # spent on failed schema attempts from the budget.
            tokens_in += getattr(loop.stats, "session_prompt_tokens", 0)
            tokens_out += getattr(loop.stats, "session_completion_tokens", 0)
            live.tokens_in = tokens_in
            live.tokens_out = tokens_out

            response_text = "".join(accumulated)

            if schema is None:
                self._finalize_agent(
                    _FinalizeArgs(
                        prompt=prompt,
                        response=response_text,
                        label=label,
                        phase=phase,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        reservation=reservation,
                        completed=completed,
                        error=error_msg,
                        cache_key=cache_key,
                        agent=agent,
                        model=model,
                        live=live,
                    )
                )
                return response_text

            try:
                parsed = orjson.loads(_strip_code_fences(response_text))
            except orjson.JSONDecodeError as e:
                last_errors = [f"JSON parse error: {e}"]
            else:
                if strip_unknown:
                    parsed = strip_unknown_properties(parsed, schema)
                errors = validate_against_schema(parsed, schema)
                if not errors:
                    self._finalize_agent(
                        _FinalizeArgs(
                            prompt=prompt,
                            response=parsed,
                            label=label,
                            phase=phase,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            reservation=reservation,
                            completed=completed,
                            error=error_msg,
                            cache_key=cache_key,
                            agent=agent,
                            model=model,
                            live=live,
                        )
                    )
                    return parsed
                last_errors = [str(e) for e in errors]

            if attempt < self.schema_retries:
                error_str = "\n".join(f"  - {e}" for e in last_errors)
                effective_prompt = (
                    f"{prompt}\n\n"
                    f"Your previous response had these validation errors:\n{error_str}\n"
                    f"Please respond again with a valid JSON object matching the schema."
                )
                if schema is not None:
                    effective_prompt += build_prompt_fallback(schema)

        if schema is None:
            # The schemaless success path returns inside the loop; reaching here
            # means act() raised. Surface the real error instead of a misleading
            # SchemaValidationError.
            self._finalize_agent(
                _FinalizeArgs(
                    prompt=prompt,
                    response="".join(accumulated),
                    label=label,
                    phase=phase,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    reservation=reservation,
                    completed=False,
                    error=error_msg,
                    cache_key=cache_key,
                    agent=agent,
                    model=model,
                    live=live,
                )
            )
            # A targeted cancel is an expected outcome, not a failure to
            # surface: the agent is already recorded as failed above, so return
            # the partial output instead of raising. Other failures still raise
            # so parallel._safe / direct awaiters see them.
            if live.cancel_requested:
                return "".join(accumulated)
            raise WorkflowError(error_msg or "Agent failed without producing output")

        if error_msg is not None:
            # act() raised before schema validation. Surface the real error
            # instead of a misleading SchemaValidationError (mirrors the
            # schemaless path above): the agent crashed, it didn't produce
            # invalid output.
            self._finalize_agent(
                _FinalizeArgs(
                    prompt=prompt,
                    response="".join(accumulated),
                    label=label,
                    phase=phase,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    reservation=reservation,
                    completed=False,
                    error=error_msg,
                    cache_key=cache_key,
                    agent=agent,
                    model=model,
                    live=live,
                )
            )
            if live.cancel_requested:
                return "".join(accumulated)
            raise WorkflowError(error_msg)

        # Genuine schema exhaustion: act() ran clean but never produced valid
        # output. Record the raw text so it survives, then either raise (strict)
        # or return a structured failure so the workflow script can recover it
        # instead of losing it to None via parallel._safe.
        schema_error_summary = (
            f"Schema validation failed after {self.schema_retries + 1} attempts"
        )
        self._finalize_agent(
            _FinalizeArgs(
                prompt=prompt,
                response="".join(accumulated),
                label=label,
                phase=phase,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                reservation=reservation,
                completed=False,
                error=schema_error_summary,
                cache_key=cache_key,
                agent=agent,
                model=model,
                live=live,
                schema_errors=list(last_errors),
            )
        )
        if self.strict_schema:
            raise SchemaValidationError(
                f"Response did not match schema after {self.schema_retries + 1} attempts. "
                f"Last errors: {'; '.join(last_errors)}"
            )
        return SchemaValidationFailure(
            raw_response="".join(accumulated),
            error=schema_error_summary,
            schema_errors=list(last_errors),
        )

    def _resolve_agent_config(
        self, *, agent: str, model: str | None = None
    ) -> VibeConfig:
        from vibe.core.config import SessionLoggingConfig, VibeConfig

        ctx = self.parent_context
        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx and ctx.session_dir else "",
            session_prefix=agent,
            enabled=ctx is not None and ctx.session_dir is not None,
        )
        # Inherit the launching session's model when the workflow step didn't
        # pin one, so the per-agent config doesn't re-derive the hardcoded
        # mistral default (which fails without MISTRAL_API_KEY). This was killing
        # workflow fan-out agents whose parent ran on glm/zai/fugu.
        cfg = ctx.agent_manager.config if ctx and ctx.agent_manager else None
        configured_subagent = cfg.subagent_model if cfg else ""
        inherited_model = (
            model
            or configured_subagent
            or (ctx.active_model if ctx else None)
            or (cfg.active_model if cfg else None)
        )
        overrides: dict[str, Any] = {}
        if inherited_model:
            overrides["active_model"] = inherited_model
        return VibeConfig.load(session_logging=session_logging, **overrides)

    def _create_loop(
        self,
        prompt: str,
        *,
        agent: str,
        model: str | None = None,
        base_config: VibeConfig | None = None,
    ) -> AgentLoop:
        if self.agent_loop_factory is not None:
            return self.agent_loop_factory(
                prompt, agent=agent, parent_context=self.parent_context
            )
        return self._create_real_loop(agent=agent, model=model, base_config=base_config)

    def _create_real_loop(
        self,
        *,
        agent: str,
        model: str | None = None,
        base_config: VibeConfig | None = None,
    ) -> AgentLoop:
        from vibe.core.agent_loop import AgentLoop as _AgentLoop

        ctx = self.parent_context

        if ctx and ctx.agent_manager:
            try:
                profile = ctx.agent_manager.get_agent(agent)
            except ValueError as e:
                raise WorkflowError(f"Unknown agent: {agent}") from e
            from vibe.core.agents.models import AgentType

            if profile.agent_type != AgentType.SUBAGENT:
                raise WorkflowError(
                    f"Agent '{agent}' is a {profile.agent_type.value} agent. "
                    f"Only subagents can be used in workflows."
                )

        if base_config is None:
            base_config = self._resolve_agent_config(agent=agent, model=model)
        # Subagents inherit the parent worktree; never call worktree_manager.enter().
        # Workflow stages share one worktree.
        loop = _AgentLoop(
            config=base_config,
            agent_name=agent,
            entrypoint_metadata=ctx.entrypoint_metadata if ctx else None,
            terminal_emulator=ctx.terminal_emulator if ctx else None,
            is_subagent=True,
            defer_heavy_init=True,
            permission_store=ctx.permission_store if ctx else None,
            hook_config_result=ctx.hook_config_result if ctx else None,
            max_turns=DEFAULT_ISOLATED_MAX_TURNS,
        )
        if ctx and ctx.session_id:
            loop.parent_session_id = ctx.session_id
        if ctx and ctx.approval_callback:
            loop.set_approval_callback(ctx.approval_callback)
        # NOTE: in-process subagents are MCP-free by design. Restricted profiles
        # (explore/research/reviewer) filter MCP out via their allowlist anyway;
        # full-tool MCP work runs as the 'worker' profile under
        # isolation='worktree', whose `vibe -p` subprocess discovers MCP itself.
        # (An earlier in-process integrate_mcp() here blocked the event loop and
        # raced the shared registry — removed.)
        return loop

    @staticmethod
    def _extract_content(event: Any) -> str | None:
        # Only the assistant's answer is part of the response. The real
        # AgentLoop.act stream also yields UserMessageEvent (the prompt echo)
        # and ReasoningEvent (chain-of-thought), both of which carry string
        # `content`; accumulating those polluted the response and broke schema
        # parsing. Restrict to AssistantEvent.
        if isinstance(event, AssistantEvent):
            content = event.content
            if content and isinstance(content, str):
                return content
        return None

    def _compute_cost(
        self, tokens_in: int, tokens_out: int, model: str | None
    ) -> float:
        ctx = self.parent_context
        if not ctx or not ctx.agent_manager:
            return 0.0
        config = ctx.agent_manager.config
        target_alias = model or config.active_model
        for m in config.models:
            if m.alias == target_alias:
                return (tokens_in / 1_000_000) * m.input_price + (
                    tokens_out / 1_000_000
                ) * m.output_price
        return 0.0

    def _finalize_agent(self, args: _FinalizeArgs) -> None:
        self._budget.reconcile(args.reservation, args.tokens_in, args.tokens_out)

        cost = self._compute_cost(args.tokens_in, args.tokens_out, args.model)
        schema_errs = args.schema_errors or []
        result = AgentResult(
            label=args.label,
            phase=args.phase,
            agent=args.agent,
            prompt=args.prompt,
            response=args.response,
            tokens_in=args.tokens_in,
            tokens_out=args.tokens_out,
            cost=cost,
            completed=args.completed,
            error=args.error,
            schema_errors=schema_errs,
        )
        self._record_agent_result(result)
        # Retire the live tracker the moment its result is recorded, so an agent
        # is reflected as live XOR finalized, never both (which would double its
        # tokens in the live view + finalized phases).
        if args.live is not None:
            self._retire_live(
                args.live,
                status="completed" if args.completed else "failed",
                error=args.error,
            )

        if args.completed:
            self._cache[args.cache_key] = CachedAgentResult(
                prompt_hash=args.cache_key,
                agent=args.agent,
                label=args.label,
                phase=args.phase,
                response=args.response,
                tokens_in=args.tokens_in,
                tokens_out=args.tokens_out,
                completed=args.completed,
                error=args.error,
                schema_errors=schema_errs,
            )

    def _record_cached_result(self, cached: CachedAgentResult) -> None:
        # A cache hit consumes no tokens in THIS run; the original spend is
        # already reflected in the restored budget. Recording the cached token
        # counts here would double-count them in run.tokens_total relative to
        # budget.spent. Report zero for this run.
        result = AgentResult(
            label=cached.label,
            phase=cached.phase,
            agent=cached.agent,
            prompt=f"[cached] {cached.agent}",
            response=cached.response,
            tokens_in=0,
            tokens_out=0,
            cost=0.0,
            completed=cached.completed,
            error=cached.error,
        )
        self._record_agent_result(result)

    async def _run_isolated_agent(
        self,
        *,
        prompt: str,
        agent: str,
        model: str | None,
        label: str | None,
        phase: str | None,
        schema: dict | None,
        reservation: Reservation,
        cache_key: str,
        strip_unknown: bool = True,
        contract: ContractSpec | None = None,
    ) -> str | dict[str, Any] | SchemaValidationFailure | ContractFailure:
        effective_prompt = prompt + (
            build_prompt_fallback(schema) if schema is not None else ""
        )
        live = self._register_live(
            agent=agent, model=model, label=label, phase=phase, prompt=prompt
        )
        live.task = asyncio.current_task()
        # contract needs the worktree's files, so only the default executor
        # (which owns the worktree lifecycle) supports it.
        if contract is not None and self.isolated_executor is not None:
            raise WorkflowError(
                "contract= is not supported with a custom isolated_executor "
                "(it requires the default worktree executor)"
            )
        (
            output,
            stats,
            contract_report,
            completed,
            error_msg,
        ) = await self._execute_isolated(
            effective_prompt, agent, label, contract, live, model=model
        )
        if completed and contract_report is not None and not contract_report.passed:
            completed = False
            error_msg = contract_report.summary()
        response, schema_errors, completed, error_msg = (
            self._finalize_isolated_response(
                output, schema, strip_unknown, completed, error_msg
            )
        )
        tokens_in, tokens_out = self._charge_isolated_tokens(
            stats, reservation, completed, label, agent
        )
        self._finalize_agent(
            _FinalizeArgs(
                prompt=prompt,
                response=response,
                label=label,
                phase=phase,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                reservation=reservation,
                completed=completed,
                error=error_msg,
                cache_key=cache_key,
                agent=agent,
                model=model,
                live=live,
                schema_errors=schema_errors,
            )
        )
        if not completed:
            failure = self._isolated_failure_value(
                contract_report, schema, schema_errors, error_msg, output
            )
            if failure is not None:
                return failure
            raise WorkflowError(f"isolated agent failed: {error_msg}")
        return response

    async def _execute_isolated(
        self,
        effective_prompt: str,
        agent: str,
        label: str | None,
        contract: ContractSpec | None,
        live: Any,
        *,
        model: str | None = None,
    ) -> tuple[str, dict[str, int] | None, ContractReport | None, bool, str | None]:
        try:
            if self.isolated_executor is not None:
                # Custom test seam: returns output (str) or (output, stats).
                raw = await self.isolated_executor(
                    effective_prompt, agent, label, DEFAULT_ISOLATED_MAX_TURNS
                )
                if isinstance(raw, tuple):
                    output, stats = raw
                else:
                    output, stats = raw, None
                return output, stats, None, True, None
            output, stats, contract_report = await self._default_isolated_executor(
                effective_prompt,
                agent,
                label,
                DEFAULT_ISOLATED_MAX_TURNS,
                contract=contract,
                model=model,
            )
            return output, stats, contract_report, True, None
        except (AgentCapExceeded, BudgetExhausted):
            raise
        except asyncio.CancelledError:
            # Whole-run stop re-raises; a targeted cancel sets cancel_requested so
            # the run continues, recording this agent as failed below.
            if not live.cancel_requested:
                raise
            return "", None, None, False, "cancelled by user"
        except Exception as e:
            return "", None, None, False, str(e)

    def _finalize_isolated_response(
        self,
        output: str,
        schema: dict | None,
        strip_unknown: bool,
        completed: bool,
        error_msg: str | None,
    ) -> tuple[str | dict[str, Any], list[str], bool, str | None]:
        response: str | dict[str, Any] = output
        if not completed or schema is None:
            return response, [], completed, error_msg
        parsed_response, schema_errors = self._parse_isolated_schema(
            output, schema, strip_unknown
        )
        if schema_errors:
            return response, schema_errors, False, "; ".join(schema_errors)
        if parsed_response is not None:
            response = parsed_response
        return response, schema_errors, completed, error_msg

    def _charge_isolated_tokens(
        self,
        stats: dict[str, int] | None,
        reservation: Reservation,
        completed: bool,
        label: str | None,
        agent: str,
    ) -> tuple[int, int]:
        if stats is not None:
            return int(stats.get("prompt_tokens", 0)), int(
                stats.get("completion_tokens", 0)
            )
        # Fall back to the reserved estimate so the budget cap stays enforced
        # (charging 0 would let isolated agents spend nothing against the cap).
        if completed:
            logger.info(
                "workflow: isolated agent %s emitted no token stats; charging "
                "the estimate (%d)",
                label or agent,
                reservation.estimate,
            )
        return 0, reservation.estimate

    def _isolated_failure_value(
        self,
        contract_report: ContractReport | None,
        schema: dict | None,
        schema_errors: list[str],
        error_msg: str | None,
        output: str,
    ) -> SchemaValidationFailure | ContractFailure | None:
        if contract_report is not None and not contract_report.passed:
            return ContractFailure(
                report=contract_report, error=error_msg or "contract failed"
            )
        if schema is None or not schema_errors:
            return None
        if self.strict_schema:
            raise SchemaValidationError(
                error_msg or "isolated agent schema validation failed"
            )
        return SchemaValidationFailure(
            raw_response=output,
            error=error_msg or "isolated agent schema validation failed",
            schema_errors=schema_errors,
        )

    @staticmethod
    def _parse_isolated_schema(
        output: str, schema: dict, strip_unknown: bool
    ) -> tuple[str | dict[str, Any] | None, list[str]]:
        try:
            parsed = orjson.loads(_strip_code_fences(output))
        except orjson.JSONDecodeError as e:
            return None, [f"isolated agent returned invalid JSON: {e}"]
        if strip_unknown:
            parsed = strip_unknown_properties(parsed, schema)
        errors = validate_against_schema(parsed, schema)
        if errors:
            return None, [str(err) for err in errors]
        return parsed, []

    async def _default_isolated_executor(
        self,
        prompt: str,
        agent: str,
        label: str | None,
        max_turns: int,
        *,
        contract: ContractSpec | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, int] | None, ContractReport | None]:
        # Delegate spawn+communicate+stats to the shared run_isolated_agent
        # (same path the task() tool uses), keeping ownership of the worktree
        # (keep_worktree=True) so contract verification can run against the live
        # tree before delivery + reap.
        from vibe.core.worktree.ephemeral import (
            deliver_ephemeral_worktree,
            remove_ephemeral_worktree,
        )

        result = await run_isolated_agent(
            prompt,
            agent,
            label=label,
            max_turns=max_turns,
            deliver=False,
            keep_worktree=True,
            model=model,
        )
        wt = result.wt
        contract_report: ContractReport | None = None
        try:
            if contract is not None and wt is not None:
                contract_report = verify_contract(wt.path, contract)
                if contract_report.passed:
                    contract_report.delivered = await asyncio.to_thread(
                        deliver_ephemeral_worktree, wt
                    )
        finally:
            if wt is not None:
                # Delivered -> work is in the parent, force-remove. Otherwise
                # keep so undelivered (failed contract or ff refused) work
                # stays recoverable via `git merge <branch>`.
                await asyncio.to_thread(
                    remove_ephemeral_worktree,
                    wt,
                    keep_if_changed=not (contract_report and contract_report.delivered),
                )
        return result.output, result.stats, contract_report

    def parallel(
        self, *thunks: Any, max_concurrency: int | None = None
    ) -> _AwaitableResult:
        if len(thunks) == 1 and isinstance(thunks[0], (list, tuple)):
            thunk_list = list(thunks[0])
        else:
            thunk_list = list(thunks)

        if max_concurrency is not None and max_concurrency < 1:
            raise WorkflowError(
                f"parallel(max_concurrency=...) must be >= 1, got {max_concurrency}"
            )
        sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )

        async def _invoke(item: Any) -> Any:
            # Accept BOTH a coroutine/awaitable (`agent(...)`) and a zero-arg
            # thunk (`lambda: agent(...)`). Python coroutines are lazy — they do
            # not run until awaited — so the bare-call form is safe and bounds
            # concurrency identically under the semaphore. (This is why the JS
            # "thunks only" rule does not apply here; requiring it was the #1
            # authoring footgun.) A thunk that returns a non-awaitable is used
            # as-is so trivial map stages still work.
            if inspect.isawaitable(item):
                return await item
            result = item()
            return await result if inspect.isawaitable(result) else result

        async def _safe(item: Any) -> Any:
            # Neither awaitable nor callable is an authoring bug (e.g. a bare
            # value passed where a coroutine/thunk was meant). Fail loud with the
            # fix rather than degrading to None and masking it.
            if not (inspect.isawaitable(item) or callable(item)):
                raise WorkflowError(
                    "parallel() items must be coroutines (`agent(...)`) or "
                    f"zero-arg callables (`lambda: agent(...)`), got "
                    f"{type(item).__name__}."
                )
            try:
                if sem is not None:
                    async with sem:
                        return await _invoke(item)
                else:
                    return await _invoke(item)
            except (AgentCapExceeded, BudgetExhausted):
                # Hard ceilings (runaway-agent / overspend backstops) must fail
                # the run, not silently become a None result that masks the
                # breach. Ordinary failures still degrade to None below.
                raise
            except Exception:
                logger.warning("workflow: parallel() item failed", exc_info=True)
                return None

        async def _run() -> list[Any]:
            return await asyncio.gather(*[_safe(t) for t in thunk_list])

        return _AwaitableResult(_run())

    @staticmethod
    def _call_stage(
        stage: Callable[..., Any], result: Any, item: Any, index: int
    ) -> Any:
        args = (result, item, index)
        try:
            params = list(inspect.signature(stage).parameters.values())
        except (TypeError, ValueError):
            return stage(result)
        if any(p.kind == p.VAR_POSITIONAL for p in params):
            return stage(*args)
        positional = sum(
            1 for p in params if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
        )
        return stage(*args[: max(1, min(3, positional))])

    @staticmethod
    def _stage_accepts_positional(stage: Callable[..., Any]) -> bool:
        try:
            params = list(inspect.signature(stage).parameters.values())
        except (TypeError, ValueError):
            return True  # uninspectable (e.g. a builtin) — assume callable
        return any(
            p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL}
            for p in params
        )

    def pipeline(
        self,
        items: list[I],
        *stages: Callable[..., Awaitable[Any]],
        max_concurrency: int | None = None,
    ) -> _AwaitableResult:
        for stage in stages:
            if not self._stage_accepts_positional(stage):
                raise WorkflowError(
                    f"pipeline stage {getattr(stage, '__name__', stage)!r} must "
                    "accept at least one positional argument (prev[, item, index])"
                )
        if max_concurrency is not None and max_concurrency < 1:
            raise WorkflowError(
                f"pipeline(max_concurrency=...) must be >= 1, got {max_concurrency}"
            )
        sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )
        items_list = list(items)

        async def _run_item(index: int, item: Any) -> Any:
            result: Any = item
            for stage in stages:
                try:
                    result = await self._call_stage(stage, result, item, index)
                except (AgentCapExceeded, BudgetExhausted):
                    # Hard ceilings fail the run (see parallel()); they must not
                    # be swallowed into a silent None item.
                    raise
                except Exception:
                    logger.warning(
                        "workflow: pipeline() stage failed for item %d",
                        index,
                        exc_info=True,
                    )
                    return None
            return result

        async def _guarded_run_item(index: int, item: Any) -> Any:
            if sem is None:
                return await _run_item(index, item)
            async with sem:
                return await _run_item(index, item)

        async def _run() -> list[Any]:
            return await asyncio.gather(*[
                _guarded_run_item(i, it) for i, it in enumerate(items_list)
            ])

        return _AwaitableResult(_run())

    def build_script_namespace(self, args: Any = None) -> dict[str, Any]:
        async def _agent(
            prompt: str,
            *,
            agent: str = "explore",
            model: str | None = None,
            label: str | None = None,
            phase: str | None = None,
            schema: dict | None = None,
            budget_estimate: int | None = None,
            isolation: str | None = None,
            strip_unknown: bool = True,
            contract: dict | None = None,
            **extra: Any,
        ) -> str | dict[str, Any] | SchemaValidationFailure | ContractFailure:
            # Tolerate unknown kwargs so one stray argument degrades a single
            # agent() call instead of crashing the whole workflow at 0 agents.
            # `agentType`/`agent_type` is a common cross-API spelling of `agent`;
            # honor it rather than silently running the default agent. Everything
            # else (e.g. max_concurrency, which belongs on parallel()/pipeline())
            # is warned about and ignored.
            if extra:
                alias = extra.pop("agentType", None) or extra.pop("agent_type", None)
                if alias and agent == "explore":
                    agent = alias
                if extra:
                    ignored = ", ".join(sorted(extra))
                    self._log(f"agent(): ignoring unsupported kwarg(s): {ignored}")
                    logger.warning(
                        "workflow agent() called with unsupported kwarg(s) %s; "
                        "ignoring",
                        ignored,
                    )
            return await self.spawn_agent(
                prompt,
                agent=agent,
                model=model,
                label=label,
                phase=phase,
                schema=schema,
                budget_estimate=budget_estimate,
                isolation=isolation,
                strip_unknown=strip_unknown,
                contract=contract,
            )

        async def _workflow(name: str, args: Any = None) -> Any:
            return await self._run_nested(name, args)

        def _post_message(channel: str, message: Any) -> None:
            self._board.post(channel, message)

        def _fetch_messages(channel: str) -> list[Any]:
            return self._board.fetch(channel)

        injected: dict[str, Any] = {
            "agent": _agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "phase": _eager_awaitable(self._set_phase),
            "log": _awaitable(self._log),
            "workflow": _workflow,
            "budget": ReadOnlyBudget(self._budget),
            "post_message": _post_message,
            "fetch_messages": _fetch_messages,
            "flatten": _flatten,
            "dedup_by": _dedup_by,
            "merge_by": _merge_by,
            "args": args,
        }
        return build_namespace(injected)

    async def _run_nested(self, name: str, args: Any) -> Any:
        if self._nesting_depth >= 1:
            raise WorkflowError(
                "workflow() can only nest one level deep "
                f"(while running nested workflow {name!r})"
            )
        if self.workflow_source_resolver is None:
            raise WorkflowError(
                "Nested workflows are not available in this context "
                f"(cannot resolve {name!r})"
            )
        source = self.workflow_source_resolver(name)
        if source is None:
            raise WorkflowError(f"Unknown workflow: {name!r}")

        violations = validate_script(source)
        if violations:
            raise WorkflowError(
                f"Nested workflow {name!r} failed validation:\n"
                + "\n".join(f"  {v}" for v in violations)
            )

        namespace = self.build_script_namespace(args)
        exec(source, namespace)
        main_fn = cast("Callable[[], Awaitable[Any]]", namespace.get("main"))
        if main_fn is None:
            raise WorkflowError(
                f"Nested workflow {name!r} must define an async `main()`"
            )

        self._log(f"nested workflow: {name}")
        self._nesting_depth += 1
        try:
            return await main_fn()
        finally:
            self._nesting_depth -= 1

    def build_run(
        self, script_path: str | None = None, args: Any = None
    ) -> WorkflowRun:
        return WorkflowRun(
            script_path=script_path,
            args=args,
            phases=[self._phases[name] for name in self._phase_order],
            status=WorkflowStatus.RUNNING,
            started_at=self._started_at,
            budget=self._budget.snapshot(),
        )

    def live_status(self) -> dict[str, Any]:
        budget = self._budget.snapshot()
        phases = []
        for name in self._phase_order:
            report = self._phases.get(name)
            if report is None:
                continue
            completed = sum(1 for r in report.agent_results if r.completed)
            failed_results = [r for r in report.agent_results if not r.completed]
            phases.append({
                "name": name,
                "agents": len(report.agent_results),
                "tokens": report.tokens_total,
                "completed": completed,
                "failed": len(failed_results),
                "failed_details": [
                    {"label": r.label, "error": r.error} for r in failed_results
                ],
            })
        live = [
            {
                "agent_id": la.agent_id,
                "agent": la.agent,
                "label": la.label,
                "phase": la.phase,
                "status": la.status,
                "tokens_in": la.tokens_in,
                "tokens_out": la.tokens_out,
                "tokens": la.tokens_total,
                "elapsed_s": round(time.monotonic() - la.started_at, 1),
                "response_preview": la.response_so_far[:2000],
            }
            for la in self._live_agents.values()
        ]
        finalized_tokens = sum(p.tokens_total for p in self._phases.values())
        live_tokens = sum(la.tokens_total for la in self._live_agents.values())
        return {
            "agent_count": self._agent_count,
            "phases": phases,
            "live_agents": live,
            "live_agent_count": len(live),
            "tokens_finalized": finalized_tokens,
            "tokens_live": live_tokens,
            "tokens_total": finalized_tokens + live_tokens,
            "budget": {
                "total": budget.total,
                "spent": budget.spent,
                "reserved": budget.reserved,
            },
        }

    async def run(self, script_source: str, args: Any = None) -> WorkflowResult:
        violations = validate_script(script_source)
        if violations:
            raise WorkflowError(
                "Script validation failed:\n" + "\n".join(f"  {v}" for v in violations)
            )

        namespace = self.build_script_namespace(args)
        exec(script_source, namespace)

        main_fn = cast("Callable[[], Awaitable[Any]]", namespace.get("main"))
        if main_fn is None:
            raise WorkflowError("Script must define an async `main()` function")

        try:
            return_value = await main_fn()
            status = WorkflowStatus.COMPLETED
            error = None
        except asyncio.CancelledError:
            # A whole-run stop cancels the task awaiting main(); surface that
            # as STOPPED (not FAILED) and still build a result so the host gets
            # the summary + any recovered agent outputs rather than a bare
            # cancel. Do not re-raise: the runner persists in its finally.
            return_value = None
            status = WorkflowStatus.STOPPED
            error = "stopped by user"
        except Exception as e:
            return_value = None
            status = WorkflowStatus.FAILED
            error = str(e)
            logger.error("Workflow script failed", exc_info=e)

        # Compute failed agents BEFORE building the run/summary so we can promote
        # a clean COMPLETED to COMPLETED_WITH_FAILURES when any agent did not
        # complete. Without this, a batch where every agent crashed (or schema-
        # exhausted) would read as an unqualified success — the status enum is
        # the machine-readable contract; the summary string is for humans.
        failed = [
            ar
            for p in self._phases.values()
            for ar in p.agent_results
            if not ar.completed
        ]
        if status == WorkflowStatus.COMPLETED and failed:
            status = WorkflowStatus.COMPLETED_WITH_FAILURES

        run = self.build_run(script_path=None, args=args)
        run.status = status
        run.finished_at = time.monotonic()
        run.budget = self._budget.snapshot()

        summary = (
            f"Workflow {status.value}: {self._agent_count} agents, "
            f"{run.tokens_total} tokens, ${run.cost_total:.4f}"
        )
        if error:
            summary += f" — error: {error}"
        # parallel()/pipeline() degrade a crashing agent to None instead of
        # failing the run (documented null-on-throw, so one bad agent doesn't
        # kill a batch). Surface the failed agents and their (deduped) errors
        # so a systemic failure is visible in the human-readable summary too.
        if failed:
            seen: list[str] = []
            for ar in failed:
                msg = ar.error or "(no error recorded)"
                # Append the first field-level schema error so a systemic schema
                # mismatch (e.g. every agent missing a required field) is named
                # in the summary, not just "Schema validation failed after N
                # attempts". Full per-agent detail lives on AgentResult and is
                # surfaced via the workflow_results tool.
                if ar.schema_errors:
                    msg = f"{msg} [{ar.schema_errors[0]}]"
                if msg not in seen:
                    seen.append(msg)
            detail = "; ".join(seen)[:500]
            summary += f" — {len(failed)}/{run.agent_count} agent(s) failed: {detail}"

        return WorkflowResult(return_value=return_value, run=run, summary=summary)

    def snapshot(
        self,
        run_id: str,
        script_source: str,
        args: Any = None,
        *,
        return_value: Any = None,
    ) -> WorkflowRunSnapshot:
        return WorkflowRunSnapshot(
            run_id=run_id,
            script_source=script_source,
            args=args,
            status=WorkflowStatus.PAUSED,
            started_at=self._started_at,
            budget_total=self.budget_total,
            budget_spent=self._budget.snapshot().spent,
            cached_results=list(self._cache.values()),
            return_value=_coerce_json_safe(return_value),
        )

    def restore_from_snapshot(self, snapshot: WorkflowRunSnapshot) -> None:
        for cached in snapshot.cached_results:
            self._cache[cached.prompt_hash] = cached
        # Restore prior spend so the budget cap is not silently reset to 0 on
        # resume (which would allow the resumed run to overspend).
        self._budget.restore_spent(snapshot.budget_spent)
        self._log(f"restored {snapshot.cached_count} cached results from snapshot")
