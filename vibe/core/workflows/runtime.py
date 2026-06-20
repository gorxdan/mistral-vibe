from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import time
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from vibe.core.logger import logger
from vibe.core.types import AssistantEvent
from vibe.core.workflows.budget import (
    Budget,
    BudgetExhausted,
    ReadOnlyBudget,
    Reservation,
)
from vibe.core.workflows.models import (
    AgentResult,
    CachedAgentResult,
    PhaseReport,
    WorkflowResult,
    WorkflowRun,
    WorkflowRunSnapshot,
    WorkflowStatus,
)
from vibe.core.workflows.schema import (
    SchemaValidationError,
    build_prompt_fallback,
    build_response_format,
    validate_against_schema,
)
from vibe.core.workflows.security import build_namespace, validate_script

if TYPE_CHECKING:
    from vibe.core.agent_loop import AgentLoop
    from vibe.core.tools.base import InvokeContext


class _AwaitableResult:
    """Wraps a coroutine so it must be awaited before unpacking.

    Gives a helpful error if the workflow script tries to unpack or index
    the result of parallel()/pipeline() without awaiting first.
    """

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

DEFAULT_MAX_CONCURRENT = 16
DEFAULT_MAX_AGENTS = 1000
DEFAULT_BUDGET_TOTAL = None
DEFAULT_SCHEMA_RETRIES = 2
DEFAULT_ISOLATED_MAX_TURNS = 40

# Signature of the isolated-agent executor seam: (prompt, agent, label,
# max_turns) -> the agent's text output. Injectable so tests can stub the
# worktree + subprocess; the default runs `vibe -p` in a fresh git worktree.
IsolatedExecutor = Callable[
    [str, str, str | None, int],
    Awaitable["str | tuple[str, dict[str, int] | None]"],
]


class AgentCapExceeded(Exception):
    pass


class WorkflowError(Exception):
    pass


@dataclass
class _LiveAgent:
    """An in-flight agent whose tokens are tracked live (before finalize).

    Recorded separately from finalized PhaseReport results so observers (the
    workflow_status tool, /workflows list) can see per-agent spend while an
    agent is still running, not only after it completes. Retired (removed) once
    the agent's AgentResult is recorded into its phase, so at any instant an
    agent is counted in exactly one place — live XOR finalized — never both.
    """

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

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class _MessageBoard:
    """In-process, single-event-loop message board shared across all agents in a
    workflow run. Lets the orchestrator (main()) route named-channel handoffs
    between phases and pipeline stages without funneling everything through
    return values at a barrier.

    Safe for concurrent use only within one asyncio loop (no locks): workflow
    agents and the orchestrator all run on the runtime's loop.
    """

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
    """Extract the real token stats line a workflow subprocess emits on stderr."""
    for line in reversed(stderr_text.splitlines()):
        if line.startswith(_ISOLATED_STATS_SENTINEL):
            try:
                data = json.loads(line[len(_ISOLATED_STATS_SENTINEL):])
                return {
                    "prompt_tokens": int(data.get("prompt_tokens", 0)),
                    "completion_tokens": int(data.get("completion_tokens", 0)),
                }
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                return None
    return None


class AgentLoopFactory(Protocol):
    def __call__(
        self, prompt: str, *, agent: str, parent_context: InvokeContext | None
    ) -> AgentLoop: ...


@dataclass
class WorkflowRuntime:
    parent_context: InvokeContext | None = None
    max_concurrent: int = DEFAULT_MAX_CONCURRENT
    max_agents: int = DEFAULT_MAX_AGENTS
    budget_total: int | None = DEFAULT_BUDGET_TOTAL
    schema_retries: int = DEFAULT_SCHEMA_RETRIES
    agent_loop_factory: AgentLoopFactory | None = None
    # Resolves a workflow name to its script source, enabling nested workflow()
    # calls. Wired from WorkflowManager at launch; None disables nesting.
    workflow_source_resolver: Callable[[str], str | None] | None = None
    # Runs an isolation="worktree" agent. None -> the default executor that
    # spawns `vibe -p` in a fresh git worktree. Injectable for tests.
    isolated_executor: IsolatedExecutor | None = None
    _semaphore: asyncio.Semaphore = field(init=False)
    _budget: Budget = field(init=False)
    _agent_count: int = field(default=0, init=False)
    _nesting_depth: int = field(default=0, init=False)
    _phases: dict[str, PhaseReport] = field(default_factory=dict, init=False)
    _phase_order: list[str] = field(default_factory=list, init=False)
    _event_sink: Callable[[str], None] | None = field(default=None, init=False)
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _cache: dict[str, CachedAgentResult] = field(default_factory=dict, init=False)
    _live_agents: dict[str, _LiveAgent] = field(default_factory=dict, init=False)
    _next_live_id: int = field(default=0, init=False)
    _board: _MessageBoard = field(default_factory=_MessageBoard, init=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._budget = Budget(total=self.budget_total)

    def set_event_sink(self, sink: Callable[[str], None]) -> None:
        self._event_sink = sink

    def _log(self, msg: str) -> None:
        logger.info("workflow: %s", msg)
        if self._event_sink:
            self._event_sink(msg)

    def _declare_phase(self, name: str) -> None:
        if name not in self._phases:
            self._phases[name] = PhaseReport(name=name)
            self._phase_order.append(name)
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
    ) -> _LiveAgent:
        live_id = f"la-{self._next_live_id}"
        self._next_live_id += 1
        live = _LiveAgent(
            agent_id=live_id, agent=agent, model=model, label=label, phase=phase
        )
        self._live_agents[live_id] = live
        return live

    def _retire_live(
        self, live: _LiveAgent, *, status: str, error: str | None = None
    ) -> None:
        live.status = status
        live.error = error
        self._live_agents.pop(live.agent_id, None)

    def _validate_workflow_profile(self, agent: str, isolation: str | None) -> None:
        """Validate the requested agent profile for a workflow spawn: it must be a
        subagent, and a full-tool profile (no enabled_tools allowlist, e.g.
        'worker') must run isolated — its write tools would race the shared tree
        and its ASK tools auto-skip headless. No-op when the profile can't be
        resolved (e.g. no agent_manager in unit contexts)."""
        ctx = self.parent_context
        if not ctx or not ctx.agent_manager:
            return
        try:
            profile = ctx.agent_manager.get_agent(agent)
        except ValueError:
            return
        from vibe.core.agents.models import AgentType

        agent_type = getattr(profile, "agent_type", AgentType.SUBAGENT)
        if agent_type != AgentType.SUBAGENT:
            raise WorkflowError(
                f"Agent '{agent}' is a {agent_type.value} agent. "
                f"Only subagents can be used in workflows."
            )
        overrides = getattr(profile, "overrides", {}) or {}
        if isolation != "worktree" and not overrides.get("enabled_tools"):
            raise WorkflowError(
                f"Agent '{agent}' has no tool allowlist (full tools incl. write); "
                f"in a workflow it must run with isolation='worktree'."
            )

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
    ) -> str | dict[str, Any]:
        cache_key = _prompt_hash(prompt, agent, phase, isolation)
        if cached := self._cache.get(cache_key):
            self._log(f"cache hit: {label or agent}")
            self._record_cached_result(cached)
            return cached.response

        # Reject non-subagents and require isolation for full-tool profiles
        # (e.g. 'worker') before reserving budget / counting the agent.
        self._validate_workflow_profile(agent, isolation)

        if self._agent_count >= self.max_agents:
            raise AgentCapExceeded(
                f"Agent cap reached: {self._agent_count}/{self.max_agents}"
            )

        reservation = self._budget.reserve(budget_estimate)
        self._agent_count += 1

        async with self._semaphore:
            if isolation == "worktree":
                return await self._run_isolated_agent(
                    prompt=prompt,
                    agent=agent,
                    model=model,
                    label=label,
                    phase=phase,
                    schema=schema,
                    reservation=reservation,
                    cache_key=cache_key,
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
                phase=phase,
                schema=schema,
                reservation=reservation,
                cache_key=cache_key,
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
    ) -> str | dict[str, Any]:
        response_format = build_response_format(schema) if schema is not None else None
        effective_prompt = prompt
        if schema is not None:
            effective_prompt = prompt + build_prompt_fallback(schema)

        live = self._register_live(agent=agent, model=model, label=label, phase=phase)

        last_errors: list[str] = []
        accumulated: list[str] = []
        tokens_in = 0
        tokens_out = 0
        completed = True
        error_msg: str | None = None

        for attempt in range(self.schema_retries + 1):
            accumulated = []
            loop = self._create_loop(effective_prompt, agent=agent, model=model)

            try:
                async with aclosing(
                    loop.act(effective_prompt, response_format=response_format)
                ) as events:
                    async for event in events:
                        content = self._extract_content(event)
                        if content:
                            accumulated.append(content)
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
            except Exception as e:
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
                return response_text

            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError as e:
                last_errors = [f"JSON parse error: {e}"]
            else:
                errors = validate_against_schema(parsed, schema)
                if not errors:
                    self._finalize_agent(
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
            raise WorkflowError(error_msg or "Agent failed without producing output")

        self._finalize_agent(
            prompt=prompt,
            response="".join(accumulated),
            label=label,
            phase=phase,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reservation=reservation,
            completed=False,
            error=f"Schema validation failed after {self.schema_retries + 1} attempts",
            cache_key=cache_key,
            agent=agent,
            model=model,
            live=live,
        )
        raise SchemaValidationError(
            f"Response did not match schema after {self.schema_retries + 1} attempts. "
            f"Last errors: {'; '.join(last_errors)}"
        )

    def _create_loop(
        self, prompt: str, *, agent: str, model: str | None = None
    ) -> AgentLoop:
        if self.agent_loop_factory is not None:
            return self.agent_loop_factory(
                prompt, agent=agent, parent_context=self.parent_context
            )
        return self._create_real_loop(agent=agent, model=model)

    def _create_real_loop(self, *, agent: str, model: str | None = None) -> AgentLoop:
        from vibe.core.agent_loop import AgentLoop as _AgentLoop
        from vibe.core.config import SessionLoggingConfig, VibeConfig

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

        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx and ctx.session_dir else "",
            session_prefix=agent,
            enabled=ctx is not None and ctx.session_dir is not None,
        )
        overrides: dict[str, Any] = {}
        if model:
            overrides["active_model"] = model
        base_config = VibeConfig.load(session_logging=session_logging, **overrides)
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

    def _finalize_agent(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        response: str | dict[str, Any],
        label: str | None,
        phase: str | None,
        tokens_in: int,
        tokens_out: int,
        reservation: Reservation,
        completed: bool,
        error: str | None,
        cache_key: str,
        agent: str,
        model: str | None = None,
        live: _LiveAgent | None = None,
    ) -> None:
        self._budget.reconcile(reservation, tokens_in, tokens_out)

        cost = self._compute_cost(tokens_in, tokens_out, model)
        result = AgentResult(
            label=label,
            phase=phase,
            prompt=prompt,
            response=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            completed=completed,
            error=error,
        )
        self._record_agent_result(result)
        # Retire the live tracker the moment its result is recorded, so an agent
        # is reflected as live XOR finalized, never both (which would double its
        # tokens in the live view + finalized phases).
        if live is not None:
            self._retire_live(live, status="completed" if completed else "failed", error=error)

        if completed:
            self._cache[cache_key] = CachedAgentResult(
                prompt_hash=cache_key,
                agent=agent,
                label=label,
                phase=phase,
                response=response,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                completed=completed,
                error=error,
            )

    def _record_cached_result(self, cached: CachedAgentResult) -> None:
        # A cache hit consumes no tokens in THIS run; the original spend is
        # already reflected in the restored budget. Recording the cached token
        # counts here would double-count them in run.tokens_total relative to
        # budget.spent. Report zero for this run.
        result = AgentResult(
            label=cached.label,
            phase=cached.phase,
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
    ) -> str | dict[str, Any]:
        """Run an isolation='worktree' agent: a `vibe -p` subprocess in a fresh
        git worktree, so file mutations cannot collide with other agents. The
        worktree's branch is kept for manual merge if the agent changed files."""
        effective_prompt = prompt + (
            build_prompt_fallback(schema) if schema is not None else ""
        )
        executor = self.isolated_executor or self._default_isolated_executor
        completed = True
        error_msg: str | None = None
        output = ""
        stats: dict[str, int] | None = None
        # Isolated agents run as a subprocess; their token usage is unknowable
        # until they exit and emit stats. The live tracker therefore shows 0
        # while running (honest), then finalize records the real total.
        live = self._register_live(agent=agent, model=model, label=label, phase=phase)
        try:
            result = await executor(
                effective_prompt, agent, label, DEFAULT_ISOLATED_MAX_TURNS
            )
            # Executor may return just the output (str) or (output, stats).
            if isinstance(result, tuple):
                output, stats = result
            else:
                output = result
        except (AgentCapExceeded, BudgetExhausted):
            raise
        except Exception as e:
            completed = False
            error_msg = str(e)

        response: str | dict[str, Any] = output
        if completed and schema is not None:
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as e:
                completed = False
                error_msg = f"isolated agent returned invalid JSON: {e}"
            else:
                errors = validate_against_schema(parsed, schema)
                if errors:
                    completed = False
                    error_msg = "; ".join(str(err) for err in errors)
                else:
                    response = parsed

        # Charge real subprocess tokens when the executor surfaced them; else
        # fall back to the reserved estimate so the budget cap stays enforced
        # (charging 0 would make isolated agents spend nothing against the cap).
        if stats is not None:
            tokens_in = int(stats.get("prompt_tokens", 0))
            tokens_out = int(stats.get("completion_tokens", 0))
        else:
            tokens_in, tokens_out = 0, reservation.estimate
            if completed:
                logger.info(
                    "workflow: isolated agent %s emitted no token stats; charging "
                    "the estimate (%d)",
                    label or agent,
                    reservation.estimate,
                )
        self._finalize_agent(
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
        )
        if not completed:
            raise WorkflowError(f"isolated agent failed: {error_msg}")
        return response

    async def _default_isolated_executor(
        self, prompt: str, agent: str, label: str | None, max_turns: int
    ) -> tuple[str, dict[str, int] | None]:
        import os
        from pathlib import Path
        import shlex
        import signal
        import sys

        from vibe.core.worktree.ephemeral import (
            create_ephemeral_worktree,
            remove_ephemeral_worktree,
        )

        wt = create_ephemeral_worktree(Path.cwd(), label or agent)
        try:
            # The binary prefix is overridable (tests point it at a fake `vibe`);
            # unset -> the real `vibe` module. Everything else (worktree, cwd,
            # env, communicate, stats parse, cleanup) runs unchanged.
            base = os.environ.get("VIBE_ISOLATED_EXECUTOR_CMD")
            prefix = shlex.split(base) if base else [sys.executable, "-m", "vibe"]
            cmd = [
                *prefix, "-p", prompt,
                "--agent", "auto-approve", "--trust",
                "--output", "text", "--max-turns", str(max_turns),
            ]
            env = os.environ.copy()
            # Ask the child to emit real token stats on stderr so we can charge
            # actual usage against the budget instead of the estimate.
            env["VIBE_WORKFLOW_EMIT_STATS"] = "1"
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(wt.path),
                # The child is a trusted `vibe` instance and needs the parent's
                # credentials (e.g. the provider API key) to run; pass the env
                # explicitly (same as teams) rather than relying on implicit
                # inheritance. Isolation bounds files, not env/secrets.
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await proc.communicate()
            except asyncio.CancelledError:
                # Reap the whole group AND wait for exit before the finally
                # removes the worktree — otherwise `git worktree remove` races a
                # process that still owns the worktree as its cwd (EBUSY -> leak).
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
            stderr_text = (stderr or b"").decode("utf-8", "replace")
            if proc.returncode != 0:
                raise WorkflowError(
                    f"isolated agent subprocess failed (rc={proc.returncode}): "
                    f"{stderr_text[:300]}"
                )
            return (stdout or b"").decode("utf-8", "replace"), _parse_stats(stderr_text)
        finally:
            remove_ephemeral_worktree(wt)

    def parallel(self, *thunks: Any) -> _AwaitableResult:
        """Run thunks concurrently and return their results in order (barrier).

        Mirrors the Claude Code Workflow contract:
        - accepts either ``parallel(t1, t2, ...)`` or ``parallel([t1, t2, ...])``;
        - a thunk that raises resolves to ``None`` (the call never rejects), so
          one bad agent does not kill the batch — filter with ``[r for r in ... if r]``.

        Concurrency is bounded inside spawn_agent (the single owner of
        self._semaphore); acquiring it here too would make each agent hold two
        permits and deadlock once concurrent thunks reach max_concurrent.
        """
        if len(thunks) == 1 and isinstance(thunks[0], (list, tuple)):
            thunk_list = list(thunks[0])
        else:
            thunk_list = list(thunks)

        async def _safe(thunk: Callable[[], Awaitable[Any]]) -> Any:
            try:
                return await thunk()
            except (AgentCapExceeded, BudgetExhausted):
                # Hard ceilings (runaway-agent / overspend backstops) must fail
                # the run, not silently become a None result that masks the
                # breach. Ordinary failures still degrade to None below.
                raise
            except Exception:
                logger.warning("workflow: parallel() thunk failed", exc_info=True)
                return None

        async def _run() -> list[Any]:
            return await asyncio.gather(*[_safe(t) for t in thunk_list])

        return _AwaitableResult(_run())

    @staticmethod
    def _call_stage(
        stage: Callable[..., Any], result: Any, item: Any, index: int
    ) -> Any:
        """Invoke a pipeline stage, passing as many of (prev, item, index) as it
        accepts (min 1). Lets a one-arg stage ``fn(item)`` and a full Claude-Code
        style ``(prev, item, index)`` stage both work.
        """
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
        """A pipeline stage must accept at least one positional arg (the item /
        prev result). Used to reject keyword-only stages up front with a clear
        error instead of silently dropping every item to None at call time.
        """
        try:
            params = list(inspect.signature(stage).parameters.values())
        except (TypeError, ValueError):
            return True  # uninspectable (e.g. a builtin) — assume callable
        return any(
            p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL}
            for p in params
        )

    def pipeline(
        self, items: list[I], *stages: Callable[..., Awaitable[Any]]
    ) -> _AwaitableResult:
        """Run each item through all stages independently — no barrier between
        stages, so item A can be in stage 3 while item B is still in stage 1
        (Claude Code pipeline semantics). Each stage receives (prevResult,
        originalItem, index). A stage that raises drops that item to ``None`` and
        skips its remaining stages. Returns one final result per item, in order.

        Limiting happens in spawn_agent, not here, to avoid nested acquisition of
        the non-reentrant semaphore (deadlock at scale).
        """
        for stage in stages:
            if not self._stage_accepts_positional(stage):
                raise WorkflowError(
                    f"pipeline stage {getattr(stage, '__name__', stage)!r} must "
                    "accept at least one positional argument (prev[, item, index])"
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

        async def _run() -> list[Any]:
            return await asyncio.gather(*[
                _run_item(i, it) for i, it in enumerate(items_list)
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
        ) -> str | dict[str, Any]:
            return await self.spawn_agent(
                prompt,
                agent=agent,
                model=model,
                label=label,
                phase=phase,
                schema=schema,
                budget_estimate=budget_estimate,
                isolation=isolation,
            )

        async def _workflow(name: str, args: Any = None) -> Any:
            return await self._run_nested(name, args)

        def _post_message(channel: str, message: Any) -> None:
            """Post a message to a named channel on this run's shared board.

            Visible to other agents/stages in the SAME run via fetch_messages.
            Use for inter-agent handoffs that don't fit the barrier-return model
            (e.g. a finder posting partial results a verifier polls for).
            """
            self._board.post(channel, message)

        def _fetch_messages(channel: str) -> list[Any]:
            """Return (a copy of) all messages posted to a channel so far."""
            return self._board.fetch(channel)

        injected: dict[str, Any] = {
            "agent": _agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "phase": self._declare_phase,
            "log": self._log,
            "workflow": _workflow,
            "budget": ReadOnlyBudget(self._budget),
            "post_message": _post_message,
            "fetch_messages": _fetch_messages,
            "args": args,
        }
        return build_namespace(injected)

    async def _run_nested(self, name: str, args: Any) -> Any:
        """Run another workflow inline as a sub-step (Claude Code workflow()).

        The child shares this runtime's budget, semaphore, agent counter and
        result cache, and its phases merge into the parent's (so it appears in
        the same live monitor). Nesting is one level only — a workflow() call
        inside a nested run raises.
        """
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
        exec(source, namespace)  # noqa: S102 — sandboxed namespace, validated above
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
        """Point-in-time, JSON-serializable view of this run for observers
        (the workflow_status tool, /workflows list). Includes both finalized
        phase results AND in-flight agents with their live token counts, so a
        caller can gauge progress mid-run — not only after each agent finishes.

        Live + finalized are mutually exclusive per agent (an agent is retired
        from _live_agents the instant its AgentResult is recorded), so summing
        the two never double-counts.
        """
        budget = self._budget.snapshot()
        phases = []
        for name in self._phase_order:
            report = self._phases.get(name)
            if report is None:
                continue
            phases.append(
                {
                    "name": name,
                    "agents": len(report.agent_results),
                    "tokens": report.tokens_total,
                }
            )
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
        except Exception as e:
            return_value = None
            status = WorkflowStatus.FAILED
            error = str(e)
            logger.error("Workflow script failed", exc_info=e)

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

        return WorkflowResult(return_value=return_value, run=run, summary=summary)

    def snapshot(
        self, run_id: str, script_source: str, args: Any = None
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
        )

    def restore_from_snapshot(self, snapshot: WorkflowRunSnapshot) -> None:
        for cached in snapshot.cached_results:
            self._cache[cached.prompt_hash] = cached
        # Restore prior spend so the budget cap is not silently reset to 0 on
        # resume (which would allow the resumed run to overspend).
        self._budget.restore_spent(snapshot.budget_spent)
        self._log(f"restored {snapshot.cached_count} cached results from snapshot")
