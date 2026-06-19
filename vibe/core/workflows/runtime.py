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


class AgentCapExceeded(Exception):
    pass


class WorkflowError(Exception):
    pass


def _prompt_hash(prompt: str, agent: str, phase: str | None = None) -> str:
    return hashlib.sha256(f"{agent}:{phase}:{prompt}".encode()).hexdigest()[:16]


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
    _semaphore: asyncio.Semaphore = field(init=False)
    _budget: Budget = field(init=False)
    _agent_count: int = field(default=0, init=False)
    _nesting_depth: int = field(default=0, init=False)
    _phases: dict[str, PhaseReport] = field(default_factory=dict, init=False)
    _phase_order: list[str] = field(default_factory=list, init=False)
    _event_sink: Callable[[str], None] | None = field(default=None, init=False)
    _started_at: float = field(default_factory=time.monotonic, init=False)
    _cache: dict[str, CachedAgentResult] = field(default_factory=dict, init=False)

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
    ) -> str | dict[str, Any]:
        cache_key = _prompt_hash(prompt, agent, phase)
        if cached := self._cache.get(cache_key):
            self._log(f"cache hit: {label or agent}")
            self._record_cached_result(cached)
            return cached.response

        if self._agent_count >= self.max_agents:
            raise AgentCapExceeded(
                f"Agent cap reached: {self._agent_count}/{self.max_agents}"
            )

        reservation = self._budget.reserve(budget_estimate)
        self._agent_count += 1

        async with self._semaphore:
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
            except Exception as e:
                completed = False
                error_msg = str(e)
                break

            # Accumulate across attempts: a fresh loop is created per retry, so
            # its stats report only that attempt. Overwriting dropped the tokens
            # spent on failed schema attempts from the budget.
            tokens_in += getattr(loop.stats, "session_prompt_tokens", 0)
            tokens_out += getattr(loop.stats, "session_completion_tokens", 0)

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
        ) -> str | dict[str, Any]:
            return await self.spawn_agent(
                prompt,
                agent=agent,
                model=model,
                label=label,
                phase=phase,
                schema=schema,
                budget_estimate=budget_estimate,
            )

        async def _workflow(name: str, args: Any = None) -> Any:
            return await self._run_nested(name, args)

        injected: dict[str, Any] = {
            "agent": _agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "phase": self._declare_phase,
            "log": self._log,
            "workflow": _workflow,
            "budget": ReadOnlyBudget(self._budget),
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
