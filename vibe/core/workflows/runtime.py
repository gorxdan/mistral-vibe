from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
import json
import time
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from vibe.core.logger import logger
from vibe.core.workflows.budget import Budget, Reservation
from vibe.core.workflows.models import (
    AgentResult,
    PhaseReport,
    WorkflowResult,
    WorkflowRun,
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

T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")

DEFAULT_MAX_CONCURRENT = 16
DEFAULT_MAX_AGENTS = 1000
DEFAULT_BUDGET_TOTAL = None
DEFAULT_SCHEMA_RETRIES = 2


class AgentCapExceeded(Exception):
    pass


class WorkflowError(Exception):
    pass


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
    _semaphore: asyncio.Semaphore = field(init=False)
    _budget: Budget = field(init=False)
    _agent_count: int = field(default=0, init=False)
    _phases: dict[str, PhaseReport] = field(default_factory=dict, init=False)
    _phase_order: list[str] = field(default_factory=list, init=False)
    _event_sink: Callable[[str], None] | None = field(default=None, init=False)
    _started_at: float = field(default_factory=time.monotonic, init=False)

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
            loop = self._create_loop(effective_prompt, agent=agent)

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

            tokens_in = getattr(loop.stats, "session_prompt_tokens", 0)
            tokens_out = getattr(loop.stats, "session_completion_tokens", 0)

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
        )
        raise SchemaValidationError(
            f"Response did not match schema after {self.schema_retries + 1} attempts. "
            f"Last errors: {'; '.join(last_errors)}"
        )

    def _create_loop(self, prompt: str, *, agent: str) -> AgentLoop:
        if self.agent_loop_factory is not None:
            return self.agent_loop_factory(
                prompt, agent=agent, parent_context=self.parent_context
            )
        return self._create_real_loop(agent=agent)

    def _create_real_loop(self, *, agent: str) -> AgentLoop:
        from vibe.core.agent_loop import AgentLoop as _AgentLoop
        from vibe.core.config import SessionLoggingConfig, VibeConfig

        base_config = VibeConfig.load(
            session_logging=SessionLoggingConfig(
                save_dir="", session_prefix=agent, enabled=False
            )
        )
        ctx = self.parent_context
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
        content = getattr(event, "content", None)
        if content and isinstance(content, str):
            return content
        return None

    def _finalize_agent(
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
    ) -> None:
        self._budget.reconcile(reservation, tokens_in, tokens_out)

        result = AgentResult(
            label=label,
            phase=phase,
            prompt=prompt,
            response=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=0.0,
            completed=completed,
            error=error,
        )
        self._record_agent_result(result)

    async def parallel(self, *thunks: Callable[[], Awaitable[T]]) -> list[T]:
        async def _run(thunk: Callable[[], Awaitable[T]]) -> T:
            async with self._semaphore:
                return await thunk()

        return await asyncio.gather(*[_run(t) for t in thunks])

    async def pipeline(
        self, items: list[I], fn: Callable[[I], Awaitable[O]]
    ) -> list[O]:
        async def _run(item: I) -> O:
            async with self._semaphore:
                return await fn(item)

        return await asyncio.gather(*[_run(i) for i in items])

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

        injected: dict[str, Any] = {
            "agent": _agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "phase": self._declare_phase,
            "log": self._log,
            "budget": self._budget,
            "args": args,
        }
        return build_namespace(injected)

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
