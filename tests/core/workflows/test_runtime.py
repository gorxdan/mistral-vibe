from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from pydantic import ValidationError
import pytest

from vibe.core.agents.manager import AgentManager
from vibe.core.llm.exceptions import BackendError, PayloadSummary
from vibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    ReasoningEvent,
    UserMessageEvent,
)
from vibe.core.workflows.contract import ContractFailure
from vibe.core.workflows.models import SchemaValidationFailure
from vibe.core.workflows.runtime import (
    AgentCapExceeded,
    WorkflowError,
    WorkflowRuntime,
    _WorkerSpawnArgs,
)
from vibe.core.workflows.schema import SchemaValidationError

pytestmark = pytest.mark.asyncio


@dataclass
class MockStats:
    session_prompt_tokens: int = 1000
    session_completion_tokens: int = 500


@dataclass
class MockAgentLoop:
    response_text: str = "mock response"
    tokens_in: int = 1000
    tokens_out: int = 500
    stats: MockStats = field(default_factory=MockStats)
    delay: float = 0.0
    _call_count: int = field(default=0, init=False)

    async def act(
        self, prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | UserMessageEvent, None]:
        self._call_count += 1
        # Mirror the real AgentLoop.act stream: a prompt echo and a
        # chain-of-thought event precede the assistant answer. Only the
        # assistant content must end up in the response.
        yield UserMessageEvent(content=prompt, message_id="u1")
        yield ReasoningEvent(content="thinking about it", message_id="r1")
        if self.delay:
            await asyncio.sleep(self.delay)
        yield AssistantEvent(content=self.response_text, message_id="a1")


def make_factory(
    response_text: str = "mock response",
    tokens_in: int = 1000,
    tokens_out: int = 500,
    delay: float = 0.0,
) -> Any:
    def factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> MockAgentLoop:
        stats = MockStats(
            session_prompt_tokens=tokens_in, session_completion_tokens=tokens_out
        )
        return MockAgentLoop(
            response_text=response_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            stats=stats,
            delay=delay,
        )

    return factory


@pytest.fixture
def runtime() -> WorkflowRuntime:
    return WorkflowRuntime(
        agent_loop_factory=make_factory(),
        max_concurrent=4,
        max_agents=100,
        budget_total=1_000_000,
    )


async def test_spawn_agent_returns_string(runtime: WorkflowRuntime) -> None:
    result = await runtime.spawn_agent("test prompt")
    assert result == "mock response"
    assert runtime._agent_count == 1


async def test_namespace_agent_tolerates_unknown_kwargs() -> None:
    # A stray kwarg (e.g. max_concurrency, which belongs on parallel/pipeline)
    # must degrade a single agent() call, not crash the whole workflow at 0
    # agents. agentType/agent_type is honored as an alias for agent.
    seen: list[str] = []

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        seen.append(agent)
        return MockAgentLoop()

    rt = WorkflowRuntime(
        agent_loop_factory=factory, max_agents=10, budget_total=1_000_000
    )
    agent_fn = rt.build_script_namespace()["agent"]

    # Unknown kwarg does not raise; the agent still runs.
    assert await agent_fn("p", max_concurrency=3) == "mock response"
    # agentType alias is honored rather than silently downgraded to the default.
    await agent_fn("p2", agentType="reviewer")
    # Unknown kwarg alongside an explicit agent= is ignored, agent= preserved.
    await agent_fn("p3", agent="planner", effort="high")

    assert rt._agent_count == 3
    assert seen == ["explore", "reviewer", "planner"]


async def test_phase_binds_subsequent_agents_implicitly() -> None:
    # i5: phase('x') sets an ambient phase. Subsequent agent() calls without an
    # explicit phase= kwarg inherit it. Explicit phase= always wins. phase(None)
    # resets so agents land in "default" again.
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_agents=10)
    rt._set_phase("audit")
    await rt.spawn_agent("a", label="a")  # no phase= → inherits "audit"
    await rt.spawn_agent("b", label="b", phase="explicit")  # explicit wins
    rt._set_phase(None)
    await rt.spawn_agent("c", label="c")  # reset → "default"

    phases = {p.name: p for p in rt._phases.values()}
    assert any(r.label == "a" for r in phases["audit"].agent_results), "a in audit"
    assert any(r.label == "b" for r in phases["explicit"].agent_results), (
        "b in explicit"
    )
    assert any(r.label == "c" for r in phases["default"].agent_results), "c in default"


async def test_phase_binding_in_workflow_script() -> None:
    # End-to-end via a script: phase() then agent() without phase= must land
    # in the declared phase, not "default".
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_agents=10)
    script = """
async def main():
    phase("research")
    await agent("find it", label="finder")
    await agent("another", label="another", phase="override")
    return {}
"""
    result = await rt.run(script)
    phase_names = {p.name for p in result.run.phases}
    assert "research" in phase_names
    assert "override" in phase_names
    assert "default" not in phase_names, (
        "agents after phase('research') must inherit it, not land in default"
    )


async def test_spawn_agent_with_schema_returns_dict() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(response_text='{"answer": "42"}'),
        max_concurrent=4,
    )
    result = await rt.spawn_agent("test", schema=schema)
    assert isinstance(result, dict)
    assert result["answer"] == "42"


async def test_spawn_agent_schema_retry_on_bad_json() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    responses = ["not json", '{"answer": "42"}']
    call_idx = [0]

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        resp = responses[min(call_idx[0], len(responses) - 1)]
        call_idx[0] += 1
        stats = MockStats()
        return MockAgentLoop(response_text=resp, stats=stats)

    rt = WorkflowRuntime(agent_loop_factory=factory, schema_retries=2)
    result = await rt.spawn_agent("test", schema=schema)
    assert isinstance(result, dict)
    assert result["answer"] == "42"
    assert call_idx[0] == 2


async def test_spawn_agent_schema_returns_failure_after_max_retries() -> None:
    # Default (strict_schema=False): schema exhaustion returns a structured
    # SchemaValidationFailure carrying the raw response, so parallel._safe
    # never silently swallows the output to None.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(response_text="not json"), schema_retries=1
    )
    result = await rt.spawn_agent("test", schema=schema)
    assert isinstance(result, SchemaValidationFailure)
    assert result.raw_response == "not json"
    assert "Schema validation failed" in result.error
    assert result.schema_errors  # last_errors captured


async def test_schema_failure_is_falsy_and_dict_like() -> None:
    # A schema-failed result degrades gracefully so one bad agent does not crash
    # the whole run: it is falsy (the canonical `[r for r in ... if r]` filter
    # drops it like a None) and `.get(...)` returns the default rather than
    # raising AttributeError. Detail stays inspectable.
    f = SchemaValidationFailure(
        raw_response="not json", error="Schema validation failed", schema_errors=["x"]
    )
    assert not f
    assert f.get("findings", []) == []
    assert f.get("anything") is None
    assert [r for r in [f, {"ok": 1}] if r] == [{"ok": 1}]
    assert isinstance(f, SchemaValidationFailure)
    assert f.schema_errors == ["x"]


def test_schema_failure_is_json_serializable() -> None:
    # Regression for the wf-2 root cause: a SchemaValidationFailure flowing into
    # a workflow script's json.dumps(results) crashed the whole run with
    # "Object of type SchemaValidationFailure is not JSON serializable". It is
    # now a dict subclass, so it serializes as a plain dict and one failed agent
    # degrades the batch instead of killing it.
    import json

    f = SchemaValidationFailure(
        raw_response="not json",
        error="Schema validation failed after 3 attempts",
        schema_errors=["$.findings[0].severity: 'medium' not in enum"],
    )
    payload = json.dumps([f, {"findings": []}])
    assert json.loads(payload) == [
        {
            "raw_response": "not json",
            "error": "Schema validation failed after 3 attempts",
            "schema_errors": ["$.findings[0].severity: 'medium' not in enum"],
        },
        {"findings": []},
    ]


def test_schema_failure_truthiness_filter_unaffected_by_dict_subclass() -> None:
    # The documented discriminator is truthiness. Filter with `if r`, never
    # `isinstance(r, dict)` (which now wrongly includes the failure since it is
    # a dict subclass). Pin both halves of that contract.
    f = SchemaValidationFailure(raw_response="x", error="bad")
    good = {"findings": [1, 2]}
    assert [r for r in [f, good] if r] == [good]
    # Guard against the anti-pattern: this is documented NOT to discriminate.
    assert [r for r in [f, good] if isinstance(r, dict)] == [f, good]


async def test_parallel_accepts_bare_coroutines_and_thunks() -> None:
    # Headline change: parallel() takes coroutines directly (the natural fan-out
    # form) as well as zero-arg thunks, and mixes them; the list form works too.
    # A non-awaitable/non-callable item fails loud instead of silently dropping.
    rt = WorkflowRuntime(agent_loop_factory=make_factory())

    async def co(v: int) -> int:
        return v

    assert await rt.parallel(co(1), co(2)) == [1, 2]
    assert await rt.parallel(lambda: co(3), lambda: co(4)) == [3, 4]
    assert await rt.parallel(co(5), lambda: co(6)) == [5, 6]
    assert await rt.parallel([co(7), co(8)]) == [7, 8]
    with pytest.raises(WorkflowError):
        await rt.parallel(123)


async def test_spawn_agent_schema_raises_after_max_retries_strict() -> None:
    # strict_schema=True preserves the legacy hard-fail behavior.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(response_text="not json"),
        schema_retries=1,
        strict_schema=True,
    )
    with pytest.raises(SchemaValidationError):
        await rt.spawn_agent("test", schema=schema)


def _structured_output_rejection() -> BackendError:
    # Mirrors the openai-chatgpt Responses-API 400 that killed wf-1 before the
    # fix: the provider rejected the response_format payload itself.
    return BackendError(
        provider="openai-chatgpt",
        endpoint="/responses",
        status=400,
        reason="Bad Request",
        headers={},
        body_text=(
            '{"error":{"message":"Missing required parameter: \'text.format.name\'."}}'
        ),
        parsed_error="Missing required parameter: 'text.format.name'.",
        model="gpt-5.5",
        payload_summary=PayloadSummary(
            model="gpt-5.5",
            message_count=2,
            approx_chars=100,
            temperature=0.2,
            has_tools=True,
            tool_choice="auto",
        ),
    )


@dataclass
class _RejectStructuredOutputThenJsonLoop:
    # Raises a structured-output rejection while response_format is set, then
    # returns valid JSON once response_format has been dropped (degraded retry).
    stats: MockStats = field(default_factory=MockStats)

    async def act(
        self, prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | UserMessageEvent, None]:
        if response_format is not None:
            raise _structured_output_rejection()
        yield AssistantEvent(content='{"answer": "42"}', message_id="a1")


@dataclass
class _Always500Loop:
    stats: MockStats = field(default_factory=MockStats)

    async def act(
        self, prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | UserMessageEvent, None]:
        raise BackendError(
            provider="test",
            endpoint="/responses",
            status=500,
            reason="Internal Server Error",
            headers={},
            body_text="boom",
            parsed_error="boom",
            model="m",
            payload_summary=_make_payload_summary(),
        )


def _make_payload_summary() -> PayloadSummary:
    return PayloadSummary(
        model="m",
        message_count=1,
        approx_chars=1,
        temperature=0.2,
        has_tools=False,
        tool_choice=None,
    )


async def test_spawn_agent_degrades_structured_output_on_provider_rejection() -> None:
    # Regression for wf-1: when the provider rejects the response_format payload
    # itself (400 "text.format.name"), the runtime drops response_format and
    # retries on the prompt-level JSON fallback instead of failing the agent.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    calls = [0]

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        calls[0] += 1
        return _RejectStructuredOutputThenJsonLoop(stats=MockStats())

    rt = WorkflowRuntime(agent_loop_factory=factory, schema_retries=2)
    result = await rt.spawn_agent("test", schema=schema)
    assert result == {"answer": "42"}
    # First attempt rejected the format; second (degraded, no response_format)
    # returned valid JSON.
    assert calls[0] == 2


async def test_spawn_agent_does_not_degrade_on_unrelated_backend_error() -> None:
    # A 500 is not a structured-output rejection: it must surface as a real
    # failure, not silently retry without response_format.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    calls = [0]

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        calls[0] += 1
        return _Always500Loop(stats=MockStats())

    rt = WorkflowRuntime(agent_loop_factory=factory, schema_retries=2)
    with pytest.raises(WorkflowError):
        await rt.spawn_agent("test", schema=schema)
    assert calls[0] == 1


async def test_spawn_agent_schema_failure_survives_parallel() -> None:
    # The P0 fix: a schema-exhausted agent must NOT become None in a parallel()
    # batch. The script author should be able to recover the raw response.
    # (Asserted here at the spawn_agent contract level; the parallel() wrapper
    # only swallows exceptions to None, and spawn_agent no longer raises on
    # schema exhaustion, so the structured failure flows through untouched.)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(response_text="not json"), schema_retries=1
    )
    single = await rt.spawn_agent("a", schema=schema)
    assert isinstance(single, SchemaValidationFailure)
    assert single.raw_response == "not json"


async def test_agent_cap_exceeded() -> None:
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_agents=2)
    await rt.spawn_agent("a")
    await rt.spawn_agent("b")
    with pytest.raises(AgentCapExceeded):
        await rt.spawn_agent("c")


async def test_budget_reconciled_after_spawn(runtime: WorkflowRuntime) -> None:
    await runtime.spawn_agent("test")
    snap = runtime._budget.snapshot()
    assert snap.spent == 1500
    assert snap.reserved == 0


async def test_budget_reservation_released_when_config_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _resolve_agent_config runs only on the real-loop path (agent_loop_factory
    # is None), before the per-attempt try that finalizes the agent. A raise
    # there must still reconcile the reservation (otherwise Budget.remaining()
    # is permanently understated) and record the agent as failed, not drop it.
    rt = WorkflowRuntime(agent_loop_factory=None, budget_total=1_000_000)

    def boom(*, agent: str, model: str | None = None) -> Any:
        raise RuntimeError("config load exploded")

    monkeypatch.setattr(rt, "_resolve_agent_config", boom)
    with pytest.raises(RuntimeError, match="config load exploded"):
        await rt.spawn_agent("test")

    assert rt._budget.snapshot().reserved == 0
    assert len(rt._live_agents) == 0
    failed = [
        r for p in rt._phases.values() for r in p.agent_results if not r.completed
    ]
    assert len(failed) == 1


async def test_budget_reservation_released_when_loop_creation_fails() -> None:
    # _create_loop (factory invocation) now runs inside the per-attempt try, so a
    # raise there is finalized instead of escaping and leaking the reservation.
    def raising_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        raise RuntimeError("loop creation exploded")

    rt = WorkflowRuntime(agent_loop_factory=raising_factory, budget_total=1_000_000)
    with pytest.raises(WorkflowError, match="loop creation exploded"):
        await rt.spawn_agent("test")

    assert rt._budget.snapshot().reserved == 0
    assert len(rt._live_agents) == 0
    failed = [
        r for p in rt._phases.values() for r in p.agent_results if not r.completed
    ]
    assert len(failed) == 1


async def test_parallel_returns_results_in_order(runtime: WorkflowRuntime) -> None:
    async def thunk_a() -> str:
        await asyncio.sleep(0.01)
        return "a"

    async def thunk_b() -> str:
        return "b"

    results = await runtime.parallel(thunk_a, thunk_b)
    assert results == ["a", "b"]


async def test_pipeline_returns_results_in_order(runtime: WorkflowRuntime) -> None:
    async def fn(x: int) -> int:
        await asyncio.sleep(0.01 * (3 - x))
        return x * 2

    results = await runtime.pipeline([1, 2, 3], fn)
    assert results == [2, 4, 6]


async def test_parallel_without_await_gives_helpful_error(
    runtime: WorkflowRuntime,
) -> None:
    async def thunk() -> str:
        return "x"

    result = runtime.parallel(thunk)
    try:
        with pytest.raises(TypeError, match="forget 'await'"):
            a, b = result  # type: ignore[misc]
    finally:
        await result  # exhaust the coroutine to avoid RuntimeWarning


async def test_pipeline_without_await_gives_helpful_error(
    runtime: WorkflowRuntime,
) -> None:
    async def fn(x: int) -> int:
        return x

    result = runtime.pipeline([1], fn)
    try:
        with pytest.raises(TypeError, match="forget 'await'"):
            result[0]  # type: ignore[index]
    finally:
        await result  # exhaust the coroutine to avoid RuntimeWarning


def _concurrency_tracking_factory(active: list[int], max_active: list[int]) -> Any:
    @dataclass
    class _TrackingLoop:
        stats: MockStats = field(default_factory=MockStats)

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            try:
                await asyncio.sleep(0.02)
                yield AssistantEvent(content="ok", message_id="a1")
            finally:
                active[0] -= 1

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _TrackingLoop()

    return factory


async def test_parallel_bounds_agent_concurrency() -> None:
    # Concurrency must be bounded by max_concurrent even though parallel() no
    # longer takes the semaphore itself — spawn_agent owns the limit.
    active = [0]
    max_active = [0]
    rt = WorkflowRuntime(
        agent_loop_factory=_concurrency_tracking_factory(active, max_active),
        max_concurrent=2,
        max_agents=100,
    )
    ns = rt.build_script_namespace()
    agent = ns["agent"]
    await rt.parallel(*[(lambda i=i: agent(f"p{i}")) for i in range(8)])
    assert max_active[0] <= 2


async def test_parallel_no_deadlock_when_exceeding_max_concurrent() -> None:
    # Regression: nested semaphore acquisition (parallel + spawn_agent) used to
    # deadlock once the number of agent thunks reached max_concurrent.
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(delay=0.01), max_concurrent=2, max_agents=100
    )
    ns = rt.build_script_namespace()
    agent = ns["agent"]
    results = await asyncio.wait_for(
        rt.parallel(*[(lambda i=i: agent(f"p{i}")) for i in range(8)]), timeout=5.0
    )
    assert results == ["mock response"] * 8


async def test_parallel_max_concurrency_caps_in_flight_below_global() -> None:
    # max_concurrency on parallel() must bound in-flight thunks independently of
    # (and below) the runtime's global max_concurrent. Regression for the
    # hand-rolled waves(tasks, n=3) chunking loop every provider-limited user
    # had to write.
    active = [0]
    max_active = [0]
    rt = WorkflowRuntime(
        agent_loop_factory=_concurrency_tracking_factory(active, max_active),
        max_concurrent=16,  # global cap is loose; the per-call cap must bind
        max_agents=100,
    )
    ns = rt.build_script_namespace()
    agent = ns["agent"]
    await rt.parallel(
        *[(lambda i=i: agent(f"p{i}")) for i in range(8)], max_concurrency=3
    )
    assert max_active[0] <= 3, (
        f"per-call cap should bound concurrency to 3, saw {max_active[0]}"
    )


async def test_pipeline_max_concurrency_caps_in_flight_items() -> None:
    active = [0]
    max_active = [0]
    rt = WorkflowRuntime(
        agent_loop_factory=_concurrency_tracking_factory(active, max_active),
        max_concurrent=16,
        max_agents=100,
    )
    ns = rt.build_script_namespace()
    agent = ns["agent"]

    async def stage(prev, item, index):
        return await agent(item)

    await rt.pipeline([f"p{i}" for i in range(8)], stage, max_concurrency=2)
    assert max_active[0] <= 2, (
        f"per-call cap should bound pipeline concurrency to 2, saw {max_active[0]}"
    )


async def test_parallel_max_concurrency_zero_is_rejected() -> None:
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_agents=100)
    with pytest.raises(WorkflowError, match="must be >= 1"):
        await rt.parallel(lambda: asyncio.sleep(0), max_concurrency=0)


async def test_pipeline_max_concurrency_zero_is_rejected() -> None:
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_agents=100)

    async def stage(prev, item, index):
        return item

    with pytest.raises(WorkflowError, match="must be >= 1"):
        await rt.pipeline([1, 2], stage, max_concurrency=0)


async def test_pipeline_no_deadlock_when_exceeding_max_concurrent() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(delay=0.01), max_concurrent=2, max_agents=100
    )
    ns = rt.build_script_namespace()
    agent = ns["agent"]

    async def fn(i: int) -> str:
        return await agent(f"p{i}")

    results = await asyncio.wait_for(rt.pipeline(list(range(8)), fn), timeout=5.0)
    assert results == ["mock response"] * 8


async def test_response_excludes_prompt_echo_and_reasoning(
    runtime: WorkflowRuntime,
) -> None:
    # The mock act() stream yields UserMessageEvent(prompt) + ReasoningEvent
    # before the AssistantEvent. Only the assistant answer must be returned.
    result = await runtime.spawn_agent("please do the thing")
    assert result == "mock response"


async def test_replan_signal_fires_on_low_success_ratio(
    runtime: WorkflowRuntime,
) -> None:
    from vibe.core.workflows.models import AgentResult

    runtime._set_phase("audit")
    runtime._record_agent_result(
        AgentResult(prompt="p", response="ok", completed=True, phase="audit")
    )
    runtime._record_agent_result(
        AgentResult(
            prompt="p", response="", completed=False, error="boom", phase="audit"
        )
    )
    runtime._record_agent_result(
        AgentResult(
            prompt="p", response="", completed=False, error="boom", phase="audit"
        )
    )

    events: list[str] = []
    runtime.set_event_sink(events.append)
    runtime._set_phase("verify")

    assert any("REPLAN SIGNAL" in e and "audit" in e for e in events), events
    assert any("low success ratio" in e for e in events), events


async def test_replan_signal_fires_on_budget_near_exhausted(
    runtime: WorkflowRuntime,
) -> None:
    from vibe.core.workflows.models import AgentResult

    assert runtime.budget_total is not None
    events: list[str] = []
    runtime.set_event_sink(events.append)
    runtime._set_phase("phase1")
    runtime._record_agent_result(
        AgentResult(prompt="p", response="ok", completed=True, phase="phase1")
    )
    runtime._record_agent_result(
        AgentResult(prompt="p", response="ok", completed=True, phase="phase1")
    )
    runtime._budget.restore_spent(int(runtime.budget_total * 0.9))
    runtime._set_phase("phase2")

    assert any("REPLAN SIGNAL" in e and "phase1" in e for e in events), events
    assert any("budget nearly exhausted" in e for e in events), events


async def test_replan_signal_quiet_on_healthy_phase(runtime: WorkflowRuntime) -> None:
    from vibe.core.workflows.models import AgentResult

    events: list[str] = []
    runtime.set_event_sink(events.append)
    runtime._set_phase("healthy")
    runtime._record_agent_result(
        AgentResult(prompt="p", response="ok", completed=True, phase="healthy")
    )
    runtime._record_agent_result(
        AgentResult(prompt="p", response="ok", completed=True, phase="healthy")
    )
    runtime._set_phase("next")

    assert not any("REPLAN SIGNAL" in e for e in events), events


async def test_replan_signal_skips_phase_with_too_few_results(
    runtime: WorkflowRuntime,
) -> None:
    from vibe.core.workflows.models import AgentResult

    events: list[str] = []
    runtime.set_event_sink(events.append)
    runtime._set_phase("small")
    runtime._record_agent_result(
        AgentResult(
            prompt="p", response="", completed=False, error="boom", phase="small"
        )
    )
    runtime._set_phase("next")

    assert not any("REPLAN SIGNAL" in e for e in events), events


async def test_phase_tracking(runtime: WorkflowRuntime) -> None:
    runtime._set_phase("Find")
    runtime._set_phase("Verify")
    await runtime.spawn_agent("find prompt", phase="Find", label="finder1")
    await runtime.spawn_agent("verify prompt", phase="Verify", label="verifier1")

    run = runtime.build_run()
    assert len(run.phases) == 2
    assert run.phases[0].name == "Find"
    assert run.phases[1].name == "Verify"
    assert len(run.phases[0].agent_results) == 1
    assert run.phases[0].agent_results[0].label == "finder1"


async def test_run_executes_script(runtime: WorkflowRuntime) -> None:
    script = """
async def main():
    phase("Test")
    result = await agent("hello")
    return {"result": result}
"""
    result = await runtime.run(script)
    assert result.run.status.value == "completed"
    assert result.return_value == {"result": "mock response"}
    assert result.run.agent_count == 1


async def test_run_rejects_unsafe_script(runtime: WorkflowRuntime) -> None:
    script = """
import os

async def main():
    os.system("rm -rf /")
"""
    with pytest.raises(WorkflowError, match="validation failed"):
        await runtime.run(script)


async def test_run_rejects_script_without_main(runtime: WorkflowRuntime) -> None:
    script = """
x = 42
"""
    with pytest.raises(WorkflowError, match="main"):
        await runtime.run(script)


async def test_run_captures_script_failure(runtime: WorkflowRuntime) -> None:
    script = """
async def main():
    raise ValueError("boom")
"""
    result = await runtime.run(script)
    assert result.run.status.value == "failed"
    assert "boom" in result.summary


@pytest.mark.asyncio
async def test_run_reports_stopped_not_failed_on_cancellation() -> None:
    # A whole-run stop cancels the task awaiting main(); the run must report
    # STOPPED (not FAILED) and still return a WorkflowResult so the host gets
    # recovered outputs instead of a bare cancel. Regression for wf-2.
    from vibe.core.workflows.runtime import WorkflowRuntime

    started = asyncio.Event()

    class _BlockingLoop:
        async def act(self, prompt, *, response_format=None):
            started.set()
            await asyncio.sleep(60)
            return
            yield  # pragma: no cover

        class stats:
            session_prompt_tokens = 10
            session_completion_tokens = 5

    def _blocking_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        return _BlockingLoop()

    rt = WorkflowRuntime(agent_loop_factory=_blocking_factory)
    script = """
async def main():
    return await agent("x")
"""
    task = asyncio.create_task(rt.run(script))
    await started.wait()
    task.cancel()
    result = await task
    assert result.run.status.value == "stopped"
    assert "stopped" in result.summary.lower()


@pytest.mark.asyncio
async def test_awaiting_log_and_phase_is_not_a_trap() -> None:
    # Regression for wf-1: `await log(...)` / `await phase(...)` used to raise
    # "NoneType can't be used in 'await'" and kill the run with 0 agents. The
    # injected wrappers are now awaitable noops.
    from vibe.core.workflows.runtime import WorkflowRuntime

    def _factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        raise AssertionError("should not be reached")

    rt = WorkflowRuntime(agent_loop_factory=_factory)
    script = """
async def main():
    await log("hi")
    await phase("p")
    return {"ok": True}
"""
    result = await rt.run(script)
    assert result.run.status.value == "completed"
    assert result.return_value == {"ok": True}


async def test_run_with_parallel_agents(runtime: WorkflowRuntime) -> None:
    script = """
async def main():
    phase("Find")
    results = await parallel(
        lambda: agent("find a", label="a", phase="Find"),
        lambda: agent("find b", label="b", phase="Find"),
    )
    return {"count": len(results)}
"""
    result = await runtime.run(script)
    assert result.run.status.value == "completed"
    assert result.return_value == {"count": 2}
    assert result.run.agent_count == 2


async def test_run_with_pipeline(runtime: WorkflowRuntime) -> None:
    script = """
async def main():
    items = ["a", "b", "c"]

    async def verify(item):
        return await agent(f"verify {item}", label=f"v:{item}")

    results = await pipeline(items, verify)
    return {"verified": len(results)}
"""
    result = await runtime.run(script)
    assert result.run.status.value == "completed"
    assert result.return_value == {"verified": 3}
    assert result.run.agent_count == 3


async def test_event_sink_receives_logs(runtime: WorkflowRuntime) -> None:
    messages: list[str] = []
    runtime.set_event_sink(messages.append)
    runtime._set_phase("Test")
    runtime._log("hello")
    assert any("phase: Test" in m for m in messages)
    assert "hello" in messages


async def test_budget_snapshot_in_run(runtime: WorkflowRuntime) -> None:
    script = """
async def main():
    await agent("test1")
    await agent("test2")
    return {}
"""
    result = await runtime.run(script)
    assert result.run.budget.spent == 3000
    assert result.run.budget.reserved == 0


def _raising_factory() -> Any:
    @dataclass
    class _RaisingLoop:
        stats: MockStats = field(default_factory=MockStats)

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            raise RuntimeError("boom from act")
            yield  # pragma: no cover  (makes this an async generator)

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _RaisingLoop()

    return factory


async def test_schemaless_agent_exception_surfaces_real_error() -> None:
    # A schemaless agent whose act() raises must surface the real error, not a
    # misleading SchemaValidationError.
    rt = WorkflowRuntime(
        agent_loop_factory=_raising_factory(), max_agents=10, budget_total=1_000_000
    )
    with pytest.raises(WorkflowError, match="boom from act"):
        await rt.spawn_agent("do it")


async def test_schema_agent_exception_still_raises() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=_raising_factory(), max_agents=10, budget_total=1_000_000
    )
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    with pytest.raises((WorkflowError, SchemaValidationError)):
        await rt.spawn_agent("do it", schema=schema)


async def test_parallel_all_agents_crash_surfaces_failure_in_summary() -> None:
    # parallel() degrades a crashing agent to None rather than failing the run
    # (documented null-on-throw). A batch where EVERY agent crashes must not
    # therefore read as an unqualified success — the summary must say how many
    # agents failed and why, or a systemic failure looks identical to success.
    rt = WorkflowRuntime(
        agent_loop_factory=_raising_factory(), max_agents=10, budget_total=1_000_000
    )
    script = """
async def main():
    phase("audit")
    results = await parallel(
        lambda: agent("a", label="a", phase="audit"),
        lambda: agent("b", label="b", phase="audit"),
    )
    return results
"""
    result = await rt.run(script)
    assert result.run.status.value == "completed_with_failures", (
        "a batch where every agent crashed must not read as unqualified success"
    )
    assert result.return_value == [None, None]
    assert "2/2" in result.summary, "summary must report that all agents failed"
    assert "boom from act" in result.summary, "summary must include the error"


async def test_parallel_partial_failure_surfaces_count_in_summary(
    runtime: WorkflowRuntime,
) -> None:
    # One agent succeeds (default mock factory), one raises — the mixed batch
    # completes but the summary must still flag the single failure.
    calls = {"n": 0}

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            return _raising_factory()(
                prompt, agent=agent, parent_context=parent_context
            )
        return make_factory()(prompt, agent=agent, parent_context=parent_context)

    rt = WorkflowRuntime(
        agent_loop_factory=factory, max_agents=10, budget_total=1_000_000
    )
    script = """
async def main():
    results = await parallel(
        lambda: agent("ok", label="ok"),
        lambda: agent("bad", label="bad"),
    )
    return results
"""
    result = await rt.run(script)
    assert result.run.status.value == "completed_with_failures"
    assert "1/2" in result.summary
    assert "boom from act" in result.summary


async def test_bare_await_agent_failure_preserves_return_value() -> None:
    # A bare await agent() must degrade to None like parallel() does, not null
    # the run's return_value via run()'s broad handler.
    calls = {"n": 0}

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        calls["n"] += 1
        # Calls 1-2 (the parallel assess agents) succeed; call 3 (the bare
        # synthesizer await) raises.
        if calls["n"] == 3:
            return _raising_factory()(
                prompt, agent=agent, parent_context=parent_context
            )
        return make_factory()(prompt, agent=agent, parent_context=parent_context)

    rt = WorkflowRuntime(
        agent_loop_factory=factory, max_agents=10, budget_total=1_000_000
    )
    script = (
        "async def main():\n"
        "    phase('assess')\n"
        "    found = await parallel(\n"
        "        lambda: agent('a', label='a', phase='assess'),\n"
        "        lambda: agent('b', label='b', phase='assess'),\n"
        "    )\n"
        "    phase('synthesize')\n"
        "    report = await agent('synthesize', label='synth', phase='synthesize')\n"
        "    return {'found': found, 'report': report}\n"
    )
    result = await rt.run(script)
    # The bare-await failure degraded to None instead of nulling return_value.
    assert result.return_value is not None, (
        "a bare await agent() failure must not null the whole run's return_value"
    )
    assert result.return_value["found"] == ["mock response", "mock response"]
    assert result.return_value["report"] is None
    assert result.run.status.value == "completed_with_failures"
    assert "1/3" in result.summary, "summary must report the single synthesizer failure"


async def test_live_status_reports_per_phase_failure_breakdown() -> None:
    # live_status() must surface a per-phase completed/failed/failed_details
    # breakdown so observers (workflow_status tool) can see which agents failed
    # without parsing the human-readable summary string. Covers the i2 contract.
    calls = {"n": 0}

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            return _raising_factory()(
                prompt, agent=agent, parent_context=parent_context
            )
        return make_factory()(prompt, agent=agent, parent_context=parent_context)

    rt = WorkflowRuntime(
        agent_loop_factory=factory, max_agents=10, budget_total=1_000_000
    )
    script = """
async def main():
    phase("audit")
    results = await parallel(
        lambda: agent("ok", label="ok", phase="audit"),
        lambda: agent("bad", label="bad", phase="audit"),
    )
    return results
"""
    result = await rt.run(script)
    # The run is over so live_status has no in-flight agents, but the finalized
    # phase breakdown must still reflect 1 succeeded / 1 failed.
    status = rt.live_status()
    audit = next(p for p in status["phases"] if p["name"] == "audit")
    assert audit["agents"] == 2
    assert audit["completed"] == 1
    assert audit["failed"] == 1
    assert len(audit["failed_details"]) == 1
    assert audit["failed_details"][0]["label"] == "bad"
    assert "boom from act" in (audit["failed_details"][0]["error"] or "")
    # And the run-level status must reflect the partial failure.
    assert result.run.status.value == "completed_with_failures"


def _retry_then_succeed_factory(per_attempt_in: int, per_attempt_out: int) -> Any:
    calls = [0]

    @dataclass
    class _RetryLoop:
        stats: MockStats = field(
            default_factory=lambda: MockStats(per_attempt_in, per_attempt_out)
        )
        _text: str = ""

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            calls[0] += 1
            text = "not json" if calls[0] == 1 else '{"a": "ok"}'
            yield AssistantEvent(content=text, message_id="a1")

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _RetryLoop()

    return factory, calls


async def test_retry_tokens_accumulate_across_attempts() -> None:
    factory, calls = _retry_then_succeed_factory(100, 50)
    rt = WorkflowRuntime(
        agent_loop_factory=factory,
        max_agents=10,
        budget_total=1_000_000,
        schema_retries=1,
    )
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    result = await rt.spawn_agent("x", schema=schema)
    assert result == {"a": "ok"}
    assert calls[0] == 2  # one failed attempt + one success
    # Both attempts' tokens counted, not just the last: 2 * (100 + 50).
    assert rt._budget.spent() == 300


async def test_cache_hit_does_not_double_count_tokens(runtime: WorkflowRuntime) -> None:
    await runtime.spawn_agent("same", agent="explore", phase="P")
    await runtime.spawn_agent("same", agent="explore", phase="P")  # cache hit
    run = runtime.build_run()
    results = run.phases[0].agent_results
    assert len(results) == 2
    # First real run records tokens; the cache hit records zero.
    assert results[0].tokens_in == 1000
    assert results[1].tokens_in == 0
    assert results[1].tokens_out == 0


async def test_parent_context_rejects_non_subagent_agent() -> None:
    from vibe.core.agents.models import AgentType
    from vibe.core.tools.base import InvokeContext

    @dataclass
    class _Profile:
        agent_type: AgentType

    class _Manager:
        def get_agent(self, name: str) -> _Profile:
            if name == "auto-approve":
                return _Profile(AgentType.AGENT)
            if name == "explore":
                return _Profile(AgentType.SUBAGENT)
            raise ValueError(name)

    ctx = InvokeContext(
        tool_call_id="wf-tool", agent_manager=cast(AgentManager, _Manager())
    )
    rt = WorkflowRuntime(parent_context=ctx, max_agents=10, budget_total=1_000_000)
    with pytest.raises(WorkflowError, match="Only subagents can be used"):
        await rt.spawn_agent("do anything", agent="auto-approve")


async def test_budget_exposed_to_script_is_read_only(runtime: WorkflowRuntime) -> None:
    ns = runtime.build_script_namespace()
    budget = ns["budget"]
    # Read accessors work.
    assert budget.spent() == 0
    assert budget.total == 1_000_000
    # Mutation is blocked.
    with pytest.raises(AttributeError, match="read-only"):
        budget._spent = 0  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="read-only"):
        budget.total = 999  # type: ignore[misc]
    # The underlying budget is unaffected.
    assert runtime._budget.spent() == 0


async def test_budget_proxy_does_not_expose_live_budget(
    runtime: WorkflowRuntime,
) -> None:
    runtime._budget.restore_spent(900_000)
    ns = runtime.build_script_namespace()
    budget = ns["budget"]
    assert budget.remaining() == 100_000
    # The old bypass attribute is gone.
    assert not hasattr(budget, "_budget")
    # remaining() still reflects the live budget (proxy isn't a stale copy).
    runtime._budget.restore_spent(950_000)
    assert budget.remaining() == 50_000


async def test_pipeline_multi_stage(runtime: WorkflowRuntime) -> None:
    async def double(x: int) -> int:
        return x * 2

    async def add_ten(prev: int, _item: int, _idx: int) -> int:
        return prev + 10

    results = await runtime.pipeline([1, 2, 3], double, add_ten)
    assert results == [12, 14, 16]


async def test_pipeline_stage_receives_prev_item_index(
    runtime: WorkflowRuntime,
) -> None:
    seen: list[tuple] = []

    async def stage1(x: int) -> int:
        return x * 10

    async def stage2(prev: int, item: int, idx: int) -> int:
        seen.append((prev, item, idx))
        return prev

    await runtime.pipeline([5, 6], stage1, stage2)
    assert (50, 5, 0) in seen
    assert (60, 6, 1) in seen


async def test_pipeline_stage_failure_drops_item_to_none(
    runtime: WorkflowRuntime,
) -> None:
    reached_stage2: list[int] = []

    async def stage1(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    async def stage2(prev: int, _item: int, _idx: int) -> int:
        reached_stage2.append(prev)
        return prev

    results = await runtime.pipeline([1, 2, 3], stage1, stage2)
    assert results == [1, None, 3]
    # The failing item never reaches stage2.
    assert 2 not in reached_stage2


async def test_parallel_thunk_failure_yields_none(runtime: WorkflowRuntime) -> None:
    async def ok() -> str:
        return "ok"

    async def boom() -> str:
        raise RuntimeError("nope")

    results = await runtime.parallel(ok, boom, ok)
    assert results == ["ok", None, "ok"]


async def test_parallel_accepts_list_form(runtime: WorkflowRuntime) -> None:
    async def a() -> str:
        return "a"

    async def b() -> str:
        return "b"

    # Claude Code style: parallel([...]) as well as parallel(*...).
    results = await runtime.parallel([a, b])
    assert results == ["a", "b"]


async def test_parallel_reraises_resource_exhaustion(runtime: WorkflowRuntime) -> None:
    from vibe.core.workflows.budget import BudgetExhausted

    async def ok() -> str:
        return "ok"

    async def hit_cap() -> str:
        raise AgentCapExceeded("cap")

    async def over_budget() -> str:
        raise BudgetExhausted("over")

    with pytest.raises(AgentCapExceeded):
        await runtime.parallel(ok, hit_cap)
    with pytest.raises(BudgetExhausted):
        await runtime.parallel(ok, over_budget)


async def test_pipeline_reraises_resource_exhaustion(runtime: WorkflowRuntime) -> None:
    from vibe.core.workflows.budget import BudgetExhausted

    async def hit_cap(_x: int) -> int:
        raise AgentCapExceeded("cap")

    async def over_budget(_x: int) -> int:
        raise BudgetExhausted("over")

    with pytest.raises(AgentCapExceeded):
        await runtime.pipeline([1, 2], hit_cap)
    with pytest.raises(BudgetExhausted):
        await runtime.pipeline([1], over_budget)


async def test_pipeline_rejects_keyword_only_stage(runtime: WorkflowRuntime) -> None:
    async def kw_only(*, prev: int) -> int:
        return prev

    with pytest.raises(WorkflowError, match="positional"):
        runtime.pipeline([1, 2], kw_only)  # raises synchronously at call time


async def test_pipeline_zero_stages_is_passthrough(runtime: WorkflowRuntime) -> None:
    assert await runtime.pipeline([1, 2, 3]) == [1, 2, 3]


async def test_nested_workflow_runs_and_shares_state() -> None:
    """workflow(name) runs another workflow inline on the SAME runtime, so its
    agents share the parent's counter/budget and its result flows back.
    """
    child_src = (
        "async def main():\n"
        "    r = await agent('child task')\n"
        "    return {'child': r}\n"
    )

    def resolver(name: str) -> str | None:
        return child_src if name == "child" else None

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        max_agents=100,
        budget_total=1_000_000,
        workflow_source_resolver=resolver,
    )
    parent_src = (
        "async def main():\n"
        "    sub = await workflow('child')\n"
        "    mine = await agent('parent task')\n"
        "    return {'sub': sub, 'mine': mine}\n"
    )
    result = await rt.run(parent_src)
    assert result.run.status.value == "completed"
    assert result.return_value["sub"] == {"child": "mock response"}
    assert result.return_value["mine"] == "mock response"
    # Shared agent counter: child agent + parent agent.
    assert rt._agent_count == 2


async def test_nested_workflow_one_level_only() -> None:
    grandchild = "async def main():\n    return 1\n"
    child = "async def main():\n    return await workflow('grandchild')\n"

    def resolver(name: str) -> str | None:
        return {"child": child, "grandchild": grandchild}.get(name)

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        workflow_source_resolver=resolver,
    )
    result = await rt.run("async def main():\n    return await workflow('child')\n")
    assert result.run.status.value == "failed"
    assert "one level" in result.summary


async def test_nested_workflow_unknown_name_fails() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        workflow_source_resolver=lambda _n: None,
    )
    result = await rt.run("async def main():\n    return await workflow('nope')\n")
    assert result.run.status.value == "failed"
    assert "Unknown workflow" in result.summary


async def test_nested_workflow_unavailable_without_resolver() -> None:
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    result = await rt.run("async def main():\n    return await workflow('x')\n")
    assert result.run.status.value == "failed"
    assert "not available" in result.summary


async def test_isolated_agent_routes_to_executor() -> None:
    """agent(isolation='worktree') routes to the injectable executor (in
    production: a `vibe -p` subprocess in a fresh worktree) and returns its
    output; it still counts against the agent cap/budget.
    """
    calls: list[tuple] = []

    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        calls.append((prompt, agent, label, max_turns))
        return "isolated result"

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        max_agents=100,
        isolated_executor=stub,
    )
    result = await rt.spawn_agent(
        "do risky thing", agent="default", label="iso1", isolation="worktree"
    )
    assert result == "isolated result"
    assert rt._agent_count == 1
    assert calls and calls[0][1] == "default" and calls[0][2] == "iso1"


async def test_isolated_agent_with_schema_parses_output() -> None:
    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        return '{"ok": true}'

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        isolated_executor=stub,
    )
    assert await rt.spawn_agent("x", schema=schema, isolation="worktree") == {
        "ok": True
    }


async def test_isolated_agent_bad_json_returns_failure() -> None:
    # Default (strict_schema=False): isolated-agent schema exhaustion returns a
    # structured SchemaValidationFailure (mirrors the in-process path) so the
    # raw output is recoverable instead of being lost to None via parallel._safe.
    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        return "not json"

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        isolated_executor=stub,
    )
    result = await rt.spawn_agent("x", schema=schema, isolation="worktree")
    assert isinstance(result, SchemaValidationFailure)
    assert result.raw_response == "not json"


async def test_isolated_agent_bad_json_raises_strict() -> None:
    # strict_schema=True preserves the legacy hard-fail behavior for isolated agents.
    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        return "not json"

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        isolated_executor=stub,
        strict_schema=True,
    )
    with pytest.raises(SchemaValidationError):
        await rt.spawn_agent("x", schema=schema, isolation="worktree")


async def test_isolated_agent_executor_failure_raises_workflow_error() -> None:
    async def boom(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        raise RuntimeError("subprocess died")

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        isolated_executor=boom,
    )
    with pytest.raises(WorkflowError, match="isolated agent failed"):
        await rt.spawn_agent("x", isolation="worktree")


async def test_unknown_isolation_mode_raises(runtime: WorkflowRuntime) -> None:
    with pytest.raises(WorkflowError, match="isolation"):
        await runtime.spawn_agent("x", isolation="container")


async def test_isolated_agent_charges_budget_estimate() -> None:
    """BUDGET-001: isolated agents can't surface real tokens, so they must charge
    the reserved estimate against budget_total (not 0) to keep the cap enforced.
    """

    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        return "done"

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        isolated_executor=stub,
    )
    await rt.spawn_agent("x", isolation="worktree", budget_estimate=12_345)
    assert rt._budget.spent() == 12_345


async def test_isolation_not_cross_cached_with_inprocess() -> None:
    """CACHE-002: an isolated result must not satisfy a later in-process call with
    the same prompt/agent/phase (different execution semantics + accounting).
    """

    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        return "ISOLATED"

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),  # in-process returns "mock response"
        budget_total=1_000_000,
        max_agents=100,
        isolated_executor=stub,
    )
    iso = await rt.spawn_agent("same", agent="explore", phase="P", isolation="worktree")
    inproc = await rt.spawn_agent("same", agent="explore", phase="P")
    assert iso == "ISOLATED"
    assert inproc == "mock response"  # not the cached isolated result
    assert rt._agent_count == 2


async def test_default_isolated_executor_spawns_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    removed: list[Any] = []
    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(
        eph, "remove_ephemeral_worktree", lambda wt, **k: removed.append(wt)
    )

    captured: dict[str, Any] = {}

    class _FakeProc:
        pid = 4242
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"agent output", b"")

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured["argv"] = args
        captured["kwargs"] = kwargs
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    out, stats, _ = await rt._default_isolated_executor(
        "do it", "auto-approve", "lbl", 40
    )

    assert out == "agent output"
    assert stats is None  # no stats line on stderr in this fake
    argv = captured["argv"]
    assert "-p" in argv and "do it" in argv
    assert "auto-approve" in argv and "--trust" in argv and "--max-turns" in argv
    assert captured["kwargs"]["cwd"] == "/tmp/iso-wt"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"].get("VIBE_WORKFLOW_EMIT_STATS") == "1"
    assert removed == [fake_wt]  # worktree cleaned up


async def test_isolated_executor_threads_requested_profile_not_auto_approve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: _spawn_isolated used to hardcode --agent auto-approve,
    # discarding the requested profile. The cmd must carry the real agent.
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)

    captured: list[Any] = []

    class _FakeProc:
        pid = 4242
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"out", b"")

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.append(args)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    await rt._default_isolated_executor(
        "do it", "editor", "lbl", 40
    )  # editor, NOT auto-approve
    argv = captured[0]
    # The requested profile appears right after --agent; auto-approve must not.
    agent_idx = argv.index("--agent")
    assert argv[agent_idx + 1] == "editor"
    assert "auto-approve" not in argv


async def test_isolated_executor_threads_model_flag_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The host's requested subagent model must reach the `vibe -p` child as a
    # --model flag; absent a model it must not appear at all (child inherits its
    # own active_model). This is what makes task(model=...) work for isolated
    # (write-capable) subagents, not just in-process ones.
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)

    captured: list[Any] = []

    class _FakeProc:
        pid = 4242
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"out", b"")

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured.append(args)
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)

    await rt._default_isolated_executor("do it", "editor", "lbl", 40, model="gpt-5.5")
    argv = captured[0]
    model_idx = argv.index("--model")
    assert argv[model_idx + 1] == "gpt-5.5"

    captured.clear()
    await rt._default_isolated_executor("do it", "editor", "lbl", 40)
    assert "--model" not in captured[0]


async def test_isolated_executor_passes_auto_approve_and_worktree_root_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the isolated subprocess used to SKIP every write/edit/bash
    # because the child had no approval callback (the critical flaw). _spawn_isolated
    # must hand the child VIBE_ISOLATED_AUTO_APPROVE (so programmatic wires an
    # auto-yes callback) and VIBE_ISOLATED_WORKTREE_ROOT (so file tools confine
    # themselves to the worktree). Both flow through the child env.
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph
    from vibe.core.worktree.ephemeral import EphemeralWorktree

    fake_wt = EphemeralWorktree(
        path=Path("/tmp/iso-wt-env"),
        branch="iso",
        repo_root=Path("/tmp/repo"),
        base_sha="0" * 40,
    )
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)

    captured: dict[str, Any] = {}

    class _FakeProc:
        pid = 7
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"out", b"")

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured["env"] = kwargs.get("env", {})
        captured["cwd"] = kwargs.get("cwd")
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    await rt._default_isolated_executor("do it", "worker", "lbl", 40)
    env = captured["env"]
    assert env.get("VIBE_ISOLATED_AUTO_APPROVE") == "1"
    assert env.get("VIBE_ISOLATED_WORKTREE_ROOT") == str(fake_wt.path)
    # cwd is the worktree so relative paths resolve inside it.
    assert captured["cwd"] == str(fake_wt.path)


async def test_isolated_executor_scrubs_host_secrets_from_child_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Isolated children must not inherit host git/gh/cloud creds.
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph
    from vibe.core.worktree.ephemeral import EphemeralWorktree

    fake_wt = EphemeralWorktree(
        path=Path("/tmp/iso-wt-scrub"),
        branch="iso",
        repo_root=Path("/tmp/repo"),
        base_sha="0" * 40,
    )
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    monkeypatch.setenv("GH_TOKEN", "ghp_must_not_leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/ssh-agent.sock")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_must_not_leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-keep")

    captured: dict[str, Any] = {}

    class _FakeProc:
        pid = 7
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"out", b"")

    async def fake_exec(*args: Any, **kwargs: Any) -> _FakeProc:
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    await rt._default_isolated_executor("do it", "worker", "lbl", 40)
    env = captured["env"]
    assert "GH_TOKEN" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env.get("OPENAI_API_KEY") == "sk-must-keep"
    assert env.get("VIBE_ISOLATED_AUTO_APPROVE") == "1"


async def test_validate_workflow_profile_rejects_editor_without_isolation() -> None:
    # editor has an enabled_tools allowlist (read/grep/write_file/edit) yet
    # writes files. The old 'no allowlist = isolate' proxy missed it; the
    # generalized predicate must force it to isolation='worktree'.
    from dataclasses import dataclass, field

    from vibe.core.agents.models import AgentType
    from vibe.core.tools.base import InvokeContext

    @dataclass
    class _Profile:
        agent_type: AgentType = AgentType.SUBAGENT
        overrides: dict = field(
            default_factory=lambda: {
                "enabled_tools": ["read", "grep", "write_file", "edit"]
            }
        )

    class _Manager:
        def get_agent(self, name: str) -> _Profile:
            if name == "editor":
                return _Profile()
            raise ValueError(name)

    ctx = InvokeContext(
        tool_call_id="wf-tool", agent_manager=cast(AgentManager, _Manager())
    )
    rt = WorkflowRuntime(parent_context=ctx, max_agents=10, budget_total=1_000_000)
    with pytest.raises(WorkflowError, match="isolation='worktree'"):
        await rt.spawn_agent("refactor foo", agent="editor")


async def test_default_isolated_executor_reaps_and_cleans_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    removed: list[Any] = []
    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt2")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(
        eph, "remove_ephemeral_worktree", lambda wt, **k: removed.append(wt)
    )

    waited = [False]

    class _HangProc:
        pid = 4243
        returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            raise asyncio.CancelledError

        async def wait(self) -> int:
            waited[0] = True
            return -15

    async def fake_exec(*args: Any, **kwargs: Any) -> _HangProc:
        return _HangProc()

    killed: list[int] = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    import os as _os

    monkeypatch.setattr(_os, "killpg", lambda pgid, sig: killed.append(sig))
    monkeypatch.setattr(_os, "getpgid", lambda pid: pid)

    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    with pytest.raises(asyncio.CancelledError):
        await rt._default_isolated_executor("do it", "auto-approve", "lbl", 40)

    assert killed  # process group was signalled
    assert waited[0]  # waited for exit before cleanup (WL-001)
    assert removed == [fake_wt]  # worktree still cleaned up despite cancel


async def test_default_isolated_executor_parses_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor parses the real token-stats line the subprocess emits on
    stderr (GAP #1 — real accounting instead of the estimate).
    """
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt3")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)

    sentinel = '__VIBE_WORKFLOW_STATS__{"prompt_tokens": 111, "completion_tokens": 22}'

    class _P:
        pid = 1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"result", ("some log\n" + sentinel + "\n").encode())

    async def fake_exec(*a: Any, **k: Any) -> _P:
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    out, stats, _ = await rt._default_isolated_executor("p", "auto-approve", "l", 40)
    assert out == "result"
    assert stats == {"prompt_tokens": 111, "completion_tokens": 22}


async def test_run_isolated_agent_redirects_stdout_to_log_file_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # When log_path is set, the child's stdout must be a file handle (tailed live
    # by the background registry) rather than a PIPE, and the final output is
    # read back from that file. Verifies the redirect path used by async
    # isolated subagents (task(async_run=true) + write-capable profile).
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-log-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    monkeypatch.setattr(eph, "deliver_ephemeral_worktree", lambda wt: True)

    captured: dict[str, Any] = {}

    class _P:
        pid = 1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"pipe-bytes-should-be-ignored", b"")

    async def fake_exec(*args: Any, **kwargs: Any) -> _P:
        captured.update(kwargs)
        # Simulate the child writing its stdout to the redirected file handle.
        # _open_isolated_log opened it with "wb" (truncate), so the pre-written
        # fixture content is gone — the child's own writes are what survive.
        stdout_fh = kwargs["stdout"]
        stdout_fh.write(b"log-stdout-output\n")
        stdout_fh.flush()
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    from vibe.core.workflows.runtime import run_isolated_agent

    log_path = tmp_path / "asub-1.log"
    log_path.touch()
    result = await run_isolated_agent(
        "do it", "worker", label="worker", max_turns=5, deliver=True, log_path=log_path
    )
    # stdout kwarg was the file handle, not a PIPE int.
    stdout_kw = captured["stdout"]
    assert not isinstance(stdout_kw, int)
    assert hasattr(stdout_kw, "write")
    # Output came from the log file, not the (ignored) PIPE bytes.
    assert result.output == "log-stdout-output\n"


async def test_run_isolated_agent_uses_pipe_when_log_path_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without log_path, the historical in-memory PIPE capture is preserved.
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    fake_wt = type("WT", (), {"path": Path("/tmp/iso-pipe-wt")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: None)
    monkeypatch.setattr(eph, "deliver_ephemeral_worktree", lambda wt: True)

    captured: dict[str, Any] = {}

    class _P:
        pid = 1
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"pipe-output", b"")

    async def fake_exec(*args: Any, **kwargs: Any) -> _P:
        captured.update(kwargs)
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    from vibe.core.workflows.runtime import run_isolated_agent

    result = await run_isolated_agent(
        "do it", "worker", label="worker", max_turns=5, deliver=True
    )
    assert captured["stdout"] == asyncio.subprocess.PIPE
    assert result.output == "pipe-output"


async def test_isolated_agent_charges_real_tokens_when_stats_present() -> None:
    async def stub(
        prompt: str, agent: str, label: str | None, max_turns: int
    ) -> tuple[str, dict[str, int]]:
        return ("done", {"prompt_tokens": 100, "completion_tokens": 50})

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        isolated_executor=stub,
    )
    await rt.spawn_agent("x", isolation="worktree", budget_estimate=99_999)
    assert rt._budget.spent() == 150  # real tokens, not the 99,999 estimate


# --- G2: live per-agent token tracking ---


def _gated_stats_factory(gate: asyncio.Event, gate_on: int = 1) -> Any:
    """A mock loop whose stats grow mid-stream. The gate_on-th invocation blocks
    on `gate` mid-stream (so a test can observe the live view); other
    invocations complete immediately with the standard 1000/500 totals. This
    lets a single runtime host both a finalized agent and an in-flight one.
    """
    calls = [0]

    @dataclass
    class _GrowingStats:
        session_prompt_tokens: int = 0
        session_completion_tokens: int = 0

    @dataclass
    class _GatedLoop:
        stats: _GrowingStats = field(default_factory=_GrowingStats)
        gated: bool = False

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            if not self.gated:
                self.stats.session_prompt_tokens = 1000
                self.stats.session_completion_tokens = 500
                yield AssistantEvent(content="mock response", message_id="a1")
                return
            # First turn completes (partial tokens), then block until released,
            # then a second turn finalizes the totals.
            self.stats.session_prompt_tokens = 600
            self.stats.session_completion_tokens = 120
            yield AssistantEvent(content="partial", message_id="a1")
            await gate.wait()
            self.stats.session_prompt_tokens = 1000
            self.stats.session_completion_tokens = 500
            yield AssistantEvent(content=" done", message_id="a2")

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        calls[0] += 1
        return _GatedLoop(gated=(calls[0] == gate_on))

    return factory


async def test_live_agent_visible_and_retired_around_execution() -> None:
    """While an agent runs it appears in live_status(); once it finalizes it
    moves to the finalized phases (live XOR finalized, never both).
    """
    gate = asyncio.Event()
    rt = WorkflowRuntime(
        agent_loop_factory=_gated_stats_factory(gate),
        max_agents=10,
        budget_total=1_000_000,
    )

    task = asyncio.create_task(rt.spawn_agent("work", phase="P", label="worker"))
    # Yield control so the agent registers and reaches the gate mid-stream.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    live = rt.live_status()
    assert live["live_agent_count"] == 1, live
    assert live["live_agents"][0]["label"] == "worker"
    assert live["live_agents"][0]["status"] == "running"
    # Partial tokens from the first turn are already visible mid-flight.
    assert live["live_agents"][0]["tokens_in"] == 600
    assert live["tokens_live"] == 720
    # Nothing finalized yet.
    assert live["tokens_finalized"] == 0

    gate.set()
    await task

    live = rt.live_status()
    assert live["live_agent_count"] == 0  # retired after finalize
    assert live["tokens_live"] == 0
    assert live["tokens_finalized"] == 1500  # 1000 + 500
    assert live["phases"] == [
        {
            "name": "P",
            "agents": 1,
            "tokens": 1500,
            "completed": 1,
            "failed": 0,
            "failed_details": [],
        }
    ]


async def test_live_tokens_do_not_double_count_with_finalized(
    runtime: WorkflowRuntime,
) -> None:
    """An agent is counted live XOR finalized. With one finished and one live,
    the live total is the sum without overlap.
    """
    gate = asyncio.Event()
    rt = WorkflowRuntime(
        agent_loop_factory=_gated_stats_factory(gate, gate_on=2),
        max_agents=10,
        budget_total=1_000_000,
    )
    # One fully finalized agent (1500 tokens) — invocation 1, completes at once.
    await rt.spawn_agent("done first", phase="P")
    # One in-flight agent blocked at the gate — invocation 2, gated.
    task = asyncio.create_task(rt.spawn_agent("still running", phase="P", label="run2"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    live = rt.live_status()
    # finalized (1500) + live partial (600+120).
    assert live["tokens_finalized"] == 1500
    assert live["tokens_live"] == 720
    assert live["tokens_total"] == 2220

    gate.set()
    await task


async def test_live_response_preview_visible_mid_run() -> None:
    # i4: response_so_far is populated on _LiveAgent as content streams, and
    # visible in live_status() as response_preview while the agent is running.
    gate = asyncio.Event()
    rt = WorkflowRuntime(
        agent_loop_factory=_gated_stats_factory(gate),
        max_agents=10,
        budget_total=1_000_000,
    )
    task = asyncio.create_task(rt.spawn_agent("work", phase="P", label="worker"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    live = rt.live_status()
    assert live["live_agent_count"] == 1
    # The gated loop yields "partial" before blocking on the gate.
    assert live["live_agents"][0]["response_preview"] == "partial"

    gate.set()
    await task


async def test_agent_timeout_cancels_hung_agent() -> None:
    # i7: agent_timeout_s wraps the agent in a watchdog that cancels it after
    # the timeout. A hung agent (blocks forever on a gate) must be cancelled
    # and recorded as failed, not block the run indefinitely.
    gate = asyncio.Event()

    @dataclass
    class _HungLoop:
        stats: MockStats = field(default_factory=MockStats)

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            yield AssistantEvent(content="starting", message_id="a1")
            await gate.wait()  # never set — simulates a hung agent
            yield AssistantEvent(content="should never reach", message_id="a2")

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _HungLoop()

    rt = WorkflowRuntime(
        agent_loop_factory=factory,
        max_agents=10,
        budget_total=1_000_000,
        agent_timeout_s=0.1,
    )
    # The agent should be cancelled by the watchdog after 0.1s, not block forever.
    result = await rt.spawn_agent("hang", label="hung")
    # A cancelled agent returns partial output (the schemaless cancel path).
    assert "starting" in result
    # The agent must be recorded as failed (not completed).
    phase = next(iter(rt._phases.values()))
    assert len(phase.agent_results) == 1
    assert phase.agent_results[0].completed is False
    assert "timed out" in (phase.agent_results[0].error or "")


async def test_agent_budget_ceiling_cancels_spendy_agent() -> None:
    # i7: agent_budget_ceiling checks per-agent token spend mid-stream and
    # cancels the agent if it exceeds the ceiling. A mock loop that reports
    # high token stats triggers the ceiling and the agent is stopped.
    @dataclass
    class _SpendyStats:
        session_prompt_tokens: int = 100_000
        session_completion_tokens: int = 100_000

    @dataclass
    class _SpendyLoop:
        stats: _SpendyStats = field(default_factory=_SpendyStats)

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            yield AssistantEvent(content="spending tokens", message_id="a1")

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _SpendyLoop()

    rt = WorkflowRuntime(
        agent_loop_factory=factory,
        max_agents=10,
        budget_total=1_000_000,
        agent_budget_ceiling=50_000,  # well below the 200K the mock reports
    )
    await rt.spawn_agent("spend", label="spendy")
    phase = next(iter(rt._phases.values()))
    assert len(phase.agent_results) == 1
    assert phase.agent_results[0].completed is False


async def test_live_status_budget_snapshot(runtime: WorkflowRuntime) -> None:
    await runtime.spawn_agent("a")
    await runtime.spawn_agent("b")
    live = runtime.live_status()
    assert live["budget"]["spent"] == 3000
    assert live["budget"]["total"] == 1_000_000
    assert live["agent_count"] == 2


# --- G4: intra-workflow message board ---


async def test_message_board_roundtrip_in_script(runtime: WorkflowRuntime) -> None:
    script = """
async def main():
    post_message("findings", {"url": "http://x", "risk": "high"})
    post_message("findings", "second finding")
    msgs = fetch_messages("findings")
    return {"count": len(msgs), "first_risk": msgs[0]["risk"], "second": msgs[1]}
"""
    result = await runtime.run(script)
    assert result.run.status.value == "completed"
    assert result.return_value == {
        "count": 2,
        "first_risk": "high",
        "second": "second finding",
    }


async def test_fetch_messages_unknown_channel_is_empty(
    runtime: WorkflowRuntime,
) -> None:
    script = """
async def main():
    return {"n": len(fetch_messages("nope"))}
"""
    result = await runtime.run(script)
    assert result.return_value == {"n": 0}


async def test_message_board_is_shared_across_parallel_agents(
    runtime: WorkflowRuntime,
) -> None:
    """Agents in the same run share one board: each posts to a channel and a
    later fetch sees them all (handoff without a barrier return).
    """
    script = """
async def main():
    async def post_one(i):
        # an agent does its work, then hands off a partial result
        await agent(f"work {i}", phase="Work")
        post_message("results", i)
        return i
    await parallel(*[(lambda i=i: post_one(i)) for i in range(3)])
    return {"results": sorted(fetch_messages("results"))}
"""
    result = await runtime.run(script)
    assert result.run.status.value == "completed"
    assert result.return_value == {"results": [0, 1, 2]}


async def test_board_survives_snapshot_restore(runtime: WorkflowRuntime) -> None:
    # Board channels roundtrip through snapshot/restore so multi-phase
    # scripts that use post_message resume with their handoff data intact.
    runtime._board.post("findings", {"risk": "high"})
    runtime._board.post("findings", "second")
    runtime._board.post("notes", "misc")
    runtime._budget.reserve(100)

    snap = runtime.snapshot("wf-snap", "src")

    fresh = WorkflowRuntime(
        agent_loop_factory=runtime.agent_loop_factory, budget_total=runtime.budget_total
    )
    fresh.restore_from_snapshot(snap)

    findings = fresh._board.fetch("findings")
    notes = fresh._board.fetch("notes")
    assert [m for m in findings if isinstance(m, dict)] == [{"risk": "high"}]
    assert "second" in findings
    assert notes == ["misc"]


class _FakeProfile:
    def __init__(self, overrides: dict[str, Any]) -> None:
        self.overrides = overrides


class _FakeConfig:
    active_model = "x"
    models: ClassVar[list[Any]] = []


class _FakeAgentManager:
    config = _FakeConfig()

    def get_agent(self, name: str) -> _FakeProfile:
        if name == "worker":
            return _FakeProfile({})  # no enabled_tools -> full tools
        return _FakeProfile({"enabled_tools": ["read", "grep"]})


async def test_full_tool_profile_requires_worktree_isolation() -> None:
    """W-001/W-002: a no-allowlist profile (worker) must run isolated — in-process
    it would race the shared tree and its headless ASK tools would auto-skip.
    """
    from vibe.core.tools.base import InvokeContext

    async def stub(p: str, a: str, lbl: str | None, mt: int) -> tuple[str, None]:
        return ("ok", None)

    ctx = InvokeContext(tool_call_id="t", agent_manager=_FakeAgentManager())  # type: ignore[arg-type]
    rt = WorkflowRuntime(
        parent_context=ctx,
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        max_agents=100,
        isolated_executor=stub,
    )
    # worker without isolation -> rejected
    with pytest.raises(WorkflowError, match="isolation='worktree'"):
        await rt.spawn_agent("x", agent="worker")
    # worker WITH isolation -> runs (isolated subprocess)
    assert await rt.spawn_agent("x", agent="worker", isolation="worktree") == "ok"
    # restricted profile (allowlist) is fine in-process
    assert await rt.spawn_agent("y", agent="explore") == "mock response"


async def test_full_tool_guard_noop_without_agent_manager() -> None:
    # No agent_manager (e.g. unit context) -> guard can't resolve, must not fire.
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(), budget_total=1_000_000, max_agents=10
    )
    assert await rt.spawn_agent("z", agent="worker") == "mock response"


# ---------------------------------------------------------------------------
# Pause / unpause gate
# ---------------------------------------------------------------------------


async def test_pause_and_unpause_toggle_is_paused() -> None:
    rt = WorkflowRuntime(agent_loop_factory=make_factory())
    assert rt.is_paused is False

    rt.pause()
    assert rt.is_paused is True
    # While paused, the gate is cleared so spawn_agent would block; the boolean
    # tracks intent for observers without needing to run an agent.
    rt.unpause()
    assert rt.is_paused is False


async def test_paused_run_blocks_new_agents_until_unpaused() -> None:
    # The pause gate sits inside spawn_agent, after the semaphore. We observe
    # whether the agent loop's act() actually begins: it can only start once
    # the gate opens, so act_started proves the gate blocked while paused.
    act_started = asyncio.Event()

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        class Loop:
            stats = MockStats()

            async def act(self, prompt: str, *, response_format: Any = None) -> Any:
                act_started.set()
                yield AssistantEvent(content="mock response", message_id="a1")

        return Loop()

    rt = WorkflowRuntime(
        agent_loop_factory=factory,
        max_concurrent=2,
        max_agents=10,
        budget_total=1_000_000,
    )

    rt.pause()
    task = asyncio.create_task(rt.spawn_agent("p", phase="work"))
    # Let the scheduler run; the paused gate must keep act() from starting.
    await asyncio.sleep(0.05)
    assert not act_started.is_set()

    rt.unpause()
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result == "mock response"
    assert act_started.is_set()


# ---------------------------------------------------------------------------
# Isolated-worker pre-flight safety judge
# ---------------------------------------------------------------------------


class _StubJudge:
    """Stand-in for SafetyJudge with a fixed verdict, recording its calls."""

    def __init__(self, *, safe: bool, reason: str = "stub") -> None:
        from vibe.core.tools.safety_judge import JudgeVerdict

        self.verdict = JudgeVerdict(safe=safe, reason=reason)
        self.calls: list[tuple[str, str, list[str]]] = []

    async def judge(self, tool_name: str, args_repr: str, flagged: list[str]) -> Any:
        self.calls.append((tool_name, args_repr, flagged))
        return self.verdict


class _RecordingApprovalCB:
    """Approval callback that records the judge_note and returns a fixed verdict."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.judge_notes: list[str | None] = []
        self.calls = 0

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        # 5th positional arg is judge_note (see ApprovalCallback signature).
        self.judge_notes.append(args[4] if len(args) >= 5 else kwargs.get("judge_note"))
        return self.response, None, None


async def test_isolated_worker_judge_approved_proceeds() -> None:
    """A worker whose prompt the judge deems safe runs without prompting."""
    from vibe.core.tools.base import InvokeContext

    async def stub(p: str, a: str, lbl: str | None, mt: int) -> tuple[str, None]:
        return ("worker-out", None)

    judge = _StubJudge(safe=True, reason="read-only refactor")
    approval = _RecordingApprovalCB(ApprovalResponse.YES)
    ctx = InvokeContext(
        tool_call_id="t", safety_judge_factory=lambda: judge, approval_callback=approval
    )
    rt = WorkflowRuntime(
        parent_context=ctx,
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        max_agents=100,
        isolated_executor=stub,
    )

    out = await rt.spawn_agent("refactor foo", agent="worker", isolation="worktree")
    assert out == "worker-out"
    assert judge.calls, "worker spawn must be pre-judged"
    # Tool name selects the workflow-aware judge prompt.
    assert judge.calls[0][0] == "launch_workflow"
    assert approval.calls == 0, "no host prompt when the judge approves"


async def test_isolated_worker_judge_deferred_surfaces_to_host() -> None:
    """When the judge defers, the host approval callback is invoked with the
    judge's reason as judge_note, and the user's YES lets the worker proceed.
    """
    from vibe.core.tools.base import InvokeContext

    async def stub(p: str, a: str, lbl: str | None, mt: int) -> tuple[str, None]:
        return ("worker-out", None)

    judge = _StubJudge(safe=False, reason="could force-push")
    approval = _RecordingApprovalCB(ApprovalResponse.YES)
    ctx = InvokeContext(
        tool_call_id="t", safety_judge_factory=lambda: judge, approval_callback=approval
    )
    rt = WorkflowRuntime(
        parent_context=ctx,
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        max_agents=100,
        isolated_executor=stub,
    )

    out = await rt.spawn_agent("git push --force", agent="worker", isolation="worktree")
    assert out == "worker-out"
    assert approval.calls == 1
    assert approval.judge_notes == ["could force-push"], (
        "the judge's deferral reason must reach the host prompt as judge_note"
    )


async def test_isolated_worker_user_denial_raises_workflow_error() -> None:
    """A user denial at the worker-spawn prompt aborts that worker (raises),
    not the whole run — distinct from a hard budget/cap ceiling.
    """
    from vibe.core.tools.base import InvokeContext

    async def stub(p: str, a: str, lbl: str | None, mt: int) -> tuple[str, None]:
        return ("should-not-run", None)

    judge = _StubJudge(safe=False, reason="destructive")
    approval = _RecordingApprovalCB(ApprovalResponse.NO)
    ctx = InvokeContext(
        tool_call_id="t", safety_judge_factory=lambda: judge, approval_callback=approval
    )
    rt = WorkflowRuntime(
        parent_context=ctx,
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        max_agents=100,
        isolated_executor=stub,
    )

    with pytest.raises(WorkflowError, match="denied by user"):
        await rt.spawn_agent("rm -rf /", agent="worker", isolation="worktree")


async def test_isolated_worker_no_judge_factory_skips_pre_judge() -> None:
    """Without a safety_judge_factory on the context, the worker runs unchecked
    (fail open at spawn; the launch-time script judge already ran).
    """
    from vibe.core.tools.base import InvokeContext

    async def stub(p: str, a: str, lbl: str | None, mt: int) -> tuple[str, None]:
        return ("ok", None)

    ctx = InvokeContext(tool_call_id="t")  # no safety_judge_factory
    rt = WorkflowRuntime(
        parent_context=ctx,
        agent_loop_factory=make_factory(),
        budget_total=1_000_000,
        max_agents=100,
        isolated_executor=stub,
    )
    assert await rt.spawn_agent("x", agent="worker", isolation="worktree") == "ok"


# ---------------------------------------------------------------------------
# Per-agent cancel
# ---------------------------------------------------------------------------


async def test_cancel_agent_aborts_one_in_flight_without_killing_others() -> None:
    """cancel_agent() cancels a single live agent's task; siblings keep running
    and the run records the cancelled one as failed.
    """
    import asyncio

    started = asyncio.Event()
    proceed = asyncio.Event()

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        class Loop:
            stats = MockStats()

            async def act(self, prompt: str, *, response_format: Any = None) -> Any:
                started.set()
                try:
                    await asyncio.wait_for(proceed.wait(), timeout=5.0)
                except TimeoutError:
                    pass
                yield AssistantEvent(content="done", message_id="a1")

        return Loop()

    rt = WorkflowRuntime(
        agent_loop_factory=factory,
        max_concurrent=4,
        max_agents=10,
        budget_total=1_000_000,
    )

    async def spawn_one(
        prompt: str,
    ) -> str | dict[str, Any] | SchemaValidationFailure | ContractFailure:
        return await rt.spawn_agent(prompt, agent="explore", phase="work")

    # Launch two agents; both register as live and block on `proceed`.
    task_a = asyncio.create_task(spawn_one("a"))
    task_b = asyncio.create_task(spawn_one("b"))
    # Let both enter act().
    await asyncio.sleep(0.05)
    live_ids = list(rt._live_agents.keys())
    assert len(live_ids) == 2, f"expected 2 live agents, got {live_ids}"

    target = live_ids[0]
    cancelled = rt.cancel_agent(target)
    assert cancelled is True

    # The cancelled agent's task completes (CancelledError swallowed inside
    # _run_agent because cancel_requested is set) and is recorded as failed.
    await asyncio.wait_for(task_a, timeout=2.0) if task_a.get_coro() else None
    # The other agent is still live and unaffected.
    remaining = list(rt._live_agents.keys())
    assert len(remaining) == 1, "only the targeted agent should be cancelled"
    assert remaining[0] != target

    # Release the survivor and let it finish.
    proceed.set()
    await asyncio.wait_for(
        asyncio.gather(task_a, task_b, return_exceptions=True), timeout=2.0
    )


async def test_cancel_agent_unknown_or_finished_returns_false() -> None:
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_agents=10)
    assert rt.cancel_agent("la-nope") is False
    # A completed agent isn't live anymore.
    await rt.spawn_agent("done", agent="explore")
    assert rt.cancel_agent("la-0") is False


async def test_worker_spawn_args_forbids_extra_fields() -> None:
    # Regression guard: model_config was briefly a raw dict with no extra=,
    # silently allowing typos in the worker-spawn approval payload.
    _WorkerSpawnArgs.model_validate({"prompt": "p", "agent": "worker"})
    # label is now a real optional field; an unknown key must still be rejected.
    _WorkerSpawnArgs.model_validate({"prompt": "p", "agent": "worker", "label": "ok"})
    with pytest.raises(ValidationError):
        _WorkerSpawnArgs.model_validate({
            "prompt": "p",
            "agent": "worker",
            "lbel": "typo",
        })


def test_create_real_loop_caps_turns() -> None:
    from unittest.mock import MagicMock, patch

    from vibe.core.workflows._limits import DEFAULT_ISOLATED_MAX_TURNS
    from vibe.core.workflows.runtime import WorkflowRuntime

    rt = WorkflowRuntime(agent_loop_factory=None)
    with patch("vibe.core.agent_loop.AgentLoop") as mock_loop:
        rt._create_real_loop(agent="explore", base_config=MagicMock())
        params = mock_loop.call_args.kwargs.get("params")
        assert params is not None
        assert params.max_turns == DEFAULT_ISOLATED_MAX_TURNS
