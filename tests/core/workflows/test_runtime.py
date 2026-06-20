from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest

from vibe.core.types import AssistantEvent, ReasoningEvent, UserMessageEvent
from vibe.core.workflows.runtime import AgentCapExceeded, WorkflowError, WorkflowRuntime
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
    assert result["answer"] == "42"
    assert call_idx[0] == 2


async def test_spawn_agent_schema_raises_after_max_retries() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(response_text="not json"), schema_retries=1
    )
    with pytest.raises(SchemaValidationError):
        await rt.spawn_agent("test", schema=schema)


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


async def test_phase_tracking(runtime: WorkflowRuntime) -> None:
    runtime._declare_phase("Find")
    runtime._declare_phase("Verify")
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
    runtime._declare_phase("Test")
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
    assert result.run.status.value == "completed"  # null-on-throw: script succeeded
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
            return _raising_factory()(prompt, agent=agent, parent_context=parent_context)
        return make_factory()(prompt, agent=agent, parent_context=parent_context)

    rt = WorkflowRuntime(agent_loop_factory=factory, max_agents=10, budget_total=1_000_000)
    script = """
async def main():
    results = await parallel(
        lambda: agent("ok", label="ok"),
        lambda: agent("bad", label="bad"),
    )
    return results
"""
    result = await rt.run(script)
    assert result.run.status.value == "completed"
    assert "1/2" in result.summary
    assert "boom from act" in result.summary


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

    ctx = InvokeContext(tool_call_id="wf-tool", agent_manager=_Manager())
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

    async def stage2(prev: int, item: int, idx: int) -> tuple:
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
    agents share the parent's counter/budget and its result flows back."""
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
    output; it still counts against the agent cap/budget."""
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
        agent_loop_factory=make_factory(), budget_total=1_000_000, isolated_executor=stub
    )
    assert await rt.spawn_agent("x", schema=schema, isolation="worktree") == {"ok": True}


async def test_isolated_agent_bad_json_raises() -> None:
    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        return "not json"

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(), budget_total=1_000_000, isolated_executor=stub
    )
    with pytest.raises(WorkflowError):
        await rt.spawn_agent("x", schema=schema, isolation="worktree")


async def test_isolated_agent_executor_failure_raises_workflow_error() -> None:
    async def boom(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        raise RuntimeError("subprocess died")

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(), budget_total=1_000_000, isolated_executor=boom
    )
    with pytest.raises(WorkflowError, match="isolated agent failed"):
        await rt.spawn_agent("x", isolation="worktree")


async def test_unknown_isolation_mode_raises(runtime: WorkflowRuntime) -> None:
    with pytest.raises(WorkflowError, match="isolation"):
        await runtime.spawn_agent("x", isolation="container")


async def test_isolated_agent_charges_budget_estimate() -> None:
    """BUDGET-001: isolated agents can't surface real tokens, so they must charge
    the reserved estimate against budget_total (not 0) to keep the cap enforced."""
    async def stub(prompt: str, agent: str, label: str | None, max_turns: int) -> str:
        return "done"

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(), budget_total=1_000_000, isolated_executor=stub
    )
    await rt.spawn_agent("x", isolation="worktree", budget_estimate=12_345)
    assert rt._budget.spent() == 12_345


async def test_isolation_not_cross_cached_with_inprocess() -> None:
    """CACHE-002: an isolated result must not satisfy a later in-process call with
    the same prompt/agent/phase (different execution semantics + accounting)."""
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
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: removed.append(wt))

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
    out, stats = await rt._default_isolated_executor("do it", "auto-approve", "lbl", 40)

    assert out == "agent output"
    assert stats is None  # no stats line on stderr in this fake
    argv = captured["argv"]
    assert "-p" in argv and "do it" in argv
    assert "auto-approve" in argv and "--trust" in argv and "--max-turns" in argv
    assert captured["kwargs"]["cwd"] == "/tmp/iso-wt"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"].get("VIBE_WORKFLOW_EMIT_STATS") == "1"
    assert removed == [fake_wt]  # worktree cleaned up


async def test_default_isolated_executor_reaps_and_cleans_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    import vibe.core.worktree.ephemeral as eph

    removed: list[Any] = []
    fake_wt = type("WT", (), {"path": Path("/tmp/iso-wt2")})()
    monkeypatch.setattr(eph, "create_ephemeral_worktree", lambda *a, **k: fake_wt)
    monkeypatch.setattr(eph, "remove_ephemeral_worktree", lambda wt, **k: removed.append(wt))

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
    stderr (GAP #1 — real accounting instead of the estimate)."""
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
    out, stats = await rt._default_isolated_executor("p", "auto-approve", "l", 40)
    assert out == "result"
    assert stats == {"prompt_tokens": 111, "completion_tokens": 22}


async def test_isolated_agent_charges_real_tokens_when_stats_present() -> None:
    async def stub(
        prompt: str, agent: str, label: str | None, max_turns: int
    ) -> tuple[str, dict[str, int]]:
        return ("done", {"prompt_tokens": 100, "completion_tokens": 50})

    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(), budget_total=1_000_000, isolated_executor=stub
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
    assert live["phases"] == [{"name": "P", "agents": 1, "tokens": 1500}]


async def test_live_tokens_do_not_double_count_with_finalized(runtime: WorkflowRuntime) -> None:
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


async def test_fetch_messages_unknown_channel_is_empty(runtime: WorkflowRuntime) -> None:
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


class _FakeProfile:
    def __init__(self, overrides: dict[str, Any]) -> None:
        self.overrides = overrides


class _FakeConfig:
    active_model = "x"
    models: list[Any] = []


class _FakeAgentManager:
    config = _FakeConfig()

    def get_agent(self, name: str) -> _FakeProfile:
        if name == "worker":
            return _FakeProfile({})  # no enabled_tools -> full tools
        return _FakeProfile({"enabled_tools": ["read", "grep"]})


async def test_full_tool_profile_requires_worktree_isolation() -> None:
    """W-001/W-002: a no-allowlist profile (worker) must run isolated — in-process
    it would race the shared tree and its headless ASK tools would auto-skip."""
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
