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

    def factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
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
        agent_loop_factory=make_factory(delay=0.01),
        max_concurrent=2,
        max_agents=100,
    )
    ns = rt.build_script_namespace()
    agent = ns["agent"]
    results = await asyncio.wait_for(
        rt.parallel(*[(lambda i=i: agent(f"p{i}")) for i in range(8)]),
        timeout=5.0,
    )
    assert results == ["mock response"] * 8


async def test_pipeline_no_deadlock_when_exceeding_max_concurrent() -> None:
    rt = WorkflowRuntime(
        agent_loop_factory=make_factory(delay=0.01),
        max_concurrent=2,
        max_agents=100,
    )
    ns = rt.build_script_namespace()
    agent = ns["agent"]

    async def fn(i: int) -> str:
        return await agent(f"p{i}")

    results = await asyncio.wait_for(
        rt.pipeline(list(range(8)), fn), timeout=5.0
    )
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
