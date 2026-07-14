from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents.manager import AgentManager
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.manager import ToolManager
from vibe.core.types import AssistantEvent as MockEvent
from vibe.core.workflows import _cache_identity
from vibe.core.workflows.models import (
    WorkflowLaneExpectation,
    WorkflowRunSnapshot,
    WorkflowStatus,
)
from vibe.core.workflows.runtime import WorkflowError, WorkflowRuntime

pytestmark = pytest.mark.asyncio
_TRUSTED_DEPENDENCIES = "a" * 64


@pytest.fixture(autouse=True)
def _stable_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _cache_identity, "_repository_fingerprint", lambda: "stable-workspace"
    )


def _runtime(**kwargs: Any) -> WorkflowRuntime:
    config = build_test_vibe_config()
    agent_manager = AgentManager(lambda: config)
    context = InvokeContext(
        tool_call_id="workflow-cache-test",
        active_model=config.active_model,
        agent_manager=agent_manager,
        tool_manager=ToolManager(lambda: config, defer_mcp=True),
    )
    kwargs.setdefault("trusted_cache_dependency_fingerprint", _TRUSTED_DEPENDENCIES)
    return WorkflowRuntime(parent_context=context, **kwargs)


@dataclass
class MockStats:
    session_prompt_tokens: int = 100
    session_completion_tokens: int = 50


@dataclass
class MockAgentLoop:
    response_text: str = "ok"
    stats: MockStats = field(default_factory=MockStats)
    call_count: int = field(default=0, init=False)

    async def act(
        self, prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[MockEvent, None]:
        self.call_count += 1
        yield MockEvent(content=self.response_text)


def make_factory(response_text: str = "ok") -> Any:
    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return MockAgentLoop(response_text=response_text)

    return factory


async def test_cache_hit_skips_agent_run() -> None:
    call_count = [0]

    def counting_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class CountingLoop:
            response_text: str = "ok"
            stats: MockStats = field(default_factory=MockStats)

            async def act(
                self, prompt: str, *, response_format: Any = None
            ) -> AsyncGenerator[MockEvent, None]:
                call_count[0] += 1
                yield MockEvent(content=self.response_text)

        return CountingLoop()

    rt = _runtime(agent_loop_factory=counting_factory, max_concurrent=2)

    result1 = await rt.spawn_agent("same prompt", agent="explore")
    assert result1 == "ok"
    assert call_count[0] == 1

    result2 = await rt.spawn_agent("same prompt", agent="explore")
    assert result2 == "ok"
    assert call_count[0] == 1

    assert len(rt._cache) == 1


async def test_different_prompts_not_cached() -> None:
    call_count = [0]

    def counting_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class CountingLoop:
            response_text: str = "ok"
            stats: MockStats = field(default_factory=MockStats)

            async def act(
                self, prompt: str, *, response_format: Any = None
            ) -> AsyncGenerator[MockEvent, None]:
                call_count[0] += 1
                yield MockEvent(content=self.response_text)

        return CountingLoop()

    rt = _runtime(agent_loop_factory=counting_factory)

    await rt.spawn_agent("prompt A", agent="explore")
    await rt.spawn_agent("prompt B", agent="explore")
    assert call_count[0] == 2
    assert len(rt._cache) == 2


async def test_different_agents_not_cached() -> None:
    call_count = [0]

    def counting_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class CountingLoop:
            response_text: str = "ok"
            stats: MockStats = field(default_factory=MockStats)

            async def act(
                self, prompt: str, *, response_format: Any = None
            ) -> AsyncGenerator[MockEvent, None]:
                call_count[0] += 1
                yield MockEvent(content=self.response_text)

        return CountingLoop()

    rt = _runtime(agent_loop_factory=counting_factory)

    await rt.spawn_agent("same prompt", agent="explore")
    await rt.spawn_agent("same prompt", agent="planner")
    assert call_count[0] == 2


async def test_snapshot_captures_cache() -> None:
    rt = _runtime(agent_loop_factory=make_factory(), max_concurrent=2)
    await rt.spawn_agent("prompt 1", agent="explore", label="a")
    await rt.spawn_agent("prompt 2", agent="explore", label="b")

    snap = rt.snapshot("wf-1", "script", args=None)
    assert snap.run_id == "wf-1"
    assert snap.cached_count == 2
    assert snap.status == WorkflowStatus.PAUSED


async def test_strategy_bound_snapshot_records_lanes_and_rejects_restore() -> None:
    lanes = (WorkflowLaneExpectation(label="audit", profile="reviewer"),)
    rt = _runtime(agent_loop_factory=make_factory(), expected_lanes=lanes, max_agents=1)
    snap = rt.snapshot("wf-bound", "script")

    assert snap.expected_lanes == lanes
    restored = WorkflowRunSnapshot.model_validate_json(snap.model_dump_json())
    assert restored.expected_lanes == lanes

    resumed = _runtime(agent_loop_factory=make_factory())
    with pytest.raises(WorkflowError, match="cannot be resumed safely"):
        resumed.restore_from_snapshot(restored)


async def test_restore_from_snapshot_populates_cache() -> None:
    rt1 = _runtime(agent_loop_factory=make_factory(), max_concurrent=2)
    await rt1.spawn_agent("cached prompt", agent="explore", label="cached")
    snap = rt1.snapshot("wf-1", "script")

    call_count = [0]

    def counting_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class CountingLoop:
            response_text: str = "should not be called"
            stats: MockStats = field(default_factory=MockStats)

            async def act(
                self, prompt: str, *, response_format: Any = None
            ) -> AsyncGenerator[MockEvent, None]:
                call_count[0] += 1
                yield MockEvent(content=self.response_text)

        return CountingLoop()

    rt2 = _runtime(agent_loop_factory=counting_factory, max_concurrent=2)
    rt2.restore_from_snapshot(snap)

    result = await rt2.spawn_agent("cached prompt", agent="explore")
    assert result == "ok"
    assert call_count[0] == 0


async def test_restore_misses_when_dependency_fingerprint_changes() -> None:
    rt1 = _runtime(agent_loop_factory=make_factory())
    await rt1.spawn_agent("cached prompt", agent="explore")
    snap = rt1.snapshot("wf-1", "script")
    call_count = [0]

    def counting_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class CountingLoop:
            stats: MockStats = field(default_factory=MockStats)

            async def act(
                self, prompt: str, *, response_format: Any = None
            ) -> AsyncGenerator[MockEvent, None]:
                call_count[0] += 1
                yield MockEvent(content="fresh")

        return CountingLoop()

    rt2 = _runtime(
        agent_loop_factory=counting_factory,
        trusted_cache_dependency_fingerprint="b" * 64,
    )
    rt2.restore_from_snapshot(snap)

    assert await rt2.spawn_agent("cached prompt", agent="explore") == "fresh"
    assert call_count[0] == 1


async def test_resume_replays_cached_and_runs_rest() -> None:
    call_count = [0]

    def counting_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class CountingLoop:
            stats: MockStats = field(default_factory=MockStats)

            async def act(
                self, prompt: str, *, response_format: Any = None
            ) -> AsyncGenerator[MockEvent, None]:
                call_count[0] += 1
                yield MockEvent(content=f"result-{call_count[0]}")

        return CountingLoop()

    rt1 = _runtime(agent_loop_factory=counting_factory, max_concurrent=2)
    await rt1.spawn_agent("cached prompt", agent="explore")
    snap = rt1.snapshot("wf-1", "script")

    rt2 = _runtime(agent_loop_factory=counting_factory, max_concurrent=2)
    rt2.restore_from_snapshot(snap)

    cached_result = await rt2.spawn_agent("cached prompt", agent="explore")
    assert cached_result == "result-1"

    new_result = await rt2.spawn_agent("new prompt", agent="explore")
    assert new_result == "result-2"

    assert call_count[0] == 2


async def test_cached_result_recorded_in_phases() -> None:
    rt = _runtime(agent_loop_factory=make_factory(), max_concurrent=2)
    await rt.spawn_agent("prompt", agent="explore", label="first", phase="Find")
    await rt.spawn_agent("prompt", agent="explore", label="second", phase="Find")

    run = rt.build_run()
    assert len(run.phases[0].agent_results) == 2
    assert run.phases[0].agent_results[0].label == "first"
    assert run.phases[0].agent_results[1].label == "first"
    assert "[cached]" in run.phases[0].agent_results[1].prompt


async def test_snapshot_serializes_to_json() -> None:
    rt = _runtime(agent_loop_factory=make_factory())
    await rt.spawn_agent("prompt", agent="explore")

    snap = rt.snapshot("wf-1", "script source", args={"key": "value"})
    json_str = snap.model_dump_json()
    assert "wf-1" in json_str
    assert "script source" in json_str
    assert "paused" in json_str

    restored = WorkflowRunSnapshot.model_validate_json(json_str)
    assert restored.run_id == "wf-1"
    assert restored.cached_count == 1
    assert restored.status == WorkflowStatus.PAUSED


async def test_restore_from_snapshot_restores_budget_spend() -> None:
    # Resuming must not silently reset the budget cap to 0 (which would let the
    # resumed run overspend).
    rt1 = _runtime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    await rt1.spawn_agent("p1", agent="explore")
    spent = rt1._budget.spent()
    assert spent > 0
    snap = rt1.snapshot("wf-1", "script")
    assert snap.budget_spent == spent

    rt2 = _runtime(agent_loop_factory=make_factory(), budget_total=1_000_000)
    assert rt2._budget.spent() == 0
    rt2.restore_from_snapshot(snap)
    assert rt2._budget.spent() == spent
    assert rt2._budget.remaining() == 1_000_000 - spent
