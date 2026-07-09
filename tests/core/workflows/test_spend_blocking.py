from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest

from vibe.core.types import AssistantEvent
from vibe.core.usage._context import (
    SpendAmount,
    SpendPurpose,
    SpendRejection,
    SpendRejectionReason,
)
from vibe.core.usage._session import SpendBudgetExceededError
from vibe.core.workflows.models import WorkflowRunSnapshot, WorkflowStatus
from vibe.core.workflows.runtime import WorkflowRuntime

pytestmark = pytest.mark.asyncio


@dataclass
class _Stats:
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0


@dataclass
class _RaisingLoop:
    error: Exception
    stats: _Stats = field(default_factory=_Stats)

    async def act(
        self, _prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[AssistantEvent, None]:
        raise self.error
        yield AssistantEvent(content="unreachable")

    async def aclose(self) -> None:
        return None


def _spend_error() -> SpendBudgetExceededError:
    rejection = SpendRejection(
        call_id="workflow-denied",
        scope_id="agent:workflow",
        purpose=SpendPurpose.WORKFLOW,
        estimate=SpendAmount(prompt_tokens=1),
        is_retry=False,
        reason=SpendRejectionReason.CALLS,
        limited_scope_id="session:test",
        timestamp=1.0,
    )
    return SpendBudgetExceededError(rejection)


def _runtime(
    error: Exception, calls: list[int] | None = None, *, budget_total: int = 100_000
) -> WorkflowRuntime:
    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        if calls is not None:
            calls.append(1)
        return _RaisingLoop(error=error)

    return WorkflowRuntime(
        agent_loop_factory=factory,
        max_agents=4,
        budget_total=budget_total,
        schema_retries=2,
    )


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    ],
)
async def test_direct_agent_preserves_spend_rejection(schema: dict | None) -> None:
    calls: list[int] = []
    runtime = _runtime(_spend_error(), calls)
    agent = runtime.build_script_namespace()["agent"]

    with pytest.raises(SpendBudgetExceededError) as exc_info:
        await agent("denied", schema=schema)

    assert exc_info.value.rejection.reason is SpendRejectionReason.CALLS
    assert len(calls) == 1
    failed = runtime.build_run().phases[0].agent_results[0]
    assert failed.completed is False
    assert "session:test" in (failed.error or "")


@pytest.mark.parametrize(
    "body",
    [
        'return await agent("denied")',
        'return await parallel(lambda: agent("denied"))',
        (
            "async def stage(item):\n"
            "        return await agent(item)\n"
            '    return await pipeline(["denied"], stage)'
        ),
    ],
    ids=["direct", "parallel", "pipeline"],
)
async def test_spend_rejection_blocks_workflow(body: str) -> None:
    runtime = _runtime(_spend_error())
    script = f"async def main():\n    {body}\n"

    result = await runtime.run(script)

    assert result.run.status is WorkflowStatus.BLOCKED
    assert result.run.status not in {
        WorkflowStatus.COMPLETED,
        WorkflowStatus.COMPLETED_WITH_FAILURES,
    }
    assert result.return_value is None
    assert "Workflow blocked" in result.summary
    assert "session:test" in result.summary


@pytest.mark.parametrize("mode", ["parallel", "pipeline"])
async def test_spend_rejection_cancels_running_siblings(mode: str) -> None:
    runtime = WorkflowRuntime(max_agents=4, budget_total=100_000)
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow(_item: str = "slow") -> str:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "late mutation"

    async def denied(_item: str = "denied") -> str:
        await started.wait()
        raise _spend_error()

    try:
        with pytest.raises(SpendBudgetExceededError):
            if mode == "parallel":
                await runtime.parallel(slow(), denied())
            else:

                async def stage(item: str) -> str:
                    return await (slow(item) if item == "slow" else denied(item))

                await runtime.pipeline(["slow", "denied"], stage)

        assert cancelled.is_set()
    finally:
        release.set()
        await asyncio.sleep(0)


async def test_blocked_status_survives_snapshot_round_trip() -> None:
    runtime = _runtime(_spend_error())
    script = 'async def main():\n    return await agent("denied")\n'
    result = await runtime.run(script)

    snapshot = runtime.snapshot("wf-blocked", script, return_value=result.return_value)
    restored = WorkflowRunSnapshot.model_validate_json(snapshot.model_dump_json())

    assert snapshot.status is WorkflowStatus.BLOCKED
    assert restored.status is WorkflowStatus.BLOCKED


async def test_isolated_agent_preserves_spend_rejection() -> None:
    async def denied_executor(
        prompt: str, agent: str, label: str | None, max_turns: int
    ) -> str:
        raise _spend_error()

    runtime = WorkflowRuntime(
        isolated_executor=denied_executor, max_agents=4, budget_total=100_000
    )
    agent = runtime.build_script_namespace()["agent"]

    with pytest.raises(SpendBudgetExceededError):
        await agent("denied", isolation="worktree")

    failed = runtime.build_run().phases[0].agent_results[0]
    assert failed.completed is False
    assert runtime.budget_snapshot().reserved == 0


async def test_local_workflow_budget_exhaustion_is_blocked() -> None:
    calls: list[int] = []
    runtime = _runtime(RuntimeError("factory must not run"), calls, budget_total=1)
    script = 'async def main():\n    return await agent("denied", budget_estimate=2)\n'

    result = await runtime.run(script)

    assert result.run.status is WorkflowStatus.BLOCKED
    assert result.return_value is None
    assert calls == []
    assert "Cannot reserve 2" in result.summary


async def test_non_budget_agent_failure_keeps_legacy_outcome() -> None:
    runtime = _runtime(RuntimeError("ordinary failure"))
    script = 'async def main():\n    return await agent("fails")\n'

    result = await runtime.run(script)

    assert result.run.status is WorkflowStatus.COMPLETED_WITH_FAILURES
    assert result.return_value is None
    assert "ordinary failure" in result.summary
