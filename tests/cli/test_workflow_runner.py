from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from vibe.cli.textual_ui.workflow_runner import WorkflowRunner
from vibe.core.workflows.models import WorkflowStatus
from vibe.core.workflows.runtime import WorkflowRuntime

pytestmark = pytest.mark.asyncio


@dataclass
class MockStats:
    session_prompt_tokens: int = 100
    session_completion_tokens: int = 50


@dataclass
class MockEvent:
    content: str | None = None


@dataclass
class MockAgentLoop:
    response_text: str = "ok"
    stats: MockStats = field(default_factory=MockStats)

    async def act(self, prompt: str, *, response_format: Any = None):
        yield MockEvent(content=self.response_text)


def make_factory(response_text: str = "ok") -> Any:
    def factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> MockAgentLoop:
        return MockAgentLoop(response_text=response_text)

    return factory


async def test_launch_and_complete() -> None:
    mounted: list[Any] = []

    async def mount(w: Any) -> None:
        mounted.append(w)

    runner = WorkflowRunner(mount=mount)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_concurrent=2)

    script = """
async def main():
    result = await agent("hello")
    return {"result": result}
"""
    run_id = runner.launch(script, runtime=rt)
    assert run_id == "wf-1"

    await asyncio.sleep(0.1)
    entry = runner.runs[0]
    assert entry.run_id == "wf-1"

    await entry.task
    assert entry.result is not None
    assert entry.result.run.status == WorkflowStatus.COMPLETED
    assert entry.status == WorkflowStatus.COMPLETED


async def test_list_runs() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    assert runner.active_runs == []
    assert runner.completed_runs == []


async def test_stop_run() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)

    def slow_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class SlowLoop:
            stats: MockStats = field(default_factory=MockStats)

            async def act(self, prompt: str, *, response_format: Any = None):
                await asyncio.sleep(10)
                yield MockEvent(content="slow")

        return SlowLoop()

    rt = WorkflowRuntime(agent_loop_factory=slow_factory, max_concurrent=1)
    script = """
async def main():
    await agent("slow")
    return {}
"""
    run_id = runner.launch(script, runtime=rt)
    await asyncio.sleep(0.05)

    stopped = await runner.stop(run_id)
    assert stopped is True

    entry = runner.runs[0]
    assert entry.error == "Cancelled"


async def test_handle_command_list() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    widget = await runner.handle_command("")
    assert "No workflow runs" in widget._content


async def test_handle_command_stop_not_found() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    widget = await runner.handle_command("stop wf-99")
    assert "not found" in widget._error or "Could not stop" in widget._error


async def test_handle_command_unknown() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    widget = await runner.handle_command("bogus")
    assert "Unknown" in widget._error


async def test_multiple_runs_increment_id() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    rt1 = WorkflowRuntime(agent_loop_factory=make_factory())
    rt2 = WorkflowRuntime(agent_loop_factory=make_factory())

    script = """
async def main():
    return {}
"""
    id1 = runner.launch(script, runtime=rt1)
    id2 = runner.launch(script, runtime=rt2)
    assert id1 == "wf-1"
    assert id2 == "wf-2"

    await asyncio.sleep(0.1)
    for entry in runner.runs:
        if entry.task and not entry.task.done():
            await entry.task


async def test_on_complete_callback() -> None:
    async def mount(w: Any) -> None:
        pass

    completed_results: list[Any] = []

    async def on_complete(result: Any) -> None:
        completed_results.append(result)

    runner = WorkflowRunner(mount=mount, on_complete=on_complete)
    rt = WorkflowRuntime(agent_loop_factory=make_factory())

    script = """
async def main():
    return {"done": True}
"""
    runner.launch(script, runtime=rt)
    await asyncio.sleep(0.1)
    entry = runner.runs[0]
    await entry.task
    assert len(completed_results) == 1
    assert completed_results[0].return_value == {"done": True}


async def test_persist_callback_fires_on_cancel() -> None:
    """WF-3: snapshots must persist on cancel/exit, not only on completion.

    Previously _run_workflow called the persist callback only on the success
    path; cancel/exit raised before persisting, so interrupted runs were never
    snapshotted. The fix moves persistence into a finally block.
    """
    persist_count = 0

    async def mount(w: Any) -> None:
        pass

    async def persist() -> None:
        nonlocal persist_count
        persist_count += 1

    runner = WorkflowRunner(mount=mount, persist_callback=persist)

    def slow_factory(
        prompt: str, *, agent: str, parent_context: Any | None = None
    ) -> Any:
        @dataclass
        class SlowLoop:
            stats: MockStats = field(default_factory=MockStats)

            async def act(self, prompt: str, *, response_format: Any = None):
                await asyncio.sleep(10)
                yield MockEvent(content="slow")

        return SlowLoop()

    rt = WorkflowRuntime(agent_loop_factory=slow_factory, max_concurrent=1)
    script = """
async def main():
    await agent("slow")
    return {}
"""
    run_id = runner.launch(script, runtime=rt)
    await asyncio.sleep(0.05)

    stopped = await runner.stop(run_id)
    assert stopped is True
    # The cancel path must have persisted a snapshot.
    assert persist_count >= 1, "persist callback must fire on cancel"


async def test_resume_reads_persisted_snapshot_and_restores_cache() -> None:
    """WF-2: resume must read a persisted snapshot back and restore cached
    results. Previously snapshots were write-only and resume() had no
    production caller, so cross-session resume was dead code.
    """
    from vibe.core.workflows.models import (
        CachedAgentResult,
        WorkflowRunSnapshot,
        WorkflowStatus,
    )

    async def mount(w: Any) -> None:
        pass

    cached_script = """
async def main():
    r = await agent("hello")
    return {"r": r}
"""
    cached = CachedAgentResult(
        prompt_hash="abc123",
        agent="explore",
        response="cached answer",
    )
    snapshot = WorkflowRunSnapshot(
        run_id="wf-1",
        script_source=cached_script,
        args=None,
        status=WorkflowStatus.PAUSED,
        budget_total=1_000_000,
        budget_spent=1500,
        cached_results=[cached],
    )
    persisted: list[dict[str, Any]] = [snapshot.model_dump(mode="json")]

    runner = WorkflowRunner(
        mount=mount,
        snapshot_loader=lambda: persisted,
        resume_runtime_factory=lambda: WorkflowRuntime(
            agent_loop_factory=make_factory(), max_concurrent=2
        ),
    )

    widget = await runner.handle_command("resume wf-1")
    assert "Resumed workflow `wf-1`" in widget._content
    assert runner.runs, "resumed run must be tracked"
    new_entry = runner.runs[-1]
    # The resumed runtime restored the cached result from the snapshot.
    assert "abc123" in new_entry.runtime._cache, "cached results must be restored"


async def test_resume_without_snapshot_is_an_error() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(
        mount=mount,
        snapshot_loader=lambda: [],
        resume_runtime_factory=lambda: WorkflowRuntime(
            agent_loop_factory=make_factory()
        ),
    )
    widget = await runner.handle_command("resume wf-missing")
    assert "No persisted snapshot" in widget._error
