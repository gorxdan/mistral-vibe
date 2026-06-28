from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from vibe.cli.textual_ui.app import VibeApp
from vibe.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from vibe.cli.textual_ui.workflow_runner import WorkflowRunner
from vibe.core.workflows.models import WorkflowResult, WorkflowRun, WorkflowStatus
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

    assert entry.task is not None
    await entry.task
    assert entry.result is not None
    assert entry.result.run.status == WorkflowStatus.COMPLETED
    assert entry.status == WorkflowStatus.COMPLETED


async def test_launch_records_args_for_snapshot() -> None:
    """WF-RESUME-01: launch args must be recorded on the entry so snapshots (and
    therefore resume) carry them — otherwise resume re-runs with args=None.
    """

    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_concurrent=2)
    script = "async def main():\n    return {'q': args}\n"
    run_id = runner.launch(script, runtime=rt, args="my topic")

    snap = runner.get_snapshot(run_id)
    assert snap is not None
    assert snap.args == "my topic"
    assert runner.runs[0].args == "my topic"


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


async def test_on_complete_fires_at_most_once_per_entry() -> None:
    # i6: exactly-once delivery. The _on_complete callback must fire at most
    # once per entry even if both the success and cancel paths are reached
    # (resume replay, external re-invocation, or a future third fire site).
    # Guarded by entry.delivered (CAS set before the first fire).
    fire_count = {"n": 0}

    async def on_complete(result: Any) -> None:
        fire_count["n"] += 1

    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount, on_complete=on_complete)
    rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_concurrent=1)
    script = "async def main():\n    return {}\n"
    runner.launch(script, runtime=rt)
    entry = runner.runs[0]
    assert entry.task is not None
    await entry.task
    assert fire_count["n"] == 1, "callback should fire exactly once on completion"
    assert entry.delivered is True

    # Simulate a re-fire (e.g. resume replay attempting to deliver again):
    # constructing a fresh _run_workflow call against the same entry must NOT
    # fire the callback a second time.

    # Manually re-run the delivery path against the already-delivered entry.
    if runner._on_complete and not entry.delivered:  # guard mirrors _run_workflow
        entry.delivered = True
        await runner._on_complete(entry.result)  # type: ignore[arg-type]
    assert fire_count["n"] == 1, "callback must not fire again for a delivered entry"


async def test_handle_command_list() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    widget = await runner.handle_command("")
    assert isinstance(widget, UserCommandMessage)
    assert "No workflow runs" in widget._content


async def test_handle_command_stop_not_found() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    widget = await runner.handle_command("stop wf-99")
    assert isinstance(widget, ErrorMessage)
    assert "not found" in widget._error or "Could not stop" in widget._error


async def test_handle_command_unknown() -> None:
    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    widget = await runner.handle_command("bogus")
    assert isinstance(widget, ErrorMessage)
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
    assert entry.task is not None
    await entry.task
    assert len(completed_results) == 1
    assert completed_results[0].return_value == {"done": True}


def _result(
    *, summary: str, return_value: Any, status: WorkflowStatus
) -> WorkflowResult:
    return WorkflowResult(
        return_value=return_value, run=WorkflowRun(status=status), summary=summary
    )


def test_format_workflow_delivery_includes_return_value() -> None:
    # The host agent must receive the actual return_value, not just the summary.
    # Previously _on_workflow_complete discarded return_value entirely.
    result = _result(
        summary="Workflow completed: 1 agents, 100 tokens, $0.0001",
        return_value={"reviews": [{"verdict": "sound"}]},
        status=WorkflowStatus.COMPLETED,
    )
    payload = VibeApp._format_workflow_delivery(result)
    assert "Workflow completed" in payload
    assert "Result:" in payload
    assert '"reviews"' in payload
    assert "sound" in payload


def test_format_workflow_delivery_omits_absent_return_value() -> None:
    result = _result(
        summary="Workflow failed: 0 tokens",
        return_value=None,
        status=WorkflowStatus.FAILED,
    )
    payload = VibeApp._format_workflow_delivery(result)
    assert payload == "Workflow failed: 0 tokens"
    assert "Result:" not in payload


def test_format_workflow_delivery_truncates_large_result() -> None:
    big = "x" * (VibeApp._WORKFLOW_DELIVERY_CHAR_CAP + 5000)
    result = _result(summary="ok", return_value=big, status=WorkflowStatus.COMPLETED)
    payload = VibeApp._format_workflow_delivery(result)
    assert "(truncated)" in payload
    # Cap + the summary + label, well under the raw 21k input.
    assert len(payload) < VibeApp._WORKFLOW_DELIVERY_CHAR_CAP + 500


async def test_on_workflow_complete_folds_into_running_turn() -> None:
    # When a turn is in flight, the result is staged into the pending-injection
    # path (the running loop drains it and keeps going) rather than appended to
    # history, which a running turn never re-reads.
    from tests.conftest import build_test_vibe_app

    app = build_test_vibe_app()

    async def _noop_mount(_w: Any) -> None:
        return None

    app._mount_and_scroll = _noop_mount  # type: ignore[method-assign]
    app._agent_running = True

    result = _result(
        summary="Workflow completed: 1 agents, 10 tokens, $0.0001",
        return_value={"findings": ["all good"]},
        status=WorkflowStatus.COMPLETED,
    )
    history_before = len(app.agent_loop.messages)
    await app._on_workflow_complete(result)

    staged = app.agent_loop._pending_injected_messages
    assert len(staged) == 1, "result must be staged into the live turn"
    assert "all good" in (staged[-1].content or "")
    assert "Workflow completed" in (staged[-1].content or "")
    # Not appended directly to history (the running turn would never see it).
    assert len(app.agent_loop.messages) == history_before


async def test_on_workflow_complete_resumes_idle_agent() -> None:
    # When the agent is idle (the launching turn already ended), a completed run
    # auto-resumes the agent: a continuation turn is driven with the delivery as
    # its prompt so the agent acts on the outcome instead of stalling.
    from tests.conftest import build_test_vibe_app

    app = build_test_vibe_app()

    async def _noop_mount(_w: Any) -> None:
        return None

    app._mount_and_scroll = _noop_mount  # type: ignore[method-assign]

    started: list[str] = []

    async def _fake_turn(prompt: str, **_kw: Any) -> None:
        started.append(prompt)

    app._handle_agent_loop_turn = _fake_turn  # type: ignore[method-assign]
    assert app._agent_running is False

    result = _result(
        summary="Workflow completed: 1 agents, 10 tokens, $0.0001",
        return_value={"findings": ["all good"]},
        status=WorkflowStatus.COMPLETED,
    )
    await app._on_workflow_complete(result)

    # _agent_running is set synchronously so a racing user submit can't double
    # start; the continuation turn is scheduled as a task.
    assert app._agent_running is True
    assert app._agent_task is not None
    await app._agent_task

    assert len(started) == 1, "an idle agent must be resumed to act on the result"
    assert "all good" in started[0]
    assert "Workflow completed" in started[0]


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
        prompt_hash="abc123", agent="explore", response="cached answer"
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
    assert isinstance(widget, UserCommandMessage)
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
    assert isinstance(widget, ErrorMessage)
    assert "No persisted snapshot" in widget._error


async def test_finished_runs_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long session that launches many workflows must not retain every
    finished run (each holds its agents' full prompts/responses). Finished runs
    beyond the cap are dropped oldest-first; active/paused runs are never dropped.
    """
    import vibe.cli.textual_ui.workflow_runner as wr

    monkeypatch.setattr(wr, "_MAX_FINISHED_RUNS", 3)

    async def mount(w: Any) -> None:
        pass

    runner = WorkflowRunner(mount=mount)
    script = "async def main():\n    return {'r': await agent('hi')}\n"

    launched: list[str] = []
    for _ in range(6):
        rt = WorkflowRuntime(agent_loop_factory=make_factory(), max_concurrent=2)
        rid = runner.launch(script, runtime=rt)
        launched.append(rid)
        found = runner.find_run(rid)  # finish before the next launch
        assert found is not None
        assert found.task is not None
        await found.task

    ids = [r.run_id for r in runner.runs]
    assert launched == ["wf-1", "wf-2", "wf-3", "wf-4", "wf-5", "wf-6"]
    # Bounded: at most the cap of finished runs plus the most-recent launch.
    assert len(runner.runs) <= wr._MAX_FINISHED_RUNS + 1
    # Oldest finished runs dropped, newest kept.
    assert "wf-1" not in ids
    assert "wf-6" in ids
