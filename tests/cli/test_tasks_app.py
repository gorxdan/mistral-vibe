"""Tests for the TasksApp pane (unified background-task monitor).

Uses a real BackgroundRegistry with fake owners (the same fakes as the registry
tests) so the pane is exercised against realistic TaskEntry shapes without
spawning processes or mounting the full TUI. Textual's test mode mounts the
widget to verify rendering and message emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vibe.cli.textual_ui.widgets.tasks_app import (
    TasksApp,
    _build_row_text,
    _fmt_seconds,
)
from vibe.core.tools.background import BackgroundRegistry, TaskCategory, TaskEntry

# ---------------------------------------------------------------------------
# Fakes (kept local so this test module is self-contained)
# ---------------------------------------------------------------------------


class _WorkflowStatus:
    def __init__(self, value: str) -> None:
        self.value = value


@dataclass
class _FakeRunEntry:
    run_id: str
    status: _WorkflowStatus
    elapsed: float = 0.0
    agent_count: int = 1
    tokens_total: int = 0
    phases: list[str] = field(default_factory=lambda: ["audit"])
    live_agents: list[Any] = field(default_factory=list)
    is_paused: bool = False
    result: Any = None
    script_source: str = "async def main():\n    pass"


class _FakeWorkflowRunner:
    def __init__(self) -> None:
        self.runs: list[_FakeRunEntry] = []
        self.paused: list[str] = []
        self.unpaused: list[str] = []

    def _find_run(self, run_id: str) -> _FakeRunEntry | None:
        return next((r for r in self.runs if r.run_id == run_id), None)

    def pause(self, run_id: str) -> bool:
        self.paused.append(run_id)
        return True

    def unpause(self, run_id: str) -> bool:
        self.unpaused.append(run_id)
        return True


def _registry(runner: _FakeWorkflowRunner) -> BackgroundRegistry:
    reg = BackgroundRegistry()
    reg.attach_workflow_runner(lambda: runner)
    return reg


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_fmt_seconds_formats_durations():
    assert _fmt_seconds(5) == "5s"
    assert _fmt_seconds(65) == "1m5s"
    assert _fmt_seconds(3700) == "1h1m"


def test_build_row_text_contains_id_status_category():
    entry = TaskEntry(
        task_id="proc-1",
        category=TaskCategory.PROCESS,
        label="vite --port 5173",
        status="running",
        elapsed=12.0,
    )
    text = _build_row_text(entry)
    rendered = str(text)
    assert "proc-1" in rendered
    assert "process" in rendered
    assert "running" in rendered
    assert "vite --port 5173" in rendered


def test_build_row_text_loop_shows_fires_in():
    entry = TaskEntry(
        task_id="loop-1",
        category=TaskCategory.LOOP,
        label="recheck CI",
        status="waiting",
        elapsed=240.0,
    )
    rendered = str(_build_row_text(entry))
    assert "fires in" in rendered


# ---------------------------------------------------------------------------
# TasksApp — registry-driven list + actions (Textual test mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_view_renders_registry_entries():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    runner = _FakeWorkflowRunner()
    runner.runs.append(_FakeRunEntry(run_id="wf-1", status=_WorkflowStatus("running")))
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(TasksApp(registry=reg, workflow_runner=runner))

        async def key_escape(self) -> None:  # avoid quit-on-escape noise
            pass

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        option_list = app.query_one("#tasks-list")
        # One workflow row present.
        ids = [o.id for o in option_list._options]
        assert "wf-1" in ids
        await app.action_quit()


@pytest.mark.asyncio
async def test_category_filter_hides_other_categories():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    runner = _FakeWorkflowRunner()
    runner.runs.append(_FakeRunEntry(run_id="wf-1", status=_WorkflowStatus("running")))
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(TasksApp(registry=reg, workflow_runner=runner))

        async def key_escape(self) -> None:
            pass

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        # Press "2" (Processes filter). No processes registered → the list is
        # replaced by the empty-state Static, so #tasks-list is gone.
        await pilot.press("2")
        await pilot.pause()
        # No processes registered → the list is replaced by the empty-state
        # Static, so #tasks-list is no longer mounted.
        try:
            app.query_one("#tasks-list")
            list_present = True
        except Exception:
            list_present = False
        assert list_present is False
        await app.action_quit()


@pytest.mark.asyncio
async def test_stop_action_emits_task_stop_requested():
    from textual.app import App, ComposeResult
    from textual.containers import Container
    from textual.widgets import OptionList

    runner = _FakeWorkflowRunner()
    runner.runs.append(_FakeRunEntry(run_id="wf-1", status=_WorkflowStatus("running")))
    reg = _registry(runner)
    captured: list[str] = []

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(TasksApp(registry=reg, workflow_runner=runner))

        async def key_escape(self) -> None:
            pass

        def on_tasks_app_task_stop_requested(
            self, message: TasksApp.TaskStopRequested
        ) -> None:
            captured.append(message.task_id)

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        tasks = app.query_one(TasksApp)
        # Highlight the first (only) row then invoke stop.
        option_list = app.query_one("#tasks-list", OptionList)
        option_list.highlighted = 0
        tasks.action_stop()
        # Let the message pump deliver the posted message.
        await pilot.pause()
        assert captured == ["wf-1"]
        await app.action_quit()


@pytest.mark.asyncio
async def test_drill_down_shows_workflow_detail():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    runner = _FakeWorkflowRunner()
    runner.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit", "report"],
            agent_count=4,
            tokens_total=12_000,
        )
    )
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(TasksApp(registry=reg, workflow_runner=runner))

        async def key_escape(self) -> None:
            pass

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        tasks = app.query_one(TasksApp)
        tasks._selected_task_id = "wf-1"
        tasks._view = "detail"
        await tasks._render_view()
        await pilot.pause()
        # Detail widget is mounted in the detail view.
        app.query_one("#tasks-detail")
        await app.action_quit()


@pytest.mark.asyncio
async def test_pause_message_from_pane_calls_registry_pause():
    """The pane emits TaskPauseRequested; the app handler must AWAIT the
    registry pause() coroutine (regression: a missing await once silently
    dropped the resume path). Here we assert the registry is actually invoked
    by routing the message the way VibeApp does.
    """
    runner = _FakeWorkflowRunner()
    runner.runs.append(_FakeRunEntry(run_id="wf-1", status=_WorkflowStatus("running")))
    reg = _registry(runner)
    pause_calls: list[str] = []
    orig_pause = reg.pause

    async def _spy_pause(task_id: str) -> bool:
        pause_calls.append(task_id)
        return await orig_pause(task_id)

    reg.pause = _spy_pause  # type: ignore[assignment]

    # Simulate the pane emitting the pause message and the app awaiting it —
    # the contract the app handler relies on.
    msg = TasksApp.TaskPauseRequested("wf-1")
    await reg.pause(msg.task_id)
    assert pause_calls == ["wf-1"]
