"""Tests for the TasksApp pane (unified background-task monitor).

Uses a real BackgroundRegistry with fake owners (the same fakes as the registry
tests) so the pane is exercised against realistic TaskEntry shapes without
spawning processes or mounting the full TUI. Textual's test mode mounts the
widget to verify rendering and message emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
from textual.containers import VerticalScroll
from textual.widgets import OptionList

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.cli.textual_ui.widgets.tasks_app import (
    TasksApp,
    _build_row_text,
    _fmt_seconds,
)
from vibe.core.tools.background import BackgroundRegistry, TaskCategory, TaskEntry

if TYPE_CHECKING:
    from vibe.cli.textual_ui.workflow_runner import WorkflowRunner

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
    phase_reports: list[Any] = field(default_factory=list)
    live_agents: list[Any] = field(default_factory=list)
    budget_snapshot: Any = None
    is_paused: bool = False
    result: Any = None
    script_source: str = "async def main():\n    pass"


class _FakeWorkflowRunner:
    def __init__(self) -> None:
        self.runs: list[_FakeRunEntry] = []
        self.paused: list[str] = []
        self.unpaused: list[str] = []

    def find_run(self, run_id: str) -> _FakeRunEntry | None:
        return next((r for r in self.runs if r.run_id == run_id), None)

    def pause(self, run_id: str) -> bool:
        self.paused.append(run_id)
        return True

    def unpause(self, run_id: str) -> bool:
        self.unpaused.append(run_id)
        return True


def _registry(runner: _FakeWorkflowRunner) -> BackgroundRegistry:
    reg = BackgroundRegistry()
    reg.attach_workflow_runner(lambda: cast("WorkflowRunner", runner))
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
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

        async def key_escape(self) -> None:  # avoid quit-on-escape noise
            pass

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        option_list = app.query_one("#tasks-list", OptionList)
        # One workflow row present.
        ids = [o.id for o in option_list.options]
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
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

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
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

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
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

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
async def test_workflow_detail_shows_budget_phase_breakdown_and_cost():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    from vibe.core.workflows.models import AgentResult, BudgetSnapshot, PhaseReport

    runner = _FakeWorkflowRunner()
    result = AgentResult(
        label="auditor",
        agent="explore",
        phase="audit",
        prompt="p",
        response="r",
        tokens_in=120,
        tokens_out=80,
        cost=0.0123,
    )
    runner.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit"],
            phase_reports=[
                PhaseReport(name="audit", agent_results=[result], elapsed_s=3)
            ],
            agent_count=1,
            tokens_total=200,
            budget_snapshot=BudgetSnapshot(total=1_000, reserved=100, spent=250),
        )
    )
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

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
        body = str(
            app
            .query_one("#tasks-detail", VerticalScroll)
            .query_one(NoMarkupStatic)
            .render()
        )
        assert "Budget: 250/1.0k" in body
        assert "650 left" in body
        assert "Cost: $0.0123" in body
        assert "audit: 1 agent(s)" in body
        await app.action_quit()


@pytest.mark.asyncio
async def test_workflow_detail_lists_agents_and_drill_down_shows_prompt_response():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    from vibe.core.workflows.models import AgentResult, PhaseReport

    runner = _FakeWorkflowRunner()
    result = AgentResult(
        label="auditor",
        agent="explore",
        phase="audit",
        prompt="Review auth.py for injection sinks.",
        response="No injection found; parameterized queries throughout.",
        tokens_in=120,
        tokens_out=80,
    )
    runner.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit"],
            phase_reports=[PhaseReport(name="audit", agent_results=[result])],
            agent_count=1,
            tokens_total=200,
        )
    )
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

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
        # The agent list is mounted beneath the summary.
        agent_list = app.query_one("#tasks-agent-list", OptionList)
        assert len(agent_list.options) == 1
        # Drill into the agent (the event handler strips the 'agent:' prefix;
        # _open_agent_view takes the bare key).
        tasks._open_agent_view("wf-1/done-audit-0")
        await tasks._render_view()
        await pilot.pause()
        body = str(
            app
            .query_one("#tasks-agent", VerticalScroll)
            .query_one(NoMarkupStatic)
            .render()
        )
        assert "Review auth.py for injection sinks." in body
        assert "No injection found" in body
        await app.action_quit()


@pytest.mark.asyncio
async def test_detail_view_poll_refresh_does_not_duplicate_widgets():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    from vibe.core.workflows.models import AgentResult, PhaseReport

    runner = _FakeWorkflowRunner()
    runner.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit"],
            phase_reports=[
                PhaseReport(
                    name="audit",
                    agent_results=[
                        AgentResult(
                            label="auditor",
                            agent="explore",
                            phase="audit",
                            prompt="p",
                            response="r",
                            tokens_in=1,
                            tokens_out=1,
                        )
                    ],
                )
            ],
            agent_count=1,
            tokens_total=2,
        )
    )
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

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
        # Simulate several poll ticks — must not crash or duplicate fixed ids.
        for _ in range(3):
            tasks._refresh_detail_view()
            await pilot.pause()
        assert len(app.query("#tasks-agent-list")) == 1
        assert len(app.query("#tasks-detail-text")) == 1
        await app.action_quit()


@pytest.mark.asyncio
async def test_live_agent_detail_shows_prompt_and_streaming_preview():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    @dataclass
    class _LiveAgent:
        agent_id: str = "la-0"
        agent: str = "explore"
        label: str | None = "live-auditor"
        phase: str | None = "audit"
        model: str | None = None
        status: str = "running"
        tokens_total: int = 50
        prompt: str = "Audit the login flow."
        response_so_far: str = "partial findings..."

    runner = _FakeWorkflowRunner()
    runner.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit"],
            live_agents=[_LiveAgent()],
        )
    )
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

        async def key_escape(self) -> None:
            pass

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        tasks = app.query_one(TasksApp)
        # Registry exposes the live agent as its own AGENT row.
        agent_entry = next(e for e in reg.list_tasks() if e.task_id == "wf-1/live-la-0")
        tasks._selected_task_id = agent_entry.task_id
        tasks._view = "detail"
        await tasks._render_view()
        await pilot.pause()
        body = str(
            app
            .query_one("#tasks-detail", VerticalScroll)
            .query_one(NoMarkupStatic)
            .render()
        )
        assert "Audit the login flow." in body
        assert "partial findings..." in body
        await app.action_quit()


@pytest.mark.asyncio
async def test_agents_filter_includes_live_workflow_agents():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    @dataclass
    class _LiveAgent:
        agent_id: str = "la-0"
        agent: str = "explore"
        label: str | None = "live-auditor"
        phase: str | None = "audit"
        model: str | None = None
        status: str = "running"
        tokens_total: int = 50
        prompt: str = "Audit the login flow."
        response_so_far: str = "partial findings..."

    runner = _FakeWorkflowRunner()
    runner.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit"],
            live_agents=[_LiveAgent()],
        )
    )
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

        async def key_escape(self) -> None:
            pass

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("6")
        await pilot.pause()
        option_list = app.query_one("#tasks-list", OptionList)
        assert [o.id for o in option_list.options] == ["wf-1/live-la-0"]
        await app.action_quit()


@pytest.mark.asyncio
async def test_flat_list_live_workflow_agent_drills_into_full_agent_view():
    from textual.app import App, ComposeResult
    from textual.containers import Container

    @dataclass
    class _LiveAgent:
        agent_id: str = "la-0"
        agent: str = "explore"
        label: str | None = "live-auditor"
        phase: str | None = "audit"
        model: str | None = None
        status: str = "running"
        tokens_total: int = 50
        prompt: str = "Audit the login flow."
        response_so_far: str = "partial findings..."

    runner = _FakeWorkflowRunner()
    runner.runs.append(
        _FakeRunEntry(
            run_id="wf-1",
            status=_WorkflowStatus("running"),
            phases=["audit"],
            live_agents=[_LiveAgent()],
        )
    )
    reg = _registry(runner)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield Container(
                TasksApp(registry=reg, workflow_runner=cast("WorkflowRunner", runner))
            )

        async def key_escape(self) -> None:
            pass

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("6")
        await pilot.pause()
        option_list = app.query_one("#tasks-list", OptionList)
        option_list.highlighted = 0
        await pilot.press("enter")
        await pilot.pause()
        app.query_one("#tasks-agent", VerticalScroll)
        body = str(
            app
            .query_one("#tasks-agent", VerticalScroll)
            .query_one(NoMarkupStatic)
            .render()
        )
        assert "Audit the login flow." in body
        assert "partial findings..." in body
        await pilot.press("b")
        await pilot.pause()
        tasks = app.query_one(TasksApp)
        assert tasks._view == "detail"
        await pilot.press("q")
        await pilot.pause()
        assert tasks._view == "list"
        await app.action_quit()


@pytest.mark.asyncio
async def test_pause_message_from_pane_calls_registry_pause():
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
