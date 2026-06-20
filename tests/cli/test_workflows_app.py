from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vibe.cli.textual_ui.widgets.workflows_app import (
    WorkflowsApp,
    _format_tokens,
    _truncate,
)
from vibe.core.workflows.models import (
    AgentResult,
    BudgetSnapshot,
    PhaseReport,
    WorkflowResult,
    WorkflowRun,
    WorkflowStatus,
)

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self) -> None:
        assert _truncate("hello", 80) == "hello"

    def test_long_text_truncated(self) -> None:
        long = "x" * 100
        result = _truncate(long, 80)
        assert len(result) == 80
        assert result.endswith("\u2026")

    def test_newlines_replaced_with_spaces(self) -> None:
        assert _truncate("line1\nline2", 80) == "line1 line2"

    def test_stripped(self) -> None:
        assert _truncate("  hello  ", 80) == "hello"


class TestFormatTokens:
    def test_small_number(self) -> None:
        assert _format_tokens(42) == "42"

    def test_thousands(self) -> None:
        assert _format_tokens(1500) == "1.5k"

    def test_millions(self) -> None:
        assert _format_tokens(2_500_000) == "2.5M"

    def test_zero(self) -> None:
        assert _format_tokens(0) == "0"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    label: str | None = None
    phase: str | None = None
    prompt: str = "test prompt"
    response: str | dict[str, Any] = "test response"
    tokens_in: int = 100
    tokens_out: int = 50
    cost: float = 0.01
    completed: bool = True
    error: str | None = None

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class FakePhaseReport:
    name: str = "default"
    agent_results: list[Any] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def tokens_total(self) -> int:
        return sum(r.tokens_total for r in self.agent_results)

    @property
    def cost_total(self) -> float:
        return sum(r.cost for r in self.agent_results)


@dataclass
class FakeBudget:
    total: int | None = 100_000

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(total=self.total, reserved=0, spent=5000)


@dataclass
class FakeRuntime:
    _phases: dict[str, FakePhaseReport] = field(default_factory=dict)
    _phase_order: list[str] = field(default_factory=list)
    _budget: FakeBudget = field(default_factory=FakeBudget)
    _agent_count: int = 0
    _live_agents: dict[str, Any] = field(default_factory=dict)
    is_paused: bool = False


@dataclass
class FakeLiveAgent:
    agent_id: str = "la-0"
    agent: str = "explore"
    label: str | None = "live-1"
    phase: str | None = "research"
    model: str | None = None
    status: str = "running"
    tokens_in: int = 300
    tokens_out: int = 120
    error: str | None = None

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class FakeRunEntry:
    run_id: str = "wf-1"
    script_source: str = "async def main():\n    return {}"
    started_at: float = 0.0
    runtime: Any = None
    task: Any = None
    result: Any = None
    error: str | None = None
    _phase_reports: list[Any] = field(default_factory=list)
    _budget_snapshot: BudgetSnapshot = field(
        default_factory=lambda: BudgetSnapshot(total=100_000, reserved=0, spent=5000)
    )
    _live_agents_override: list[Any] | None = None

    @property
    def status(self) -> WorkflowStatus:
        if self.result is not None:
            return self.result.run.status
        if self.error is not None:
            return WorkflowStatus.FAILED
        if self.task is not None and self.task.done():
            return WorkflowStatus.FAILED
        if self.runtime is not None and getattr(self.runtime, "is_paused", False):
            return WorkflowStatus.PAUSED
        return WorkflowStatus.RUNNING

    @property
    def elapsed(self) -> float:
        return 12.5

    @property
    def agent_count(self) -> int:
        return 3

    @property
    def tokens_total(self) -> int:
        return 15420

    @property
    def phases(self) -> list[str]:
        if self.result is not None:
            return [p.name for p in self.result.run.phases]
        return list(self.runtime._phase_order) if self.runtime else []

    @property
    def phase_reports(self) -> list[Any]:
        if self.result is not None:
            return self.result.run.phases
        if self.runtime:
            return [
                self.runtime._phases[name]
                for name in self.runtime._phase_order
                if name in self.runtime._phases
            ]
        return self._phase_reports

    @property
    def budget_snapshot(self) -> BudgetSnapshot:
        if self.result is not None:
            return self.result.run.budget
        if self.runtime:
            return self.runtime._budget.snapshot()
        return self._budget_snapshot

    @property
    def is_paused(self) -> bool:
        if self.result is not None:
            return False
        if self.runtime is not None:
            return getattr(self.runtime, "is_paused", False)
        return False

    @property
    def live_agents(self) -> list[Any]:
        if self.result is not None:
            return []
        if self._live_agents_override is not None:
            return self._live_agents_override
        if self.runtime is not None:
            return list(getattr(self.runtime, "_live_agents", {}).values())
        return []


@dataclass
class FakeRunner:
    _runs: list[Any] = field(default_factory=list)

    @property
    def runs(self) -> list[Any]:
        return list(self._runs)

    def _find_run(self, run_id: str) -> Any | None:
        return next((r for r in self._runs if r.run_id == run_id), None)


class FakeOption:
    def __init__(self, option_id: str | None) -> None:
        self.id = option_id


class FakeOptionSelectedEvent:
    def __init__(self, option_id: str | None, option_index: int = 0) -> None:
        self.option = FakeOption(option_id)
        self.option_index = option_index


class FakeOptionList:
    def __init__(self, highlighted_option_id: str | None = None) -> None:
        self.highlighted_option = (
            FakeOption(highlighted_option_id)
            if highlighted_option_id is not None
            else None
        )
        self.highlighted: int | None = None
        self._options: list[Any] = []

    def clear_options(self) -> None:
        self._options = []

    def add_options(self, options: list[Any]) -> None:
        self._options.extend(options)


# ---------------------------------------------------------------------------
# Widget init tests
# ---------------------------------------------------------------------------


class TestWorkflowsAppInit:
    def test_init_sets_runner(self) -> None:
        runner = FakeRunner()
        app = WorkflowsApp(runner=runner)
        assert app._runner is runner

    def test_id_is_workflows_app(self) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        assert app.id == "workflows-app"

    def test_initial_view_is_list(self) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        assert app._view == "list"

    def test_initial_selections_are_none(self) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        assert app._selected_run_id is None
        assert app._selected_agent_idx is None

    def test_can_focus_children_is_true(self) -> None:
        assert WorkflowsApp.can_focus_children is True


# ---------------------------------------------------------------------------
# Binding tests
# ---------------------------------------------------------------------------


class TestWorkflowsAppBindings:
    def _get_binding_keys(self) -> list[str]:
        keys: list[str] = []
        for binding in WorkflowsApp.BINDINGS:
            keys.extend(binding.key.split(","))
        return keys

    def test_has_escape_binding(self) -> None:
        assert "escape" in self._get_binding_keys()

    def test_has_refresh_binding(self) -> None:
        assert "r" in self._get_binding_keys()

    def test_has_stop_binding(self) -> None:
        # Stop moved to `x` (Claude Code parity); `s` is now Save.
        assert "x" in self._get_binding_keys()

    def test_has_pause_binding(self) -> None:
        assert "p" in self._get_binding_keys()

    def test_has_save_binding(self) -> None:
        assert "s" in self._get_binding_keys()

    def test_has_script_binding(self) -> None:
        assert "o" in self._get_binding_keys()


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------


class TestWorkflowsAppMessages:
    def test_closed_message(self) -> None:
        msg = WorkflowsApp.Closed()
        assert isinstance(msg, WorkflowsApp.Closed)

    def test_stop_requested_stores_run_id(self) -> None:
        msg = WorkflowsApp.StopRequested("wf-1")
        assert msg.run_id == "wf-1"


# ---------------------------------------------------------------------------
# View transition tests
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_mock_render(monkeypatch: pytest.MonkeyPatch) -> WorkflowsApp:
    app = WorkflowsApp(runner=FakeRunner())
    rendered_views: list[str] = []

    async def fake_render() -> None:
        rendered_views.append(app._view)

    def fake_run_worker(coro: Any, **kw: Any) -> None:
        coro.close()

    monkeypatch.setattr(app, "_render_view", fake_render)
    monkeypatch.setattr(app, "run_worker", fake_run_worker)
    app._rendered_views = rendered_views  # type: ignore[attr-defined]
    return app


class TestViewTransitions:
    @pytest.mark.asyncio
    async def test_back_from_list_posts_closed(
        self,
        app_with_mock_render: WorkflowsApp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = app_with_mock_render
        app._view = "list"
        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_back()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowsApp.Closed)

    @pytest.mark.asyncio
    async def test_back_from_detail_goes_to_list(
        self, app_with_mock_render: WorkflowsApp
    ) -> None:
        app = app_with_mock_render
        app._view = "detail"
        app._selected_run_id = "wf-1"

        app.action_back()

        assert app._view == "list"
        assert app._selected_run_id is None

    @pytest.mark.asyncio
    async def test_back_from_agent_goes_to_detail(
        self, app_with_mock_render: WorkflowsApp
    ) -> None:
        app = app_with_mock_render
        app._view = "agent"
        app._selected_agent_idx = 0
        app._selected_run_id = "wf-1"

        app.action_back()

        assert app._view == "detail"
        assert app._selected_agent_idx is None
        assert app._selected_run_id == "wf-1"

    @pytest.mark.asyncio
    async def test_select_run_from_list_goes_to_detail(
        self,
        app_with_mock_render: WorkflowsApp,
    ) -> None:
        app = app_with_mock_render
        app._view = "list"

        event = FakeOptionSelectedEvent(option_id="wf-1", option_index=0)
        await app.on_option_list_option_selected(event)  # type: ignore[arg-type]

        assert app._view == "detail"
        assert app._selected_run_id == "wf-1"

    @pytest.mark.asyncio
    async def test_select_agent_from_detail_goes_to_agent(
        self,
        app_with_mock_render: WorkflowsApp,
    ) -> None:
        app = app_with_mock_render
        app._view = "detail"
        app._selected_run_id = "wf-1"

        event = FakeOptionSelectedEvent(option_id="agent-2", option_index=2)
        await app.on_option_list_option_selected(event)  # type: ignore[arg-type]

        assert app._view == "agent"
        assert app._selected_agent_idx == 2


# ---------------------------------------------------------------------------
# Stop action tests
# ---------------------------------------------------------------------------


class TestStopAction:
    def test_stop_in_list_posts_stop_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "list"

        fake_option_list = FakeOptionList(highlighted_option_id="wf-1")
        monkeypatch.setattr(app, "query_one", lambda *args: fake_option_list)

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_stop()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowsApp.StopRequested)
        assert posted[0].run_id == "wf-1"

    def test_stop_in_detail_posts_stop_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "detail"
        app._selected_run_id = "wf-2"

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_stop()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowsApp.StopRequested)
        assert posted[0].run_id == "wf-2"

    def test_stop_in_agent_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "agent"

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_stop()

        assert posted == []

    def test_stop_with_no_highlighted_run_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "list"

        fake_option_list = FakeOptionList(highlighted_option_id=None)
        monkeypatch.setattr(app, "query_one", lambda *args: fake_option_list)

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_stop()

        assert posted == []


# ---------------------------------------------------------------------------
# WorkflowRunEntry property tests
# ---------------------------------------------------------------------------


class TestWorkflowRunEntryProperties:
    def test_phase_reports_from_result(self) -> None:
        from vibe.cli.textual_ui.workflow_runner import WorkflowRunEntry
        from vibe.core.workflows.runtime import WorkflowRuntime

        phase = PhaseReport(
            name="explore",
            agent_results=[
                AgentResult(prompt="p1", response="r1", tokens_in=10, tokens_out=5)
            ],
        )
        run = WorkflowRun(
            phases=[phase],
            status=WorkflowStatus.COMPLETED,
            budget=BudgetSnapshot(total=100_000, reserved=0, spent=15),
        )
        result = WorkflowResult(return_value={}, run=run, summary="done")
        entry = WorkflowRunEntry(
            run_id="wf-1",
            script_source="",
            started_at=0.0,
            runtime=WorkflowRuntime(),
            result=result,
        )

        reports = entry.phase_reports
        assert len(reports) == 1
        assert reports[0].name == "explore"
        assert len(reports[0].agent_results) == 1

    def test_phase_reports_from_runtime(self) -> None:
        from vibe.cli.textual_ui.workflow_runner import WorkflowRunEntry
        from vibe.core.workflows.runtime import WorkflowRuntime

        runtime = WorkflowRuntime()
        runtime._declare_phase("research")
        runtime._record_agent_result(
            AgentResult(
                prompt="search",
                response="found",
                phase="research",
                tokens_in=20,
                tokens_out=10,
            )
        )

        entry = WorkflowRunEntry(
            run_id="wf-1",
            script_source="",
            started_at=0.0,
            runtime=runtime,
        )

        reports = entry.phase_reports
        assert len(reports) == 1
        assert reports[0].name == "research"
        assert len(reports[0].agent_results) == 1

    def test_budget_snapshot_from_result(self) -> None:
        from vibe.cli.textual_ui.workflow_runner import WorkflowRunEntry
        from vibe.core.workflows.runtime import WorkflowRuntime

        budget = BudgetSnapshot(total=200_000, reserved=1000, spent=50000)
        run = WorkflowRun(
            status=WorkflowStatus.COMPLETED,
            budget=budget,
        )
        result = WorkflowResult(return_value=None, run=run, summary="")
        entry = WorkflowRunEntry(
            run_id="wf-1",
            script_source="",
            started_at=0.0,
            runtime=WorkflowRuntime(),
            result=result,
        )

        snap = entry.budget_snapshot
        assert snap.total == 200_000
        assert snap.reserved == 1000
        assert snap.spent == 50000

    def test_budget_snapshot_from_runtime(self) -> None:
        from vibe.cli.textual_ui.workflow_runner import WorkflowRunEntry
        from vibe.core.workflows.runtime import WorkflowRuntime

        runtime = WorkflowRuntime(budget_total=50_000)
        entry = WorkflowRunEntry(
            run_id="wf-1",
            script_source="",
            started_at=0.0,
            runtime=runtime,
        )

        snap = entry.budget_snapshot
        assert snap.total == 50_000
        assert snap.spent == 0


# ---------------------------------------------------------------------------
# Pause / save / script message + action tests
# ---------------------------------------------------------------------------


class TestWorkflowsAppNewMessages:
    def test_pause_toggle_requested_stores_run_id(self) -> None:
        msg = WorkflowsApp.PauseToggleRequested("wf-3")
        assert msg.run_id == "wf-3"

    def test_save_requested_stores_run_id_and_source(self) -> None:
        msg = WorkflowsApp.SaveRequested("wf-3", "async def main(): ...", name=None)
        assert msg.run_id == "wf-3"
        assert msg.script_source == "async def main(): ..."
        assert msg.name is None

    def test_save_requested_carries_name(self) -> None:
        msg = WorkflowsApp.SaveRequested(
            "wf-3", "src", name="audit"
        )
        assert msg.name == "audit"


class TestTogglePauseAction:
    def test_toggle_pause_in_detail_posts_pause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "detail"
        app._selected_run_id = "wf-2"

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_toggle_pause()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowsApp.PauseToggleRequested)
        assert posted[0].run_id == "wf-2"

    def test_toggle_pause_in_list_uses_highlighted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "list"
        fake_option_list = FakeOptionList(highlighted_option_id="wf-5")
        monkeypatch.setattr(app, "query_one", lambda *a: fake_option_list)

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_toggle_pause()

        assert isinstance(posted[0], WorkflowsApp.PauseToggleRequested)
        assert posted[0].run_id == "wf-5"

    def test_toggle_pause_in_agent_view_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "agent"
        app._selected_run_id = "wf-2"

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_toggle_pause()
        assert posted == []


class TestSaveAction:
    def test_save_in_detail_posts_save_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = FakeRunEntry(run_id="wf-1", script_source="async def main(): return 1")
        runner = FakeRunner(_runs=[entry])
        app = WorkflowsApp(runner=runner)
        app._view = "detail"
        app._selected_run_id = "wf-1"

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_save()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowsApp.SaveRequested)
        assert posted[0].run_id == "wf-1"
        assert posted[0].script_source == "async def main(): return 1"

    def test_save_with_no_focus_does_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "agent"

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_save()
        assert posted == []


class TestScriptAction:
    def test_script_in_detail_transitions_to_script_view(
        self, app_with_mock_render: WorkflowsApp
    ) -> None:
        app = app_with_mock_render
        app._view = "detail"
        app._selected_run_id = "wf-1"

        app.action_script()

        assert app._view == "script"

    def test_script_in_list_does_nothing(self) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        app._view = "list"
        app.action_script()
        assert app._view == "list"


# ---------------------------------------------------------------------------
# Unified agent-row builder (live + done)
# ---------------------------------------------------------------------------


class TestAgentCancelAction:
    def test_x_in_agent_view_cancels_live_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In the agent-detail view, `x` cancels the focused in-flight agent
        (posts AgentCancelRequested), not the whole run.
        """
        live = FakeLiveAgent(agent_id="la-2", label="running-agent")
        entry = FakeRunEntry(
            run_id="wf-1",
            runtime=FakeRuntime(),
            _live_agents_override=[live],
        )
        runner = FakeRunner(_runs=[entry])
        app = WorkflowsApp(runner=runner)
        app._view = "agent"
        app._selected_run_id = "wf-1"
        app._selected_agent_idx = 0  # the live row

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_stop()

        assert len(posted) == 1
        assert isinstance(posted[0], WorkflowsApp.AgentCancelRequested)
        assert posted[0].run_id == "wf-1"
        assert posted[0].agent_id == "la-2"

    def test_x_in_agent_view_does_nothing_for_finalized_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finalized agent can't be cancelled; `x` posts nothing."""
        phase = FakePhaseReport(
            name="done",
            agent_results=[FakeAgentResult(label="finished", phase="done")],
        )
        entry = FakeRunEntry(
            run_id="wf-1",
            runtime=FakeRuntime(_phases={"done": phase}, _phase_order=["done"]),
        )
        runner = FakeRunner(_runs=[entry])
        app = WorkflowsApp(runner=runner)
        app._view = "agent"
        app._selected_run_id = "wf-1"
        app._selected_agent_idx = 0  # the done row

        posted: list[object] = []
        monkeypatch.setattr(app, "post_message", posted.append)

        app.action_stop()
        assert posted == []


class TestGetAgentRows:
    def test_live_agents_listed_before_done(self) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        phase = FakePhaseReport(
            name="research",
            agent_results=[
                FakeAgentResult(label="done-1", phase="research"),
            ],
        )
        runtime = FakeRuntime(_phases={"research": phase}, _phase_order=["research"])
        entry = FakeRunEntry(
            run_id="wf-1",
            runtime=runtime,
            _live_agents_override=[FakeLiveAgent(agent_id="la-0", label="live-1")],
        )

        rows = app._get_agent_rows(entry)

        assert len(rows) == 2
        assert rows[0].kind == "live"
        assert rows[0].label == "live-1"
        assert rows[0].status == "running"
        assert rows[0].key == "live-la-0"
        assert rows[1].kind == "done"
        assert rows[1].label == "done-1"
        # live carries no response; done does
        assert rows[0].response is None
        assert rows[1].response == "test response"

    def test_empty_when_no_live_and_no_done(self) -> None:
        app = WorkflowsApp(runner=FakeRunner())
        runtime = FakeRuntime()
        entry = FakeRunEntry(run_id="wf-1", runtime=runtime)
        assert app._get_agent_rows(entry) == []
