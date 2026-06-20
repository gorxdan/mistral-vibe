from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.core.workflows.models import AgentResult

if TYPE_CHECKING:
    from vibe.cli.textual_ui.workflow_runner import WorkflowRunEntry, WorkflowRunner


_STATUS_COLORS: dict[str, str] = {
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "paused": "blue",
}

_TRUNCATE_LEN = 80
_SCRIPT_PREVIEW_LINES = 5
_POLL_INTERVAL = 1.0
_TOKEN_K_THRESHOLD = 1_000
_TOKEN_M_THRESHOLD = 1_000_000


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def _format_tokens(n: int) -> str:
    if n >= _TOKEN_M_THRESHOLD:
        return f"{n / _TOKEN_M_THRESHOLD:.1f}M"
    if n >= _TOKEN_K_THRESHOLD:
        return f"{n / _TOKEN_K_THRESHOLD:.1f}k"
    return str(n)


def _build_run_option_text(entry: WorkflowRunEntry) -> Text:
    text = Text(no_wrap=True)
    text.append(f"{entry.run_id:6}", style="cyan")
    text.append("  ")
    status_color = _STATUS_COLORS.get(entry.status.value, "white")
    text.append(f"{entry.status.value:10}", style=status_color)
    text.append("  ")
    text.append(f"{entry.elapsed:.1f}s", style="dim")
    text.append("  ")
    agents = entry.agent_count
    text.append(f"{agents} agent{'s' if agents != 1 else ''}", style="dim")
    text.append("  ")
    text.append(f"{_format_tokens(entry.tokens_total)} tok", style="dim")
    phases = ", ".join(entry.phases) or "(none)"
    text.append(f"  {phases}")
    return text


def _build_agent_option_text(
    phase_name: str, result: AgentResult, idx: int
) -> Text:
    text = Text(no_wrap=True)
    text.append(f"[{phase_name}]", style="cyan")
    label = result.label or f"agent-{idx + 1}"
    text.append(f" {label}", style="bold")
    status = "completed" if result.completed else "failed"
    status_color = "green" if result.completed else "red"
    text.append(f" {status}", style=status_color)
    text.append(f" {_format_tokens(result.tokens_total)} tok", style="dim")
    if result.error:
        text.append(f" {_truncate(result.error, 40)}", style="red")
    else:
        text.append(f" {_truncate(result.prompt, 60)}", style="dim")
    return text


def _build_run_detail_header(entry: WorkflowRunEntry) -> Text:
    text = Text()
    text.append(f"Run: {entry.run_id}", style="bold cyan")
    status_color = _STATUS_COLORS.get(entry.status.value, "white")
    text.append(f"  Status: {entry.status.value}", style=status_color)
    text.append(f"\nElapsed: {entry.elapsed:.1f}s", style="dim")
    text.append(f"  Agents: {entry.agent_count}", style="dim")
    text.append(f"  Tokens: {_format_tokens(entry.tokens_total)}", style="dim")
    budget = entry.budget_snapshot
    if budget.total is not None:
        text.append(
            f"  Budget: {_format_tokens(budget.spent)}/{_format_tokens(budget.total)}",
            style="dim",
        )
    if entry.error:
        text.append(f"\nError: {entry.error}", style="red")
    return text


def _build_agent_detail(phase_name: str, result: AgentResult, idx: int) -> Text:
    text = Text()
    label = result.label or f"agent-{idx + 1}"
    text.append(f"Agent: {label}", style="bold cyan")
    text.append(f"  Phase: {phase_name}", style="cyan")
    status = "completed" if result.completed else "failed"
    status_color = "green" if result.completed else "red"
    text.append(f"  Status: {status}", style=status_color)
    text.append(
        f"\nTokens: {_format_tokens(result.tokens_total)} "
        f"(in: {_format_tokens(result.tokens_in)}, "
        f"out: {_format_tokens(result.tokens_out)})",
        style="dim",
    )
    text.append(f"  Cost: ${result.cost:.4f}", style="dim")
    text.append("\n\n--- Prompt ---", style="bold")
    text.append(f"\n{result.prompt}")
    text.append("\n\n--- Response ---", style="bold")
    response = result.response
    if isinstance(response, dict):
        text.append(f"\n{json.dumps(response, indent=2, default=str)}")
    else:
        text.append(f"\n{response}")
    if result.error:
        text.append("\n\n--- Error ---", style="bold red")
        text.append(f"\n{result.error}", style="red")
    return text


@dataclass
class _AgentRow:
    """Unified view-model row for one agent in the detail drill-down.

    Covers both in-flight agents (kind='live', no response yet) and finalized
    agents (kind='done'). The detail list shows live agents first so the user
    sees what is running now, above what has completed.
    """

    kind: str  # "live" | "done"
    key: str  # option id: "live-<agent_id>" | "agent-<idx>"
    phase: str
    label: str
    status: str  # "running" | "completed" | "failed"
    tokens_total: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    prompt: str = ""
    response: Any = None
    error: str | None = None
    agent: str | None = None
    model: str | None = None


def _build_live_agent_option_text(row: _AgentRow) -> Text:
    text = Text(no_wrap=True)
    text.append(f"[{row.phase}]", style="cyan")
    text.append(f" {row.label}", style="bold")
    text.append(" running", style="yellow")
    text.append(f" {_format_tokens(row.tokens_total)} tok", style="dim")
    detail = row.agent or row.model
    if detail:
        text.append(f" {detail}", style="dim")
    return text


def _build_live_agent_detail(row: _AgentRow) -> Text:
    text = Text()
    text.append(f"Agent: {row.label}", style="bold cyan")
    text.append(f"  Phase: {row.phase}", style="cyan")
    text.append("  Status: running", style="yellow")
    text.append(
        f"\nTokens (so far): {_format_tokens(row.tokens_total)} "
        f"(in: {_format_tokens(row.tokens_in)}, "
        f"out: {_format_tokens(row.tokens_out)})",
        style="dim",
    )
    if row.agent:
        text.append(f"\nProfile: {row.agent}", style="dim")
    if row.model:
        text.append(f"  Model: {row.model}", style="dim")
    text.append("\n\n(In-flight — response not available until the agent finishes.)")
    if row.error:
        text.append("\n\n--- Error ---", style="bold red")
        text.append(f"\n{row.error}", style="red")
    return text


class WorkflowsApp(Container):
    """Interactive workflow monitor with drill-down into runs and agents."""

    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", show=False),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("x", "stop", "Stop", show=True),
        Binding("p", "toggle_pause", "Pause/Resume", show=True),
        Binding("s", "save", "Save", show=True),
        Binding("o", "script", "Script", show=True),
    ]

    class Closed(Message):
        pass

    class StopRequested(Message):
        run_id: str

        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            super().__init__()

    class PauseToggleRequested(Message):
        """Toggle pause/resume for the focused run (handled by the app)."""

        run_id: str

        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            super().__init__()

    class AgentCancelRequested(Message):
        """Cancel a single in-flight agent (from the agent-detail view).

        Carries both the run id and the live agent id so the app can call
        WorkflowRunner.cancel_agent. Only meaningful for live agents; the
        widget only emits this when the focused row is an in-flight agent.
        """

        run_id: str
        agent_id: str

        def __init__(self, run_id: str, agent_id: str) -> None:
            self.run_id = run_id
            self.agent_id = agent_id
            super().__init__()

    class SaveRequested(Message):
        """Persist the focused run's script as a reusable /<name> command."""

        run_id: str
        script_source: str
        name: str | None

        def __init__(
            self, run_id: str, script_source: str, name: str | None = None
        ) -> None:
            self.run_id = run_id
            self.script_source = script_source
            self.name = name
            super().__init__()

    def __init__(self, runner: WorkflowRunner, **kwargs: Any) -> None:
        super().__init__(id="workflows-app", **kwargs)
        self._runner = runner
        self._view: str = "list"
        self._selected_run_id: str | None = None
        self._selected_agent_idx: int | None = None
        self._agent_rows: list[_AgentRow] = []
        self._poll_timer: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="workflows-content"):
            yield NoMarkupStatic(
                "", id="workflows-header", classes="workflows-header"
            )
            yield Vertical(id="workflows-body")
            yield NoMarkupStatic("", id="workflows-help", classes="workflows-help")

    async def on_mount(self) -> None:
        await self._render_view()
        self._schedule_poll()

    def on_unmount(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()

    # --- Polling ---

    def _schedule_poll(self) -> None:
        self._poll_timer = self.set_timer(_POLL_INTERVAL, self._tick)

    def _tick(self) -> None:
        if not self.is_running:
            return
        self._refresh_current_view()
        self._schedule_poll()

    def _refresh_current_view(self) -> None:
        if self._view == "list":
            self._refresh_list_view()
        elif self._view == "detail":
            self._refresh_detail_view()
        # "agent" and "script" views are static snapshots — no live refresh.

    # --- View rendering ---

    async def _render_view(self) -> None:
        body = self.query_one("#workflows-body", Vertical)
        await body.remove_children()
        if self._view == "list":
            await self._render_list_view(body)
        elif self._view == "detail":
            await self._render_detail_view(body)
        elif self._view == "agent":
            await self._render_agent_view(body)
        elif self._view == "script":
            await self._render_script_view(body)
        self._update_help()

    async def _render_list_view(self, body: Vertical) -> None:
        runs = self._runner.runs
        header = self.query_one("#workflows-header", NoMarkupStatic)
        header.update(Text("Workflow Runs", style="bold"))

        if not runs:
            await body.mount(
                NoMarkupStatic(
                    "No workflow runs. Use /<workflow-name> to launch one.",
                    classes="workflows-empty",
                )
            )
            return

        options = [
            Option(_build_run_option_text(entry), id=entry.run_id)
            for entry in runs
        ]
        option_list = OptionList(*options, id="workflows-run-list")
        await body.mount(option_list)
        option_list.focus()

    def _refresh_list_view(self) -> None:
        try:
            option_list = self.query_one("#workflows-run-list", OptionList)
        except Exception:
            return
        runs = self._runner.runs
        highlighted = option_list.highlighted
        option_list.clear_options()
        option_list.add_options(
            [
                Option(_build_run_option_text(entry), id=entry.run_id)
                for entry in runs
            ]
        )
        if highlighted is not None and highlighted < len(runs):
            option_list.highlighted = highlighted

    async def _render_detail_view(self, body: Vertical) -> None:
        entry = self._find_selected_run()
        if entry is None:
            self._view = "list"
            await self._render_view()
            return

        header = self.query_one("#workflows-header", NoMarkupStatic)
        header.update(_build_run_detail_header(entry))

        script_lines = entry.script_source.strip().split("\n")[:_SCRIPT_PREVIEW_LINES]
        script_preview = Text("Script preview (press o for full):\n", style="dim")
        script_preview.append("\n".join(script_lines), style="dim")
        await body.mount(
            NoMarkupStatic(script_preview, classes="workflows-script")
        )

        rows = self._get_agent_rows(entry)
        if not rows:
            await body.mount(
                NoMarkupStatic(
                    "No agents yet.",
                    classes="workflows-empty",
                )
            )
            return

        options = [Option(self._build_row_option_text(row), id=row.key) for row in rows]
        option_list = OptionList(*options, id="workflows-agent-list")
        await body.mount(option_list)
        option_list.focus()

    def _refresh_detail_view(self) -> None:
        entry = self._find_selected_run()
        if entry is None:
            return
        header = self.query_one("#workflows-header", NoMarkupStatic)
        header.update(_build_run_detail_header(entry))

        try:
            option_list = self.query_one("#workflows-agent-list", OptionList)
        except Exception:
            return
        rows = self._get_agent_rows(entry)
        highlighted = option_list.highlighted
        option_list.clear_options()
        option_list.add_options(
            [Option(self._build_row_option_text(row), id=row.key) for row in rows]
        )
        if highlighted is not None and highlighted < len(rows):
            option_list.highlighted = highlighted

    @staticmethod
    def _build_row_option_text(row: _AgentRow) -> Text:
        if row.kind == "live":
            return _build_live_agent_option_text(row)
        # Finalized row: reuse the AgentResult renderer via a throwaway result.
        result = AgentResult(
            label=row.label,
            prompt=row.prompt,
            response=row.response,
            tokens_in=row.tokens_in,
            tokens_out=row.tokens_out,
            cost=row.cost,
            completed=(row.status == "completed"),
            error=row.error,
        )
        return _build_agent_option_text(row.phase, result, 0)

    async def _render_agent_view(self, body: Vertical) -> None:
        entry = self._find_selected_run()
        if entry is None or self._selected_agent_idx is None:
            self._view = "detail"
            await self._render_view()
            return

        rows = self._get_agent_rows(entry)
        if self._selected_agent_idx >= len(rows):
            self._view = "detail"
            await self._render_view()
            return

        row = rows[self._selected_agent_idx]

        header = self.query_one("#workflows-header", NoMarkupStatic)
        header.update(Text(f"Run {entry.run_id} \u2192 Agent Detail", style="bold"))

        scroll = VerticalScroll(id="workflows-agent-detail")
        await body.mount(scroll)
        if row.kind == "live":
            await scroll.mount(NoMarkupStatic(_build_live_agent_detail(row)))
        else:
            result = AgentResult(
                label=row.label,
                prompt=row.prompt,
                response=row.response,
                tokens_in=row.tokens_in,
                tokens_out=row.tokens_out,
                cost=row.cost,
                completed=(row.status == "completed"),
                error=row.error,
            )
            await scroll.mount(
                NoMarkupStatic(_build_agent_detail(row.phase, result, self._selected_agent_idx))
            )

    async def _render_script_view(self, body: Vertical) -> None:
        entry = self._find_selected_run()
        if entry is None:
            self._view = "list"
            await self._render_view()
            return

        header = self.query_one("#workflows-header", NoMarkupStatic)
        header.update(Text(f"Run {entry.run_id} \u2192 Script", style="bold"))

        scroll = VerticalScroll(id="workflows-script-detail")
        await body.mount(scroll)
        await scroll.mount(
            NoMarkupStatic(
                Text(entry.script_source or "(empty script)", style="dim")
            )
        )

    # --- Helpers ---

    def _find_selected_run(self) -> WorkflowRunEntry | None:
        if self._selected_run_id is None:
            return None
        return self._runner._find_run(self._selected_run_id)

    def _get_agent_rows(self, entry: WorkflowRunEntry) -> list[_AgentRow]:
        """In-flight agents first, then finalized results, as unified rows.

        Live agents carry running token totals; finalized agents carry their
        recorded prompt/response. Positional indices in this list back the
        detail OptionList and the agent-detail view, so a refresh that adds or
        retires a live agent can shift indices — selection is restored by
        highlighted position, not by id.
        """
        rows: list[_AgentRow] = []

        live = getattr(entry, "live_agents", None) or []
        for la in live:
            label = getattr(la, "label", None) or la.agent_id
            rows.append(
                _AgentRow(
                    kind="live",
                    key=f"live-{la.agent_id}",
                    phase=getattr(la, "phase", None) or "default",
                    label=label,
                    status="running",
                    tokens_total=getattr(la, "tokens_total", 0),
                    tokens_in=getattr(la, "tokens_in", 0),
                    tokens_out=getattr(la, "tokens_out", 0),
                    agent=getattr(la, "agent", None),
                    model=getattr(la, "model", None),
                    error=getattr(la, "error", None),
                )
            )

        for phase in entry.phase_reports:
            for idx, ar in enumerate(phase.agent_results):
                rows.append(
                    _AgentRow(
                        kind="done",
                        key=f"agent-{len(rows)}",
                        phase=phase.name,
                        label=ar.label or f"agent-{idx + 1}",
                        status="completed" if ar.completed else "failed",
                        tokens_total=ar.tokens_total,
                        tokens_in=ar.tokens_in,
                        tokens_out=ar.tokens_out,
                        cost=ar.cost,
                        prompt=ar.prompt,
                        response=ar.response,
                        error=ar.error,
                    )
                )
        self._agent_rows = rows
        return rows

    def _highlighted_run_id(self) -> str | None:
        try:
            option_list = self.query_one("#workflows-run-list", OptionList)
        except Exception:
            return None
        option = option_list.highlighted_option
        if option is None or option.id is None:
            return None
        return str(option.id)

    def _focused_run_id(self) -> str | None:
        """The run the current view is focused on, for key actions.

        Run-level actions (stop/pause/save) are scoped to the list and detail
        views. Inside the agent- or script-detail views, Esc returns to the
        run detail first, so a stray key there does not stop the whole run.
        """
        if self._view == "detail":
            return self._selected_run_id
        if self._view == "list":
            return self._highlighted_run_id()
        return None

    def _update_help(self) -> None:
        help_widget = self.query_one("#workflows-help", NoMarkupStatic)
        if self._view == "list":
            help_widget.update(
                "\u2191\u2193 Navigate  Enter Select  x Stop  p Pause  s Save  "
                "r Refresh  Esc Back"
            )
        elif self._view == "detail":
            help_widget.update(
                "\u2191\u2193 Navigate  Enter Agent Detail  x Stop  p Pause  "
                "s Save  o Script  Esc Back"
            )
        elif self._view == "agent":
            help_widget.update("x Cancel agent  Esc Back to Run Detail")
        elif self._view == "script":
            help_widget.update("Esc Back to Run Detail")

    # --- Actions ---

    def action_back(self) -> None:
        if self._view in {"agent", "script"}:
            self._view = "detail"
            self._selected_agent_idx = None
            self.run_worker(self._render_view(), exclusive=True)
        elif self._view == "detail":
            self._view = "list"
            self._selected_run_id = None
            self.run_worker(self._render_view(), exclusive=True)
        else:
            self.post_message(self.Closed())

    def action_stop(self) -> None:
        # In the agent-detail view, `x` cancels the focused in-flight agent
        # (Claude Code parity) rather than the whole run. A finalized agent
        # can't be cancelled, so fall through to nothing.
        if self._view == "agent" and self._selected_run_id is not None:
            entry = self._find_selected_run()
            if entry is not None and self._selected_agent_idx is not None:
                rows = self._get_agent_rows(entry)
                if self._selected_agent_idx < len(rows):
                    row = rows[self._selected_agent_idx]
                    if row.kind == "live":
                        # row.key is "live-<agent_id>"
                        agent_id = row.key.removeprefix("live-")
                        self.post_message(
                            self.AgentCancelRequested(self._selected_run_id, agent_id)
                        )
            return
        run_id = self._focused_run_id()
        if run_id:
            self.post_message(self.StopRequested(run_id))

    def action_toggle_pause(self) -> None:
        run_id = self._focused_run_id()
        if run_id:
            self.post_message(self.PauseToggleRequested(run_id))

    def action_save(self) -> None:
        run_id = self._focused_run_id()
        if not run_id:
            return
        entry = self._find_selected_run() if run_id == self._selected_run_id else None
        if entry is None:
            entry = self._runner._find_run(run_id)
        if entry is None:
            return
        self.post_message(
            self.SaveRequested(run_id, entry.script_source, name=None)
        )

    def action_script(self) -> None:
        if self._view != "detail":
            return
        if self._selected_run_id is None:
            return
        self._view = "script"
        self.run_worker(self._render_view(), exclusive=True)

    def action_refresh(self) -> None:
        self._refresh_current_view()

    # --- OptionList events ---

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if self._view == "list":
            if event.option.id:
                self._selected_run_id = str(event.option.id)
                self._view = "detail"
                await self._render_view()
        elif self._view == "detail":
            self._selected_agent_idx = event.option_index
            self._view = "agent"
            await self._render_view()
