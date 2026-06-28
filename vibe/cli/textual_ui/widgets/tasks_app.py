"""Tasks pane — unified background-task monitor.

Replaces WorkflowsApp as the ctrl+w surface. Aggregates five categories via the
BackgroundRegistry into one list with a category filter, and routes stop/pause
back through the registry. Per-category detail views: process rows show a live
log tail; workflow rows show phases/tokens/script; agent/team/loop rows show
their status card.

Mirrors WorkflowsApp's structure (1s poll, drill-down views, message-based
actions handled on VibeApp) so the existing app wiring transfers cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from vibe.core.tools.background import BackgroundRegistry, TaskCategory, TaskEntry

if TYPE_CHECKING:
    from vibe.cli.textual_ui.workflow_runner import WorkflowRunner

_POLL_INTERVAL = 1.0
_TRUNCATE_LEN = 70
_TOKEN_K = 1_000

_STATUS_COLORS: dict[str, str] = {
    "running": "yellow",
    "completed": "green",
    "failed": "red",
    "paused": "blue",
    "stopped": "magenta",
    "waiting": "cyan",
}

# Filter order for the header row + number-key bindings (1=All ... 6=Subagents).
_FILTERS: list[tuple[str, TaskCategory | None]] = [
    ("All", None),
    ("Processes", TaskCategory.PROCESS),
    ("Workflows", TaskCategory.WORKFLOW),
    ("Teams", TaskCategory.TEAM),
    ("Loops", TaskCategory.LOOP),
    ("Subagents", TaskCategory.ASYNC_AGENT),
]


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"


def _fmt_tokens(n: int) -> str:
    return f"{n / _TOKEN_K:.1f}k" if n >= _TOKEN_K else str(n)


def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m{s % 60}s"
    return f"{s}s"


def _build_row_text(entry: TaskEntry) -> Text:
    text = Text(no_wrap=True)
    text.append(f"{entry.task_id:18}", style="cyan")
    text.append("  ")
    text.append(f"{entry.category.value:9}", style="dim")
    text.append("  ")
    color = _STATUS_COLORS.get(entry.status, "white")
    text.append(f"{entry.status:9}", style=color)
    text.append("  ")
    if entry.category == TaskCategory.LOOP:
        text.append(f"fires in {_fmt_seconds(entry.elapsed):>6}", style="cyan")
    else:
        text.append(f"{_fmt_seconds(entry.elapsed):>6}", style="dim")
    text.append("  ")
    text.append(_truncate(entry.label), style="white")
    return text


# Snippet caps for the inline agent summary in the detail view.
_AGENT_PROMPT_SNIPPET = 280
_AGENT_RESPONSE_SNIPPET = 700


@dataclass
class _AgentRow:
    """A browsable agent within a workflow run — live or finalized.

    Unifies an in-flight _LiveAgent (partial response in response_so_far) and a
    finalized AgentResult (full response) so the workflow detail can list both
    and the drill-down view can render a full prompt + response for either.
    """

    key: str
    label: str
    phase: str | None
    status: str
    profile: str | None
    model: str | None
    tokens: int
    prompt: str
    response: str
    run_id: str
    error: str | None = None
    schema_errors: list[str] | None = None


def _agent_row_text(agent: _AgentRow) -> Text:
    text = Text(no_wrap=True)
    color = _STATUS_COLORS.get(agent.status, "white")
    text.append(f"{agent.status:9}", style=color)
    text.append("  ")
    text.append(f"{_fmt_tokens(agent.tokens):>7}", style="dim")
    text.append("  ")
    text.append(f"{(agent.phase or '-'):14}", style="cyan")
    text.append("  ")
    text.append(_truncate(agent.label or agent.profile or agent.key), style="white")
    return text


def _gather_workflow_agents(runner: WorkflowRunner, run_id: str) -> list[_AgentRow]:
    """Collect a run's agents (finalized first, then live) for the detail list.

    Finalized agents come straight from the runner's phase reports — the registry
    only carries live-agent rows, so finalized results are read here to avoid
    duplicating them into the flat task list. Each carries its full prompt and
    response; live agents carry their prompt and the streaming response_so_far.
    """
    entry = runner.find_run(run_id)
    if entry is None:
        return []
    rows: list[_AgentRow] = []
    for report in entry.phase_reports:
        for i, res in enumerate(report.agent_results):
            response = res.response
            if not isinstance(response, str):
                response = json.dumps(response, indent=2)
            rows.append(
                _AgentRow(
                    key=f"{run_id}/done-{report.name}-{i}",
                    label=res.label or res.agent or f"agent-{i}",
                    phase=res.phase or report.name,
                    status="completed" if res.completed else "failed",
                    profile=res.agent,
                    model=None,
                    tokens=res.tokens_total,
                    prompt=res.prompt or "",
                    response=response,
                    run_id=run_id,
                    error=res.error,
                    schema_errors=list(res.schema_errors)
                    if res.schema_errors
                    else None,
                )
            )
    for la in entry.live_agents:
        rows.append(
            _AgentRow(
                key=f"{run_id}/live-{getattr(la, 'agent_id', id(la))}",
                label=getattr(la, "label", None) or getattr(la, "agent", "agent"),
                phase=getattr(la, "phase", None),
                status=getattr(la, "status", "running") or "running",
                profile=getattr(la, "agent", None),
                model=getattr(la, "model", None),
                tokens=getattr(la, "tokens_total", 0),
                prompt=getattr(la, "prompt", "") or "",
                response=getattr(la, "response_so_far", "") or "",
                run_id=run_id,
            )
        )
    return rows


@dataclass
class _AgentViewData:
    """Carries a drilled-into agent's full prompt + response into the view."""

    title: str
    phase: str | None
    status: str
    tokens: int
    prompt: str
    response: str
    run_id: str
    key: str
    error: str | None = None
    schema_errors: list[str] = field(default_factory=list)


@dataclass
class _WorkflowScriptRef:
    """Carries a workflow run's script source so SaveRequested/script view work
    without TasksApp importing the WorkflowRunEntry type.
    """

    run_id: str
    script_source: str


class TasksApp(Container):
    """Unified background-task monitor with drill-down per category."""

    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", show=False),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("x", "stop", "Stop", show=True),
        Binding("p", "toggle_pause", "Pause/Resume", show=True),
        Binding("s", "save", "Save", show=True),
        Binding("o", "script", "Script", show=True),
        Binding("1", "filter_all", "All", show=False),
        Binding("2", "filter_processes", "Processes", show=False),
        Binding("3", "filter_workflows", "Workflows", show=False),
        Binding("4", "filter_teams", "Teams", show=False),
        Binding("5", "filter_loops", "Loops", show=False),
        Binding("6", "filter_subagents", "Subagents", show=False),
    ]

    class Closed(Message):
        pass

    class TaskStopRequested(Message):
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            super().__init__()

    class TaskPauseRequested(Message):
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            super().__init__()

    class SaveRequested(Message):
        """Persist the focused workflow run's script as a reusable command."""

        def __init__(
            self, run_id: str, script_source: str, name: str | None = None
        ) -> None:
            self.run_id = run_id
            self.script_source = script_source
            self.name = name
            super().__init__()

    def __init__(
        self,
        registry: BackgroundRegistry,
        workflow_runner: WorkflowRunner,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="tasks-app", **kwargs)
        self._registry = registry
        self._workflow_runner = workflow_runner
        self._view: str = "list"  # list | detail | script | agent
        self._filter_idx = 0  # index into _FILTERS
        self._selected_task_id: str | None = None
        self._agent_view_data: _AgentViewData | None = None
        self._poll_timer: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="tasks-content"):
            yield NoMarkupStatic("", id="tasks-header", classes="tasks-header")
            yield Vertical(id="tasks-body")
            yield NoMarkupStatic("", id="tasks-help", classes="tasks-help")

    async def on_mount(self) -> None:
        await self._render_view()
        self._schedule_poll()

    def on_unmount(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()

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
        elif self._view == "agent":
            self._refresh_agent_view()
        # "script" view is static.

    async def _render_view(self) -> None:
        body = self.query_one("#tasks-body", Vertical)
        await body.remove_children()
        if self._view == "list":
            await self._render_list_view(body)
        elif self._view == "detail":
            await self._render_detail_view(body)
        elif self._view == "script":
            await self._render_script_view(body)
        elif self._view == "agent":
            await self._render_agent_view(body)
        self._update_header()
        self._update_help()

    def _current_filter(self) -> TaskCategory | None:
        return _FILTERS[self._filter_idx][1]

    def _update_header(self) -> None:
        header = self.query_one("#tasks-header", NoMarkupStatic)
        if self._view == "list":
            chips = []
            for idx, (label, _cat) in enumerate(_FILTERS):
                marker = "[" if idx == self._filter_idx else " "
                closer = "]" if idx == self._filter_idx else " "
                style = "bold cyan" if idx == self._filter_idx else "dim"
                text = Text()
                text.append(marker, style=style)
                text.append(label, style=style)
                text.append(closer, style=style)
                chips.append(text)
            joined = Text()
            for i, c in enumerate(chips):
                if i:
                    joined.append("  ")
                joined.append_text(c)
            joined.append("\nBackground Tasks", style="bold")
            header.update(joined)
        elif self._view == "detail":
            header.update(Text(f"Task: {self._selected_task_id}", style="bold cyan"))
        elif self._view == "script":
            header.update(
                Text(f"{self._selected_task_id} \u2192 Script", style="bold cyan")
            )
        elif self._view == "agent":
            title = self._agent_view_data.title if self._agent_view_data else "Agent"
            header.update(Text(f"Agent: {title}", style="bold cyan"))

    def _entries(self) -> list[TaskEntry]:
        return self._registry.list_tasks(category=self._current_filter())

    async def _render_list_view(self, body: Vertical) -> None:
        entries = self._entries()
        if not entries:
            await body.mount(
                NoMarkupStatic(
                    "No background tasks. Launch a server with bash "
                    "background=true, or /<workflow-name>.",
                    classes="tasks-empty",
                )
            )
            return
        options = [Option(_build_row_text(e), id=e.task_id) for e in entries]
        option_list = OptionList(*options, id="tasks-list")
        await body.mount(option_list)
        option_list.focus()

    def _refresh_list_view(self) -> None:
        try:
            option_list = self.query_one("#tasks-list", OptionList)
        except Exception:
            return
        entries = self._entries()
        highlighted = option_list.highlighted
        option_list.clear_options()
        option_list.add_options([
            Option(_build_row_text(e), id=e.task_id) for e in entries
        ])
        if highlighted is not None and highlighted < len(entries):
            option_list.highlighted = highlighted

    async def _render_detail_view(self, body: Vertical) -> None:
        entry = self._find_selected()
        if entry is None:
            self._view = "list"
            await self._render_view()
            return
        scroll = VerticalScroll(id="tasks-detail")
        await body.mount(scroll)
        await scroll.mount(
            NoMarkupStatic(self._build_detail_text(entry), id="tasks-detail-text")
        )
        # Workflows get a navigable agent list below the summary so each agent's
        # prompt + response is one Enter away. Other categories are read-only.
        if entry.category == TaskCategory.WORKFLOW:
            agents = _gather_workflow_agents(self._workflow_runner, entry.task_id)
            if agents:
                options = [
                    Option(_agent_row_text(a), id=f"agent:{a.key}") for a in agents
                ]
                await scroll.mount(OptionList(*options, id="tasks-agent-list"))

    def _refresh_detail_view(self) -> None:
        if self._view != "detail":
            return
        entry = self._find_selected()
        if entry is None:
            return
        try:
            self.query_one("#tasks-detail", VerticalScroll)
        except Exception:
            return
        # Update widgets IN PLACE — do NOT remove + re-mount each tick. remove()
        # is deferred in Textual, so a re-mounted fixed-id widget (tasks-agent-
        # list) collides with the not-yet-removed old one -> DuplicateIds crash.
        try:
            self.query_one("#tasks-detail-text", NoMarkupStatic).update(
                self._build_detail_text(entry)
            )
        except Exception:
            pass

        agents = (
            _gather_workflow_agents(self._workflow_runner, entry.task_id)
            if entry.category == TaskCategory.WORKFLOW
            else []
        )
        try:
            agent_list = self.query_one("#tasks-agent-list", OptionList)
        except Exception:
            agent_list = None

        if agents:
            options = [Option(_agent_row_text(a), id=f"agent:{a.key}") for a in agents]
            if agent_list is None:
                self.query_one("#tasks-detail", VerticalScroll).mount(
                    OptionList(*options, id="tasks-agent-list")
                )
            else:
                highlighted = agent_list.highlighted
                agent_list.clear_options()
                agent_list.add_options(options)
                if highlighted is not None and highlighted < len(agents):
                    agent_list.highlighted = highlighted
        elif agent_list is not None:
            agent_list.remove()

    async def _render_script_view(self, body: Vertical) -> None:
        ref = self._workflow_script_ref()
        scroll = VerticalScroll(id="tasks-script")
        await body.mount(scroll)
        await scroll.mount(
            NoMarkupStatic(
                Text(ref.script_source if ref else "(no script)", style="dim")
            )
        )

    async def _render_agent_view(self, body: Vertical) -> None:
        scroll = VerticalScroll(id="tasks-agent")
        await body.mount(scroll)
        await scroll.mount(
            NoMarkupStatic(self._build_agent_view_text(), id="tasks-agent-text")
        )

    def _refresh_agent_view(self) -> None:
        if self._view != "agent" or self._agent_view_data is None:
            return
        # Live agents stream: refresh the carried data from the live object so
        # the partial response updates while the view is open. Update in place —
        # re-mounting each tick races Textual's deferred remove() (see
        # _refresh_detail_view).
        self._refresh_agent_view_data()
        try:
            self.query_one("#tasks-agent-text", NoMarkupStatic).update(
                self._build_agent_view_text()
            )
        except Exception:
            pass

    def _refresh_agent_view_data(self) -> None:
        """Re-read a live agent's streaming response into the view data.

        Finalized agents are immutable, so only live agents need refreshing.
        """
        av = self._agent_view_data
        if av is None or "/live-" not in av.key:
            return
        for la in self._workflow_runner_live_agents(av.run_id):
            la_id = getattr(la, "agent_id", None)
            if la_id and av.key.endswith(f"live-{la_id}"):
                av.response = getattr(la, "response_so_far", "") or ""
                av.tokens = getattr(la, "tokens_total", 0)
                av.status = getattr(la, "status", av.status) or av.status
                return

    def _workflow_runner_live_agents(self, run_id: str) -> list[Any]:
        entry = self._workflow_runner.find_run(run_id)
        return list(entry.live_agents) if entry is not None else []

    def _build_agent_view_text(self) -> Text:
        av = self._agent_view_data
        text = Text()
        if av is None:
            text.append("(no agent selected)", style="dim")
            return text
        text.append(av.title, style="bold cyan")
        color = _STATUS_COLORS.get(av.status, "white")
        text.append(f"  {av.status}", style=color)
        text.append(f"  {_fmt_tokens(av.tokens)} tokens", style="dim")
        if av.phase:
            text.append(f"  phase: {av.phase}", style="dim")
        text.append("\n\n--- Prompt ---", style="bold")
        text.append(f"\n{av.prompt or '(empty)'}")
        text.append("\n\n--- Response ---", style="bold")
        if av.error:
            text.append(f"\nERROR: {av.error}", style="red")
        if av.schema_errors:
            text.append("\n\nSchema validation errors:", style="bold red")
            for e in av.schema_errors:
                text.append(f"\n  {e}", style="red")
        if av.response:
            text.append(f"\n{av.response}")
        elif not av.error:
            text.append(
                "\n(Streaming — response appears here as the agent produces it.)",
                style="dim",
            )
        return text

    @staticmethod
    def _append_streaming_response(text: Text, d: dict[str, Any], status: str) -> None:
        preview = d.get("response_preview") or ""
        if preview:
            text.append("\n\n--- Response (streaming) ---", style="bold")
            text.append(f"\n{_truncate(preview, _AGENT_RESPONSE_SNIPPET)}")
        elif status == "running":
            text.append(
                "\n\n(Streaming — response appears here as it is produced.)",
                style="dim",
            )
        else:
            text.append("\n\n(No response captured.)", style="dim")

    def _build_detail_text(self, entry: TaskEntry) -> Text:
        text = Text()
        text.append(f"{entry.task_id}", style="bold cyan")
        text.append(f"  [{entry.category.value}]", style="dim")
        color = _STATUS_COLORS.get(entry.status, "white")
        text.append(f"  {entry.status}", style=color)
        d = entry.detail
        if entry.category == TaskCategory.PROCESS:
            text.append(f"\nCommand: {entry.label}", style="white")
            text.append(
                f"\nPID: {d.get('pid')}  Return code: {d.get('returncode')}",
                style="dim",
            )
            text.append(f"\nCWD: {d.get('cwd')}", style="dim")
            log_tail = self._registry.read_log_tail(entry.task_id, lines=80)
            text.append("\n\n--- Log (last 80 lines) ---", style="bold")
            text.append(f"\n{log_tail}" if log_tail else "\n(no output yet)")
        elif entry.category == TaskCategory.WORKFLOW:
            text.append(f"\nPhases: {entry.label}", style="white")
            text.append(
                f"\nAgents: {d.get('agent_count', 0)} "
                f"({d.get('live_agent_count', 0)} live)  "
                f"Tokens: {_fmt_tokens(d.get('tokens_total', 0))}",
                style="dim",
            )
            text.append(f"\nElapsed: {_fmt_seconds(entry.elapsed)}", style="dim")
            agents = _gather_workflow_agents(self._workflow_runner, entry.task_id)
            if agents:
                text.append(
                    f"\n\nAgents ({len(agents)}):  "
                    "\u2191\u2193 select, Enter to view prompt + response",
                    style="bold",
                )
            else:
                text.append("\n\n(no agents yet)", style="dim")
        elif entry.category == TaskCategory.AGENT:
            text.append(f"\nAgent: {entry.label}", style="white")
            text.append(
                f"\nPhase: {d.get('phase')}  "
                f"Tokens: {_fmt_tokens(d.get('tokens_total', 0))}",
                style="dim",
            )
            if d.get("agent"):
                text.append(f"\nProfile: {d['agent']}", style="dim")
            if d.get("model"):
                text.append(f"  Model: {d['model']}", style="dim")
            prompt = d.get("prompt") or ""
            if prompt:
                text.append("\n\n--- Prompt ---", style="bold")
                text.append(f"\n{_truncate(prompt, _AGENT_PROMPT_SNIPPET)}")
            self._append_streaming_response(text, d, entry.status)
        elif entry.category == TaskCategory.TEAM:
            text.append(f"\nName: {d.get('name')}", style="white")
            text.append(f"\nType: {entry.label}", style="dim")
            text.append(f"\nPID: {d.get('pid')}", style="dim")
            text.append(f"\nStatus: {d.get('raw_status')}", style="dim")
        elif entry.category == TaskCategory.LOOP:
            text.append(f"\nPrompt: {entry.label}", style="white")
            text.append(
                f"\nEvery {_fmt_seconds(d.get('interval_seconds', 0))}  "
                f"{'recurring' if d.get('recurring') else 'one-shot'}  "
                f"fires in {_fmt_seconds(d.get('remaining_seconds', 0))}",
                style="dim",
            )
        elif entry.category == TaskCategory.ASYNC_AGENT:
            text.append(f"\nSubagent: {entry.label}", style="white")
            if d.get("agent"):
                text.append(f"\nProfile: {d['agent']}", style="dim")
            text.append(f"\nElapsed: {_fmt_seconds(entry.elapsed)}", style="dim")
            if d.get("error"):
                text.append(f"\nError: {d['error']}", style="red")
            self._append_streaming_response(text, d, entry.status)
        return text

    def _find_selected(self) -> TaskEntry | None:
        if self._selected_task_id is None:
            return None
        for e in self._entries():
            if e.task_id == self._selected_task_id:
                return e
        # Fall back to an unfiltered lookup (the filter may have hidden it).
        for e in self._registry.list_tasks():
            if e.task_id == self._selected_task_id:
                return e
        return None

    def _highlighted_task_id(self) -> str | None:
        try:
            option_list = self.query_one("#tasks-list", OptionList)
        except Exception:
            return None
        option = option_list.highlighted_option
        if option is None or option.id is None:
            return None
        return str(option.id)

    def _focused_task_id(self) -> str | None:
        """The task the current view is acting on (selection or highlight)."""
        if self._view == "detail":
            return self._selected_task_id
        if self._view == "list":
            return self._highlighted_task_id()
        return None

    def _focused_entry(self) -> TaskEntry | None:
        tid = self._focused_task_id()
        if tid is None:
            return None
        for e in self._registry.list_tasks():
            if e.task_id == tid:
                return e
        return None

    def _workflow_script_ref(self) -> _WorkflowScriptRef | None:
        """Look up the selected workflow run's script source for save/script."""
        if self._selected_task_id is None or not self._selected_task_id.startswith(
            "wf-"
        ):
            return None
        entry = self._workflow_runner.find_run(self._selected_task_id)
        if entry is None:
            return None
        return _WorkflowScriptRef(
            run_id=entry.run_id, script_source=getattr(entry, "script_source", "") or ""
        )

    def _open_agent_view(self, key: str) -> bool:
        """Resolve an agent key to view data and switch to the agent view.

        Returns False (no transition) if the agent no longer exists — e.g. a
        finalized agent whose run was pruned between render and Enter.
        """
        run_id = key.split("/")[0] if "/" in key else self._selected_task_id
        if run_id is None:
            return False
        for a in _gather_workflow_agents(self._workflow_runner, run_id):
            if a.key == key:
                self._agent_view_data = _AgentViewData(
                    title=a.label or a.profile or key,
                    phase=a.phase,
                    status=a.status,
                    tokens=a.tokens,
                    prompt=a.prompt,
                    response=a.response,
                    run_id=a.run_id,
                    key=a.key,
                    error=a.error,
                    schema_errors=list(a.schema_errors) if a.schema_errors else [],
                )
                self._view = "agent"
                return True
        return False

    def _update_help(self) -> None:
        help_widget = self.query_one("#tasks-help", NoMarkupStatic)
        if self._view == "list":
            help_widget.update(
                "1-5 Filter  \u2191\u2193 Navigate  Enter Detail  x Stop  "
                "p Pause  s Save  r Refresh  Esc Back"
            )
        elif self._view == "detail":
            entry = self._focused_entry()
            hints = "x Stop  Esc Back"
            if entry is not None and entry.category == TaskCategory.WORKFLOW:
                hints = "\u2191\u2193 Agents  Enter View  x Stop  p Pause  s Save  o Script  Esc Back"
            help_widget.update(hints)
        elif self._view == "script":
            help_widget.update("Esc Back")
        elif self._view == "agent":
            help_widget.update("Esc Back")

    def action_back(self) -> None:
        if self._view == "agent":
            self._view = "detail"
            self._agent_view_data = None
            self.run_worker(self._render_view(), exclusive=True)
        elif self._view == "script":
            self._view = "detail"
            self.run_worker(self._render_view(), exclusive=True)
        elif self._view == "detail":
            self._view = "list"
            self._selected_task_id = None
            self.run_worker(self._render_view(), exclusive=True)
        else:
            self.post_message(self.Closed())

    def action_stop(self) -> None:
        tid = self._focused_task_id()
        if tid:
            self.post_message(self.TaskStopRequested(tid))

    def action_toggle_pause(self) -> None:
        entry = self._focused_entry()
        if entry is not None and entry.can_pause:
            self.post_message(self.TaskPauseRequested(entry.task_id))

    def action_save(self) -> None:
        if self._view != "detail" or self._selected_task_id is None:
            return
        ref = self._workflow_script_ref()
        if ref is None:
            return
        self.post_message(self.SaveRequested(ref.run_id, ref.script_source, name=None))

    def action_script(self) -> None:
        if self._view != "detail" or self._selected_task_id is None:
            return
        if self._workflow_script_ref() is None:
            return
        self._view = "script"
        self.run_worker(self._render_view(), exclusive=True)

    def action_refresh(self) -> None:
        self._refresh_current_view()

    def _set_filter(self, idx: int) -> None:
        if idx == self._filter_idx:
            return
        self._filter_idx = idx
        if self._view == "list":
            self.run_worker(self._render_view(), exclusive=True)
        else:
            # Switching filter from any drill-down view jumps back to the list.
            self._view = "list"
            self._selected_task_id = None
            self._agent_view_data = None
            self.run_worker(self._render_view(), exclusive=True)

    def action_filter_all(self) -> None:
        self._set_filter(0)

    def action_filter_processes(self) -> None:
        self._set_filter(1)

    def action_filter_workflows(self) -> None:
        self._set_filter(2)

    def action_filter_teams(self) -> None:
        self._set_filter(3)

    def action_filter_loops(self) -> None:
        self._set_filter(4)

    def action_filter_subagents(self) -> None:
        self._set_filter(5)

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if self._view == "list" and event.option.id:
            self._selected_task_id = str(event.option.id)
            self._view = "detail"
            await self._render_view()
        elif (
            self._view == "detail"
            and event.option.id
            and str(event.option.id).startswith("agent:")
        ):
            key = str(event.option.id)[len("agent:") :]
            if self._open_agent_view(key):
                await self._render_view()
