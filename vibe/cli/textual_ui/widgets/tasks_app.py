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

from dataclasses import dataclass
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

# Filter order for the header row + number-key bindings (1=All ... 5=Loops).
_FILTERS: list[tuple[str, TaskCategory | None]] = [
    ("All", None),
    ("Processes", TaskCategory.PROCESS),
    ("Workflows", TaskCategory.WORKFLOW),
    ("Teams", TaskCategory.TEAM),
    ("Loops", TaskCategory.LOOP),
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
        self._view: str = "list"  # list | detail | script
        self._filter_idx = 0  # index into _FILTERS
        self._selected_task_id: str | None = None
        self._poll_timer: Any = None

    # --- lifecycle / compose ---

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

    # --- polling ---

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
        # "script" view is static.

    # --- rendering ---

    async def _render_view(self) -> None:
        body = self.query_one("#tasks-body", Vertical)
        await body.remove_children()
        if self._view == "list":
            await self._render_list_view(body)
        elif self._view == "detail":
            await self._render_detail_view(body)
        elif self._view == "script":
            await self._render_script_view(body)
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
        await scroll.mount(NoMarkupStatic(self._build_detail_text(entry)))

    def _refresh_detail_view(self) -> None:
        if self._view != "detail":
            return
        entry = self._find_selected()
        if entry is None:
            return
        try:
            scroll = self.query_one("#tasks-detail", VerticalScroll)
        except Exception:
            return
        # Replace the single child with refreshed content (cheap; one widget).
        for child in list(scroll.children):
            child.remove()
        scroll.mount(NoMarkupStatic(self._build_detail_text(entry)))

    async def _render_script_view(self, body: Vertical) -> None:
        ref = self._workflow_script_ref()
        scroll = VerticalScroll(id="tasks-script")
        await body.mount(scroll)
        await scroll.mount(
            NoMarkupStatic(
                Text(ref.script_source if ref else "(no script)", style="dim")
            )
        )

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
            text.append(
                "\n\n(In-flight — response not available until the agent finishes.)"
            )
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
        return text

    # --- helpers ---

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
                hints = "x Stop  p Pause  s Save  o Script  Esc Back"
            help_widget.update(hints)
        elif self._view == "script":
            help_widget.update("Esc Back")

    # --- actions ---

    def action_back(self) -> None:
        if self._view == "script":
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
            # Switching filter from a detail view jumps back to the list.
            self._view = "list"
            self._selected_task_id = None
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

    # --- OptionList events ---

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if self._view == "list" and event.option.id:
            self._selected_task_id = str(event.option.id)
            self._view = "detail"
            await self._render_view()
