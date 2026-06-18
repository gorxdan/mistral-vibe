from __future__ import annotations

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


class WorkflowsApp(Container):
    """Interactive workflow monitor with drill-down into runs and agents."""

    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", show=False),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("s", "stop", "Stop", show=True),
    ]

    class Closed(Message):
        pass

    class StopRequested(Message):
        run_id: str

        def __init__(self, run_id: str) -> None:
            self.run_id = run_id
            super().__init__()

    def __init__(self, runner: WorkflowRunner, **kwargs: Any) -> None:
        super().__init__(id="workflows-app", **kwargs)
        self._runner = runner
        self._view: str = "list"
        self._selected_run_id: str | None = None
        self._selected_agent_idx: int | None = None
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
        script_preview = Text("Script preview:\n", style="dim")
        script_preview.append("\n".join(script_lines), style="dim")
        await body.mount(
            NoMarkupStatic(script_preview, classes="workflows-script")
        )

        agents = self._get_agent_results(entry)
        if not agents:
            await body.mount(
                NoMarkupStatic(
                    "No agents completed yet.",
                    classes="workflows-empty",
                )
            )
            return

        options = [
            Option(
                _build_agent_option_text(phase_name, result, idx),
                id=f"agent-{idx}",
            )
            for idx, (phase_name, result) in enumerate(agents)
        ]
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
        agents = self._get_agent_results(entry)
        highlighted = option_list.highlighted
        option_list.clear_options()
        option_list.add_options(
            [
                Option(
                    _build_agent_option_text(phase_name, result, idx),
                    id=f"agent-{idx}",
                )
                for idx, (phase_name, result) in enumerate(agents)
            ]
        )
        if highlighted is not None and highlighted < len(agents):
            option_list.highlighted = highlighted

    async def _render_agent_view(self, body: Vertical) -> None:
        entry = self._find_selected_run()
        if entry is None or self._selected_agent_idx is None:
            self._view = "detail"
            await self._render_view()
            return

        agents = self._get_agent_results(entry)
        if self._selected_agent_idx >= len(agents):
            self._view = "detail"
            await self._render_view()
            return

        phase_name, result = agents[self._selected_agent_idx]

        header = self.query_one("#workflows-header", NoMarkupStatic)
        header.update(Text(f"Run {entry.run_id} \u2192 Agent Detail", style="bold"))

        scroll = VerticalScroll(id="workflows-agent-detail")
        await body.mount(scroll)
        await scroll.mount(
            NoMarkupStatic(
                _build_agent_detail(
                    phase_name, result, self._selected_agent_idx
                )
            )
        )

    # --- Helpers ---

    def _find_selected_run(self) -> WorkflowRunEntry | None:
        if self._selected_run_id is None:
            return None
        return self._runner._find_run(self._selected_run_id)

    def _get_agent_results(
        self, entry: WorkflowRunEntry
    ) -> list[tuple[str, AgentResult]]:
        results: list[tuple[str, AgentResult]] = []
        for phase in entry.phase_reports:
            for ar in phase.agent_results:
                results.append((phase.name, ar))
        return results

    def _highlighted_run_id(self) -> str | None:
        try:
            option_list = self.query_one("#workflows-run-list", OptionList)
        except Exception:
            return None
        option = option_list.highlighted_option
        if option is None or option.id is None:
            return None
        return str(option.id)

    def _update_help(self) -> None:
        help_widget = self.query_one("#workflows-help", NoMarkupStatic)
        if self._view == "list":
            help_widget.update(
                "\u2191\u2193 Navigate  Enter Select  s Stop  r Refresh  Esc Back"
            )
        elif self._view == "detail":
            help_widget.update(
                "\u2191\u2193 Navigate  Enter Agent Detail  s Stop  Esc Back to List"
            )
        elif self._view == "agent":
            help_widget.update("Esc Back to Run Detail")

    # --- Actions ---

    def action_back(self) -> None:
        if self._view == "agent":
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
        if self._view == "list":
            run_id = self._highlighted_run_id()
        elif self._view == "detail":
            run_id = self._selected_run_id
        else:
            return
        if run_id:
            self.post_message(self.StopRequested(run_id))

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
