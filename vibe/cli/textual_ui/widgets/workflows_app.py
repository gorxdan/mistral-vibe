from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic

if TYPE_CHECKING:
    from vibe.cli.textual_ui.workflow_runner import WorkflowRunEntry


def _build_run_table(runs: list[WorkflowRunEntry]) -> Table:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Agents", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("Phases")

    for entry in runs:
        elapsed = f"{entry.elapsed:.1f}s"
        phases = ", ".join(entry.phases) or "(none)"
        status_color = {
            "running": "yellow",
            "completed": "green",
            "failed": "red",
            "paused": "blue",
        }.get(entry.status.value, "white")
        table.add_row(
            entry.run_id,
            Text(entry.status.value, style=status_color),
            str(entry.agent_count),
            str(entry.tokens_total),
            elapsed,
            phases,
        )

    return table


class WorkflowsApp(Container):
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Back", show=False),
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

    def __init__(self, runs: list[WorkflowRunEntry], **kwargs: Any) -> None:
        super().__init__(id="workflows-app", **kwargs)
        self._runs = runs

    def compose(self) -> ComposeResult:
        with Vertical(id="workflows-content"):
            yield NoMarkupStatic("Workflow Runs", classes="workflows-title")
            if not self._runs:
                yield NoMarkupStatic(
                    "No workflow runs. Use /<workflow-name> to launch one.",
                    classes="workflows-empty",
                )
            else:
                yield NoMarkupStatic(
                    _build_run_table(self._runs), classes="workflows-table"
                )
            yield NoMarkupStatic(
                "r Refresh  s Stop  Esc Back", classes="workflows-help"
            )

    def on_mount(self) -> None:
        self.query_one(NoMarkupStatic).focus()

    def action_cancel(self) -> None:
        self.post_message(self.Closed())

    def action_refresh(self) -> None:
        table_widget = self.query_one(".workflows-table", NoMarkupStatic)
        table_widget.update(_build_run_table(self._runs))

    def action_stop(self) -> None:
        active = [r for r in self._runs if r.status.value == "running"]
        if active:
            self.post_message(self.StopRequested(active[0].run_id))
