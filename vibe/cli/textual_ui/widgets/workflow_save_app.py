"""Save-as-command dialog for workflow runs.

A switch-in overlay (same pattern as ApprovalApp) that lets the user name a
run's script and pick where it saves before it becomes a reusable ``/<name>``
command. Prefilled with a name derived from the run id; Tab toggles between
project (``.vibe/workflows/``) and personal (``~/.vibe/workflows/``).
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Static

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput

_LOCATIONS = ("project", "user")
_LOC_LABELS = {
    "project": "Project (.vibe/workflows/)",
    "user": "Personal (~/.vibe/workflows/)",
}


def _default_name(run_id: str) -> str:
    return f"workflow-{run_id.removeprefix('wf-')}"


class WorkflowSaveApp(Container):
    """Name + location picker for saving a run's script as a command."""

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "confirm", "Save", show=False),
        Binding("tab", "toggle_location", "Toggle location", show=False),
    ]

    class SaveConfirmed(Message):
        def __init__(
            self, run_id: str, script_source: str, name: str, location: str
        ) -> None:
            super().__init__()
            self.run_id = run_id
            self.script_source = script_source
            self.name = name
            self.location = location

    class Cancelled(Message):
        def __init__(self, run_id: str) -> None:
            super().__init__()
            self.run_id = run_id

    def __init__(
        self,
        run_id: str,
        script_source: str,
        default_name: str | None = None,
        default_location: str = "project",
        **kwargs: object,
    ) -> None:
        super().__init__(id="workflow-save-app", **kwargs)  # type: ignore[arg-type]
        self.run_id = run_id
        self.script_source = script_source
        self._name_default = default_name or _default_name(run_id)
        self._location = (
            default_location if default_location in _LOCATIONS else "project"
        )

    def _loc_label(self) -> str:
        # Show both options, marking the active one; mirrors Claude Code's
        # Tab-to-toggle save dialog.
        parts = []
        for loc in _LOCATIONS:
            marker = "[*]" if loc == self._location else "[ ]"
            parts.append(f"{marker} {_LOC_LABELS[loc]}")
        return "  ".join(parts)

    def compose(self) -> ComposeResult:
        # Child widgets are created here (not __init__) so the widget can be
        # constructed without an active Textual app — Input() needs one.
        with Vertical(id="workflow-save-content"):
            yield NoMarkupStatic(
                f"Save workflow {self.run_id} as a command", classes="save-title"
            )
            yield NoMarkupStatic("Name:", classes="save-field-label")
            yield VscodeCompatInput(value=self._name_default, id="workflow-save-name")
            yield NoMarkupStatic(
                self._loc_label(), id="workflow-save-location", classes="save-location"
            )
            yield Static(
                "Enter Save  Tab Toggle location  Esc Cancel", classes="save-help"
            )

    async def on_mount(self) -> None:
        name_input = self.query_one("#workflow-save-name", VscodeCompatInput)
        name_input.focus()
        # Pre-select the placeholder name for easy overtyping.
        name_input.cursor_position = len(name_input.value)

    # --- Actions ---

    def action_toggle_location(self) -> None:
        idx = _LOCATIONS.index(self._location)
        self._location = _LOCATIONS[(idx + 1) % len(_LOCATIONS)]
        self.query_one("#workflow-save-location", NoMarkupStatic).update(
            self._loc_label()
        )

    def action_confirm(self) -> None:
        name_input = self.query_one("#workflow-save-name", VscodeCompatInput)
        name = name_input.value.strip() or self._name_default
        self.post_message(
            self.SaveConfirmed(
                run_id=self.run_id,
                script_source=self.script_source,
                name=name,
                location=self._location,
            )
        )

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled(run_id=self.run_id))
