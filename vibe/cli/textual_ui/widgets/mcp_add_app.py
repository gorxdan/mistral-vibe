from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Input, Static

from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from vibe.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput

_FIELD_DEFS: list[tuple[str, str, str]] = [
    ("name", "Server name (alias)", ""),
    ("transport", "Transport: http, streamable-http, or stdio", "http"),
    ("url", "URL (for http / streamable-http)", ""),
    ("command", "Command (for stdio, e.g. 'npx -y @my/mcp-server')", ""),
    ("auth", "Auth: static or oauth (http only)", "static"),
    ("api_key_env", "API key env var (for static auth)", ""),
    ("scopes", "OAuth scopes, comma-separated (for oauth)", ""),
]

_HELP_TEXT = "↑↓ navigate  Enter add & exit  ESC cancel"


class MCPAddApp(Container):
    """Bottom-panel form for adding a new MCP server."""

    can_focus = True
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "focus_previous", "Up", show=False),
        Binding("down", "focus_next", "Down", show=False),
    ]

    class MCPAddClosed(Message):
        def __init__(
            self, *, added: bool, name: str = "", error: str | None = None
        ) -> None:
            super().__init__()
            self.added = added
            self.name = name
            self.error = error

    def __init__(self) -> None:
        super().__init__(id="mcpadd-app")
        self.inputs: dict[str, Input] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="mcpadd-content"):
            yield NoMarkupStatic("Add MCP Server", classes="settings-title")
            for key, label, default in _FIELD_DEFS:
                yield Static(f"[bold $primary]{label}[/]", classes="mcpadd-label")
                widget = VscodeCompatInput(
                    value=default,
                    placeholder=label,
                    id=f"mcpadd-{key}",
                    classes="mcpadd-input",
                )
                self.inputs[key] = widget
                yield widget
            yield NoMarkupStatic(_HELP_TEXT, classes="settings-help")

    def focus(self, scroll_visible: bool = True) -> MCPAddApp:
        if self.inputs:
            list(self.inputs.values())[0].focus(scroll_visible=scroll_visible)
        else:
            super().focus(scroll_visible=scroll_visible)
        return self

    def action_focus_next(self) -> None:
        inputs = list(self.inputs.values())
        focused = self.screen.focused
        if focused is not None and isinstance(focused, Input) and focused in inputs:
            idx = inputs.index(focused)
            inputs[(idx + 1) % len(inputs)].focus()

    def action_focus_previous(self) -> None:
        inputs = list(self.inputs.values())
        focused = self.screen.focused
        if focused is not None and isinstance(focused, Input) and focused in inputs:
            idx = inputs.index(focused)
            inputs[(idx - 1) % len(inputs)].focus()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._save_and_close()

    def on_blur(self, _event: events.Blur) -> None:
        self.call_after_refresh(self._refocus_if_needed)

    def _refocus_if_needed(self) -> None:
        if self.has_focus or any(inp.has_focus for inp in self.inputs.values()):
            return
        self.focus()

    def _build_entry(self) -> dict[str, object]:
        transport = self.inputs["transport"].value.strip() or "http"
        name = self.inputs["name"].value.strip()
        entry: dict[str, object] = {"name": name, "transport": transport}

        if transport == "stdio":
            command = self.inputs["command"].value.strip()
            if command:
                entry["command"] = command
        else:
            url = self.inputs["url"].value.strip()
            if url:
                entry["url"] = url

        auth = self.inputs["auth"].value.strip() or "static"
        if transport != "stdio" and auth == "oauth":
            scopes_raw = self.inputs["scopes"].value.strip()
            scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()]
            entry["auth"] = {"type": "oauth", "scopes": scopes}
        elif transport != "stdio":
            api_key_env = self.inputs["api_key_env"].value.strip()
            if api_key_env:
                entry["auth"] = {"type": "static", "api_key_env": api_key_env}

        return entry

    def _save_and_close(self) -> None:
        name = self.inputs["name"].value.strip()
        if not name:
            self.post_message(
                self.MCPAddClosed(added=False, error="Server name is required.")
            )
            return

        try:
            entry = self._build_entry()
            _validate_mcp_entry(entry)

            from vibe.core.config import VibeConfig as _VC

            persisted = _VC.get_persisted_config()
            servers: list[dict[str, object]] = list(persisted.get("mcp_servers", []))
            servers.append(entry)
            _VC.save_updates({"mcp_servers": servers})
        except Exception as exc:
            self.post_message(self.MCPAddClosed(added=False, name=name, error=str(exc)))
            return

        self.post_message(self.MCPAddClosed(added=True, name=name))

    def action_close(self) -> None:
        self.post_message(self.MCPAddClosed(added=False))


def _validate_mcp_entry(entry: dict[str, object]) -> None:
    from vibe.core.config import MCPHttp, MCPStdio, MCPStreamableHttp

    transport = entry.get("transport", "http")
    match transport:
        case "http":
            MCPHttp.model_validate(entry)
        case "streamable-http":
            MCPStreamableHttp.model_validate(entry)
        case "stdio":
            MCPStdio.model_validate(entry)
        case _:
            raise ValueError(f"Unknown transport: {transport!r}")
