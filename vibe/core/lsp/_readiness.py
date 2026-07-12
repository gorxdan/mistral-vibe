from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from vibe.core.lsp._server import LanguageServer
from vibe.core.lsp._types import ServerState


class LSPReadinessState(StrEnum):
    DISABLED = auto()
    ISOLATED = auto()
    UNINITIALIZED = auto()
    UNCONFIGURED = auto()
    NO_MATCH = auto()
    COLD = auto()
    STARTING = auto()
    READY = auto()
    DEGRADED = auto()
    ERRORED = auto()


class LSPServerReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    state: ServerState
    extensions: tuple[str, ...]
    language_ids: tuple[str, ...]
    ready: bool
    can_attempt: bool
    restartable: bool
    operations: tuple[str, ...] | None
    error: str | None


class LSPReadinessSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: LSPReadinessState
    generation: int
    enabled: bool
    isolated: bool
    requested_path: str | None
    requested_extension: str | None
    selected_server: str | None
    configured: bool
    ready: bool
    can_attempt: bool
    servers: tuple[LSPServerReadiness, ...]
    reason: str


_OPERATION_PROVIDERS = (
    ("go_to_definition", "definitionProvider"),
    ("find_references", "referencesProvider"),
    ("hover", "hoverProvider"),
    ("document_symbol", "documentSymbolProvider"),
    ("workspace_symbol", "workspaceSymbolProvider"),
    ("go_to_implementation", "implementationProvider"),
    ("prepare_call_hierarchy", "callHierarchyProvider"),
    ("incoming_calls", "callHierarchyProvider"),
    ("outgoing_calls", "callHierarchyProvider"),
)


def _normalize_extension(extension: str) -> str:
    normalized = extension.lower().lstrip(".")
    return f".{normalized}" if normalized else ""


def _provider_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return value is not None


def _supported_operations(server: LanguageServer) -> tuple[str, ...] | None:
    if server.state is not ServerState.RUNNING:
        return None
    return tuple(
        operation
        for operation, provider in _OPERATION_PROVIDERS
        if _provider_enabled(server.capabilities.get(provider))
    )


def _can_attempt(server: LanguageServer) -> bool:
    if server.state in {ServerState.RUNNING, ServerState.STARTING}:
        return True
    return not server.restarts_exhausted


def _public_error(error: str | None) -> str | None:
    if error is None:
        return None
    normalized = error.casefold()
    if "not found" in normalized or "no such file" in normalized:
        return "Language server executable was not found."
    if "timed out" in normalized or "timeout" in normalized:
        return "Language server request timed out."
    if "permission" in normalized or "access denied" in normalized:
        return "Language server could not start because access was denied."
    return "Language server failed; inspect the local Vibe log for details."


def _server_readiness(name: str, server: LanguageServer) -> LSPServerReadiness:
    extensions = tuple(
        sorted({
            ext for raw in server.config.languages if (ext := _normalize_extension(raw))
        })
    )
    return LSPServerReadiness(
        name=name,
        state=server.state,
        extensions=extensions,
        language_ids=tuple(sorted(set(server.config.languages.values()))),
        ready=server.state is ServerState.RUNNING,
        can_attempt=_can_attempt(server),
        restartable=not server.restarts_exhausted,
        operations=_supported_operations(server),
        error=_public_error(server.last_error),
    )


def _request(file_path: str | Path | None) -> tuple[str | None, str | None]:
    if file_path is None:
        return None, None
    requested_path = str(file_path)
    extension = _normalize_extension(Path(requested_path).suffix)
    return requested_path, extension or None


def _terminal_snapshot(
    state: LSPReadinessState,
    *,
    generation: int,
    enabled: bool,
    isolated: bool,
    requested_path: str | None,
    requested_extension: str | None,
    reason: str,
) -> LSPReadinessSnapshot:
    return LSPReadinessSnapshot(
        state=state,
        generation=generation,
        enabled=enabled,
        isolated=isolated,
        requested_path=requested_path,
        requested_extension=requested_extension,
        selected_server=None,
        configured=False,
        ready=False,
        can_attempt=False,
        servers=(),
        reason=reason,
    )


def _selected_server_name(
    servers: Mapping[str, LanguageServer], selected_server: LanguageServer
) -> str:
    return next(
        (name for name, candidate in servers.items() if candidate is selected_server),
        selected_server.config.name,
    )


def _file_snapshot(
    statuses: tuple[LSPServerReadiness, ...],
    servers: Mapping[str, LanguageServer],
    selected_server: LanguageServer | None,
    *,
    generation: int,
    requested_path: str,
    requested_extension: str | None,
) -> LSPReadinessSnapshot:
    extension_label = requested_extension or "extensionless"
    if selected_server is None:
        return LSPReadinessSnapshot(
            state=LSPReadinessState.NO_MATCH,
            generation=generation,
            enabled=True,
            isolated=False,
            requested_path=requested_path,
            requested_extension=requested_extension,
            selected_server=None,
            configured=False,
            ready=False,
            can_attempt=False,
            servers=statuses,
            reason=f"No language server is selected for {extension_label} files.",
        )

    selected_name = _selected_server_name(servers, selected_server)
    selected = _server_readiness(selected_name, selected_server)
    if all(status.name != selected_name for status in statuses):
        statuses = (*statuses, selected)

    if selected.state is ServerState.RUNNING:
        state = LSPReadinessState.READY
        reason = f"{selected_name} is running for {extension_label} files."
    elif selected.state is ServerState.STARTING:
        state = LSPReadinessState.STARTING
        reason = f"{selected_name} is starting for {extension_label} files."
    elif selected.state is ServerState.STOPPED and selected.can_attempt:
        state = LSPReadinessState.COLD
        reason = (
            f"{selected_name} is configured for {extension_label} files but has "
            "not started."
        )
    else:
        state = LSPReadinessState.ERRORED
        qualifier = "can retry" if selected.can_attempt else "cannot restart"
        reason = f"{selected_name} errored for {extension_label} files and {qualifier}."

    return LSPReadinessSnapshot(
        state=state,
        generation=generation,
        enabled=True,
        isolated=False,
        requested_path=requested_path,
        requested_extension=requested_extension,
        selected_server=selected_name,
        configured=True,
        ready=selected.ready,
        can_attempt=selected.can_attempt,
        servers=statuses,
        reason=reason,
    )


def _workspace_state(
    statuses: tuple[LSPServerReadiness, ...],
) -> tuple[LSPReadinessState, str]:
    running = sum(status.ready for status in statuses)
    if running == len(statuses):
        return LSPReadinessState.READY, "All configured language servers are running."
    if running:
        return (
            LSPReadinessState.DEGRADED,
            f"{running} of {len(statuses)} configured language servers are running.",
        )
    if any(status.state is ServerState.STARTING for status in statuses):
        return LSPReadinessState.STARTING, "Configured language servers are starting."
    if any(
        status.state is ServerState.STOPPED and status.can_attempt
        for status in statuses
    ):
        return (
            LSPReadinessState.COLD,
            "Language servers are configured but have not started.",
        )
    return LSPReadinessState.ERRORED, "No configured language server is running."


def build_lsp_readiness(
    servers: Mapping[str, LanguageServer] | None,
    *,
    enabled: bool,
    generation: int = 0,
    isolated: bool = False,
    file_path: str | Path | None = None,
    selected_server: LanguageServer | None = None,
) -> LSPReadinessSnapshot:
    """Capture immutable LSP state without discovering or starting servers.

    ``servers=None`` means no manager is active; an empty mapping means an active
    manager has no configured servers. For a file-specific snapshot, callers
    supply the result of their real router as ``selected_server``. This preserves
    precedence when multiple servers claim the same extension.
    """
    requested_path, requested_extension = _request(file_path)
    if not enabled:
        return _terminal_snapshot(
            LSPReadinessState.DISABLED,
            generation=generation,
            enabled=False,
            isolated=False,
            requested_path=requested_path,
            requested_extension=requested_extension,
            reason="The LSP feature is disabled.",
        )
    if isolated:
        return _terminal_snapshot(
            LSPReadinessState.ISOLATED,
            generation=generation,
            enabled=True,
            isolated=True,
            requested_path=requested_path,
            requested_extension=requested_extension,
            reason="LSP is unavailable in isolated worktree execution.",
        )
    if servers is None:
        return _terminal_snapshot(
            LSPReadinessState.UNINITIALIZED,
            generation=generation,
            enabled=True,
            isolated=False,
            requested_path=requested_path,
            requested_extension=requested_extension,
            reason="No LSP manager is active.",
        )
    if not servers:
        return _terminal_snapshot(
            LSPReadinessState.UNCONFIGURED,
            generation=generation,
            enabled=True,
            isolated=False,
            requested_path=requested_path,
            requested_extension=requested_extension,
            reason="No language servers are configured.",
        )

    statuses = tuple(
        _server_readiness(name, server) for name, server in servers.items()
    )
    if requested_path is not None:
        return _file_snapshot(
            statuses,
            servers,
            selected_server,
            generation=generation,
            requested_path=requested_path,
            requested_extension=requested_extension,
        )

    state, reason = _workspace_state(statuses)
    return LSPReadinessSnapshot(
        state=state,
        generation=generation,
        enabled=True,
        isolated=False,
        requested_path=None,
        requested_extension=None,
        selected_server=None,
        configured=True,
        ready=any(status.ready for status in statuses),
        can_attempt=any(status.can_attempt for status in statuses),
        servers=statuses,
        reason=reason,
    )


__all__ = [
    "LSPReadinessSnapshot",
    "LSPReadinessState",
    "LSPServerReadiness",
    "build_lsp_readiness",
]
