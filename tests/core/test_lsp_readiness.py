from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError
import pytest

from vibe.core.lsp._readiness import LSPReadinessState, build_lsp_readiness
from vibe.core.lsp._server import LanguageServer, ServerConfig
from vibe.core.lsp._types import ServerState


def _server(
    name: str,
    state: ServerState,
    *,
    languages: dict[str, str] | None = None,
    capabilities: dict[str, object] | None = None,
    crash_count: int = 0,
    max_restarts: int = 3,
    error: str | None = None,
) -> LanguageServer:
    server = LanguageServer(
        ServerConfig(
            name=name,
            command=[name],
            languages=languages or {".py": "python"},
            max_restarts=max_restarts,
        )
    )
    server._state = state
    server._capabilities = capabilities or {}
    server._crash_count = crash_count
    server._last_error = error
    return server


def _mapping(*servers: LanguageServer) -> Mapping[str, LanguageServer]:
    return {server.config.name: server for server in servers}


def test_disabled_and_isolated_states_override_runtime_servers() -> None:
    running = _server("pyright", ServerState.RUNNING)

    disabled = build_lsp_readiness(_mapping(running), enabled=False, generation=4)
    isolated = build_lsp_readiness(
        _mapping(running), enabled=True, isolated=True, generation=5
    )

    assert disabled.state is LSPReadinessState.DISABLED
    assert disabled.ready is False
    assert disabled.servers == ()
    assert isolated.state is LSPReadinessState.ISOLATED
    assert isolated.ready is False
    assert isolated.servers == ()


def test_none_and_empty_server_mappings_have_distinct_states() -> None:
    missing = build_lsp_readiness(None, enabled=True, generation=7)
    empty = build_lsp_readiness({}, enabled=True, generation=8)

    assert missing.state is LSPReadinessState.UNINITIALIZED
    assert missing.generation == 7
    assert empty.state is LSPReadinessState.UNCONFIGURED
    assert empty.generation == 8


def test_file_snapshot_uses_caller_selected_server_not_any_matching_server() -> None:
    first = _server("first", ServerState.RUNNING)
    selected = _server(
        "selected",
        ServerState.ERRORED,
        crash_count=3,
        max_restarts=3,
        error="restart cap exhausted",
    )

    snapshot = build_lsp_readiness(
        _mapping(first, selected),
        enabled=True,
        file_path="src/example.py",
        selected_server=selected,
    )

    assert snapshot.state is LSPReadinessState.ERRORED
    assert snapshot.selected_server == "selected"
    assert snapshot.requested_extension == ".py"
    assert snapshot.ready is False
    assert snapshot.can_attempt is False


def test_running_nonmatching_server_does_not_make_file_ready() -> None:
    running = _server("pyright", ServerState.RUNNING)

    snapshot = build_lsp_readiness(
        _mapping(running), enabled=True, file_path="src/main.rs", selected_server=None
    )

    assert snapshot.state is LSPReadinessState.NO_MATCH
    assert snapshot.configured is False
    assert snapshot.ready is False
    assert snapshot.can_attempt is False


@pytest.mark.parametrize(
    ("server_state", "expected"),
    [
        (ServerState.STOPPED, LSPReadinessState.COLD),
        (ServerState.STARTING, LSPReadinessState.STARTING),
        (ServerState.RUNNING, LSPReadinessState.READY),
    ],
)
def test_file_snapshot_reports_live_selected_server_state(
    server_state: ServerState, expected: LSPReadinessState
) -> None:
    server = _server("pyright", server_state, languages={"PY": "python"})

    snapshot = build_lsp_readiness(
        _mapping(server),
        enabled=True,
        file_path="pkg/MODULE.PY",
        selected_server=server,
    )

    assert snapshot.state is expected
    assert snapshot.ready is (server_state is ServerState.RUNNING)
    assert snapshot.can_attempt is True
    assert snapshot.servers[0].extensions == (".py",)


def test_workspace_snapshot_is_degraded_when_only_some_servers_run() -> None:
    running = _server("pyright", ServerState.RUNNING)
    cold = _server("rust-analyzer", ServerState.STOPPED, languages={".rs": "rust"})

    snapshot = build_lsp_readiness(_mapping(running, cold), enabled=True, generation=11)

    assert snapshot.state is LSPReadinessState.DEGRADED
    assert snapshot.ready is True
    assert snapshot.can_attempt is True
    assert snapshot.generation == 11


@pytest.mark.parametrize(
    ("server_state", "crash_count", "expected"),
    [
        (ServerState.RUNNING, 0, LSPReadinessState.READY),
        (ServerState.STARTING, 0, LSPReadinessState.STARTING),
        (ServerState.STOPPED, 0, LSPReadinessState.COLD),
        (ServerState.ERRORED, 3, LSPReadinessState.ERRORED),
    ],
)
def test_workspace_snapshot_aggregates_server_states(
    server_state: ServerState, crash_count: int, expected: LSPReadinessState
) -> None:
    server = _server("pyright", server_state, crash_count=crash_count, max_restarts=3)

    snapshot = build_lsp_readiness(_mapping(server), enabled=True)

    assert snapshot.state is expected


def test_retryable_error_remains_attemptable_but_not_ready() -> None:
    server = _server(
        "pyright",
        ServerState.ERRORED,
        crash_count=1,
        max_restarts=3,
        error="server exited",
    )

    snapshot = build_lsp_readiness(
        _mapping(server), enabled=True, file_path="app.py", selected_server=server
    )

    assert snapshot.state is LSPReadinessState.ERRORED
    assert snapshot.ready is False
    assert snapshot.can_attempt is True
    assert snapshot.servers[0].restartable is True
    assert snapshot.servers[0].error == (
        "Language server failed; inspect the local Vibe log for details."
    )


def test_readiness_redacts_raw_server_error_details() -> None:
    server = _server(
        "python",
        ServerState.ERRORED,
        languages={".py": "python"},
        error="stderr: API_TOKEN=top-secret-value",
    )

    snapshot = build_lsp_readiness(
        {"python": server}, enabled=True, selected_server=server, file_path="main.py"
    )

    assert snapshot.servers[0].error is not None
    assert "top-secret-value" not in snapshot.servers[0].error


def test_operations_are_unknown_until_server_is_running() -> None:
    capabilities = {
        "definitionProvider": True,
        "hoverProvider": {},
        "referencesProvider": False,
        "callHierarchyProvider": {"workDoneProgress": False},
    }
    starting = _server("starting", ServerState.STARTING, capabilities=capabilities)
    running = _server("running", ServerState.RUNNING, capabilities=capabilities)

    snapshot = build_lsp_readiness(_mapping(starting, running), enabled=True)
    by_name = {server.name: server for server in snapshot.servers}

    assert by_name["starting"].operations is None
    assert by_name["running"].operations == (
        "go_to_definition",
        "hover",
        "prepare_call_hierarchy",
        "incoming_calls",
        "outgoing_calls",
    )


def test_running_server_with_no_providers_reports_known_empty_operations() -> None:
    running = _server("running", ServerState.RUNNING)

    snapshot = build_lsp_readiness(_mapping(running), enabled=True)

    assert snapshot.servers[0].operations == ()


def test_readiness_models_are_immutable() -> None:
    snapshot = build_lsp_readiness({}, enabled=True)

    with pytest.raises(ValidationError):
        snapshot.ready = True
