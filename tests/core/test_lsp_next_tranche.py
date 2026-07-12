from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

import pytest

from vibe.core.lsp._manager import LSPManager
from vibe.core.lsp._readiness import LSPReadinessState
from vibe.core.lsp._registry import format_diagnostics_for_model
from vibe.core.lsp._server import ServerConfig
from vibe.core.lsp._types import (
    Diagnostic,
    DiagnosticSeverity,
    Position,
    Range,
    ServerState,
    path_from_uri,
    uri_from_path,
)
from vibe.core.tools.base import InvokeContext, ToolError
from vibe.core.tools.builtins.lsp import (
    Lsp,
    LspArgs,
    LspConfig,
    LspOperation,
    LspResult,
    LspState,
)
from vibe.core.utils.io import write_safe


class _ConfigSource:
    def __init__(self, configs: list[ServerConfig]) -> None:
        self._configs = configs

    def load(self) -> list[ServerConfig]:
        return self._configs


class _RecordingServer:
    def __init__(self, response: Any) -> None:
        self.config = ServerConfig(
            name="test", command=["test-server"], languages={".py": "python"}
        )
        self.response = response


class _RecordingManager:
    def __init__(self, response: Any) -> None:
        self.server = _RecordingServer(response)
        self.requests: list[tuple[str, dict[str, Any] | None]] = []

    def get_server_for_file(self, _path: str | Path) -> _RecordingServer:
        return self.server

    async def open_document(self, _path: str, _text: str, _language_id: str) -> None:
        return None

    async def send_request(
        self, _path: str, method: str, params: dict[str, Any] | None = None
    ) -> tuple[Any, _RecordingServer]:
        self.requests.append((method, params))
        return self.server.response, self.server


async def _collect(tool: Lsp, args: LspArgs) -> list[LspResult]:
    events: AsyncGenerator[Any, None] = tool.run(args)
    return [event async for event in events if isinstance(event, LspResult)]


@pytest.mark.asyncio
async def test_lsp_converts_human_column_to_utf16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "emoji.py"
    write_safe(path, "😀target\n")
    manager = _RecordingManager({"contents": "target"})
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    await _collect(
        tool,
        LspArgs(operation=LspOperation.HOVER, file_path=str(path), line=1, character=2),
    )

    hover = next(
        params for method, params in manager.requests if method == "textDocument/hover"
    )
    assert hover is not None
    assert hover["position"] == {"line": 0, "character": 2}


@pytest.mark.asyncio
async def test_lsp_converts_utf16_location_to_human_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "emoji.py"
    write_safe(path, "😀target\n")
    manager = _RecordingManager([
        {
            "uri": uri_from_path(path),
            "range": {
                "start": {"line": 0, "character": 2},
                "end": {"line": 0, "character": 8},
            },
        }
    ])
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)

    async def keep_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return locations

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    monkeypatch.setattr(tool, "_filter_gitignored", keep_locations)

    results = await _collect(
        tool,
        LspArgs(
            operation=LspOperation.GO_TO_DEFINITION,
            file_path=str(path),
            line=1,
            character=2,
        ),
    )

    assert len(results) == 1
    assert f"{path}:1:2" in results[0].summary
    assert results[0].locations[0]["range"]["start"]["character"] == 1


@pytest.mark.parametrize("text,line,character", [("", 1, 1), ("x\n", 2, 1)])
def test_validate_position_accepts_protocol_empty_lines(
    text: str, line: int, character: int
) -> None:
    Lsp._validate_position(Path("sample.py"), line, character, text)


def test_validate_position_does_not_split_unicode_line_separator() -> None:
    Lsp._validate_position(Path("sample.py"), 1, 4, "a\u2028b")


def test_manager_routes_nested_manifest_root_before_server_start(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "services" / "api"
    source = nested / "src" / "main.py"
    source.parent.mkdir(parents=True)
    write_safe(nested / "pyproject.toml", "[project]\nname = 'api'\n")
    write_safe(source, "")
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
        manifest_markers=("pyproject.toml",),
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()

    server = manager.get_server_for_file(source)

    assert server is not None
    assert server.config.root_uri is not None
    assert Path(path_from_uri(server.config.root_uri)) == nested
    assert server.config.cwd == str(nested)


def test_manager_routes_glob_manifest_root(tmp_path: Path) -> None:
    nested = tmp_path / "services" / "api"
    source = nested / "src" / "Main.cs"
    source.parent.mkdir(parents=True)
    write_safe(nested / "Api.csproj", "<Project />\n")
    write_safe(source, "")
    config = ServerConfig(
        name="omnisharp",
        command=["OmniSharp"],
        languages={".cs": "csharp"},
        manifest_markers=("*.csproj",),
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()

    server = manager.get_server_for_file(source)

    assert server is not None
    assert server.config.root_uri is not None
    assert Path(path_from_uri(server.config.root_uri)) == nested


def test_manager_pools_same_language_by_workspace_root(tmp_path: Path) -> None:
    files: list[Path] = []
    for name in ("api", "worker"):
        root = tmp_path / "services" / name
        source = root / "src" / "main.py"
        source.parent.mkdir(parents=True)
        write_safe(root / "pyproject.toml", f"[project]\nname = '{name}'\n")
        write_safe(source, "")
        files.append(source)
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
        manifest_markers=("pyproject.toml",),
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()

    first = manager.get_server_for_file(files[0])
    second = manager.get_server_for_file(files[1])

    assert first is not None and second is not None
    assert first is not second
    assert first.config.root_uri != second.config.root_uri


def test_manager_uses_enclosing_git_root_when_launched_from_subdirectory(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    launch_dir = repository / "packages" / "api"
    source = launch_dir / "src" / "main.py"
    (repository / ".git").mkdir(parents=True)
    source.parent.mkdir(parents=True)
    write_safe(repository / "pyproject.toml", "[project]\nname = 'repo'\n")
    write_safe(source, "")
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
        manifest_markers=("pyproject.toml",),
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(launch_dir)
    manager.initialize()

    server = manager.get_server_for_file(source)

    assert manager.root_path == repository
    assert server is not None and server.config.root_uri is not None
    assert Path(path_from_uri(server.config.root_uri)) == repository


def test_manager_preserves_explicit_workspace_root(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    explicit = tmp_path / "explicit"
    source = nested / "main.py"
    nested.mkdir()
    explicit.mkdir()
    write_safe(nested / "pyproject.toml", "[project]\nname = 'nested'\n")
    write_safe(source, "")
    explicit_uri = uri_from_path(explicit)
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
        root_uri=explicit_uri,
        manifest_markers=("pyproject.toml",),
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()

    server = manager.get_server_for_file(source)

    assert server is not None
    assert server.config.root_uri == explicit_uri
    assert server.config.cwd == str(explicit)


@pytest.mark.asyncio
async def test_warmup_skips_synthetic_marker_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_path / "services" / "api"
    nested.mkdir(parents=True)
    write_safe(nested / "pyproject.toml", "[project]\nname = 'api'\n")
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
        manifest_markers=("pyproject.toml",),
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()
    base_server = manager.servers["pyright"]
    starts = 0

    async def record_start() -> None:
        nonlocal starts
        starts += 1

    monkeypatch.setattr(base_server, "ensure_started", record_start)
    manager.start_warmup()
    assert manager._warmup_task is not None
    await manager._warmup_task

    assert starts == 0


def test_document_symbols_preserve_hierarchy() -> None:
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    raw = [
        {
            "name": "Outer",
            "kind": 5,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 4, "character": 0},
            },
            "selectionRange": {
                "start": {"line": 0, "character": 6},
                "end": {"line": 0, "character": 11},
            },
            "children": [
                {
                    "name": "inner",
                    "kind": 6,
                    "range": {
                        "start": {"line": 1, "character": 4},
                        "end": {"line": 3, "character": 0},
                    },
                    "selectionRange": {
                        "start": {"line": 1, "character": 8},
                        "end": {"line": 1, "character": 13},
                    },
                }
            ],
        }
    ]

    result = tool._format_symbols("Document symbols", raw)

    assert result.symbol_names == ["Outer", "inner"]
    assert [symbol["depth"] for symbol in result.symbols] == [0, 1]
    assert result.symbols[1]["container_path"] == ["Outer"]
    assert "  Outer" in result.summary
    assert "    inner" in result.summary


@pytest.mark.asyncio
async def test_location_continuation_pages_losslessly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.py"
    write_safe(path, "target\n")
    raw = [
        {
            "uri": uri_from_path(tmp_path / f"target-{index}.py"),
            "range": {
                "start": {"line": index, "character": 0},
                "end": {"line": index, "character": 1},
            },
        }
        for index in range(55)
    ]
    manager = _RecordingManager(raw)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)

    async def keep_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return locations

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    monkeypatch.setattr(tool, "_filter_gitignored", keep_locations)
    first_args = LspArgs(
        operation=LspOperation.GO_TO_DEFINITION,
        file_path=str(path),
        line=1,
        character=1,
    )

    first = (await _collect(tool, first_args))[0]
    requests_after_first_page = len(manager.requests)
    assert len(first.locations) == 50
    assert first.continuation_token is not None
    assert first.page_offset == 0
    assert first.has_more

    second = (
        await _collect(
            tool,
            first_args.model_copy(
                update={"continuation_token": first.continuation_token}
            ),
        )
    )[0]

    assert len(second.locations) == 5
    assert second.page_offset == 50
    assert second.continuation_token is None
    assert second.was_truncated
    assert not second.has_more
    assert len(manager.requests) == requests_after_first_page
    assert "showing 51–55" in second.summary
    assert {location["uri"] for location in [*first.locations, *second.locations]} == {
        location["uri"] for location in raw
    }


@pytest.mark.asyncio
async def test_continuation_token_is_bound_to_original_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.py"
    write_safe(path, "target\n")
    raw = [
        {
            "uri": uri_from_path(tmp_path / f"target-{index}.py"),
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
        }
        for index in range(51)
    ]
    manager = _RecordingManager(raw)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)

    async def keep_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return locations

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    monkeypatch.setattr(tool, "_filter_gitignored", keep_locations)
    args = LspArgs(
        operation=LspOperation.GO_TO_DEFINITION,
        file_path=str(path),
        line=1,
        character=1,
    )
    first = (await _collect(tool, args))[0]
    assert first.continuation_token is not None
    requests_after_first_page = len(manager.requests)

    with pytest.raises(ToolError, match="Invalid or expired"):
        await _collect(
            tool,
            args.model_copy(
                update={"character": 2, "continuation_token": first.continuation_token}
            ),
        )
    assert len(manager.requests) == requests_after_first_page


@pytest.mark.asyncio
async def test_continuation_token_is_bound_to_lsp_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.py"
    write_safe(path, "target\n")
    raw = [
        {
            "uri": uri_from_path(tmp_path / f"target-{index}.py"),
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
        }
        for index in range(51)
    ]
    manager = _RecordingManager(raw)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    generation = 1
    monkeypatch.setattr(
        "vibe.core.tools.builtins.lsp.current_lsp_generation", lambda: generation
    )

    async def keep_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return locations

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    monkeypatch.setattr(tool, "_filter_gitignored", keep_locations)
    args = LspArgs(
        operation=LspOperation.GO_TO_DEFINITION,
        file_path=str(path),
        line=1,
        character=1,
    )
    first = (await _collect(tool, args))[0]
    assert first.continuation_token is not None
    requests_after_first_page = len(manager.requests)

    generation = 2
    with pytest.raises(ToolError, match="Invalid or expired"):
        await _collect(
            tool,
            args.model_copy(update={"continuation_token": first.continuation_token}),
        )

    assert len(manager.requests) == requests_after_first_page


@pytest.mark.asyncio
async def test_workspace_symbol_continuation_replays_without_server_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _WorkspaceServer:
        def __init__(self) -> None:
            self.requests = 0

        async def send_request(
            self, _method: str, _params: dict[str, Any]
        ) -> list[dict[str, Any]]:
            self.requests += 1
            return [
                {
                    "name": f"Target{index}",
                    "location": {"uri": uri_from_path(tmp_path / f"target-{index}.py")},
                }
                for index in range(105)
            ]

    server = _WorkspaceServer()
    manager = type(
        "WorkspaceManager", (), {"servers": {"python": server}, "root_path": tmp_path}
    )()
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    args = LspArgs(operation=LspOperation.WORKSPACE_SYMBOL, query="Target")

    first = (await _collect(tool, args))[0]
    second = (
        await _collect(
            tool,
            args.model_copy(update={"continuation_token": first.continuation_token}),
        )
    )[0]

    assert len(first.symbols) == 100
    assert len(second.symbols) == 5
    assert server.requests == 1


@pytest.mark.asyncio
async def test_location_continuation_preserves_normalized_snapshot_after_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    target = tmp_path / "target.py"
    write_safe(source, "target\n")
    write_safe(target, "😀target\n")
    raw = [
        {
            "uri": uri_from_path(target),
            "range": {
                "start": {"line": 0, "character": 2},
                "end": {"line": 0, "character": 8},
            },
        }
        for _ in range(55)
    ]
    manager = _RecordingManager(raw)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)

    async def keep_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return locations

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    monkeypatch.setattr(tool, "_filter_gitignored", keep_locations)
    args = LspArgs(
        operation=LspOperation.GO_TO_DEFINITION,
        file_path=str(source),
        line=1,
        character=1,
    )
    first = (await _collect(tool, args))[0]
    assert first.continuation_token is not None
    assert first.locations[0]["range"]["start"]["character"] == 1

    write_safe(target, "abtarget\n")
    second = (
        await _collect(
            tool,
            args.model_copy(update={"continuation_token": first.continuation_token}),
        )
    )[0]

    assert second.locations[0]["range"]["start"]["character"] == 1
    assert second.locations[0]["position_encoding"] == "unicode-codepoint"


@pytest.mark.asyncio
async def test_task_scope_is_applied_before_first_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    allowed = tmp_path / "allowed.py"
    write_safe(source, "target\n")
    write_safe(allowed, "target\n")
    raw = [
        {
            "uri": uri_from_path(tmp_path / "denied" / f"target-{index}.py"),
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
        }
        for index in range(50)
    ]
    raw.append({
        "uri": uri_from_path(allowed),
        "range": {
            "start": {"line": 0, "character": 0},
            "end": {"line": 0, "character": 1},
        },
    })
    manager = _RecordingManager(raw)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)

    async def keep_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return locations

    class _Contract:
        brief_hash = "bound-task"

        def allows_search_result(self, candidate: str) -> bool:
            return Path(candidate) == allowed

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    monkeypatch.setattr(tool, "_filter_gitignored", keep_locations)
    ctx = InvokeContext(
        tool_call_id="scope", session_id="session", task_contract=cast(Any, _Contract())
    )

    results = [
        item
        async for item in tool.run(
            LspArgs(
                operation=LspOperation.GO_TO_DEFINITION,
                file_path=str(source),
                line=1,
                character=1,
            ),
            ctx,
        )
        if isinstance(item, LspResult)
    ]

    assert len(results) == 1
    assert [location["uri"] for location in results[0].locations] == [
        uri_from_path(allowed)
    ]
    assert results[0].total_count == 1
    assert results[0].continuation_token is None


def test_manager_readiness_requires_running_matching_server(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    write_safe(source, "target\n")
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()
    server = manager.get_server_for_file(source)
    assert server is not None

    cold = manager.readiness(source)
    server._state = ServerState.RUNNING
    server._capabilities = {"definitionProvider": True, "hoverProvider": True}
    ready = manager.readiness(source)
    no_match = manager.readiness(tmp_path / "source.rs")

    assert cold.state is LSPReadinessState.COLD
    assert ready.state is LSPReadinessState.READY
    assert ready.servers[0].operations == ("go_to_definition", "hover")
    assert no_match.state is LSPReadinessState.NO_MATCH
    assert no_match.ready is False


def test_manager_readiness_requires_advertised_operation(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    write_safe(source, "target\n")
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()
    server = manager.get_server_for_file(source)
    assert server is not None
    server._state = ServerState.RUNNING
    server._capabilities = {"definitionProvider": True}

    assert manager.has_running_server_for(
        file_path=source, operation="go_to_definition"
    )
    assert not manager.has_running_server_for(
        file_path=source, operation="workspace_symbol"
    )


@pytest.mark.asyncio
async def test_lsp_status_reports_cold_server_without_starting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    results = await _collect(tool, LspArgs(operation=LspOperation.STATUS))

    assert len(results) == 1
    assert results[0].readiness is not None
    assert results[0].readiness["state"] == "cold"
    assert "pyright: stopped" in results[0].summary
    assert manager.servers["pyright"].state.value == "stopped"


@pytest.mark.asyncio
async def test_lsp_status_does_not_reject_oversized_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "large.py"
    path.write_bytes(b"x" * (11 * 1024 * 1024))
    config = ServerConfig(
        name="pyright",
        command=["pyright-langserver", "--stdio"],
        languages={".py": "python"},
    )
    manager = LSPManager(_ConfigSource([config]))
    manager.set_root(tmp_path)
    manager.initialize()
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    [result] = await _collect(
        tool, LspArgs(operation=LspOperation.STATUS, file_path=str(path))
    )

    assert result.readiness is not None
    assert result.readiness["requested_extension"] == ".py"


@pytest.mark.asyncio
async def test_hierarchical_symbol_continuation_counts_flattened_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "symbols.py"
    write_safe(path, "class Outer:\n    pass\n")
    child_range = {
        "start": {"line": 1, "character": 4},
        "end": {"line": 1, "character": 8},
    }
    raw = [
        {
            "name": "Outer",
            "kind": 5,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 1, "character": 8},
            },
            "selectionRange": {
                "start": {"line": 0, "character": 6},
                "end": {"line": 0, "character": 11},
            },
            "children": [
                {
                    "name": f"member_{index}",
                    "kind": 6,
                    "range": child_range,
                    "selectionRange": child_range,
                }
                for index in range(105)
            ],
        }
    ]
    manager = _RecordingManager(raw)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    args = LspArgs(operation=LspOperation.DOCUMENT_SYMBOL, file_path=str(path))

    first = (await _collect(tool, args))[0]
    second = (
        await _collect(
            tool,
            args.model_copy(update={"continuation_token": first.continuation_token}),
        )
    )[0]

    assert first.total_count == 106
    assert len(first.symbols) == 100
    assert len(second.symbols) == 6
    assert second.page_offset == 100
    assert second.continuation_token is None
    assert all(symbol["container_path"] == ["Outer"] for symbol in second.symbols)


@pytest.mark.asyncio
async def test_call_item_continuation_pages_losslessly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "calls.py"
    write_safe(path, "def target():\n    pass\n")
    raw = [
        {
            "name": f"caller_{index}",
            "uri": uri_from_path(path),
            "range": {
                "start": {"line": 0, "character": 4},
                "end": {"line": 0, "character": 10},
            },
            "selectionRange": {
                "start": {"line": 0, "character": 4},
                "end": {"line": 0, "character": 10},
            },
        }
        for index in range(55)
    ]
    manager = _RecordingManager(raw)
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    args = LspArgs(
        operation=LspOperation.PREPARE_CALL_HIERARCHY,
        file_path=str(path),
        line=1,
        character=5,
    )

    first = (await _collect(tool, args))[0]
    second = (
        await _collect(
            tool,
            args.model_copy(update={"continuation_token": first.continuation_token}),
        )
    )[0]

    assert len(first.locations) == 50
    assert len(second.locations) == 5
    assert second.page_offset == 50
    assert second.continuation_token is None


def test_diagnostic_columns_are_presented_as_codepoints(tmp_path: Path) -> None:
    path = tmp_path / "emoji.py"
    write_safe(path, "😀target\n")
    diagnostic = Diagnostic(
        range=Range(start=Position(0, 2), end=Position(0, 8)),
        severity=DiagnosticSeverity.ERROR,
        message="broken",
    )

    rendered = format_diagnostics_for_model({
        "sources": ["test"],
        "files": [{"path": str(path), "diagnostics": [diagnostic]}],
    })

    assert "line 1, col 2" in rendered


def test_diagnostic_columns_identify_utf16_when_source_is_unavailable(
    tmp_path: Path,
) -> None:
    diagnostic = Diagnostic(
        range=Range(start=Position(0, 2), end=Position(0, 8)),
        severity=DiagnosticSeverity.ERROR,
        message="broken",
    )

    rendered = format_diagnostics_for_model({
        "sources": ["test"],
        "files": [{"path": str(tmp_path / "missing.py"), "diagnostics": [diagnostic]}],
    })

    assert "col 3 (UTF-16 column)" in rendered


@pytest.mark.asyncio
async def test_unreadable_symbol_location_keeps_utf16_provenance() -> None:
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    raw = [
        {
            "name": "target",
            "location": {
                "uri": "file:///definitely-missing/target.py",
                "range": {
                    "start": {"line": 0, "character": 2},
                    "end": {"line": 0, "character": 8},
                },
            },
        }
    ]

    [symbol] = await tool._normalize_symbols(raw, "")
    result = tool._format_symbol_records("Symbols", [symbol])

    assert result.symbols[0]["position_encoding"] == "utf-16"
    assert "[UTF-16 column]" in result.summary


@pytest.mark.asyncio
async def test_malformed_local_location_degrades_without_raw_exception(
    tmp_path: Path,
) -> None:
    path = tmp_path / "target.py"
    write_safe(path, "target\n")
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    [location] = await tool._normalize_location_positions([
        {
            "uri": uri_from_path(path),
            "range": {
                "start": {"line": 0, "character": None},
                "end": {"line": 0, "character": 4},
            },
        }
    ])

    assert location["position_encoding"] == "utf-16"
