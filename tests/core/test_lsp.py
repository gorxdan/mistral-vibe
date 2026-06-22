from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import pytest

from vibe.core.config import VibeConfig
from vibe.core.lsp._jsonrpc import JsonRpcConnection
from vibe.core.lsp._manager import LSPManager
from vibe.core.lsp._registry import DiagnosticRegistry
from vibe.core.lsp._server import LanguageServer, ServerConfig
from vibe.core.lsp._types import (
    Diagnostic,
    DiagnosticSeverity,
    LSPError,
    LSPProtocolError,
    Position,
    Range,
    ServerState,
)


class _NullWriter:
    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


def _feed(reader: asyncio.StreamReader, body: bytes) -> None:
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    reader.feed_data(header + body)


async def _delayed_feed(
    reader: asyncio.StreamReader, body: bytes, delay: float = 0.05
) -> None:
    await asyncio.sleep(delay)
    _feed(reader, body)


@pytest.mark.asyncio
async def test_jsonrpc_request_response_round_trip() -> None:
    reader = asyncio.StreamReader()
    conn = JsonRpcConnection(reader, _NullWriter())
    conn.start()
    asyncio.create_task(
        _delayed_feed(reader, b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}')
    )
    result = await conn.request("test", {"x": 1}, timeout=2.0)
    assert result == {"ok": True}
    await conn.close()


@pytest.mark.asyncio
async def test_jsonrpc_error_response_raises_protocol_error() -> None:
    reader = asyncio.StreamReader()
    conn = JsonRpcConnection(reader, _NullWriter())
    conn.start()
    asyncio.create_task(
        _delayed_feed(
            reader, b'{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"bad"}}'
        )
    )
    with pytest.raises(LSPProtocolError, match="bad"):
        await conn.request("f", timeout=2.0)
    await conn.close()


@pytest.mark.asyncio
async def test_jsonrpc_notification_dispatch() -> None:
    received: list[dict] = []

    async def handler(params: dict) -> None:
        received.append(params)

    reader = asyncio.StreamReader()
    conn = JsonRpcConnection(reader, _NullWriter())
    conn.on_notification("test/event", handler)
    conn.start()
    asyncio.create_task(
        _delayed_feed(
            reader, b'{"jsonrpc":"2.0","method":"test/event","params":{"v":42}}'
        )
    )
    await asyncio.sleep(0.2)
    assert received == [{"v": 42}]
    await conn.close()


def test_diagnostic_registry_dedup_across_turns() -> None:
    registry = DiagnosticRegistry()
    diag = {
        "uri": "file:///tmp/test.py",
        "diagnostics": [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 5},
                },
                "severity": 1,
                "message": "Undefined variable",
                "source": "pyright",
            }
        ],
    }
    registry.publish(diag, "pyright")
    first = registry.consume()
    assert len(first) == 1
    registry.publish(diag, "pyright")
    assert registry.consume() == []


def test_diagnostic_registry_groups_by_file() -> None:
    registry = DiagnosticRegistry()
    for name, msg in [("a.py", "err1"), ("b.py", "warn1")]:
        registry.publish(
            {
                "uri": f"file:///tmp/{name}",
                "diagnostics": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 1},
                        },
                        "severity": 1,
                        "message": msg,
                    }
                ],
            },
            "pyright",
        )
    batches = registry.consume()
    assert len(batches) == 1
    assert len(batches[0]["files"]) == 2


def test_manager_routes_by_extension() -> None:
    config = ServerConfig(
        name="pyright", command=["pyright-langserver"], languages={".py": "python"}
    )
    manager = LSPManager()
    manager._configs = [config]
    manager._servers = {config.name: LanguageServer(config)}
    assert manager.get_server_for_file("foo.py") is not None
    assert manager.get_server_for_file("foo.ts") is None
    assert manager.get_server_for_file("Makefile") is None


def test_manager_multi_language_routing() -> None:
    py = ServerConfig(name="py", command=["x"], languages={".py": "python"})
    ts = ServerConfig(
        name="ts",
        command=["x"],
        languages={".ts": "typescript", ".tsx": "typescriptreact"},
    )
    manager = LSPManager()
    manager._configs = [py, ts]
    manager._servers = {py.name: LanguageServer(py), ts.name: LanguageServer(ts)}
    assert manager.get_server_for_file("app.py").config.name == "py"
    assert manager.get_server_for_file("app.ts").config.name == "ts"
    assert manager.get_server_for_file("app.tsx").config.name == "ts"


def test_server_config_matches_case_insensitive() -> None:
    config = ServerConfig(name="x", command=["x"], languages={".PY": "python"})
    assert config.matches(".py")
    assert config.matches(".PY")
    assert config.language_id_for(".py") == "python"


def test_diagnostic_dedup_key_stable() -> None:
    d = Diagnostic(
        range=Range(start=Position(1, 2), end=Position(1, 5)),
        severity=DiagnosticSeverity.ERROR,
        message="test",
        source="pyright",
        code="E001",
    )
    assert d.dedup_key == d.dedup_key
    assert "test" in d.dedup_key
    assert d.label == "error"


def test_ensure_manager_returns_none_when_not_installed(monkeypatch) -> None:
    from vibe.core.lsp import clear_lsp_manager
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    clear_lsp_manager()
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: False))
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    assert tool._ensure_manager() is None


def test_ensure_manager_lazy_initializes_when_installed(monkeypatch) -> None:
    from vibe.core.lsp import clear_lsp_manager, get_lsp_manager
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    clear_lsp_manager()
    assert get_lsp_manager() is None
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))

    initialized: list[bool] = []

    def fake_setup(config, getter, root):
        initialized.append(True)
        mgr = LSPManager()
        mgr.initialize()
        from vibe.core.lsp._manager import init_lsp_manager

        init_lsp_manager(mgr)
        return mgr

    monkeypatch.setattr("vibe.core.lsp._lifecycle.setup_lsp_for_config", fake_setup)
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    mgr = tool._ensure_manager()
    assert mgr is not None
    assert initialized == [True]
    tool._ensure_manager()
    assert initialized == [True]
    clear_lsp_manager()


def test_preset_probe_returns_false_when_binary_missing(monkeypatch) -> None:
    from vibe.core.lsp import _defaults
    from vibe.core.lsp._defaults import PRESETS, preset_probe_passes

    monkeypatch.setattr(_defaults.shutil, "which", lambda _: None)
    assert preset_probe_passes(PRESETS["rust"]) is False


def test_preset_probe_returns_false_when_version_check_exits_nonzero(
    monkeypatch,
) -> None:
    from vibe.core.lsp import _defaults
    from vibe.core.lsp._defaults import PRESETS, preset_probe_passes

    monkeypatch.setattr(_defaults.shutil, "which", lambda _: "/fake/bin")
    monkeypatch.setattr(
        _defaults.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="boom", stdout=""),
    )
    assert preset_probe_passes(PRESETS["rust"]) is False


def test_preset_probe_returns_true_when_version_check_exits_zero(monkeypatch) -> None:
    from vibe.core.lsp import _defaults
    from vibe.core.lsp._defaults import PRESETS, preset_probe_passes

    monkeypatch.setattr(_defaults.shutil, "which", lambda _: "/fake/bin")
    monkeypatch.setattr(
        _defaults.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stderr="", stdout="ok"),
    )
    assert preset_probe_passes(PRESETS["rust"]) is True


def test_available_presets_excludes_pathed_but_broken_preset(monkeypatch) -> None:
    from vibe.core.lsp import _defaults
    from vibe.core.lsp._defaults import available_presets

    which_map = {"rust-analyzer": "/fake/rust-analyzer", "pyright": "/fake/pyright"}

    def fake_run(cmd: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        if cmd[0] == "rust-analyzer":
            return SimpleNamespace(returncode=1, stderr="rustup-proxy: boom", stdout="")
        return SimpleNamespace(returncode=0, stderr="", stdout="ok")

    monkeypatch.setattr(_defaults.shutil, "which", lambda name: which_map.get(name))
    monkeypatch.setattr(_defaults.subprocess, "run", fake_run)
    keys = {p.key for p in available_presets()}
    assert "rust" not in keys
    assert "pyright" in keys


@pytest.mark.asyncio
async def test_start_surfaces_server_stderr_on_crash() -> None:
    config = ServerConfig(
        name="boom",
        command=["sh", "-c", "echo rustup-proxy-error >&2; exit 1"],
        languages={".rs": "rust"},
        startup_timeout=3.0,
    )
    server = LanguageServer(config)
    with pytest.raises(LSPError, match="rustup-proxy-error"):
        await server.start()
    assert "rustup-proxy-error" in (server.last_error or "")
    await server.stop()


def test_restarts_exhausted_after_crash_count_reaches_max() -> None:
    from vibe.core.lsp._types import LSPServerCrashedError

    config = ServerConfig(
        name="boom", command=["x"], languages={".py": "python"}, max_restarts=2
    )
    server = LanguageServer(config)
    server._crash_count = 2
    assert server.restarts_exhausted is True
    server._state = ServerState.ERRORED
    with pytest.raises(LSPServerCrashedError, match="restart cap"):
        asyncio.run(server.ensure_started())


def test_max_restarts_defaults_to_three() -> None:
    config = ServerConfig(name="x", command=["x"], languages={".py": "python"})
    assert config.max_restarts == 3


def test_current_lsp_generation_bumps_on_init() -> None:
    from vibe.core.lsp._manager import (
        clear_lsp_manager,
        current_lsp_generation,
        init_lsp_manager,
    )

    clear_lsp_manager()
    before = current_lsp_generation()
    first = init_lsp_manager(LSPManager())
    second = init_lsp_manager(LSPManager())
    assert first == before + 1
    assert second == before + 2
    clear_lsp_manager()


@pytest.mark.asyncio
async def test_setup_skips_install_when_superseded(monkeypatch, tmp_path) -> None:
    from typing import ClassVar, cast

    from vibe.core.config import VibeConfig
    from vibe.core.lsp import _lifecycle as lifecycle
    from vibe.core.lsp._lifecycle import setup_lsp_for_config
    from vibe.core.lsp._manager import init_lsp_manager

    class _StubConfig:
        installed_components: ClassVar[list[str]] = ["lsp"]

    config = cast(VibeConfig, _StubConfig())
    winner = LSPManager()
    init_lsp_manager(winner)
    started = lifecycle.current_lsp_generation()
    calls: list[int] = []

    def fake_generation() -> int:
        calls.append(1)
        return started if len(calls) == 1 else started + 1

    monkeypatch.setattr(lifecycle, "current_lsp_generation", fake_generation)
    result = setup_lsp_for_config(config, lambda: config, tmp_path)
    assert result is winner


def test_jsonrpc_trace_flag_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VIBE_LSP_TRACE", raising=False)
    import importlib

    from vibe.core.lsp import _jsonrpc as jsonrpc_mod

    importlib.reload(jsonrpc_mod)
    assert jsonrpc_mod._TRACE is False


@pytest.mark.asyncio
async def test_filter_gitignored_drops_gitignored_locations(
    tmp_path, monkeypatch
) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("build/\n")
    ignored = tmp_path / "build" / "target.py"
    ignored.parent.mkdir()
    ignored.write_text("x = 1\n")
    kept = tmp_path / "src.py"
    kept.write_text("y = 2\n")
    monkeypatch.chdir(tmp_path)

    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    locations = [
        {"uri": ignored.as_uri(), "range": {"start": {"line": 0, "character": 0}}},
        {"uri": kept.as_uri(), "range": {"start": {"line": 0, "character": 0}}},
    ]
    filtered = await tool._filter_gitignored(locations)
    uris = {loc["uri"] for loc in filtered}
    assert kept.as_uri() in uris
    assert ignored.as_uri() not in uris


@pytest.mark.asyncio
async def test_filter_gitignored_fails_open_when_no_git(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    locations = [{"uri": (tmp_path / "a.py").as_uri(), "range": {"start": {"line": 0}}}]
    filtered = await tool._filter_gitignored(locations)
    assert filtered == locations


@pytest.mark.asyncio
async def test_filter_gitignored_with_out_of_repo_path_in_batch(
    tmp_path, tmp_path_factory, monkeypatch
) -> None:
    """A single out-of-repo path (e.g. a definition resolving into
    site-packages) must not disable filtering for the rest of the batch.
    Regression: ``git check-ignore`` exits 128 on any out-of-tree path, which
    previously read as "nothing ignored" and leaked the whole batch.
    """
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("build/\n")
    ignored = tmp_path / "build" / "target.py"
    ignored.parent.mkdir()
    ignored.write_text("x = 1\n")
    kept = tmp_path / "src.py"
    kept.write_text("y = 2\n")
    outside_dir = tmp_path_factory.mktemp("outside_repo")
    outside = outside_dir / "site_package_def.py"
    outside.write_text("z = 3\n")
    monkeypatch.chdir(tmp_path)

    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    locations = [
        {"uri": outside.as_uri()},
        {"uri": ignored.as_uri()},
        {"uri": kept.as_uri()},
    ]
    filtered = await tool._filter_gitignored(locations)
    uris = {loc["uri"] for loc in filtered}
    assert outside.as_uri() in uris
    assert kept.as_uri() in uris
    assert ignored.as_uri() not in uris


@pytest.mark.asyncio
async def test_resolve_path_rejects_oversized_file(tmp_path) -> None:
    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    big = tmp_path / "huge.py"
    big.write_bytes(b"a" * (11 * 1024 * 1024))
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    with pytest.raises(ToolError, match="MiB"):
        tool._resolve_path(str(big))


class _FakeCallHierarchyManager:
    """Replays canned responses keyed by (method, position) and records the
    sequence of requests so tests assert retry behavior.
    """

    def __init__(
        self,
        prepare_responses: dict[tuple[int, int], list[dict]],
        document_symbols: list[dict],
        call_edges: dict[str, list[dict]] | None = None,
    ) -> None:
        self._prepare = prepare_responses
        self._symbols = document_symbols
        self._edges = call_edges or {}
        self.requests: list[tuple[str, dict]] = []

    async def send_request(self, file_path: str, method: str, params: dict):
        self.requests.append((method, params))
        if method == "textDocument/prepareCallHierarchy":
            pos = params.get("position") or {}
            return self._prepare.get(
                (pos.get("line", -1), pos.get("character", -1)), []
            ), None
        if method == "textDocument/documentSymbol":
            return self._symbols, None
        if method in {"callHierarchy/incomingCalls", "callHierarchy/outgoingCalls"}:
            item = (params.get("item") or {}).get("name", "")
            return self._edges.get(item, []), None
        return None, None


def _make_lsp_tool():
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    return Lsp(config_getter=lambda: LspConfig(), state=LspState())


def _fn_symbol(name: str, sel_start: tuple[int, int], rng_end: tuple[int, int]):
    # A DocumentSymbol whose identifier (selectionRange) is at sel_start.
    return {
        "name": name,
        "kind": 12,
        "range": {
            "start": {"line": sel_start[0], "character": 0},
            "end": {"line": rng_end[0], "character": rng_end[1]},
        },
        "selectionRange": {
            "start": {"line": sel_start[0], "character": sel_start[1]},
            "end": {"line": sel_start[0], "character": sel_start[1] + len(name)},
        },
    }


@pytest.mark.asyncio
async def test_call_hierarchy_resolves_off_identifier_position_and_retries() -> None:
    # The agent passed character=1 (on the `fn`/`def` keyword), not on the
    # identifier. prepareCallHierarchy at (10,0) returns []; the tool resolves
    # via documentSymbol to the identifier at (10,4) and retries successfully.
    from vibe.core.tools.builtins.lsp import LspOperation

    tool = _make_lsp_tool()
    # identifier `foo` starts at line 10 (0-based), char 4.
    symbols = [_fn_symbol("foo", (10, 4), (12, 0))]
    manager = _FakeCallHierarchyManager(
        prepare_responses={(10, 4): [{"name": "foo"}]},
        document_symbols=symbols,
        call_edges={"foo": [{"from": {"name": "caller", "uri": "file:///x.rs"}}]},
    )
    args = type(
        "A", (), {"operation": LspOperation.INCOMING_CALLS, "file_path": "/x.rs"}
    )()
    result = await tool._call_hierarchy(
        manager,
        args,
        "/x.rs",
        {"textDocument": {"uri": "file:///x.rs"}},
        {"line": 10, "character": 0},
    )
    # First prepare at (10,0) empty -> documentSymbol -> retry prepare at (10,4).
    prepare_positions = [
        p["position"]
        for m, p in manager.requests
        if m == "textDocument/prepareCallHierarchy"
    ]
    assert prepare_positions == [
        {"line": 10, "character": 0},
        {"line": 10, "character": 4},
    ]
    # Incoming edge carries `from` (the caller) per LSP spec.
    assert result.locations
    assert any("caller" in (loc.get("name") or "") for loc in result.locations)


@pytest.mark.asyncio
async def test_call_hierarchy_no_retry_when_position_already_on_identifier() -> None:
    # Position lands exactly on the identifier: prepare succeeds first try, no
    # documentSymbol lookup, no retry.
    from vibe.core.tools.builtins.lsp import LspOperation

    tool = _make_lsp_tool()
    symbols = [_fn_symbol("foo", (10, 4), (12, 0))]
    manager = _FakeCallHierarchyManager(
        prepare_responses={(10, 4): [{"name": "foo"}]},
        document_symbols=symbols,
        call_edges={"foo": [{"from": {"name": "caller", "uri": "file:///x.rs"}}]},
    )
    args = type("A", (), {"operation": LspOperation.INCOMING_CALLS})()
    result = await tool._call_hierarchy(
        manager,
        args,
        "/x.rs",
        {"textDocument": {"uri": "file:///x.rs"}},
        {"line": 10, "character": 4},
    )
    methods = [m for m, _ in manager.requests]
    assert methods == [
        "textDocument/prepareCallHierarchy",
        "callHierarchy/incomingCalls",
    ]
    assert result.locations


@pytest.mark.asyncio
async def test_outgoing_calls_extract_to_field_not_from() -> None:
    # Locks the corrected direction: CallHierarchyOutgoingCall carries `to`
    # (the callee). A stray `from` must be ignored.
    from vibe.core.tools.builtins.lsp import LspOperation

    tool = _make_lsp_tool()
    symbols = [_fn_symbol("foo", (2, 4), (4, 0))]
    manager = _FakeCallHierarchyManager(
        prepare_responses={(2, 4): [{"name": "foo"}]},
        document_symbols=symbols,
        call_edges={
            "foo": [
                {"to": {"name": "bar", "uri": "file:///x.rs"}},
                {"from": {"name": "stray", "uri": "file:///x.rs"}},
            ]
        },
    )
    args = type("A", (), {"operation": LspOperation.OUTGOING_CALLS})()
    result = await tool._call_hierarchy(
        manager,
        args,
        "/x.rs",
        {"textDocument": {"uri": "file:///x.rs"}},
        {"line": 2, "character": 4},
    )
    names = {loc.get("name") for loc in result.locations}
    assert "bar" in names
    assert "stray" not in names


@pytest.mark.asyncio
async def test_call_hierarchy_actionable_message_when_genuinely_empty() -> None:
    # No callable anywhere at the position and documentSymbol finds nothing
    # spanning it: surface an actionable fallback message pointing to
    # find_references instead of a bare "no call hierarchy at position".
    from vibe.core.tools.builtins.lsp import LspOperation

    tool = _make_lsp_tool()
    manager = _FakeCallHierarchyManager(prepare_responses={}, document_symbols=[])
    args = type("A", (), {"operation": LspOperation.INCOMING_CALLS})()
    result = await tool._call_hierarchy(
        manager,
        args,
        "/x.rs",
        {"textDocument": {"uri": "file:///x.rs"}},
        {"line": 99, "character": 0},
    )
    assert "no callable at line 100" in result.summary
    assert "find_references" in result.summary


def test_range_contains_bounds() -> None:
    from vibe.core.tools.builtins.lsp import Lsp

    rng = {"start": {"line": 5, "character": 4}, "end": {"line": 9, "character": 1}}
    assert Lsp._range_contains(rng, {"line": 5, "character": 4})
    assert Lsp._range_contains(rng, {"line": 7, "character": 0})
    assert Lsp._range_contains(rng, {"line": 9, "character": 1})
    assert not Lsp._range_contains(rng, {"line": 5, "character": 3})
    assert not Lsp._range_contains(rng, {"line": 9, "character": 2})
    assert not Lsp._range_contains(rng, {"line": 4, "character": 9})


def test_deepest_symbol_at_picks_innermost() -> None:
    from vibe.core.tools.builtins.lsp import Lsp

    inner = _fn_symbol("inner", (3, 8), (3, 20))
    outer = _fn_symbol("outer", (1, 0), (5, 0))
    outer["children"] = [inner]
    node = Lsp._deepest_symbol_at([outer], {"line": 3, "character": 10})
    assert node is not None
    assert node["name"] == "inner"


@pytest.mark.asyncio
async def test_notify_file_changed_skips_oversized_text(monkeypatch) -> None:
    from vibe.core.lsp import _integration as integration

    called: list[bool] = []

    class _FakeManager:
        def get_server_for_file(self, path):
            return None

    monkeypatch.setattr(integration, "get_lsp_manager", lambda: _FakeManager())
    await integration.notify_file_changed("/tmp/x.py", "x" * (11 * 1024 * 1024))
    assert called == []


def _config_without_lsp() -> VibeConfig:
    return cast(VibeConfig, SimpleNamespace(installed_components=[]))


def _config_with_lsp() -> VibeConfig:
    return cast(VibeConfig, SimpleNamespace(installed_components=["lsp"]))


def test_nudge_returns_install_hint_when_server_binary_absent(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.lsp import _nudge as nudge
    from vibe.core.lsp._defaults import _RUST_ANALYZER

    # No presets available, none broken -> rust binary absent.
    monkeypatch.setattr(nudge, "available_presets", lambda: [])
    monkeypatch.setattr(nudge, "broken_presets", lambda: [])
    monkeypatch.setattr(nudge, "preset_for_extension", lambda ext: _RUST_ANALYZER)
    # Fresh nudge state.
    monkeypatch.setattr(nudge, "_read_nudge_state", lambda _: {})
    decision = nudge.evaluate_nudge(
        "/x/main.rs", _config_without_lsp(), tmp_path / "cache.json"
    )
    assert decision.kind == "install_hint"
    assert decision.install_hint == _RUST_ANALYZER.install_hint
    assert decision.preset_display_name == _RUST_ANALYZER.display_name


def test_nudge_returns_skip_when_server_binary_broken(monkeypatch, tmp_path) -> None:
    """Half-installed servers belong in /lsp status, not a passive nudge."""
    from vibe.core.lsp import _nudge as nudge
    from vibe.core.lsp._defaults import _RUST_ANALYZER, PresetProbe

    monkeypatch.setattr(nudge, "available_presets", lambda: [])
    monkeypatch.setattr(
        nudge,
        "broken_presets",
        lambda: [PresetProbe(preset=_RUST_ANALYZER, status="broken")],
    )
    monkeypatch.setattr(nudge, "preset_for_extension", lambda ext: _RUST_ANALYZER)
    monkeypatch.setattr(nudge, "_read_nudge_state", lambda _: {})
    decision = nudge.evaluate_nudge(
        "/x/main.rs", _config_without_lsp(), tmp_path / "cache.json"
    )
    assert decision.kind == "skip"


def test_nudge_returns_first_prompt_when_server_available_and_lsp_off(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.lsp import _nudge as nudge
    from vibe.core.lsp._defaults import _RUST_ANALYZER

    monkeypatch.setattr(nudge, "available_presets", lambda: [_RUST_ANALYZER])
    monkeypatch.setattr(nudge, "broken_presets", lambda: [])
    monkeypatch.setattr(nudge, "preset_for_extension", lambda ext: _RUST_ANALYZER)
    monkeypatch.setattr(nudge, "_read_nudge_state", lambda _: {})
    decision = nudge.evaluate_nudge(
        "/x/main.rs", _config_without_lsp(), tmp_path / "cache.json"
    )
    assert decision.kind == "first_prompt"


def test_nudge_returns_skip_when_lsp_already_installed(monkeypatch, tmp_path) -> None:
    from vibe.core.lsp import _nudge as nudge
    from vibe.core.lsp._defaults import _RUST_ANALYZER

    monkeypatch.setattr(nudge, "available_presets", lambda: [_RUST_ANALYZER])
    monkeypatch.setattr(nudge, "broken_presets", lambda: [])
    monkeypatch.setattr(nudge, "preset_for_extension", lambda ext: _RUST_ANALYZER)
    decision = nudge.evaluate_nudge(
        "/x/main.rs", _config_with_lsp(), tmp_path / "cache.json"
    )
    assert decision.kind == "skip"


def test_install_hint_nudge_silent_after_decline_within_interval(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.lsp import _nudge as nudge
    from vibe.core.lsp._defaults import _RUST_ANALYZER

    monkeypatch.setattr(nudge, "available_presets", lambda: [])
    monkeypatch.setattr(nudge, "broken_presets", lambda: [])
    monkeypatch.setattr(nudge, "preset_for_extension", lambda ext: _RUST_ANALYZER)
    monkeypatch.setattr(
        nudge,
        "_read_nudge_state",
        lambda _: {"hint_declined:rust": True, "hint_shown:rust": 1},
    )
    decision = nudge.evaluate_nudge(
        "/x/main.rs", _config_without_lsp(), tmp_path / "cache.json", turns_since_last=2
    )
    assert decision.kind == "silent"


def test_install_hint_nudge_reminder_after_decline_and_interval(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.lsp import _nudge as nudge
    from vibe.core.lsp._defaults import _RUST_ANALYZER

    monkeypatch.setattr(nudge, "available_presets", lambda: [])
    monkeypatch.setattr(nudge, "broken_presets", lambda: [])
    monkeypatch.setattr(nudge, "preset_for_extension", lambda ext: _RUST_ANALYZER)
    monkeypatch.setattr(
        nudge,
        "_read_nudge_state",
        lambda _: {"hint_declined:rust": True, "hint_shown:rust": 1},
    )
    decision = nudge.evaluate_nudge(
        "/x/main.rs",
        _config_without_lsp(),
        tmp_path / "cache.json",
        turns_since_last=nudge.REMINDER_INTERVAL_TURNS,
    )
    assert decision.kind == "install_hint"


def test_broken_presets_returns_only_broken(monkeypatch) -> None:
    from vibe.core.lsp import _defaults as defaults
    from vibe.core.lsp._defaults import _PYRIGHT, _RUST_ANALYZER, PresetProbe

    probes = [
        PresetProbe(preset=_PYRIGHT, status="available"),
        PresetProbe(
            preset=_RUST_ANALYZER,
            status="broken",
            returncode=1,
            stderr="rustup-proxy: Unknown binary",
        ),
    ]
    monkeypatch.setattr(
        defaults,
        "_probe",
        lambda preset: next(
            (p for p in probes if p.preset.key == preset.key),
            PresetProbe(preset=preset, status="absent"),
        ),
    )
    broken = defaults.broken_presets()
    assert [p.preset.key for p in broken] == ["rust"]
    assert "rustup-proxy" in broken[0].stderr


def test_preset_states_returns_all_in_declaration_order(monkeypatch) -> None:
    from vibe.core.lsp import _defaults as defaults
    from vibe.core.lsp._defaults import _PYRIGHT, PresetProbe

    monkeypatch.setattr(
        defaults, "_probe", lambda preset: PresetProbe(preset=preset, status="absent")
    )
    states = defaults.preset_states()
    assert [p.status for p in states] == ["absent"] * len(states)
    assert states[0].preset.key == _PYRIGHT.key
