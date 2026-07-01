from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.conftest import build_test_vibe_config
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
async def test_jsonrpc_method_not_found_error_carries_code() -> None:
    reader = asyncio.StreamReader()
    conn = JsonRpcConnection(reader, _NullWriter())
    conn.start()
    asyncio.create_task(
        _delayed_feed(
            reader,
            b'{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"method not found"}}',
        )
    )
    with pytest.raises(LSPProtocolError) as exc_info:
        await conn.request("f", timeout=2.0)
    assert exc_info.value.code == -32601
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


@pytest.mark.asyncio
async def test_on_publish_is_async_and_does_not_raise() -> None:
    manager = LSPManager()
    coro = manager._on_publish(
        {"uri": "file:///tmp/test.py", "diagnostics": []}, "pyright"
    )
    assert hasattr(coro, "__await__")
    await coro


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


def test_method_not_found_surfaces_actionable_tool_error(monkeypatch, tmp_path) -> None:
    import asyncio

    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspState,
    )

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    class _FakeServer:
        def __init__(self) -> None:
            self.config = ServerConfig(
                name="pyright", command=["x"], languages={".py": "python"}
            )

    class _FakeManager:
        def __init__(self) -> None:
            self.server = _FakeServer()

        def get_server_for_file(self, path):
            return self.server

        async def open_document(self, path, text, language_id):
            pass

    monkeypatch.setattr(
        "vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: _FakeManager()
    )
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))

    async def fake_dispatch(self, manager, args, file_path, position):
        raise LSPProtocolError("method not found", code=-32601)

    monkeypatch.setattr(Lsp, "_dispatch", fake_dispatch)

    tmp = tmp_path / "test.py"
    tmp.write_text("x = 1\n")
    args = LspArgs(
        operation=LspOperation.GO_TO_IMPLEMENTATION,
        file_path=str(tmp),
        line=1,
        character=1,
    )

    def _run() -> None:
        async def _inner() -> None:
            async for _ in tool.run(args):
                pass

        asyncio.run(_inner())

    try:
        _run()
    except ToolError as exc:
        assert "does not support" in str(exc)
        assert "find_references" in str(exc)
        assert "pyright" in str(exc)
    else:
        raise AssertionError("expected ToolError")


def test_method_not_found_implementation_does_not_promise_caller_callee(
    monkeypatch, tmp_path
) -> None:
    # go_to_implementation finds concrete overrides of an interface, not
    # "caller/callee info". The method-not-found fallback must not promise
    # call-graph data it cannot deliver; find_references (usages) is the real
    # fallback and workspace_symbol locates subclasses by name.
    import asyncio

    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspState,
    )

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    class _FakeServer:
        def __init__(self) -> None:
            self.config = ServerConfig(
                name="pyright", command=["x"], languages={".py": "python"}
            )

    class _FakeManager:
        def __init__(self) -> None:
            self.server = _FakeServer()

        def get_server_for_file(self, path):
            return self.server

        async def open_document(self, path, text, language_id):
            pass

    monkeypatch.setattr(
        "vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: _FakeManager()
    )
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))

    async def fake_dispatch(self, manager, args, file_path, position):
        raise LSPProtocolError("method not found", code=-32601)

    monkeypatch.setattr(Lsp, "_dispatch", fake_dispatch)

    tmp = tmp_path / "test.py"
    tmp.write_text("x = 1\n")
    args = LspArgs(
        operation=LspOperation.GO_TO_IMPLEMENTATION,
        file_path=str(tmp),
        line=1,
        character=1,
    )

    def _run() -> None:
        async def _inner() -> None:
            async for _ in tool.run(args):
                pass

        asyncio.run(_inner())

    try:
        _run()
    except ToolError as exc:
        msg = str(exc)
        assert "go_to_implementation" in msg
        assert "caller/callee" not in msg
        assert "find_references" in msg
    else:
        raise AssertionError("expected ToolError")


def test_resolve_manifest_root_finds_nearest_marker_dir(tmp_path) -> None:
    from pathlib import Path

    cargo_dir = tmp_path / "backend"
    src_dir = cargo_dir / "src"
    src_dir.mkdir(parents=True)
    (cargo_dir / "Cargo.toml").write_text("[package]\nname='x'\n")
    lib_file = src_dir / "lib.rs"
    lib_file.write_text("")
    manager = LSPManager()
    manager.set_root(tmp_path)
    found = manager._resolve_manifest_root(Path(lib_file), ("Cargo.toml",), tmp_path)
    assert found == cargo_dir


def test_resolve_manifest_root_falls_back_to_default_when_no_marker(tmp_path) -> None:
    from pathlib import Path

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    lib_file = src_dir / "lib.rs"
    lib_file.write_text("")
    manager = LSPManager()
    manager.set_root(tmp_path)
    found = manager._resolve_manifest_root(Path(lib_file), ("Cargo.toml",), tmp_path)
    assert found == tmp_path


def test_resolve_manifest_root_returns_default_when_no_markers(tmp_path) -> None:
    from pathlib import Path

    manager = LSPManager()
    manager.set_root(tmp_path)
    found = manager._resolve_manifest_root(Path(tmp_path / "x.rs"), (), tmp_path)
    assert found == tmp_path


def test_rust_preset_carries_cargo_manifest_marker() -> None:
    from vibe.core.lsp._defaults import _RUST_ANALYZER

    assert "Cargo.toml" in _RUST_ANALYZER.server.manifest_markers


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
    py_server = manager.get_server_for_file("app.py")
    ts_server = manager.get_server_for_file("app.ts")
    tsx_server = manager.get_server_for_file("app.tsx")
    assert py_server is not None and py_server.config.name == "py"
    assert ts_server is not None and ts_server.config.name == "ts"
    assert tsx_server is not None and tsx_server.config.name == "ts"


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
        # _probe now passes the resolved absolute path (_resolve_binary output),
        # so match on the basename rather than the bare command name.
        if cmd[0].endswith("rust-analyzer"):
            return SimpleNamespace(returncode=1, stderr="rustup-proxy: boom", stdout="")
        return SimpleNamespace(returncode=0, stderr="", stdout="ok")

    monkeypatch.setattr(_defaults.shutil, "which", lambda name: which_map.get(name))
    monkeypatch.setattr(_defaults.subprocess, "run", fake_run)
    keys = {p.key for p in available_presets()}
    assert "rust" not in keys
    assert "pyright" in keys


def test_preset_matches_root_requires_manifest_marker(tmp_path) -> None:
    from vibe.core.lsp._defaults import _GOPLS, _PYRIGHT, preset_matches_root

    assert preset_matches_root(_PYRIGHT, tmp_path) is False
    (tmp_path / "pyproject.toml").write_text("")
    assert preset_matches_root(_PYRIGHT, tmp_path) is True
    assert preset_matches_root(_GOPLS, tmp_path) is False
    (tmp_path / "go.mod").write_text("")
    assert preset_matches_root(_GOPLS, tmp_path) is True


def test_build_server_configs_filters_to_project_languages(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.lsp import _config_bridge, _defaults
    from vibe.core.lsp._config_bridge import build_server_configs
    from vibe.core.lsp._defaults import _GOPLS, _PYRIGHT, _RUST_ANALYZER

    probed: list[str] = []

    def preset_probe_passes(preset) -> bool:
        probed.append(preset.key)
        return True

    monkeypatch.setattr(
        _config_bridge,
        "available_presets",
        lambda root_path=None: [
            p
            for p in [_PYRIGHT, _RUST_ANALYZER, _GOPLS]
            if root_path is None or _defaults.preset_matches_root(p, root_path)
            if preset_probe_passes(p)
        ],
    )
    (tmp_path / "pyproject.toml").write_text("")
    config = build_test_vibe_config(lsp_auto_discover=True)
    names = {s.name for s in build_server_configs(config, root_path=tmp_path)}
    assert names == {"pyright"}
    assert probed == ["pyright"]


def test_build_server_configs_auto_discover_false_skips_presets(
    monkeypatch, tmp_path
) -> None:
    from vibe.core.lsp import _config_bridge
    from vibe.core.lsp._config_bridge import build_server_configs
    from vibe.core.lsp._defaults import _PYRIGHT

    monkeypatch.setattr(_config_bridge, "available_presets", lambda: [_PYRIGHT])
    config = build_test_vibe_config(lsp_auto_discover=False)
    assert build_server_configs(config, root_path=tmp_path) == []


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


@pytest.mark.asyncio
async def test_sync_if_changed_resends_on_edit_not_on_identity() -> None:
    # Regression: a document opened via didOpen must be re-synced with
    # didChange when its on-disk content has changed, so post-edit queries see
    # fresh text. Identical content is a no-op (no spurious version bump).
    from asyncio import StreamReader, StreamWriter
    from typing import cast as tcast

    server = LanguageServer(
        ServerConfig(name="t", command=["t"], languages={".py": "python"})
    )
    conn = JsonRpcConnection(StreamReader(), tcast(StreamWriter, _NullWriter()))
    server._conn = conn
    original = "def foo():\n    return 1\n"
    await server.did_open("/x/foo.py", original, "python")
    assert server.is_open("/x/foo.py")
    # Identical content: no didChange, only the initial didOpen.
    await server.sync_if_changed("/x/foo.py", original)
    assert [n for n in server._open_docs.values()] == [1]
    # Edited content: sync_if_changed fires didChange, bumping the version.
    await server.sync_if_changed("/x/foo.py", "def foo():\n    return 2\n")
    assert server._open_docs[next(iter(server._open_docs))] == 2


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


@pytest.mark.asyncio
async def test_setup_schedules_warmup_in_running_loop(monkeypatch, tmp_path) -> None:
    from vibe.core.lsp._lifecycle import setup_lsp_for_config, teardown_lsp_async

    warmed: list[LSPManager] = []
    monkeypatch.setattr(LSPManager, "start_warmup", lambda self: warmed.append(self))

    manager = setup_lsp_for_config(
        _config_with_lsp(), _config_with_lsp, tmp_path, warmup=True
    )

    assert manager is not None
    assert warmed == [manager]
    await teardown_lsp_async()


@pytest.mark.asyncio
async def test_manager_warmup_starts_servers_without_blocking() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _FakeServer:
        async def ensure_started(self) -> None:
            started.set()
            await release.wait()

        async def stop(self) -> None:
            pass

    manager = LSPManager()
    manager._servers = {"python": cast(Any, _FakeServer())}
    manager.start_warmup()

    await asyncio.wait_for(started.wait(), timeout=1)
    assert manager._warmup_task is not None
    assert not manager._warmup_task.done()

    release.set()
    await manager._warmup_task
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_warmup_is_idempotent() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    starts = 0

    class _FakeServer:
        async def ensure_started(self) -> None:
            nonlocal starts
            starts += 1
            started.set()
            await release.wait()

        async def stop(self) -> None:
            pass

    manager = LSPManager()
    manager._servers = {"python": cast(Any, _FakeServer())}
    manager.start_warmup()
    first_task = manager._warmup_task
    manager.start_warmup()

    assert manager._warmup_task is first_task
    await asyncio.wait_for(started.wait(), timeout=1)
    assert starts == 1

    release.set()
    assert manager._warmup_task is not None
    await manager._warmup_task
    await manager.shutdown()


@pytest.mark.asyncio
async def test_manager_shutdown_cancels_warmup() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class _FakeServer:
        async def ensure_started(self) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def stop(self) -> None:
            pass

    manager = LSPManager()
    manager._servers = {"python": cast(Any, _FakeServer())}
    manager.start_warmup()
    await asyncio.wait_for(started.wait(), timeout=1)

    await manager.shutdown()

    assert cancelled.is_set()
    assert manager._warmup_task is None


@pytest.mark.asyncio
async def test_teardown_awaits_retired_managers(monkeypatch, tmp_path) -> None:
    from vibe.core.lsp import _lifecycle as lifecycle
    from vibe.core.lsp._lifecycle import setup_lsp_for_config, teardown_lsp_async

    async def slow_shutdown(self: LSPManager) -> None:
        await asyncio.sleep(0.01)

    monkeypatch.setattr(LSPManager, "shutdown", slow_shutdown)

    setup_lsp_for_config(_config_with_lsp(), _config_with_lsp, tmp_path, warmup=False)
    setup_lsp_for_config(_config_with_lsp(), _config_with_lsp, tmp_path, warmup=False)

    assert len(lifecycle._retirement_tasks) >= 1
    await teardown_lsp_async()
    assert len(lifecycle._retirement_tasks) == 0


@pytest.mark.asyncio
async def test_warmup_one_server_failure_does_not_block_others() -> None:
    started_other = asyncio.Event()
    release = asyncio.Event()

    class _FailingServer:
        async def ensure_started(self) -> None:
            raise RuntimeError("boom")

        async def stop(self) -> None:
            pass

    class _SlowServer:
        async def ensure_started(self) -> None:
            started_other.set()
            await release.wait()

        async def stop(self) -> None:
            pass

    manager = LSPManager()
    manager._servers = {
        "rust": cast(Any, _FailingServer()),
        "python": cast(Any, _SlowServer()),
    }
    manager.start_warmup()

    await asyncio.wait_for(started_other.wait(), timeout=1)
    assert manager._warmup_task is not None
    assert not manager._warmup_task.done()

    release.set()
    await manager._warmup_task
    await manager.shutdown()


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


def test_validate_position_rejects_line_beyond_file_end(tmp_path) -> None:
    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    f = tmp_path / "small.py"
    f.write_text("x = 1\ny = 2\n")
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    with pytest.raises(ToolError, match="line 99.*2 lines"):
        tool._validate_position(f, 99, 1, "x = 1\ny = 2\n")


def test_validate_position_rejects_column_beyond_line_length(tmp_path) -> None:
    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    f = tmp_path / "small.py"
    f.write_text("x = 1\n")
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    with pytest.raises(ToolError, match="column 50.*has 5 character"):
        tool._validate_position(f, 1, 50, "x = 1\n")


def test_validate_position_passes_for_valid_coords(tmp_path) -> None:
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    f = tmp_path / "small.py"
    f.write_text("x = 1\ny = 2\n")
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    tool._validate_position(f, 2, 3, "x = 1\ny = 2\n")


def test_workspace_symbol_ranks_exact_match_above_substring() -> None:
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    raw = [
        {"name": "test_base_tool_works", "location": {"uri": "file:///x.py"}},
        {"name": "BaseTool", "location": {"uri": "file:///y.py"}},
        {"name": "base_tool_helper", "location": {"uri": "file:///z.py"}},
    ]
    result = tool._format_symbols("Workspace symbols", raw, query="BaseTool")
    names = result.symbol_names
    assert names[0] == "BaseTool"
    assert "test_base_tool_works" in names[-2:]


def test_workspace_symbol_deprioritizes_test_symbols() -> None:
    from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())
    raw = [
        {"name": "test_base_tool", "location": {"uri": "file:///t.py"}},
        {"name": "BaseTool", "location": {"uri": "file:///y.py"}},
        {"name": "BaseToolConfig", "location": {"uri": "file:///z.py"}},
    ]
    result = tool._format_symbols("Workspace symbols", raw, query="BaseTool")
    names = result.symbol_names
    assert names[-1] == "test_base_tool"
    assert names[0] == "BaseTool"


class _FakeWorkspaceServer:
    def __init__(
        self, response: list[dict] | None = None, error: Exception | None = None
    ) -> None:
        self._response = response or []
        self._error = error
        self.requests: list[tuple[str, dict]] = []

    async def send_request(self, method: str, params: dict) -> list[dict]:
        self.requests.append((method, params))
        if self._error is not None:
            raise self._error
        return self._response


class _FakeWorkspaceManager:
    def __init__(self, servers: dict[str, _FakeWorkspaceServer]) -> None:
        self._servers = servers

    @property
    def servers(self) -> dict[str, _FakeWorkspaceServer]:
        return self._servers


@pytest.mark.asyncio
async def test_workspace_symbol_runs_without_file_path(monkeypatch) -> None:
    # Regression: file_path was required for every operation, contradicting the
    # docs. workspace_symbol is workspace-wide and must work with only a query.
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspResult,
        LspState,
    )

    server = _FakeWorkspaceServer([
        {"name": "BaseTool", "location": {"uri": "file:///x.py"}}
    ])
    manager = _FakeWorkspaceManager({"pyright": server})
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    args = LspArgs(operation=LspOperation.WORKSPACE_SYMBOL, query="BaseTool")
    assert args.file_path is None

    collected = [r async for r in tool.run(args)]
    assert len(collected) == 1
    result = collected[0]
    assert isinstance(result, LspResult)
    assert "BaseTool" in result.summary
    assert server.requests == [("workspace/symbol", {"query": "BaseTool"})]


@pytest.mark.asyncio
async def test_non_workspace_op_without_file_path_raises(monkeypatch) -> None:
    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspState,
    )

    manager = _FakeWorkspaceManager({"pyright": _FakeWorkspaceServer([])})
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    args = LspArgs(operation=LspOperation.FIND_REFERENCES, line=1, character=1)
    with pytest.raises(ToolError, match="requires file_path"):
        async for _ in tool.run(args):
            pass


@pytest.mark.asyncio
async def test_workspace_symbol_without_file_path_queries_all_servers(
    monkeypatch,
) -> None:
    # Regression: omitting file_path used to query only the first configured
    # server, so symbols from other languages were missed. workspace_symbol is
    # workspace-wide, so it fans out to every server and merges results.
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspResult,
        LspState,
    )

    py = _FakeWorkspaceServer([
        {"name": "PyClass", "location": {"uri": "file:///a.py"}}
    ])
    go = _FakeWorkspaceServer([
        {"name": "GoStruct", "location": {"uri": "file:///b.go"}}
    ])
    manager = _FakeWorkspaceManager({"pyright": py, "gopls": go})
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    args = LspArgs(operation=LspOperation.WORKSPACE_SYMBOL, query="Thing")
    collected = [r async for r in tool.run(args)]
    assert len(collected) == 1
    result = collected[0]
    assert isinstance(result, LspResult)
    assert "PyClass" in result.summary
    assert "GoStruct" in result.summary
    # Both servers were queried, not just the first.
    assert py.requests and go.requests


@pytest.mark.asyncio
async def test_workspace_symbol_merges_ignoring_unsupported_server(monkeypatch) -> None:
    # A server that rejects workspace/symbol (no workspace index) must not fail
    # the whole query; symbols from the servers that do support it still surface.
    from vibe.core.lsp._types import LSPProtocolError
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspResult,
        LspState,
    )

    py = _FakeWorkspaceServer([
        {"name": "PyClass", "location": {"uri": "file:///a.py"}}
    ])
    dead = _FakeWorkspaceServer(error=LSPProtocolError("method not found", code=-32601))
    manager = _FakeWorkspaceManager({"pyright": py, "stub": dead})
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    args = LspArgs(operation=LspOperation.WORKSPACE_SYMBOL, query="PyClass")
    collected = [r async for r in tool.run(args)]
    result = collected[0]
    assert isinstance(result, LspResult)
    assert "PyClass" in result.summary


@pytest.mark.asyncio
async def test_workspace_symbol_raises_when_no_server_supports_it(monkeypatch) -> None:
    from vibe.core.lsp._types import LSPProtocolError
    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspState,
    )

    dead = _FakeWorkspaceServer(error=LSPProtocolError("method not found", code=-32601))
    manager = _FakeWorkspaceManager({"stub": dead})
    monkeypatch.setattr("vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: manager)
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))
    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    args = LspArgs(operation=LspOperation.WORKSPACE_SYMBOL, query="Thing")
    with pytest.raises(ToolError, match="No configured server supports"):
        async for _ in tool.run(args):
            pass


class _FakeCallHierarchyManager:
    """Replays canned responses keyed by (method, position) and records the
    sequence of requests so tests assert retry behavior.
    """

    def __init__(
        self,
        prepare_responses: dict[tuple[int, int], list[dict]],
        document_symbols: list[dict],
        call_edges: dict[str, list[dict]] | None = None,
        cold_rounds: int = 0,
    ) -> None:
        self._prepare = prepare_responses
        self._symbols = document_symbols
        self._edges = call_edges or {}
        self._cold_rounds = cold_rounds
        self._call_attempts = 0
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
            self._call_attempts += 1
            if self._call_attempts <= self._cold_rounds:
                return [], None
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
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    tool = _make_lsp_tool()
    # identifier `foo` starts at line 10 (0-based), char 4.
    symbols = [_fn_symbol("foo", (10, 4), (12, 0))]
    manager = _FakeCallHierarchyManager(
        prepare_responses={(10, 4): [{"name": "foo"}]},
        document_symbols=symbols,
        call_edges={"foo": [{"from": {"name": "caller", "uri": "file:///x.rs"}}]},
    )
    args = LspArgs(operation=LspOperation.INCOMING_CALLS, file_path="/x.rs")
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
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    tool = _make_lsp_tool()
    symbols = [_fn_symbol("foo", (10, 4), (12, 0))]
    manager = _FakeCallHierarchyManager(
        prepare_responses={(10, 4): [{"name": "foo"}]},
        document_symbols=symbols,
        call_edges={"foo": [{"from": {"name": "caller", "uri": "file:///x.rs"}}]},
    )
    args = LspArgs(operation=LspOperation.INCOMING_CALLS, file_path="/x.rs")
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
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

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
    args = LspArgs(operation=LspOperation.OUTGOING_CALLS, file_path="/x.rs")
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
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    tool = _make_lsp_tool()
    manager = _FakeCallHierarchyManager(prepare_responses={}, document_symbols=[])
    args = LspArgs(operation=LspOperation.INCOMING_CALLS, file_path="/x.rs")
    result = await tool._call_hierarchy(
        manager,
        args,
        "/x.rs",
        {"textDocument": {"uri": "file:///x.rs"}},
        {"line": 99, "character": 0},
    )
    assert "no callable at line 100" in result.summary
    assert "find_references" in result.summary


@pytest.mark.asyncio
async def test_call_hierarchy_retries_followup_on_cold_index(monkeypatch) -> None:
    # prepareCallHierarchy succeeds (syntax-only) but incomingCalls returns
    # empty on the first attempt because the package graph isn't loaded yet.
    # The tool retries with a short backoff and gets data on the second try.
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    delays: list[float] = []

    async def _fake_sleep(d: float) -> None:
        delays.append(d)

    monkeypatch.setattr("vibe.core.tools.builtins.lsp.asyncio.sleep", _fake_sleep)
    tool = _make_lsp_tool()
    symbols = [_fn_symbol("foo", (10, 4), (12, 0))]
    manager = _FakeCallHierarchyManager(
        prepare_responses={(10, 4): [{"name": "foo"}]},
        document_symbols=symbols,
        call_edges={"foo": [{"from": {"name": "caller", "uri": "file:///x.rs"}}]},
        cold_rounds=1,
    )
    args = LspArgs(
        operation=LspOperation.INCOMING_CALLS, file_path="/x.rs", line=11, character=5
    )
    result = await tool._call_hierarchy(
        manager,
        args,
        "/x.rs",
        {"textDocument": {"uri": "file:///x.rs"}},
        {"line": 10, "character": 4},
    )
    assert result.locations
    assert any("caller" in (loc.get("name") or "") for loc in result.locations)
    incoming_calls = [
        m for m, _ in manager.requests if m == "callHierarchy/incomingCalls"
    ]
    assert len(incoming_calls) >= 2


@pytest.mark.asyncio
async def test_call_hierarchy_no_indexing_note_for_class_symbol(monkeypatch) -> None:
    # pyright returns a CallHierarchyItem for a class, but no incoming/outgoing
    # edges (it does not model instantiation as call edges). The empty result is
    # correct and final — no backoff retry, no misleading "server was indexing"
    # caveat. SymbolKind Class == 5.
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    tool = _make_lsp_tool()
    manager = _FakeCallHierarchyManager(
        prepare_responses={(10, 4): [{"name": "Widget", "kind": 5}]},
        document_symbols=[],
        call_edges={},
    )
    args = LspArgs(
        operation=LspOperation.INCOMING_CALLS, file_path="/x.py", line=11, character=5
    )
    result = await tool._call_hierarchy(
        manager,
        args,
        "/x.py",
        {"textDocument": {"uri": "file:///x.py"}},
        {"line": 10, "character": 4},
    )
    incoming_calls = [
        m for m, _ in manager.requests if m == "callHierarchy/incomingCalls"
    ]
    assert len(incoming_calls) == 1
    assert "indexing" not in result.summary
    assert "retried" not in result.summary
    assert result.locations == []


class _FakePositionalManager:
    # Replays canned responses keyed by (method, line, character) for the
    # position-based ops, plus a static documentSymbol list. An empty result
    # at the keyword position should trigger a documentSymbol lookup and a
    # retry at the identifier — the off-identifier self-heal.

    def __init__(
        self, responses: dict[tuple[str, int, int], Any], document_symbols: list[dict]
    ) -> None:
        self._responses = responses
        self._symbols = document_symbols
        self.requests: list[tuple[str, dict]] = []

    async def send_request(self, file_path: str, method: str, params: dict):
        self.requests.append((method, params))
        if method == "textDocument/documentSymbol":
            return self._symbols, None
        pos = params.get("position") or {}
        key = (method, pos.get("line", -1), pos.get("character", -1))
        return self._responses.get(key), None


@pytest.mark.asyncio
async def test_hover_self_heals_off_identifier_position() -> None:
    # Cursor at col 1 (the `class` keyword) returns no hover; the tool resolves
    # via documentSymbol to the identifier at col 7 and retries successfully.
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    tool = _make_lsp_tool()
    symbols = [_fn_symbol("LaunchWorkflow", (99, 6), (110, 0))]
    manager = _FakePositionalManager(
        responses={
            ("textDocument/hover", 99, 6): {"contents": "(class) LaunchWorkflow"}
        },
        document_symbols=symbols,
    )
    args = LspArgs(
        operation=LspOperation.HOVER, file_path="/x.py", line=100, character=1
    )
    result = await tool._dispatch(manager, args, "/x.py", {"line": 99, "character": 0})
    assert "LaunchWorkflow" in result.summary
    hover_positions = [
        p["position"] for m, p in manager.requests if m == "textDocument/hover"
    ]
    assert hover_positions == [
        {"line": 99, "character": 0},
        {"line": 99, "character": 6},
    ]


@pytest.mark.asyncio
async def test_find_references_self_heals_off_identifier_position() -> None:
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    tool = _make_lsp_tool()
    symbols = [_fn_symbol("LaunchWorkflow", (99, 6), (110, 0))]
    manager = _FakePositionalManager(
        responses={
            ("textDocument/references", 99, 6): [
                {
                    "uri": "file:///x.py",
                    "range": {"start": {"line": 50, "character": 4}},
                }
            ]
        },
        document_symbols=symbols,
    )
    args = LspArgs(
        operation=LspOperation.FIND_REFERENCES, file_path="/x.py", line=100, character=1
    )
    result = await tool._dispatch(manager, args, "/x.py", {"line": 99, "character": 0})
    assert result.locations
    ref_positions = [
        p["position"] for m, p in manager.requests if m == "textDocument/references"
    ]
    assert ref_positions == [{"line": 99, "character": 0}, {"line": 99, "character": 6}]


@pytest.mark.asyncio
async def test_go_to_definition_self_heals_off_identifier_position() -> None:
    from vibe.core.tools.builtins.lsp import LspArgs, LspOperation

    tool = _make_lsp_tool()
    symbols = [_fn_symbol("LaunchWorkflow", (99, 6), (110, 0))]
    manager = _FakePositionalManager(
        responses={
            ("textDocument/definition", 99, 6): [
                {
                    "uri": "file:///defs.py",
                    "range": {"start": {"line": 10, "character": 0}},
                }
            ]
        },
        document_symbols=symbols,
    )
    args = LspArgs(
        operation=LspOperation.GO_TO_DEFINITION,
        file_path="/x.py",
        line=100,
        character=1,
    )
    result = await tool._dispatch(manager, args, "/x.py", {"line": 99, "character": 0})
    assert result.locations
    def_positions = [
        p["position"] for m, p in manager.requests if m == "textDocument/definition"
    ]
    assert def_positions == [{"line": 99, "character": 0}, {"line": 99, "character": 6}]


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


def test_out_of_range_position_fails_soft(monkeypatch, tmp_path) -> None:
    import asyncio

    from vibe.core.tools.base import ToolError
    from vibe.core.tools.builtins.lsp import (
        Lsp,
        LspArgs,
        LspConfig,
        LspOperation,
        LspResult,
        LspState,
    )

    tool = Lsp(config_getter=lambda: LspConfig(), state=LspState())

    class _FakeServer:
        def __init__(self) -> None:
            self.config = ServerConfig(
                name="pyright", command=["x"], languages={".py": "python"}
            )

    class _FakeManager:
        def __init__(self) -> None:
            self.server = _FakeServer()

        def get_server_for_file(self, path):
            return self.server

        async def open_document(self, path, text, language_id):
            pass

    monkeypatch.setattr(
        "vibe.core.tools.builtins.lsp.get_lsp_manager", lambda: _FakeManager()
    )
    monkeypatch.setattr(Lsp, "_lsp_installed", staticmethod(lambda: True))

    tmp = tmp_path / "test.py"
    tmp.write_text("x = 1\ny = 2\n")
    args = LspArgs(
        operation=LspOperation.GO_TO_DEFINITION,
        file_path=str(tmp),
        line=99,
        character=1,
    )

    events: list = []

    async def _inner() -> None:
        async for ev in tool.run(args):
            events.append(ev)

    try:
        asyncio.run(_inner())
    except ToolError as exc:  # pragma: no cover - asserts the bug, must not fire
        raise AssertionError(f"expected fail-soft, got hard ToolError: {exc}") from exc

    result = events[-1]
    assert isinstance(result, LspResult)
    assert "out of range" in result.summary
    assert "document_symbol" in result.summary


def test_diagnostic_registry_suppresses_stale_import_when_module_exists(
    tmp_path,
) -> None:
    """The new-module gap: pyright cached 'not found', then the module
    appeared via git. The diagnostic is provably stale — the module file is on
    disk — so it must not stage to the model.
    """
    (tmp_path / "vibe").mkdir()
    pkg = tmp_path / "vibe" / "core"
    pkg.mkdir(parents=True)
    (tmp_path / "vibe" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "verifier.py").write_text("")

    registry = DiagnosticRegistry(root_path=tmp_path)
    diag = {
        "uri": "file:///tmp/agent_loop.py",
        "diagnostics": [
            {
                "range": {
                    "start": {"line": 222, "character": 9},
                    "end": {"line": 222, "character": 60},
                },
                "severity": 1,
                "message": 'Import "vibe.core.verifier" could not be resolved',
                "source": "pyright",
            }
        ],
    }
    registry.publish(diag, "pyright")
    assert registry.consume() == []


def test_diagnostic_registry_keeps_real_import_error_when_module_absent(
    tmp_path,
) -> None:
    """A genuinely missing module must still stage — only provably-stale
    (module exists on disk) errors are suppressed.
    """
    registry = DiagnosticRegistry(root_path=tmp_path)
    diag = {
        "uri": "file:///tmp/agent_loop.py",
        "diagnostics": [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 30},
                },
                "severity": 1,
                "message": 'Import "vibe.core.nonexistent" could not be resolved',
                "source": "pyright",
            }
        ],
    }
    registry.publish(diag, "pyright")
    batches = registry.consume()
    assert len(batches) == 1


def test_diagnostic_registry_set_root_enables_suppression(tmp_path) -> None:
    """Suppression engages only after set_root wires the workspace root."""
    (tmp_path / "vibe").mkdir()
    pkg = tmp_path / "vibe" / "core"
    pkg.mkdir(parents=True)
    (tmp_path / "vibe" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "verifier.py").write_text("")

    registry = DiagnosticRegistry()
    diag = {
        "uri": "file:///tmp/agent_loop.py",
        "diagnostics": [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 10},
                },
                "severity": 1,
                "message": 'Import "vibe.core.verifier" could not be resolved',
                "source": "pyright",
            }
        ],
    }
    registry.publish(diag, "pyright")
    assert len(registry.consume()) == 1
    registry.set_root(tmp_path)
    registry.publish(diag, "pyright")
    assert registry.consume() == []


def test_diagnostic_registry_stale_filter_is_source_specific(tmp_path) -> None:
    """The import-resolution filter is keyed to the pyright source. A
    diagnostic with an identical message shape from another server passes
    through unchanged — the registry never silently applies one server's
    heuristic to another's diagnostics.
    """
    (tmp_path / "vibe").mkdir()
    pkg = tmp_path / "vibe" / "core"
    pkg.mkdir(parents=True)
    (tmp_path / "vibe" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "verifier.py").write_text("")

    registry = DiagnosticRegistry(root_path=tmp_path)
    diag = {
        "uri": "file:///tmp/agent_loop.py",
        "diagnostics": [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 10},
                },
                "severity": 1,
                "message": 'Import "vibe.core.verifier" could not be resolved',
                "source": "pyright",
            }
        ],
    }
    registry.publish(diag, "typescript-language-server")
    assert len(registry.consume()) == 1


def test_resolve_binary_prefers_project_venv(tmp_path, monkeypatch) -> None:
    """The project-venv binary wins over a stray PATH global install — closing
    the version-skew class where the LSP tool spawns a different binary than
    the project's own toolchain uses.
    """
    from vibe.core.lsp._defaults import _resolve_binary

    venv_bin = tmp_path / ".venv" / "bin" / "pyright-langserver"
    venv_bin.parent.mkdir(parents=True)
    venv_bin.write_text("#!/bin/sh\nexit 0\n")
    venv_bin.chmod(0o755)

    def fake_which(_name: str) -> str | None:
        return "/usr/local/bin/stray-pyright"

    monkeypatch.setattr("vibe.core.lsp._defaults.shutil.which", fake_which)
    resolved = _resolve_binary("pyright-langserver", tmp_path)
    assert resolved == str(venv_bin)


def test_resolve_binary_falls_back_to_path_when_no_venv(tmp_path, monkeypatch) -> None:
    from vibe.core.lsp._defaults import _resolve_binary

    monkeypatch.setattr(
        "vibe.core.lsp._defaults.shutil.which",
        lambda _name: "/usr/local/bin/pyright-langserver",
    )
    resolved = _resolve_binary("pyright-langserver", tmp_path)
    assert resolved == "/usr/local/bin/pyright-langserver"


def test_resolve_binary_returns_none_when_absent(tmp_path, monkeypatch) -> None:
    from vibe.core.lsp._defaults import _resolve_binary

    monkeypatch.setattr("vibe.core.lsp._defaults.shutil.which", lambda _name: None)
    assert _resolve_binary("pyright-langserver", tmp_path) is None


def test_probe_passes_root_to_resolve_binary(tmp_path, monkeypatch) -> None:
    """_probe honors root_path so available_presets() uses the venv binary."""
    from vibe.core.lsp import _defaults
    from vibe.core.lsp._defaults import PRESETS

    calls: list[Path | None] = []

    def fake_resolve(_name: str, root: Path | None) -> str | None:
        calls.append(root)
        return None  # forces "absent" without spawning a process

    monkeypatch.setattr(_defaults, "_resolve_binary", fake_resolve)
    _defaults._probe(PRESETS["pyright"], root_path=tmp_path)
    assert calls == [tmp_path]


def test_build_server_configs_resolves_command_to_venv_binary(
    tmp_path, monkeypatch
) -> None:
    """The spawned ServerConfig.command[0] must be the venv binary when one
    exists, not whatever stray global install is first on PATH.
    """
    from vibe.core.config import VibeConfig
    from vibe.core.lsp import _config_bridge
    from vibe.core.lsp._defaults import _PYRIGHT

    venv_bin = tmp_path / ".venv" / "bin" / "pyright-langserver"
    venv_bin.parent.mkdir(parents=True)
    venv_bin.write_text("#!/bin/sh\nexit 0\n")
    venv_bin.chmod(0o755)

    def fake_available(_root=None):
        return [_PYRIGHT]

    monkeypatch.setattr(_config_bridge, "available_presets", fake_available)

    config = VibeConfig(lsp_servers=[], lsp_auto_discover=True)
    configs = _config_bridge.build_server_configs(config, tmp_path)
    pyright = next(c for c in configs if c.name == "pyright")
    assert pyright.command[0] == str(venv_bin)


def test_install_for_preset_no_command_returns_hint_only_error() -> None:
    """A preset without install_command falls back to the hint path."""
    from vibe.core.lsp import install_for_preset
    from vibe.core.lsp._defaults import _CLANGD

    result = install_for_preset(_CLANGD)
    assert not result.success
    assert "no install_command" in result.error


def test_install_for_preset_unsupported_channel_returns_hint_only_error() -> None:
    """A preset whose install_command uses a tool vibe does not bootstrap
    (e.g. a hand-rolled curl) stays hint-only — the user installs manually.
    """
    from vibe.core.config import LSPServer
    from vibe.core.lsp import install_for_preset
    from vibe.core.lsp._defaults import ServerPreset

    preset = ServerPreset(
        key="exotic",
        display_name="Exotic",
        server=LSPServer(
            name="exotic", command="exotic-ls", languages={".ex": "exotic"}
        ),
        install_hint="curl https://example.com/exotic-ls | sh",
        detection_command=("exotic-ls", "--version"),
        install_command=("curl", "https://example.com/install.sh"),
    )
    result = install_for_preset(preset)
    assert not result.success
    assert "does not bootstrap" in result.error


def test_install_for_preset_channel_absent_returns_explanatory_error(
    monkeypatch,
) -> None:
    """The install_command's tool exists in the channel map but the binary
    itself is not on PATH — the error names the missing tool.
    """
    from vibe.core.lsp import _installer, install_for_preset
    from vibe.core.lsp._defaults import _PYRIGHT

    monkeypatch.setattr(_installer.shutil, "which", lambda _name: None)
    result = install_for_preset(_PYRIGHT)
    assert not result.success
    assert "not on PATH" in result.error


def test_install_for_preset_declined_without_consent_callback() -> None:
    """A None consent_callback declines by default — install never fires."""
    from vibe.core.lsp import _installer, install_for_preset
    from vibe.core.lsp._defaults import _PYRIGHT

    _installer.channel_available("pip")  # touch to ensure import is live
    result = install_for_preset(_PYRIGHT, consent_callback=None)
    assert not result.success
    assert result.error == "declined by user"


def test_install_for_preset_runs_after_consent(monkeypatch) -> None:
    """A returning-True consent_callback triggers the subprocess; the post-run
    binary resolution check decides success.
    """
    from types import SimpleNamespace

    from vibe.core.lsp import _installer, install_for_preset
    from vibe.core.lsp._defaults import _PYRIGHT

    ran: list[tuple[str, ...]] = []

    def fake_run(cmd, **kwargs):
        ran.append(tuple(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    # The installer imports _resolve_binary inline from _defaults, so patch the
    # source module to make the post-install existence check succeed.
    from vibe.core.lsp import _defaults

    monkeypatch.setattr(_installer.subprocess, "run", fake_run)
    monkeypatch.setattr(
        _defaults,
        "_resolve_binary",
        lambda _binary, _root: "/fake/venv/bin/pyright-langserver",
    )
    result = install_for_preset(_PYRIGHT, consent_callback=lambda _desc: True)
    assert ran and ran[0] == _PYRIGHT.install_command
    assert result.success


def test_install_for_preset_nonzero_exit_is_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    from vibe.core.lsp import _installer, install_for_preset
    from vibe.core.lsp._defaults import _PYRIGHT

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="permission denied")

    monkeypatch.setattr(_installer.subprocess, "run", fake_run)
    result = install_for_preset(_PYRIGHT, consent_callback=lambda _desc: True)
    assert not result.success
    assert "exit 1" in result.error
    assert "permission denied" in result.output


def test_preset_install_command_field_round_trips() -> None:
    """7 bootstrap-eligible presets carry a channel-mapped install_command;
    hint-only presets (clangd, jdtls, sourcekit) leave it empty.
    """
    from vibe.core.lsp._defaults import PRESETS

    bootstrap_keys = {"pyright", "typescript", "rust", "go", "csharp", "php", "ruby"}
    hint_only_keys = {"clangd", "java", "swift"}
    for key in bootstrap_keys:
        assert PRESETS[key].install_command, f"{key} should bootstrap"
        assert PRESETS[key].install_command[0] in {
            "pip",
            "npm",
            "rustup",
            "go",
            "dotnet",
            "gem",
        }, f"{key} channel not in bootstrap set"
    for key in hint_only_keys:
        assert not PRESETS[key].install_command, f"{key} should stay hint-only"
