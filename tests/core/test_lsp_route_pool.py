from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import ValidationError
import pytest

from vibe.core.config import VibeConfig
from vibe.core.lsp._manager import LSPManager
from vibe.core.lsp._route_pool import WorkspaceRoutePool
from vibe.core.lsp._server import LanguageServer, ServerConfig
from vibe.core.lsp._types import uri_from_path
from vibe.core.tools.builtins.lsp import Lsp, LspResult
from vibe.core.utils.io import write_safe


class _ConfigSource:
    def __init__(self, configs: list[ServerConfig]) -> None:
        self.configs = configs

    def load(self) -> list[ServerConfig]:
        return self.configs


def _config(name: str = "pyright", extension: str = ".py") -> ServerConfig:
    return ServerConfig(
        name=name,
        command=[name],
        languages={extension: name},
        manifest_markers=("workspace.marker",),
    )


def _workspace_file(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_safe(root / "workspace.marker", "")
    source = root / name
    write_safe(source, "")
    return source


async def _leased_server(manager: LSPManager, path: Path) -> LanguageServer:
    async with manager.lease_server_for_file(path) as server:
        assert server is not None
        return server


@pytest.mark.asyncio
async def test_route_pool_evicts_least_recent_idle_root() -> None:
    pool = WorkspaceRoutePool(max_dynamic_roots=2)

    await pool.acquire("a")
    await pool.release("a")
    await pool.acquire("b")
    await pool.release("b")
    await pool.acquire("a")
    await pool.release("a")
    admission = await pool.acquire("c")

    assert admission.evicted_roots == ("b",)
    assert pool.is_resident("a")
    assert pool.is_resident("c")
    assert not pool.is_resident("b")


@pytest.mark.asyncio
async def test_route_pool_waiter_cancellation_does_not_leak_capacity() -> None:
    pool = WorkspaceRoutePool(max_dynamic_roots=1)
    await pool.acquire("a")
    waiter = asyncio.create_task(pool.acquire("b"))
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await pool.release("a")
    admission = await pool.acquire("b")

    assert admission.evicted_roots == ("a",)
    assert pool.snapshot().leased_dynamic_roots == 1


@pytest.mark.asyncio
async def test_protected_roots_do_not_consume_dynamic_capacity() -> None:
    pool = WorkspaceRoutePool(max_dynamic_roots=1)
    await pool.acquire("explicit", protected=True)
    await pool.release("explicit")
    await pool.acquire("a")
    await pool.release("a")

    admission = await pool.acquire("b")

    assert admission.evicted_roots == ("a",)
    assert pool.is_resident("explicit")
    assert pool.snapshot().resident_dynamic_roots == 1


@pytest.mark.asyncio
async def test_manager_evicts_whole_root_bucket_in_lru_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    manager = LSPManager(_ConfigSource([config]), max_workspace_roots=2)
    manager.set_root(tmp_path)
    manager.initialize()
    generation = manager.generation
    a = _workspace_file(tmp_path / "a", "main.py")
    b = _workspace_file(tmp_path / "b", "main.py")
    c = _workspace_file(tmp_path / "c", "main.py")

    a_server = await _leased_server(manager, a)
    b_server = await _leased_server(manager, b)
    assert await _leased_server(manager, a) is a_server
    retired: list[LanguageServer] = []

    async def stop_b() -> None:
        retired.append(b_server)

    monkeypatch.setattr(b_server, "stop", stop_b)
    c_server = await _leased_server(manager, c)

    assert retired == [b_server]
    assert manager.generation == generation
    assert manager._servers_by_route[(config.name, uri_from_path(a.parent))] is a_server
    assert manager._servers_by_route[(config.name, uri_from_path(c.parent))] is c_server
    assert (config.name, uri_from_path(b.parent)) not in manager._servers_by_route
    route_pool = manager.readiness().route_pool
    assert route_pool is not None
    assert route_pool.resident_roots == 3
    assert route_pool.known_roots == 4
    assert route_pool.workspace_symbol_partial


@pytest.mark.asyncio
async def test_manager_waits_for_active_root_and_clean_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    manager = LSPManager(_ConfigSource([config]), max_workspace_roots=1)
    manager.set_root(tmp_path)
    manager.initialize()
    a = _workspace_file(tmp_path / "a", "main.py")
    b = _workspace_file(tmp_path / "b", "main.py")
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()
    b_entered = asyncio.Event()

    async with manager.lease_server_for_file(a) as a_server:
        assert a_server is not None

        async def blocked_stop() -> None:
            stop_started.set()
            await allow_stop.wait()

        monkeypatch.setattr(a_server, "stop", blocked_stop)

        async def use_b() -> None:
            async with manager.lease_server_for_file(b) as b_server:
                assert b_server is not None
                b_entered.set()

        waiter = asyncio.create_task(use_b())
        await asyncio.sleep(0)
        assert not stop_started.is_set()
        assert not b_entered.is_set()

    await asyncio.wait_for(stop_started.wait(), timeout=1)
    assert not b_entered.is_set()
    allow_stop.set()
    await asyncio.wait_for(waiter, timeout=1)
    assert b_entered.is_set()


@pytest.mark.asyncio
async def test_late_diagnostics_from_evicted_server_are_ignored(tmp_path: Path) -> None:
    config = _config()
    manager = LSPManager(_ConfigSource([config]), max_workspace_roots=1)
    manager.set_root(tmp_path)
    manager.initialize()
    a = _workspace_file(tmp_path / "a", "main.py")
    b = _workspace_file(tmp_path / "b", "main.py")
    a_server = await _leased_server(manager, a)
    handler = next(
        callback
        for method, callback in a_server._pending_handlers
        if method == "textDocument/publishDiagnostics"
    )

    await _leased_server(manager, b)
    await handler({
        "uri": uri_from_path(a),
        "diagnostics": [
            {
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 1},
                },
                "severity": 1,
                "message": "stale",
            }
        ],
    })

    assert manager.consume_diagnostics_text() is None


@pytest.mark.asyncio
async def test_reload_retires_prior_servers_before_replacement_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _ConfigSource([_config()])
    manager = LSPManager(source)
    (tmp_path / ".git").mkdir()
    manager.set_root(tmp_path)
    manager.initialize()
    old_server = manager.servers["pyright"]
    stopped = asyncio.Event()

    async def stop_old() -> None:
        stopped.set()

    monkeypatch.setattr(old_server, "stop", stop_old)
    manager.reload()
    replacement = manager.servers["pyright"]

    await asyncio.wait_for(stopped.wait(), timeout=1)
    assert replacement is not old_server
    assert old_server not in manager._servers_by_route.values()


@pytest.mark.asyncio
async def test_reload_defers_retirement_until_active_lease_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _ConfigSource([_config()])
    manager = LSPManager(source)
    (tmp_path / ".git").mkdir()
    manager.set_root(tmp_path)
    manager.initialize()
    a = _workspace_file(tmp_path / "a", "main.py")
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()
    replacement_entered = asyncio.Event()

    async with manager.lease_server_for_file(a) as old_server:
        assert old_server is not None

        async def stop_old() -> None:
            stop_started.set()
            await allow_stop.wait()

        monkeypatch.setattr(old_server, "stop", stop_old)
        manager.reload()
        assert manager.get_server_for_file(a) is None

        async def use_replacement() -> None:
            async with manager.lease_server_for_file(a) as replacement:
                assert replacement is not None
                assert replacement is not old_server
                replacement_entered.set()

        waiter = asyncio.create_task(use_replacement())
        await asyncio.sleep(0)
        assert not stop_started.is_set()
        assert not replacement_entered.is_set()

    await asyncio.wait_for(stop_started.wait(), timeout=1)
    assert not replacement_entered.is_set()
    allow_stop.set()
    await asyncio.wait_for(waiter, timeout=1)
    assert replacement_entered.is_set()


@pytest.mark.asyncio
async def test_reload_blocks_workspace_broadcast_until_old_server_retires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _ConfigSource([_config()])
    manager = LSPManager(source)
    (tmp_path / ".git").mkdir()
    manager.set_root(tmp_path)
    session_file = _workspace_file(tmp_path, "main.py")
    manager.initialize()
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()
    replacement_used = asyncio.Event()

    async with manager.lease_server_for_file(session_file) as old_server:
        assert old_server is not None

        async def stop_old() -> None:
            stop_started.set()
            await allow_stop.wait()

        monkeypatch.setattr(old_server, "stop", stop_old)
        manager.reload()
        replacement = manager.servers["pyright"]

        async def request_replacement(_method: str, _params: object) -> list[object]:
            replacement_used.set()
            return []

        monkeypatch.setattr(replacement, "send_request", request_replacement)
        broadcast = asyncio.create_task(
            manager.send_request_all("workspace/symbol", {"query": "Thing"})
        )
        await asyncio.sleep(0)
        assert not replacement_used.is_set()

    await asyncio.wait_for(stop_started.wait(), timeout=1)
    assert not replacement_used.is_set()
    allow_stop.set()
    await asyncio.wait_for(broadcast, timeout=1)
    assert replacement_used.is_set()


@pytest.mark.asyncio
async def test_reload_blocks_warmup_until_old_server_retires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _ConfigSource([_config()])
    manager = LSPManager(source)
    (tmp_path / ".git").mkdir()
    manager.set_root(tmp_path)
    session_file = _workspace_file(tmp_path, "main.py")
    manager.initialize()
    stop_started = asyncio.Event()
    allow_stop = asyncio.Event()
    replacement_started = asyncio.Event()

    async with manager.lease_server_for_file(session_file) as old_server:
        assert old_server is not None

        async def stop_old() -> None:
            stop_started.set()
            await allow_stop.wait()

        monkeypatch.setattr(old_server, "stop", stop_old)
        manager.reload()
        replacement = manager.servers["pyright"]

        async def start_replacement() -> None:
            replacement_started.set()

        monkeypatch.setattr(replacement, "ensure_started", start_replacement)
        manager.start_warmup()
        await asyncio.sleep(0)
        assert not replacement_started.is_set()

    await asyncio.wait_for(stop_started.wait(), timeout=1)
    assert not replacement_started.is_set()
    allow_stop.set()
    assert manager._warmup_task is not None
    await asyncio.wait_for(manager._warmup_task, timeout=1)
    assert replacement_started.is_set()


@pytest.mark.asyncio
async def test_waiting_route_rechecks_config_after_reload(tmp_path: Path) -> None:
    source = _ConfigSource([_config()])
    manager = LSPManager(source, max_workspace_roots=1)
    manager.set_root(tmp_path)
    manager.initialize()
    a = _workspace_file(tmp_path / "a", "main.py")
    b = _workspace_file(tmp_path / "b", "main.py")
    selected: list[LanguageServer | None] = []

    async with manager.lease_server_for_file(a):

        async def use_b() -> None:
            async with manager.lease_server_for_file(b) as server:
                selected.append(server)

        waiter = asyncio.create_task(use_b())
        await asyncio.sleep(0)
        source.configs = []
        manager.reload()

    await asyncio.wait_for(waiter, timeout=1)
    assert selected == [None]


@pytest.mark.asyncio
async def test_multiple_server_configs_share_one_root_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    py = _config("pyright", ".py")
    ts = _config("typescript", ".ts")
    manager = LSPManager(_ConfigSource([py, ts]), max_workspace_roots=1)
    manager.set_root(tmp_path)
    manager.initialize()
    a_py = _workspace_file(tmp_path / "a", "main.py")
    a_ts = a_py.with_suffix(".ts")
    write_safe(a_ts, "")
    b_py = _workspace_file(tmp_path / "b", "main.py")
    py_server = await _leased_server(manager, a_py)
    ts_server = await _leased_server(manager, a_ts)
    retired: set[LanguageServer] = set()

    async def stop_py() -> None:
        retired.add(py_server)

    async def stop_ts() -> None:
        retired.add(ts_server)

    monkeypatch.setattr(py_server, "stop", stop_py)
    monkeypatch.setattr(ts_server, "stop", stop_ts)
    await _leased_server(manager, b_py)

    assert retired == {py_server, ts_server}
    route_pool = manager.readiness().route_pool
    assert route_pool is not None
    assert route_pool.resident_dynamic_roots == 1


@pytest.mark.asyncio
async def test_readiness_is_observational_for_nonresident_root(tmp_path: Path) -> None:
    config = _config()
    manager = LSPManager(_ConfigSource([config]), max_workspace_roots=1)
    manager.set_root(tmp_path)
    manager.initialize()
    a = _workspace_file(tmp_path / "a", "main.py")
    b = _workspace_file(tmp_path / "b", "main.py")
    a_server = await _leased_server(manager, a)

    snapshot = manager.readiness(b)

    assert snapshot.selected_workspace_root == uri_from_path(b.parent)
    assert snapshot.route_pool is not None
    assert snapshot.route_pool.resident_dynamic_roots == 1
    assert manager._servers_by_route[(config.name, uri_from_path(a.parent))] is a_server
    assert (config.name, uri_from_path(b.parent)) not in manager._servers_by_route


@pytest.mark.asyncio
async def test_workspace_broadcast_pins_routes_until_requests_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    manager = LSPManager(_ConfigSource([config]), max_workspace_roots=1)
    manager.set_root(tmp_path)
    manager.initialize()
    a = _workspace_file(tmp_path / "a", "main.py")
    b = _workspace_file(tmp_path / "b", "main.py")
    a_server = await _leased_server(manager, a)
    request_started = asyncio.Event()
    allow_request = asyncio.Event()
    b_entered = asyncio.Event()

    async def blocked_request(_method: str, _params: object) -> list[object]:
        request_started.set()
        await allow_request.wait()
        return []

    monkeypatch.setattr(a_server, "send_request", blocked_request)
    for server in manager.servers.values():
        if server is a_server:
            continue

        async def empty_request(_method: str, _params: object) -> list[object]:
            return []

        monkeypatch.setattr(server, "send_request", empty_request)
    broadcast = asyncio.create_task(
        manager.send_request_all("workspace/symbol", {"query": "Thing"})
    )
    await asyncio.wait_for(request_started.wait(), timeout=1)

    async def use_b() -> None:
        async with manager.lease_server_for_file(b) as b_server:
            assert b_server is not None
            b_entered.set()

    waiter = asyncio.create_task(use_b())
    await asyncio.sleep(0)
    assert not b_entered.is_set()
    allow_request.set()
    await asyncio.wait_for(broadcast, timeout=1)
    await asyncio.wait_for(waiter, timeout=1)

    assert b_entered.is_set()


def test_workspace_root_limit_is_validated_and_reported(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        VibeConfig(lsp_max_workspace_roots=0)

    manager = LSPManager(max_workspace_roots=3)
    manager.set_root(tmp_path)
    snapshot = manager.readiness()

    assert snapshot.route_pool is not None
    assert snapshot.route_pool.max_dynamic_roots == 3


def test_workspace_symbol_marks_resident_only_results_partial() -> None:
    result = Lsp._apply_workspace_coverage(
        LspResult(operation="symbols", summary="Workspace symbols: 1 found."),
        {"resident_roots": 2, "known_roots": 3, "partial": True},
    )

    assert result.partial_coverage
    assert result.workspace_coverage == {
        "resident_roots": 2,
        "known_roots": 3,
        "partial": True,
    }
    assert "Results are partial" in result.summary
    assert "lsp_max_workspace_roots" in result.summary
