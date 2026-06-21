from __future__ import annotations

import asyncio

import pytest

from vibe.core.lsp._jsonrpc import JsonRpcConnection
from vibe.core.lsp._manager import LSPManager
from vibe.core.lsp._registry import DiagnosticRegistry
from vibe.core.lsp._server import LanguageServer, ServerConfig
from vibe.core.lsp._types import (
    Diagnostic,
    DiagnosticSeverity,
    LSPProtocolError,
    Position,
    Range,
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
