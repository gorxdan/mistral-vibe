from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time
from typing import Any, Protocol

from vibe.core.logger import logger
from vibe.core.lsp._registry import DiagnosticRegistry, format_diagnostics_for_model
from vibe.core.lsp._server import LanguageServer, ServerConfig
from vibe.core.lsp._types import (
    LSPError,
    LSPNotConnectedError,
    ServerState,
    path_from_uri,
    uri_from_path,
)


class LSPServerSource(Protocol):
    def load(self) -> list[ServerConfig]: ...


class LSPManager:
    """Owns the registry of language servers, file-to-server routing, and the
    diagnostic registry that feeds passive next-turn context.
    """

    def __init__(self, source: LSPServerSource | None = None) -> None:
        self._source = source
        self._servers: dict[str, LanguageServer] = {}
        self._configs: list[ServerConfig] = []
        self._diagnostics = DiagnosticRegistry()
        self._initialized = False
        self._root_uri: str | None = None
        self._root_path: Path | None = None

    def set_source(self, source: LSPServerSource) -> None:
        self._source = source

    def set_root(self, root_path: str | Path) -> None:
        self._root_uri = uri_from_path(root_path)
        self._root_path = Path(root_path).resolve()

    @property
    def diagnostics(self) -> DiagnosticRegistry:
        return self._diagnostics

    @property
    def servers(self) -> dict[str, LanguageServer]:
        return dict(self._servers)

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.reload()

    def reload(self) -> None:
        if self._source is None:
            return
        try:
            configs = self._source.load()
        except Exception:
            logger.exception("lsp config load failed; servers unavailable")
            configs = []
        self._configs = configs
        self._servers = {cfg.name: self._build_server(cfg) for cfg in configs}
        for name, server in self._servers.items():
            server.on_notification(
                "textDocument/publishDiagnostics",
                lambda params, _n=name: self._on_publish(params, _n),
            )
        if configs:
            logger.info(
                "lsp configured %d server(s): %s",
                len(configs),
                ", ".join(c.name for c in configs),
            )

    def _build_server(self, config: ServerConfig) -> LanguageServer:
        if self._root_uri and config.root_uri is None:
            config.root_uri = self._root_uri
        return LanguageServer(config)

    def _resolve_manifest_root(
        self, file_path: Path, markers: tuple[str, ...], default_root: Path
    ) -> Path:
        """Find the nearest ancestor dir containing any of ``markers``.

        Falls back to ``default_root`` when no marker is found or ``markers``
        is empty. Search is bounded by the session root so it never escapes
        the project — a marker above the session root (rare) is ignored.
        """
        if not markers:
            return default_root
        root_resolved = default_root.resolve()
        search_from = file_path.parent if file_path.is_file() else file_path
        for candidate in [search_from, *search_from.parents]:
            if any((candidate / marker).exists() for marker in markers):
                return candidate
            if candidate.resolve() == root_resolved:
                break
        return default_root

    async def _on_publish(self, params: dict[str, Any], server_name: str) -> None:
        self._diagnostics.publish(params, server_name)

    def get_server_for_file(self, path: str | Path) -> LanguageServer | None:
        ext = Path(path).suffix
        if not ext:
            return None
        for config in self._configs:
            if config.matches(ext):
                return self._servers.get(config.name)
        return None

    def status(self) -> dict[str, Any]:
        return {
            "servers": [
                {"name": name, "state": str(s.state), "error": s.last_error}
                for name, s in self._servers.items()
            ]
        }

    async def send_request(
        self, path: str | Path, method: str, params: dict[str, Any] | None = None
    ) -> tuple[Any, LanguageServer]:
        server = self.get_server_for_file(path)
        if server is None:
            raise LSPNotConnectedError(
                f"no LSP server registered for {Path(path).suffix or 'unknown'}"
            )
        total_start = time.perf_counter()
        try:
            result = await server.send_request(method, params)
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                total_ms = (time.perf_counter() - total_start) * 1000.0
                logger.debug("lsp request %s total=%.1fms", method, total_ms)
        return result, server

    async def open_document(
        self, path: str | Path, text: str, language_id: str
    ) -> None:
        server = self.get_server_for_file(path)
        if server is None:
            return
        markers = server.config.manifest_markers
        if markers and self._root_path is not None and server.config.root_uri is None:
            resolved = self._resolve_manifest_root(Path(path), markers, self._root_path)
            server.config.root_uri = uri_from_path(resolved)
            if server.config.cwd is None:
                server.config.cwd = str(resolved)
        await server.ensure_started()
        if not server.is_open(str(path)):
            await server.did_open(str(path), text, language_id)
        else:
            await server.sync_if_changed(str(path), text)

    async def notify_change(self, path: str | Path, text: str) -> None:
        server = self.get_server_for_file(path)
        if server is None or not server.is_open(str(path)):
            return
        await server.did_change(str(path), text)
        self._diagnostics.clear_for_path(str(path))

    async def notify_save(self, path: str | Path, text: str) -> None:
        server = self.get_server_for_file(path)
        if server is None or not server.is_open(str(path)):
            return
        await server.did_save(str(path), text)

    async def reinitialize(self) -> None:
        await self.shutdown()
        self._initialized = False
        self.initialize()

    async def shutdown(self) -> None:
        stops = [server.stop() for server in self._servers.values()]
        if stops:
            await asyncio.gather(*stops, return_exceptions=True)
        self._servers.clear()
        self._configs.clear()
        self._initialized = False

    def consume_diagnostics_text(self) -> str | None:
        batches = self._diagnostics.consume()
        if not batches:
            return None
        return "\n\n".join(format_diagnostics_for_model(b) for b in batches)

    def clear_diagnostics_for(self, path: str | Path) -> None:
        self._diagnostics.clear_for_path(str(path))


def uris_equal(a: str, b: str) -> bool:
    return path_from_uri(a) == path_from_uri(b)


_global_manager: LSPManager | None = None
_global_generation: int = 0


def init_lsp_manager(manager: LSPManager) -> int:
    """Install ``manager`` as the process-wide LSP singleton.

    Returns the new generation counter. Callers that captured a generation
    before doing async work can compare against the return to detect that a
    newer setup superseded them.
    """
    global _global_manager, _global_generation
    _global_generation += 1
    _global_manager = manager
    return _global_generation


def get_lsp_manager() -> LSPManager | None:
    """Return the process LSP manager, or ``None`` if LSP is not active."""
    return _global_manager


def current_lsp_generation() -> int:
    """Monotonic counter bumped on every ``init_lsp_manager`` install."""
    return _global_generation


def clear_lsp_manager() -> None:
    """Drop the singleton reference (does not shut the manager down)."""
    global _global_manager
    _global_manager = None


__all__ = [
    "LSPError",
    "LSPManager",
    "LSPNotConnectedError",
    "LSPServerSource",
    "ServerState",
    "clear_lsp_manager",
    "current_lsp_generation",
    "get_lsp_manager",
    "init_lsp_manager",
    "uris_equal",
]
