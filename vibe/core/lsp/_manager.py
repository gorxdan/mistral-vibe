from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
import logging
from pathlib import Path
import time
from typing import Any, Protocol

from vibe.core.logger import logger
from vibe.core.lsp._readiness import (
    LSPReadinessSnapshot,
    LSPRoutePoolReadiness,
    build_lsp_readiness,
)
from vibe.core.lsp._registry import DiagnosticRegistry, format_diagnostics_for_model
from vibe.core.lsp._roots import (
    ResolvedWorkspaceRoot,
    directory_matches_markers,
    nearest_manifest_root,
    resolve_workspace_root,
)
from vibe.core.lsp._route_pool import DEFAULT_MAX_WORKSPACE_ROOTS, WorkspaceRoutePool
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

    def __init__(
        self,
        source: LSPServerSource | None = None,
        *,
        max_workspace_roots: int = DEFAULT_MAX_WORKSPACE_ROOTS,
    ) -> None:
        self._source = source
        self._servers: dict[str, LanguageServer] = {}
        self._servers_by_route: dict[tuple[str, str], LanguageServer] = {}
        self._route_pool = WorkspaceRoutePool(max_workspace_roots)
        self._retirement_tasks: set[asyncio.Task[None]] = set()
        self._retirement_tasks_by_root: dict[str, set[asyncio.Task[None]]] = {}
        self._retiring_servers: set[LanguageServer] = set()
        self._deferred_retirements_by_root: dict[str, set[LanguageServer]] = {}
        self._route_drain_events: dict[str, asyncio.Event] = {}
        self._configs: list[ServerConfig] = []
        self._diagnostics = DiagnosticRegistry()
        self._initialized = False
        self._root_uri: str | None = None
        self._root_path: Path | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        self._routing_generation = 0

    def set_source(self, source: LSPServerSource) -> None:
        self._source = source

    def set_root(self, root_path: str | Path) -> None:
        requested_root = Path(root_path).resolve()
        self._root_path = next(
            (
                candidate
                for candidate in (requested_root, *requested_root.parents)
                if (candidate / ".git").exists()
            ),
            requested_root,
        )
        self._root_uri = uri_from_path(self._root_path)
        # Keep the registry's workspace root in sync so it can suppress
        # provably-stale import-resolution diagnostics against the live tree.
        self._diagnostics.set_root(self._root_path)

    @property
    def diagnostics(self) -> DiagnosticRegistry:
        return self._diagnostics

    @property
    def root_path(self) -> Path | None:
        return self._root_path

    @property
    def generation(self) -> int:
        return (current_lsp_generation() << 32) | self._routing_generation

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
        self._routing_generation += 1
        try:
            configs = self._source.load()
        except Exception:
            logger.exception("lsp config load failed; servers unavailable")
            configs = []
        prior = tuple(self._servers_by_route.items())
        idle = tuple(
            (route, server)
            for route, server in prior
            if not self._route_pool.is_leased(route[1])
        )
        active = tuple(
            (route, server)
            for route, server in prior
            if self._route_pool.is_leased(route[1])
        )
        if idle:
            self._schedule_retirements(
                tuple(server for _route, server in idle),
                tuple(route for route, _server in idle),
            )
        if active:
            for route, _server in active:
                self._route_drain_events.setdefault(route[1], asyncio.Event())
            self._defer_retirements(active)
        self._configs = configs
        self._servers.clear()
        self._servers_by_route.clear()
        self._route_pool.reset(preserve_leases=True)
        for config in configs:
            self._server_for_config(config, self._default_route_probe())
        if configs:
            logger.info(
                "lsp configured %d server(s): %s",
                len(configs),
                ", ".join(c.name for c in configs),
            )

    def _default_route_probe(self) -> Path:
        root = self._root_path or Path.cwd().resolve()
        return root / ".vibe-lsp-workspace-root"

    def _server_for_config(
        self, config: ServerConfig, file_path: str | Path
    ) -> LanguageServer | None:
        root = self._resolved_root(config, file_path)
        if not self._route_pool.touch_or_admit(
            root.uri, protected=self._root_is_protected(root)
        ):
            return None
        return self._server_for_root(config, root)

    def _server_for_root(
        self, config: ServerConfig, root: ResolvedWorkspaceRoot
    ) -> LanguageServer:
        route = (config.name, root.uri)
        existing = self._servers_by_route.get(route)
        if existing is not None:
            return existing

        server = LanguageServer(self._routed_config(config, root))
        self._servers_by_route[route] = server
        key = config.name
        if key in self._servers:
            index = 2
            while f"{config.name}@{index}" in self._servers:
                index += 1
            key = f"{config.name}@{index}"
        self._servers[key] = server

        async def publish_diagnostics(params: dict[str, Any]) -> None:
            if self._servers_by_route.get(route) is not server:
                return
            await self._on_publish(params, config.name)

        server.on_notification("textDocument/publishDiagnostics", publish_diagnostics)
        return server

    def _root_is_protected(self, root: ResolvedWorkspaceRoot) -> bool:
        return root.explicit or root.uri == self._root_uri

    def _detach_roots(
        self, root_uris: tuple[str, ...]
    ) -> tuple[tuple[tuple[str, str], ...], tuple[LanguageServer, ...]]:
        if not root_uris:
            return (), ()
        roots = set(root_uris)
        routes = tuple(route for route in self._servers_by_route if route[1] in roots)
        servers = tuple(self._servers_by_route.pop(route) for route in routes)
        detached = set(servers)
        for name, server in tuple(self._servers.items()):
            if server in detached:
                del self._servers[name]
        return routes, servers

    def _defer_retirements(
        self,
        routes_and_servers: tuple[tuple[tuple[str, str], LanguageServer], ...],
        *,
        barrier_roots: tuple[str, ...] = (),
    ) -> None:
        for route, server in routes_and_servers:
            roots = {route[1], *barrier_roots}
            for root_uri in roots:
                self._deferred_retirements_by_root.setdefault(root_uri, set()).add(
                    server
                )

    def _activate_deferred_retirements(self, root_uri: str) -> None:
        if self._route_pool.is_leased(root_uri):
            return
        servers = self._deferred_retirements_by_root.pop(root_uri, set())
        if not servers:
            return
        for candidate, pending in tuple(self._deferred_retirements_by_root.items()):
            pending.difference_update(servers)
            if not pending:
                del self._deferred_retirements_by_root[candidate]
        routes = tuple((server.config.name, root_uri) for server in servers)
        self._schedule_retirements(tuple(servers), routes, barrier_roots=(root_uri,))

    async def _await_root_drain(self, root_uri: str) -> None:
        event = self._route_drain_events.get(root_uri)
        if event is not None:
            await event.wait()

    def _finish_root_drain_if_ready(self, root_uri: str) -> None:
        event = self._route_drain_events.get(root_uri)
        if event is None:
            return
        if (
            self._route_pool.is_leased(root_uri)
            or self._deferred_retirements_by_root.get(root_uri)
            or self._retirement_tasks_by_root.get(root_uri)
        ):
            return
        del self._route_drain_events[root_uri]
        event.set()

    def _schedule_retirements(
        self,
        servers: tuple[LanguageServer, ...],
        routes: tuple[tuple[str, str], ...],
        *,
        barrier_roots: tuple[str, ...] = (),
    ) -> asyncio.Task[None] | None:
        retiring = tuple(
            dict.fromkeys(
                server for server in servers if server not in self._retiring_servers
            )
        )
        if not retiring:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._defer_retirements(
                tuple(zip(routes, servers, strict=True)), barrier_roots=barrier_roots
            )
            return None
        self._retiring_servers.update(retiring)
        task = loop.create_task(
            self._stop_servers(retiring), name="lsp-route-retirement"
        )
        self._retirement_tasks.add(task)
        associated_roots = {*(route[1] for route in routes), *barrier_roots}
        for root_uri in associated_roots:
            self._retirement_tasks_by_root.setdefault(root_uri, set()).add(task)

        def retirement_finished(done: asyncio.Task[None]) -> None:
            self._retirement_tasks.discard(done)
            for root_uri in associated_roots:
                tasks = self._retirement_tasks_by_root.get(root_uri)
                if tasks is None:
                    continue
                tasks.discard(done)
                if not tasks:
                    del self._retirement_tasks_by_root[root_uri]
                self._finish_root_drain_if_ready(root_uri)

        task.add_done_callback(retirement_finished)
        return task

    async def _stop_servers(self, servers: tuple[LanguageServer, ...]) -> None:
        try:
            await asyncio.gather(
                *(server.stop() for server in servers), return_exceptions=True
            )
        finally:
            self._retiring_servers.difference_update(servers)

    async def _await_route_retirements(self, root_uri: str) -> None:
        tasks = tuple(self._retirement_tasks_by_root.get(root_uri, ()))
        for task in tasks:
            await asyncio.shield(task)

    def _retire_evicted_roots(
        self, root_uris: tuple[str, ...], *, replacement_root: str
    ) -> None:
        routes, servers = self._detach_roots(root_uris)
        if not servers:
            return
        logger.info(
            "lsp workspace route pool evicted %d root(s), retiring %d server(s)",
            len(root_uris),
            len(servers),
        )
        self._schedule_retirements(servers, routes, barrier_roots=(replacement_root,))

    def _resolved_root(
        self, config: ServerConfig, file_path: str | Path
    ) -> ResolvedWorkspaceRoot:
        return resolve_workspace_root(
            file_path,
            self._root_path or Path.cwd(),
            config.manifest_markers,
            explicit_root_uri=config.root_uri,
        )

    @staticmethod
    def _routed_config(
        config: ServerConfig, root: ResolvedWorkspaceRoot
    ) -> ServerConfig:
        return replace(
            config,
            command=list(config.command),
            languages=dict(config.languages),
            env=dict(config.env),
            root_uri=root.uri,
            cwd=config.cwd or (str(root.path) if root.path is not None else None),
        )

    def _matching_config(self, path: str | Path) -> ServerConfig | None:
        extension = Path(path).suffix
        if not extension:
            return None
        return next(
            (config for config in self._configs if config.matches(extension)), None
        )

    def _resolve_manifest_root(
        self, file_path: Path, markers: tuple[str, ...], default_root: Path
    ) -> Path:
        """Find the nearest ancestor dir containing any of ``markers``.

        Falls back to ``default_root`` when no marker is found or ``markers``
        is empty. Search is bounded by the session root so it never escapes
        the project — a marker above the session root (rare) is ignored.
        """
        return nearest_manifest_root(file_path, default_root, markers)

    async def _on_publish(self, params: dict[str, Any], server_name: str) -> None:
        self._diagnostics.publish(params, server_name)

    def get_server_for_file(self, path: str | Path) -> LanguageServer | None:
        config = self._matching_config(path)
        if config is None:
            return None
        root = self._resolved_root(config, path)
        if root.uri in self._route_drain_events:
            return None
        if not self._route_pool.touch_or_admit(
            root.uri, protected=self._root_is_protected(root)
        ):
            return None
        server = self._server_for_root(config, root)
        if server is not None and server.config.root_uri is not None:
            self._activate_deferred_retirements(server.config.root_uri)
        if (
            server is not None
            and server.config.root_uri in self._retirement_tasks_by_root
        ):
            return None
        return server

    @asynccontextmanager
    async def lease_server_for_file(
        self, path: str | Path
    ) -> AsyncIterator[LanguageServer | None]:
        while True:
            config = self._matching_config(path)
            if config is None:
                yield None
                return
            generation = self._routing_generation
            root = self._resolved_root(config, path)
            await self._await_root_drain(root.uri)
            if generation != self._routing_generation:
                continue
            self._activate_deferred_retirements(root.uri)
            admission = await self._route_pool.acquire(
                root.uri, protected=self._root_is_protected(root)
            )
            if generation == self._routing_generation:
                break
            await self._route_pool.release(root.uri)
        try:
            self._retire_evicted_roots(
                admission.evicted_roots, replacement_root=root.uri
            )
            server = self._server_for_root(config, root)
            await self._await_route_retirements(root.uri)
            yield server
        finally:
            await self._route_pool.release(root.uri)
            self._activate_deferred_retirements(root.uri)
            self._finish_root_drain_if_ready(root.uri)

    @asynccontextmanager
    async def _lease_existing_server(
        self, route: tuple[str, str], server: LanguageServer
    ) -> AsyncIterator[bool]:
        generation = self._routing_generation
        await self._await_root_drain(route[1])
        if (
            generation != self._routing_generation
            or self._servers_by_route.get(route) is not server
        ):
            yield False
            return
        pinned = await self._route_pool.pin_resident((route[1],))
        if not pinned or self._servers_by_route.get(route) is not server:
            if pinned:
                await self._route_pool.release_many(pinned)
                for root_uri in pinned:
                    self._activate_deferred_retirements(root_uri)
            yield False
            return
        try:
            await self._await_route_retirements(route[1])
            yield True
        finally:
            await self._route_pool.release_many(pinned)
            for root_uri in pinned:
                self._activate_deferred_retirements(root_uri)

    def status(self) -> dict[str, Any]:
        return self.readiness().model_dump(mode="json")

    def readiness(self, file_path: str | Path | None = None) -> LSPReadinessSnapshot:
        servers = self._servers
        selected: LanguageServer | None = None
        selected_root: ResolvedWorkspaceRoot | None = None
        if file_path is not None and (config := self._matching_config(file_path)):
            selected_root = self._resolved_root(config, file_path)
            selected = self._servers_by_route.get((config.name, selected_root.uri))
            if selected is None:
                selected = LanguageServer(self._routed_config(config, selected_root))
            servers = {
                name: server
                for name, server in self._servers.items()
                if server.config.name != config.name
            }
            servers[config.name] = selected
        snapshot = build_lsp_readiness(
            servers,
            enabled=True,
            generation=self.generation,
            file_path=file_path,
            selected_server=selected,
        )
        pool = self._route_pool.snapshot()
        deferred_servers = {
            server
            for servers in self._deferred_retirements_by_root.values()
            for server in servers
        }
        return snapshot.model_copy(
            update={
                "selected_workspace_root": (
                    selected_root.uri if selected_root is not None else None
                ),
                "route_pool": LSPRoutePoolReadiness(
                    resident_dynamic_roots=pool.resident_dynamic_roots,
                    max_dynamic_roots=pool.max_dynamic_roots,
                    leased_dynamic_roots=pool.leased_dynamic_roots,
                    resident_roots=pool.resident_roots,
                    known_roots=pool.known_roots,
                    workspace_symbol_partial=pool.workspace_symbol_partial,
                    retiring_servers=len(self._retiring_servers | deferred_servers),
                    revision=pool.revision,
                ),
            }
        )

    def readiness_fingerprint(self) -> tuple[Any, ...]:
        snapshot = self.readiness()
        return (
            snapshot.generation,
            snapshot.state,
            snapshot.route_pool,
            tuple(
                (
                    server.name,
                    server.state,
                    server.extensions,
                    server.operations,
                    server.error,
                )
                for server in snapshot.servers
            ),
        )

    def running_extensions(self) -> tuple[str, ...]:
        return tuple(
            sorted({
                extension
                for server in self.readiness().servers
                if server.ready and server.operations
                for extension in server.extensions
            })
        )

    @staticmethod
    def _server_is_ready_for(server: Any, operation: str | None) -> bool:
        return server.ready and (
            operation is None
            or (server.operations is not None and operation in server.operations)
        )

    def has_running_server_for(
        self,
        *,
        file_path: str | Path | None = None,
        extensions: tuple[str, ...] = (),
        language_id: str | None = None,
        operation: str | None = None,
    ) -> bool:
        if file_path is not None:
            snapshot = self.readiness(file_path)
            return any(
                server.name == snapshot.selected_server
                and self._server_is_ready_for(server, operation)
                for server in snapshot.servers
            )
        if extensions:
            normalized_extensions = {
                f".{extension.lower().lstrip('.')}"
                for extension in extensions
                if extension.lstrip(".")
            }
            return any(
                self._server_is_ready_for(server, operation)
                and not normalized_extensions.isdisjoint(server.extensions)
                for server in self.readiness().servers
            )
        snapshot = self.readiness()
        if language_id is not None:
            normalized = language_id.casefold()
            return any(
                self._server_is_ready_for(server, operation)
                and any(value.casefold() == normalized for value in server.language_ids)
                for server in snapshot.servers
            )
        return any(
            self._server_is_ready_for(server, operation) for server in snapshot.servers
        )

    def start_warmup(self) -> None:
        if self._warmup_task is not None and not self._warmup_task.done():
            return
        self._warmup_task = asyncio.create_task(self._warmup(), name="lsp-warmup")

    async def _warmup(self) -> None:
        async def start_server(
            name: str, route: tuple[str, str] | None, server: LanguageServer
        ) -> None:
            try:
                if route is None:
                    await server.ensure_started()
                    return
                async with self._lease_existing_server(route, server) as leased:
                    if leased:
                        await server.ensure_started()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("lsp warmup failed for %s", name, exc_info=True)

        routes_by_server = {
            server: route for route, server in self._servers_by_route.items()
        }

        await asyncio.gather(
            *(
                start_server(name, routes_by_server.get(server), server)
                for name, server in tuple(self._servers.items())
                if self._should_warm(server)
            )
        )

    def _should_warm(self, server: LanguageServer) -> bool:
        server_config = getattr(server, "config", None)
        if server_config is None:
            return True
        markers = server_config.manifest_markers
        if not markers:
            return True
        template = next(
            (config for config in self._configs if config.name == server_config.name),
            None,
        )
        if template is not None and template.root_uri is not None:
            return True
        root_uri = server_config.root_uri
        if root_uri is None or not root_uri.startswith("file:"):
            return False
        return directory_matches_markers(Path(path_from_uri(root_uri)), markers)

    async def send_request(
        self, path: str | Path, method: str, params: dict[str, Any] | None = None
    ) -> tuple[Any, LanguageServer]:
        async with self.lease_server_for_file(path) as server:
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

    async def send_request_all(
        self, method: str, params: dict[str, Any] | None = None
    ) -> list[Any]:
        while True:
            generation = self._routing_generation
            routes = tuple(self._servers_by_route.items())
            roots = tuple(dict.fromkeys(route[1] for route, _server in routes))
            for root_uri in roots:
                await self._await_root_drain(root_uri)
            if generation != self._routing_generation:
                continue
            pinned = await self._route_pool.pin_resident(roots)
            if generation == self._routing_generation:
                break
            await self._route_pool.release_many(pinned)
            for root_uri in pinned:
                self._activate_deferred_retirements(root_uri)
        pinned_set = set(pinned)
        leased = tuple(
            (route, server)
            for route, server in routes
            if route[1] in pinned_set and self._servers_by_route.get(route) is server
        )
        try:
            for root_uri in pinned:
                await self._await_route_retirements(root_uri)
            return list(
                await asyncio.gather(
                    *(server.send_request(method, params) for _route, server in leased),
                    return_exceptions=True,
                )
            )
        finally:
            await self._route_pool.release_many(pinned)
            for root_uri in pinned:
                self._activate_deferred_retirements(root_uri)

    async def open_document(
        self, path: str | Path, text: str, language_id: str
    ) -> LanguageServer | None:
        async with self.lease_server_for_file(path) as server:
            if server is None:
                return None
            await server.ensure_started()
            if not server.is_open(str(path)):
                await server.did_open(str(path), text, language_id)
            else:
                await server.sync_if_changed(str(path), text)
            return server

    async def notify_change(self, path: str | Path, text: str) -> None:
        async with self.lease_server_for_file(path) as server:
            if server is None or not server.is_open(str(path)):
                return
            await server.did_change(str(path), text)
            self._diagnostics.clear_for_path(str(path))

    async def notify_save(self, path: str | Path, text: str) -> None:
        async with self.lease_server_for_file(path) as server:
            if server is None or not server.is_open(str(path)):
                return
            await server.did_save(str(path), text)

    async def reinitialize(self) -> None:
        await self.shutdown()
        self._initialized = False
        self.initialize()

    async def shutdown(self) -> None:
        warmup_task = self._warmup_task
        self._warmup_task = None
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        deferred_servers = {
            server
            for servers in self._deferred_retirements_by_root.values()
            for server in servers
        }
        stops = [
            server.stop() for server in {*self._servers.values(), *deferred_servers}
        ]
        if stops:
            await asyncio.gather(*stops, return_exceptions=True)
        retirement_tasks = tuple(self._retirement_tasks)
        if retirement_tasks:
            await asyncio.gather(*retirement_tasks, return_exceptions=True)
        self._servers.clear()
        self._servers_by_route.clear()
        self._route_pool.reset()
        self._retirement_tasks.clear()
        self._retirement_tasks_by_root.clear()
        self._retiring_servers.clear()
        self._deferred_retirements_by_root.clear()
        for event in self._route_drain_events.values():
            event.set()
        self._route_drain_events.clear()
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
