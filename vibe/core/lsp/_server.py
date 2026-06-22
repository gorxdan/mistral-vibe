from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from typing import Any

from vibe.core.logger import logger
from vibe.core.lsp._jsonrpc import JsonRpcConnection
from vibe.core.lsp._types import (
    LSPError,
    LSPProtocolError,
    LSPServerCrashedError,
    ServerState,
    uri_from_path,
)

DEFAULT_STARTUP_TIMEOUT = 20.0
DEFAULT_REQUEST_TIMEOUT = 10.0
DEFAULT_MAX_RESTARTS = 3
_CONTENT_MODIFIED_RETRIES = 3
_CONTENT_MODIFIED_BACKOFF = (0.25, 0.5, 1.0)
_STDERR_TAIL = 10
_STDERR_DRAIN_WAIT = 0.5


@dataclass
class ServerConfig:
    name: str
    command: list[str]
    languages: dict[str, str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    root_uri: str | None = None
    initialization_options: dict[str, Any] | None = None
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_restarts: int = DEFAULT_MAX_RESTARTS

    def matches(self, extension: str) -> bool:
        ext = extension.lower().lstrip(".")
        return any(k.lower().lstrip(".") == ext for k in self.languages)

    def language_id_for(self, extension: str) -> str:
        ext = extension.lower().lstrip(".")
        for key, lang in self.languages.items():
            if key.lower().lstrip(".") == ext:
                return lang
        return "plaintext"


class LanguageServer:
    """One language-server process and its JSON-RPC connection."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._state = ServerState.STOPPED
        self._proc: asyncio.subprocess.Process | None = None
        self._conn: JsonRpcConnection | None = None
        self._capabilities: dict[str, Any] = {}
        self._open_docs: dict[str, int] = {}
        self._start_lock = asyncio.Lock()
        self._last_error: str | None = None
        self._crash_watcher: asyncio.Task[None] | None = None
        self._pending_handlers: list[tuple[str, Any]] = []
        self._stderr_lines: list[str] = []
        self._stderr_done: asyncio.Event = asyncio.Event()
        self._crash_count = 0

    @property
    def state(self) -> ServerState:
        return self._state

    @property
    def capabilities(self) -> dict[str, Any]:
        return self._capabilities

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def crash_count(self) -> int:
        return self._crash_count

    @property
    def restarts_exhausted(self) -> bool:
        return self._crash_count >= self.config.max_restarts

    @property
    def supports_diagnostics(self) -> bool:
        sync = self._capabilities.get("textDocumentSync")
        return bool(sync) or "textDocumentSync" in self._capabilities

    def on_notification(self, method: str, handler: Any) -> None:
        self._pending_handlers.append((method, handler))
        if self._conn is not None:
            self._conn.on_notification(method, handler)

    async def start(self) -> None:
        async with self._start_lock:
            if self._state in {ServerState.RUNNING, ServerState.STARTING}:
                return
            self._state = ServerState.STARTING
            self._last_error = None
            self._stderr_lines.clear()
            self._stderr_done = asyncio.Event()
            try:
                await self._spawn()
                await self._initialize()
            except (LSPError, OSError) as exc:
                await self._await_stderr_drain()
                if self._last_error is None:
                    self._last_error = self._with_stderr(str(exc) or type(exc).__name__)
                self._state = ServerState.ERRORED
                await self._force_kill()
                if isinstance(exc, LSPError):
                    raise type(exc)(self._last_error) from exc
                raise
            self._state = ServerState.RUNNING

    def _with_stderr(self, base: str) -> str:
        tail = "; ".join(self._stderr_lines[-_STDERR_TAIL:])
        return f"{base}; stderr: {tail}" if tail else base

    async def _await_stderr_drain(self) -> None:
        if self._proc is None:
            return
        try:
            await asyncio.wait_for(self._stderr_done.wait(), timeout=_STDERR_DRAIN_WAIT)
        except TimeoutError:
            pass

    async def _spawn(self) -> None:
        env = {**os.environ, **self.config.env}
        cwd = self.config.cwd or None
        logger.debug(
            "lsp starting server %s: %s", self.config.name, self.config.command
        )
        self._proc = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        assert self._proc.stdout is not None and self._proc.stdin is not None
        self._conn = JsonRpcConnection(self._proc.stdout, self._proc.stdin)
        for method, handler in self._pending_handlers:
            self._conn.on_notification(method, handler)
        self._conn.start()
        self._crash_watcher = asyncio.create_task(
            self._watch_exit(), name=f"lsp-exit-{self.config.name}"
        )
        if self._proc.stderr is not None:
            asyncio.create_task(
                self._drain_stderr(), name=f"lsp-stderr-{self.config.name}"
            )

    async def _initialize(self) -> None:
        assert self._conn is not None
        params: dict[str, Any] = {
            "processId": os.getpid(),
            "clientInfo": {"name": "chaton", "version": "0.1.0"},
            "locale": "en",
            "rootUri": self.config.root_uri,
            "capabilities": self._client_capabilities(),
            "workspaceFolders": (
                [{"uri": self.config.root_uri, "name": "workspace"}]
                if self.config.root_uri
                else None
            ),
        }
        if self.config.initialization_options is not None:
            params["initializationOptions"] = self.config.initialization_options
        result = await self._conn.request(
            "initialize", params, timeout=self.config.startup_timeout
        )
        self._capabilities = (result or {}).get("capabilities", {})
        await self._conn.notify("initialized", {})
        self._conn.on_notification(
            "workspace/configuration", lambda _params: self._config_response()
        )

    @staticmethod
    def _client_capabilities() -> dict[str, Any]:
        return {
            "workspace": {
                "configuration": False,
                "workspaceFolders": False,
                "applyEdit": False,
            },
            "window": {"workDoneProgress": False},
            "textDocument": {
                "synchronization": {
                    "didSave": True,
                    "willSave": False,
                    "willSaveWaitUntil": False,
                },
                "hover": {"contentFormat": ["markdown", "plaintext"]},
                "definition": {"linkSupport": True},
                "typeDefinition": {"linkSupport": True},
                "implementation": {"linkSupport": True},
                "references": {},
                "documentSymbol": {
                    "hierarchicalDocumentSymbolSupport": True,
                    "symbolKind": {"valueSet": list(range(1, 27))},
                },
                "workspaceSymbol": {"symbolKind": {"valueSet": list(range(1, 27))}},
                "callHierarchy": {"dynamicRegistration": False},
                "publishDiagnostics": {
                    "relatedInformation": True,
                    "versionSupport": False,
                    "tagSupport": {"valueSet": [1, 2]},
                },
            },
            "general": {"positionEncodings": ["utf-16"]},
            "offsetEncoding": ["utf-16"],
        }

    async def _config_response(self) -> list[Any]:
        return [{}]

    async def _watch_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            await proc.wait()
        except asyncio.CancelledError:
            raise
        if self._state not in {ServerState.RUNNING, ServerState.STARTING}:
            return
        self._crash_count += 1
        try:
            await asyncio.wait_for(self._stderr_done.wait(), timeout=_STDERR_DRAIN_WAIT)
        except TimeoutError:
            pass
        base = (
            f"server exited (code {proc.returncode}); "
            f"{self.config.max_restarts - self._crash_count} restarts left"
        )
        self._last_error = self._with_stderr(base)
        self._state = ServerState.ERRORED
        if self._conn is not None:
            await self._conn.close()

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            self._stderr_done.set()
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").rstrip()
                self._stderr_lines.append(text)
                if len(self._stderr_lines) > _STDERR_TAIL:
                    self._stderr_lines.pop(0)
                logger.debug("lsp %s stderr: %s", self.config.name, text)
        except (asyncio.CancelledError, OSError):
            pass
        finally:
            self._stderr_done.set()

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        await self.ensure_started()
        assert self._conn is not None
        last_exc: Exception | None = None
        for attempt in range(_CONTENT_MODIFIED_RETRIES):
            try:
                return await self._conn.request(
                    method, params, timeout=self.config.request_timeout
                )
            except LSPProtocolError as exc:
                last_exc = exc
                if attempt < _CONTENT_MODIFIED_RETRIES - 1:
                    await asyncio.sleep(_CONTENT_MODIFIED_BACKOFF[attempt])
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    async def ensure_started(self) -> None:
        if self._state == ServerState.RUNNING:
            return
        if self.restarts_exhausted:
            raise LSPServerCrashedError(
                f"{self.config.name} crashed {self._crash_count} times; "
                f"restart cap ({self.config.max_restarts}) exhausted"
            )
        await self.start()

    async def did_open(self, path: str, text: str, language_id: str) -> None:
        assert self._conn is not None
        uri = uri_from_path(path)
        self._open_docs[uri] = 1
        await self._conn.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )

    async def did_change(self, path: str, text: str) -> None:
        assert self._conn is not None
        uri = uri_from_path(path)
        version = self._open_docs.get(uri, 0) + 1
        self._open_docs[uri] = version
        await self._conn.notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": version},
                "contentChanges": [{"text": text}],
            },
        )

    async def did_save(self, path: str, text: str) -> None:
        assert self._conn is not None
        uri = uri_from_path(path)
        await self._conn.notify(
            "textDocument/didSave", {"textDocument": {"uri": uri}, "text": text}
        )

    async def did_close(self, path: str) -> None:
        assert self._conn is not None
        uri = uri_from_path(path)
        self._open_docs.pop(uri, None)
        await self._conn.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

    def is_open(self, path: str) -> bool:
        return uri_from_path(path) in self._open_docs

    async def stop(self) -> None:
        if self._state == ServerState.STOPPED:
            return
        self._state = ServerState.STOPPED
        if self._crash_watcher is not None:
            self._crash_watcher.cancel()
            self._crash_watcher = None
        if self._conn is not None:
            try:
                await self._conn.request("shutdown", None, timeout=5.0)
            except Exception:
                pass
            try:
                await self._conn.notify("exit", None)
            except Exception:
                pass
            await self._conn.close()
            self._conn = None
        await self._force_kill()
        self._open_docs.clear()

    async def _force_kill(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except (ProcessLookupError, OSError):
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except TimeoutError:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
