from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import os
from typing import Any

from vibe.core.logger import logger
from vibe.core.lsp._types import (
    CONTENT_MODIFIED_CODE,
    LSPProtocolError,
    LSPServerCrashedError,
    LSPTimeoutError,
)

JsonRpcHandler = Callable[[dict[str, Any]], Awaitable[Any | None]]
_HEADER_TERM = b"\r\n\r\n"
_LINE_TERM = b"\r\n"
_TRACE = os.environ.get("VIBE_LSP_TRACE", "") in {"1", "true", "TRUE"}


class _RequestCancelled(Exception):
    pass


class JsonRpcConnection:
    """JSON-RPC 2.0 over the LSP base protocol (Content-Length framing).

    Owns request/response correlation, outbound notifications/requests, and
    inbound server notifications and reverse-requests. The transport is a pair
    of asyncio streams (stdout reader, stdin writer) supplied by the caller.
    """

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._handlers: dict[str, JsonRpcHandler] = {}
        self._read_task: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()

    def on_notification(self, method: str, handler: JsonRpcHandler) -> None:
        self._handlers[method] = handler

    def start(self) -> None:
        if self._read_task is None:
            self._read_task = asyncio.create_task(
                self._read_loop(), name="lsp-rpc-read"
            )

    async def request(
        self, method: str, params: Any | None = None, *, timeout: float | None = None
    ) -> Any:
        if self._closed.is_set():
            raise LSPServerCrashedError(f"connection closed before {method}")
        self._next_id += 1
        req_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = future
        await self._write(payload)
        try:
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            self._pending.pop(req_id, None)
            raise LSPTimeoutError(f"{method} timed out after {timeout}s") from exc
        except _RequestCancelled as exc:
            raise LSPProtocolError(str(exc)) from exc

    async def notify(self, method: str, params: Any | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._write(payload)

    async def respond(self, req_id: Any, result: Any | None) -> None:
        await self._write({"jsonrpc": "2.0", "id": req_id, "result": result})

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(LSPServerCrashedError("connection closed"))
        self._pending.clear()
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except (asyncio.CancelledError, Exception):
                pass
            self._read_task = None
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    async def _write(self, payload: dict[str, Any]) -> None:
        if _TRACE:
            logger.debug(
                "lsp jsonrpc >>> %s id=%s",
                payload.get("method", "?"),
                payload.get("id", "-"),
            )
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = (
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n"
        ).encode("ascii")
        self._writer.write(header + body)
        try:
            await self._writer.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self._closed.set()
            raise LSPServerCrashedError("write failed") from exc

    async def _read_loop(self) -> None:
        try:
            while not self._closed.is_set():
                message = await self._read_message()
                if message is None:
                    break
                asyncio.create_task(self._dispatch(message))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("lsp jsonrpc read loop ended: %s", exc)
        finally:
            self._closed.set()
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(LSPServerCrashedError("server closed connection"))

    async def _read_message(self) -> dict[str, Any] | None:
        header = await self._reader.readuntil(_HEADER_TERM)
        if not header:
            return None
        length = self._parse_content_length(header)
        if length <= 0:
            return None
        body = await self._reader.readexactly(length)
        return json.loads(body.decode("utf-8"))

    @staticmethod
    def _parse_content_length(header: bytes) -> int:
        for line in header[: -len(_HEADER_TERM)].split(_LINE_TERM):
            if line.lower().startswith(b"content-length:"):
                try:
                    return int(line.split(b":", 1)[1].strip())
                except ValueError:
                    return 0
        return 0

    async def _dispatch(self, message: dict[str, Any]) -> None:
        if _TRACE:
            kind = (
                "response"
                if "id" in message and "method" not in message
                else message.get("method", "?")
            )
            logger.debug("lsp jsonrpc <<< %s id=%s", kind, message.get("id", "-"))
        if "id" in message and "method" in message:
            await self._handle_reverse_request(message)
        elif "id" in message:
            self._resolve_response(message)
        elif "method" in message:
            await self._handle_notification(message)

    def _resolve_response(self, message: dict[str, Any]) -> None:
        req_id = message["id"]
        future = self._pending.pop(req_id, None)
        if future is None or future.done():
            return
        if "error" in message and message["error"] is not None:
            err = message["error"]
            code = err.get("code")
            msg = err.get("message", "unknown error")
            if code == CONTENT_MODIFIED_CODE:
                future.set_exception(_RequestCancelled(msg))
            else:
                future.set_exception(
                    LSPProtocolError(
                        f"{msg} (code {code}): {err.get('data')}",
                        code=int(code) if code is not None else None,
                    )
                )
        else:
            future.set_result(message.get("result"))

    async def _handle_notification(self, message: dict[str, Any]) -> None:
        method = message.get("method", "")
        handler = self._handlers.get(method)
        if handler is None:
            return
        try:
            await handler(message.get("params") or {})
        except Exception:
            logger.exception("lsp notification handler failed for %s", method)

    async def _handle_reverse_request(self, message: dict[str, Any]) -> None:
        method = message.get("method", "")
        req_id = message["id"]
        handler = self._handlers.get(method)
        if handler is None:
            await self.respond(req_id, None)
            return
        try:
            result = await handler(message.get("params") or {})
        except Exception as exc:
            logger.debug("lsp reverse-request %s failed: %s", method, exc)
            await self.respond(req_id, None)
            return
        await self.respond(req_id, result)
