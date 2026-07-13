from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any, cast

from vibe.core.logger import logger
from vibe.core.lsp._manager import get_lsp_manager

_MAX_NOTIFY_BYTES = 10 * 1024 * 1024
_MAX_UTF8_BYTES_PER_CODEPOINT = 4


@asynccontextmanager
async def _lease_server(manager: Any, path: str | Path) -> AsyncIterator[Any]:
    lease = getattr(manager, "lease_server_for_file", None)
    if callable(lease):
        lease_server = cast(
            Callable[[str | Path], AbstractAsyncContextManager[Any]], lease
        )
        async with lease_server(path) as server:
            yield server
        return
    yield manager.get_server_for_file(path)


def readiness_fingerprint() -> tuple[object, ...]:
    manager = get_lsp_manager()
    if manager is None:
        return ("inactive",)
    return manager.readiness_fingerprint()


def running_extensions() -> tuple[str, ...]:
    manager = get_lsp_manager()
    if manager is None:
        return ()
    return manager.running_extensions()


async def notify_file_changed(path: str | Path, text: str) -> None:
    """Tell the LSP server that ``path`` was modified.

    Opens the document if needed (didOpen), then sends didChange + didSave so
    the server re-analyzes and republishes diagnostics. Safe no-op when LSP is
    inactive or no server matches the file. Never raises — LSP failures must
    not break the host tool's write.
    """
    manager = get_lsp_manager()
    if manager is None:
        return
    try:
        p = str(path)
        manager.clear_diagnostics_for(p)
        text_length = len(text)
        if text_length > _MAX_NOTIFY_BYTES or (
            text_length > _MAX_NOTIFY_BYTES // _MAX_UTF8_BYTES_PER_CODEPOINT
            and len(text.encode("utf-8")) > _MAX_NOTIFY_BYTES
        ):
            return
        async with _lease_server(manager, path) as server:
            if server is None:
                return
            await server.ensure_started()
            if not server.is_open(p):
                language_id = server.config.language_id_for(Path(path).suffix)
                await server.did_open(p, text, language_id)
            else:
                await server.did_change(p, text)
            await server.did_save(p, text)
    except Exception:
        logger.debug("lsp notify_file_changed failed for %s", path, exc_info=True)


def drain_diagnostics_into(stage: Callable[[str], None]) -> bool:
    """Drain pending LSP diagnostics and stage them for the next model turn.

    Returns True if anything was staged. No-op when LSP is inactive.
    """
    manager = get_lsp_manager()
    if manager is None:
        return False
    text = manager.consume_diagnostics_text()
    if not text:
        return False
    stage(text)
    return True


def schedule_notify_file_changed(path: str | Path, text: str) -> None:
    """Fire-and-forget variant for callers that can't await.

    Schedules the notification on the running loop without blocking.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(notify_file_changed(path, text))
    except RuntimeError:
        pass
