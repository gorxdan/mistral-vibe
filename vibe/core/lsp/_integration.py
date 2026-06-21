from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from vibe.core.logger import logger
from vibe.core.lsp._manager import get_lsp_manager


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
        server = manager.get_server_for_file(path)
        if server is None:
            return
        p = str(path)
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
