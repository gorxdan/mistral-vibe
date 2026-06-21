from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vibe.core.logger import logger
from vibe.core.lsp._config_bridge import ConfigServerSource
from vibe.core.lsp._manager import (
    LSPManager,
    clear_lsp_manager,
    get_lsp_manager,
    init_lsp_manager,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe.core.config import VibeConfig


def setup_lsp_for_config(
    config: VibeConfig, config_getter: Callable[[], VibeConfig], root_path: str | Path
) -> LSPManager | None:
    """Initialize (or replace) the process LSP manager from ``config``.

    Returns the manager, or ``None`` if LSP is not installed. Safe to call on
    every config reload — replaces the prior manager.
    """
    if "lsp" not in getattr(config, "installed_components", []):
        teardown_lsp()
        return None
    prior = get_lsp_manager()
    if prior is not None:
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(prior.shutdown())
            else:
                loop.run_until_complete(prior.shutdown())
        except Exception:
            logger.debug("prior lsp manager shutdown failed", exc_info=True)
    manager = LSPManager(source=ConfigServerSource(config_getter))
    manager.set_root(root_path)
    manager.initialize()
    init_lsp_manager(manager)
    logger.info(
        "lsp manager initialized: %d server(s) configured", len(manager.servers)
    )
    return manager


def teardown_lsp() -> None:
    """Shut down and drop the process LSP manager if one is active."""
    manager = get_lsp_manager()
    if manager is None:
        return
    try:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(manager.shutdown())
        else:
            loop.run_until_complete(manager.shutdown())
    except Exception:
        logger.debug("lsp teardown failed", exc_info=True)
    finally:
        clear_lsp_manager()


async def teardown_lsp_async() -> None:
    """Async-safe variant for shutdown sequences that can await."""
    manager = get_lsp_manager()
    if manager is None:
        return
    try:
        await manager.shutdown()
    except Exception:
        logger.debug("lsp teardown failed", exc_info=True)
    finally:
        clear_lsp_manager()
