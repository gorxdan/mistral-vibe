from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from vibe.core.logger import logger
from vibe.core.lsp._config_bridge import ConfigServerSource
from vibe.core.lsp._manager import (
    LSPManager,
    clear_lsp_manager,
    current_lsp_generation,
    get_lsp_manager,
    init_lsp_manager,
)
from vibe.core.lsp._route_pool import DEFAULT_MAX_WORKSPACE_ROOTS
from vibe.core.tools.utils import isolated_worktree_root

if TYPE_CHECKING:
    from collections.abc import Callable

    from vibe.core.config import VibeConfig


_retirement_tasks: set[asyncio.Task[None]] = set()


def _retire_manager(manager: LSPManager, loop: asyncio.AbstractEventLoop) -> None:
    task = loop.create_task(manager.shutdown())
    _retirement_tasks.add(task)
    task.add_done_callback(_retirement_tasks.discard)


def setup_lsp_for_config(
    config: VibeConfig,
    config_getter: Callable[[], VibeConfig],
    root_path: str | Path,
    *,
    warmup: bool = False,
) -> LSPManager | None:
    """Initialize (or replace) the process LSP manager from ``config``.

    Returns the manager, or ``None`` if LSP is not installed. Safe to call on
    every config reload — replaces the prior manager.
    """
    recipe = getattr(config, "trusted_verification_recipe", None)
    if recipe is not None and recipe.execution_topology is not None:
        teardown_lsp()
        logger.info(
            "lsp disabled for managed execution topology; project language "
            "servers are outside the host-pinned capability set"
        )
        return None
    if isolated_worktree_root() is not None:
        teardown_lsp()
        logger.info(
            "lsp disabled for isolated worktree execution until language "
            "servers can run inside the process sandbox"
        )
        return None
    if "lsp" not in getattr(config, "installed_components", []):
        teardown_lsp()
        return None
    started_at = current_lsp_generation()
    prior = get_lsp_manager()
    if prior is not None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                _retire_manager(prior, loop)
            else:
                loop.run_until_complete(prior.shutdown())
        except Exception:
            logger.debug("prior lsp manager shutdown failed", exc_info=True)
    if current_lsp_generation() != started_at:
        logger.debug(
            "lsp setup superseded by a newer generation %d->%d; not installing",
            started_at,
            current_lsp_generation(),
        )
        return get_lsp_manager()
    manager = LSPManager(
        source=ConfigServerSource(config_getter, root_path=root_path),
        max_workspace_roots=getattr(
            config, "lsp_max_workspace_roots", DEFAULT_MAX_WORKSPACE_ROOTS
        ),
    )
    manager.set_root(root_path)
    manager.initialize()
    init_lsp_manager(manager)
    if warmup:
        manager.start_warmup()
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
        loop = asyncio.get_event_loop()
        if loop.is_running():
            _retire_manager(manager, loop)
        else:
            loop.run_until_complete(manager.shutdown())
    except Exception:
        logger.debug("lsp teardown failed", exc_info=True)
    finally:
        clear_lsp_manager()


async def teardown_lsp_async() -> None:
    """Async-safe variant for shutdown sequences that can await."""
    manager = get_lsp_manager()
    pending = list(_retirement_tasks)
    try:
        if manager is not None:
            await manager.shutdown()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    except Exception:
        logger.debug("lsp teardown failed", exc_info=True)
    finally:
        clear_lsp_manager()
