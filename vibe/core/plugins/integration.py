"""Shared plugin-loading entrypoint glue.

All three entrypoints (cli, acp new_session/load_session, and — via cli —
programmatic) call :func:`load_and_apply_plugins` so plugin discovery can't
diverge between them. Fully defensive: a bad plugin logs a warning, never
breaks launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe.core.logger import logger
from vibe.core.plugins.loader import apply_plugin_result, load_plugins_from_fs
from vibe.core.plugins.models import PluginLoadResult

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


def load_and_apply_plugins(config: VibeConfig) -> PluginLoadResult:
    """Discover plugins (explicit plugin_paths + trust-gated project/user dirs),
    fold their paths/mcp_servers into *config* in place, and return the result
    (the caller passes ``result.hooks`` to ``load_hooks_from_fs``).
    """
    from vibe.core.config.harness_files import get_harness_files_manager
    from vibe.core.paths import VIBE_HOME

    try:
        plugin_dirs = get_harness_files_manager().plugin_dirs
    except Exception as e:
        logger.warning("plugin dir discovery failed: %s", e)
        plugin_dirs = [VIBE_HOME.path / "plugins"]

    try:
        result = load_plugins_from_fs(
            config.plugin_paths,
            plugin_dirs,
            enabled=config.enabled_plugins,
            disabled=config.disabled_plugins,
        )
        apply_plugin_result(config, result)
        for issue in result.issues:
            logger.warning("plugin: %s", issue)
        return result
    except Exception as e:
        logger.warning("plugin loading failed: %s", e)
        return PluginLoadResult()
