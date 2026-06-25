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

    Gating model: plugins are opt-in by filesystem placement, not a config
    flag. Discovery only sees ``plugin_paths`` the user explicitly declared and
    ``<trusted-root>/.vibe/plugins`` + ``~/.vibe/plugins`` dirs, the latter two
    re-validated by ``HarnessFilesManager.plugin_dirs`` (resolved_within per
    root). So a plugin cannot load from an untrusted or unopened location. The
    plugin's own component dirs are confined to the plugin root by the loader.
    Plugin-contributed HOOKS additionally respect ``enable_experimental_hooks``
    (enforced in ``load_hooks_from_fs``); component paths (agents/skills/tools/
    workflows/mcp) are not behind a flag — placement in a trusted root IS the
    consent.
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
