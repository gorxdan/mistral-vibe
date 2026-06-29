from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from typing import TYPE_CHECKING

from vibe.core.lsp._defaults import _resolve_binary, available_presets
from vibe.core.lsp._manager import LSPServerSource
from vibe.core.lsp._server import ServerConfig

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


def _resolve_command_inplace(config: ServerConfig, root: Path | None) -> None:
    """Rewrite ``config.command[0]`` to the resolved absolute binary path.

    Ensures the spawned server process is the project-venv binary when one
    exists, not whatever stray global install is first on PATH.
    """
    if not config.command:
        return
    binary = config.command[0]
    if os.path.isabs(binary):
        return
    resolved = _resolve_binary(binary, root)
    if resolved is not None:
        config.command[0] = resolved


def build_server_configs(
    config: VibeConfig, root_path: str | Path | None = None
) -> list[ServerConfig]:
    """Merge manual ``[[lsp_servers]]`` entries with auto-discovered presets.

    Manual entries take precedence on name collision (a user who declares a
    custom ``pyright`` server wins over the preset). Presets whose binary is
    not on PATH are skipped. This is why a user with pyright installed gets
    Python support automatically once LSP is enabled — no config required.

    When ``root_path`` is given and ``config.lsp_auto_discover`` is True (the
    default), auto-discovered presets are filtered to those whose
    ``manifest_markers`` exist at the project root — so a Python-only repo no
    longer eagerly spawns rust-analyzer, gopls, and clangd. Setting
    ``lsp_auto_discover = false`` disables preset auto-discovery entirely,
    leaving only manually-declared ``[[lsp_servers]]`` entries (explicit-only,
    like MCP server config).
    """
    root = Path(root_path) if root_path is not None else None
    manual = [entry.to_server_config() for entry in config.lsp_servers]
    for cfg in manual:
        _resolve_command_inplace(cfg, root)
    manual_names = {s.name for s in manual}
    if not getattr(config, "lsp_auto_discover", True):
        return manual
    auto_presets = available_presets(root)
    auto = [p.server.to_server_config() for p in auto_presets]
    for cfg in auto:
        _resolve_command_inplace(cfg, root)
    return manual + [s for s in auto if s.name not in manual_names]


class ConfigServerSource(LSPServerSource):
    """Loads language-server definitions from config plus installed presets."""

    def __init__(
        self,
        config_getter: Callable[[], VibeConfig],
        root_path: str | Path | None = None,
    ) -> None:
        self._get = config_getter
        self._root_path = root_path

    def load(self) -> list[ServerConfig]:
        return build_server_configs(self._get(), self._root_path)
