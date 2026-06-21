from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vibe.core.lsp._defaults import available_presets
from vibe.core.lsp._manager import LSPServerSource
from vibe.core.lsp._server import ServerConfig

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


def build_server_configs(config: VibeConfig) -> list[ServerConfig]:
    """Merge manual ``[[lsp_servers]]`` entries with auto-discovered presets.

    Manual entries take precedence on name collision (a user who declares a
    custom ``pyright`` server wins over the preset). Presets whose binary is
    not on PATH are skipped. This is why a user with pyright installed gets
    Python support automatically once LSP is enabled — no config required.
    """
    manual = [entry.to_server_config() for entry in config.lsp_servers]
    manual_names = {s.name for s in manual}
    auto = [p.server.to_server_config() for p in available_presets()]
    return manual + [s for s in auto if s.name not in manual_names]


class ConfigServerSource(LSPServerSource):
    """Loads language-server definitions from config plus installed presets."""

    def __init__(self, config_getter: Callable[[], VibeConfig]) -> None:
        self._get = config_getter

    def load(self) -> list[ServerConfig]:
        return build_server_configs(self._get())
