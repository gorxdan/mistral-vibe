from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vibe.core.lsp._manager import LSPServerSource
from vibe.core.lsp._server import ServerConfig

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


def build_server_configs(config: VibeConfig) -> list[ServerConfig]:
    return [entry.to_server_config() for entry in config.lsp_servers]


class ConfigServerSource(LSPServerSource):
    """Loads language-server definitions from :class:`VibeConfig.lsp_servers`."""

    def __init__(self, config_getter: Callable[[], VibeConfig]) -> None:
        self._get = config_getter

    def load(self) -> list[ServerConfig]:
        return build_server_configs(self._get())
