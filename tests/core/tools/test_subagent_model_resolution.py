from __future__ import annotations

from unittest.mock import MagicMock

from vibe.core.config import VibeConfig
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.builtins.task import _configured_subagent_model


def _ctx(config: VibeConfig) -> InvokeContext:
    manager = MagicMock()
    manager.config = config
    return InvokeContext(tool_call_id="test", agent_manager=manager)


def test_returns_none_when_unconfigured() -> None:
    config = VibeConfig(active_model="host")
    assert _configured_subagent_model(_ctx(config)) is None


def test_returns_none_when_no_manager() -> None:
    ctx = InvokeContext(tool_call_id="test", agent_manager=None)
    assert _configured_subagent_model(ctx) is None


def test_returns_alias_when_configured() -> None:
    config = VibeConfig(active_model="host", subagent_model="glm")
    assert _configured_subagent_model(_ctx(config)) == "glm"
