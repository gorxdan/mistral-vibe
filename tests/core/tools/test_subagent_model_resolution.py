from __future__ import annotations

from unittest.mock import MagicMock

from vibe.core.config import PurposeModelRoutingConfig, VibeConfig
from vibe.core.tasking import TaskBrief, TaskManifestIdentity
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.builtins.task import (
    TaskArgs,
    _configured_grunt_model,
    _configured_subagent_model,
    _effective_subagent_model,
)


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


def test_grunt_model_resolver() -> None:
    config = VibeConfig(active_model="host", grunt_model="haiku")
    assert _configured_grunt_model(_ctx(config)) == "haiku"
    empty = VibeConfig(active_model="host")
    assert _configured_grunt_model(_ctx(empty)) is None


def test_effective_model_grunt_prefers_grunt_model() -> None:
    config = VibeConfig(active_model="host", grunt_model="haiku", subagent_model="glm")
    args = TaskArgs(task="rename X", agent="grunt")
    assert _effective_subagent_model(args, _ctx(config)) == "haiku"


def test_effective_model_grunt_falls_back_to_subagent_model() -> None:
    config = VibeConfig(active_model="host", subagent_model="glm")
    args = TaskArgs(task="rename X", agent="grunt")
    assert _effective_subagent_model(args, _ctx(config)) == "glm"


def test_effective_model_grunt_falls_back_to_host() -> None:
    config = VibeConfig(active_model="host")
    args = TaskArgs(task="rename X", agent="grunt")
    assert _effective_subagent_model(args, _ctx(config)) == "host"


def test_effective_model_explicit_arg_wins() -> None:
    config = VibeConfig(active_model="host", grunt_model="haiku", subagent_model="glm")
    args = TaskArgs(task="rename X", agent="grunt", model="spark")
    assert _effective_subagent_model(args, _ctx(config)) == "spark"


def test_effective_model_non_grunt_ignores_grunt_model() -> None:
    config = VibeConfig(active_model="host", grunt_model="haiku", subagent_model="glm")
    args = TaskArgs(task="explore", agent="explore")
    assert _effective_subagent_model(args, _ctx(config)) == "glm"


def test_mechanical_contract_forces_configured_cheap_model() -> None:
    config = VibeConfig(
        active_model="host",
        model_routing=PurposeModelRoutingConfig(mechanical_model="cheap"),
    )
    brief = TaskBrief(
        objective="apply mechanical edits",
        allowed_paths=["src/**"],
        acceptance_checks=["focused"],
        manifest=TaskManifestIdentity(name="mechanical-edit", version="1"),
    )
    args = TaskArgs(task=brief, agent="worker", model="strong")

    assert _effective_subagent_model(args, _ctx(config)) == "cheap"
