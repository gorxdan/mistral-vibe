from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.agents.manager import AgentManager
from vibe.core.config import PurposeModelRoutingConfig
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.builtins.read import Read
from vibe.core.tools.manager import ToolManager
from vibe.core.types import AssistantEvent
from vibe.core.workflows import _cache_identity
from vibe.core.workflows._cache_identity import (
    workflow_cache_context,
    workflow_cache_identity,
)
from vibe.core.workflows.runtime import WorkflowRuntime, _prompt_hash

_TRUSTED_DEPENDENCIES = "a" * 64


def _context() -> InvokeContext:
    config = build_test_vibe_config()
    agent_manager = AgentManager(lambda: config)
    tool_manager = ToolManager(lambda: config, defer_mcp=True)
    return InvokeContext(
        tool_call_id="workflow-cache-test",
        active_model=agent_manager.config.active_model,
        agent_manager=agent_manager,
        tool_manager=tool_manager,
    )


def test_cache_context_changes_with_repository_state(monkeypatch) -> None:
    repository = "tree-a"
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: repository)

    first = workflow_cache_context(
        _context(),
        agent="explore",
        model=None,
        trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
    )
    repository = "tree-b"
    second = workflow_cache_context(
        _context(),
        agent="explore",
        model=None,
        trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
    )

    assert first != second


def test_cache_context_changes_with_effective_model(monkeypatch) -> None:
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "same-tree")
    context = _context()
    assert context.agent_manager is not None

    primary = workflow_cache_context(
        context,
        agent="explore",
        model=None,
        trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
    )
    context.agent_manager.config.models[0].temperature = 0.25
    alternate = workflow_cache_context(
        context,
        agent="explore",
        model=None,
        trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
    )

    assert primary != alternate


def test_cache_context_changes_with_formatter_routing(monkeypatch) -> None:
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "tree")
    context = _context()
    assert context.agent_manager is not None

    without_formatter = workflow_cache_context(
        context,
        agent="explore",
        model=None,
        trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
    )
    context.agent_manager.config.model_routing = PurposeModelRoutingConfig(
        formatter_model=context.agent_manager.config.active_model
    )
    with_formatter = workflow_cache_context(
        context,
        agent="explore",
        model=None,
        trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
    )

    assert without_formatter != with_formatter


@pytest.mark.parametrize("agent", ["worker", "grunt", "verifier", "research"])
def test_cache_context_rejects_unsafe_profiles(monkeypatch, agent: str) -> None:
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "tree")

    assert (
        workflow_cache_context(
            _context(),
            agent=agent,
            model=None,
            trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
        )
        is None
    )


def test_cache_context_requires_complete_trusted_dependency_fingerprint(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "tree")
    context = _context()

    assert workflow_cache_context(context, agent="explore", model=None) is None
    assert (
        workflow_cache_context(
            context,
            agent="explore",
            model=None,
            trusted_dependency_fingerprint="not-a-sha256",
        )
        is None
    )


def test_cache_context_binds_complete_dependency_fingerprint(monkeypatch) -> None:
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "tree")
    context = _context()

    first = workflow_cache_context(
        context, agent="explore", model=None, trusted_dependency_fingerprint="a" * 64
    )
    second = workflow_cache_context(
        context, agent="explore", model=None, trusted_dependency_fingerprint="b" * 64
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_cache_context_rejects_shadowed_builtin_tool(monkeypatch) -> None:
    class _ShadowRead(Read):
        read_only = True

    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "tree")
    context = _context()
    assert context.tool_manager is not None
    context.tool_manager._all_tools["read"] = _ShadowRead

    assert (
        workflow_cache_context(
            context,
            agent="explore",
            model=None,
            trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
        )
        is None
    )


def test_cache_identity_rejects_incomplete_or_effectful_runs(monkeypatch) -> None:
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "tree")
    context = _context()

    assert (
        workflow_cache_context(
            None,
            agent="explore",
            model=None,
            trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
        )
        is None
    )
    assert (
        workflow_cache_identity(
            context,
            agent="explore",
            model=None,
            isolation="worktree",
            then=None,
            contract=None,
            trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
        )
        is None
    )
    assert (
        workflow_cache_identity(
            context,
            agent="explore",
            model=None,
            isolation=None,
            then=None,
            contract=None,
            citations={"items_path": "findings"},
            trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
        )
        is None
    )
    assert (
        workflow_cache_identity(
            context,
            agent="explore",
            model=None,
            isolation=None,
            then="verifier",
            contract=None,
            trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
        )
        is None
    )
    assert (
        workflow_cache_identity(
            context,
            agent="explore",
            model=None,
            isolation=None,
            then=None,
            contract={"outputs": []},
            trusted_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
        )
        is None
    )


def test_prompt_hash_changes_with_cache_context() -> None:
    first = _prompt_hash("prompt", "explore", context_fingerprint="tree-a")
    second = _prompt_hash("prompt", "explore", context_fingerprint="tree-b")

    assert first != second


@pytest.mark.asyncio
async def test_workflow_cache_misses_after_repository_change(monkeypatch) -> None:
    repository = "tree-a"
    calls = 0
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: repository)

    @dataclass
    class _Stats:
        session_prompt_tokens: int = 1
        session_completion_tokens: int = 1

    @dataclass
    class _Loop:
        stats: _Stats = field(default_factory=_Stats)

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            nonlocal calls
            calls += 1
            yield AssistantEvent(content=f"result-{calls}")

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _Loop()

    runtime = WorkflowRuntime(
        agent_loop_factory=factory,
        parent_context=_context(),
        trusted_cache_dependency_fingerprint=_TRUSTED_DEPENDENCIES,
    )
    first = await runtime.spawn_agent("same", agent="explore")
    repository = "tree-b"
    second = await runtime.spawn_agent("same", agent="explore")

    assert first == "result-1"
    assert second == "result-2"
    assert calls == 2


@pytest.mark.asyncio
async def test_workflow_cache_is_disabled_without_trusted_dependencies(
    monkeypatch,
) -> None:
    calls = 0
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "tree")

    @dataclass
    class _Stats:
        session_prompt_tokens: int = 1
        session_completion_tokens: int = 1

    @dataclass
    class _Loop:
        stats: _Stats = field(default_factory=_Stats)

        async def act(
            self, prompt: str, *, response_format: Any = None
        ) -> AsyncGenerator[AssistantEvent, None]:
            nonlocal calls
            calls += 1
            yield AssistantEvent(content=f"result-{calls}")

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _Loop()

    runtime = WorkflowRuntime(agent_loop_factory=factory, parent_context=_context())

    assert await runtime.spawn_agent("same", agent="explore") == "result-1"
    assert await runtime.spawn_agent("same", agent="explore") == "result-2"
    assert calls == 2
    assert runtime._cache == {}
