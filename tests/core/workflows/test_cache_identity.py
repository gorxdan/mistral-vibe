from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from vibe.core.agents.models import BUILTIN_AGENTS
from vibe.core.config import ModelConfig
from vibe.core.types import AssistantEvent
from vibe.core.workflows import _cache_identity
from vibe.core.workflows._cache_identity import workflow_cache_context
from vibe.core.workflows.runtime import WorkflowRuntime, _prompt_hash


class _AgentManager:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            active_model="primary",
            subagent_model="",
            grunt_model="grunt-model",
            models=[
                ModelConfig(name="primary-api", provider="test", alias="primary"),
                ModelConfig(name="grunt-api", provider="test", alias="grunt-model"),
            ],
        )

    def get_agent(self, name: str):
        return BUILTIN_AGENTS[name]


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        active_model="primary", agent_manager=_AgentManager(), tool_manager=None
    )


def test_cache_context_changes_with_repository_state(monkeypatch) -> None:
    repository = "tree-a"
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: repository)

    first = workflow_cache_context(_context(), agent="explore", model=None)
    repository = "tree-b"
    second = workflow_cache_context(_context(), agent="explore", model=None)

    assert first != second


def test_cache_context_changes_with_effective_model(monkeypatch) -> None:
    monkeypatch.setattr(_cache_identity, "_repository_fingerprint", lambda: "same-tree")

    primary = workflow_cache_context(_context(), agent="explore", model=None)
    grunt = workflow_cache_context(_context(), agent="grunt", model=None)

    assert primary != grunt


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

    runtime = WorkflowRuntime(agent_loop_factory=factory)
    first = await runtime.spawn_agent("same", agent="explore")
    repository = "tree-b"
    second = await runtime.spawn_agent("same", agent="explore")

    assert first == "result-1"
    assert second == "result-2"
    assert calls == 2
