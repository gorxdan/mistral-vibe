from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.tools.base import InvokeContext
from vibe.core.types import AgentStats, AssistantEvent
from vibe.core.workflows.runtime import IsolatedStats, WorkflowRuntime, _parse_stats


class _CostedLoop:
    def __init__(self, cost: float, *, estimated: bool = False) -> None:
        self.stats = AgentStats(
            session_prompt_tokens=1_000_000,
            session_completion_tokens=100_000,
            accumulated_cost_usd=cost,
            accumulated_cost_initialized=True,
            cost_is_estimated=estimated,
        )

    async def act(
        self, prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[AssistantEvent, None]:
        yield AssistantEvent(content="done", message_id="assistant")


@pytest.mark.asyncio
async def test_workflow_uses_exact_child_session_cost() -> None:
    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        return _CostedLoop(0.77)

    runtime = WorkflowRuntime(agent_loop_factory=factory, budget_total=2_000_000)

    await runtime.spawn_agent("work")

    result = runtime._phases["default"].agent_results[0]
    assert result.cost == pytest.approx(0.77)
    assert result.cost_estimated is False


def test_workflow_fallback_quote_honors_subscription_mode() -> None:
    model = ModelConfig(
        name="glm-5.2", provider="zai", alias="glm", pricing_mode="subscription"
    )
    config = build_test_vibe_config(
        active_model="glm",
        models=[model],
        providers=[ProviderConfig(name="zai", api_base="https://example.test/v1")],
    )
    manager = SimpleNamespace(config=config)
    context = InvokeContext(tool_call_id="workflow", agent_manager=cast(Any, manager))
    runtime = WorkflowRuntime(parent_context=context)

    assert runtime._compute_cost(1_000_000, 100_000, "glm") == 0.0


def test_workflow_fallback_quote_uses_builtin_api_pricing() -> None:
    model = ModelConfig(
        name="gpt-5.6-luna", provider="openai", alias="gpt", pricing_mode="api"
    )
    config = build_test_vibe_config(
        active_model="gpt",
        models=[model],
        providers=[ProviderConfig(name="openai", api_base="https://api.openai.com/v1")],
    )
    context = InvokeContext(
        tool_call_id="workflow", agent_manager=cast(Any, SimpleNamespace(config=config))
    )

    assert WorkflowRuntime(parent_context=context)._compute_cost(
        1_000_000, 100_000, "gpt"
    ) == pytest.approx(1.6)


@pytest.mark.asyncio
async def test_isolated_workflow_uses_child_reported_cost() -> None:
    async def isolated(
        prompt: str, agent: str, label: str | None, max_turns: int
    ) -> tuple[str, IsolatedStats]:
        return (
            "done",
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost_usd": 0.42,
                "cost_initialized": True,
                "cost_estimated": False,
            },
        )

    runtime = WorkflowRuntime(
        agent_loop_factory=cast(Any, lambda *args, **kwargs: _CostedLoop(0.0)),
        isolated_executor=isolated,
        budget_total=2_000_000,
    )

    await runtime.spawn_agent("work", isolation="worktree")

    result = runtime._phases["default"].agent_results[0]
    assert result.cost == pytest.approx(0.42)
    assert result.cost_estimated is False


def test_isolated_stats_parser_keeps_cache_and_cost_metadata() -> None:
    stats = _parse_stats(
        "log\n__VIBE_WORKFLOW_STATS__"
        '{"prompt_tokens":100,"completion_tokens":50,"cached_tokens":60,'
        '"cache_write_tokens":20,"reasoning_tokens":30,"cost_usd":0.42,'
        '"cost_initialized":true,"cost_estimated":false}\n'
    )

    assert stats == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cached_tokens": 60,
        "cache_write_tokens": 20,
        "reasoning_tokens": 30,
        "cost_usd": 0.42,
        "cost_initialized": True,
        "cost_estimated": False,
    }
