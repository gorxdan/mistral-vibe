from __future__ import annotations

from vibe.core.workflows._limits import (
    DEFAULT_BUDGET_TOTAL,
    DEFAULT_ISOLATED_MAX_TURNS,
    DEFAULT_MAX_AGENTS,
    DEFAULT_MAX_CONCURRENT,
)
from vibe.core.workflows.runtime import WorkflowRuntime


def test_workflow_defaults_bound_paid_fanout() -> None:
    runtime = WorkflowRuntime()

    assert runtime.max_concurrent == DEFAULT_MAX_CONCURRENT == 2
    assert runtime.max_agents == DEFAULT_MAX_AGENTS == 32
    assert runtime.budget_total == DEFAULT_BUDGET_TOTAL == 500_000
    assert DEFAULT_ISOLATED_MAX_TURNS == 60
    assert runtime.budget_snapshot().total == DEFAULT_BUDGET_TOTAL
