from __future__ import annotations

import time

from vibe.core.workflows.models import (
    AgentResult,
    BudgetSnapshot,
    PhaseReport,
    WorkflowResult,
    WorkflowRun,
    WorkflowStatus,
)


def test_budget_snapshot_remaining_with_total() -> None:
    snap = BudgetSnapshot(total=100_000, reserved=30_000, spent=20_000)
    assert snap.remaining == 50_000


def test_budget_snapshot_remaining_unlimited() -> None:
    snap = BudgetSnapshot(total=None, reserved=0, spent=0)
    assert snap.remaining == float("inf")


def test_agent_result_tokens_total() -> None:
    r = AgentResult(prompt="test", response="ok", tokens_in=100, tokens_out=50)
    assert r.tokens_total == 150


def test_phase_report_aggregates() -> None:
    results = [
        AgentResult(prompt="a", response="a", tokens_in=100, tokens_out=50, cost=0.01),
        AgentResult(prompt="b", response="b", tokens_in=200, tokens_out=100, cost=0.02),
    ]
    phase = PhaseReport(name="Find", agent_results=results, elapsed_s=5.0)
    assert phase.tokens_total == 450
    assert phase.cost_total == 0.03


def test_workflow_run_aggregates() -> None:
    results = [AgentResult(prompt="a", response="a", tokens_in=100, tokens_out=50)]
    phases = [PhaseReport(name="Find", agent_results=results)]
    run = WorkflowRun(
        phases=phases, status=WorkflowStatus.COMPLETED, started_at=0.0, finished_at=10.0
    )
    assert run.tokens_total == 150
    assert run.agent_count == 1


def test_workflow_run_default_budget() -> None:
    run = WorkflowRun()
    assert run.budget.total is None
    assert run.budget.remaining == float("inf")


def test_workflow_result_serializes() -> None:
    results = [
        AgentResult(prompt="a", response={"key": "value"}, tokens_in=10, tokens_out=5)
    ]
    phases = [PhaseReport(name="Find", agent_results=results)]
    run = WorkflowRun(
        phases=phases, status=WorkflowStatus.COMPLETED, started_at=time.time()
    )
    result = WorkflowResult(return_value={"findings": []}, run=run, summary="done")
    json_str = result.model_dump_json()
    assert "findings" in json_str
    assert "COMPLETED" in json_str or "completed" in json_str


def test_workflow_status_enum() -> None:
    assert WorkflowStatus.RUNNING != WorkflowStatus.COMPLETED
    assert WorkflowStatus.PAUSED == "paused"
