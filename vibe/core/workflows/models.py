from __future__ import annotations

from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStatus(StrEnum):
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int | None
    reserved: int
    spent: int

    @property
    def remaining(self) -> int | float:
        if self.total is None:
            return float("inf")
        return self.total - self.reserved - self.spent


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    phase: str | None = None
    prompt: str
    response: str | dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    completed: bool = True
    error: str | None = None

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class PhaseReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    agent_results: list[AgentResult] = Field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def tokens_total(self) -> int:
        return sum(r.tokens_total for r in self.agent_results)

    @property
    def cost_total(self) -> float:
        return sum(r.cost for r in self.agent_results)


class WorkflowRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script_path: str | None = None
    args: Any = None
    phases: list[PhaseReport] = Field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.RUNNING
    started_at: float = 0.0
    finished_at: float | None = None
    budget: BudgetSnapshot = Field(
        default_factory=lambda: BudgetSnapshot(total=None, reserved=0, spent=0)
    )

    @property
    def tokens_total(self) -> int:
        return sum(p.tokens_total for p in self.phases)

    @property
    def cost_total(self) -> float:
        return sum(p.cost_total for p in self.phases)

    @property
    def agent_count(self) -> int:
        return sum(len(p.agent_results) for p in self.phases)


class WorkflowResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    return_value: Any = None
    run: WorkflowRun
    summary: str = ""


class CachedAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_hash: str
    agent: str
    label: str | None = None
    phase: str | None = None
    response: str | dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0
    completed: bool = True
    error: str | None = None


class WorkflowRunSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    script_source: str
    args: Any = None
    status: WorkflowStatus = WorkflowStatus.PAUSED
    started_at: float = 0.0
    budget_total: int | None = None
    budget_spent: int = 0
    cached_results: list[CachedAgentResult] = Field(default_factory=list)

    @property
    def cached_count(self) -> int:
        return len(self.cached_results)
