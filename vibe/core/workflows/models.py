from __future__ import annotations

from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStatus(StrEnum):
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    COMPLETED_WITH_FAILURES = auto()
    FAILED = auto()
    STOPPED = auto()


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
    agent: str | None = None
    prompt: str
    response: str | dict[str, Any]
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    completed: bool = True
    error: str | None = None
    # Field-level detail for schema-validation failures (e.g.
    # "$.findings[0].severity: 'medium' not in enum"). Empty unless this agent
    # exhausted its schema-retry budget; surfaced via workflow_results so the
    # launching model sees *why* output was rejected, not just *that* it was.
    schema_errors: list[str] = Field(default_factory=list)

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out


class SchemaValidationFailure(BaseModel):
    """Structured return value when an agent exhausts its schema-retry budget.

    Returned (not raised) by ``WorkflowRuntime.spawn_agent`` in the default
    non-strict mode so a workflow script never silently loses the agent's raw
    output to ``None``. Callers that want the legacy hard-fail behavior can set
    ``strict_schema=True`` on the runtime, in which case ``spawn_agent`` raises
    ``SchemaValidationError`` instead.

    Scripts check for failure with ``isinstance(result, SchemaValidationFailure)``
    — but they don't have to. This is also a falsy, dict-like empty value so the
    common idioms degrade gracefully instead of crashing the whole run on one
    failed agent: the canonical filter ``[r for r in results if r]`` drops it
    (like the ``None`` from a raised agent), and ``r.get("findings", [])`` returns
    the default. The failure detail stays available via ``.error`` /
    ``.schema_errors`` / ``.raw_response``.
    """

    model_config = ConfigDict(extra="forbid")

    raw_response: str
    error: str
    schema_errors: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return False

    def get(self, key: str, default: Any = None) -> Any:
        return default


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
    schema_errors: list[str] = Field(default_factory=list)


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
    # The script's return value, captured so a completed run's result survives
    # session exit and can be re-read after resume. None until the run finishes
    # (or for runs that failed/cancelled before main() returned). Coerced to a
    # JSON-safe form at snapshot time, so a non-serializable return value degrades
    # to its string form rather than dropping the whole snapshot.
    return_value: Any = None

    @property
    def cached_count(self) -> int:
        return len(self.cached_results)
