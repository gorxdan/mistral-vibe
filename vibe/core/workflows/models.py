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


class SchemaValidationFailure(dict):
    """Falsy dict returned (not raised) by ``spawn_agent`` when a schema-tagged
    agent exhausts its retries in non-strict mode. A ``dict`` subclass, not a
    pydantic model, so it survives ``json.dumps(results)`` -- the previous form
    crashed the whole run on one failed agent.

    Filter with ``[r for r in results if r]`` (truthiness), NOT
    ``isinstance(r, dict)``: it is a dict subclass, so isinstance would wrongly
    keep it. ``isinstance(r, SchemaValidationFailure)`` / ``r.schema_errors``
    still work for introspection.
    """

    def __init__(
        self,
        *,
        raw_response: str = "",
        error: str = "",
        schema_errors: list[str] | None = None,
    ) -> None:
        super().__init__(
            raw_response=raw_response,
            error=error,
            schema_errors=list(schema_errors or []),
        )

    def __bool__(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        # Attribute-style access (``f.schema_errors``). Real dict attributes
        # resolve first and never reach here; raise AttributeError so hasattr()
        # behaves correctly.
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


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
    # Inter-agent message board channels, captured so multi-phase scripts that
    # use post_message survive resume. Empty unless the script used the board.
    board: dict[str, list[Any]] = Field(default_factory=dict)
    return_value: Any = None

    @property
    def cached_count(self) -> int:
        return len(self.cached_results)
