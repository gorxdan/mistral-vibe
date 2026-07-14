from __future__ import annotations

from enum import StrEnum, auto
from typing import Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class OrchestrationRoute(StrEnum):
    DIRECT = auto()
    TASK = auto()
    WORKFLOW = auto()
    TEAM = auto()


class OrchestrationState(StrEnum):
    OFF = auto()
    PROVISIONAL_LOCAL = auto()
    ROUTE_REQUIRED = auto()
    DIRECT = auto()
    DELEGATION_PENDING = auto()
    DISTRIBUTED = auto()
    RECOVERY = auto()


class StrategyReason(StrEnum):
    LOCALIZED = auto()
    SEQUENTIALLY_COUPLED = auto()
    INDEPENDENT_LANES = auto()
    ADVERSARIAL_REVIEW = auto()
    LONG_RUNNING = auto()
    USER_CONSTRAINED = auto()
    USER_FORBIDS_AGENTS = auto()
    CAPABILITY_UNAVAILABLE = auto()
    CAPABILITY_FALLBACK = auto()


class WorkRisk(StrEnum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


class LaneOwner(StrEnum):
    HOST = auto()
    AGENT = auto()


class OrchestrationLane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    objective: str = Field(
        min_length=1,
        description="One independently actionable question or implementation unit.",
    )
    owner: LaneOwner = Field(
        default=LaneOwner.AGENT,
        description="Assign agent only when the lane benefits from delegation.",
    )
    profile: str | None = Field(
        default=None,
        description="Optional exact subagent profile required for this lane.",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("dependencies", "depends_on"),
        description="Lane IDs that must complete successfully before this lane.",
    )
    acceptance: list[str] = Field(
        default_factory=list,
        description="Observable facts that establish this lane is complete.",
    )
    expected_paths: list[str] = Field(
        default_factory=list,
        description="Narrow files or directories this lane is expected to inspect.",
    )

    @property
    def depends_on(self) -> list[str]:
        return self.dependencies


class StrategyEvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_id: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
        description="Host-issued ID of the completed strategy that exposed the gap.",
    )
    lane_ids: list[str] = Field(
        min_length=1,
        description="Completed lane IDs whose returned evidence exposed the gap.",
    )
    description: str = Field(
        min_length=1,
        description="Concrete missing evidence that justifies another bounded fan-out.",
    )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        description = value.strip()
        if not description:
            raise ValueError("Evidence-gap description cannot be blank")
        return description

    @model_validator(mode="after")
    def validate_lane_ids(self) -> Self:
        if len(self.lane_ids) != len(set(self.lane_ids)):
            raise ValueError("Evidence-gap lane IDs must be unique")
        return self


class OrchestrationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: OrchestrationRoute = Field(
        description=(
            "direct for localized/sequential host work; task for independent lanes; "
            "workflow for staged/adversarial fan-out; team for long-running coordination."
        )
    )
    objective: str = Field(default="Adaptive turn strategy", min_length=1)
    risk: WorkRisk = Field(description="Impact if the chosen route or result is wrong.")
    reason: StrategyReason = Field(
        description="The observed topology or constraint that justifies the route."
    )
    expected_paths: list[str] = Field(
        default_factory=list,
        description="Narrow files or directories the host expects to mutate directly.",
    )
    lanes: list[OrchestrationLane] = Field(
        default_factory=list,
        description=(
            "Concrete host and agent work lanes with dependencies; delegated routes "
            "may start at most two agent-owned evidence lanes."
        ),
    )
    evidence_gap: StrategyEvidenceGap | None = Field(
        default=None,
        description=(
            "Required for delegated expansion after a prior delegated strategy; "
            "binds the expansion to returned evidence from completed lanes."
        ),
    )

    @model_validator(mode="after")
    def validate_lane_graph(self) -> Self:
        lane_ids = [lane.id for lane in self.lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("Lane IDs must be unique")
        known = set(lane_ids)
        for lane in self.lanes:
            if lane.id in lane.dependencies:
                raise ValueError(f"Lane '{lane.id}' cannot depend on itself")
        dependencies = {lane.id: lane.dependencies for lane in self.lanes}
        visiting: list[str] = []
        visited: set[str] = set()

        def visit(lane_id: str) -> None:
            if lane_id in visited:
                return
            if lane_id in visiting:
                start = visiting.index(lane_id)
                cycle = " -> ".join([*visiting[start:], lane_id])
                raise ValueError(f"Lane dependencies contain a cycle: {cycle}")
            visiting.append(lane_id)
            for dependency in dependencies[lane_id]:
                if dependency in known:
                    visit(dependency)
            visiting.pop()
            visited.add(lane_id)

        for lane_id in lane_ids:
            visit(lane_id)
        return self


class OrchestrationCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: bool = False
    workflow: bool = False
    team: bool = False
    background_delivery: bool = False


class StrategyReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: OrchestrationRoute
    state: OrchestrationState
    message: str
    accepted: bool = True
    reason: StrategyReason | None = None
    required_delegations: int = Field(ge=0)
    strategy_id: str | None = None


class OrchestrationTurnSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: OrchestrationState
    route: OrchestrationRoute | None = None
    reason: StrategyReason | None = None
    capabilities: OrchestrationCapabilities
    reconnaissance_calls: int = 0
    direct_mutations: int = 0
    unique_paths: int = 0
    productive_delegations: int = 0
    completed_delegations: int = 0
    pending_delegations: int = 0
    verifier_delegations: int = 0
    required_delegations: int = 0
    failed_delegations: int = 0
    scope_drift: bool = False
    policy_nudges: int = 0
    user_allows_agents: bool = True
    user_allows_workflow: bool = True
    user_allows_team: bool = True
