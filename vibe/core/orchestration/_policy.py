from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Literal

from vibe.core._agent_limits import HOST_AGENT_LANE_LIMIT
from vibe.core.orchestration.models import (
    LaneOwner,
    OrchestrationCapabilities,
    OrchestrationDecision,
    OrchestrationLane,
    OrchestrationRoute,
    OrchestrationState,
    OrchestrationTurnSummary,
    StrategyReason,
    StrategyReceipt,
    WorkRisk,
)
from vibe.core.tasking._path_scope import path_matches_scope
from vibe.core.workflows.models import WorkflowLaneAttestation, WorkflowLaneExpectation

_CONTROL_TOOLS = frozenset({
    "ask_user_question",
    "background",
    "enter_plan_mode",
    "exit_plan_mode",
    "skill",
    "todo",
    "tool_search",
    "verify_work",
    "work_strategy",
    "workflow_results",
    "workflow_status",
})
_DELEGATION_TO_ROUTE = {
    "task": OrchestrationRoute.TASK,
    "launch_workflow": OrchestrationRoute.WORKFLOW,
    "team_spawn": OrchestrationRoute.TEAM,
}
_PATH_KEYS = ("file_path", "path")
_NO_AGENTS = re.compile(
    r"\b(?:do not|don't|without|no)\s+(?:(?:use|spawn|delegate(?:\s+to)?)\s+)?"
    r"(?:sub[- ]?)?agents?\b",
    re.IGNORECASE,
)
_NO_WORKFLOW = re.compile(
    r"\b(?:do not|don't|without|no)\s+"
    r"(?:(?:use|launch|run)\s+)?workflows?\b",
    re.IGNORECASE,
)
_NO_TEAM = re.compile(
    r"\b(?:do not|don't|without|no)\s+"
    r"(?:(?:use|spawn|launch)\s+)?teams?\b",
    re.IGNORECASE,
)
_EXPLICIT_AGENT_DELEGATION = re.compile(
    r"\b(?:(?:use|spawn)\s+(?:(?:multiple|several)\s+)?(?:sub[- ]?)?agents?|"
    r"delegate(?:\s+\w+){0,3}\s+to\s+(?:sub[- ]?)?agents?|"
    r"parallel(?:ize|ise)|fan[- ]?out|"
    r"(?:use|run|perform|do)\s+(?:an?\s+)?multi[- ]?agent)\b",
    re.IGNORECASE,
)
_EXPLICIT_WORKFLOW = re.compile(
    r"\b(?:(?:use|launch|run)\s+(?:an?\s+|the\s+)?workflows?|"
    r"(?:using|via|with)\s+(?:an?\s+|the\s+)?workflows?)\b",
    re.IGNORECASE,
)
_EXPLICIT_TEAM = re.compile(
    r"\b(?:(?:use|spawn|launch)\s+(?:an?\s+|the\s+)?teams?|"
    r"(?:using|via|with)\s+(?:an?\s+|the\s+)?teams?)\b",
    re.IGNORECASE,
)
_EXPLICIT_DIRECT = re.compile(
    r"\b(?:(?:do|handle|proceed|continue|work)\s+(?:it\s+)?direct(?:ly)?|"
    r"(?:use|choose|go)\s+(?:the\s+)?direct(?:\s+route)?|"
    r"direct(?:ly)?\s+(?:instead|only))\b",
    re.IGNORECASE,
)
_SUBSTANTIVE = re.compile(
    r"\b(?:audit|migration|refactor|implement|investigate|system[- ]wide|"
    r"cross[- ]cutting|multi[- ]file|architecture|security review|analy[sz]e\s+"
    r"(?:the\s+)?(?:logs|traces|sessions|repository|repo|codebase))\b",
    re.IGNORECASE,
)
_HOST_HIGH_RISK = re.compile(
    r"\b(?:high[- ]risk|multi[- ]system|cross[- ]system|"
    r"production[- ]critical|security[- ]critical|release acceptance|"
    r"credential rotation)\b",
    re.IGNORECASE,
)
_DIRECT_MUTATION_LIMIT = 8
_DIRECT_PATH_LIMIT = 2
_IMPLICIT_DIRECT_RECON_LIMIT = 2
_RECON_NUDGE_THRESHOLD = 4
_WORKFLOW_MIN_AGENT_LANES = 2
_WORKFLOW_RESERVED_HELPERS = frozenset({"agent", "parallel", "pipeline"})
_WORKFLOW_UNKNOWN = object()
_WORK_RISK_RANK = {WorkRisk.LOW: 0, WorkRisk.MEDIUM: 1, WorkRisk.HIGH: 2}


@dataclass(frozen=True)
class _RecoveryDecisionIdentity:
    route: OrchestrationRoute
    objective: str
    risk: WorkRisk
    expected_paths: tuple[str, ...]


@dataclass(frozen=True)
class _RecoveryLaneIdentity:
    id: str
    agent_profile: str | None
    objective: str
    dependencies: tuple[str, ...]
    acceptance: tuple[str, ...]
    expected_paths: tuple[str, ...]


@dataclass(frozen=True)
class _RecoveryIdentity:
    decision: _RecoveryDecisionIdentity
    lanes: tuple[_RecoveryLaneIdentity, ...]


class OrchestrationController:
    def __init__(self, *, workspace_root: Path | None = None) -> None:
        self._workspace_root = (workspace_root or Path.cwd()).expanduser().resolve()
        self.state = OrchestrationState.OFF
        self.capabilities = OrchestrationCapabilities()
        self.decision: OrchestrationDecision | None = None
        self.user_allows_agents = True
        self._user_allows_workflow = True
        self._user_allows_team = True
        self._user_prompt = ""
        self._requires_strategy = False
        self._explicit_delegation = False
        self._reconnaissance_calls = 0
        self._mutation_calls = 0
        self._mutation_paths: set[str] = set()
        self._total_mutation_calls = 0
        self._total_mutation_paths: set[str] = set()
        self._productive_delegations = 0
        self._verifier_delegations = 0
        self._required_delegations = 0
        self._delegation_failures = 0
        self._policy_nudges = 0
        self._inferred_route: OrchestrationRoute | None = None
        self._implicit_direct_path: str | None = None
        self._scope_drift = False
        self._launched_lane_ids: set[str] = set()
        self._completed_lane_ids: set[str] = set()
        self._task_lanes_by_id: dict[str, tuple[int, set[str]]] = {}
        self._init_workflow_tracking()
        self._team_lanes_by_id: dict[str, tuple[int, set[str]]] = {}
        self._deferred_task_results: dict[str, bool] = {}
        self._deferred_team_results: dict[str, bool] = {}
        self._reserved_lanes_by_call: dict[str, set[str]] = {}
        self._failed_lane_ids: set[str] = set()
        self._failed_lane_groups: set[frozenset[str]] = set()
        self._failed_recovery_identities: dict[frozenset[str], _RecoveryIdentity] = {}
        self._unbound_terminal_failures = 0
        self._unresolved_terminal_failures = 0
        self._lifecycle_generation = 0
        self._terminal_deliveries: dict[tuple[OrchestrationRoute, str], int] = {}
        self._continuation_markers: dict[
            str, tuple[int, frozenset[tuple[OrchestrationRoute, str]]]
        ] = {}
        self._continuation_counter = 0
        self._risk_floor = WorkRisk.LOW
        self._current_strategy_id: str | None = None
        self._delegated_strategy_started = False
        self._terminal_strategy_evidence: dict[str, frozenset[str]] = {}
        self._consumed_evidence_strategy_ids: set[str] = set()
        self._successful_lane_ids: set[str] = set()
        self._active_external_dependencies: set[str] = set()
        self._route_revalidation_required = False

    def _init_workflow_tracking(self) -> None:
        self._workflow_lanes_by_id: dict[str, tuple[int, set[str]]] = {}
        self._workflow_expectations_by_id: dict[
            str, tuple[WorkflowLaneExpectation, ...]
        ] = {}
        self._deferred_workflow_results: dict[
            str, tuple[bool, WorkflowLaneAttestation | None]
        ] = {}
        self._reserved_workflow_expectations_by_call: dict[
            str, tuple[WorkflowLaneExpectation, ...]
        ] = {}

    def begin_turn(
        self,
        *,
        enabled: bool,
        user_prompt: str,
        capabilities: OrchestrationCapabilities,
        continuation_id: str | None = None,
    ) -> None:
        self._reserved_lanes_by_call.clear()
        self._reserved_workflow_expectations_by_call.clear()
        synthetic_continuation = self._consume_continuation_marker(
            continuation_id, enabled=enabled
        )
        open_debt = enabled and self._has_open_strategy_debt()
        if not enabled:
            self._lifecycle_generation += 1
            self._risk_floor = WorkRisk.LOW
            self.state = OrchestrationState.OFF
            self.decision = None
            self._clear_terminal_failures()
            self._productive_delegations = 0
            self._required_delegations = 0
            self._inferred_route = None
            self._launched_lane_ids.clear()
            self._completed_lane_ids.clear()
            self._reset_strategy_campaign()
        elif synthetic_continuation or open_debt:
            if self._unresolved_terminal_failures:
                self.state = OrchestrationState.RECOVERY
        else:
            self._lifecycle_generation += 1
            self._risk_floor = WorkRisk.LOW
            self.state = (
                OrchestrationState.DISTRIBUTED
                if self._pending_delegations
                else OrchestrationState.PROVISIONAL_LOCAL
            )
            self.decision = None
            self._productive_delegations = 0
            self._required_delegations = 0
            self._inferred_route = None
            self._launched_lane_ids.clear()
            self._completed_lane_ids.clear()
            self._reset_strategy_campaign()
        self.capabilities = capabilities
        if synthetic_continuation:
            pass
        elif open_debt:
            self._merge_user_intent(user_prompt)
        else:
            self._replace_user_intent(user_prompt)
        self._reconnaissance_calls = 0
        self._mutation_calls = 0
        self._mutation_paths.clear()
        self._total_mutation_calls = 0
        self._total_mutation_paths.clear()
        self._verifier_delegations = 0
        self._delegation_failures = self._unresolved_terminal_failures
        self._policy_nudges = 0
        self._implicit_direct_path = None
        self._scope_drift = False
        self._implicit_lane_counter = 0

    def _reset_strategy_campaign(self) -> None:
        self._current_strategy_id = None
        self._delegated_strategy_started = False
        self._terminal_strategy_evidence.clear()
        self._consumed_evidence_strategy_ids.clear()
        self._successful_lane_ids.clear()
        self._failed_recovery_identities.clear()
        self._active_external_dependencies.clear()
        self._route_revalidation_required = False

    def _clear_terminal_failures(self) -> None:
        self._unresolved_terminal_failures = 0
        self._failed_lane_ids.clear()
        self._failed_lane_groups.clear()
        self._failed_recovery_identities.clear()
        self._unbound_terminal_failures = 0

    def issue_continuation(
        self, *, route: OrchestrationRoute | None = None, launch_id: str | None = None
    ) -> str | None:
        if (route is None) != (launch_id is None):
            raise ValueError("route and launch_id must be provided together")
        if self.state is OrchestrationState.OFF:
            return None

        terminal_deliveries = {
            key
            for key, generation in self._terminal_deliveries.items()
            if generation == self._lifecycle_generation
        }
        bound_deliveries: set[tuple[OrchestrationRoute, str]] = set()
        if route is not None and launch_id is not None:
            key = (route, launch_id)
            launch_generation = self._launch_generation(route, launch_id)
            if key not in terminal_deliveries and (
                launch_generation != self._lifecycle_generation
            ):
                return None
            if key in terminal_deliveries:
                bound_deliveries.add(key)
        elif terminal_deliveries:
            bound_deliveries.update(terminal_deliveries)
        elif not self._has_open_strategy_debt():
            return None

        self._continuation_counter += 1
        continuation_id = (
            f"orchestration-{self._lifecycle_generation}-{self._continuation_counter}"
        )
        self._continuation_markers[continuation_id] = (
            self._lifecycle_generation,
            frozenset(bound_deliveries),
        )
        return continuation_id

    def _consume_continuation_marker(
        self, continuation_id: str | None, *, enabled: bool
    ) -> bool:
        marker = (
            self._continuation_markers.pop(continuation_id, None)
            if continuation_id is not None
            else None
        )
        self._continuation_markers.clear()
        valid = bool(
            enabled and marker is not None and marker[0] == self._lifecycle_generation
        )
        if valid and marker is not None:
            for delivery in marker[1]:
                self._terminal_deliveries.pop(delivery, None)
            return True
        self._terminal_deliveries.clear()
        return False

    def _launch_generation(
        self, route: OrchestrationRoute, launch_id: str
    ) -> int | None:
        launches = {
            OrchestrationRoute.TASK: self._task_lanes_by_id,
            OrchestrationRoute.WORKFLOW: self._workflow_lanes_by_id,
            OrchestrationRoute.TEAM: self._team_lanes_by_id,
        }[route]
        launch = launches.get(launch_id)
        return launch[0] if launch is not None else None

    def _replace_user_intent(self, user_prompt: str) -> None:
        self.user_allows_agents = _NO_AGENTS.search(user_prompt) is None
        self._user_allows_workflow = (
            self.user_allows_agents and _NO_WORKFLOW.search(user_prompt) is None
        )
        self._user_allows_team = (
            self.user_allows_agents and _NO_TEAM.search(user_prompt) is None
        )
        self._user_prompt = user_prompt.lower()
        self._requires_strategy = _SUBSTANTIVE.search(user_prompt) is not None
        self._explicit_delegation = bool(
            (self.user_allows_agents and _EXPLICIT_AGENT_DELEGATION.search(user_prompt))
            or (self._user_allows_workflow and _EXPLICIT_WORKFLOW.search(user_prompt))
            or (self._user_allows_team and _EXPLICIT_TEAM.search(user_prompt))
        )
        self._raise_risk_floor_from_prompt(user_prompt)

    def _merge_user_intent(self, user_prompt: str) -> None:
        no_agents = _NO_AGENTS.search(user_prompt) is not None
        no_workflow = _NO_WORKFLOW.search(user_prompt) is not None
        no_team = _NO_TEAM.search(user_prompt) is not None
        use_agents = _EXPLICIT_AGENT_DELEGATION.search(user_prompt) is not None
        use_workflow = _EXPLICIT_WORKFLOW.search(user_prompt) is not None
        use_team = _EXPLICIT_TEAM.search(user_prompt) is not None

        if no_agents:
            self.user_allows_agents = False
            self._user_allows_workflow = False
            self._user_allows_team = False
            self._explicit_delegation = False
        else:
            if use_agents or use_workflow or use_team:
                self.user_allows_agents = True
                self._explicit_delegation = True
            if no_workflow:
                self._user_allows_workflow = False
            elif use_workflow:
                self._user_allows_workflow = True
            if no_team:
                self._user_allows_team = False
            elif use_team:
                self._user_allows_team = True
            if _EXPLICIT_DIRECT.search(user_prompt):
                self._explicit_delegation = False

        self._user_prompt = f"{self._user_prompt}\n{user_prompt.lower()}"
        self._requires_strategy = bool(
            self._requires_strategy or _SUBSTANTIVE.search(user_prompt)
        )
        self._raise_risk_floor_from_prompt(user_prompt)

    def _raise_risk_floor_from_prompt(self, user_prompt: str) -> None:
        if _HOST_HIGH_RISK.search(user_prompt) is None:
            return
        self._risk_floor = WorkRisk.HIGH
        if self.decision is not None:
            self.decision = self.decision.model_copy(update={"risk": WorkRisk.HIGH})
        for group, identity in self._failed_recovery_identities.items():
            self._failed_recovery_identities[group] = replace(
                identity, decision=replace(identity.decision, risk=WorkRisk.HIGH)
            )

    def declare(self, decision: OrchestrationDecision) -> StrategyReceipt:
        if self.state is OrchestrationState.OFF:
            raise ValueError("Work strategy is only available in Le Chaton host turns")

        previous_state = self.state
        previous_decision = self.decision
        submitted_risk = decision.risk
        decision = self._apply_risk_floor(
            decision, latch=previous_state is not OrchestrationState.RECOVERY
        )
        fallback = self._fallback_decision(decision)
        if fallback is not None:
            decision = fallback
        try:
            (decision, evidence_strategy_id, external_dependencies) = (
                self._validate_strategy_replacement(
                    decision,
                    previous_state=previous_state,
                    previous_decision=previous_decision,
                    submitted_risk=submitted_risk,
                )
            )
        except ValueError as exc:
            return self._rejected_strategy_receipt(
                decision,
                exc,
                previous_state=previous_state,
                previous_decision=previous_decision,
            )
        self._latch_accepted_risk(decision.risk)
        self._activate_strategy(
            decision,
            evidence_strategy_id=evidence_strategy_id,
            external_dependencies=external_dependencies,
        )
        message = self._apply_strategy_route(decision)

        return StrategyReceipt(
            route=decision.route,
            state=self.state,
            message=message,
            reason=(
                StrategyReason.CAPABILITY_FALLBACK
                if fallback is not None
                else decision.reason
            ),
            required_delegations=self._required_delegations,
            strategy_id=self._current_strategy_id,
        )

    def _activate_strategy(
        self,
        decision: OrchestrationDecision,
        *,
        evidence_strategy_id: str | None,
        external_dependencies: set[str],
    ) -> None:
        self._lifecycle_generation += 1
        self._current_strategy_id = f"strategy-{self._lifecycle_generation}"
        self._terminal_deliveries.clear()
        self._continuation_markers.clear()
        self.decision = decision
        self._active_external_dependencies = set(external_dependencies)
        self._clear_terminal_failures()
        self._inferred_route = None
        self._productive_delegations = 0
        self._launched_lane_ids.clear()
        self._completed_lane_ids.clear()
        self._reserved_lanes_by_call.clear()
        self._reserved_workflow_expectations_by_call.clear()
        self._mutation_calls = 0
        self._mutation_paths.clear()
        self._implicit_direct_path = None
        self._route_revalidation_required = False
        if decision.route is not OrchestrationRoute.DIRECT:
            self._delegated_strategy_started = True
            if evidence_strategy_id is not None:
                self._consumed_evidence_strategy_ids.add(evidence_strategy_id)

    def _apply_strategy_route(self, decision: OrchestrationDecision) -> str:
        agent_lanes = sum(lane.owner is LaneOwner.AGENT for lane in decision.lanes)
        lane_ids = [lane.id for lane in decision.lanes if lane.owner is LaneOwner.AGENT]
        match decision.route:
            case OrchestrationRoute.DIRECT:
                self._required_delegations = 0
                self.state = OrchestrationState.DIRECT
                message = "Direct strategy recorded; the host remains hands-on."
            case OrchestrationRoute.TASK:
                self._required_delegations = max(1, agent_lanes)
                self.state = OrchestrationState.DELEGATION_PENDING
                message = (
                    "Task strategy recorded; launch each productive lane once and "
                    "include its marker in the task text: "
                    + ", ".join(f"[lane:{lane_id}]" for lane_id in lane_ids)
                )
            case OrchestrationRoute.WORKFLOW:
                self._required_delegations = 1
                self.state = OrchestrationState.DELEGATION_PENDING
                message = (
                    "Workflow strategy recorded; bind each planned agent() call to "
                    "one literal label: "
                    + ", ".join(f"label='{lane_id}'" for lane_id in lane_ids)
                )
            case OrchestrationRoute.TEAM:
                self._required_delegations = max(1, agent_lanes)
                self.state = OrchestrationState.DELEGATION_PENDING
                message = (
                    "Team strategy recorded; spawn each declared lane with its "
                    "[lane:<id>] marker before substantive mutation."
                )
        return message

    def _validate_strategy_replacement(
        self,
        decision: OrchestrationDecision,
        *,
        previous_state: OrchestrationState,
        previous_decision: OrchestrationDecision | None,
        submitted_risk: WorkRisk,
    ) -> tuple[OrchestrationDecision, str | None, set[str]]:
        decision = self._canonicalize_expected_paths(decision)
        external_dependencies = self._candidate_external_dependencies(
            decision, previous_state=previous_state, previous_decision=previous_decision
        )
        self._validate_decision(
            decision, allowed_external_dependencies=external_dependencies
        )
        if previous_decision is not None and self._has_active_strategy_launch():
            raise ValueError(
                "The accepted strategy still has an active delegation; wait "
                "for its terminal result before replacing the route or lanes"
            )
        if (
            previous_decision is not None
            and previous_state is not OrchestrationState.RECOVERY
        ):
            unfinished = {
                lane.id
                for lane in previous_decision.lanes
                if lane.owner is LaneOwner.AGENT
            } - self._completed_lane_ids
            if unfinished:
                names = ", ".join(sorted(unfinished))
                raise ValueError(
                    "The accepted strategy still owes terminal evidence for "
                    f"lane(s): {names}"
                )
        exact_recovery = self._validate_recovery_replacement(
            decision, previous_state=previous_state, submitted_risk=submitted_risk
        )
        if decision.route is OrchestrationRoute.DIRECT:
            if decision.evidence_gap is not None:
                raise ValueError("evidence_gap applies only to a delegated expansion")
            return decision, None, external_dependencies
        if exact_recovery:
            return decision, None, external_dependencies
        evidence_strategy_id = self._validate_delegated_expansion(decision)
        return decision, evidence_strategy_id, external_dependencies

    def _validate_recovery_replacement(
        self,
        decision: OrchestrationDecision,
        *,
        previous_state: OrchestrationState,
        submitted_risk: WorkRisk,
    ) -> bool:
        if previous_state is not OrchestrationState.RECOVERY:
            return False
        exact_recovery = self._matches_failed_recovery(
            decision, submitted_risk=submitted_risk
        )
        if exact_recovery:
            if decision.evidence_gap is not None:
                raise ValueError(
                    "An exact failed-lane retry does not accept evidence_gap"
                )
            return True
        if decision.evidence_gap is None:
            raise ValueError(
                "A gap-free recovery must exactly match the failed route, lane set, "
                "decision objective, risk, expected paths, and each failed lane's "
                "profile, objective, dependencies, acceptance, and expected paths"
            )
        return False

    def _validate_delegated_expansion(
        self, decision: OrchestrationDecision
    ) -> str | None:
        gap = decision.evidence_gap
        if not self._delegated_strategy_started:
            if gap is not None:
                raise ValueError(
                    "The first delegated strategy cannot claim a prior evidence gap"
                )
            return None

        if gap is None:
            raise ValueError(
                "Another delegated strategy requires evidence_gap bound to a "
                "completed strategy and its returned lane evidence"
            )
        proposed_lane_ids = {lane.id for lane in decision.lanes}
        if reused := proposed_lane_ids & self._successful_lane_ids:
            names = ", ".join(sorted(reused))
            raise ValueError(
                "A delegated expansion cannot reuse successful lane identities: "
                f"{names}"
            )
        completed_lane_ids = self._terminal_strategy_evidence.get(gap.strategy_id)
        if completed_lane_ids is None:
            raise ValueError(
                f"Evidence strategy '{gap.strategy_id}' is not terminal and successful"
            )
        if gap.strategy_id in self._consumed_evidence_strategy_ids:
            raise ValueError(
                f"Evidence strategy '{gap.strategy_id}' already authorized an expansion"
            )
        referenced_lane_ids = set(gap.lane_ids)
        if not referenced_lane_ids <= completed_lane_ids:
            unknown = ", ".join(sorted(referenced_lane_ids - completed_lane_ids))
            raise ValueError(
                "Evidence gap references lane(s) without successful terminal "
                f"evidence in strategy '{gap.strategy_id}': {unknown}"
            )
        return gap.strategy_id

    def _matches_failed_recovery(
        self, decision: OrchestrationDecision, *, submitted_risk: WorkRisk
    ) -> bool:
        if self._unbound_terminal_failures or not self._failed_lane_ids:
            return False
        candidate_lane_ids = {lane.id for lane in decision.lanes}
        candidate_agent_lane_ids = {
            lane.id for lane in decision.lanes if lane.owner is LaneOwner.AGENT
        }
        if (
            candidate_lane_ids != self._failed_lane_ids
            or candidate_agent_lane_ids != self._failed_lane_ids
            or set(self._failed_recovery_identities) != self._failed_lane_groups
        ):
            return False

        candidate = self._recovery_identity(decision, self._failed_lane_ids)
        expected_decisions = {
            identity.decision for identity in self._failed_recovery_identities.values()
        }
        if expected_decisions != {candidate.decision} or any(
            identity.risk is not submitted_risk for identity in expected_decisions
        ):
            return False
        expected_lanes = {
            lane
            for identity in self._failed_recovery_identities.values()
            for lane in identity.lanes
        }
        return expected_lanes == set(candidate.lanes)

    def _candidate_external_dependencies(
        self,
        decision: OrchestrationDecision,
        *,
        previous_state: OrchestrationState,
        previous_decision: OrchestrationDecision | None,
    ) -> set[str]:
        lane_ids = {lane.id for lane in decision.lanes}
        if (
            previous_decision is not None
            and decision.route is previous_decision.route
            and lane_ids == {lane.id for lane in previous_decision.lanes}
        ):
            return set(self._active_external_dependencies)
        if (
            previous_state is not OrchestrationState.RECOVERY
            or decision.evidence_gap is not None
        ):
            return set()
        agent_lane_ids = {
            lane.id for lane in decision.lanes if lane.owner is LaneOwner.AGENT
        }
        if lane_ids != self._failed_lane_ids or agent_lane_ids != self._failed_lane_ids:
            return set()
        dependencies = {
            dependency
            for identity in self._failed_recovery_identities.values()
            for lane in identity.lanes
            for dependency in lane.dependencies
        }
        prior_host_lane_ids = (
            {lane.id for lane in self.decision.lanes if lane.owner is LaneOwner.HOST}
            if self.decision is not None
            else set()
        )
        satisfied_dependencies = self._successful_lane_ids | prior_host_lane_ids
        return (dependencies - self._failed_lane_ids) & satisfied_dependencies

    @staticmethod
    def _recovery_identity(
        decision: OrchestrationDecision, lane_ids: set[str] | frozenset[str]
    ) -> _RecoveryIdentity:
        decision_identity = _RecoveryDecisionIdentity(
            route=decision.route,
            objective=decision.objective.strip(),
            risk=decision.risk,
            expected_paths=tuple(sorted(set(decision.expected_paths))),
        )
        lanes = tuple(
            sorted(
                (
                    _RecoveryLaneIdentity(
                        id=lane.id,
                        agent_profile=lane.profile,
                        objective=lane.objective.strip(),
                        dependencies=tuple(sorted(set(lane.dependencies))),
                        acceptance=tuple(
                            sorted({item.strip() for item in lane.acceptance})
                        ),
                        expected_paths=tuple(sorted(set(lane.expected_paths))),
                    )
                    for lane in decision.lanes
                    if lane.id in lane_ids
                ),
                key=lambda lane: lane.id,
            )
        )
        return _RecoveryIdentity(decision=decision_identity, lanes=lanes)

    def _rejected_strategy_receipt(
        self,
        decision: OrchestrationDecision,
        error: ValueError,
        *,
        previous_state: OrchestrationState,
        previous_decision: OrchestrationDecision | None,
    ) -> StrategyReceipt:
        if previous_decision is None:
            self.state = OrchestrationState.ROUTE_REQUIRED
            message = str(error)
            required_delegations = 0
            receipt_route = decision.route
            strategy_id = None
        else:
            receipt_route = previous_decision.route
            strategy_id = self._current_strategy_id
            required_delegations = self._required_delegations
            previous_at_floor = previous_decision.model_copy(
                update={"risk": self._risk_floor}
            )
            try:
                if self._route_revalidation_required:
                    raise ValueError("the retained route already requires revalidation")
                self._validate_decision(
                    previous_at_floor,
                    allowed_external_dependencies=self._active_external_dependencies,
                )
            except ValueError as previous_error:
                self.state = OrchestrationState.ROUTE_REQUIRED
                self._route_revalidation_required = True
                message = (
                    f"{error}. Existing {previous_decision.route} strategy no longer "
                    "satisfies the current risk, intent, or capability constraints: "
                    f"{previous_error}. Record a new valid strategy before mutation."
                )
            else:
                self.decision = previous_at_floor
                self.state = previous_state
                message = (
                    f"{error}. Existing {previous_decision.route} strategy remains "
                    f"active in state '{previous_state}'."
                )
        return StrategyReceipt(
            route=receipt_route,
            state=self.state,
            message=message,
            accepted=False,
            reason=self._rejection_reason(decision),
            required_delegations=required_delegations,
            strategy_id=strategy_id,
        )

    def _apply_risk_floor(
        self, decision: OrchestrationDecision, *, latch: bool
    ) -> OrchestrationDecision:
        submitted_rank = _WORK_RISK_RANK[decision.risk]
        floor_rank = _WORK_RISK_RANK[self._risk_floor]
        if submitted_rank > floor_rank:
            if latch:
                self._risk_floor = decision.risk
            return decision
        if submitted_rank == floor_rank:
            return decision
        return decision.model_copy(update={"risk": self._risk_floor})

    def _latch_accepted_risk(self, risk: WorkRisk) -> None:
        if _WORK_RISK_RANK[risk] > _WORK_RISK_RANK[self._risk_floor]:
            self._risk_floor = risk

    def before_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        read_only: bool,
        call_id: str = "",
    ) -> str | None:
        if (
            self.state is OrchestrationState.OFF
            or tool_name in _CONTROL_TOOLS
            or (self._route_revalidation_required and read_only)
        ):
            return None
        if self._route_revalidation_required:
            return (
                "The retained strategy no longer satisfies current risk, intent, "
                "or capability constraints. Record a new valid work_strategy "
                "before another effectful tool."
            )

        route = _DELEGATION_TO_ROUTE.get(tool_name)
        if route is not None:
            return self._before_delegation(tool_name, args, route, call_id=call_id)

        if not read_only:
            if path_error := self._workspace_path_error(args):
                return path_error
            return self._before_mutation(tool_name, args)
        return None

    def _workspace_path_error(self, args: dict[str, Any]) -> str | None:
        raw_path = self._raw_tool_path(args)
        if raw_path is None or self._tool_path(args) is not None:
            return None
        self._scope_drift = True
        self.state = OrchestrationState.ROUTE_REQUIRED
        return (
            f"Path '{raw_path}' escapes the workspace. Record a strategy "
            "scoped to workspace-relative paths before mutating it."
        )

    def _before_delegation(
        self,
        tool_name: str,
        args: dict[str, Any],
        route: OrchestrationRoute,
        *,
        call_id: str,
    ) -> str | None:
        if constraint := self._delegation_constraint_error(route):
            self._reserved_lanes_by_call.pop(call_id, None)
            return constraint
        verifier = tool_name == "task" and args.get("agent") == "verifier"
        if self.decision is None and not verifier:
            self.state = OrchestrationState.ROUTE_REQUIRED
            self._reserved_lanes_by_call.pop(call_id, None)
            return (
                "Record work_strategy before launching a productive delegation so "
                "the harness can bind its route and lane identity."
            )
        if self.state in {
            OrchestrationState.PROVISIONAL_LOCAL,
            OrchestrationState.ROUTE_REQUIRED,
        } and (self._requires_strategy or self._explicit_delegation):
            self.state = OrchestrationState.ROUTE_REQUIRED
            self._reserved_lanes_by_call.pop(call_id, None)
            return (
                "Record work_strategy before launching substantive or explicitly "
                "requested orchestration so the harness can bind its lanes."
            )
        error = self._planned_delegation_error(tool_name, args, route, call_id=call_id)
        if error is not None:
            self._reserved_lanes_by_call.pop(call_id, None)
            return error
        if (
            call_id
            and self.decision is not None
            and self.decision.route is route
            and not (tool_name == "task" and args.get("agent") == "verifier")
        ):
            self._reserved_lanes_by_call[call_id] = self._bound_lane_ids(
                tool_name, args
            )
        return None

    def _delegation_constraint_error(self, route: OrchestrationRoute) -> str | None:
        if not self.user_allows_agents:
            return (
                "The user explicitly constrained agent use. Record a direct "
                "work_strategy with reason='user_constrained' instead."
            )
        if route is OrchestrationRoute.WORKFLOW and not self._user_allows_workflow:
            return (
                "The user explicitly prohibited workflows. Choose task, team, or "
                "a direct work_strategy consistent with that constraint."
            )
        if route is OrchestrationRoute.TEAM and not self._user_allows_team:
            return (
                "The user explicitly prohibited teams. Choose task, workflow, or "
                "a direct work_strategy consistent with that constraint."
            )
        route_available = {
            OrchestrationRoute.TASK: self.capabilities.task,
            OrchestrationRoute.WORKFLOW: self.capabilities.workflow,
            OrchestrationRoute.TEAM: self.capabilities.team,
        }.get(route, True)
        if not route_available:
            return (
                f"The '{route}' route is unavailable with terminal result delivery "
                "in this host. Choose an available route with work_strategy."
            )
        return None

    def _planned_delegation_error(
        self,
        tool_name: str,
        args: dict[str, Any],
        route: OrchestrationRoute,
        *,
        call_id: str,
    ) -> str | None:
        if self.state is OrchestrationState.DISTRIBUTED and self.decision is not None:
            if tool_name == "task" and args.get("agent") == "verifier":
                return None
            return (
                "Every declared productive lane has already launched. Reassess "
                "with work_strategy before adding another delegation lane."
            )
        if self.state not in {
            OrchestrationState.DELEGATION_PENDING,
            OrchestrationState.RECOVERY,
        }:
            return None
        planned = self.decision.route if self.decision is not None else None
        if planned is not route:
            return (
                f"The recorded strategy requires route '{planned}', not "
                f"'{route}'. Reassess with work_strategy before switching routes."
            )
        if tool_name == "task" and args.get("agent") == "verifier":
            return None
        return self._lane_binding_error(tool_name, args, call_id=call_id)

    def _before_mutation(self, tool_name: str, args: dict[str, Any]) -> str | None:
        if self.state in {
            OrchestrationState.PROVISIONAL_LOCAL,
            OrchestrationState.ROUTE_REQUIRED,
        }:
            if self._can_claim_implicit_direct(tool_name, args):
                self.state = OrchestrationState.DIRECT
                self._inferred_route = OrchestrationRoute.DIRECT
                self._implicit_direct_path = self._tool_path(args)
                return None
            self.state = OrchestrationState.ROUTE_REQUIRED
            return (
                "Le Chaton requires an adaptive work_strategy before the first "
                "substantive mutating tool. Choose direct, task, workflow, or team "
                "from observed scope; this does not remove the host's tools."
            )
        if self.state is OrchestrationState.DELEGATION_PENDING:
            return (
                "The declared delegation has not launched yet. Launch its productive "
                "task, workflow, or team lane before substantive host mutation."
            )
        if self.state is OrchestrationState.RECOVERY:
            return (
                "The planned delegation failed. Reassess with work_strategy or retry "
                "a viable orchestration route before continuing mutation."
            )
        if self.state is OrchestrationState.DIRECT:
            drift = self._direct_scope_drift(args)
            if drift is not None:
                self.state = OrchestrationState.ROUTE_REQUIRED
                return drift
        return None

    def _can_claim_implicit_direct(self, tool_name: str, args: dict[str, Any]) -> bool:
        if self.state is not OrchestrationState.PROVISIONAL_LOCAL:
            return False
        if (
            self._risk_floor is WorkRisk.HIGH
            or self._requires_strategy
            or self._explicit_delegation
        ):
            return False
        if (
            self._mutation_calls
            or self._reconnaissance_calls > _IMPLICIT_DIRECT_RECON_LIMIT
        ):
            return False
        path = self._tool_path(args)
        if tool_name not in {"edit", "write_file"} or path is None:
            return False
        normalized = path.lower().replace("\\", "/")
        basename = normalized.rsplit("/", maxsplit=1)[-1]
        return normalized in self._user_prompt or basename in self._user_prompt

    def record_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        status: Literal["success", "failure", "skipped"],
        result: dict[str, Any] | None = None,
        *,
        read_only: bool | None = None,
        call_id: str = "",
    ) -> None:
        if self.state is OrchestrationState.OFF:
            return

        route = _DELEGATION_TO_ROUTE.get(tool_name)
        if route is not None:
            self._reserved_lanes_by_call.pop(call_id, None)
            workflow_expectations = self._reserved_workflow_expectations_by_call.pop(
                call_id, None
            )
            self._record_delegation(
                tool_name,
                args,
                route,
                status,
                result,
                workflow_expectations=workflow_expectations,
            )
            return
        if status != "success" or tool_name in _CONTROL_TOOLS:
            return
        if read_only:
            self._reconnaissance_calls += 1
            return

        self._mutation_calls += 1
        self._total_mutation_calls += 1
        if path := self._tool_path(args):
            self._mutation_paths.add(path)
            self._total_mutation_paths.add(path)
        if self.state is OrchestrationState.DIRECT:
            expected = self.decision.expected_paths if self.decision is not None else []
            outside_expected = bool(
                path
                and expected
                and not any(self._path_matches(path, item) for item in expected)
            )
            if (
                outside_expected
                or self._mutation_calls >= _DIRECT_MUTATION_LIMIT
                or len(self._mutation_paths) > _DIRECT_PATH_LIMIT
            ):
                self._scope_drift = True
                self.state = OrchestrationState.ROUTE_REQUIRED

    def release_reservation(self, call_id: str) -> None:
        self._reserved_lanes_by_call.pop(call_id, None)
        self._reserved_workflow_expectations_by_call.pop(call_id, None)

    def workflow_lane_expectations(
        self, call_id: str, script: str
    ) -> tuple[WorkflowLaneExpectation, ...] | None:
        if (
            self.decision is None
            or self.decision.route is not OrchestrationRoute.WORKFLOW
        ):
            return None
        reserved = self._reserved_lanes_by_call.get(call_id)
        lanes = {lane.id: lane for lane in self._agent_lanes()}
        if not reserved or reserved != set(lanes):
            return None
        calls = self._workflow_agent_calls(script)
        expected = tuple(
            WorkflowLaneExpectation(
                label=lane_id, profile=lanes[lane_id].profile or calls.get(lane_id)
            )
            for lane_id in sorted(reserved)
        )
        if any(lane.profile is None for lane in expected):
            return None
        self._reserved_workflow_expectations_by_call[call_id] = expected
        return expected

    def record_task_completion(self, task_id: str, *, succeeded: bool) -> None:
        launch = self._task_lanes_by_id.pop(task_id, None)
        if launch is None:
            if self.state is not OrchestrationState.OFF:
                self._deferred_task_results[task_id] = succeeded
            return
        generation, lane_ids = launch
        if self.state is OrchestrationState.OFF:
            return
        self._register_terminal_delivery(
            OrchestrationRoute.TASK, task_id, generation, lane_ids
        )
        self._finish_lanes(lane_ids, succeeded=succeeded, launch_generation=generation)

    def record_workflow_completion(
        self,
        run_id: str,
        *,
        succeeded: bool,
        attestation: WorkflowLaneAttestation | None,
    ) -> None:
        launch = self._workflow_lanes_by_id.pop(run_id, None)
        if launch is None:
            if self.state is not OrchestrationState.OFF:
                self._deferred_workflow_results[run_id] = (succeeded, attestation)
            return
        generation, lane_ids = launch
        expected = self._workflow_expectations_by_id.pop(run_id, None)
        if self.state is OrchestrationState.OFF:
            return
        self._register_terminal_delivery(
            OrchestrationRoute.WORKFLOW, run_id, generation, lane_ids
        )
        attested = bool(
            succeeded
            and expected is not None
            and attestation is not None
            and attestation.satisfies(expected)
        )
        self._finish_lanes(lane_ids, succeeded=attested, launch_generation=generation)

    def record_team_completion(self, launch_id: str, *, succeeded: bool) -> None:
        launch = self._team_lanes_by_id.pop(launch_id, None)
        if launch is None:
            if self.state is not OrchestrationState.OFF:
                self._deferred_team_results[launch_id] = succeeded
            return
        generation, lane_ids = launch
        if self.state is OrchestrationState.OFF:
            return
        self._register_terminal_delivery(
            OrchestrationRoute.TEAM, launch_id, generation, lane_ids
        )
        self._finish_lanes(lane_ids, succeeded=succeeded, launch_generation=generation)

    def _register_terminal_delivery(
        self,
        route: OrchestrationRoute,
        launch_id: str,
        generation: int,
        lane_ids: set[str],
    ) -> None:
        if generation != self._lifecycle_generation or not (
            lane_ids & self._launched_lane_ids
        ):
            return
        self._terminal_deliveries[(route, launch_id)] = generation

    def completion_nudge(self) -> str | None:
        if self.state is OrchestrationState.OFF or self._policy_nudges:
            return None

        message: str | None = None
        if self.state is OrchestrationState.PROVISIONAL_LOCAL and (
            self._requires_strategy
            or self._explicit_delegation
            or self._reconnaissance_calls >= _RECON_NUDGE_THRESHOLD
        ):
            self.state = OrchestrationState.ROUTE_REQUIRED
            message = (
                "Before finishing this substantive Le Chaton turn, record an "
                "adaptive work_strategy from the scope you observed. Direct work "
                "is valid when localized; independent lanes require delegation."
            )
        elif self.state is OrchestrationState.ROUTE_REQUIRED:
            message = (
                "The observed work exceeded the current local strategy. Reassess "
                "with work_strategy before claiming completion."
            )
        elif self.state is OrchestrationState.DELEGATION_PENDING:
            if self._waiting_for_lane_dependencies():
                message = (
                    f"{self._current_pending_delegations} prerequisite lane(s) are "
                    "still running. Report an in-progress handoff and wait for their "
                    "terminal results before launching dependent lanes."
                )
            else:
                remaining = max(
                    0, self._required_delegations - self._productive_delegations
                )
                route = (
                    self.decision.route if self.decision is not None else "delegation"
                )
                message = (
                    f"The declared {route} route still owes {remaining} productive "
                    f"{route} launch(es). Launch them or revise work_strategy before "
                    "claiming completion."
                )
        elif self.state is OrchestrationState.RECOVERY:
            message = (
                "A planned delegation failed. Retry with lower concurrency, choose "
                "another available route, or record an honest constrained fallback "
                "with work_strategy before finishing."
            )
        elif self._pending_delegations:
            message = (
                f"{self._pending_delegations} productive delegation lane(s) are "
                "launched but still running. Report an in-progress handoff, not "
                "completed work; terminal results will be delivered separately."
            )

        if message is not None:
            self._policy_nudges += 1
        return message

    def completion_blocker(self) -> str | None:
        if not self._policy_nudges:
            return None
        unresolved = {OrchestrationState.ROUTE_REQUIRED, OrchestrationState.RECOVERY}
        if self.state in unresolved or (
            self.state is OrchestrationState.DELEGATION_PENDING
            and not self._waiting_for_lane_dependencies()
        ):
            return (
                "Le Chaton cannot report completion because orchestration policy "
                f"remains unresolved in state '{self.state}'. This turn is ending as "
                "blocked, not successful; revise the strategy or delegation on the "
                "next turn."
            )
        if self._pending_delegations:
            return (
                "Le Chaton cannot report delegated work complete while "
                f"{self._pending_delegations} productive lane(s) are still "
                "running. This turn is ending with an in-progress handoff; "
                "terminal results will be delivered separately."
            )
        return None

    @property
    def summary(self) -> OrchestrationTurnSummary:
        decision = self.decision
        return OrchestrationTurnSummary(
            state=self.state,
            route=decision.route if decision is not None else self._inferred_route,
            reason=decision.reason if decision is not None else None,
            capabilities=self.capabilities,
            reconnaissance_calls=self._reconnaissance_calls,
            direct_mutations=self._total_mutation_calls,
            unique_paths=len(self._total_mutation_paths),
            productive_delegations=self._productive_delegations,
            completed_delegations=len(self._completed_lane_ids),
            pending_delegations=self._pending_delegations,
            verifier_delegations=self._verifier_delegations,
            required_delegations=self._required_delegations,
            failed_delegations=self._delegation_failures,
            scope_drift=self._scope_drift,
            policy_nudges=self._policy_nudges,
            user_allows_agents=self.user_allows_agents,
            user_allows_workflow=self._user_allows_workflow,
            user_allows_team=self._user_allows_team,
        )

    @property
    def has_open_debt(self) -> bool:
        return self._has_open_strategy_debt()

    @property
    def _pending_delegations(self) -> int:
        launches = (
            *self._task_lanes_by_id.values(),
            *self._workflow_lanes_by_id.values(),
            *self._team_lanes_by_id.values(),
        )
        return sum(
            len(lane_ids)
            for generation, lane_ids in launches
            if generation == self._lifecycle_generation
        )

    @property
    def _current_pending_delegations(self) -> int:
        return self._pending_delegations

    def _waiting_for_lane_dependencies(self) -> bool:
        if self.state is not OrchestrationState.DELEGATION_PENDING:
            return False
        if not self._current_pending_delegations:
            return False
        lanes = self._agent_lanes()
        agent_lane_ids = {lane.id for lane in lanes}
        unlaunched = [lane for lane in lanes if lane.id not in self._launched_lane_ids]
        if not unlaunched:
            return False
        return all(
            bool((set(lane.dependencies) & agent_lane_ids) - self._completed_lane_ids)
            for lane in unlaunched
        )

    def _has_open_strategy_debt(self) -> bool:
        if self.state in {
            OrchestrationState.ROUTE_REQUIRED,
            OrchestrationState.DELEGATION_PENDING,
            OrchestrationState.RECOVERY,
        }:
            return True
        return any(
            generation == self._lifecycle_generation
            for generation, _ in (
                *self._task_lanes_by_id.values(),
                *self._workflow_lanes_by_id.values(),
                *self._team_lanes_by_id.values(),
            )
        )

    def _has_active_strategy_launch(self) -> bool:
        return any(
            generation == self._lifecycle_generation
            for generation, _ in (
                *self._task_lanes_by_id.values(),
                *self._workflow_lanes_by_id.values(),
                *self._team_lanes_by_id.values(),
            )
        )

    def _validate_decision(
        self,
        decision: OrchestrationDecision,
        *,
        allowed_external_dependencies: set[str] | None = None,
    ) -> None:
        lane_ids = {lane.id for lane in decision.lanes}
        allowed_dependencies = allowed_external_dependencies or set()
        for lane in decision.lanes:
            unknown = set(lane.dependencies) - lane_ids - allowed_dependencies
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Lane '{lane.id}' has unknown dependencies: {names}")
        agent_lanes = [lane for lane in decision.lanes if lane.owner is LaneOwner.AGENT]
        if (
            not self.user_allows_agents
            and decision.route is not OrchestrationRoute.DIRECT
        ):
            raise ValueError("The user explicitly prohibited agent delegation")
        if not self.user_allows_agents and decision.reason not in {
            StrategyReason.USER_CONSTRAINED,
            StrategyReason.USER_FORBIDS_AGENTS,
        }:
            raise ValueError(
                "A direct strategy under an explicit no-agent constraint must use "
                "reason='user_constrained'"
            )
        if (
            decision.route is not OrchestrationRoute.DIRECT
            and len(agent_lanes) > HOST_AGENT_LANE_LIMIT
        ):
            raise ValueError(
                "A strategy may start at most two agent-owned evidence lanes; "
                "declare another bounded strategy only after returned evidence "
                "identifies a concrete gap"
            )

        match decision.route:
            case OrchestrationRoute.DIRECT:
                self._validate_direct_decision(decision, bool(agent_lanes))
            case OrchestrationRoute.TASK:
                self._validate_agent_route(
                    self.capabilities.task,
                    bool(agent_lanes),
                    unavailable="Task delegation is unavailable in this host",
                    missing="Task route requires an agent-owned lane",
                )
            case OrchestrationRoute.WORKFLOW:
                if not self._user_allows_workflow:
                    raise ValueError("The user explicitly prohibited workflows")
                if not self.capabilities.workflow:
                    raise ValueError(
                        "Workflow orchestration is unavailable in this host"
                    )
                if len(agent_lanes) < _WORKFLOW_MIN_AGENT_LANES:
                    raise ValueError(
                        "Workflow route requires at least two agent-owned lanes"
                    )
            case OrchestrationRoute.TEAM:
                if not self._user_allows_team:
                    raise ValueError("The user explicitly prohibited teams")
                self._validate_agent_route(
                    self.capabilities.team,
                    bool(agent_lanes),
                    unavailable="Team orchestration is unavailable in this host",
                    missing="Team route requires an agent-owned lane",
                )

    def _validate_direct_decision(
        self, decision: OrchestrationDecision, has_agent_lanes: bool
    ) -> None:
        if has_agent_lanes:
            raise ValueError("A direct strategy cannot assign agent-owned lanes")
        if self._explicit_delegation and self.user_allows_agents:
            raise ValueError(
                "The user explicitly requested orchestration; direct is not a valid route"
            )
        if decision.reason in {
            StrategyReason.INDEPENDENT_LANES,
            StrategyReason.ADVERSARIAL_REVIEW,
            StrategyReason.LONG_RUNNING,
        }:
            raise ValueError(
                f"Reason '{decision.reason}' requires an orchestration route"
            )
        user_reasons = {
            StrategyReason.USER_CONSTRAINED,
            StrategyReason.USER_FORBIDS_AGENTS,
        }
        if decision.reason in user_reasons and self.user_allows_agents:
            raise ValueError("No user constraint prevents agent delegation")
        capability_reasons = {
            StrategyReason.CAPABILITY_UNAVAILABLE,
            StrategyReason.CAPABILITY_FALLBACK,
        }
        if decision.reason in capability_reasons and any((
            self.capabilities.task,
            self.capabilities.workflow,
            self.capabilities.team,
        )):
            raise ValueError("An orchestration capability is available to this host")
        constrained = {*user_reasons, *capability_reasons}
        if decision.risk is WorkRisk.HIGH and decision.reason not in constrained:
            raise ValueError(
                "High-risk direct work requires a user or capability constraint"
            )

    @staticmethod
    def _validate_agent_route(
        available: bool, has_agent_lanes: bool, *, unavailable: str, missing: str
    ) -> None:
        if not available:
            raise ValueError(unavailable)
        if not has_agent_lanes:
            raise ValueError(missing)

    def _record_delegation(
        self,
        tool_name: str,
        args: dict[str, Any],
        route: OrchestrationRoute,
        status: Literal["success", "failure", "skipped"],
        result: dict[str, Any] | None,
        *,
        workflow_expectations: tuple[WorkflowLaneExpectation, ...] | None = None,
    ) -> None:
        verifier = tool_name == "task" and args.get("agent") == "verifier"
        if verifier:
            if status == "success":
                self._verifier_delegations += 1
            return
        if status != "success":
            if self.state in {
                OrchestrationState.DELEGATION_PENDING,
                OrchestrationState.RECOVERY,
            }:
                self._delegation_failures += 1
                lane_ids = (
                    self._bound_lane_ids(tool_name, args)
                    if self.decision is not None and self.decision.route is route
                    else set()
                )
                self._register_failed_lanes(lane_ids)
                self.state = OrchestrationState.RECOVERY
            return

        if self.decision is None:
            self._delegation_failures += 1
            self._unbound_terminal_failures += 1
            self.state = OrchestrationState.RECOVERY
            self._sync_terminal_failures()
            return
        if self.decision.route is not route:
            return
        lane_ids = self._bound_lane_ids(tool_name, args)
        if not lane_ids:
            return

        self._launched_lane_ids.update(lane_ids)
        self._update_delegation_state()
        if tool_name == "task":
            self._record_task_launch(lane_ids, result)
        elif tool_name == "launch_workflow":
            self._record_workflow_launch(
                lane_ids, result, expectations=workflow_expectations
            )
        elif tool_name == "team_spawn":
            self._record_team_launch(lane_ids, result)

    def _record_task_launch(
        self, lane_ids: set[str], result: dict[str, Any] | None
    ) -> None:
        lane_id = next(iter(lane_ids))
        task_id = result.get("task_id") if result is not None else None
        if isinstance(task_id, str) and task_id:
            self._task_lanes_by_id[task_id] = (self._lifecycle_generation, {lane_id})
            if task_id in self._deferred_task_results:
                succeeded = self._deferred_task_results.pop(task_id)
                self.record_task_completion(task_id, succeeded=succeeded)
            return
        succeeded = self._terminal_task_succeeded(result)
        self._finish_lanes(lane_ids, succeeded=succeeded)

    def _record_workflow_launch(
        self,
        lane_ids: set[str],
        result: dict[str, Any] | None,
        *,
        expectations: tuple[WorkflowLaneExpectation, ...] | None,
    ) -> None:
        if result is None:
            self._finish_lanes(lane_ids, succeeded=False)
            return
        run_id = result.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            self._finish_lanes(lane_ids, succeeded=False)
            return
        self._workflow_lanes_by_id[run_id] = (self._lifecycle_generation, set(lane_ids))
        if expectations is not None:
            self._workflow_expectations_by_id[run_id] = expectations
        if run_id in self._deferred_workflow_results:
            succeeded, attestation = self._deferred_workflow_results.pop(run_id)
            self.record_workflow_completion(
                run_id, succeeded=succeeded, attestation=attestation
            )

    def _record_team_launch(
        self, lane_ids: set[str], result: dict[str, Any] | None
    ) -> None:
        launch_id = result.get("launch_id") if result is not None else None
        if not isinstance(launch_id, str) or not launch_id:
            self._finish_lanes(lane_ids, succeeded=False)
            return
        self._team_lanes_by_id[launch_id] = (self._lifecycle_generation, set(lane_ids))
        if launch_id in self._deferred_team_results:
            succeeded = self._deferred_team_results.pop(launch_id)
            self.record_team_completion(launch_id, succeeded=succeeded)

    @staticmethod
    def _terminal_task_succeeded(result: dict[str, Any] | None) -> bool:
        if result is None:
            return True
        if not bool(result.get("completed", False)):
            return False
        outcome = result.get("outcome")
        if not isinstance(outcome, dict):
            return True
        status = outcome.get("status")
        return getattr(status, "value", status) == "succeeded"

    def _finish_lanes(
        self,
        lane_ids: set[str],
        *,
        succeeded: bool,
        launch_generation: int | None = None,
    ) -> None:
        if (
            launch_generation is not None
            and launch_generation != self._lifecycle_generation
        ):
            return
        relevant = lane_ids & self._launched_lane_ids
        if not relevant:
            return
        if succeeded:
            resolved_groups = {
                group for group in self._failed_lane_groups if group <= relevant
            }
            self._failed_lane_groups.difference_update(resolved_groups)
            for group in resolved_groups:
                self._failed_recovery_identities.pop(group, None)
            self._sync_terminal_failures()
            self._completed_lane_ids.update(relevant)
            self._successful_lane_ids.update(relevant)
            self._record_terminal_strategy_evidence()
            self._update_delegation_state()
            return
        self._launched_lane_ids.difference_update(relevant)
        self._completed_lane_ids.difference_update(relevant)
        self._delegation_failures += 1
        self._register_failed_lanes(relevant)
        self._update_productive_count()
        self.state = OrchestrationState.RECOVERY

    def _update_delegation_state(self) -> None:
        self._update_productive_count()
        if self._route_revalidation_required:
            self.state = OrchestrationState.ROUTE_REQUIRED
            return
        if self._unresolved_terminal_failures:
            self.state = OrchestrationState.RECOVERY
            return
        required = {lane.id for lane in self._agent_lanes()}
        if required:
            self.state = (
                OrchestrationState.DISTRIBUTED
                if required <= self._launched_lane_ids
                else OrchestrationState.DELEGATION_PENDING
            )
            return
        self.state = (
            OrchestrationState.DISTRIBUTED
            if self._productive_delegations >= self._required_delegations
            else OrchestrationState.DELEGATION_PENDING
        )

    def _record_terminal_strategy_evidence(self) -> None:
        strategy_id = self._current_strategy_id
        if strategy_id is None or self.decision is None:
            return
        required = {lane.id for lane in self._agent_lanes()}
        if required and required <= self._completed_lane_ids:
            self._terminal_strategy_evidence[strategy_id] = frozenset(required)

    def _register_failed_lanes(self, lane_ids: set[str]) -> None:
        if lane_ids:
            group = frozenset(lane_ids)
            self._failed_lane_groups.add(group)
            if self.decision is not None:
                decision_lane_ids = {
                    lane.id
                    for lane in self.decision.lanes
                    if lane.owner is LaneOwner.AGENT
                }
                if group <= decision_lane_ids:
                    self._failed_recovery_identities[group] = self._recovery_identity(
                        self.decision, group
                    )
        else:
            self._unbound_terminal_failures += 1
        self._sync_terminal_failures()

    def _sync_terminal_failures(self) -> None:
        stale_groups = set(self._failed_recovery_identities) - self._failed_lane_groups
        for group in stale_groups:
            self._failed_recovery_identities.pop(group, None)
        self._failed_lane_ids = set().union(*self._failed_lane_groups)
        self._unresolved_terminal_failures = (
            len(self._failed_lane_groups) + self._unbound_terminal_failures
        )

    def _update_productive_count(self) -> None:
        self._productive_delegations = len(self._launched_lane_ids)

    def _lane_binding_error(
        self, tool_name: str, args: dict[str, Any], *, call_id: str
    ) -> str | None:
        lanes = self._agent_lanes()
        reserved_elsewhere = set().union(
            *(
                lane_ids
                for reservation_id, lane_ids in self._reserved_lanes_by_call.items()
                if reservation_id != call_id
            )
        )
        pending = [
            lane
            for lane in lanes
            if lane.id not in self._launched_lane_ids
            and lane.id not in reserved_elsewhere
        ]
        bound = self._bound_lane_ids(tool_name, args)
        if duplicate := bound & reserved_elsewhere:
            names = ", ".join(sorted(duplicate))
            return f"Strategy lane(s) already reserved by another tool call: {names}"
        if tool_name == "launch_workflow":
            return self._workflow_lane_binding_error(
                lanes, pending, bound, str(args.get("script", ""))
            )
        return self._single_lane_binding_error(tool_name, args, lanes, pending, bound)

    def _workflow_lane_binding_error(
        self,
        lanes: list[OrchestrationLane],
        pending: list[OrchestrationLane],
        bound: set[str],
        script: str,
    ) -> str | None:
        missing = [lane for lane in pending if lane.id not in bound]
        if not missing and len(bound) == len(lanes):
            return self._workflow_dependency_error(script, lanes)
        labels = ", ".join(f"label='{lane.id}'" for lane in missing or pending)
        return (
            "Bind every declared workflow lane to one literal agent() label: "
            f"{labels}. Dynamic labels and comments do not satisfy lane debt."
        )

    def _single_lane_binding_error(
        self,
        tool_name: str,
        args: dict[str, Any],
        lanes: list[OrchestrationLane],
        pending: list[OrchestrationLane],
        bound: set[str],
    ) -> str | None:
        if len(bound) != 1:
            markers = ", ".join(f"[lane:{lane.id}]" for lane in pending)
            return (
                "Bind this delegation to exactly one pending strategy lane by "
                f"including its marker in the task/prompt: {markers}."
            )
        lane_id = next(iter(bound))
        if lane_id in self._launched_lane_ids:
            return f"Strategy lane '{lane_id}' has already launched"
        lane = next(lane for lane in lanes if lane.id == lane_id)
        if lane.profile is not None and args.get("agent") != lane.profile:
            return (
                f"Strategy lane '{lane.id}' requires agent profile "
                f"'{lane.profile}', not '{args.get('agent')}'."
            )
        agent_lane_ids = {item.id for item in lanes}
        unmet = (set(lane.dependencies) & agent_lane_ids) - self._completed_lane_ids
        if unmet:
            names = ", ".join(sorted(unmet))
            return (
                f"Strategy lane '{lane.id}' depends on incomplete lane(s): "
                f"{names}. Wait for their terminal results before launching it."
            )
        return None

    @classmethod
    def _workflow_dependency_error(
        cls, script: str, lanes: list[OrchestrationLane]
    ) -> str | None:
        entrypoint = cls._workflow_entrypoint(script)
        if isinstance(entrypoint, str):
            return entrypoint
        tree, main = entrypoint

        calls_result = cls._workflow_declared_agent_calls(tree, lanes)
        if isinstance(calls_result, str):
            return calls_result
        calls = calls_result

        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        pipeline_result = cls._workflow_pipeline_positions(tree, main, calls, parents)
        if isinstance(pipeline_result, str):
            return pipeline_result
        pipeline_positions = pipeline_result
        parallel_positions = cls._workflow_parallel_positions(main, calls, parents)
        execution_anchors: dict[str, ast.Call] = {}
        for lane in lanes:
            [agent_call] = calls[lane.id]
            if (
                cls._workflow_main_sequence_position(agent_call, main, parents)
                is not None
            ):
                execution_anchors[lane.id] = agent_call
                continue
            if parallel_call := parallel_positions.get(lane.id):
                execution_anchors[lane.id] = parallel_call
                continue
            if pipeline := pipeline_positions.get(lane.id):
                execution_anchors[lane.id] = pipeline[0]
                continue
            return (
                f"Workflow lane '{lane.id}' is not directly executable, so the "
                "script cannot prove that the declared lane will run. "
                "Use a top-level awaited agent() statement, a direct agent() "
                "coroutine or lambda in a top-level awaited parallel(), or a "
                "canonical named or lambda stage in a top-level awaited pipeline()."
            )

        dependencies = {
            lane.id: set(lane.dependencies) for lane in lanes if lane.dependencies
        }
        agent_lane_ids = {lane.id for lane in lanes}
        for lane_id, lane_dependencies in dependencies.items():
            for dependency in lane_dependencies & agent_lane_ids:
                if cls._workflow_labels_are_ordered(
                    dependency,
                    lane_id,
                    execution_anchors,
                    pipeline_positions,
                    main,
                    parents,
                ):
                    continue
                return (
                    f"Workflow lane '{lane_id}' depends on '{dependency}', but the "
                    "script does not establish that order with distinct awaited "
                    "main() statements or ordered stages of one awaited pipeline()."
                )
        return None

    @staticmethod
    def _workflow_declared_agent_calls(
        tree: ast.Module, lanes: list[OrchestrationLane]
    ) -> dict[str, list[ast.Call]] | str:
        calls: dict[str, list[ast.Call]] = {}
        direct_agent_names: set[ast.Name] = set()
        unlabeled_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "agent":
                continue
            direct_agent_names.add(node.func)
            keywords = {item.arg: item.value for item in node.keywords if item.arg}
            label = keywords.get("label")
            if isinstance(label, ast.Constant) and isinstance(label.value, str):
                calls.setdefault(label.value, []).append(node)
            else:
                unlabeled_calls += 1
        indirect_reference = any(
            isinstance(node, ast.Name)
            and node.id == "agent"
            and isinstance(node.ctx, ast.Load)
            and node not in direct_agent_names
            for node in ast.walk(tree)
        )
        if indirect_reference:
            return (
                "Workflow scripts cannot carry or invoke agent indirectly; use "
                "literal agent(...) calls bound to declared lane labels."
            )
        if unlabeled_calls:
            return (
                "Every workflow agent() call must use one declared literal lane "
                "label; dynamic or unlabeled calls can bypass the strategy limit."
            )
        lane_ids = {lane.id for lane in lanes}
        if unexpected := sorted(set(calls) - lane_ids):
            names = ", ".join(unexpected)
            return (
                f"Workflow agent() label(s) are not declared strategy lanes: {names}."
            )
        if missing := sorted(lane_ids - set(calls)):
            names = ", ".join(missing)
            return f"Workflow strategy lane(s) have no direct agent() call: {names}."
        duplicate = next(
            (label for label, items in calls.items() if len(items) != 1), None
        )
        if duplicate is not None:
            return f"Workflow lane label '{duplicate}' must identify exactly one agent() call."
        return calls

    @classmethod
    def _workflow_entrypoint(
        cls, script: str
    ) -> tuple[ast.Module, ast.AsyncFunctionDef] | str:
        try:
            tree = ast.parse(script)
        except SyntaxError:
            return "Workflow lane bindings require a valid workflow script."

        main_nodes = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == "main"
        ]
        if (
            len(main_nodes) != 1
            or not isinstance(main_nodes[0], ast.AsyncFunctionDef)
            or main_nodes[0].decorator_list
        ):
            return (
                "Workflow lane bindings require exactly one undecorated top-level "
                "async main() with no reassignment."
            )
        main = main_nodes[0]
        if rebound := cls._workflow_reserved_binding(tree, main):
            if rebound == "main":
                return (
                    "Workflow lane bindings require exactly one undecorated top-level "
                    "async main() with no reassignment."
                )
            return (
                "Workflow scripts cannot bind reserved orchestration helper "
                f"'{rebound}'."
            )
        return tree, main

    @staticmethod
    def _workflow_reserved_binding(
        tree: ast.Module, main: ast.AsyncFunctionDef
    ) -> str | None:
        reserved = _WORKFLOW_RESERVED_HELPERS | {"main"}
        for node in ast.walk(tree):
            if node is main:
                continue
            if (
                matches := OrchestrationController._workflow_bound_names(node)
                & reserved
            ):
                return min(matches)
        return None

    @staticmethod
    def _workflow_bound_names(node: ast.AST) -> set[str]:
        names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            names.add(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        return names

    @classmethod
    def _workflow_parallel_positions(
        cls,
        main: ast.AsyncFunctionDef,
        calls: dict[str, list[ast.Call]],
        parents: dict[ast.AST, ast.AST],
    ) -> dict[str, ast.Call]:
        positions: dict[str, ast.Call] = {}
        for node in ast.walk(main):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "parallel":
                continue
            if cls._workflow_main_sequence_position(node, main, parents) is None:
                continue
            for label, [agent_call] in calls.items():
                if any(
                    argument is agent_call
                    or (
                        isinstance(argument, ast.Lambda) and argument.body is agent_call
                    )
                    for argument in node.args
                ):
                    positions[label] = node
        return positions

    @classmethod
    def _workflow_pipeline_positions(
        cls,
        tree: ast.Module,
        main: ast.AsyncFunctionDef,
        calls: dict[str, list[ast.Call]],
        parents: dict[ast.AST, ast.AST],
    ) -> dict[str, tuple[ast.Call, int]] | str:
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        positions: dict[str, tuple[ast.Call, int]] = {}
        for node in ast.walk(main):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "pipeline":
                continue
            if cls._workflow_main_sequence_position(node, main, parents) is None:
                continue
            pipeline_positions: dict[str, tuple[ast.Call, int]] = {}
            named_stages: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
            for index, stage in enumerate(node.args[1:]):
                target: ast.AST = stage
                if isinstance(stage, ast.Name) and stage.id in functions:
                    function = functions[stage.id]
                    target = function
                matched_stage = False
                for label, [agent_call] in calls.items():
                    if cls._workflow_stage_runs_agent(target, agent_call, parents):
                        pipeline_positions[label] = (node, index)
                        matched_stage = True
                if (
                    matched_stage
                    and isinstance(stage, ast.Name)
                    and stage.id in functions
                ):
                    named_stages.append((stage.id, functions[stage.id]))
            if not pipeline_positions:
                continue
            if not node.args or not cls._workflow_pipeline_seed_is_singleton(
                node.args[0]
            ):
                return (
                    "Workflow pipeline lanes require a statically provable "
                    "singleton seed so each declared lane executes once."
                )
            for name, function in named_stages:
                if cls._workflow_named_stage_is_stable(tree, name, function):
                    continue
                return (
                    f"Workflow pipeline stage '{name}' must have one stable module "
                    "binding. Rebinding, deletion, and global or nonlocal mutation "
                    "paths are not allowed."
                )
            positions.update(pipeline_positions)
        return positions

    @classmethod
    def _workflow_named_stage_is_stable(
        cls,
        tree: ast.Module,
        name: str,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        bindings = [
            node for node in ast.walk(tree) if cls._workflow_node_binds_name(node, name)
        ]
        return bindings == [function]

    @staticmethod
    def _workflow_node_binds_name(node: ast.AST, name: str) -> bool:
        return name in OrchestrationController._workflow_bound_names(node)

    @classmethod
    def _workflow_pipeline_seed_is_singleton(cls, node: ast.AST) -> bool:
        value = cls._workflow_static_value(node)
        if value is _WORKFLOW_UNKNOWN:
            return False
        supported = (list, tuple, set, frozenset, dict, str, bytes, bytearray, range)
        return isinstance(value, supported) and len(value) == 1

    @classmethod
    def _workflow_static_value(cls, node: ast.AST) -> Any:
        value: Any = _WORKFLOW_UNKNOWN
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            value = cls._workflow_static_collection(node)
        elif isinstance(node, ast.Dict):
            value = cls._workflow_static_dict(node)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            truth = cls._workflow_static_truth(node.operand)
            value = _WORKFLOW_UNKNOWN if truth is None else not truth
        elif isinstance(node, ast.Call):
            value = cls._workflow_static_call(node)
        elif isinstance(node, ast.Compare):
            value = cls._workflow_static_compare(node)
        return value

    @classmethod
    def _workflow_static_collection(cls, node: ast.List | ast.Tuple | ast.Set) -> Any:
        values = cls._workflow_static_items(node.elts)
        if not isinstance(values, list):
            return _WORKFLOW_UNKNOWN
        try:
            if isinstance(node, ast.List):
                return values
            if isinstance(node, ast.Tuple):
                return tuple(values)
            return set(values)
        except TypeError:
            return _WORKFLOW_UNKNOWN

    @classmethod
    def _workflow_static_items(cls, nodes: list[ast.expr]) -> list[Any] | object:
        values: list[Any] = []
        for node in nodes:
            value = cls._workflow_static_value(node)
            if value is _WORKFLOW_UNKNOWN:
                return _WORKFLOW_UNKNOWN
            values.append(value)
        return values

    @classmethod
    def _workflow_static_dict(cls, node: ast.Dict) -> dict[Any, Any] | object:
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                return _WORKFLOW_UNKNOWN
            key = cls._workflow_static_value(key_node)
            value = cls._workflow_static_value(value_node)
            if key is _WORKFLOW_UNKNOWN or value is _WORKFLOW_UNKNOWN:
                return _WORKFLOW_UNKNOWN
            try:
                result[key] = value
            except TypeError:
                return _WORKFLOW_UNKNOWN
        return result

    @classmethod
    def _workflow_static_call(cls, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.keywords:
            return _WORKFLOW_UNKNOWN
        empty_factories: dict[str, Any] = {
            "bytearray": bytearray(),
            "bytes": b"",
            "dict": {},
            "frozenset": frozenset(),
            "list": [],
            "set": set(),
            "str": "",
            "tuple": (),
        }
        if not node.args and node.func.id in empty_factories:
            return empty_factories[node.func.id]
        if node.func.id not in {"bool", "len"} or len(node.args) != 1:
            return _WORKFLOW_UNKNOWN
        value = cls._workflow_static_value(node.args[0])
        if value is _WORKFLOW_UNKNOWN:
            return _WORKFLOW_UNKNOWN
        try:
            return bool(value) if node.func.id == "bool" else len(value)
        except (TypeError, ValueError):
            return _WORKFLOW_UNKNOWN

    @classmethod
    def _workflow_static_compare(cls, node: ast.Compare) -> bool | object:
        left = cls._workflow_static_value(node.left)
        if left is _WORKFLOW_UNKNOWN:
            return _WORKFLOW_UNKNOWN
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = cls._workflow_static_value(comparator)
            if right is _WORKFLOW_UNKNOWN:
                return _WORKFLOW_UNKNOWN
            compared = cls._workflow_apply_compare(left, operator, right)
            if compared is _WORKFLOW_UNKNOWN:
                return _WORKFLOW_UNKNOWN
            if not compared:
                return False
            left = right
        return True

    @staticmethod
    def _workflow_apply_compare(
        left: Any, operator: ast.cmpop, right: Any
    ) -> bool | object:
        result: bool | object = _WORKFLOW_UNKNOWN
        try:
            if isinstance(operator, ast.Eq):
                result = left == right
            elif isinstance(operator, ast.NotEq):
                result = left != right
            elif isinstance(operator, ast.Lt):
                result = left < right
            elif isinstance(operator, ast.LtE):
                result = left <= right
            elif isinstance(operator, ast.Gt):
                result = left > right
            elif isinstance(operator, ast.GtE):
                result = left >= right
        except TypeError:
            result = _WORKFLOW_UNKNOWN
        return result

    @classmethod
    def _workflow_static_truth(cls, node: ast.AST) -> bool | None:
        value = cls._workflow_static_value(node)
        if value is _WORKFLOW_UNKNOWN:
            return None
        try:
            return bool(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _workflow_stage_runs_agent(
        cls, stage: ast.AST, agent_call: ast.Call, parents: dict[ast.AST, ast.AST]
    ) -> bool:
        if isinstance(stage, ast.Lambda):
            return stage.body is agent_call
        if isinstance(stage, ast.AsyncFunctionDef):
            if stage.decorator_list:
                return False
            return (
                cls._workflow_main_sequence_position(agent_call, stage, parents)
                is not None
            )
        if not isinstance(stage, ast.FunctionDef):
            return False
        if stage.decorator_list:
            return False
        return_node = parents.get(agent_call)
        return bool(
            isinstance(return_node, ast.Return)
            and return_node.value is agent_call
            and parents.get(return_node) is stage
            and return_node in stage.body
            and not cls._workflow_sequence_terminates(
                stage.body[: stage.body.index(return_node)]
            )
        )

    @classmethod
    def _workflow_labels_are_ordered(
        cls,
        dependency: str,
        lane_id: str,
        execution_anchors: dict[str, ast.Call],
        pipelines: dict[str, tuple[ast.Call, int]],
        main: ast.AsyncFunctionDef,
        parents: dict[ast.AST, ast.AST],
    ) -> bool:
        dependency_pipeline = pipelines.get(dependency)
        lane_pipeline = pipelines.get(lane_id)
        if dependency_pipeline is not None and lane_pipeline is not None:
            if dependency_pipeline[0] is lane_pipeline[0]:
                return dependency_pipeline[1] < lane_pipeline[1]

        dependency_node = execution_anchors[dependency]
        lane_node = execution_anchors[lane_id]
        if dependency_node is lane_node:
            return False
        dependency_position = cls._workflow_main_sequence_position(
            dependency_node, main, parents
        )
        lane_position = cls._workflow_main_sequence_position(lane_node, main, parents)
        return bool(
            dependency_position is not None
            and lane_position is not None
            and dependency_position < lane_position
        )

    @classmethod
    def _workflow_main_sequence_position(
        cls, node: ast.AST, main: ast.AsyncFunctionDef, parents: dict[ast.AST, ast.AST]
    ) -> int | None:
        await_node = parents.get(node)
        if not isinstance(await_node, ast.Await) or await_node.value is not node:
            return None
        statement = parents.get(await_node)
        if (
            not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return))
            or getattr(statement, "value", None) is not await_node
            or parents.get(statement) is not main
            or statement not in main.body
        ):
            return None
        position = main.body.index(statement)
        if cls._workflow_sequence_terminates(main.body[:position]):
            return None
        return position

    @classmethod
    def _workflow_sequence_terminates(cls, statements: list[ast.stmt]) -> bool:
        return any(
            cls._workflow_statement_terminates(statement) for statement in statements
        )

    @classmethod
    def _workflow_statement_terminates(cls, statement: ast.stmt) -> bool:
        terminates = False
        if isinstance(statement, (ast.Return, ast.Raise)):
            terminates = True
        elif isinstance(statement, ast.If):
            truth = cls._workflow_static_truth(statement.test)
            branches = (
                [statement.body]
                if truth is True
                else [statement.orelse]
                if truth is False
                else [statement.body, statement.orelse]
            )
            terminates = any(
                cls._workflow_sequence_terminates(branch) for branch in branches
            )
        elif isinstance(statement, ast.While):
            truth = cls._workflow_static_truth(statement.test)
            if truth is False:
                terminates = cls._workflow_sequence_terminates(statement.orelse)
            elif cls._workflow_sequence_terminates(statement.body):
                terminates = True
            elif truth is True and not cls._workflow_loop_has_break(statement):
                terminates = True
            elif truth is None:
                terminates = cls._workflow_sequence_terminates(statement.orelse)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            value = cls._workflow_static_value(statement.iter)
            if value is not _WORKFLOW_UNKNOWN and not bool(value):
                terminates = cls._workflow_sequence_terminates(statement.orelse)
            else:
                terminates = cls._workflow_sequence_terminates(
                    statement.body
                ) or cls._workflow_sequence_terminates(statement.orelse)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            sequences = [statement.body, statement.orelse, statement.finalbody]
            sequences.extend(handler.body for handler in statement.handlers)
            terminates = any(
                cls._workflow_sequence_terminates(items) for items in sequences
            )
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            terminates = cls._workflow_sequence_terminates(statement.body)
        elif isinstance(statement, ast.Match):
            terminates = any(
                (
                    case.guard is None
                    or cls._workflow_static_truth(case.guard) is not False
                )
                and cls._workflow_sequence_terminates(case.body)
                for case in statement.cases
            )
        return terminates

    @classmethod
    def _workflow_loop_has_break(cls, statement: ast.While) -> bool:
        return cls._workflow_statements_have_break(statement.body)

    @classmethod
    def _workflow_statements_have_break(cls, statements: list[ast.stmt]) -> bool:
        for statement in statements:
            if isinstance(statement, ast.Break):
                return True
            if isinstance(statement, ast.If):
                truth = cls._workflow_static_truth(statement.test)
                branches = (
                    [statement.body]
                    if truth is True
                    else [statement.orelse]
                    if truth is False
                    else [statement.body, statement.orelse]
                )
                if any(
                    cls._workflow_statements_have_break(items) for items in branches
                ):
                    return True
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                sequences = [statement.body, statement.orelse, statement.finalbody]
                sequences.extend(handler.body for handler in statement.handlers)
                if any(
                    cls._workflow_statements_have_break(items) for items in sequences
                ):
                    return True
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                if cls._workflow_statements_have_break(statement.body):
                    return True
            elif isinstance(statement, ast.Match):
                if any(
                    (
                        case.guard is None
                        or cls._workflow_static_truth(case.guard) is not False
                    )
                    and cls._workflow_statements_have_break(case.body)
                    for case in statement.cases
                ):
                    return True
        return False

    def _bound_lane_ids(self, tool_name: str, args: dict[str, Any]) -> set[str]:
        lanes = self._agent_lanes()
        if tool_name == "launch_workflow":
            calls = self._workflow_agent_calls(str(args.get("script", "")))
            return {
                lane.id
                for lane in lanes
                if calls.get(lane.id) == lane.profile
                or (lane.profile is None and lane.id in calls)
            }
        payload = args.get("task") if tool_name == "task" else args.get("prompt")
        text = (
            json.dumps(payload, sort_keys=True)
            if isinstance(payload, dict)
            else str(payload or "")
        )
        return {lane.id for lane in lanes if f"[lane:{lane.id}]" in text}

    def _agent_lanes(self) -> list[OrchestrationLane]:
        if self.decision is None:
            return []
        return [lane for lane in self.decision.lanes if lane.owner is LaneOwner.AGENT]

    @staticmethod
    def _workflow_agent_calls(script: str) -> dict[str, str]:
        try:
            tree = ast.parse(script)
        except SyntaxError:
            return {}
        calls: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "agent":
                continue
            keywords = {item.arg: item.value for item in node.keywords if item.arg}
            label_node = keywords.get("label")
            if not isinstance(label_node, ast.Constant) or not isinstance(
                label_node.value, str
            ):
                continue
            profile_node = keywords.get("agent")
            profile = "explore"
            if isinstance(profile_node, ast.Constant) and isinstance(
                profile_node.value, str
            ):
                profile = profile_node.value
            calls[label_node.value] = profile
        return calls

    def _direct_scope_drift(self, args: dict[str, Any]) -> str | None:
        if self._mutation_calls >= _DIRECT_MUTATION_LIMIT:
            self._scope_drift = True
            return (
                "The direct route reached its bounded mutation envelope. Reassess "
                "with work_strategy before another mutation."
            )
        path = self._tool_path(args)
        if path is None or path in self._mutation_paths:
            return None
        if (
            self._implicit_direct_path is not None
            and path != self._implicit_direct_path
        ):
            self._scope_drift = True
            return (
                f"Path '{path}' is outside the inferred direct-work scope "
                f"'{self._implicit_direct_path}'. Reassess with work_strategy "
                "before mutating it."
            )
        expected = self.decision.expected_paths if self.decision is not None else []
        if expected and not any(self._path_matches(path, item) for item in expected):
            self._scope_drift = True
            return (
                f"Path '{path}' is outside the declared direct-work scope. Reassess "
                "with work_strategy before mutating it."
            )
        if len(self._mutation_paths) >= _DIRECT_PATH_LIMIT:
            self._scope_drift = True
            return (
                "The direct route expanded beyond two mutation paths. Reassess "
                "whether independent lanes now justify task, workflow, or team."
            )
        return None

    def _fallback_decision(
        self, decision: OrchestrationDecision
    ) -> OrchestrationDecision | None:
        if (
            decision.route is OrchestrationRoute.WORKFLOW
            and not self.capabilities.workflow
            and self.capabilities.task
            and self.user_allows_agents
            and self._user_allows_workflow
        ):
            return decision.model_copy(
                update={
                    "route": OrchestrationRoute.TASK,
                    "reason": StrategyReason.CAPABILITY_FALLBACK,
                }
            )
        return None

    def _rejection_reason(self, decision: OrchestrationDecision) -> StrategyReason:
        if not self.user_allows_agents:
            return StrategyReason.USER_FORBIDS_AGENTS
        if (
            decision.route is OrchestrationRoute.WORKFLOW
            and not self._user_allows_workflow
        ) or (decision.route is OrchestrationRoute.TEAM and not self._user_allows_team):
            return StrategyReason.USER_CONSTRAINED
        if (
            decision.route is OrchestrationRoute.WORKFLOW
            and not self.capabilities.workflow
        ):
            return StrategyReason.CAPABILITY_UNAVAILABLE
        return decision.reason

    @staticmethod
    def _raw_tool_path(args: dict[str, Any]) -> str | None:
        for key in _PATH_KEYS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _tool_path(self, args: dict[str, Any]) -> str | None:
        raw_path = self._raw_tool_path(args)
        if raw_path is None:
            return None
        return self._workspace_relative_path(raw_path)

    def _workspace_relative_path(self, value: str) -> str | None:
        candidate = Path(value.replace("\\", "/")).expanduser()
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate
        try:
            return candidate.resolve().relative_to(self._workspace_root).as_posix()
        except (OSError, ValueError):
            return None

    def _canonicalize_expected_paths(
        self, decision: OrchestrationDecision
    ) -> OrchestrationDecision:
        expected_paths = self._canonical_path_list(decision.expected_paths)
        lanes = [
            lane.model_copy(
                update={
                    "expected_paths": self._canonical_path_list(lane.expected_paths)
                }
            )
            for lane in decision.lanes
        ]
        return decision.model_copy(
            update={"expected_paths": expected_paths, "lanes": lanes}
        )

    def _canonical_path_list(self, paths: list[str]) -> list[str]:
        expected_paths: list[str] = []
        for path in paths:
            normalized = self._workspace_relative_path(path)
            if normalized is None:
                raise ValueError(f"Expected path '{path}' escapes the workspace")
            expected_paths.append(normalized)
        return expected_paths

    @staticmethod
    def _path_matches(path: str, expected: str) -> bool:
        return path_matches_scope(path, expected)
