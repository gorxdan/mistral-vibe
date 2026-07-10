from __future__ import annotations

from dataclasses import dataclass, field
import math

from vibe.core.failure_diagnostic import FailureCategory, FailureDiagnostic
from vibe.core.repair.models import (
    ProgressSnapshot,
    RepairAction,
    RepairDecision,
    RepairEpisodeMetrics,
    RepairEpisodeOutcome,
    RetryBudgetSet,
)

_ESCALATION_ELIGIBLE = frozenset({
    FailureCategory.TOOL_ARGUMENT_PARSE,
    FailureCategory.TOOL_ARGUMENT_SCHEMA,
    FailureCategory.RESULT_SCHEMA,
    FailureCategory.ACCEPTANCE_CHECK,
    FailureCategory.PROVIDER_TRANSPORT,
})
_NON_REPAIRABLE = frozenset({FailureCategory.POLICY, FailureCategory.BUDGET})
_STOP_STRIKES = 2


@dataclass(slots=True)
class _EpisodeState:
    attempts: int = 0
    added_tokens: int = 0
    added_cost_usd: float = 0.0
    seen_snapshots: set[str] = field(default_factory=set)
    no_progress_strikes: int = 0
    finished: bool = False
    recovered: bool = False
    escalation_reason: str | None = None
    terminal_reason: str | None = None


class RepairController:
    def __init__(self, budgets: RetryBudgetSet) -> None:
        self._budgets = budgets
        self._states: dict[FailureCategory, _EpisodeState] = {}

    @classmethod
    def with_finite_defaults(cls) -> RepairController:
        return cls(RetryBudgetSet.finite_defaults())

    def observe_failure(
        self,
        diagnostic: FailureDiagnostic,
        snapshot: ProgressSnapshot,
        *,
        caller_budget_remaining: bool,
        added_tokens: int = 0,
        added_cost_usd: float = 0.0,
    ) -> RepairDecision:
        if snapshot.error_fingerprint != diagnostic.fingerprint:
            raise ValueError(
                "progress snapshot error fingerprint must match the diagnostic"
            )
        self._validate_cost(added_tokens, added_cost_usd)
        category = diagnostic.category
        state = self._state(category)
        self._require_open(state, category)
        state.attempts += 1
        state.added_tokens += added_tokens
        state.added_cost_usd += added_cost_usd

        fingerprint = snapshot.semantic_fingerprint
        repeated = fingerprint in state.seen_snapshots
        if repeated:
            state.no_progress_strikes += 1
        else:
            state.seen_snapshots.add(fingerprint)
            state.no_progress_strikes = 0

        maximum = self._budgets.max_attempts_for(category)
        remaining = max(maximum - state.attempts, 0)
        made_progress = not repeated

        if category in _NON_REPAIRABLE or not diagnostic.retryable:
            reason = f"{category.value} failures are not repairable in this episode"
            return self._finish(
                state,
                diagnostic,
                snapshot,
                action=RepairAction.STOP,
                remaining=remaining,
                made_progress=made_progress,
                reason=reason,
            )

        if state.no_progress_strikes >= _STOP_STRIKES:
            if (
                category in _ESCALATION_ELIGIBLE
                and caller_budget_remaining
                and remaining > 0
            ):
                reason = f"Repeated {category.value} state without semantic progress"
                state.escalation_reason = reason
                return self._finish(
                    state,
                    diagnostic,
                    snapshot,
                    action=RepairAction.ESCALATE,
                    remaining=remaining,
                    made_progress=False,
                    reason=reason,
                    escalation_reason=reason,
                )
            reason = "Second no-progress repeat reached the bounded stop"
            if not caller_budget_remaining:
                reason += "; caller budget is exhausted"
            return self._finish(
                state,
                diagnostic,
                snapshot,
                action=RepairAction.STOP,
                remaining=remaining,
                made_progress=False,
                reason=reason,
            )

        if remaining == 0:
            reason = f"Retry budget exhausted for {category.value}"
            return self._finish(
                state,
                diagnostic,
                snapshot,
                action=RepairAction.STOP,
                remaining=0,
                made_progress=made_progress,
                reason=reason,
            )

        if state.no_progress_strikes == 1:
            return self._decision(
                state,
                diagnostic,
                snapshot,
                action=RepairAction.WARN,
                remaining=remaining,
                made_progress=False,
                reason=(
                    "No semantic progress: change the repair action before retrying"
                ),
            )

        return self._decision(
            state,
            diagnostic,
            snapshot,
            action=RepairAction.CONTINUE,
            remaining=remaining,
            made_progress=True,
            reason="New semantic state observed; bounded repair may continue",
        )

    def record_recovered(
        self,
        category: FailureCategory,
        *,
        added_tokens: int = 0,
        added_cost_usd: float = 0.0,
    ) -> RepairDecision:
        self._validate_cost(added_tokens, added_cost_usd)
        state = self._state(category)
        self._require_open(state, category)
        state.added_tokens += added_tokens
        state.added_cost_usd += added_cost_usd
        state.finished = True
        state.recovered = True
        state.terminal_reason = "Repair recovered the failure"
        remaining = max(self._budgets.max_attempts_for(category) - state.attempts, 0)
        return RepairDecision(
            action=RepairAction.RECOVERED,
            category=category,
            attempt=state.attempts,
            remaining_attempts=remaining,
            made_progress=True,
            no_progress_strikes=state.no_progress_strikes,
            reason=state.terminal_reason,
            metrics=self._metrics(category, state),
        )

    def record_escalated_recovery(
        self,
        category: FailureCategory,
        *,
        added_tokens: int = 0,
        added_cost_usd: float = 0.0,
    ) -> RepairDecision:
        self._validate_cost(added_tokens, added_cost_usd)
        state = self._state(category)
        if not state.finished or state.escalation_reason is None or state.recovered:
            raise RuntimeError(
                f"repair episode for {category.value} has no open escalation"
            )
        state.added_tokens += added_tokens
        state.added_cost_usd += added_cost_usd
        state.recovered = True
        state.terminal_reason = "Semantic escalation recovered the failure"
        remaining = max(self._budgets.max_attempts_for(category) - state.attempts, 0)
        return RepairDecision(
            action=RepairAction.RECOVERED,
            category=category,
            attempt=state.attempts,
            remaining_attempts=remaining,
            made_progress=True,
            no_progress_strikes=state.no_progress_strikes,
            reason=state.terminal_reason,
            escalation_reason=state.escalation_reason,
            metrics=self._metrics(category, state),
        )

    def record_escalation_failure(
        self,
        category: FailureCategory,
        reason: str,
        *,
        added_tokens: int = 0,
        added_cost_usd: float = 0.0,
    ) -> None:
        self._validate_cost(added_tokens, added_cost_usd)
        state = self._state(category)
        if not state.finished or state.escalation_reason is None or state.recovered:
            raise RuntimeError(
                f"repair episode for {category.value} has no open escalation"
            )
        state.added_tokens += added_tokens
        state.added_cost_usd += added_cost_usd
        state.terminal_reason = reason

    def metrics(self, category: FailureCategory) -> RepairEpisodeMetrics:
        return self._metrics(category, self._state(category))

    def reset(self, category: FailureCategory | None = None) -> None:
        if category is None:
            self._states.clear()
            return
        self._states.pop(category, None)

    def _state(self, category: FailureCategory) -> _EpisodeState:
        return self._states.setdefault(category, _EpisodeState())

    @staticmethod
    def _validate_cost(added_tokens: int, added_cost_usd: float) -> None:
        if added_tokens < 0 or added_cost_usd < 0 or not math.isfinite(added_cost_usd):
            raise ValueError(
                "repair token and cost additions must be non-negative and finite"
            )

    @staticmethod
    def _require_open(state: _EpisodeState, category: FailureCategory) -> None:
        if state.finished:
            raise RuntimeError(
                f"repair episode for {category.value} is already finished"
            )

    def _finish(
        self,
        state: _EpisodeState,
        diagnostic: FailureDiagnostic,
        snapshot: ProgressSnapshot,
        *,
        action: RepairAction,
        remaining: int,
        made_progress: bool,
        reason: str,
        escalation_reason: str | None = None,
    ) -> RepairDecision:
        state.finished = True
        state.terminal_reason = reason
        return self._decision(
            state,
            diagnostic,
            snapshot,
            action=action,
            remaining=remaining,
            made_progress=made_progress,
            reason=reason,
            escalation_reason=escalation_reason,
        )

    def _decision(
        self,
        state: _EpisodeState,
        diagnostic: FailureDiagnostic,
        snapshot: ProgressSnapshot,
        *,
        action: RepairAction,
        remaining: int,
        made_progress: bool,
        reason: str,
        escalation_reason: str | None = None,
    ) -> RepairDecision:
        return RepairDecision(
            action=action,
            category=diagnostic.category,
            attempt=state.attempts,
            remaining_attempts=remaining,
            made_progress=made_progress,
            no_progress_strikes=state.no_progress_strikes,
            reason=reason,
            escalation_reason=escalation_reason,
            diagnostic=diagnostic,
            snapshot_fingerprint=snapshot.semantic_fingerprint,
            metrics=self._metrics(diagnostic.category, state),
        )

    @staticmethod
    def _metrics(
        category: FailureCategory, state: _EpisodeState
    ) -> RepairEpisodeMetrics:
        return RepairEpisodeMetrics(
            category=category,
            outcome=(
                RepairEpisodeOutcome.RECOVERED
                if state.recovered
                else RepairEpisodeOutcome.NOT_RECOVERED
            ),
            finished=state.finished,
            attempts=state.attempts,
            added_tokens=state.added_tokens,
            added_cost_usd=state.added_cost_usd,
            escalation_reason=state.escalation_reason,
            terminal_reason=state.terminal_reason,
        )
