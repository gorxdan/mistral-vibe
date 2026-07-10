from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Annotated, Literal, final
from uuid import uuid4

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from vibe.core.usage._context import (
    MAX_RESERVATION_LEASE_S,
    PromptTokenEstimate,
    SpendAmount,
    SpendContext,
    SpendEnvelope,
    SpendEnvelopeLimits,
    SpendEnvelopeSnapshot,
    SpendRejection,
    SpendRejectionReason,
    SpendReservation,
    SpendRetryAuthorization,
    SpendRetryCause,
    SpendRetryPolicyReason,
    SpendScopeKind,
    SpendSettlement,
    SpendSettlementDisposition,
)
from vibe.core.usage._prompt_estimator import (
    PROMPT_OBSERVATION_WINDOW,
    PromptObservation,
    PromptReservationPlan,
    estimate_prompt_tokens,
)
from vibe.core.utils.io import read_safe, write_durable

__all__ = [
    "LedgerEvent",
    "SpendLedger",
    "SpendLedgerBusyError",
    "SpendLedgerConflictError",
    "SpendLedgerCorruptError",
    "SpendLedgerError",
]


class SpendLedgerError(RuntimeError):
    pass


class SpendLedgerBusyError(SpendLedgerError):
    pass


class SpendLedgerConflictError(SpendLedgerError):
    pass


class SpendLedgerCorruptError(SpendLedgerError):
    pass


class _EventBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sequence: int = Field(ge=1)
    event_id: str = Field(min_length=1, max_length=256)
    timestamp: float = Field(ge=0.0)


@final
class _ScopeDefinedEvent(_EventBase):
    kind: Literal["scope_defined"] = "scope_defined"
    envelope: SpendEnvelope


@final
class _EnvelopeTightenedEvent(_EventBase):
    kind: Literal["envelope_tightened"] = "envelope_tightened"
    scope_id: str
    limits: SpendEnvelopeLimits


type _LegacyTokenLimitField = Literal[
    "max_prompt_tokens", "max_completion_tokens", "max_total_tokens"
]

_LEGACY_TOKEN_LIMIT_DEFAULTS: dict[_LegacyTokenLimitField, int] = {
    "max_prompt_tokens": 400_000,
    "max_completion_tokens": 100_000,
    "max_total_tokens": 500_000,
}
_LEGACY_TOKEN_LIMIT_FIELDS = tuple(_LEGACY_TOKEN_LIMIT_DEFAULTS)


@final
class _EnvelopePolicyMigratedEvent(_EventBase):
    kind: Literal["envelope_policy_migrated"] = "envelope_policy_migrated"
    scope_id: str
    from_policy_version: Literal[1] = 1
    to_policy_version: Literal[2] = 2
    cleared_fields: tuple[_LegacyTokenLimitField, ...] = Field(min_length=1)
    limits: SpendEnvelopeLimits


@final
class _ReservedEvent(_EventBase):
    kind: Literal["reserved"] = "reserved"
    reservation: SpendReservation


@final
class _DispatchStartedEvent(_EventBase):
    kind: Literal["dispatch_started"] = "dispatch_started"
    reservation_id: str


@final
class _ReconciledEvent(_EventBase):
    kind: Literal["reconciled"] = "reconciled"
    reservation_id: str
    amount: SpendAmount
    estimated: bool


@final
class _ReleasedEvent(_EventBase):
    kind: Literal["released"] = "released"
    reservation_id: str
    reason: str


@final
class _ExpiredEvent(_EventBase):
    kind: Literal["expired"] = "expired"
    reservation_id: str
    charge_estimate: bool = True


@final
class _LeaseRenewedEvent(_EventBase):
    kind: Literal["lease_renewed"] = "lease_renewed"
    reservation_id: str
    lease_expires_at: float


@final
class _RejectedEvent(_EventBase):
    kind: Literal["rejected"] = "rejected"
    rejection: SpendRejection


@final
class _RetryAuthorizedEvent(_EventBase):
    kind: Literal["retry_authorized"] = "retry_authorized"
    authorization: SpendRetryAuthorization


@final
class _RetryBudgetRejectedEvent(_EventBase):
    kind: Literal["retry_budget_rejected"] = "retry_budget_rejected"
    cause: SpendRetryCause
    attempt: int = Field(ge=1)
    rejection: SpendRejection


@final
class _RetryPolicyRejectedEvent(_EventBase):
    kind: Literal["retry_policy_rejected"] = "retry_policy_rejected"
    reservation_id: str = Field(min_length=1, max_length=256)
    cause: SpendRetryCause
    attempt: int = Field(ge=1)
    reason: SpendRetryPolicyReason
    elapsed_s: float = Field(ge=0.0)
    max_elapsed_s: float = Field(ge=0.0)
    next_delay_s: float = Field(ge=0.0)
    max_retries: int = Field(ge=0)


LedgerEvent = Annotated[
    _ScopeDefinedEvent
    | _EnvelopeTightenedEvent
    | _EnvelopePolicyMigratedEvent
    | _ReservedEvent
    | _DispatchStartedEvent
    | _ReconciledEvent
    | _ReleasedEvent
    | _ExpiredEvent
    | _LeaseRenewedEvent
    | _RejectedEvent
    | _RetryAuthorizedEvent
    | _RetryBudgetRejectedEvent
    | _RetryPolicyRejectedEvent,
    Field(discriminator="kind"),
]
_EVENT_ADAPTER = TypeAdapter(LedgerEvent)


@dataclass(slots=True)
class _Totals:
    spent: SpendAmount = field(default_factory=SpendAmount)
    reserved: SpendAmount = field(default_factory=SpendAmount)
    rejected: SpendAmount = field(default_factory=SpendAmount)
    spent_calls: int = 0
    reserved_calls: int = 0
    rejected_calls: int = 0
    spent_retries: int = 0
    reserved_retries: int = 0
    rejected_retries: int = 0


@dataclass(slots=True)
class _State:
    sequence: int = 0
    scopes: dict[str, SpendEnvelope] = field(default_factory=dict)
    reservations: dict[str, SpendReservation] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)
    dispatched: set[str] = field(default_factory=set)
    settlements: dict[str, SpendSettlement] = field(default_factory=dict)
    rejections: list[SpendRejection] = field(default_factory=list)
    retry_authorizations: list[SpendRetryAuthorization] = field(default_factory=list)
    retry_budget_rejections: list[SpendRejection] = field(default_factory=list)
    retry_policy_rejections: list[_RetryPolicyRejectedEvent] = field(
        default_factory=list
    )
    prompt_observations: dict[str, list[PromptObservation]] = field(
        default_factory=dict
    )
    envelope_policy_migrations: set[str] = field(default_factory=set)


def _add_amount(left: SpendAmount, right: SpendAmount) -> SpendAmount:
    return SpendAmount(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        cached_tokens=left.cached_tokens + right.cached_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def _remaining(limit: int | None, used: int) -> int | None:
    return None if limit is None else max(limit - used, 0)


def _remaining_cost(limit: float | None, used: float) -> float | None:
    return None if limit is None else max(limit - used, 0.0)


def _tighter_limit[T: int | float](current: T | None, requested: T | None) -> T | None:
    if requested is None:
        return current
    if current is None:
        return requested
    return min(current, requested)


def _tighten_limits(
    current: SpendEnvelopeLimits, requested: SpendEnvelopeLimits
) -> SpendEnvelopeLimits:
    return SpendEnvelopeLimits(
        max_prompt_tokens=_tighter_limit(
            current.max_prompt_tokens, requested.max_prompt_tokens
        ),
        max_completion_tokens=_tighter_limit(
            current.max_completion_tokens, requested.max_completion_tokens
        ),
        max_total_tokens=_tighter_limit(
            current.max_total_tokens, requested.max_total_tokens
        ),
        max_cost_usd=_tighter_limit(current.max_cost_usd, requested.max_cost_usd),
        max_calls=_tighter_limit(current.max_calls, requested.max_calls),
        max_concurrent_calls=_tighter_limit(
            current.max_concurrent_calls, requested.max_concurrent_calls
        ),
        max_retries=_tighter_limit(current.max_retries, requested.max_retries),
        deadline_at=_tighter_limit(current.deadline_at, requested.deadline_at),
    )


class SpendLedger:
    """File-lock-backed append-only budget ledger shared by local processes."""

    def __init__(
        self,
        path: Path,
        *,
        lock_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self._events_dir = path / "events"
        self._lock_path = path / "ledger.lock"
        self._lock_timeout_s = lock_timeout_s
        self._clock = clock

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._lock_path), timeout=self._lock_timeout_s)
        try:
            with lock:
                self._events_dir.mkdir(parents=True, exist_ok=True)
                yield
        except Timeout as e:
            raise SpendLedgerBusyError(str(self._lock_path)) from e

    def _load(self) -> _State:
        state = _State()
        event_paths = sorted(self._events_dir.glob("*.json"))
        for expected_sequence, event_path in enumerate(event_paths, start=1):
            try:
                event = _EVENT_ADAPTER.validate_json(read_safe(event_path).text)
            except (OSError, ValidationError) as e:
                raise SpendLedgerCorruptError(str(event_path)) from e
            if event.sequence != expected_sequence:
                raise SpendLedgerCorruptError(
                    f"non-contiguous event sequence at {event_path}"
                )
            self._apply(state, event)
        return state

    def _apply(self, state: _State, event: LedgerEvent) -> None:
        if event.sequence != state.sequence + 1:
            raise SpendLedgerCorruptError("event sequence is out of order")
        state.sequence = event.sequence
        if isinstance(
            event,
            (_ScopeDefinedEvent, _EnvelopeTightenedEvent, _EnvelopePolicyMigratedEvent),
        ):
            if isinstance(event, _ScopeDefinedEvent):
                self._apply_scope(state, event.envelope)
            elif isinstance(event, _EnvelopeTightenedEvent):
                self._apply_tightening(state, event)
            else:
                self._apply_policy_migration(state, event)
            return
        if isinstance(event, _ReservedEvent):
            reservation = event.reservation
            if reservation.reservation_id in state.reservations:
                raise SpendLedgerCorruptError("duplicate reservation event")
            if reservation.scope_id not in state.scopes:
                raise SpendLedgerCorruptError("reservation references unknown scope")
            if (
                tuple(self._scope_chain(state, reservation.scope_id))
                + (reservation.call_scope_id,)
                != reservation.scope_chain
            ):
                raise SpendLedgerCorruptError("reservation scope chain is invalid")
            state.reservations[reservation.reservation_id] = reservation
            state.active.add(reservation.reservation_id)
            return
        if isinstance(event, (_DispatchStartedEvent, _LeaseRenewedEvent)):
            self._apply_active_update(state, event)
            return
        if isinstance(event, (_ReleasedEvent, _ExpiredEvent, _ReconciledEvent)):
            self._apply_settlement_event(state, event)
            return
        if isinstance(event, _RejectedEvent):
            state.rejections.append(event.rejection)
            return
        self._apply_retry_event(state, event)

    def _apply_settlement_event(
        self, state: _State, event: _ReleasedEvent | _ExpiredEvent | _ReconciledEvent
    ) -> None:
        if isinstance(event, _ExpiredEvent):
            self._apply_expiration(state, event)
            return
        if isinstance(event, _ReconciledEvent):
            self._apply_reconciliation(state, event)
            return
        self._active_reservation(state, event.reservation_id)
        state.active.remove(event.reservation_id)
        state.settlements[event.reservation_id] = SpendSettlement(
            reservation_id=event.reservation_id,
            disposition=SpendSettlementDisposition.RELEASED,
            amount=SpendAmount(),
            estimated=False,
            applied=True,
            timestamp=event.timestamp,
            reason=event.reason,
        )

    def _apply_retry_event(
        self,
        state: _State,
        event: _RetryAuthorizedEvent
        | _RetryBudgetRejectedEvent
        | _RetryPolicyRejectedEvent,
    ) -> None:
        if isinstance(event, _RetryAuthorizedEvent):
            self._apply_retry_authorization(state, event.authorization)
            return
        if isinstance(event, _RetryBudgetRejectedEvent):
            self._apply_retry_budget_rejection(state, event)
            return
        self._apply_retry_policy_denial(state, event)

    def _apply_active_update(
        self, state: _State, event: _DispatchStartedEvent | _LeaseRenewedEvent
    ) -> None:
        reservation = self._active_reservation(state, event.reservation_id)
        if isinstance(event, _DispatchStartedEvent):
            if event.reservation_id in state.dispatched:
                raise SpendLedgerCorruptError("reservation dispatched more than once")
            state.dispatched.add(event.reservation_id)
            return
        state.reservations[event.reservation_id] = reservation.model_copy(
            update={"lease_expires_at": event.lease_expires_at}
        )

    def _apply_retry_authorization(
        self, state: _State, authorization: SpendRetryAuthorization
    ) -> None:
        reservation = self._retry_reservation(state, authorization.reservation_id)
        if authorization.call_scope_id != reservation.call_scope_id:
            raise SpendLedgerCorruptError("retry references the wrong call scope")
        if authorization.scope_chain != reservation.scope_chain:
            raise SpendLedgerCorruptError("retry scope chain is invalid")
        if authorization.attempt != self._next_retry_attempt(
            state, authorization.reservation_id
        ):
            raise SpendLedgerCorruptError("retry attempt is out of order")
        if (
            self._retry_budget_denial(state, reservation, authorization.timestamp)
            is not None
        ):
            raise SpendLedgerCorruptError("retry authorization exceeds its envelope")
        state.retry_authorizations.append(authorization)

    def _apply_retry_budget_rejection(
        self, state: _State, event: _RetryBudgetRejectedEvent
    ) -> None:
        reservation = self._retry_reservation(state, event.rejection.call_id)
        if event.attempt != self._next_retry_attempt(state, reservation.reservation_id):
            raise SpendLedgerCorruptError("retry rejection attempt is out of order")
        expected = self._retry_budget_denial(state, reservation, event.timestamp)
        if expected is None or expected != event.rejection:
            raise SpendLedgerCorruptError("retry budget rejection is invalid")
        state.retry_budget_rejections.append(event.rejection)

    def _apply_retry_policy_denial(
        self, state: _State, event: _RetryPolicyRejectedEvent
    ) -> None:
        reservation = self._retry_reservation(state, event.reservation_id)
        if event.attempt != self._next_retry_attempt(state, reservation.reservation_id):
            raise SpendLedgerCorruptError("retry rejection attempt is out of order")
        authorized = event.attempt - 1
        if (
            event.reason is SpendRetryPolicyReason.ATTEMPT_LIMIT
            and authorized < event.max_retries
        ):
            raise SpendLedgerCorruptError("retry attempt-limit rejection is invalid")
        if (
            event.reason is SpendRetryPolicyReason.ELAPSED_LIMIT
            and event.elapsed_s + event.next_delay_s < event.max_elapsed_s
        ):
            raise SpendLedgerCorruptError("retry elapsed-limit rejection is invalid")
        state.retry_policy_rejections.append(event)

    @staticmethod
    def _retry_reservation(state: _State, reservation_id: str) -> SpendReservation:
        reservation = SpendLedger._active_reservation(state, reservation_id)
        if reservation_id not in state.dispatched:
            raise SpendLedgerCorruptError(
                "retry references an undispatched reservation"
            )
        return reservation

    @staticmethod
    def _next_retry_attempt(state: _State, reservation_id: str) -> int:
        return (
            sum(
                authorization.reservation_id == reservation_id
                for authorization in state.retry_authorizations
            )
            + 1
        )

    def _retry_budget_denial(
        self, state: _State, reservation: SpendReservation, now: float
    ) -> SpendRejection | None:
        reason: SpendRejectionReason | None = None
        limited_scope_id: str | None = None
        for scope_id in reservation.scope_chain[:-1]:
            limits = state.scopes[scope_id].limits
            totals = self._totals(state, scope_id)
            projected = _add_amount(
                _add_amount(totals.spent, totals.reserved), reservation.estimate
            )
            if limits.deadline_at is not None and now >= limits.deadline_at:
                reason = SpendRejectionReason.DEADLINE
            elif (
                limits.max_prompt_tokens is not None
                and projected.prompt_tokens > limits.max_prompt_tokens
            ):
                reason = SpendRejectionReason.PROMPT_TOKENS
            elif (
                limits.max_completion_tokens is not None
                and projected.completion_tokens > limits.max_completion_tokens
            ):
                reason = SpendRejectionReason.COMPLETION_TOKENS
            elif (
                limits.max_total_tokens is not None
                and projected.total_tokens > limits.max_total_tokens
            ):
                reason = SpendRejectionReason.TOTAL_TOKENS
            elif (
                limits.max_cost_usd is not None
                and projected.cost_usd > limits.max_cost_usd
            ):
                reason = SpendRejectionReason.COST_USD
            elif (
                limits.max_retries is not None
                and totals.spent_retries + totals.reserved_retries + 1
                > limits.max_retries
            ):
                reason = SpendRejectionReason.RETRIES
            if reason is not None:
                limited_scope_id = scope_id
                break
        if reason is None:
            return None
        return SpendRejection(
            call_id=reservation.reservation_id,
            scope_id=reservation.scope_id,
            scope_chain=reservation.scope_chain[:-1],
            purpose=reservation.purpose,
            estimate=reservation.estimate,
            is_retry=True,
            reason=reason,
            limited_scope_id=limited_scope_id,
            timestamp=now,
        )

    def _apply_expiration(self, state: _State, event: _ExpiredEvent) -> None:
        reservation = self._active_reservation(state, event.reservation_id)
        expected_charge = (
            reservation.dispatch_tracking_version == 0
            or event.reservation_id in state.dispatched
        )
        if event.charge_estimate != expected_charge:
            raise SpendLedgerCorruptError("expired reservation disposition is invalid")
        state.active.remove(event.reservation_id)
        state.settlements[event.reservation_id] = SpendSettlement(
            reservation_id=event.reservation_id,
            disposition=(
                SpendSettlementDisposition.EXPIRED
                if event.charge_estimate
                else SpendSettlementDisposition.RELEASED
            ),
            amount=reservation.estimate if event.charge_estimate else SpendAmount(),
            estimated=event.charge_estimate,
            applied=True,
            timestamp=event.timestamp,
            reason=(
                "dispatched reservation lease expired"
                if event.charge_estimate
                else "undispatched reservation lease expired"
            ),
        )

    def _apply_scope(self, state: _State, scope: SpendEnvelope) -> None:
        if scope.scope_id in state.scopes:
            raise SpendLedgerCorruptError("duplicate scope definition")
        parent = (
            state.scopes.get(scope.parent_scope_id) if scope.parent_scope_id else None
        )
        if scope.kind == SpendScopeKind.SESSION:
            if parent is not None or scope.parent_scope_id is not None:
                raise SpendLedgerCorruptError("session scope cannot have a parent")
        elif parent is None:
            raise SpendLedgerCorruptError("child scope references unknown parent")
        elif not self._valid_parent(scope.kind, parent.kind):
            raise SpendLedgerCorruptError("scope hierarchy is invalid")
        state.scopes[scope.scope_id] = scope

    def _apply_tightening(self, state: _State, event: _EnvelopeTightenedEvent) -> None:
        scope = state.scopes.get(event.scope_id)
        if scope is None:
            raise SpendLedgerCorruptError(
                "envelope tightening references unknown scope"
            )
        if _tighten_limits(scope.limits, event.limits) != event.limits:
            raise SpendLedgerCorruptError("envelope limits were not monotonic")
        state.scopes[event.scope_id] = scope.model_copy(update={"limits": event.limits})

    def _apply_policy_migration(
        self, state: _State, event: _EnvelopePolicyMigratedEvent
    ) -> None:
        scope = state.scopes.get(event.scope_id)
        if scope is None:
            raise SpendLedgerCorruptError(
                "envelope policy migration references unknown scope"
            )
        if scope.kind != SpendScopeKind.SESSION:
            raise SpendLedgerCorruptError(
                "envelope policy migration requires a session scope"
            )
        if scope.policy_version != event.from_policy_version:
            raise SpendLedgerCorruptError(
                "envelope policy migration has the wrong source version"
            )
        if event.scope_id in state.envelope_policy_migrations:
            raise SpendLedgerCorruptError("envelope policy migrated more than once")
        requested = set(event.cleared_fields)
        canonical_fields = tuple(
            field_name
            for field_name in _LEGACY_TOKEN_LIMIT_FIELDS
            if field_name in requested
        )
        if event.cleared_fields != canonical_fields:
            raise SpendLedgerCorruptError(
                "envelope policy migration fields are not canonical"
            )
        for field_name in event.cleared_fields:
            if (
                getattr(scope.limits, field_name)
                != _LEGACY_TOKEN_LIMIT_DEFAULTS[field_name]
            ):
                raise SpendLedgerCorruptError(
                    "envelope policy migration did not start from a legacy default"
                )
        expected_limits = scope.limits.model_copy(
            update={field_name: None for field_name in event.cleared_fields}
        )
        if event.limits != expected_limits:
            raise SpendLedgerCorruptError(
                "envelope policy migration changed unrelated limits"
            )
        state.scopes[event.scope_id] = scope.model_copy(
            update={"limits": event.limits, "policy_version": event.to_policy_version}
        )
        state.envelope_policy_migrations.add(event.scope_id)

    @staticmethod
    def _valid_parent(kind: SpendScopeKind, parent_kind: SpendScopeKind) -> bool:
        if kind in {SpendScopeKind.WORKFLOW, SpendScopeKind.TEAM}:
            return parent_kind == SpendScopeKind.SESSION
        if kind == SpendScopeKind.AGENT:
            return parent_kind in {
                SpendScopeKind.SESSION,
                SpendScopeKind.WORKFLOW,
                SpendScopeKind.TEAM,
            }
        return kind == SpendScopeKind.CALL and parent_kind == SpendScopeKind.AGENT

    def _apply_reconciliation(self, state: _State, event: _ReconciledEvent) -> None:
        reservation_id = event.reservation_id
        if reservation_id not in state.reservations:
            raise SpendLedgerCorruptError(
                "reconciliation references unknown reservation"
            )
        previous = state.settlements.get(reservation_id)
        if reservation_id not in state.active and (
            previous is None
            or not previous.estimated
            or previous.disposition == SpendSettlementDisposition.RELEASED
        ):
            raise SpendLedgerCorruptError("reservation reconciled more than once")
        state.active.discard(reservation_id)
        state.settlements[reservation_id] = SpendSettlement(
            reservation_id=reservation_id,
            disposition=SpendSettlementDisposition.RECONCILED,
            amount=event.amount,
            estimated=event.estimated,
            applied=True,
            timestamp=event.timestamp,
        )
        prompt_estimate = state.reservations[reservation_id].prompt_estimate
        if (
            not event.estimated
            and prompt_estimate is not None
            and event.amount.prompt_tokens > 0
        ):
            observations = state.prompt_observations.setdefault(
                prompt_estimate.profile_key, []
            )
            observations.append(
                PromptObservation(
                    base_tokens=prompt_estimate.base_tokens,
                    actual_tokens=event.amount.prompt_tokens,
                )
            )
            del observations[:-PROMPT_OBSERVATION_WINDOW]

    @staticmethod
    def _active_reservation(state: _State, reservation_id: str) -> SpendReservation:
        reservation = state.reservations.get(reservation_id)
        if reservation is None or reservation_id not in state.active:
            raise SpendLedgerCorruptError("event references inactive reservation")
        return reservation

    def _append(self, state: _State, event: LedgerEvent) -> None:
        if event.sequence != state.sequence + 1:
            raise SpendLedgerConflictError("event sequence changed during transaction")
        event_path = self._events_dir / f"{event.sequence:020d}.json"
        if event_path.exists():
            raise SpendLedgerConflictError(str(event_path))
        write_durable(
            event_path, event.model_dump_json().encode("utf-8"), suffix=".event.tmp"
        )
        self._apply(state, event)

    def define_envelope(self, scope: SpendEnvelope) -> SpendEnvelope:
        if scope.kind == SpendScopeKind.CALL:
            raise SpendLedgerConflictError("call scopes are created by reservations")
        with self._locked():
            state = self._load()
            existing = state.scopes.get(scope.scope_id)
            if existing is not None:
                if existing == scope:
                    return existing
                raise SpendLedgerConflictError(
                    f"scope {scope.scope_id!r} already has a different definition"
                )
            self._validate_new_scope(state, scope)
            now = self._clock()
            self._append(
                state,
                _ScopeDefinedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=now,
                    envelope=scope,
                ),
            )
            return scope

    def define_or_tighten_envelope(self, scope: SpendEnvelope) -> SpendEnvelope:
        if scope.kind == SpendScopeKind.CALL:
            raise SpendLedgerConflictError("call scopes are created by reservations")
        with self._locked():
            state = self._load()
            existing = state.scopes.get(scope.scope_id)
            if existing is None:
                self._validate_new_scope(state, scope)
                self._append(
                    state,
                    _ScopeDefinedEvent(
                        sequence=state.sequence + 1,
                        event_id=uuid4().hex,
                        timestamp=self._clock(),
                        envelope=scope,
                    ),
                )
                return scope
            if existing.model_copy(update={"limits": scope.limits}) != scope:
                raise SpendLedgerConflictError(
                    f"scope {scope.scope_id!r} already has a different definition"
                )
            limits = _tighten_limits(existing.limits, scope.limits)
            if limits == existing.limits:
                return existing
            self._append(
                state,
                _EnvelopeTightenedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=self._clock(),
                    scope_id=scope.scope_id,
                    limits=limits,
                ),
            )
            return state.scopes[scope.scope_id]

    def get_envelope(self, scope_id: str) -> SpendEnvelope | None:
        with self._locked():
            return self._load().scopes.get(scope_id)

    def tighten_envelope(
        self, scope_id: str, requested: SpendEnvelopeLimits
    ) -> SpendEnvelope:
        with self._locked():
            state = self._load()
            scope = state.scopes.get(scope_id)
            if scope is None:
                raise SpendLedgerConflictError(f"unknown scope {scope_id!r}")
            limits = _tighten_limits(scope.limits, requested)
            if limits == scope.limits:
                return scope
            self._append(
                state,
                _EnvelopeTightenedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=self._clock(),
                    scope_id=scope_id,
                    limits=limits,
                ),
            )
            return state.scopes[scope_id]

    def migrate_legacy_default_token_limits(
        self,
        scope_id: str,
        *,
        clear_prompt_tokens: bool,
        clear_completion_tokens: bool,
        clear_total_tokens: bool,
    ) -> SpendEnvelope:
        requested_fields: list[_LegacyTokenLimitField] = []
        if clear_prompt_tokens:
            requested_fields.append("max_prompt_tokens")
        if clear_completion_tokens:
            requested_fields.append("max_completion_tokens")
        if clear_total_tokens:
            requested_fields.append("max_total_tokens")
        if not requested_fields:
            raise ValueError("at least one legacy token limit must be selected")
        with self._locked():
            state = self._load()
            scope = state.scopes.get(scope_id)
            if scope is None:
                raise SpendLedgerConflictError(f"unknown scope {scope_id!r}")
            if scope.kind != SpendScopeKind.SESSION:
                raise SpendLedgerConflictError(
                    "legacy token limit migration requires a session scope"
                )
            fields_to_clear: list[_LegacyTokenLimitField] = []
            for field_name in requested_fields:
                current = getattr(scope.limits, field_name)
                if current is None:
                    continue
                if current != _LEGACY_TOKEN_LIMIT_DEFAULTS[field_name]:
                    raise SpendLedgerConflictError(
                        f"{field_name} is not the legacy default"
                    )
                fields_to_clear.append(field_name)
            if not fields_to_clear:
                return scope
            if scope_id in state.envelope_policy_migrations:
                raise SpendLedgerConflictError(
                    "envelope policy migration was already applied"
                )
            if scope.policy_version != 1:
                raise SpendLedgerConflictError(
                    "legacy token limit migration requires policy version 1"
                )
            limits = scope.limits.model_copy(
                update={field_name: None for field_name in fields_to_clear}
            )
            self._append(
                state,
                _EnvelopePolicyMigratedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=self._clock(),
                    scope_id=scope_id,
                    cleared_fields=tuple(fields_to_clear),
                    limits=limits,
                ),
            )
            return state.scopes[scope_id]

    def _validate_new_scope(self, state: _State, scope: SpendEnvelope) -> None:
        if scope.kind == SpendScopeKind.SESSION:
            if scope.parent_scope_id is not None:
                raise SpendLedgerConflictError("session scope cannot have a parent")
            return
        parent = state.scopes.get(scope.parent_scope_id or "")
        if parent is None:
            raise SpendLedgerConflictError("parent scope must be defined first")
        if not self._valid_parent(scope.kind, parent.kind):
            raise SpendLedgerConflictError(
                f"{scope.kind.value} cannot be a child of {parent.kind.value}"
            )

    def try_reserve(
        self, context: SpendContext, estimate: SpendAmount, *, lease_s: float
    ) -> SpendReservation | SpendRejection:
        if lease_s <= 0 or lease_s > MAX_RESERVATION_LEASE_S:
            raise ValueError(f"lease_s must be in (0, {MAX_RESERVATION_LEASE_S:g}]")
        with self._locked():
            state = self._load()
            return self._try_reserve_loaded(state, context, estimate, lease_s=lease_s)

    def try_reserve_prompt(
        self, context: SpendContext, plan: PromptReservationPlan, *, lease_s: float
    ) -> SpendReservation | SpendRejection:
        if lease_s <= 0 or lease_s > MAX_RESERVATION_LEASE_S:
            raise ValueError(f"lease_s must be in (0, {MAX_RESERVATION_LEASE_S:g}]")
        with self._locked():
            state = self._load()
            self._expire_stale(state, self._clock())
            prompt_estimate = estimate_prompt_tokens(
                plan, state.prompt_observations.get(plan.footprint.profile_key, [])
            )
            completion_tokens = plan.completion_tokens
            completion_cost_per_token = (
                plan.completion_cost_usd / plan.completion_tokens
                if plan.completion_tokens > 0
                else 0.0
            )
            if plan.allow_completion_reduction:
                chain = self._known_scope_chain(state, context.scope_id)
                affordable = self._affordable_completion_tokens(
                    state,
                    chain,
                    prompt_tokens=prompt_estimate.estimated_tokens,
                    input_cost_usd_per_token=plan.input_cost_usd_per_token,
                    completion_cost_usd_per_token=completion_cost_per_token,
                    desired=completion_tokens,
                )
                minimum = min(plan.minimum_completion_tokens, completion_tokens)
                completion_tokens = (
                    min(completion_tokens, affordable)
                    if affordable >= minimum
                    else minimum
                )
            estimate = SpendAmount(
                prompt_tokens=prompt_estimate.estimated_tokens,
                completion_tokens=completion_tokens,
                cost_usd=(
                    prompt_estimate.estimated_tokens * plan.input_cost_usd_per_token
                    + completion_tokens * completion_cost_per_token
                ),
            )
            return self._try_reserve_loaded(
                state,
                context,
                estimate,
                lease_s=lease_s,
                prompt_estimate=prompt_estimate,
            )

    def _affordable_completion_tokens(
        self,
        state: _State,
        chain: tuple[str, ...],
        *,
        prompt_tokens: int,
        input_cost_usd_per_token: float,
        completion_cost_usd_per_token: float,
        desired: int,
    ) -> int:
        affordable = desired
        prompt_cost = prompt_tokens * input_cost_usd_per_token
        for scope_id in chain:
            limits = state.scopes[scope_id].limits
            totals = self._totals(state, scope_id)
            used = _add_amount(totals.spent, totals.reserved)
            if limits.max_completion_tokens is not None:
                affordable = min(
                    affordable,
                    max(limits.max_completion_tokens - used.completion_tokens, 0),
                )
            if limits.max_total_tokens is not None:
                affordable = min(
                    affordable,
                    max(limits.max_total_tokens - used.total_tokens - prompt_tokens, 0),
                )
            if limits.max_cost_usd is not None:
                remaining_cost = limits.max_cost_usd - used.cost_usd - prompt_cost
                if completion_cost_usd_per_token > 0:
                    affordable = min(
                        affordable,
                        max(
                            math.floor(
                                max(remaining_cost, 0.0) / completion_cost_usd_per_token
                            ),
                            0,
                        ),
                    )
                elif remaining_cost < 0:
                    affordable = 0
        return max(affordable, 0)

    def _try_reserve_loaded(
        self,
        state: _State,
        context: SpendContext,
        estimate: SpendAmount,
        *,
        lease_s: float,
        prompt_estimate: PromptTokenEstimate | None = None,
    ) -> SpendReservation | SpendRejection:
        now = self._clock()
        self._expire_stale(state, now)
        call_id = context.call_id or uuid4().hex
        chain = self._known_scope_chain(state, context.scope_id)
        rejection = self._reservation_rejection(
            state,
            context,
            estimate,
            call_id,
            chain,
            now,
            prompt_estimate=prompt_estimate,
        )
        if rejection is not None:
            self._record_rejection(state, rejection, now)
            return rejection
        deadline = self._earliest_deadline(state, chain)
        lease_expires_at = now + lease_s
        if deadline is not None:
            lease_expires_at = min(lease_expires_at, deadline)
        call_scope_id = f"call:{call_id}"
        reservation = SpendReservation(
            reservation_id=call_id,
            call_scope_id=call_scope_id,
            scope_id=context.scope_id,
            scope_chain=(*chain, call_scope_id),
            purpose=context.purpose,
            estimate=estimate,
            prompt_estimate=prompt_estimate,
            is_retry=context.is_retry,
            created_at=now,
            lease_expires_at=lease_expires_at,
            dispatch_tracking_version=1,
        )
        self._append(
            state,
            _ReservedEvent(
                sequence=state.sequence + 1,
                event_id=uuid4().hex,
                timestamp=now,
                reservation=reservation,
            ),
        )
        return reservation

    def mark_dispatched(self, reservation_id: str) -> bool:
        with self._locked():
            state = self._load()
            now = self._clock()
            self._expire_stale(state, now)
            if reservation_id not in state.active:
                raise SpendLedgerConflictError(
                    f"unknown active reservation {reservation_id!r}"
                )
            if reservation_id in state.dispatched:
                return False
            self._append(
                state,
                _DispatchStartedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=now,
                    reservation_id=reservation_id,
                ),
            )
            return True

    def authorize_retry(
        self, reservation_id: str, cause: SpendRetryCause
    ) -> SpendRetryAuthorization | SpendRejection:
        with self._locked():
            state = self._load()
            now = self._clock()
            self._expire_stale(state, now)
            if (
                reservation_id not in state.active
                or reservation_id not in state.dispatched
            ):
                raise SpendLedgerConflictError(
                    f"unknown dispatched reservation {reservation_id!r}"
                )
            reservation = state.reservations[reservation_id]
            attempt = self._next_retry_attempt(state, reservation_id)
            rejection = self._retry_budget_denial(state, reservation, now)
            if rejection is not None:
                self._append(
                    state,
                    _RetryBudgetRejectedEvent(
                        sequence=state.sequence + 1,
                        event_id=uuid4().hex,
                        timestamp=now,
                        cause=cause,
                        attempt=attempt,
                        rejection=rejection,
                    ),
                )
                return rejection
            authorization = SpendRetryAuthorization(
                reservation_id=reservation_id,
                call_scope_id=reservation.call_scope_id,
                scope_chain=reservation.scope_chain,
                attempt=attempt,
                cause=cause,
                timestamp=now,
            )
            self._append(
                state,
                _RetryAuthorizedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=now,
                    authorization=authorization,
                ),
            )
            return authorization

    def reject_retry_policy(
        self,
        reservation_id: str,
        cause: SpendRetryCause,
        reason: SpendRetryPolicyReason,
        *,
        elapsed_s: float,
        max_elapsed_s: float,
        next_delay_s: float,
        max_retries: int,
    ) -> None:
        with self._locked():
            state = self._load()
            now = self._clock()
            self._expire_stale(state, now)
            if (
                reservation_id not in state.active
                or reservation_id not in state.dispatched
            ):
                raise SpendLedgerConflictError(
                    f"unknown dispatched reservation {reservation_id!r}"
                )
            event = _RetryPolicyRejectedEvent(
                sequence=state.sequence + 1,
                event_id=uuid4().hex,
                timestamp=now,
                reservation_id=reservation_id,
                cause=cause,
                attempt=self._next_retry_attempt(state, reservation_id),
                reason=reason,
                elapsed_s=elapsed_s,
                max_elapsed_s=max_elapsed_s,
                next_delay_s=next_delay_s,
                max_retries=max_retries,
            )
            self._append(state, event)

    def _known_scope_chain(self, state: _State, scope_id: str) -> tuple[str, ...]:
        if scope_id not in state.scopes:
            return ()
        return tuple(self._scope_chain(state, scope_id))

    @staticmethod
    def _scope_chain(state: _State, scope_id: str) -> list[str]:
        chain: list[str] = []
        current: str | None = scope_id
        while current is not None:
            scope = state.scopes.get(current)
            if scope is None:
                raise SpendLedgerCorruptError("scope chain references unknown scope")
            chain.append(current)
            current = scope.parent_scope_id
        chain.reverse()
        return chain

    def _reservation_rejection(
        self,
        state: _State,
        context: SpendContext,
        estimate: SpendAmount,
        call_id: str,
        chain: tuple[str, ...],
        now: float,
        *,
        prompt_estimate: PromptTokenEstimate | None = None,
    ) -> SpendRejection | None:
        reason: SpendRejectionReason | None = None
        limited_scope_id: str | None = None
        if not chain or state.scopes[chain[-1]].kind != SpendScopeKind.AGENT:
            reason = SpendRejectionReason.UNKNOWN_SCOPE
        elif call_id in state.reservations:
            reason = SpendRejectionReason.DUPLICATE_CALL
            limited_scope_id = context.scope_id
        else:
            limit = self._first_exceeded_limit(state, chain, estimate, context, now)
            if limit is not None:
                reason, limited_scope_id = limit
        if reason is None:
            return None
        return SpendRejection(
            call_id=call_id,
            scope_id=context.scope_id,
            scope_chain=chain,
            purpose=context.purpose,
            estimate=estimate,
            prompt_estimate=prompt_estimate,
            is_retry=context.is_retry,
            reason=reason,
            limited_scope_id=limited_scope_id,
            timestamp=now,
        )

    def _first_exceeded_limit(
        self,
        state: _State,
        chain: tuple[str, ...],
        estimate: SpendAmount,
        context: SpendContext,
        now: float,
    ) -> tuple[SpendRejectionReason, str] | None:
        for scope_id in chain:
            limits = state.scopes[scope_id].limits
            totals = self._totals(state, scope_id)
            reason = self._exceeded_reason(
                limits, totals, estimate, context.is_retry, now
            )
            if reason is not None:
                return reason, scope_id
        return None

    @staticmethod
    def _exceeded_reason(
        limits: SpendEnvelopeLimits,
        totals: _Totals,
        estimate: SpendAmount,
        is_retry: bool,
        now: float,
    ) -> SpendRejectionReason | None:
        projected = _add_amount(_add_amount(totals.spent, totals.reserved), estimate)
        projected_calls = totals.spent_calls + totals.reserved_calls + 1
        projected_retries = (
            totals.spent_retries + totals.reserved_retries + int(is_retry)
        )
        checks = (
            (
                limits.deadline_at is not None and now >= limits.deadline_at,
                SpendRejectionReason.DEADLINE,
            ),
            (
                limits.max_prompt_tokens is not None
                and projected.prompt_tokens > limits.max_prompt_tokens,
                SpendRejectionReason.PROMPT_TOKENS,
            ),
            (
                limits.max_completion_tokens is not None
                and projected.completion_tokens > limits.max_completion_tokens,
                SpendRejectionReason.COMPLETION_TOKENS,
            ),
            (
                limits.max_total_tokens is not None
                and projected.total_tokens > limits.max_total_tokens,
                SpendRejectionReason.TOTAL_TOKENS,
            ),
            (
                limits.max_cost_usd is not None
                and projected.cost_usd > limits.max_cost_usd,
                SpendRejectionReason.COST_USD,
            ),
            (
                limits.max_calls is not None and projected_calls > limits.max_calls,
                SpendRejectionReason.CALLS,
            ),
            (
                limits.max_concurrent_calls is not None
                and totals.reserved_calls + 1 > limits.max_concurrent_calls,
                SpendRejectionReason.CONCURRENT_CALLS,
            ),
            (
                limits.max_retries is not None
                and projected_retries > limits.max_retries,
                SpendRejectionReason.RETRIES,
            ),
        )
        return next((reason for exceeded, reason in checks if exceeded), None)

    def _record_rejection(
        self, state: _State, rejection: SpendRejection, now: float
    ) -> None:
        self._append(
            state,
            _RejectedEvent(
                sequence=state.sequence + 1,
                event_id=uuid4().hex,
                timestamp=now,
                rejection=rejection,
            ),
        )

    def _earliest_deadline(self, state: _State, chain: tuple[str, ...]) -> float | None:
        deadlines: list[float] = []
        for scope_id in chain:
            deadline = state.scopes[scope_id].limits.deadline_at
            if deadline is not None:
                deadlines.append(deadline)
        return min(deadlines) if deadlines else None

    def reconcile(
        self, reservation_id: str, actual: SpendAmount | None
    ) -> SpendSettlement:
        with self._locked():
            state = self._load()
            now = self._clock()
            self._expire_stale(state, now)
            reservation = state.reservations.get(reservation_id)
            if reservation is None:
                raise SpendLedgerConflictError(
                    f"unknown reservation {reservation_id!r}"
                )
            previous = state.settlements.get(reservation_id)
            amount = actual or reservation.estimate
            estimated = actual is None
            if previous is not None:
                return self._reconcile_settled(
                    state, reservation_id, amount, estimated, previous, now
                )
            event = _ReconciledEvent(
                sequence=state.sequence + 1,
                event_id=uuid4().hex,
                timestamp=now,
                reservation_id=reservation_id,
                amount=amount,
                estimated=estimated,
            )
            self._append(state, event)
            return state.settlements[reservation_id]

    def _reconcile_settled(
        self,
        state: _State,
        reservation_id: str,
        amount: SpendAmount,
        estimated: bool,
        previous: SpendSettlement,
        now: float,
    ) -> SpendSettlement:
        if previous.disposition == SpendSettlementDisposition.RELEASED:
            raise SpendLedgerConflictError("released reservation cannot be reconciled")
        if not previous.estimated:
            if previous.amount != amount or estimated:
                raise SpendLedgerConflictError("exact usage already reconciled")
            return previous.model_copy(update={"applied": False})
        if estimated:
            return previous.model_copy(update={"applied": False})
        self._append(
            state,
            _ReconciledEvent(
                sequence=state.sequence + 1,
                event_id=uuid4().hex,
                timestamp=now,
                reservation_id=reservation_id,
                amount=amount,
                estimated=False,
            ),
        )
        return state.settlements[reservation_id]

    def release(self, reservation_id: str, *, reason: str) -> SpendSettlement:
        if not reason.strip():
            raise ValueError("release reason cannot be blank")
        with self._locked():
            state = self._load()
            now = self._clock()
            self._expire_stale(state, now)
            previous = state.settlements.get(reservation_id)
            if previous is not None:
                return previous.model_copy(update={"applied": False})
            if reservation_id not in state.active:
                raise SpendLedgerConflictError(
                    f"unknown reservation {reservation_id!r}"
                )
            self._append(
                state,
                _ReleasedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=now,
                    reservation_id=reservation_id,
                    reason=reason,
                ),
            )
            return state.settlements[reservation_id]

    def renew(self, reservation_id: str, *, lease_s: float) -> bool:
        if lease_s <= 0 or lease_s > MAX_RESERVATION_LEASE_S:
            raise ValueError(f"lease_s must be in (0, {MAX_RESERVATION_LEASE_S:g}]")
        with self._locked():
            state = self._load()
            now = self._clock()
            self._expire_stale(state, now)
            reservation = state.reservations.get(reservation_id)
            if reservation is None or reservation_id not in state.active:
                return False
            chain = reservation.scope_chain[:-1]
            deadline = self._earliest_deadline(state, chain)
            lease_expires_at = max(reservation.lease_expires_at, now + lease_s)
            if deadline is not None:
                lease_expires_at = min(lease_expires_at, deadline)
            self._append(
                state,
                _LeaseRenewedEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=now,
                    reservation_id=reservation_id,
                    lease_expires_at=lease_expires_at,
                ),
            )
            return True

    def reap_expired(self) -> list[SpendSettlement]:
        with self._locked():
            state = self._load()
            expired_ids = self._expire_stale(state, self._clock())
            return [state.settlements[reservation_id] for reservation_id in expired_ids]

    def _expire_stale(self, state: _State, now: float) -> list[str]:
        expired_ids = sorted(
            reservation_id
            for reservation_id in state.active
            if state.reservations[reservation_id].lease_expires_at <= now
        )
        for reservation_id in expired_ids:
            reservation = state.reservations[reservation_id]
            charge_estimate = (
                reservation.dispatch_tracking_version == 0
                or reservation_id in state.dispatched
            )
            self._append(
                state,
                _ExpiredEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=now,
                    reservation_id=reservation_id,
                    charge_estimate=charge_estimate,
                ),
            )
        return expired_ids

    def snapshot(self, scope_id: str) -> SpendEnvelopeSnapshot:
        with self._locked():
            state = self._load()
            self._expire_stale(state, self._clock())
            scope = state.scopes.get(scope_id)
            if scope is None:
                raise SpendLedgerConflictError(f"unknown scope {scope_id!r}")
            totals = self._totals(state, scope_id)
            limits = scope.limits
            used = _add_amount(totals.spent, totals.reserved)
            return SpendEnvelopeSnapshot(
                envelope=scope,
                spent=totals.spent,
                reserved=totals.reserved,
                rejected=totals.rejected,
                spent_calls=totals.spent_calls,
                reserved_calls=totals.reserved_calls,
                rejected_calls=totals.rejected_calls,
                spent_retries=totals.spent_retries,
                reserved_retries=totals.reserved_retries,
                rejected_retries=totals.rejected_retries,
                remaining_prompt_tokens=_remaining(
                    limits.max_prompt_tokens, used.prompt_tokens
                ),
                remaining_completion_tokens=_remaining(
                    limits.max_completion_tokens, used.completion_tokens
                ),
                remaining_total_tokens=_remaining(
                    limits.max_total_tokens, used.total_tokens
                ),
                remaining_cost_usd=_remaining_cost(limits.max_cost_usd, used.cost_usd),
                remaining_calls=_remaining(
                    limits.max_calls, totals.spent_calls + totals.reserved_calls
                ),
                remaining_concurrent_calls=_remaining(
                    limits.max_concurrent_calls, totals.reserved_calls
                ),
                remaining_retries=_remaining(
                    limits.max_retries, totals.spent_retries + totals.reserved_retries
                ),
                deadline_at=limits.deadline_at,
            )

    def _totals(self, state: _State, scope_id: str) -> _Totals:
        totals = _Totals()
        for reservation_id, reservation in state.reservations.items():
            if scope_id not in reservation.scope_chain:
                continue
            if reservation_id in state.active:
                totals.reserved = _add_amount(totals.reserved, reservation.estimate)
                totals.reserved_calls += 1
                totals.reserved_retries += int(reservation.is_retry)
                continue
            settlement = state.settlements.get(reservation_id)
            if (
                settlement is None
                or settlement.disposition == SpendSettlementDisposition.RELEASED
            ):
                continue
            totals.spent = _add_amount(totals.spent, settlement.amount)
            totals.spent_calls += 1
            totals.spent_retries += int(reservation.is_retry)
        for rejection in state.rejections:
            if scope_id not in rejection.scope_chain:
                continue
            totals.rejected = _add_amount(totals.rejected, rejection.estimate)
            totals.rejected_calls += 1
        for authorization in state.retry_authorizations:
            if scope_id in authorization.scope_chain:
                reservation = state.reservations[authorization.reservation_id]
                totals.spent = _add_amount(totals.spent, reservation.estimate)
                totals.spent_retries += 1
        for rejection in state.retry_budget_rejections:
            if scope_id in rejection.scope_chain:
                totals.rejected_retries += 1
        for rejection in state.retry_policy_rejections:
            reservation = state.reservations[rejection.reservation_id]
            if scope_id in reservation.scope_chain:
                totals.rejected_retries += 1
        return totals

    def events(self) -> list[LedgerEvent]:
        with self._locked():
            state = self._load()
            event_paths = sorted(self._events_dir.glob("*.json"))
            if state.sequence != len(event_paths):
                raise SpendLedgerCorruptError("event count changed during read")
            return [
                _EVENT_ADAPTER.validate_json(read_safe(event_path).text)
                for event_path in event_paths
            ]
