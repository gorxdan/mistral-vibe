from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Annotated, Literal, final
from uuid import uuid4

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from vibe.core.usage._context import (
    MAX_RESERVATION_LEASE_S,
    SpendAmount,
    SpendContext,
    SpendEnvelope,
    SpendEnvelopeLimits,
    SpendEnvelopeSnapshot,
    SpendRejection,
    SpendRejectionReason,
    SpendReservation,
    SpendScopeKind,
    SpendSettlement,
    SpendSettlementDisposition,
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
class _ReservedEvent(_EventBase):
    kind: Literal["reserved"] = "reserved"
    reservation: SpendReservation


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


@final
class _LeaseRenewedEvent(_EventBase):
    kind: Literal["lease_renewed"] = "lease_renewed"
    reservation_id: str
    lease_expires_at: float


@final
class _RejectedEvent(_EventBase):
    kind: Literal["rejected"] = "rejected"
    rejection: SpendRejection


LedgerEvent = Annotated[
    _ScopeDefinedEvent
    | _ReservedEvent
    | _ReconciledEvent
    | _ReleasedEvent
    | _ExpiredEvent
    | _LeaseRenewedEvent
    | _RejectedEvent,
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


@dataclass(slots=True)
class _State:
    sequence: int = 0
    scopes: dict[str, SpendEnvelope] = field(default_factory=dict)
    reservations: dict[str, SpendReservation] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)
    settlements: dict[str, SpendSettlement] = field(default_factory=dict)
    rejections: list[SpendRejection] = field(default_factory=list)


def _add_amount(left: SpendAmount, right: SpendAmount) -> SpendAmount:
    return SpendAmount(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def _remaining(limit: int | None, used: int) -> int | None:
    return None if limit is None else max(limit - used, 0)


def _remaining_cost(limit: float | None, used: float) -> float | None:
    return None if limit is None else max(limit - used, 0.0)


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
        if isinstance(event, _ScopeDefinedEvent):
            self._apply_scope(state, event.envelope)
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
        if isinstance(event, _LeaseRenewedEvent):
            reservation = self._active_reservation(state, event.reservation_id)
            state.reservations[event.reservation_id] = reservation.model_copy(
                update={"lease_expires_at": event.lease_expires_at}
            )
            return
        if isinstance(event, _ReleasedEvent):
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
            return
        if isinstance(event, _ExpiredEvent):
            reservation = self._active_reservation(state, event.reservation_id)
            state.active.remove(event.reservation_id)
            state.settlements[event.reservation_id] = SpendSettlement(
                reservation_id=event.reservation_id,
                disposition=SpendSettlementDisposition.EXPIRED,
                amount=reservation.estimate,
                estimated=True,
                applied=True,
                timestamp=event.timestamp,
                reason="reservation lease expired",
            )
            return
        if isinstance(event, _ReconciledEvent):
            self._apply_reconciliation(state, event)
            return
        state.rejections.append(event.rejection)

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
            now = self._clock()
            self._expire_stale(state, now)
            call_id = context.call_id or uuid4().hex
            chain = self._known_scope_chain(state, context.scope_id)
            rejection = self._reservation_rejection(
                state, context, estimate, call_id, chain, now
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
                is_retry=context.is_retry,
                created_at=now,
                lease_expires_at=lease_expires_at,
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
            self._append(
                state,
                _ExpiredEvent(
                    sequence=state.sequence + 1,
                    event_id=uuid4().hex,
                    timestamp=now,
                    reservation_id=reservation_id,
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
