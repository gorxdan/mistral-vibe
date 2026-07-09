from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

from vibe.core.usage._context import (
    DEFAULT_RESERVATION_LEASE_S,
    SpendAmount,
    SpendContext,
    SpendEnvelope,
    SpendEnvelopeSnapshot,
    SpendRejection,
    SpendReservation,
    SpendSettlement,
)
from vibe.core.usage._ledger import LedgerEvent, SpendLedger

__all__ = ["SpendBroker"]


class SpendBroker:
    """Transactional admission facade over a shared spend ledger directory."""

    def __init__(
        self,
        ledger_path: Path,
        *,
        lock_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._ledger = SpendLedger(
            ledger_path, lock_timeout_s=lock_timeout_s, clock=clock
        )

    @property
    def ledger_path(self) -> Path:
        return self._ledger.path

    def define_envelope(self, envelope: SpendEnvelope) -> SpendEnvelope:
        return self._ledger.define_envelope(envelope)

    def try_reserve(
        self,
        context: SpendContext,
        estimate: SpendAmount,
        *,
        lease_s: float = DEFAULT_RESERVATION_LEASE_S,
    ) -> SpendReservation | SpendRejection:
        return self._ledger.try_reserve(context, estimate, lease_s=lease_s)

    def reconcile(
        self, reservation: SpendReservation | str, actual: SpendAmount | None
    ) -> SpendSettlement:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, SpendReservation)
            else reservation
        )
        return self._ledger.reconcile(reservation_id, actual)

    def release(
        self, reservation: SpendReservation | str, *, reason: str
    ) -> SpendSettlement:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, SpendReservation)
            else reservation
        )
        return self._ledger.release(reservation_id, reason=reason)

    def renew(self, reservation: SpendReservation | str, *, lease_s: float) -> bool:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, SpendReservation)
            else reservation
        )
        return self._ledger.renew(reservation_id, lease_s=lease_s)

    def reap_expired(self) -> list[SpendSettlement]:
        return self._ledger.reap_expired()

    def snapshot(self, scope_id: str) -> SpendEnvelopeSnapshot:
        return self._ledger.snapshot(scope_id)

    def events(self) -> list[LedgerEvent]:
        return self._ledger.events()
