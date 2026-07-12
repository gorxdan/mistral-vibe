from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

from vibe.core.usage._context import (
    DEFAULT_RESERVATION_LEASE_S,
    SpendAmount,
    SpendContext,
    SpendEnvelope,
    SpendEnvelopeLimits,
    SpendEnvelopeSnapshot,
    SpendRejection,
    SpendReservation,
    SpendRetryAuthorization,
    SpendRetryCause,
    SpendRetryPolicyReason,
    SpendSettlement,
)
from vibe.core.usage._ledger import LedgerEvent, SpendLedger
from vibe.core.usage._prompt_estimator import PromptReservationPlan

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

    def define_or_tighten_envelope(self, envelope: SpendEnvelope) -> SpendEnvelope:
        return self._ledger.define_or_tighten_envelope(envelope)

    def get_envelope(self, scope_id: str) -> SpendEnvelope | None:
        return self._ledger.get_envelope(scope_id)

    def tighten_envelope(
        self, scope_id: str, limits: SpendEnvelopeLimits
    ) -> SpendEnvelope:
        return self._ledger.tighten_envelope(scope_id, limits)

    def replace_envelope_limits(
        self, scope_id: str, limits: SpendEnvelopeLimits
    ) -> SpendEnvelope:
        return self._ledger.replace_envelope_limits(scope_id, limits)

    def migrate_legacy_default_token_limits(
        self,
        scope_id: str,
        *,
        clear_prompt_tokens: bool,
        clear_completion_tokens: bool,
        clear_total_tokens: bool,
    ) -> SpendEnvelope:
        return self._ledger.migrate_legacy_default_token_limits(
            scope_id,
            clear_prompt_tokens=clear_prompt_tokens,
            clear_completion_tokens=clear_completion_tokens,
            clear_total_tokens=clear_total_tokens,
        )

    def try_reserve(
        self,
        context: SpendContext,
        estimate: SpendAmount,
        *,
        lease_s: float = DEFAULT_RESERVATION_LEASE_S,
    ) -> SpendReservation | SpendRejection:
        return self._ledger.try_reserve(context, estimate, lease_s=lease_s)

    def try_reserve_prompt(
        self,
        context: SpendContext,
        plan: PromptReservationPlan,
        *,
        lease_s: float = DEFAULT_RESERVATION_LEASE_S,
    ) -> SpendReservation | SpendRejection:
        return self._ledger.try_reserve_prompt(context, plan, lease_s=lease_s)

    def reconcile(
        self,
        reservation: SpendReservation | str,
        actual: SpendAmount | None,
        *,
        estimated: bool | None = None,
    ) -> SpendSettlement:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, SpendReservation)
            else reservation
        )
        return self._ledger.reconcile(reservation_id, actual, estimated=estimated)

    def mark_dispatched(self, reservation: SpendReservation | str) -> bool:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, SpendReservation)
            else reservation
        )
        return self._ledger.mark_dispatched(reservation_id)

    def authorize_retry(
        self, reservation: SpendReservation | str, cause: SpendRetryCause
    ) -> SpendRetryAuthorization | SpendRejection:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, SpendReservation)
            else reservation
        )
        return self._ledger.authorize_retry(reservation_id, cause)

    def reject_retry_policy(
        self,
        reservation: SpendReservation | str,
        cause: SpendRetryCause,
        reason: SpendRetryPolicyReason,
        *,
        elapsed_s: float,
        max_elapsed_s: float,
        next_delay_s: float,
        max_retries: int,
    ) -> None:
        reservation_id = (
            reservation.reservation_id
            if isinstance(reservation, SpendReservation)
            else reservation
        )
        self._ledger.reject_retry_policy(
            reservation_id,
            cause,
            reason,
            elapsed_s=elapsed_s,
            max_elapsed_s=max_elapsed_s,
            next_delay_s=next_delay_s,
            max_retries=max_retries,
        )

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
