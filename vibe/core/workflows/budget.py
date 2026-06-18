from __future__ import annotations

from dataclasses import dataclass, field

from vibe.core.workflows.models import BudgetSnapshot


class BudgetExhausted(Exception):
    pass


class DoubleReconcileError(Exception):
    pass


@dataclass
class Reservation:
    estimate: int
    reconciled: bool = False


@dataclass
class Budget:
    total: int | None
    default_reservation: int = 50_000
    _reserved: int = field(default=0, init=False, repr=False)
    _spent: int = field(default=0, init=False, repr=False)
    _agent_count: int = field(default=0, init=False, repr=False)

    def reserve(self, estimate: int | None = None) -> Reservation:
        est = estimate if estimate is not None else self.default_reservation

        if self.total is not None:
            remaining = self.total - self._reserved - self._spent
            if remaining - est < 0:
                raise BudgetExhausted(
                    f"Cannot reserve {est}: remaining {remaining}, "
                    f"would go to {remaining - est}"
                )

        self._reserved += est
        self._agent_count += 1
        return Reservation(estimate=est)

    def reconcile(
        self, reservation: Reservation, actual_in: int, actual_out: int
    ) -> None:
        if reservation.reconciled:
            raise DoubleReconcileError("Reservation already reconciled")

        actual = actual_in + actual_out
        reservation.reconciled = True
        self._reserved -= reservation.estimate
        self._spent += actual

    def restore_spent(self, spent: int) -> None:
        """Restore prior spend when resuming from a snapshot, so the cap still
        accounts for tokens already consumed in the paused run."""
        self._spent = spent

    def spent(self) -> int:
        return self._spent

    def remaining(self) -> int | float:
        if self.total is None:
            return float("inf")
        return self.total - self._reserved - self._spent

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            total=self.total, reserved=self._reserved, spent=self._spent
        )

    @property
    def agent_count(self) -> int:
        return self._agent_count
