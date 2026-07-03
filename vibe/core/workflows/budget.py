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
        if est < 0:
            raise ValueError(f"budget reserve estimate must be non-negative, got {est}")

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
        accounts for tokens already consumed in the paused run.
        """
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


class ReadOnlyBudget:
    """Read-only view of a Budget exposed to workflow scripts.

    The live Budget is mutable (reserve/reconcile/restore_spent mutate
    _spent/_reserved). Injecting it directly let a script reset spend
    (budget._spent = 0) and bypass the cap. Blocking writes on a proxy that
    merely *holds* the Budget was not enough: a script could read it back via
    the proxy's storage attribute (budget._budget._spent = 0) since a single-
    underscore name is not blocked by the script sandbox.

    So this proxy stores no readable reference to the Budget — only bound
    accessor callables. The only paths from those callables back to the live
    Budget are dunder attributes (__self__ / __closure__), which the sandbox's
    AST checks reject, so a script cannot reach the Budget to mutate it.
    """

    __slots__ = (
        "_total_fn",
        "_spent_fn",
        "_remaining_fn",
        "_snapshot_fn",
        "_agent_count_fn",
    )

    def __init__(self, budget: Budget) -> None:
        object.__setattr__(self, "_spent_fn", budget.spent)
        object.__setattr__(self, "_remaining_fn", budget.remaining)
        object.__setattr__(self, "_snapshot_fn", budget.snapshot)
        object.__setattr__(self, "_total_fn", lambda: budget.total)
        object.__setattr__(self, "_agent_count_fn", lambda: budget.agent_count)

    @property
    def total(self) -> int | None:
        return self._total_fn()

    def spent(self) -> int:
        return self._spent_fn()

    def remaining(self) -> int | float:
        return self._remaining_fn()

    def snapshot(self) -> BudgetSnapshot:
        return self._snapshot_fn()

    @property
    def agent_count(self) -> int:
        return self._agent_count_fn()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"budget is read-only in workflow scripts; cannot set attribute {name!r}"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"budget is read-only in workflow scripts; cannot delete attribute {name!r}"
        )
