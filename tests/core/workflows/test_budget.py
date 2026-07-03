from __future__ import annotations

import asyncio

import pytest

from vibe.core.workflows.budget import Budget, BudgetExhausted, DoubleReconcileError


def test_reserve_drops_remaining_reconcile_recovers() -> None:
    b = Budget(total=100_000, default_reservation=50_000)
    assert b.remaining() == 100_000

    r = b.reserve()
    assert b.remaining() == 50_000

    b.reconcile(r, actual_in=20_000, actual_out=10_000)
    assert b.remaining() == 70_000
    assert b.snapshot().spent == 30_000
    assert b.snapshot().reserved == 0


def test_reserve_past_total_raises() -> None:
    b = Budget(total=100_000, default_reservation=50_000)
    b.reserve()

    with pytest.raises(BudgetExhausted):
        b.reserve(60_000)

    r2 = b.reserve(50_000)
    assert b.remaining() == 0
    b.reconcile(r2, 25_000, 25_000)


def test_double_reconcile_raises() -> None:
    b = Budget(total=100_000)
    r = b.reserve()
    b.reconcile(r, 10_000, 10_000)
    with pytest.raises(DoubleReconcileError):
        b.reconcile(r, 10_000, 10_000)


def test_unlimited_budget_never_raises() -> None:
    b = Budget(total=None)
    assert b.remaining() == float("inf")

    for _ in range(100):
        r = b.reserve(999_999)
        b.reconcile(r, 500, 500)

    assert b.remaining() == float("inf")
    assert b.snapshot().spent == 100_000


def test_overspend_discovered_at_reconcile() -> None:
    b = Budget(total=100_000, default_reservation=10_000)
    r1 = b.reserve(10_000)
    r2 = b.reserve(10_000)
    r3 = b.reserve(10_000)

    b.reconcile(r1, actual_in=30_000, actual_out=20_000)
    assert b.remaining() == 30_000

    b.reconcile(r2, actual_in=30_000, actual_out=20_000)
    assert b.remaining() == -10_000

    with pytest.raises(BudgetExhausted):
        b.reserve(1_000)

    b.reconcile(r3, actual_in=30_000, actual_out=20_000)
    assert b.snapshot().spent == 150_000
    assert b.snapshot().reserved == 0
    assert b.remaining() == -50_000


def test_concurrent_reserve_and_reconcile() -> None:
    async def run() -> None:
        b = Budget(total=500_000, default_reservation=50_000)

        async def spawn_and_finish(i: int) -> int:
            r = b.reserve()
            await asyncio.sleep(0.001)
            b.reconcile(r, actual_in=15_000, actual_out=10_000)
            return i

        results = await asyncio.gather(*[spawn_and_finish(i) for i in range(10)])

        assert sorted(results) == list(range(10))
        assert b.snapshot().spent == 250_000
        assert b.snapshot().reserved == 0
        assert b.remaining() == 250_000
        assert b.agent_count == 10

    asyncio.run(run())


def test_budget_guard_in_loop() -> None:
    budget_floor = 60_000
    b = Budget(total=100_000, default_reservation=30_000)
    rounds = 0

    while b.remaining() > budget_floor:
        rounds += 1
        r = b.reserve(30_000)
        b.reconcile(r, actual_in=10_000, actual_out=10_000)

    assert rounds == 2
    assert b.remaining() == 60_000


def test_negative_estimate_rejected() -> None:
    # A negative estimate must not be accepted: it lowers _reserved and inflates
    # remaining(), letting a workflow author bypass the budget cap.
    b = Budget(total=100_000, default_reservation=50_000)
    b.reserve()
    assert b.remaining() == 50_000
    with pytest.raises(ValueError):
        b.reserve(-1_000_000)
    # remaining() is unchanged — the rejected call did not mutate _reserved.
    assert b.remaining() == 50_000
