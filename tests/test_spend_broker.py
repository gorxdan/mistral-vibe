from __future__ import annotations

import multiprocessing
from pathlib import Path

import orjson
from pydantic import ValidationError
import pytest

from vibe.core.usage._broker import SpendBroker
from vibe.core.usage._context import (
    SpendAmount,
    SpendContext,
    SpendEnvelope,
    SpendEnvelopeLimits,
    SpendPurpose,
    SpendRejection,
    SpendRejectionReason,
    SpendReservation,
    SpendScopeKind,
    SpendSettlementDisposition,
)
from vibe.core.usage._ledger import SpendLedgerConflictError, SpendLedgerCorruptError
from vibe.core.utils.io import read_safe, write_safe


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _define_agent(
    broker: SpendBroker,
    *,
    session_limits: SpendEnvelopeLimits | None = None,
    agent_limits: SpendEnvelopeLimits | None = None,
    agent_id: str = "agent",
    group_kind: SpendScopeKind = SpendScopeKind.WORKFLOW,
) -> None:
    broker.define_envelope(
        SpendEnvelope(
            scope_id="session",
            kind=SpendScopeKind.SESSION,
            limits=session_limits or SpendEnvelopeLimits(),
        )
    )
    broker.define_envelope(
        SpendEnvelope(scope_id="group", kind=group_kind, parent_scope_id="session")
    )
    broker.define_envelope(
        SpendEnvelope(
            scope_id=agent_id,
            kind=SpendScopeKind.AGENT,
            parent_scope_id="group",
            limits=agent_limits or SpendEnvelopeLimits(),
        )
    )


def _context(
    call_id: str, *, agent_id: str = "agent", is_retry: bool = False
) -> SpendContext:
    return SpendContext(
        scope_id=agent_id,
        purpose=SpendPurpose.PRIMARY,
        call_id=call_id,
        is_retry=is_retry,
    )


def _reserve_worker(ledger_path: str, start, results, call_id: str) -> None:
    broker = SpendBroker(Path(ledger_path))
    start.wait(timeout=10)
    decision = broker.try_reserve(
        _context(call_id), SpendAmount(prompt_tokens=60), lease_s=60
    )
    if isinstance(decision, SpendReservation):
        results.put((True, None))
        return
    results.put((False, decision.reason.value))


def test_hierarchy_charges_session_group_agent_and_call(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    _define_agent(
        broker,
        session_limits=SpendEnvelopeLimits(max_total_tokens=100),
        agent_limits=SpendEnvelopeLimits(max_total_tokens=80),
    )

    reservation = broker.try_reserve(
        _context("call-1"),
        SpendAmount(prompt_tokens=40, completion_tokens=20, cost_usd=0.25),
    )

    assert isinstance(reservation, SpendReservation)
    assert reservation.scope_chain == ("session", "group", "agent", "call:call-1")
    for scope_id in ("session", "group", "agent"):
        snapshot = broker.snapshot(scope_id)
        assert snapshot.reserved.total_tokens == 60
        assert snapshot.reserved_calls == 1

    rejection = broker.try_reserve(_context("call-2"), SpendAmount(prompt_tokens=21))
    assert isinstance(rejection, SpendRejection)
    assert rejection.reason == SpendRejectionReason.TOTAL_TOKENS
    assert rejection.limited_scope_id == "agent"
    assert broker.snapshot("agent").rejected.prompt_tokens == 21


@pytest.mark.parametrize(
    ("limits", "estimate", "is_retry", "reason"),
    [
        (
            SpendEnvelopeLimits(max_prompt_tokens=9),
            SpendAmount(prompt_tokens=10),
            False,
            SpendRejectionReason.PROMPT_TOKENS,
        ),
        (
            SpendEnvelopeLimits(max_completion_tokens=9),
            SpendAmount(completion_tokens=10),
            False,
            SpendRejectionReason.COMPLETION_TOKENS,
        ),
        (
            SpendEnvelopeLimits(max_total_tokens=9),
            SpendAmount(prompt_tokens=5, completion_tokens=5),
            False,
            SpendRejectionReason.TOTAL_TOKENS,
        ),
        (
            SpendEnvelopeLimits(max_cost_usd=0.09),
            SpendAmount(cost_usd=0.1),
            False,
            SpendRejectionReason.COST_USD,
        ),
        (
            SpendEnvelopeLimits(max_calls=0),
            SpendAmount(),
            False,
            SpendRejectionReason.CALLS,
        ),
        (
            SpendEnvelopeLimits(max_concurrent_calls=0),
            SpendAmount(),
            False,
            SpendRejectionReason.CONCURRENT_CALLS,
        ),
        (
            SpendEnvelopeLimits(max_retries=0),
            SpendAmount(),
            True,
            SpendRejectionReason.RETRIES,
        ),
    ],
)
def test_each_capacity_limit_rejects_before_dispatch(
    tmp_path,
    limits: SpendEnvelopeLimits,
    estimate: SpendAmount,
    is_retry: bool,
    reason: SpendRejectionReason,
) -> None:
    broker = SpendBroker(tmp_path / reason.value)
    _define_agent(broker, session_limits=limits)

    decision = broker.try_reserve(_context("limited", is_retry=is_retry), estimate)

    assert isinstance(decision, SpendRejection)
    assert decision.reason == reason
    assert decision.limited_scope_id == "session"


def test_parent_cap_rejects_sibling_agent_spend(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    _define_agent(
        broker,
        session_limits=SpendEnvelopeLimits(max_total_tokens=100),
        agent_limits=SpendEnvelopeLimits(max_total_tokens=100),
        agent_id="agent-a",
        group_kind=SpendScopeKind.TEAM,
    )
    broker.define_envelope(
        SpendEnvelope(
            scope_id="agent-b",
            kind=SpendScopeKind.AGENT,
            parent_scope_id="group",
            limits=SpendEnvelopeLimits(max_total_tokens=100),
        )
    )
    first = broker.try_reserve(
        _context("call-a", agent_id="agent-a"), SpendAmount(prompt_tokens=60)
    )

    second = broker.try_reserve(
        _context("call-b", agent_id="agent-b"), SpendAmount(prompt_tokens=60)
    )

    assert isinstance(first, SpendReservation)
    assert isinstance(second, SpendRejection)
    assert second.limited_scope_id == "session"


def test_release_returns_capacity_without_counting_a_call(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    _define_agent(
        broker, session_limits=SpendEnvelopeLimits(max_total_tokens=10, max_calls=1)
    )
    reservation = broker.try_reserve(_context("aborted"), SpendAmount(prompt_tokens=10))
    assert isinstance(reservation, SpendReservation)

    settlement = broker.release(reservation, reason="dispatch was cancelled")
    replacement = broker.try_reserve(
        _context("replacement"), SpendAmount(prompt_tokens=10)
    )

    assert settlement.disposition == SpendSettlementDisposition.RELEASED
    assert isinstance(replacement, SpendReservation)
    snapshot = broker.snapshot("session")
    assert snapshot.spent_calls == 0
    assert snapshot.reserved_calls == 1


def test_exact_and_estimated_reconciliation_are_idempotent(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    _define_agent(broker)
    exact_reservation = broker.try_reserve(
        _context("exact"), SpendAmount(prompt_tokens=100, cost_usd=1.0)
    )
    estimated_reservation = broker.try_reserve(
        _context("estimated"), SpendAmount(prompt_tokens=40, cost_usd=0.4)
    )
    assert isinstance(exact_reservation, SpendReservation)
    assert isinstance(estimated_reservation, SpendReservation)

    exact = SpendAmount(prompt_tokens=20, completion_tokens=5, cost_usd=0.2)
    first = broker.reconcile(exact_reservation, exact)
    duplicate = broker.reconcile(exact_reservation, exact)
    estimated = broker.reconcile(estimated_reservation, None)
    corrected = broker.reconcile(
        estimated_reservation,
        SpendAmount(prompt_tokens=10, completion_tokens=5, cost_usd=0.1),
    )

    assert first.applied is True
    assert duplicate.applied is False
    assert estimated.estimated is True
    assert corrected.estimated is False
    snapshot = broker.snapshot("session")
    assert snapshot.spent.prompt_tokens == 30
    assert snapshot.spent.completion_tokens == 10
    assert snapshot.spent.cost_usd == pytest.approx(0.3)
    assert snapshot.spent_calls == 2
    with pytest.raises(SpendLedgerConflictError, match="exact usage"):
        broker.reconcile(exact_reservation, SpendAmount(prompt_tokens=21))


def test_expired_lease_commits_estimate_and_accepts_late_exact_usage(tmp_path) -> None:
    clock = _Clock(100.0)
    broker = SpendBroker(tmp_path / "ledger", clock=clock)
    _define_agent(
        broker,
        session_limits=SpendEnvelopeLimits(
            max_total_tokens=100, max_concurrent_calls=1
        ),
    )
    reservation = broker.try_reserve(
        _context("stale"), SpendAmount(prompt_tokens=60), lease_s=5
    )
    assert isinstance(reservation, SpendReservation)
    assert broker.mark_dispatched(reservation) is True
    assert broker.mark_dispatched(reservation) is False

    clock.now = 106.0
    expired = broker.reap_expired()
    replacement = broker.try_reserve(
        _context("replacement"), SpendAmount(prompt_tokens=40), lease_s=5
    )

    assert expired[0].disposition == SpendSettlementDisposition.EXPIRED
    assert expired[0].estimated is True
    assert isinstance(replacement, SpendReservation)
    assert broker.snapshot("session").remaining_total_tokens == 0

    actual = SpendAmount(prompt_tokens=20)
    corrected = broker.reconcile(reservation, actual)
    duplicate = broker.reconcile(reservation, actual)
    assert corrected.applied is True
    assert corrected.estimated is False
    assert duplicate.applied is False
    snapshot = broker.snapshot("session")
    assert snapshot.spent.prompt_tokens == 20
    assert snapshot.reserved.prompt_tokens == 40
    assert snapshot.remaining_total_tokens == 40
    assert any(event.kind == "expired" for event in broker.events())


def test_expired_undispatched_lease_releases_reserved_capacity(tmp_path) -> None:
    clock = _Clock(100.0)
    broker = SpendBroker(tmp_path / "ledger", clock=clock)
    _define_agent(
        broker, session_limits=SpendEnvelopeLimits(max_total_tokens=10, max_calls=1)
    )
    reservation = broker.try_reserve(
        _context("never-dispatched"), SpendAmount(prompt_tokens=10), lease_s=5
    )
    assert isinstance(reservation, SpendReservation)

    clock.now = 106.0
    expired = broker.reap_expired()
    replacement = broker.try_reserve(
        _context("replacement"), SpendAmount(prompt_tokens=10), lease_s=5
    )

    assert expired[0].disposition == SpendSettlementDisposition.RELEASED
    assert expired[0].amount == SpendAmount()
    assert expired[0].estimated is False
    assert expired[0].reason == "undispatched reservation lease expired"
    assert isinstance(replacement, SpendReservation)


def test_legacy_untracked_lease_expiry_remains_conservative(tmp_path) -> None:
    clock = _Clock(100.0)
    ledger_path = tmp_path / "ledger"
    broker = SpendBroker(ledger_path, clock=clock)
    _define_agent(broker)
    reservation = broker.try_reserve(
        _context("legacy"), SpendAmount(prompt_tokens=10), lease_s=5
    )
    assert isinstance(reservation, SpendReservation)
    reserved_path = next(
        path
        for path in sorted((ledger_path / "events").glob("*.json"))
        if orjson.loads(read_safe(path).text).get("kind") == "reserved"
    )
    payload = orjson.loads(read_safe(reserved_path).text)
    payload["reservation"].pop("dispatch_tracking_version")
    write_safe(reserved_path, orjson.dumps(payload).decode())

    clock.now = 106.0
    expired = SpendBroker(ledger_path, clock=clock).reap_expired()

    assert expired[0].disposition == SpendSettlementDisposition.EXPIRED
    assert expired[0].amount == reservation.estimate
    assert expired[0].estimated is True


def test_renewal_and_deadline_bound_the_lease(tmp_path) -> None:
    clock = _Clock(100.0)
    broker = SpendBroker(tmp_path / "ledger", clock=clock)
    _define_agent(broker, session_limits=SpendEnvelopeLimits(deadline_at=120.0))
    reservation = broker.try_reserve(
        _context("renewed"), SpendAmount(prompt_tokens=1), lease_s=10
    )
    assert isinstance(reservation, SpendReservation)

    clock.now = 105.0
    assert broker.renew(reservation, lease_s=30) is True
    renewal = [event for event in broker.events() if event.kind == "lease_renewed"]
    assert renewal[0].lease_expires_at == 120.0

    clock.now = 120.0
    rejection = broker.try_reserve(_context("too-late"), SpendAmount(prompt_tokens=1))
    assert isinstance(rejection, SpendRejection)
    assert rejection.reason == SpendRejectionReason.DEADLINE


def test_reopen_replays_durable_events_and_corruption_fails_closed(tmp_path) -> None:
    ledger_path = tmp_path / "ledger"
    broker = SpendBroker(ledger_path)
    _define_agent(broker)
    reservation = broker.try_reserve(_context("durable"), SpendAmount(prompt_tokens=12))
    assert isinstance(reservation, SpendReservation)
    broker.reconcile(reservation, SpendAmount(prompt_tokens=7))

    reopened = SpendBroker(ledger_path)
    assert reopened.snapshot("session").spent.prompt_tokens == 7

    first_event = ledger_path / "events" / "00000000000000000001.json"
    write_safe(first_event, "not-json")
    with pytest.raises(SpendLedgerCorruptError):
        reopened.snapshot("session")


def test_conflicting_scope_definition_is_rejected(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    scope = SpendEnvelope(scope_id="session", kind=SpendScopeKind.SESSION)
    assert broker.define_envelope(scope) == scope
    assert broker.define_envelope(scope) == scope

    with pytest.raises(SpendLedgerConflictError, match="different definition"):
        broker.define_envelope(
            SpendEnvelope(
                scope_id="session",
                kind=SpendScopeKind.SESSION,
                limits=SpendEnvelopeLimits(max_calls=1),
            )
        )


def test_legacy_default_token_limit_migration_is_durable_and_idempotent(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "ledger"
    broker = SpendBroker(ledger_path, clock=_Clock(100.0))
    _define_agent(
        broker,
        session_limits=SpendEnvelopeLimits(
            max_prompt_tokens=400_000,
            max_completion_tokens=100_000,
            max_total_tokens=500_000,
            max_cost_usd=3.0,
            max_calls=7,
            max_concurrent_calls=2,
            max_retries=4,
            deadline_at=999.0,
        ),
    )
    reservation = broker.try_reserve(
        _context("before-migration"), SpendAmount(prompt_tokens=12)
    )
    assert isinstance(reservation, SpendReservation)
    broker.reconcile(reservation, SpendAmount(prompt_tokens=7))

    migrated = broker.migrate_legacy_default_token_limits(
        "session",
        clear_prompt_tokens=True,
        clear_completion_tokens=True,
        clear_total_tokens=True,
    )
    event_count = len(broker.events())
    repeated = broker.migrate_legacy_default_token_limits(
        "session",
        clear_prompt_tokens=True,
        clear_completion_tokens=True,
        clear_total_tokens=True,
    )

    assert migrated == repeated
    assert migrated.policy_version == 2
    assert len(broker.events()) == event_count
    assert migrated.limits == SpendEnvelopeLimits(
        max_cost_usd=3.0,
        max_calls=7,
        max_concurrent_calls=2,
        max_retries=4,
        deadline_at=999.0,
    )
    migrations = [
        event for event in broker.events() if event.kind == "envelope_policy_migrated"
    ]
    assert len(migrations) == 1
    assert migrations[0].from_policy_version == 1
    assert migrations[0].to_policy_version == 2
    assert migrations[0].cleared_fields == (
        "max_prompt_tokens",
        "max_completion_tokens",
        "max_total_tokens",
    )

    scope_path = ledger_path / "events" / "00000000000000000001.json"
    scope_payload = orjson.loads(read_safe(scope_path).text)
    scope_payload["envelope"].pop("policy_version")
    write_safe(scope_path, orjson.dumps(scope_payload).decode())

    reopened = SpendBroker(ledger_path)
    assert reopened.get_envelope("session") == migrated
    snapshot = reopened.snapshot("session")
    assert snapshot.spent.prompt_tokens == 7
    assert snapshot.remaining_prompt_tokens is None
    assert snapshot.remaining_completion_tokens is None
    assert snapshot.remaining_total_tokens is None


def test_legacy_migration_preserves_unselected_explicit_token_limit(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    broker.define_envelope(
        SpendEnvelope(
            scope_id="session",
            kind=SpendScopeKind.SESSION,
            limits=SpendEnvelopeLimits(
                max_prompt_tokens=400_000,
                max_completion_tokens=75_000,
                max_total_tokens=500_000,
            ),
        )
    )

    with pytest.raises(SpendLedgerConflictError, match="max_completion_tokens"):
        broker.migrate_legacy_default_token_limits(
            "session",
            clear_prompt_tokens=True,
            clear_completion_tokens=True,
            clear_total_tokens=True,
        )
    unchanged = broker.get_envelope("session")
    assert unchanged is not None
    assert unchanged.limits.max_prompt_tokens == 400_000
    assert unchanged.limits.max_completion_tokens == 75_000
    assert unchanged.limits.max_total_tokens == 500_000

    migrated = broker.migrate_legacy_default_token_limits(
        "session",
        clear_prompt_tokens=True,
        clear_completion_tokens=False,
        clear_total_tokens=True,
    )
    assert migrated.limits.max_prompt_tokens is None
    assert migrated.limits.max_completion_tokens == 75_000
    assert migrated.limits.max_total_tokens is None


def test_legacy_migration_is_session_only_and_cannot_run_twice(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    _define_agent(
        broker,
        session_limits=SpendEnvelopeLimits(
            max_prompt_tokens=400_000,
            max_completion_tokens=100_000,
            max_total_tokens=500_000,
        ),
        agent_limits=SpendEnvelopeLimits(max_prompt_tokens=400_000),
    )

    with pytest.raises(SpendLedgerConflictError, match="session scope"):
        broker.migrate_legacy_default_token_limits(
            "agent",
            clear_prompt_tokens=True,
            clear_completion_tokens=False,
            clear_total_tokens=False,
        )

    broker.migrate_legacy_default_token_limits(
        "session",
        clear_prompt_tokens=True,
        clear_completion_tokens=False,
        clear_total_tokens=False,
    )
    with pytest.raises(SpendLedgerConflictError, match="already applied"):
        broker.migrate_legacy_default_token_limits(
            "session",
            clear_prompt_tokens=False,
            clear_completion_tokens=True,
            clear_total_tokens=False,
        )


def test_new_policy_envelope_cannot_be_mistaken_for_legacy_default(tmp_path) -> None:
    broker = SpendBroker(tmp_path / "ledger")
    envelope = SpendEnvelope(
        scope_id="session",
        kind=SpendScopeKind.SESSION,
        policy_version=2,
        limits=SpendEnvelopeLimits(max_prompt_tokens=400_000),
    )
    assert broker.define_envelope(envelope) == envelope
    assert broker.define_envelope(envelope) == envelope

    with pytest.raises(SpendLedgerConflictError, match="policy version 1"):
        broker.migrate_legacy_default_token_limits(
            "session",
            clear_prompt_tokens=True,
            clear_completion_tokens=False,
            clear_total_tokens=False,
        )


def test_corrupt_legacy_migration_cannot_change_unrelated_limits(tmp_path) -> None:
    ledger_path = tmp_path / "ledger"
    broker = SpendBroker(ledger_path)
    broker.define_envelope(
        SpendEnvelope(
            scope_id="session",
            kind=SpendScopeKind.SESSION,
            limits=SpendEnvelopeLimits(max_prompt_tokens=400_000, max_cost_usd=3.0),
        )
    )
    broker.migrate_legacy_default_token_limits(
        "session",
        clear_prompt_tokens=True,
        clear_completion_tokens=False,
        clear_total_tokens=False,
    )
    migration_path = ledger_path / "events" / "00000000000000000002.json"
    payload = orjson.loads(read_safe(migration_path).text)
    payload["limits"]["max_cost_usd"] = 4.0
    write_safe(migration_path, orjson.dumps(payload).decode())

    with pytest.raises(SpendLedgerCorruptError, match="unrelated limits"):
        SpendBroker(ledger_path).get_envelope("session")


def test_envelope_cannot_collide_with_transient_call_scope() -> None:
    with pytest.raises(ValidationError, match="reserved call"):
        SpendEnvelope(scope_id="call:other", kind=SpendScopeKind.AGENT)


@pytest.mark.parametrize(
    ("session_limits", "rejection_reason"),
    [
        (SpendEnvelopeLimits(max_total_tokens=100), SpendRejectionReason.TOTAL_TOKENS),
        (
            SpendEnvelopeLimits(max_total_tokens=1_000, max_concurrent_calls=1),
            SpendRejectionReason.CONCURRENT_CALLS,
        ),
    ],
)
def test_cross_process_reservations_share_parent_limits(
    tmp_path,
    session_limits: SpendEnvelopeLimits,
    rejection_reason: SpendRejectionReason,
) -> None:
    ledger_path = tmp_path / "ledger"
    broker = SpendBroker(ledger_path)
    _define_agent(
        broker,
        session_limits=session_limits,
        agent_limits=SpendEnvelopeLimits(max_total_tokens=100),
    )
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_reserve_worker,
            args=(str(ledger_path), start, results, f"process-{index}"),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    assert sorted(accepted for accepted, _reason in outcomes) == [False, True]
    assert [reason for accepted, reason in outcomes if not accepted] == [
        rejection_reason.value
    ]
    snapshot = broker.snapshot("session")
    assert snapshot.reserved.prompt_tokens == 60
    assert snapshot.rejected.prompt_tokens == 60
