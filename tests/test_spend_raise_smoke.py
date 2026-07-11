from __future__ import annotations

from pathlib import Path

from vibe.core.usage._context import SpendEnvelope, SpendEnvelopeLimits, SpendScopeKind
from vibe.core.usage._ledger import SpendLedger


def test_replace_envelope_limits_can_raise(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path)
    ledger.define_envelope(
        SpendEnvelope(scope_id="session:t", kind=SpendScopeKind.SESSION)
    )
    ledger.tighten_envelope(
        "session:t", SpendEnvelopeLimits(max_cost_usd=5.0, max_calls=10)
    )
    ledger.replace_envelope_limits(
        "session:t", SpendEnvelopeLimits(max_cost_usd=50.0, max_calls=500)
    )
    env = ledger.get_envelope("session:t")
    assert env is not None
    assert env.limits.max_cost_usd == 50.0
    assert env.limits.max_calls == 500


def test_replaced_limits_persist_across_reload(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path)
    ledger.define_envelope(
        SpendEnvelope(scope_id="session:t", kind=SpendScopeKind.SESSION)
    )
    ledger.replace_envelope_limits(
        "session:t", SpendEnvelopeLimits(max_cost_usd=100.0, max_calls=1000)
    )
    reloaded_env = SpendLedger(tmp_path).get_envelope("session:t")
    assert reloaded_env is not None
    assert reloaded_env.limits.max_cost_usd == 100.0
    assert reloaded_env.limits.max_calls == 1000


def test_tighten_still_cannot_raise(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path)
    ledger.define_envelope(
        SpendEnvelope(scope_id="session:t", kind=SpendScopeKind.SESSION)
    )
    ledger.tighten_envelope("session:t", SpendEnvelopeLimits(max_cost_usd=5.0))
    ledger.tighten_envelope("session:t", SpendEnvelopeLimits(max_cost_usd=50.0))
    env = ledger.get_envelope("session:t")
    assert env is not None
    assert env.limits.max_cost_usd == 5.0
