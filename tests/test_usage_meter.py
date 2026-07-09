from __future__ import annotations

from pydantic import ValidationError
import pytest

from vibe.core.config import AuxiliaryBudgetConfig, ModelConfig, ProviderConfig
from vibe.core.types import Backend, LLMUsage
from vibe.core.usage import (
    CallKind,
    SpendLimits,
    UsageMeter,
    UsageRecord,
    UsageRecorder,
)


def _model(*, input_price: float = 0.0, output_price: float = 0.0) -> ModelConfig:
    return ModelConfig(
        name="meter-test",
        provider="test",
        alias="meter-test",
        input_price=input_price,
        output_price=output_price,
    )


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="test", api_base="https://example.test", backend=Backend.GENERIC
    )


def test_auxiliary_budget_has_finite_defaults() -> None:
    budget = AuxiliaryBudgetConfig()

    assert budget.max_tokens == 50_000
    assert budget.max_calls == 24
    assert budget.max_cost_usd == 1.0


@pytest.mark.parametrize("field", ["max_tokens", "max_calls", "max_cost_usd"])
def test_auxiliary_budget_rejects_negative_limits(field: str) -> None:
    with pytest.raises(ValidationError):
        AuxiliaryBudgetConfig.model_validate({field: -1})


def test_usage_record_reads_legacy_shape() -> None:
    record = UsageRecord.model_validate({
        "timestamp": 1.0,
        "provider": "test",
        "model": "meter-test",
    })

    assert record.call_kind == "main"
    assert record.result_used is None


def test_meter_enforces_token_and_call_reservations(tmp_path) -> None:
    meter = UsageMeter(
        "session",
        limits=SpendLimits(max_tokens=100, max_calls=1),
        recorder=UsageRecorder(tmp_path / "usage.jsonl"),
    )

    reservation = meter.try_reserve(80)
    assert reservation is not None
    assert meter.try_reserve(1) is None
    assert meter.snapshot().reserved_tokens == 80

    meter.release(reservation)

    assert meter.snapshot().calls == 0
    assert meter.try_reserve(101) is None
    assert meter.try_reserve(100) is not None


def test_meter_reserves_projected_cost_before_dispatch(tmp_path) -> None:
    meter = UsageMeter(
        "session",
        limits=SpendLimits(max_cost_usd=0.001),
        recorder=UsageRecorder(tmp_path / "usage.jsonl"),
    )

    assert meter.try_reserve(10, estimated_cost_usd=0.0011) is None
    reservation = meter.try_reserve(10, estimated_cost_usd=0.0006)
    assert reservation is not None
    assert meter.try_reserve(10, estimated_cost_usd=0.0005) is None
    assert meter.snapshot().reserved_cost_usd == 0.0006

    meter.release(reservation)

    assert meter.snapshot().reserved_cost_usd == 0.0


def test_meter_reconciles_usage_and_persists_attribution(tmp_path) -> None:
    recorder = UsageRecorder(tmp_path / "usage.jsonl")
    meter = UsageMeter("session", recorder=recorder)
    reservation = meter.try_reserve(500)
    assert reservation is not None

    meter.reconcile(
        reservation,
        usage=LLMUsage(prompt_tokens=100, completion_tokens=20),
        model=_model(input_price=2.0, output_price=4.0),
        provider=_provider(),
        call_kind=CallKind.MEMORY_SELECT,
        duration_s=0.25,
        result_used=True,
    )

    snapshot = meter.snapshot()
    assert snapshot.tokens == 120
    assert snapshot.calls == 1
    assert snapshot.reserved_tokens == 0
    assert snapshot.cost_usd == 0.00028

    records = recorder.read_all()
    assert len(records) == 1
    assert records[0].harness is True
    assert records[0].call_kind == "memory_select"
    assert records[0].result_used is True


def test_meter_charges_estimate_when_usage_is_unavailable(tmp_path) -> None:
    recorder = UsageRecorder(tmp_path / "usage.jsonl")
    meter = UsageMeter("session", recorder=recorder)
    reservation = meter.try_reserve(75)
    assert reservation is not None

    meter.reconcile(
        reservation,
        usage=None,
        model=_model(),
        provider=_provider(),
        call_kind=CallKind.SAFETY_JUDGE,
        duration_s=0.1,
    )

    assert meter.snapshot().tokens == 75
    assert recorder.read_all()[0].prompt_tokens == 75


def test_reservation_keeps_session_attribution_across_rebind(tmp_path) -> None:
    recorder = UsageRecorder(tmp_path / "usage.jsonl")
    meter = UsageMeter("session-a", recorder=recorder)
    reservation_a = meter.try_reserve(10)
    assert reservation_a is not None

    meter.rebind_session("session-b")
    reservation_b = meter.try_reserve(10)
    assert reservation_b is not None
    for reservation in (reservation_a, reservation_b):
        meter.reconcile(
            reservation,
            usage=LLMUsage(prompt_tokens=5),
            model=_model(),
            provider=_provider(),
            call_kind=CallKind.MEMORY_SELECT,
            duration_s=0.1,
        )

    assert [record.session_id for record in recorder.read_all()] == [
        "session-a",
        "session-b",
    ]
