from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum, auto
import threading
import time

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.logger import logger
from vibe.core.types import LLMUsage
from vibe.core.usage._pricing_policy import (
    CostQuote,
    quote_cold_reservation,
    quote_usage,
)
from vibe.core.usage._recorder import UsageRecorder, get_usage_recorder
from vibe.core.usage.models import UsageRecord

_DEFAULT_UNPRICED_INPUT_USD_PER_MILLION = 10.0
_DEFAULT_UNPRICED_OUTPUT_USD_PER_MILLION = 30.0


class CallKind(StrEnum):
    MAIN = auto()
    SUBAGENT = auto()
    COMPACTION = auto()
    MEMORY_SELECT = auto()
    MEMORY_EXTRACT = auto()
    MEMORY_CONSOLIDATE = auto()
    MEMORY_VERIFY = auto()
    SAFETY_JUDGE = auto()
    NARRATOR = auto()


@dataclass(frozen=True, slots=True)
class SpendLimits:
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_calls: int | None = None


@dataclass(frozen=True, slots=True)
class SpendSnapshot:
    tokens: int
    cost_usd: float
    calls: int
    reserved_tokens: int
    reserved_cost_usd: float


@dataclass(slots=True)
class UsageReservation:
    estimated_tokens: int
    session_id: str
    estimated_cost_usd: float = 0.0
    active: bool = True


def usage_cost(
    model: ModelConfig,
    usage: LLMUsage,
    *,
    unpriced_input_usd_per_million: float = _DEFAULT_UNPRICED_INPUT_USD_PER_MILLION,
    unpriced_output_usd_per_million: float = _DEFAULT_UNPRICED_OUTPUT_USD_PER_MILLION,
) -> float:
    return quote_usage(
        model,
        usage,
        unpriced_input_price=max(unpriced_input_usd_per_million, 0.0),
        unpriced_output_price=max(unpriced_output_usd_per_million, 0.0),
    ).cost_usd


class UsageMeter:
    """Process-local call admission and persisted usage accounting.

    A meter can be shared by a host loop and its in-process children. The
    cross-process broker remains a separate roadmap item; subprocesses still
    persist their exact calls through the shared UsageRecorder.
    """

    def __init__(
        self,
        session_id: str,
        *,
        limits: SpendLimits | None = None,
        recorder: UsageRecorder | None = None,
        unpriced_input_usd_per_million: float = _DEFAULT_UNPRICED_INPUT_USD_PER_MILLION,
        unpriced_output_usd_per_million: float = _DEFAULT_UNPRICED_OUTPUT_USD_PER_MILLION,
        on_reconcile: Callable[[LLMUsage, float, bool], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self.limits = limits or SpendLimits()
        self._recorder = recorder or get_usage_recorder()
        self._tokens = 0
        self._cost_usd = 0.0
        self._calls = 0
        self._reserved_tokens = 0
        self._reserved_cost_usd = 0.0
        self._unpriced_input_usd_per_million = max(unpriced_input_usd_per_million, 0.0)
        self._unpriced_output_usd_per_million = max(
            unpriced_output_usd_per_million, 0.0
        )
        self._on_reconcile = on_reconcile
        self._lock = threading.Lock()

    def quote(self, model: ModelConfig, usage: LLMUsage) -> CostQuote:
        return quote_usage(
            model,
            usage,
            unpriced_input_price=self._unpriced_input_usd_per_million,
            unpriced_output_price=self._unpriced_output_usd_per_million,
        )

    def quote_reservation(self, model: ModelConfig, usage: LLMUsage) -> CostQuote:
        return quote_cold_reservation(
            model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            unpriced_input_price=self._unpriced_input_usd_per_million,
            unpriced_output_price=self._unpriced_output_usd_per_million,
        )

    def try_reserve(
        self, estimated_tokens: int = 0, *, estimated_cost_usd: float = 0.0
    ) -> UsageReservation | None:
        estimate = max(estimated_tokens, 0)
        cost_estimate = max(estimated_cost_usd, 0.0)
        with self._lock:
            limits = self.limits
            if limits.max_calls is not None and self._calls >= limits.max_calls:
                return None
            projected = self._tokens + self._reserved_tokens + estimate
            if limits.max_tokens is not None and projected > limits.max_tokens:
                return None
            if limits.max_cost_usd is not None:
                projected_cost = (
                    self._cost_usd + self._reserved_cost_usd + cost_estimate
                )
                if (
                    self._cost_usd >= limits.max_cost_usd
                    or projected_cost > limits.max_cost_usd
                ):
                    return None
            self._calls += 1
            self._reserved_tokens += estimate
            self._reserved_cost_usd += cost_estimate
            session_id = self.session_id
        return UsageReservation(
            estimated_tokens=estimate,
            estimated_cost_usd=cost_estimate,
            session_id=session_id,
        )

    def rebind_session(self, session_id: str) -> None:
        with self._lock:
            self.session_id = session_id

    def release(self, reservation: UsageReservation) -> None:
        with self._lock:
            if not reservation.active:
                return
            reservation.active = False
            self._reserved_tokens -= reservation.estimated_tokens
            self._reserved_cost_usd -= reservation.estimated_cost_usd
            self._calls -= 1

    def reconcile(
        self,
        reservation: UsageReservation,
        *,
        usage: LLMUsage | None,
        model: ModelConfig,
        provider: ProviderConfig,
        call_kind: CallKind,
        duration_s: float,
        result_used: bool | None = None,
    ) -> None:
        usage_was_missing = usage is None
        if usage_was_missing:
            usage = LLMUsage(
                prompt_tokens=reservation.estimated_tokens, completion_tokens=0
            )
        quote = self.quote(model, usage)
        usage = usage.model_copy(
            update={
                "prompt_tokens": quote.prompt_tokens,
                "cached_tokens": quote.cached_tokens,
                "cache_write_tokens": quote.cache_write_tokens,
                "completion_tokens": quote.completion_tokens,
                "reasoning_tokens": min(
                    usage.reasoning_tokens, quote.completion_tokens
                ),
            }
        )
        cost = quote.cost_usd
        if usage_was_missing:
            cost = max(cost, reservation.estimated_cost_usd)
        with self._lock:
            if not reservation.active:
                return
            reservation.active = False
            self._reserved_tokens -= reservation.estimated_tokens
            self._reserved_cost_usd -= reservation.estimated_cost_usd
            self._tokens += usage.prompt_tokens + usage.completion_tokens
            self._cost_usd += cost
        estimated = usage_was_missing or quote.estimated
        if self._on_reconcile is not None:
            try:
                self._on_reconcile(usage, cost, estimated)
            except Exception:
                logger.warning("Usage reconciliation observer failed", exc_info=True)
        self._recorder.record(
            UsageRecord.from_usage(
                timestamp=time.time(),
                provider=provider.name,
                model=model.name,
                usage=usage,
                cost_usd=cost,
                duration_s=duration_s,
                session_id=reservation.session_id,
                cost_estimated=estimated,
                pricing_mode=quote.pricing_mode,
                harness=True,
                call_kind=call_kind.value,
                result_used=result_used,
            )
        )

    def snapshot(self) -> SpendSnapshot:
        with self._lock:
            return SpendSnapshot(
                tokens=self._tokens,
                cost_usd=self._cost_usd,
                calls=self._calls,
                reserved_tokens=self._reserved_tokens,
                reserved_cost_usd=self._reserved_cost_usd,
            )
