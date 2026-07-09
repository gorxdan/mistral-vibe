from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
import threading
import time

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.types import LLMUsage
from vibe.core.usage._pricing import compute_cost, lookup_pricing
from vibe.core.usage._recorder import UsageRecorder, get_usage_recorder
from vibe.core.usage.models import UsageRecord


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


def usage_cost(model: ModelConfig, usage: LLMUsage) -> float:
    if model.input_price > 0 or model.output_price > 0:
        return (
            usage.prompt_tokens * model.input_price
            + usage.completion_tokens * model.output_price
        ) / 1_000_000
    pricing = lookup_pricing(model.name)
    if pricing is None:
        return 0.0
    return compute_cost(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        pricing=pricing,
    )


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
    ) -> None:
        self.session_id = session_id
        self.limits = limits or SpendLimits()
        self._recorder = recorder or get_usage_recorder()
        self._tokens = 0
        self._cost_usd = 0.0
        self._calls = 0
        self._reserved_tokens = 0
        self._reserved_cost_usd = 0.0
        self._lock = threading.Lock()

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
        cost = usage_cost(model, usage)
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
        self._recorder.record(
            UsageRecord.from_usage(
                timestamp=time.time(),
                provider=provider.name,
                model=model.name,
                usage=usage,
                cost_usd=cost,
                duration_s=duration_s,
                session_id=reservation.session_id,
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
