from __future__ import annotations

from enum import StrEnum, auto
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.llm.provider_retry import SpendRetryCause, SpendRetryPolicyReason

__all__ = [
    "DEFAULT_RESERVATION_LEASE_S",
    "MAX_RESERVATION_LEASE_S",
    "PromptTokenEstimate",
    "SpendAmount",
    "SpendContext",
    "SpendEnvelope",
    "SpendEnvelopeLimits",
    "SpendEnvelopeSnapshot",
    "SpendPurpose",
    "SpendRejection",
    "SpendRejectionReason",
    "SpendReservation",
    "SpendRetryAuthorization",
    "SpendRetryCause",
    "SpendRetryPolicyReason",
    "SpendScopeKind",
    "SpendSettlement",
    "SpendSettlementDisposition",
]

DEFAULT_RESERVATION_LEASE_S = 300.0
MAX_RESERVATION_LEASE_S = 3600.0


class SpendScopeKind(StrEnum):
    SESSION = auto()
    WORKFLOW = auto()
    TEAM = auto()
    AGENT = auto()
    CALL = auto()


class SpendPurpose(StrEnum):
    PRIMARY = auto()
    COMPACTION = auto()
    MEMORY_RECALL = auto()
    MEMORY_EXTRACT = auto()
    MEMORY_CONSOLIDATE = auto()
    MEMORY_VERIFY = auto()
    SAFETY_JUDGE = auto()
    NARRATION = auto()
    WORKFLOW = auto()
    TEAM = auto()
    REPAIR = auto()
    VERIFICATION = auto()


class SpendRejectionReason(StrEnum):
    UNKNOWN_SCOPE = auto()
    DUPLICATE_CALL = auto()
    DEADLINE = auto()
    PROMPT_TOKENS = auto()
    COMPLETION_TOKENS = auto()
    TOTAL_TOKENS = auto()
    COST_USD = auto()
    CALLS = auto()
    CONCURRENT_CALLS = auto()
    RETRIES = auto()


class SpendSettlementDisposition(StrEnum):
    RECONCILED = auto()
    RELEASED = auto()
    EXPIRED = auto()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class PromptTokenEstimate(_FrozenModel):
    estimator_version: int = Field(ge=1)
    profile_key: str = Field(min_length=1, max_length=512)
    base_tokens: int = Field(ge=1)
    strict_tokens: int = Field(ge=1)
    estimated_tokens: int = Field(ge=1)
    factor: float = Field(gt=0.0)
    sample_count: int = Field(ge=0)
    adaptive: bool


class SpendAmount(_FrozenModel):
    prompt_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_cached_tokens(self) -> Self:
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens cannot exceed prompt_tokens")
        return self

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class SpendEnvelopeLimits(_FrozenModel):
    max_prompt_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    max_total_tokens: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    max_calls: int | None = Field(default=None, ge=0)
    max_concurrent_calls: int | None = Field(default=None, ge=0)
    max_retries: int | None = Field(default=None, ge=0)
    deadline_at: float | None = Field(default=None, ge=0.0)


class SpendEnvelope(_FrozenModel):
    scope_id: str = Field(min_length=1, max_length=256)
    kind: SpendScopeKind
    policy_version: int = Field(default=1, ge=1)
    limits: SpendEnvelopeLimits = Field(default_factory=SpendEnvelopeLimits)
    parent_scope_id: str | None = Field(default=None, min_length=1, max_length=256)
    task_brief_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("scope_id")
    @classmethod
    def _reserve_call_namespace(cls, value: str) -> str:
        if value.startswith("call:"):
            raise ValueError("scope_id cannot use the reserved call: namespace")
        return value

    @model_validator(mode="after")
    def _validate_task_binding(self) -> Self:
        if self.task_brief_hash is not None and self.kind is not SpendScopeKind.AGENT:
            raise ValueError("task brief hash can bind only an agent scope")
        return self


class SpendContext(_FrozenModel):
    scope_id: str = Field(min_length=1, max_length=256)
    purpose: SpendPurpose
    call_id: str | None = Field(default=None, min_length=1, max_length=256)
    is_retry: bool = False


class SpendReservation(_FrozenModel):
    reservation_id: str = Field(min_length=1, max_length=256)
    call_scope_id: str = Field(min_length=1, max_length=512)
    scope_id: str = Field(min_length=1, max_length=256)
    scope_chain: tuple[str, ...] = Field(min_length=2)
    purpose: SpendPurpose
    estimate: SpendAmount
    prompt_estimate: PromptTokenEstimate | None = None
    is_retry: bool
    created_at: float = Field(ge=0.0)
    lease_expires_at: float = Field(ge=0.0)
    # Missing from older ledger events; version 0 keeps their expiry conservative.
    dispatch_tracking_version: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_scope_chain(self) -> Self:
        if self.call_scope_id != f"call:{self.reservation_id}":
            raise ValueError("call_scope_id must derive from reservation_id")
        if self.scope_chain[-1] != self.call_scope_id:
            raise ValueError("call scope must terminate scope_chain")
        if self.scope_chain[-2] != self.scope_id:
            raise ValueError("agent scope must precede call scope")
        if self.lease_expires_at <= self.created_at:
            raise ValueError("reservation lease must expire after creation")
        return self


class SpendRejection(_FrozenModel):
    call_id: str = Field(min_length=1, max_length=256)
    scope_id: str = Field(min_length=1, max_length=256)
    scope_chain: tuple[str, ...] = ()
    purpose: SpendPurpose
    estimate: SpendAmount
    prompt_estimate: PromptTokenEstimate | None = None
    is_retry: bool
    reason: SpendRejectionReason
    limited_scope_id: str | None = Field(default=None, min_length=1, max_length=256)
    timestamp: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_scope_chain(self) -> Self:
        if self.scope_chain and self.scope_chain[-1] != self.scope_id:
            raise ValueError("rejection scope must terminate scope_chain")
        return self


class SpendRetryAuthorization(_FrozenModel):
    reservation_id: str = Field(min_length=1, max_length=256)
    call_scope_id: str = Field(min_length=1, max_length=512)
    scope_chain: tuple[str, ...] = Field(min_length=2)
    attempt: int = Field(ge=1)
    cause: SpendRetryCause
    timestamp: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_scope_chain(self) -> Self:
        if self.call_scope_id != f"call:{self.reservation_id}":
            raise ValueError("call_scope_id must derive from reservation_id")
        if self.scope_chain[-1] != self.call_scope_id:
            raise ValueError("call scope must terminate scope_chain")
        return self


class SpendSettlement(_FrozenModel):
    reservation_id: str = Field(min_length=1, max_length=256)
    disposition: SpendSettlementDisposition
    amount: SpendAmount
    estimated: bool
    applied: bool
    timestamp: float = Field(ge=0.0)
    reason: str | None = Field(default=None, max_length=512)


class SpendEnvelopeSnapshot(_FrozenModel):
    envelope: SpendEnvelope
    spent: SpendAmount
    reserved: SpendAmount
    rejected: SpendAmount
    spent_calls: int = Field(ge=0)
    reserved_calls: int = Field(ge=0)
    rejected_calls: int = Field(ge=0)
    spent_retries: int = Field(ge=0)
    reserved_retries: int = Field(ge=0)
    rejected_retries: int = Field(default=0, ge=0)
    remaining_prompt_tokens: int | None = Field(default=None, ge=0)
    remaining_completion_tokens: int | None = Field(default=None, ge=0)
    remaining_total_tokens: int | None = Field(default=None, ge=0)
    remaining_cost_usd: float | None = Field(default=None, ge=0.0)
    remaining_calls: int | None = Field(default=None, ge=0)
    remaining_concurrent_calls: int | None = Field(default=None, ge=0)
    remaining_retries: int | None = Field(default=None, ge=0)
    deadline_at: float | None = Field(default=None, ge=0.0)
