from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
import time
from uuid import uuid4

import orjson

from vibe.core.config._spend_config import SpendConfig
from vibe.core.config.models import ModelConfig
from vibe.core.llm.types import BackendLike, CompletionRequest
from vibe.core.paths import VIBE_HOME
from vibe.core.types import FileImageSource, LLMChunk, LLMUsage
from vibe.core.usage._broker import SpendBroker
from vibe.core.usage._context import (
    DEFAULT_RESERVATION_LEASE_S,
    SpendAmount,
    SpendContext,
    SpendEnvelope,
    SpendEnvelopeLimits,
    SpendEnvelopeSnapshot,
    SpendPurpose,
    SpendRejection,
    SpendReservation,
    SpendScopeKind,
)
from vibe.core.usage._ledger import LedgerEvent
from vibe.core.usage._pricing import compute_cost, lookup_pricing

__all__ = [
    "SPEND_SESSION_ID_METADATA_KEY",
    "UNROUTED_PAID_CALL_BOUNDARIES",
    "SessionSpendAdapter",
    "SpendAdmissionBlockedError",
    "SpendBudgetExceededError",
    "estimate_request_tokens",
]


UNROUTED_PAID_CALL_BOUNDARIES = frozenset({
    "isolated_subprocess",
    "mcp_sampling",
    "narration",
})
SPEND_SESSION_ID_METADATA_KEY = "spend_session_id"
_RESERVATION_LEASE_S = DEFAULT_RESERVATION_LEASE_S
_RESERVATION_RENEW_INTERVAL_S = DEFAULT_RESERVATION_LEASE_S / 3


class SpendBudgetExceededError(RuntimeError):
    def __init__(self, rejection: SpendRejection) -> None:
        self.rejection = rejection
        limited_scope = rejection.limited_scope_id or rejection.scope_id
        super().__init__(
            f"Spend budget rejected {rejection.purpose.value} call "
            f"({rejection.reason.value}) at scope {limited_scope!r}."
        )


class SpendAdmissionBlockedError(RuntimeError):
    pass


def estimate_request_tokens(request: CompletionRequest) -> int:
    payload = {
        "messages": [
            message.model_dump(mode="json", exclude_none=True)
            for message in request.messages
        ],
        "tools": [
            tool.model_dump(mode="json", exclude_none=True)
            for tool in request.tools or []
        ],
        "response_format": request.response_format,
    }
    expanded_image_bytes = 0
    for message in request.messages:
        for image in message.images or []:
            if not isinstance(image.source, FileImageSource):
                continue
            size = image.source.path.stat().st_size
            base64_size = 4 * ((size + 2) // 3)
            expanded_image_bytes += base64_size + len(
                f"data:{image.mime_type};base64,".encode()
            )
    return max(1, len(orjson.dumps(payload)) + expanded_image_bytes)


def _cost(model: ModelConfig, usage: LLMUsage, config: SpendConfig) -> float:
    if model.input_price > 0 or model.output_price > 0:
        return (
            usage.prompt_tokens * model.input_price
            + usage.completion_tokens * model.output_price
        ) / 1_000_000
    pricing = lookup_pricing(model.name)
    if pricing is None:
        return (
            usage.prompt_tokens * config.unpriced_input_usd_per_million
            + usage.completion_tokens * config.unpriced_output_usd_per_million
        ) / 1_000_000
    return compute_cost(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        pricing=pricing,
    )


def _session_limits(
    config: SpendConfig,
    *,
    now: float,
    runtime_max_cost_usd: float | None,
    runtime_max_total_tokens: int | None,
) -> SpendEnvelopeLimits:
    max_total_tokens = min(
        config.max_total_tokens,
        max(runtime_max_total_tokens, 0)
        if runtime_max_total_tokens is not None
        else config.max_total_tokens,
    )
    max_cost_usd = min(
        config.max_cost_usd,
        max(runtime_max_cost_usd, 0.0)
        if runtime_max_cost_usd is not None
        else config.max_cost_usd,
    )
    deadline_at = (
        now + config.deadline_seconds if config.deadline_seconds is not None else None
    )
    return SpendEnvelopeLimits(
        max_prompt_tokens=min(config.max_prompt_tokens, max_total_tokens),
        max_completion_tokens=min(config.max_completion_tokens, max_total_tokens),
        max_total_tokens=max_total_tokens,
        max_cost_usd=max_cost_usd,
        max_calls=config.max_calls,
        max_concurrent_calls=config.max_concurrent_calls,
        max_retries=config.max_retries,
        deadline_at=deadline_at,
    )


@dataclass(slots=True)
class _AdmissionState:
    config: SpendConfig
    error: SpendAdmissionBlockedError | None = None


@dataclass(slots=True)
class _SessionSpendCall:
    adapter: SessionSpendAdapter
    request: CompletionRequest
    reservation: SpendReservation
    config: SpendConfig
    dispatched: bool = False
    settled: bool = False

    def mark_dispatched(self) -> None:
        self.dispatched = True

    async def renew_while_active(self) -> None:
        while True:
            await asyncio.sleep(_RESERVATION_RENEW_INTERVAL_S)
            if self.settled:
                return
            renewed = self.adapter._broker.renew(
                self.reservation, lease_s=_RESERVATION_LEASE_S
            )
            if not renewed:
                raise RuntimeError(
                    f"Spend reservation {self.reservation.reservation_id!r} "
                    "expired while its provider call was still active."
                )

    def settle(self, usage: LLMUsage | None) -> None:
        if self.settled:
            return
        self.settled = True
        if not self.dispatched:
            self.adapter._broker.release(
                self.reservation, reason="backend dispatch did not start"
            )
            return
        if usage is None:
            self.adapter._broker.reconcile(self.reservation, None)
            return
        cost = _cost(self.request.model, usage, self.config)
        actual = SpendAmount(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
        )
        self.adapter._broker.reconcile(self.reservation, actual)


class SessionSpendAdapter:
    """Scoped paid-call admission for one agent in a shared session ledger.

    Narration, MCP sampling, and isolated subprocesses remain explicit later
    integration boundaries. They do not silently borrow this session adapter's
    scope.
    """

    def __init__(
        self,
        *,
        broker: SpendBroker,
        config: SpendConfig,
        session_scope_id: str,
        agent_scope_id: str,
        default_purpose: SpendPurpose,
        clock: Callable[[], float],
        admission_state: _AdmissionState | None = None,
    ) -> None:
        self._broker = broker
        self.session_scope_id = session_scope_id
        self.agent_scope_id = agent_scope_id
        self.default_purpose = default_purpose
        self._clock = clock
        self._admission_state = admission_state or _AdmissionState(config=config)
        self.last_admitted_completion_tokens: int | None = None

    @property
    def _config(self) -> SpendConfig:
        return self._admission_state.config

    @classmethod
    def create(
        cls,
        config: SpendConfig,
        session_id: str,
        *,
        ledger_path: Path | None = None,
        clock: Callable[[], float] = time.time,
        runtime_max_cost_usd: float | None = None,
        runtime_max_total_tokens: int | None = None,
    ) -> SessionSpendAdapter:
        path = ledger_path or VIBE_HOME.path / "spend" / session_id
        broker = SpendBroker(path, clock=clock)
        session_scope_id = f"session:{session_id}"
        session_envelope = SpendEnvelope(
            scope_id=session_scope_id,
            kind=SpendScopeKind.SESSION,
            limits=_session_limits(
                config,
                now=clock(),
                runtime_max_cost_usd=runtime_max_cost_usd,
                runtime_max_total_tokens=runtime_max_total_tokens,
            ),
        )
        existing_session_envelope = broker.get_envelope(session_scope_id)
        if existing_session_envelope is None:
            broker.define_envelope(session_envelope)
        elif existing_session_envelope.kind != SpendScopeKind.SESSION:
            broker.define_envelope(session_envelope)
        else:
            broker.tighten_envelope(session_scope_id, session_envelope.limits)
        agent_scope_id = f"agent:{session_id}:primary"
        broker.define_envelope(
            SpendEnvelope(
                scope_id=agent_scope_id,
                kind=SpendScopeKind.AGENT,
                parent_scope_id=session_scope_id,
            )
        )
        return cls(
            broker=broker,
            config=config,
            session_scope_id=session_scope_id,
            agent_scope_id=agent_scope_id,
            default_purpose=SpendPurpose.PRIMARY,
            clock=clock,
        )

    @property
    def ledger_path(self) -> Path:
        return self._broker.ledger_path

    @property
    def spend_session_id(self) -> str:
        return self.session_scope_id.removeprefix("session:")

    def tighten_limits(
        self,
        config: SpendConfig,
        *,
        runtime_max_cost_usd: float | None = None,
        runtime_max_total_tokens: int | None = None,
    ) -> None:
        self._admission_state.error = SpendAdmissionBlockedError(
            "spend admission is blocked while limits are updating"
        )
        try:
            self._broker.tighten_envelope(
                self.session_scope_id,
                _session_limits(
                    config,
                    now=self._clock(),
                    runtime_max_cost_usd=runtime_max_cost_usd,
                    runtime_max_total_tokens=runtime_max_total_tokens,
                ),
            )
        except Exception as e:
            error = SpendAdmissionBlockedError(f"spend admission is blocked: {e}")
            self._admission_state.error = error
            raise error from e
        self._admission_state.config = config
        self._admission_state.error = None

    def child_agent(
        self,
        *,
        group_kind: SpendScopeKind | None = None,
        group_id: str | None = None,
        agent_id: str | None = None,
        purpose: SpendPurpose | None = None,
    ) -> SessionSpendAdapter:
        if group_kind not in {None, SpendScopeKind.WORKFLOW, SpendScopeKind.TEAM}:
            raise ValueError("child group must be a workflow or team scope")
        if group_kind is None and group_id is not None:
            raise ValueError("group_id requires group_kind")
        parent_scope_id = self.session_scope_id
        if group_kind is not None:
            resolved_group_id = group_id or f"{group_kind.value}:{uuid4().hex}"
            self._broker.define_envelope(
                SpendEnvelope(
                    scope_id=resolved_group_id,
                    kind=group_kind,
                    parent_scope_id=self.session_scope_id,
                )
            )
            parent_scope_id = resolved_group_id
        resolved_agent_id = agent_id or f"agent:{uuid4().hex}"
        self._broker.define_envelope(
            SpendEnvelope(
                scope_id=resolved_agent_id,
                kind=SpendScopeKind.AGENT,
                parent_scope_id=parent_scope_id,
            )
        )
        return SessionSpendAdapter(
            broker=self._broker,
            config=self._config,
            session_scope_id=self.session_scope_id,
            agent_scope_id=resolved_agent_id,
            default_purpose=purpose or self.default_purpose,
            clock=self._clock,
            admission_state=self._admission_state,
        )

    def _completion_token_estimate(self, request: CompletionRequest) -> int:
        if request.max_tokens is not None:
            if request.max_tokens < 0:
                raise ValueError("max_tokens cannot be negative")
            return request.max_tokens
        model_limit = request.model.max_output_tokens
        return (
            model_limit
            if model_limit is not None and model_limit > 0
            else self._config.default_max_output_tokens
        )

    def _reserve(
        self, request: CompletionRequest, *, purpose: SpendPurpose, is_retry: bool
    ) -> _SessionSpendCall:
        if self._admission_state.error is not None:
            raise self._admission_state.error
        prompt_tokens = estimate_request_tokens(request)
        completion_tokens = self._completion_token_estimate(request)
        estimated_usage = LLMUsage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        estimated_cost = _cost(request.model, estimated_usage, self._config)
        estimate = SpendAmount(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=estimated_cost,
        )
        decision = self._broker.try_reserve(
            SpendContext(
                scope_id=self.agent_scope_id, purpose=purpose, is_retry=is_retry
            ),
            estimate,
            lease_s=_RESERVATION_LEASE_S,
        )
        if isinstance(decision, SpendRejection):
            raise SpendBudgetExceededError(decision)
        self.last_admitted_completion_tokens = completion_tokens
        admitted_request = (
            request
            if request.max_tokens is not None
            else replace(request, max_tokens=completion_tokens)
        )
        return _SessionSpendCall(
            adapter=self,
            request=admitted_request,
            reservation=decision,
            config=self._config,
        )

    @staticmethod
    async def _stop_renewal(task: asyncio.Task[None]) -> None:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def complete(
        self,
        backend: BackendLike,
        request: CompletionRequest,
        *,
        purpose: SpendPurpose | None = None,
        is_retry: bool = False,
        response_headers_sink: dict[str, str] | None = None,
    ) -> LLMChunk:
        resolved_purpose = purpose or self.default_purpose
        call = self._reserve(request, purpose=resolved_purpose, is_retry=is_retry)
        call.mark_dispatched()
        renewal = asyncio.create_task(call.renew_while_active())
        try:
            result = await backend.complete(
                call.request, response_headers_sink=response_headers_sink
            )
        except BaseException:
            try:
                await self._stop_renewal(renewal)
            finally:
                call.settle(None)
            raise
        try:
            await self._stop_renewal(renewal)
        finally:
            call.settle(result.usage)
        return result

    async def complete_streaming(
        self,
        backend: BackendLike,
        request: CompletionRequest,
        *,
        purpose: SpendPurpose | None = None,
        is_retry: bool = False,
        response_headers_sink: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        resolved_purpose = purpose or self.default_purpose
        call = self._reserve(request, purpose=resolved_purpose, is_retry=is_retry)
        final_usage: LLMUsage | None = None
        call.mark_dispatched()
        renewal = asyncio.create_task(call.renew_while_active())
        try:
            async for chunk in backend.complete_streaming(
                call.request, response_headers_sink=response_headers_sink
            ):
                if chunk.usage is not None:
                    final_usage = (
                        chunk.usage
                        if final_usage is None
                        else final_usage + chunk.usage
                    )
                yield chunk
        except BaseException:
            try:
                await self._stop_renewal(renewal)
            finally:
                call.settle(None)
            raise
        try:
            await self._stop_renewal(renewal)
        finally:
            call.settle(final_usage)

    def snapshot(self) -> SpendEnvelopeSnapshot:
        return self._broker.snapshot(self.session_scope_id)

    def events(self) -> list[LedgerEvent]:
        return self._broker.events()
