from __future__ import annotations

import time

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.llm.types import BackendLike, CompletionRequest
from vibe.core.types import LLMChunk, LLMUsage
from vibe.core.usage._context import SpendPurpose
from vibe.core.usage._meter import CallKind, UsageMeter, UsageReservation, usage_cost
from vibe.core.usage._session import SessionSpendAdapter, SpendBudgetExceededError
from vibe.core.utils.tokens import approx_token_count

__all__ = ["complete_auxiliary"]


async def complete_auxiliary(
    backend: BackendLike,
    request: CompletionRequest,
    *,
    model: ModelConfig,
    provider: ProviderConfig,
    call_kind: CallKind,
    purpose: SpendPurpose,
    usage_meter: UsageMeter | None,
    spend_adapter: SessionSpendAdapter | None,
    is_retry: bool = False,
) -> LLMChunk | None:
    reservation = _reserve_local(usage_meter, request, model)
    if usage_meter is not None and reservation is None:
        return None

    started = time.monotonic()
    try:
        result = (
            await spend_adapter.complete(
                backend, request, purpose=purpose, is_retry=is_retry
            )
            if spend_adapter is not None
            else await backend.complete(request)
        )
    except SpendBudgetExceededError:
        if usage_meter is not None and reservation is not None:
            usage_meter.release(reservation)
        raise
    except BaseException:
        _reconcile_local(
            usage_meter,
            reservation,
            usage=None,
            model=model,
            provider=provider,
            call_kind=call_kind,
            started=started,
        )
        raise

    _reconcile_local(
        usage_meter,
        reservation,
        usage=getattr(result, "usage", None),
        model=model,
        provider=provider,
        call_kind=call_kind,
        started=started,
        result_used=True,
    )
    return result


def _reserve_local(
    usage_meter: UsageMeter | None, request: CompletionRequest, model: ModelConfig
) -> UsageReservation | None:
    if usage_meter is None:
        return None
    prompt_tokens = sum(
        approx_token_count(message.content or "") for message in request.messages
    )
    usage = LLMUsage(
        prompt_tokens=prompt_tokens, completion_tokens=max(request.max_tokens or 0, 0)
    )
    return usage_meter.try_reserve(
        usage.prompt_tokens + usage.completion_tokens,
        estimated_cost_usd=usage_cost(model, usage),
    )


def _reconcile_local(
    usage_meter: UsageMeter | None,
    reservation: UsageReservation | None,
    *,
    usage: LLMUsage | None,
    model: ModelConfig,
    provider: ProviderConfig,
    call_kind: CallKind,
    started: float,
    result_used: bool | None = None,
) -> None:
    if usage_meter is None or reservation is None:
        return
    usage_meter.reconcile(
        reservation,
        usage=usage,
        model=model,
        provider=provider,
        call_kind=call_kind,
        duration_s=time.monotonic() - started,
        result_used=result_used,
    )
