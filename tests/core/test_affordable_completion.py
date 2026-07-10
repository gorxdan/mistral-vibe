from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
import types
from typing import Literal

import pytest

from vibe.core.config import ModelConfig
from vibe.core.config._spend_config import SpendConfig
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import LLMChunk, LLMMessage, LLMUsage, Role
from vibe.core.usage import SpendEnvelopeLimits, SpendRejectionReason
from vibe.core.usage._session import (
    SessionSpendAdapter,
    SpendBudgetExceededError,
    estimate_request_tokens,
)


class _CaptureBackend:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.entered = asyncio.Event()
        self.requests: list[CompletionRequest] = []

    async def __aenter__(self) -> _CaptureBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        return None

    async def complete(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> LLMChunk:
        self.requests.append(request)
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        return LLMChunk(
            message=LLMMessage(role=Role.ASSISTANT, content="ok"),
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
        )

    async def complete_streaming(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ) -> AsyncGenerator[LLMChunk, None]:
        yield await self.complete(request, response_headers_sink=response_headers_sink)


def _request(*, max_tokens: int | None = None) -> CompletionRequest:
    return CompletionRequest(
        model=ModelConfig(
            name="affordable-model",
            provider="test",
            alias="affordable-model",
            input_price=1.0,
            output_price=2.0,
        ),
        messages=[LLMMessage(role=Role.USER, content="hello")],
        max_tokens=max_tokens,
    )


def _limit(
    kind: Literal["completion", "total", "cost"],
    *,
    prompt_tokens: int,
    output_tokens: int,
) -> SpendEnvelopeLimits:
    match kind:
        case "completion":
            return SpendEnvelopeLimits(max_completion_tokens=output_tokens)
        case "total":
            return SpendEnvelopeLimits(max_total_tokens=prompt_tokens + output_tokens)
        case "cost":
            return SpendEnvelopeLimits(
                max_cost_usd=(prompt_tokens + output_tokens * 2) / 1_000_000
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "agent"])
@pytest.mark.parametrize("kind", ["completion", "total", "cost"])
async def test_omitted_max_tokens_shrinks_to_affordable_scope_bound(
    tmp_path: Path,
    scope: Literal["session", "agent"],
    kind: Literal["completion", "total", "cost"],
) -> None:
    request = _request()
    prompt_tokens = estimate_request_tokens(request)
    limit = _limit(kind, prompt_tokens=prompt_tokens, output_tokens=200)
    config_values: dict[str, int | float] = {
        "default_max_output_tokens": 1_000,
        "minimum_admitted_output_tokens": 10,
    }
    if scope == "session":
        config_values.update(limit.model_dump(exclude_none=True))
    root = SessionSpendAdapter.create(
        SpendConfig.model_validate(config_values),
        f"affordable-{scope}-{kind}",
        ledger_path=tmp_path / f"{scope}-{kind}",
    )
    adapter = root if scope == "session" else root.child_agent(limits=limit)
    backend = _CaptureBackend()

    await adapter.complete(backend, request)

    assert request.max_tokens is None
    assert backend.requests[0].max_tokens == 200
    reservation = next(
        event.reservation for event in adapter.events() if event.kind == "reserved"
    )
    assert reservation.estimate.completion_tokens == 200


@pytest.mark.asyncio
async def test_explicit_max_tokens_remains_a_strict_bound(tmp_path: Path) -> None:
    adapter = SessionSpendAdapter.create(
        SpendConfig(max_completion_tokens=200),
        "explicit-bound",
        ledger_path=tmp_path / "explicit",
    )
    backend = _CaptureBackend()

    with pytest.raises(SpendBudgetExceededError) as exc_info:
        await adapter.complete(backend, _request(max_tokens=300))

    assert exc_info.value.rejection.reason is SpendRejectionReason.COMPLETION_TOKENS
    assert not backend.requests
    assert not [event for event in adapter.events() if event.kind == "reserved"]


@pytest.mark.asyncio
async def test_minimum_unaffordable_output_is_rejected(tmp_path: Path) -> None:
    request = _request()
    prompt_tokens = estimate_request_tokens(request)
    adapter = SessionSpendAdapter.create(
        SpendConfig(
            max_cost_usd=(prompt_tokens + 99 * 2) / 1_000_000,
            default_max_output_tokens=1_000,
            minimum_admitted_output_tokens=100,
        ),
        "minimum-rejection",
        ledger_path=tmp_path / "minimum",
    )
    backend = _CaptureBackend()

    with pytest.raises(SpendBudgetExceededError) as exc_info:
        await adapter.complete(backend, request)

    assert exc_info.value.rejection.reason is SpendRejectionReason.COST_USD
    assert exc_info.value.rejection.estimate.completion_tokens == 100
    assert not backend.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "cost"])
async def test_concurrent_reservations_share_the_atomic_affordable_remainder(
    tmp_path: Path, kind: Literal["completion", "cost"]
) -> None:
    request = _request()
    prompt_tokens = estimate_request_tokens(request)
    config_values: dict[str, int | float] = {
        "default_max_output_tokens": 250,
        "minimum_admitted_output_tokens": 50,
        "max_calls": 2,
        "max_concurrent_calls": 2,
    }
    if kind == "completion":
        config_values["max_completion_tokens"] = 300
    else:
        config_values["max_cost_usd"] = (prompt_tokens * 2 + 300 * 2) / 1_000_000
    adapter = SessionSpendAdapter.create(
        SpendConfig.model_validate(config_values),
        f"concurrent-{kind}",
        ledger_path=tmp_path / kind,
    )
    gate = asyncio.Event()
    backends = (_CaptureBackend(gate), _CaptureBackend(gate))
    tasks = [
        asyncio.create_task(adapter.complete(backend, request)) for backend in backends
    ]
    try:
        await asyncio.wait_for(
            asyncio.gather(*(backend.entered.wait() for backend in backends)), timeout=2
        )
        admitted = {backend.requests[0].max_tokens for backend in backends}
        assert admitted == {50, 250}
        reservations = [
            event.reservation for event in adapter.events() if event.kind == "reserved"
        ]
        assert sum(item.estimate.completion_tokens for item in reservations) == 300
    finally:
        gate.set()
        await asyncio.gather(*tasks)
