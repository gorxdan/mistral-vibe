from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
import respx

from vibe.core.config import ModelConfig, ProviderConfig
from vibe.core.config._spend_config import SpendConfig
from vibe.core.llm.backend.generic import GenericBackend
from vibe.core.llm.backend.mistral import MistralBackend
from vibe.core.llm.exceptions import BackendError
from vibe.core.llm.provider_retry import SpendRetryCause
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import LLMMessage, Role
from vibe.core.usage import SpendEnvelopeLimits, SpendRejectionReason
from vibe.core.usage._session import (
    SessionSpendAdapter,
    SpendBudgetExceededError,
    estimate_request_tokens,
)

_ENDPOINT = "https://retry.test/v1/chat/completions"
_SUCCESS = {
    "choices": [
        {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}
_STREAM_CHUNK = (
    b'data: {"choices":[{"delta":{"role":"assistant","content":"ok"},'
    b'"finish_reason":null}]}\n\n'
)
_STREAM_DONE = (
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
    b"data: [DONE]\n\n"
)


class _ChunkThenReadError(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield _STREAM_CHUNK
        raise httpx.ReadError("midstream disconnect")


def _request(
    *,
    thinking: Literal["off", "max"] = "off",
    input_price: float = 0.0,
    output_price: float = 0.0,
) -> CompletionRequest:
    return CompletionRequest(
        model=ModelConfig(
            name="test-model",
            provider="retry",
            alias="test-model",
            thinking=thinking,
            max_output_tokens=10,
            input_price=input_price,
            output_price=output_price,
        ),
        messages=[LLMMessage(role=Role.USER, content="hello")],
        max_tokens=10,
    )


def _backend(*, retry_max_elapsed_time: float = 300.0) -> GenericBackend:
    return GenericBackend(
        provider=ProviderConfig(name="retry", api_base="https://retry.test/v1"),
        retry_max_elapsed_time=retry_max_elapsed_time,
    )


def _spend_adapter(
    path: Path, *, max_retries: int, session_id: str
) -> SessionSpendAdapter:
    return SessionSpendAdapter.create(
        SpendConfig(max_retries=max_retries, enforce_limits=True),
        session_id,
        ledger_path=path,
    )


def _retry_events(adapter: SessionSpendAdapter, kind: str) -> list[Any]:
    return [event for event in adapter.events() if event.kind == kind]


@pytest.mark.asyncio
async def test_503_retry_is_authorized_on_the_logical_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibe.core.utils.retry._retry_delay", lambda *args: 0.0)
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=1, session_id="503")
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=_SUCCESS)]
        )
        result = await adapter.complete(_backend(), _request())

    assert result.message.content == "ok"
    assert route.call_count == 2
    reservation = _retry_events(adapter, "reserved")[0].reservation
    retry = _retry_events(adapter, "retry_authorized")[0].authorization
    assert retry.attempt == 1
    assert retry.cause is SpendRetryCause.HTTP_STATUS
    snapshot = adapter.snapshot()
    assert snapshot.spent_retries == 1
    assert snapshot.spent.prompt_tokens == reservation.estimate.prompt_tokens + 3
    assert snapshot.spent.completion_tokens == (
        reservation.estimate.completion_tokens + 2
    )


@pytest.mark.parametrize("cap_scope", ["session", "child"])
@pytest.mark.asyncio
async def test_retry_token_exposure_is_admitted_at_every_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap_scope: str
) -> None:
    monkeypatch.setattr("vibe.core.utils.retry._retry_delay", lambda *args: 0.0)
    config = SpendConfig(
        max_completion_tokens=10 if cap_scope == "session" else None,
        max_retries=1,
        enforce_limits=True,
    )
    adapter = SessionSpendAdapter.create(
        config, f"token-{cap_scope}", ledger_path=tmp_path / cap_scope
    )
    if cap_scope == "child":
        adapter = adapter.child_agent(
            limits=SpendEnvelopeLimits(max_completion_tokens=10)
        )
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=_SUCCESS)]
        )
        with pytest.raises(SpendBudgetExceededError) as exc_info:
            await adapter.complete(_backend(), _request())

    assert route.call_count == 1
    assert exc_info.value.rejection.reason is SpendRejectionReason.COMPLETION_TOKENS
    assert exc_info.value.rejection.estimate.completion_tokens == 10


@pytest.mark.asyncio
async def test_retry_usd_exposure_is_admitted_before_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibe.core.utils.retry._retry_delay", lambda *args: 0.0)
    request = _request(input_price=1_000_000, output_price=1_000_000)
    first_attempt_cost = float(estimate_request_tokens(request) + 10)
    adapter = SessionSpendAdapter.create(
        SpendConfig(
            max_cost_usd=first_attempt_cost, max_retries=1, enforce_limits=True
        ),
        "retry-usd",
        ledger_path=tmp_path / "usd",
    )
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=_SUCCESS)]
        )
        with pytest.raises(SpendBudgetExceededError) as exc_info:
            await adapter.complete(_backend(), request)

    assert route.call_count == 1
    assert exc_info.value.rejection.reason is SpendRejectionReason.COST_USD
    assert exc_info.value.rejection.estimate.cost_usd == first_attempt_cost


@pytest.mark.asyncio
async def test_timeout_retry_uses_transport_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibe.core.utils.retry._retry_delay", lambda *args: 0.0)
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=1, session_id="timeout")
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            side_effect=[
                httpx.ConnectTimeout("connect timeout"),
                httpx.Response(200, json=_SUCCESS),
            ]
        )
        await adapter.complete(_backend(), _request())

    assert route.call_count == 2
    retry = _retry_events(adapter, "retry_authorized")[0].authorization
    assert retry.cause is SpendRetryCause.TRANSPORT


@pytest.mark.asyncio
async def test_mistral_sdk_retry_is_disabled_and_outer_retry_is_brokered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "vibe.core.llm.backend._mistral_retry.provider_retry_delay", lambda *args: 0.0
    )
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=0, session_id="mistral")
    backend = MistralBackend(
        ProviderConfig(name="retry", api_base="https://retry.test/v1")
    )
    assert backend._retry_config.strategy == "none"

    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(return_value=httpx.Response(503))
        with pytest.raises(SpendBudgetExceededError):
            async with backend:
                await adapter.complete(backend, _request())

    assert route.call_count == 1
    rejected = _retry_events(adapter, "retry_budget_rejected")[0]
    assert rejected.cause is SpendRetryCause.HTTP_STATUS


@pytest.mark.asyncio
async def test_retry_budget_denial_stops_before_second_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibe.core.utils.retry._retry_delay", lambda *args: 0.0)
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=0, session_id="denied")
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=_SUCCESS)]
        )
        with pytest.raises(SpendBudgetExceededError):
            await adapter.complete(_backend(), _request())

    assert route.call_count == 1
    assert not _retry_events(adapter, "retry_authorized")
    assert len(_retry_events(adapter, "retry_budget_rejected")) == 1
    assert adapter.snapshot().rejected_retries == 1


@pytest.mark.asyncio
async def test_quota_429_is_not_retried(tmp_path: Path) -> None:
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=2, session_id="quota")
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"type": "insufficient_quota"}},
            )
        )
        with pytest.raises(BackendError):
            await adapter.complete(_backend(), _request())

    assert route.call_count == 1
    assert not _retry_events(adapter, "retry_authorized")
    assert not _retry_events(adapter, "retry_budget_rejected")


@pytest.mark.asyncio
async def test_effort_resend_requires_retry_authorization(tmp_path: Path) -> None:
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=0, session_id="effort")
    error = "Unsupported value: 'max'. Supported values are: 'low', 'medium', 'high'."
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            side_effect=[
                httpx.Response(400, json={"error": {"message": error}}),
                httpx.Response(200, json=_SUCCESS),
            ]
        )
        with pytest.raises(SpendBudgetExceededError):
            await adapter.complete(_backend(), _request(thinking="max"))

    assert route.call_count == 1
    rejected = _retry_events(adapter, "retry_budget_rejected")[0]
    assert rejected.cause is SpendRetryCause.REASONING_EFFORT


@pytest.mark.asyncio
async def test_elapsed_limit_is_durable_and_prevents_retry(tmp_path: Path) -> None:
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=2, session_id="elapsed")
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(return_value=httpx.Response(503))
        with pytest.raises(BackendError):
            await adapter.complete(_backend(retry_max_elapsed_time=0.0), _request())

    assert route.call_count == 1
    assert len(_retry_events(adapter, "retry_policy_rejected")) == 1
    snapshot = adapter.snapshot()
    assert snapshot.spent_retries == 0
    assert snapshot.rejected_retries == 1


@pytest.mark.asyncio
async def test_stream_retries_only_before_first_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vibe.core.utils.retry._retry_delay", lambda *args: 0.0)
    adapter = _spend_adapter(
        tmp_path / "ledger", max_retries=2, session_id="stream-open"
    )
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(
                    200,
                    stream=httpx.ByteStream(_STREAM_CHUNK + _STREAM_DONE),
                    headers={"content-type": "text/event-stream"},
                ),
            ]
        )
        chunks = [
            chunk async for chunk in adapter.complete_streaming(_backend(), _request())
        ]

    assert route.call_count == 2
    assert any(chunk.message.content == "ok" for chunk in chunks)
    assert len(_retry_events(adapter, "retry_authorized")) == 1

    midstream = _spend_adapter(
        tmp_path / "midstream-ledger", max_retries=2, session_id="stream-mid"
    )
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                stream=_ChunkThenReadError(),
                headers={"content-type": "text/event-stream"},
            )
        )
        received = []
        with pytest.raises(BackendError):
            async for chunk in midstream.complete_streaming(_backend(), _request()):
                received.append(chunk)

    assert route.call_count == 1
    assert received[0].message.content == "ok"
    assert not _retry_events(midstream, "retry_authorized")


@pytest.mark.asyncio
async def test_cancellation_during_backoff_does_not_authorize_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered_backoff = asyncio.Event()
    from vibe.core.utils import retry as retry_module

    original_authorize = retry_module.authorize_provider_retry

    async def blocking_authorize(cause, *, delay_s):
        entered_backoff.set()
        return await original_authorize(cause, delay_s=30.0)

    monkeypatch.setattr(retry_module, "authorize_provider_retry", blocking_authorize)
    adapter = _spend_adapter(tmp_path / "ledger", max_retries=2, session_id="cancel")
    with respx.mock() as mock_api:
        route = mock_api.post(_ENDPOINT).mock(return_value=httpx.Response(503))
        task = asyncio.create_task(adapter.complete(_backend(), _request()))
        await entered_backoff.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert route.call_count == 1
    assert not _retry_events(adapter, "retry_authorized")
    assert not _retry_events(adapter, "retry_policy_rejected")
