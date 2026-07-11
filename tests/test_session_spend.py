from __future__ import annotations

import ast
import asyncio
import multiprocessing
from pathlib import Path
import types
from unittest.mock import patch

from pydantic import ValidationError
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import collect_result, mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.manager import AgentManager
from vibe.core.config import (
    ModelConfig,
    PromptEstimatorMode,
    SessionLoggingConfig,
    SpendConfig,
)
from vibe.core.llm.types import CompletionRequest
from vibe.core.tasking import TaskBrief, TaskManifestIdentity, TaskOutcomeStatus
from vibe.core.tools.base import BaseToolState, InvokeContext
from vibe.core.tools.builtins.task import Task, TaskArgs, TaskToolConfig
from vibe.core.types import (
    AvailableFunction,
    AvailableTool,
    FileImageSource,
    ImageAttachment,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    UnclassifiedBackendError,
)
from vibe.core.usage import (
    SpendLedgerConflictError,
    UsageRecorder,
    get_usage_recorder,
    reset_usage_recorder_for_tests,
)
from vibe.core.usage._broker import SpendBroker
from vibe.core.usage._context import (
    SpendAmount,
    SpendEnvelope,
    SpendEnvelopeLimits,
    SpendPurpose,
    SpendRejection,
    SpendRejectionReason,
    SpendScopeKind,
)
from vibe.core.usage._process_context import (
    SPEND_PROCESS_CONTEXT_ENV,
    SpendProcessContext,
    SpendProcessContextError,
    decode_spend_process_context,
    install_spend_process_context,
    load_spend_process_context,
)
from vibe.core.usage._prompt_estimator import request_prompt_footprint
from vibe.core.usage._session import (
    UNROUTED_PAID_CALL_BOUNDARIES,
    SessionSpendAdapter,
    SpendAdmissionBlockedError,
    SpendBudgetExceededError,
    estimate_request_tokens,
)
from vibe.core.utils.io import read_safe, write_safe
from vibe.core.workflows.runtime import WorkflowRuntime


class _CountingBackend:
    def __init__(
        self,
        *,
        result: LLMChunk | None = None,
        error: Exception | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.result = result or mock_llm_chunk()
        self.error = error
        self.gate = gate
        self.complete_calls = 0
        self.streaming_calls = 0
        self.requests: list[CompletionRequest] = []

    async def __aenter__(self) -> _CountingBackend:
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
        self.complete_calls += 1
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def complete_streaming(
        self,
        request: CompletionRequest,
        *,
        response_headers_sink: dict[str, str] | None = None,
    ):
        self.streaming_calls += 1
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        yield self.result


def _model() -> ModelConfig:
    return ModelConfig(
        name="spend-test",
        provider="test",
        alias="spend-test",
        input_price=1.0,
        output_price=2.0,
    )


def _request(
    *,
    max_tokens: int | None = None,
    tools: list[AvailableTool] | None = None,
    response_format: dict | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        model=_model(),
        messages=[LLMMessage(role=Role.USER, content="hello é")],
        tools=tools,
        response_format=response_format,
        max_tokens=max_tokens,
    )


def _adapter(tmp_path: Path, **overrides) -> SessionSpendAdapter:
    config = SpendConfig(**overrides)
    return SessionSpendAdapter.create(
        config, "test-session", ledger_path=tmp_path / "ledger"
    )


def _attached_call_worker(encoded: str, start, results) -> None:
    adapter = SessionSpendAdapter.attach(
        SpendConfig(), decode_spend_process_context(encoded)
    )
    backend = _CountingBackend(
        result=LLMChunk(
            message=LLMMessage(role=Role.ASSISTANT),
            usage=LLMUsage(prompt_tokens=2, completion_tokens=1),
        )
    )
    start.wait(timeout=10)
    try:
        asyncio.run(adapter.complete(backend, _request(max_tokens=1)))
    except SpendBudgetExceededError as e:
        results.put((False, e.rejection.reason.value))
        return
    results.put((True, None))


def test_spend_config_defaults_use_dynamic_token_admission() -> None:
    config = SpendConfig()

    assert config.max_prompt_tokens is None
    assert config.max_completion_tokens is None
    assert config.max_total_tokens is None
    assert config.max_cost_usd == 10.0
    assert config.max_calls == 128
    assert config.max_concurrent_calls == 2
    assert config.max_retries == 16
    assert config.deadline_seconds is None

    with pytest.raises(ValidationError, match="max_prompt_tokens"):
        SpendConfig(max_prompt_tokens=11, max_total_tokens=10)

    for field in (
        "max_cost_usd",
        "deadline_seconds",
        "unpriced_input_usd_per_million",
        "unpriced_output_usd_per_million",
    ):
        for value in (float("inf"), float("nan")):
            with pytest.raises(ValidationError):
                SpendConfig.model_validate({field: value})


def test_process_context_round_trip_replaces_inherited_nested_scope(tmp_path) -> None:
    parent = _adapter(tmp_path)
    first = parent.child_agent(agent_id="agent:first")
    second = parent.child_agent(agent_id="agent:second", purpose=SpendPurpose.TEAM)
    env = {SPEND_PROCESS_CONTEXT_ENV: first.export_process_context().model_dump_json()}

    install_spend_process_context(env, second.export_process_context())
    loaded = load_spend_process_context(env)

    assert loaded == second.export_process_context()
    assert loaded is not None
    assert loaded.agent_scope_id == "agent:second"
    assert loaded.purpose is SpendPurpose.TEAM
    install_spend_process_context(env, None)
    assert SPEND_PROCESS_CONTEXT_ENV not in env


@pytest.mark.parametrize(
    "payload",
    [
        '{"version":2,"ledger_path":"/tmp/x","session_scope_id":"s",'
        '"agent_scope_id":"a","purpose":"primary"}',
        '{"version":1,"ledger_path":"relative","session_scope_id":"s",'
        '"agent_scope_id":"a","purpose":"primary"}',
        '{"version":1,"ledger_path":"/tmp/x","session_scope_id":"s",'
        '"agent_scope_id":"a","purpose":"primary","extra":true}',
    ],
)
def test_process_context_rejects_tampered_payload(payload: str) -> None:
    with pytest.raises(SpendProcessContextError, match="invalid"):
        decode_spend_process_context(payload)


def test_attached_adapter_validates_existing_scope_kind_and_ancestry(tmp_path) -> None:
    parent = _adapter(tmp_path)
    child = parent.child_agent(
        group_kind=SpendScopeKind.WORKFLOW,
        group_id="workflow:attached",
        agent_id="agent:attached",
        purpose=SpendPurpose.WORKFLOW,
    )
    context = child.export_process_context()
    before = len(parent.events())

    attached = SessionSpendAdapter.attach(SpendConfig(max_calls=999), context)

    assert attached.ledger_path == parent.ledger_path
    assert attached.session_scope_id == parent.session_scope_id
    assert attached.agent_scope_id == child.agent_scope_id
    assert attached.default_purpose is SpendPurpose.WORKFLOW
    assert len(parent.events()) == before

    wrong_kind = context.model_copy(update={"agent_scope_id": parent.session_scope_id})
    with pytest.raises(SpendAdmissionBlockedError, match="invalid agent"):
        SessionSpendAdapter.attach(SpendConfig(), wrong_kind)

    broker = SpendBroker(parent.ledger_path)
    broker.define_envelope(
        SpendEnvelope(scope_id="session:foreign", kind=SpendScopeKind.SESSION)
    )
    broker.define_envelope(
        SpendEnvelope(
            scope_id="agent:foreign",
            kind=SpendScopeKind.AGENT,
            parent_scope_id="session:foreign",
        )
    )
    foreign = context.model_copy(update={"agent_scope_id": "agent:foreign"})
    with pytest.raises(SpendAdmissionBlockedError, match="outside"):
        SessionSpendAdapter.attach(SpendConfig(), foreign)


def test_attach_rejects_missing_ledger_without_creating_it(tmp_path) -> None:
    missing = tmp_path / "missing"
    context = SpendProcessContext(
        ledger_path=str(missing.resolve()),
        session_scope_id="session:missing",
        agent_scope_id="agent:missing",
        purpose=SpendPurpose.PRIMARY,
    )

    with pytest.raises(SpendAdmissionBlockedError, match="missing ledger"):
        SessionSpendAdapter.attach(SpendConfig(), context)

    assert not missing.exists()


def test_attach_rejects_task_hash_rebound_in_process_context(tmp_path) -> None:
    original_hash = "a" * 64
    tampered_hash = "b" * 64
    limits = SpendEnvelopeLimits(max_calls=1)
    parent = _adapter(tmp_path)
    child = parent.child_agent(
        agent_id="agent:task-bound", limits=limits, task_brief_hash=original_hash
    )
    context = child.export_process_context()

    assert context.task_brief_hash == original_hash
    attached = SessionSpendAdapter.attach(
        SpendConfig(),
        context,
        required_task_brief_hash=original_hash,
        required_limits=limits,
    )
    assert attached.agent_scope_id == child.agent_scope_id

    tampered = context.model_copy(update={"task_brief_hash": tampered_hash})
    with pytest.raises(SpendAdmissionBlockedError, match="agent task brief"):
        SessionSpendAdapter.attach(
            SpendConfig(),
            tampered,
            required_task_brief_hash=tampered_hash,
            required_limits=limits,
        )


@pytest.mark.asyncio
async def test_reissued_task_reuses_and_tightens_immutable_scope(tmp_path) -> None:
    parent = _adapter(tmp_path, max_calls=10)
    brief_hash = "c" * 64
    first = parent.child_task(
        task_brief_hash=brief_hash, limits=SpendEnvelopeLimits(max_calls=2)
    )
    await first.complete(_CountingBackend(), _request(max_tokens=1))

    tightened = parent.child_task(
        task_brief_hash=brief_hash, limits=SpendEnvelopeLimits(max_calls=1)
    )
    attempted_widen = parent.child_task(
        task_brief_hash=brief_hash, limits=SpendEnvelopeLimits(max_calls=5)
    )
    envelope = SpendBroker(parent.ledger_path).get_envelope(first.agent_scope_id)

    assert tightened.agent_scope_id == first.agent_scope_id
    assert attempted_widen.agent_scope_id == first.agent_scope_id
    assert envelope is not None
    assert envelope.task_brief_hash == brief_hash
    assert envelope.limits.max_calls == 1
    with pytest.raises(SpendBudgetExceededError):
        await attempted_widen.complete(_CountingBackend(), _request(max_tokens=1))

    parent.child_task(
        task_brief_hash=brief_hash,
        task_id="host-task-1",
        limits=SpendEnvelopeLimits(max_calls=5),
    )
    with pytest.raises(SpendLedgerConflictError):
        parent.child_task(
            task_brief_hash="d" * 64,
            task_id="host-task-1",
            limits=SpendEnvelopeLimits(max_calls=5),
        )


@pytest.mark.asyncio
async def test_concurrent_task_scope_creation_tightens_atomically(tmp_path) -> None:
    parent = _adapter(tmp_path, max_calls=10)
    brief_hash = "e" * 64

    children = await asyncio.gather(
        *(
            asyncio.to_thread(
                parent.child_task,
                task_brief_hash=brief_hash,
                limits=SpendEnvelopeLimits(max_calls=max_calls),
            )
            for max_calls in (5, 3, 4, 2)
        )
    )
    scope_ids = {child.agent_scope_id for child in children}
    envelope = SpendBroker(parent.ledger_path).get_envelope(children[0].agent_scope_id)

    assert len(scope_ids) == 1
    assert envelope is not None
    assert envelope.limits.max_calls == 2


@pytest.mark.asyncio
async def test_child_agent_limits_reject_without_tightening_parent(tmp_path) -> None:
    parent = _adapter(tmp_path, max_calls=2)
    child = parent.child_agent(
        agent_id="agent:bounded", limits=SpendEnvelopeLimits(max_calls=0)
    )
    backend = _CountingBackend()

    with pytest.raises(SpendBudgetExceededError) as exc_info:
        await child.complete(backend, _request(max_tokens=1))
    await parent.complete(_CountingBackend(), _request(max_tokens=1))

    assert exc_info.value.rejection.limited_scope_id == child.agent_scope_id
    assert backend.complete_calls == 0


def test_attached_processes_share_parent_call_cap(tmp_path) -> None:
    parent = _adapter(tmp_path, max_calls=1, max_concurrent_calls=2)
    child = parent.child_agent(agent_id="agent:process")
    encoded = child.export_process_context().model_dump_json()
    process_context = multiprocessing.get_context("spawn")
    start = process_context.Event()
    results = process_context.Queue()
    processes = [
        process_context.Process(
            target=_attached_call_worker, args=(encoded, start, results)
        )
        for _ in range(2)
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
        SpendRejectionReason.CALLS.value
    ]
    assert parent.snapshot().spent_calls == 1


def test_request_estimate_is_below_semantic_byte_ceiling() -> None:
    tools = [
        AvailableTool(
            function=AvailableFunction(
                name="lookup",
                description="search café",
                parameters={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        )
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }
    request = _request(tools=tools, response_format=response_format)
    footprint = request_prompt_footprint(request)

    assert estimate_request_tokens(request) <= footprint.strict_tokens
    assert estimate_request_tokens(request) >= len("hello é".encode())


def test_request_estimate_includes_file_image_base64_expansion(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    image_bytes = b"x" * 301
    write_safe(image_path, image_bytes.decode())
    image = ImageAttachment(
        source=FileImageSource(path=image_path),
        alias="image.png",
        mime_type="image/png",
    )
    request = CompletionRequest(
        model=_model(),
        messages=[LLMMessage(role=Role.USER, content="inspect", images=[image])],
    )
    without_image = CompletionRequest(
        model=_model(), messages=[LLMMessage(role=Role.USER, content="inspect")]
    )

    assert estimate_request_tokens(request) > estimate_request_tokens(without_image)


def _large_request(
    *, content_size: int = 180_000, model_name: str = "spend-test"
) -> CompletionRequest:
    model = _model().model_copy(update={"name": model_name, "alias": model_name})
    return CompletionRequest(
        model=model,
        messages=[LLMMessage(role=Role.USER, content="x" * content_size)],
        max_tokens=1,
    )


@pytest.mark.asyncio
async def test_adaptive_estimator_admits_long_cached_session_under_explicit_cap(
    tmp_path,
) -> None:
    config = SpendConfig(
        max_prompt_tokens=400_000,
        max_completion_tokens=100_000,
        max_total_tokens=500_000,
        max_cost_usd=1_000.0,
        max_calls=20,
    )
    usage = LLMUsage(prompt_tokens=45_000, cached_tokens=40_000, completion_tokens=1)
    result = LLMChunk(message=LLMMessage(role=Role.ASSISTANT), usage=usage)
    request = _large_request()
    adaptive_backend = _CountingBackend(result=result)
    adaptive = SessionSpendAdapter.create(
        config, "adaptive-session", ledger_path=tmp_path / "adaptive-ledger"
    )

    for _ in range(8):
        await adaptive.complete(adaptive_backend, request)

    reservations = [
        event.reservation for event in adaptive.events() if event.kind == "reserved"
    ]
    assert adaptive_backend.complete_calls == 8
    assert reservations[0].prompt_estimate is not None
    assert reservations[-1].prompt_estimate is not None
    assert (
        reservations[-1].prompt_estimate.estimated_tokens
        < reservations[0].prompt_estimate.estimated_tokens
    )
    assert reservations[-1].prompt_estimate.factor > 1.15
    assert reservations[-1].estimate.cost_usd == pytest.approx(
        reservations[-1].estimate.prompt_tokens / 1_000_000 + 2 / 1_000_000
    )
    assert adaptive.snapshot().spent.cached_tokens == 320_000

    strict_backend = _CountingBackend(result=result)
    strict = SessionSpendAdapter.create(
        config.model_copy(update={"prompt_estimator_mode": PromptEstimatorMode.STRICT}),
        "strict-session",
        ledger_path=tmp_path / "strict-ledger",
    )
    for _ in range(5):
        await strict.complete(strict_backend, request)
    with pytest.raises(SpendBudgetExceededError) as exc_info:
        await strict.complete(strict_backend, request)

    assert exc_info.value.rejection.reason == SpendRejectionReason.PROMPT_TOKENS
    assert strict_backend.complete_calls == 5


@pytest.mark.asyncio
async def test_prompt_calibration_replays_and_isolates_models(tmp_path) -> None:
    config = SpendConfig(max_cost_usd=1_000.0, max_calls=10)
    result = LLMChunk(
        message=LLMMessage(role=Role.ASSISTANT),
        usage=LLMUsage(prompt_tokens=25_000, completion_tokens=1),
    )
    ledger_path = tmp_path / "replay-ledger"
    first = SessionSpendAdapter.create(
        config, "replay-session", ledger_path=ledger_path
    )
    await first.complete(
        _CountingBackend(result=result), _large_request(content_size=100_000)
    )

    resumed = SessionSpendAdapter.create(
        config, "replay-session", ledger_path=ledger_path
    )
    await resumed.complete(
        _CountingBackend(result=result), _large_request(content_size=100_000)
    )
    await resumed.complete(
        _CountingBackend(result=result),
        _large_request(content_size=100_000, model_name="other-model"),
    )

    reservations = [
        event.reservation for event in resumed.events() if event.kind == "reserved"
    ]
    assert reservations[0].prompt_estimate is not None
    assert reservations[1].prompt_estimate is not None
    assert reservations[2].prompt_estimate is not None
    assert reservations[0].prompt_estimate.sample_count == 0
    assert reservations[1].prompt_estimate.sample_count == 1
    assert reservations[2].prompt_estimate.sample_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, LLMUsage()])
async def test_absent_or_zero_usage_does_not_train_prompt_estimator(
    tmp_path, usage: LLMUsage | None
) -> None:
    adapter = _adapter(tmp_path, max_calls=3)
    missing = LLMChunk(message=LLMMessage(role=Role.ASSISTANT), usage=usage)
    request = _large_request(content_size=20_000)

    await adapter.complete(_CountingBackend(result=missing), request)
    await adapter.complete(_CountingBackend(), request)

    reservations = [
        event.reservation for event in adapter.events() if event.kind == "reserved"
    ]
    assert reservations[1].prompt_estimate is not None
    assert reservations[1].prompt_estimate.sample_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_none_max_tokens_dispatches_admitted_bound_without_mutating_caller(
    tmp_path, streaming: bool
) -> None:
    adapter = _adapter(tmp_path, default_max_output_tokens=123)
    backend = _CountingBackend()
    request = _request()

    if streaming:
        _ = [chunk async for chunk in adapter.complete_streaming(backend, request)]
    else:
        await adapter.complete(backend, request)

    assert request.max_tokens is None
    assert backend.requests[0] is not request
    assert backend.requests[0].max_tokens == 123
    reserved = next(event for event in adapter.events() if event.kind == "reserved")
    assert reserved.reservation.estimate.completion_tokens == 123


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_budget_rejection_never_invokes_backend(
    tmp_path, streaming: bool
) -> None:
    adapter = _adapter(tmp_path, max_calls=0)
    backend = _CountingBackend()

    with pytest.raises(SpendBudgetExceededError) as exc_info:
        if streaming:
            _ = [
                chunk async for chunk in adapter.complete_streaming(backend, _request())
            ]
        else:
            await adapter.complete(backend, _request())

    assert exc_info.value.rejection.reason == SpendRejectionReason.CALLS
    assert "configured call limit is reached" in str(exc_info.value)
    assert "session:" not in str(exc_info.value)
    assert backend.complete_calls == 0
    assert backend.streaming_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_usage", [False, True])
async def test_dispatched_error_or_missing_usage_reconciles_estimate_once(
    tmp_path, missing_usage: bool
) -> None:
    result = LLMChunk(message=LLMMessage(role=Role.ASSISTANT), usage=None)
    backend = _CountingBackend(
        result=result, error=None if missing_usage else RuntimeError("provider failed")
    )
    adapter = _adapter(tmp_path, default_max_output_tokens=50)

    if missing_usage:
        await adapter.complete(backend, _request())
    else:
        with pytest.raises(RuntimeError, match="provider failed"):
            await adapter.complete(backend, _request())

    snapshot = adapter.snapshot()
    events = adapter.events()
    reservation = next(
        event.reservation for event in events if event.kind == "reserved"
    )
    reconciled = [event for event in events if event.kind == "reconciled"]
    assert backend.complete_calls == 1
    assert snapshot.reserved_calls == 0
    assert snapshot.spent_calls == 1
    assert snapshot.spent == reservation.estimate
    assert len(reconciled) == 1
    assert reconciled[0].estimated is True


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_inflight_call_renews_lease_until_provider_finishes(
    tmp_path, monkeypatch, streaming: bool
) -> None:
    monkeypatch.setattr("vibe.core.usage._session._RESERVATION_LEASE_S", 0.08)
    monkeypatch.setattr("vibe.core.usage._session._RESERVATION_RENEW_INTERVAL_S", 0.02)
    adapter = _adapter(tmp_path, max_concurrent_calls=1)
    gate = asyncio.Event()
    backend = _CountingBackend(gate=gate)

    async def invoke() -> None:
        if streaming:
            _ = [
                chunk async for chunk in adapter.complete_streaming(backend, _request())
            ]
            return
        await adapter.complete(backend, _request())

    provider_call = asyncio.create_task(invoke())
    while backend.complete_calls + backend.streaming_calls < 1:
        await asyncio.sleep(0)
    await asyncio.sleep(0.18)

    snapshot = adapter.snapshot()
    assert snapshot.reserved_calls == 1
    assert snapshot.spent_calls == 0
    assert any(event.kind == "lease_renewed" for event in adapter.events())

    gate.set()
    await provider_call


@pytest.mark.asyncio
async def test_child_workflow_adapter_cannot_borrow_past_parent_cap(tmp_path) -> None:
    parent = _adapter(tmp_path, max_calls=1)
    child = parent.child_agent(
        group_kind=SpendScopeKind.WORKFLOW,
        group_id="workflow:test",
        purpose=SpendPurpose.WORKFLOW,
    )
    await parent.complete(_CountingBackend(), _request(max_tokens=1))
    child_backend = _CountingBackend()

    with pytest.raises(SpendBudgetExceededError) as exc_info:
        await child.complete(child_backend, _request(max_tokens=1))

    assert child.ledger_path == parent.ledger_path
    assert child.session_scope_id == parent.session_scope_id
    assert exc_info.value.rejection.limited_scope_id == parent.session_scope_id
    assert child_backend.complete_calls == 0


@pytest.mark.asyncio
async def test_explicit_retry_bits_remain_correct_under_concurrency(tmp_path) -> None:
    adapter = _adapter(tmp_path, max_calls=2, max_concurrent_calls=2, max_retries=1)
    gate = asyncio.Event()
    backend = _CountingBackend(gate=gate)
    initial = asyncio.create_task(
        adapter.complete(backend, _request(max_tokens=1), is_retry=False)
    )
    retry = asyncio.create_task(
        adapter.complete(backend, _request(max_tokens=1), is_retry=True)
    )
    while backend.complete_calls < 2:
        await asyncio.sleep(0)

    snapshot = adapter.snapshot()
    gate.set()
    await asyncio.gather(initial, retry)

    assert snapshot.reserved_calls == 2
    assert snapshot.reserved_retries == 1


@pytest.mark.asyncio
async def test_agent_loop_retry_state_is_isolated_between_shared_children(
    tmp_path,
) -> None:
    parent = _adapter(tmp_path, max_calls=3, max_retries=1)
    first_adapter = parent.child_agent()
    second_adapter = parent.child_agent()
    first_backend = _CountingBackend(error=RuntimeError("provider failed"))
    second_backend = _CountingBackend()
    first = build_test_agent_loop(backend=first_backend, spend_adapter=first_adapter)
    second = build_test_agent_loop(backend=second_backend, spend_adapter=second_adapter)

    with pytest.raises(UnclassifiedBackendError, match="provider failed"):
        await first._chat()
    await second._chat()
    first_backend.error = None
    await first._chat()

    reservations = [
        event.reservation for event in parent.events() if event.kind == "reserved"
    ]
    first_retries = [
        reservation.is_retry
        for reservation in reservations
        if reservation.scope_id == first_adapter.agent_scope_id
    ]
    second_retries = [
        reservation.is_retry
        for reservation in reservations
        if reservation.scope_id == second_adapter.agent_scope_id
    ]
    assert first_retries == [False, True]
    assert second_retries == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"max_session_tokens": 1}, SpendRejectionReason.PROMPT_TOKENS),
        ({"max_price": 0.0}, SpendRejectionReason.COST_USD),
    ],
)
async def test_agent_loop_runtime_caps_reject_before_dispatch(
    tmp_path, params: dict, reason: SpendRejectionReason
) -> None:
    backend = FakeBackend(mock_llm_chunk())
    agent = build_test_agent_loop(backend=backend, **params)

    with pytest.raises(SpendBudgetExceededError) as exc_info:
        await agent._chat()

    assert exc_info.value.rejection.reason == reason
    if reason is SpendRejectionReason.PROMPT_TOKENS:
        assert "adaptive prompt estimate" in str(exc_info.value)
        assert "before dispatch" in str(exc_info.value)
    assert backend.requests_messages == []


def test_resume_rebinds_usage_and_spend_to_loaded_session(tmp_path) -> None:
    agent = build_test_agent_loop()
    initial_ledger = agent._spend_adapter.ledger_path

    agent.resume_existing_session("resumed-session", "parent-session", tmp_path)

    assert agent.session_id == "resumed-session"
    assert agent.parent_session_id == "parent-session"
    assert agent._usage_meter.session_id == "resumed-session"
    assert agent._spend_adapter.session_scope_id == "session:resumed-session"
    assert agent._spend_adapter.ledger_path != initial_ledger


@pytest.mark.asyncio
async def test_compaction_rotation_resume_reuses_root_spend_ledger(tmp_path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        ),
        spend=SpendConfig(max_calls=1),
    )
    root_adapter = SessionSpendAdapter.create(config.spend, "root-spend-session")
    await root_adapter.complete(_CountingBackend(), _request(max_tokens=1))
    agent = build_test_agent_loop(config=config, spend_adapter=root_adapter)
    old_session_id = agent.session_id

    await agent._reset_session()
    rotated_session_id = agent.session_id
    agent.messages.append(LLMMessage(role=Role.USER, content="continue after compact"))
    await agent.session_logger.save_interaction(
        agent.messages,
        agent.stats,
        agent.base_config,
        agent.tool_manager,
        agent.agent_profile,
    )
    session_path = agent.session_logger.session_dir
    assert session_path is not None
    assert agent.session_logger.session_metadata is not None
    assert (
        agent.session_logger.session_metadata.environment["spend_session_id"]
        == "root-spend-session"
    )

    backend = _CountingBackend()
    resumed = build_test_agent_loop(config=config, backend=backend)
    resumed.resume_existing_session(rotated_session_id, old_session_id, session_path)

    assert resumed._spend_adapter.ledger_path == root_adapter.ledger_path
    assert resumed._spend_adapter.session_scope_id == "session:root-spend-session"
    with pytest.raises(SpendBudgetExceededError):
        await resumed._chat()
    assert backend.complete_calls == 0


@pytest.mark.asyncio
async def test_reopening_session_tightens_caps_and_preserves_deadline(tmp_path) -> None:
    now = 100.0

    def clock() -> float:
        return now

    config = SpendConfig(deadline_seconds=10.0)
    ledger_path = tmp_path / "deadline-ledger"
    first = SessionSpendAdapter.create(
        config, "deadline-session", ledger_path=ledger_path, clock=clock
    )
    await first.complete(_CountingBackend(), _request(max_tokens=1))
    original_deadline = first.snapshot().envelope.limits.deadline_at

    now = 105.0
    resumed = SessionSpendAdapter.create(
        config,
        "deadline-session",
        ledger_path=ledger_path,
        clock=clock,
        runtime_max_cost_usd=0.0,
        runtime_max_total_tokens=1,
    )

    assert original_deadline == 110.0
    limits = resumed.snapshot().envelope.limits
    assert limits.deadline_at == original_deadline
    assert limits.max_total_tokens == 1
    assert limits.max_cost_usd == 0.0
    backend = _CountingBackend()
    with pytest.raises(SpendBudgetExceededError):
        await resumed.complete(backend, _request(max_tokens=1))
    assert backend.complete_calls == 0


def test_reopening_legacy_session_migrates_omitted_default_token_caps(tmp_path) -> None:
    ledger_path = tmp_path / "legacy-ledger"
    broker = SpendBroker(ledger_path)
    broker.define_envelope(
        SpendEnvelope(
            scope_id="session:legacy-session",
            kind=SpendScopeKind.SESSION,
            limits=SpendEnvelopeLimits(
                max_prompt_tokens=400_000,
                max_completion_tokens=100_000,
                max_total_tokens=500_000,
                max_cost_usd=10.0,
                max_calls=128,
                max_concurrent_calls=2,
                max_retries=16,
            ),
        )
    )

    resumed = SessionSpendAdapter.create(
        SpendConfig(), "legacy-session", ledger_path=ledger_path
    )

    envelope = resumed.snapshot().envelope
    assert envelope.policy_version == 2
    assert envelope.limits.max_prompt_tokens is None
    assert envelope.limits.max_completion_tokens is None
    assert envelope.limits.max_total_tokens is None
    assert (
        len([
            event
            for event in resumed.events()
            if event.kind == "envelope_policy_migrated"
        ])
        == 1
    )


def test_reopening_legacy_session_preserves_explicit_default_token_caps(
    tmp_path,
) -> None:
    ledger_path = tmp_path / "explicit-legacy-ledger"
    broker = SpendBroker(ledger_path)
    legacy_limits = SpendEnvelopeLimits(
        max_prompt_tokens=400_000,
        max_completion_tokens=100_000,
        max_total_tokens=500_000,
        max_cost_usd=10.0,
        max_calls=128,
        max_concurrent_calls=2,
        max_retries=16,
    )
    broker.define_envelope(
        SpendEnvelope(
            scope_id="session:explicit-legacy-session",
            kind=SpendScopeKind.SESSION,
            limits=legacy_limits,
        )
    )

    resumed = SessionSpendAdapter.create(
        SpendConfig(
            max_prompt_tokens=400_000,
            max_completion_tokens=100_000,
            max_total_tokens=500_000,
        ),
        "explicit-legacy-session",
        ledger_path=ledger_path,
    )

    envelope = resumed.snapshot().envelope
    assert envelope.policy_version == 1
    assert envelope.limits == legacy_limits
    assert not any(
        event.kind == "envelope_policy_migrated" for event in resumed.events()
    )


@pytest.mark.asyncio
async def test_live_config_reload_sets_current_spend_adapter(tmp_path) -> None:
    initial = build_test_vibe_config(spend=SpendConfig(max_calls=2))
    adapter = SessionSpendAdapter.create(
        initial.spend, "reload-session", ledger_path=tmp_path / "reload-ledger"
    )
    agent = build_test_agent_loop(config=initial, spend_adapter=adapter)
    await adapter.complete(_CountingBackend(), _request(max_tokens=1))
    tightened = initial.model_copy(update={"spend": SpendConfig(max_calls=1)})

    await agent.reload_with_initial_messages(base_config=tightened)

    assert agent._spend_adapter is adapter
    assert adapter.snapshot().envelope.limits.max_calls == 1
    backend = _CountingBackend()
    with pytest.raises(SpendBudgetExceededError):
        await adapter.complete(backend, _request(max_tokens=1))
    assert backend.complete_calls == 0

    # set_limits can raise, unlike tighten: reloading back to the original
    # higher limit must take effect so a session can recover from a cap.
    await agent.reload_with_initial_messages(base_config=initial)
    assert adapter.snapshot().envelope.limits.max_calls == 2


@pytest.mark.asyncio
async def test_sync_config_refresh_sets_current_spend_adapter(tmp_path) -> None:
    initial = build_test_vibe_config(spend=SpendConfig(max_calls=2))
    adapter = SessionSpendAdapter.create(
        initial.spend, "refresh-session", ledger_path=tmp_path / "refresh-ledger"
    )
    agent = build_test_agent_loop(config=initial, spend_adapter=adapter)
    await adapter.complete(_CountingBackend(), _request(max_tokens=1))
    tightened = initial.model_copy(update={"spend": SpendConfig(max_calls=1)})

    with patch("vibe.core.agent_loop.VibeConfig.load", return_value=tightened):
        agent.refresh_config()

    assert agent.base_config == tightened
    assert adapter.snapshot().envelope.limits.max_calls == 1
    backend = _CountingBackend()
    with pytest.raises(SpendBudgetExceededError):
        await adapter.complete(backend, _request(max_tokens=1))
    assert backend.complete_calls == 0


@pytest.mark.asyncio
async def test_failed_live_spend_set_blocks_later_dispatches(tmp_path) -> None:
    initial = build_test_vibe_config(spend=SpendConfig(max_calls=2))
    adapter = SessionSpendAdapter.create(
        initial.spend, "failed-reload", ledger_path=tmp_path / "failed-ledger"
    )
    child = adapter.child_agent()
    agent = build_test_agent_loop(config=initial, spend_adapter=adapter)
    tightened = initial.model_copy(update={"spend": SpendConfig(max_calls=1)})

    with (
        patch.object(
            adapter._broker,
            "replace_envelope_limits",
            side_effect=RuntimeError("ledger unavailable"),
        ),
        pytest.raises(RuntimeError, match="ledger unavailable"),
    ):
        await agent.reload_with_initial_messages(base_config=tightened)

    backend = _CountingBackend()
    with pytest.raises(RuntimeError, match="spend admission is blocked"):
        await adapter.complete(backend, _request(max_tokens=1))
    assert backend.complete_calls == 0
    with pytest.raises(RuntimeError, match="spend admission is blocked"):
        await child.complete(backend, _request(max_tokens=1))
    assert backend.complete_calls == 0


@pytest.mark.asyncio
async def test_reset_spend_starts_fresh_ledger_without_clearing_history(
    tmp_path,
) -> None:
    initial = build_test_vibe_config(spend=SpendConfig(max_calls=2))
    adapter = SessionSpendAdapter.create(
        initial.spend, "reset-session", ledger_path=tmp_path / "reset-ledger"
    )
    agent = build_test_agent_loop(config=initial, spend_adapter=adapter)
    await adapter.complete(_CountingBackend(), _request(max_tokens=1))
    assert adapter.snapshot().spent_calls == 1

    old_session_id = adapter.spend_session_id
    new_session_id = agent.reset_spend()

    assert new_session_id != old_session_id
    assert agent._spend_adapter.spend_session_id == new_session_id
    assert agent._spend_adapter.snapshot().spent_calls == 0
    # Fresh ledger has the full configured budget again.
    assert (
        agent._spend_adapter.snapshot().envelope.limits.max_calls
        == initial.spend.max_calls
    )
    # A call that would have exceeded the old budget now succeeds.
    await agent._spend_adapter.complete(_CountingBackend(), _request(max_tokens=1))
    assert agent._spend_adapter.snapshot().spent_calls == 1


def test_cost_usd_rejection_message_quotes_projected_total(tmp_path) -> None:
    from vibe.core.usage._session import _spend_rejection_message

    rejection = SpendRejection(
        call_id="c1",
        scope_id="agent:s:test:primary",
        scope_chain=("session:test", "agent:s:test:primary"),
        purpose=SpendPurpose.PRIMARY,
        estimate=SpendAmount(cost_usd=1.0),
        is_retry=False,
        reason=SpendRejectionReason.COST_USD,
        timestamp=0.0,
        projected_cost_usd=9.50,
        limit_cost_usd=8.00,
    )
    message = _spend_rejection_message(rejection)
    assert "$9.5000" in message
    assert "$8.0000" in message
    assert "$1.0000" not in message


def test_calls_rejection_message_quotes_projected_and_limit() -> None:
    from vibe.core.usage._session import _spend_rejection_message

    rejection = SpendRejection(
        call_id="c1",
        scope_id="agent:s:test:primary",
        scope_chain=("session:test", "agent:s:test:primary"),
        purpose=SpendPurpose.PRIMARY,
        estimate=SpendAmount(),
        is_retry=False,
        reason=SpendRejectionReason.CALLS,
        timestamp=0.0,
        projected_calls=129,
        limit_calls=128,
    )
    message = _spend_rejection_message(rejection)
    assert "129" in message
    assert "128" in message


@pytest.mark.asyncio
async def test_agent_loop_records_usage_once_while_broker_tracks_admission(
    tmp_path,
) -> None:
    original = get_usage_recorder()
    recorder = UsageRecorder(tmp_path / "usage.jsonl")
    reset_usage_recorder_for_tests(recorder)
    try:
        backend = FakeBackend(mock_llm_chunk(prompt_tokens=12, completion_tokens=3))
        agent = build_test_agent_loop(backend=backend)

        await agent._chat()

        records = recorder.read_all()
        snapshot = agent._spend_adapter.snapshot()
        assert len(records) == 1
        assert records[0].prompt_tokens == 12
        assert records[0].completion_tokens == 3
        assert snapshot.spent.prompt_tokens == 12
        assert snapshot.spent.completion_tokens == 3
        assert snapshot.spent_calls == 1
    finally:
        reset_usage_recorder_for_tests(original)


def test_task_and_workflow_in_process_children_inherit_session_adapter(
    tmp_path,
) -> None:
    parent = _adapter(tmp_path)
    config = build_test_vibe_config()
    manager = AgentManager(lambda: config)
    ctx = InvokeContext(
        tool_call_id="task-call",
        agent_manager=manager,
        session_id="parent",
        spend_adapter=parent,
    )
    task = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    with (
        patch("vibe.core.tools.builtins.task.AgentLoop") as task_loop,
        patch("vibe.core.tools.builtins.task.VibeConfig.load", return_value=config),
    ):
        task._build_subagent_loop(TaskArgs(task="inspect", agent="explore"), ctx)
    task_adapter = task_loop.call_args.kwargs["params"].spend_adapter

    runtime = WorkflowRuntime(parent_context=ctx)
    with patch("vibe.core.agent_loop.AgentLoop") as workflow_loop:
        runtime._create_real_loop(agent="explore", base_config=config)
    workflow_adapter = workflow_loop.call_args.kwargs["params"].spend_adapter

    for child in (task_adapter, workflow_adapter):
        assert child is not None
        assert child.ledger_path == parent.ledger_path
        assert child.session_scope_id == parent.session_scope_id
        assert child.agent_scope_id != parent.agent_scope_id
    assert workflow_adapter.default_purpose == SpendPurpose.WORKFLOW


def test_later_paid_call_integration_boundaries_are_explicit() -> None:
    assert UNROUTED_PAID_CALL_BOUNDARIES == {
        "mcp_sampling",
        "transcription",
        "tts",
        "websearch",
    }


def test_production_paid_calls_are_brokered_or_documented() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    observed: list[tuple[str, str, str]] = []
    for path in sorted((repository_root / "vibe").rglob("*.py")):
        if "bundled" in path.parts:
            continue
        source = read_safe(path, raise_on_error=True).text
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr not in {
                "complete",
                "complete_streaming",
                "complete_async",
                "start_async",
                "stream_async",
                "transcribe_stream",
            }:
                continue
            observed.append((
                str(path.relative_to(repository_root)),
                node.func.attr,
                ast.unparse(node.func.value),
            ))

    assert sorted(observed) == [
        ("vibe/cli/turn_summary/tracker.py", "complete", "spend_adapter"),
        ("vibe/core/agent_loop.py", "complete", "self._spend_adapter"),
        ("vibe/core/agent_loop.py", "complete_streaming", "self._spend_adapter"),
        ("vibe/core/llm/backend/generic.py", "complete_streaming", "self"),
        (
            "vibe/core/llm/backend/mistral.py",
            "complete_async",
            "self._get_client().chat",
        ),
        ("vibe/core/llm/backend/mistral.py", "stream_async", "self._get_client().chat"),
        (
            "vibe/core/tools/builtins/websearch.py",
            "start_async",
            "client.beta.conversations",
        ),
        ("vibe/core/tools/mcp_sampling.py", "complete", "self._backend_getter()"),
        (
            "vibe/core/transcribe/mistral_transcribe_client.py",
            "transcribe_stream",
            "client.audio.realtime",
        ),
        (
            "vibe/core/tts/mistral_tts_client.py",
            "complete_async",
            "client.audio.speech",
        ),
        ("vibe/core/usage/_auxiliary.py", "complete", "spend_adapter"),
        ("vibe/core/usage/_session.py", "complete", "backend"),
        ("vibe/core/usage/_session.py", "complete_streaming", "backend"),
        ("vibe/core/workflows/_formatter_repair.py", "complete", "spend_adapter"),
        ("vibe/core/workflows/_semantic_repair.py", "complete", "spend_adapter"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("collect", [False, True])
async def test_task_maps_spend_rejection_to_blocked_outcome(collect: bool) -> None:
    rejection = SpendRejection(
        call_id="denied",
        scope_id="agent:test",
        purpose=SpendPurpose.PRIMARY,
        estimate=SpendAmount(prompt_tokens=1),
        is_retry=False,
        reason=SpendRejectionReason.CALLS,
        limited_scope_id="session:test",
        timestamp=1.0,
    )

    class _DeniedLoop:
        def __init__(self) -> None:
            self.messages: list[LLMMessage] = []

        async def act(self, _prompt: str):
            raise SpendBudgetExceededError(rejection)
            yield

        async def aclose(self) -> None:
            return None

    brief = TaskBrief(
        objective="inspect the cap",
        allowed_paths=["vibe"],
        acceptance_checks=["report the result"],
        manifest=TaskManifestIdentity(name="test", version="1"),
    )
    args = TaskArgs(task=brief, agent="explore", async_run=False)
    ctx = InvokeContext(tool_call_id="task-call")
    task = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    with patch.object(
        Task, "_build_subagent_loop", return_value=(_DeniedLoop(), "prompt")
    ):
        if collect:
            result = await task._run_in_process_collect(args, ctx)
            outcome = result.outcome
        else:
            result = await collect_result(task._run_in_process(args, ctx))
            outcome = result.outcome

    assert outcome is not None
    assert outcome.status == TaskOutcomeStatus.BLOCKED
    assert any("calls" in diagnostic for diagnostic in outcome.diagnostics)


@pytest.mark.asyncio
async def test_attached_team_task_children_preserve_group_and_shared_cap(
    tmp_path: Path,
) -> None:
    root = _adapter(tmp_path, max_calls=1)
    teammate_a = root.child_agent(
        group_kind=SpendScopeKind.TEAM,
        group_id="team:shared-cap",
        agent_id="agent:teammate-a",
        purpose=SpendPurpose.TEAM,
    )
    teammate_b = root.child_agent(
        group_kind=SpendScopeKind.TEAM,
        group_id="team:shared-cap",
        agent_id="agent:teammate-b",
        purpose=SpendPurpose.TEAM,
    )
    attached_a = SessionSpendAdapter.attach(
        SpendConfig(), teammate_a.export_process_context()
    )
    attached_b = SessionSpendAdapter.attach(
        SpendConfig(), teammate_b.export_process_context()
    )
    task_a = attached_a.child_agent(agent_id="agent:task-a")
    task_b = attached_b.child_agent(agent_id="agent:task-b")
    broker = SpendBroker(root.ledger_path)
    scope_a = broker.get_envelope(task_a.agent_scope_id)
    scope_b = broker.get_envelope(task_b.agent_scope_id)

    assert scope_a is not None
    assert scope_b is not None
    assert scope_a.parent_scope_id == "team:shared-cap"
    assert scope_b.parent_scope_id == "team:shared-cap"

    await task_a.complete(_CountingBackend(), _request(max_tokens=1))
    with pytest.raises(SpendBudgetExceededError):
        await task_b.complete(_CountingBackend(), _request(max_tokens=1))
