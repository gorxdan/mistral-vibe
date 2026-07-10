from __future__ import annotations

import asyncio
from pathlib import Path
import types
from unittest.mock import patch

import orjson
from pydantic import ValidationError
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import collect_result, mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.manager import AgentManager
from vibe.core.config import ModelConfig, SessionLoggingConfig, SpendConfig
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
    Role,
    UnclassifiedBackendError,
)
from vibe.core.usage import (
    UsageRecorder,
    get_usage_recorder,
    reset_usage_recorder_for_tests,
)
from vibe.core.usage._context import (
    SpendAmount,
    SpendPurpose,
    SpendRejection,
    SpendRejectionReason,
    SpendScopeKind,
)
from vibe.core.usage._session import (
    UNROUTED_PAID_CALL_BOUNDARIES,
    SessionSpendAdapter,
    SpendBudgetExceededError,
    estimate_request_tokens,
)
from vibe.core.utils.io import write_safe
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


def test_spend_config_defaults_are_finite() -> None:
    config = SpendConfig()

    assert config.max_prompt_tokens == 400_000
    assert config.max_completion_tokens == 100_000
    assert config.max_total_tokens == 500_000
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


def test_request_estimate_is_serialized_utf8_byte_upper_bound() -> None:
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
    serialized = orjson.dumps({
        "messages": [
            message.model_dump(mode="json", exclude_none=True)
            for message in request.messages
        ],
        "tools": [
            tool.model_dump(mode="json", exclude_none=True)
            for tool in request.tools or []
        ],
        "response_format": response_format,
    })

    assert estimate_request_tokens(request) == len(serialized)
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
    payload = {
        "messages": [
            message.model_dump(mode="json", exclude_none=True)
            for message in request.messages
        ],
        "tools": [],
        "response_format": None,
    }
    expanded_size = 4 * ((len(image_bytes) + 2) // 3) + len(b"data:image/png;base64,")

    assert (
        estimate_request_tokens(request) == len(orjson.dumps(payload)) + expanded_size
    )


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


@pytest.mark.asyncio
async def test_live_config_reload_tightens_current_spend_adapter(tmp_path) -> None:
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

    await agent.reload_with_initial_messages(base_config=initial)
    assert adapter.snapshot().envelope.limits.max_calls == 1


@pytest.mark.asyncio
async def test_sync_config_refresh_tightens_current_spend_adapter(tmp_path) -> None:
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
async def test_failed_live_spend_tightening_blocks_later_dispatches(tmp_path) -> None:
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
            "tighten_envelope",
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
        "isolated_subprocess",
        "mcp_sampling",
        "narration",
    }


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
