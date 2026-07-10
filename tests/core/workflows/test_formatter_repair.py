from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from vibe.core.agents.manager import AgentManager
from vibe.core.config import (
    ModelConfig,
    ProviderConfig,
    PurposeModelRoutingConfig,
    VibeConfig,
)
from vibe.core.llm.types import CompletionRequest
from vibe.core.repair import RepairController
from vibe.core.tools.base import InvokeContext
from vibe.core.types import (
    AssistantEvent,
    Backend,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
)
from vibe.core.usage._context import (
    SpendAmount,
    SpendPurpose,
    SpendRejection,
    SpendRejectionReason,
)
from vibe.core.usage._session import SessionSpendAdapter, SpendBudgetExceededError
from vibe.core.workflows._formatter_repair import format_workflow_result
from vibe.core.workflows._result_repair import (
    WorkflowRepairHandoff,
    repair_progress_snapshot,
    repair_workflow_result,
)
from vibe.core.workflows.models import SchemaValidationFailure
from vibe.core.workflows.runtime import WorkflowRuntime

pytestmark = pytest.mark.asyncio

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


@dataclass
class _Backend:
    provider: ProviderConfig

    async def __aenter__(self) -> _Backend:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


@dataclass
class _SpendAdapter:
    outcomes: list[LLMChunk | Exception]
    calls: list[tuple[_Backend, CompletionRequest, SpendPurpose | None, bool]] = field(
        default_factory=list
    )

    async def complete(
        self,
        backend: _Backend,
        request: CompletionRequest,
        *,
        purpose: SpendPurpose | None = None,
        is_retry: bool = False,
    ) -> LLMChunk:
        self.calls.append((backend, request, purpose, is_retry))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class _Stats:
    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0


@dataclass
class _SequenceLoop:
    responses: list[str]
    stats: _Stats = field(default_factory=_Stats)
    calls: int = 0

    async def act(
        self, prompt: str, *, response_format: Any = None
    ) -> AsyncGenerator[AssistantEvent, None]:
        del prompt, response_format
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        self.stats.session_prompt_tokens += 20
        self.stats.session_completion_tokens += 10
        yield AssistantEvent(content=response, message_id=f"a{self.calls}")


class _Manager:
    def __init__(self, config: VibeConfig) -> None:
        self.config = config

    def get_agent(self, name: str) -> object:
        raise ValueError(name)


def _config(*, formatter_model: str = "cheap", semantic_model: str = "") -> VibeConfig:
    providers = [
        ProviderConfig(
            name="primary-provider",
            api_base="https://primary.invalid",
            backend=Backend.GENERIC,
        ),
        ProviderConfig(
            name="cheap-provider",
            api_base="https://cheap.invalid",
            backend=Backend.GENERIC,
        ),
        ProviderConfig(
            name="strong-provider",
            api_base="https://strong.invalid",
            backend=Backend.GENERIC,
        ),
    ]
    models = [
        ModelConfig(name="primary-model", alias="primary", provider="primary-provider"),
        ModelConfig(
            name="cheap-model",
            alias="cheap",
            provider="cheap-provider",
            max_output_tokens=2_000,
        ),
        ModelConfig(
            name="strong-model",
            alias="strong",
            provider="strong-provider",
            max_output_tokens=4_000,
        ),
    ]
    return VibeConfig(
        active_model="primary",
        providers=providers,
        models=models,
        model_routing=PurposeModelRoutingConfig(
            formatter_model=formatter_model, semantic_escalation_model=semantic_model
        ),
    )


def _chunk(content: str) -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.ASSISTANT, content=content),
        usage=LLMUsage(prompt_tokens=11, completion_tokens=7),
    )


def _handoff(raw: str = "broken json") -> WorkflowRepairHandoff:
    repaired = repair_workflow_result(raw, _SCHEMA, strip_unknown=True)
    assert repaired.diagnostic is not None
    controller = RepairController.with_finite_defaults()
    decision = controller.observe_failure(
        repaired.diagnostic,
        repair_progress_snapshot(repaired.diagnostic, repaired.errors),
        caller_budget_remaining=True,
    )
    return repaired.handoff(_SCHEMA, decision)


def _runtime(
    loop: _SequenceLoop,
    adapter: _SpendAdapter,
    *,
    formatter_model: str = "cheap",
    semantic_model: str = "",
) -> WorkflowRuntime:
    config = _config(formatter_model=formatter_model, semantic_model=semantic_model)
    context = InvokeContext(
        tool_call_id="workflow",
        agent_manager=cast(AgentManager, _Manager(config)),
        spend_adapter=cast(SessionSpendAdapter, adapter),
    )

    def factory(prompt: str, *, agent: str, parent_context: Any | None = None) -> Any:
        del prompt, agent, parent_context
        return loop

    return WorkflowRuntime(
        parent_context=context,
        agent_loop_factory=factory,
        schema_retries=2,
        budget_total=100_000,
    )


async def test_formatter_uses_configured_model_without_tools_or_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[ProviderConfig, float, float]] = []

    def create_backend(
        *, provider: ProviderConfig, timeout: float, retry_max_elapsed_time: float
    ) -> _Backend:
        created.append((provider, timeout, retry_max_elapsed_time))
        return _Backend(provider)

    monkeypatch.setattr(
        "vibe.core.workflows._formatter_repair.create_backend", create_backend
    )
    adapter = _SpendAdapter([_chunk('{"answer":"fixed"}')])
    handoff = _handoff("broken-marker")

    result = await format_workflow_result(
        handoff, config=_config(), spend_adapter=cast(SessionSpendAdapter, adapter)
    )

    assert result.content == '{"answer":"fixed"}'
    assert created[0][0].name == "cheap-provider"
    assert created[0][2] == 0.0
    _, request, purpose, is_retry = adapter.calls[0]
    assert request.model.alias == "cheap"
    assert request.max_tokens == 512
    assert request.tools is None
    assert request.tool_choice is None
    assert purpose is SpendPurpose.REPAIR
    assert is_retry is True
    assert len(request.messages) == 2
    request_text = "\n".join(message.content or "" for message in request.messages)
    assert "broken-marker" in request_text
    assert handoff.diagnostic.for_model() in request_text
    assert "SECRET_PARENT_TRANSCRIPT" not in request_text


async def test_local_repair_does_not_call_formatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.workflows._formatter_repair.create_backend",
        lambda **kwargs: _Backend(kwargs["provider"]),
    )
    adapter = _SpendAdapter([_chunk('{"answer":"unused"}')])
    loop = _SequenceLoop(['{"answer":"local",}'])

    result = await _runtime(loop, adapter).spawn_agent("test", schema=_SCHEMA)

    assert result == {"answer": "local"}
    assert adapter.calls == []


async def test_strong_model_runs_only_after_bounded_semantic_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[ProviderConfig] = []

    def create_backend(**kwargs: Any) -> _Backend:
        created.append(kwargs["provider"])
        return _Backend(kwargs["provider"])

    monkeypatch.setattr(
        "vibe.core.workflows._semantic_repair.create_backend", create_backend
    )
    adapter = _SpendAdapter([_chunk('{"answer":"strong repair"}')])
    loop = _SequenceLoop(['{"answer":7}'])
    runtime = _runtime(loop, adapter, formatter_model="", semantic_model="strong")

    result = await runtime.spawn_agent("test", schema=_SCHEMA)

    assert result == {"answer": "strong repair"}
    assert loop.calls == 3
    assert [provider.name for provider in created] == ["strong-provider"]
    _, request, purpose, is_retry = adapter.calls[0]
    assert request.model.alias == "strong"
    assert request.tools is None
    assert request.max_tokens == 2_048
    assert purpose is SpendPurpose.REPAIR
    assert is_retry is True


async def test_invalid_strong_result_preserves_exact_evidence_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.workflows._semantic_repair.create_backend",
        lambda **kwargs: _Backend(kwargs["provider"]),
    )
    strong_response = '{"answer":false}'
    expected = repair_workflow_result(strong_response, _SCHEMA, strip_unknown=True)
    assert expected.value is None
    assert expected.diagnostic is not None
    adapter = _SpendAdapter([_chunk(strong_response)])
    loop = _SequenceLoop(['{"answer":7}'])
    runtime = _runtime(loop, adapter, formatter_model="", semantic_model="strong")

    result = await runtime.spawn_agent("test", schema=_SCHEMA)

    assert isinstance(result, SchemaValidationFailure)
    assert result.raw_response == strong_response
    assert result.schema_errors == list(expected.errors)
    assert expected.diagnostic.for_model() in result.error
    recorded = runtime._phases["default"].agent_results[0]
    assert recorded.response == strong_response
    assert recorded.schema_errors == list(expected.errors)
    assert recorded.tokens_in == 71
    assert recorded.tokens_out == 37


async def test_valid_formatter_result_avoids_worker_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.workflows._formatter_repair.create_backend",
        lambda **kwargs: _Backend(kwargs["provider"]),
    )
    adapter = _SpendAdapter([_chunk('{"answer":"formatter"}')])
    loop = _SequenceLoop(["broken json", '{"answer":"unused worker"}'])
    runtime = _runtime(loop, adapter)

    result = await runtime.spawn_agent("test", schema=_SCHEMA)

    assert result == {"answer": "formatter"}
    assert loop.calls == 1
    assert len(adapter.calls) == 1
    recorded = runtime._phases["default"].agent_results[0]
    assert recorded.tokens_in == 31
    assert recorded.tokens_out == 17


async def test_broker_rejection_falls_back_to_retained_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.workflows._formatter_repair.create_backend",
        lambda **kwargs: _Backend(kwargs["provider"]),
    )
    rejection = SpendRejection(
        call_id="formatter-denied",
        scope_id="agent:worker",
        purpose=SpendPurpose.REPAIR,
        estimate=SpendAmount(prompt_tokens=1),
        is_retry=True,
        reason=SpendRejectionReason.RETRIES,
        limited_scope_id="session:test",
        timestamp=1.0,
    )
    adapter = _SpendAdapter([SpendBudgetExceededError(rejection)])
    loop = _SequenceLoop(["broken json", '{"answer":"worker"}'])

    result = await _runtime(loop, adapter).spawn_agent("test", schema=_SCHEMA)

    assert result == {"answer": "worker"}
    assert loop.calls == 2
    assert len(adapter.calls) == 1


async def test_malformed_formatter_output_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.workflows._formatter_repair.create_backend",
        lambda **kwargs: _Backend(kwargs["provider"]),
    )
    adapter = _SpendAdapter([_chunk("still broken")])
    loop = _SequenceLoop(["broken json", '{"answer":"worker"}'])

    result = await _runtime(loop, adapter).spawn_agent("test", schema=_SCHEMA)

    assert result == {"answer": "worker"}
    assert loop.calls == 2
    assert len(adapter.calls) == 1


async def test_semantic_schema_failure_never_calls_formatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibe.core.workflows._formatter_repair.create_backend",
        lambda **kwargs: _Backend(kwargs["provider"]),
    )
    adapter = _SpendAdapter([_chunk('{"answer":"unused"}')])
    loop = _SequenceLoop(['{"answer":1}', '{"answer":"worker"}'])

    result = await _runtime(loop, adapter).spawn_agent("test", schema=_SCHEMA)

    assert result == {"answer": "worker"}
    assert loop.calls == 2
    assert adapter.calls == []
