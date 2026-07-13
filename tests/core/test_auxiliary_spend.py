from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types
from typing import Any

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from vibe.core.config import ModelConfig, ProviderConfig, SafetyJudgeConfig, SpendConfig
from vibe.core.memory.consolidator import MemoryConsolidator
from vibe.core.memory.extractor import MemoryExtractor
from vibe.core.memory.selector import MemorySelector
from vibe.core.tools.base import BaseToolState
from vibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from vibe.core.tools.safety_judge import SafetyJudge
from vibe.core.types import (
    ApprovalResponse,
    Backend,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
)
from vibe.core.usage import SpendLimits, SpendPurpose, UsageMeter, UsageRecorder
from vibe.core.usage._session import SessionSpendAdapter


@dataclass
class _BackendProbe:
    result: LLMChunk
    complete_calls: int = 0

    def backend_class(self) -> type[Any]:
        probe = self

        class _Backend:
            def __init__(self, **_: object) -> None:
                pass

            async def __aenter__(self) -> _Backend:
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: types.TracebackType | None,
            ) -> None:
                return None

            async def complete(self, *args: object, **kwargs: object) -> LLMChunk:
                probe.complete_calls += 1
                return probe.result

        return _Backend


class _RecordingApproval:
    def __init__(self) -> None:
        self.called = False

    async def __call__(
        self, *args: object, **kwargs: object
    ) -> tuple[ApprovalResponse, None, None]:
        self.called = True
        return ApprovalResponse.NO, None, None


def _model() -> ModelConfig:
    return ModelConfig(
        name="aux-model",
        provider="aux-provider",
        alias="aux-model",
        input_price=1.0,
        output_price=2.0,
    )


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="aux-provider", api_base="https://example.test", backend=Backend.GENERIC
    )


def _adapter(
    tmp_path: Path, *, max_calls: int = 128, session_id: str = "aux-session"
) -> SessionSpendAdapter:
    return SessionSpendAdapter.create(
        SpendConfig(max_calls=max_calls, enforce_limits=True),
        session_id,
        ledger_path=tmp_path / "ledger",
    )


def _meter(tmp_path: Path) -> tuple[UsageMeter, UsageRecorder]:
    recorder = UsageRecorder(tmp_path / "usage.jsonl")
    meter = UsageMeter(
        "aux-session",
        limits=SpendLimits(max_tokens=100_000, max_cost_usd=10.0, max_calls=20),
        recorder=recorder,
    )
    return meter, recorder


def _result(content: str, usage: LLMUsage | None) -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.ASSISTANT, content=content), usage=usage
    )


def test_agent_loop_routes_memory_and_safety_with_distinct_purposes() -> None:
    config = build_test_vibe_config(
        models=[_model()],
        providers=[_provider()],
        safety_judge=SafetyJudgeConfig(enabled=True, model="aux-model"),
    )
    loop = build_test_agent_loop(config=config)

    memory_clients = [
        (loop._resolve_memory_selector(), SpendPurpose.MEMORY_RECALL),
        (loop._resolve_memory_extractor(), SpendPurpose.MEMORY_EXTRACT),
        (loop._resolve_memory_consolidator(), SpendPurpose.MEMORY_CONSOLIDATE),
        (loop._resolve_memory_verifier(), SpendPurpose.MEMORY_VERIFY),
    ]

    for client, purpose in memory_clients:
        assert client is not None
        assert client._spend_adapter is loop._spend_adapter
        assert client._spend_purpose == purpose
    judge = loop._resolve_safety_judge()
    assert judge is not None
    assert judge._spend_adapter is loop._spend_adapter


@pytest.mark.asyncio
async def test_memory_without_broker_fails_closed_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = _BackendProbe(
        _result('{"ids": ["one"]}', LLMUsage(prompt_tokens=12, completion_tokens=3))
    )
    monkeypatch.setattr(
        "vibe.core.memory._llm_client.BACKEND_FACTORY",
        {Backend.GENERIC: probe.backend_class()},
    )
    selector = MemorySelector(
        model=_model(), provider=_provider(), max_selected=1, spend_adapter=None
    )

    selected = await selector.select(["- [one] useful"], "question", {"one"})

    assert selected == []
    assert probe.complete_calls == 0


@pytest.mark.asyncio
async def test_memory_broker_rejection_fails_open_without_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    probe = _BackendProbe(
        _result('{"ids": ["one"]}', LLMUsage(prompt_tokens=12, completion_tokens=3))
    )
    monkeypatch.setattr(
        "vibe.core.memory._llm_client.BACKEND_FACTORY",
        {Backend.GENERIC: probe.backend_class()},
    )
    adapter = _adapter(tmp_path, max_calls=0)
    meter, recorder = _meter(tmp_path)
    selector = MemorySelector(
        model=_model(),
        provider=_provider(),
        max_selected=1,
        usage_meter=meter,
        spend_adapter=adapter,
    )

    selected = await selector.select(["- [one] useful"], "question", {"one"})

    assert selected == []
    assert probe.complete_calls == 0
    assert meter.snapshot().calls == 0
    assert meter.snapshot().reserved_tokens == 0
    assert recorder.read_all() == []
    rejected = [event for event in adapter.events() if event.kind == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].rejection.purpose == SpendPurpose.MEMORY_RECALL


@pytest.mark.asyncio
async def test_exact_auxiliary_usage_reconciles_once_in_each_accounting_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usage = LLMUsage(prompt_tokens=19, completion_tokens=7)
    probe = _BackendProbe(_result('{"memories": []}', usage))
    monkeypatch.setattr(
        "vibe.core.memory._llm_client.BACKEND_FACTORY",
        {Backend.GENERIC: probe.backend_class()},
    )
    adapter = _adapter(tmp_path)
    meter, recorder = _meter(tmp_path)
    extractor = MemoryExtractor(
        model=_model(), provider=_provider(), usage_meter=meter, spend_adapter=adapter
    )

    assert await extractor.extract("user: remember this", "") == []

    records = recorder.read_all()
    reconciled = [event for event in adapter.events() if event.kind == "reconciled"]
    reserved = [event for event in adapter.events() if event.kind == "reserved"]
    assert probe.complete_calls == 1
    assert len(records) == 1
    assert records[0].call_kind == "memory_extract"
    assert records[0].prompt_tokens == 19
    assert records[0].completion_tokens == 7
    assert meter.snapshot().tokens == 26
    assert meter.snapshot().reserved_tokens == 0
    assert len(reserved) == 1
    assert reserved[0].reservation.purpose == SpendPurpose.MEMORY_EXTRACT
    assert len(reconciled) == 1
    assert reconciled[0].estimated is False
    assert reconciled[0].amount.prompt_tokens == 19
    assert reconciled[0].amount.completion_tokens == 7


@pytest.mark.asyncio
async def test_missing_auxiliary_usage_retains_each_estimate_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    probe = _BackendProbe(_result('{"actions": []}', None))
    monkeypatch.setattr(
        "vibe.core.memory._llm_client.BACKEND_FACTORY",
        {Backend.GENERIC: probe.backend_class()},
    )
    adapter = _adapter(tmp_path)
    meter, recorder = _meter(tmp_path)
    consolidator = MemoryConsolidator(
        model=_model(), provider=_provider(), usage_meter=meter, spend_adapter=adapter
    )

    assert (
        await consolidator.consolidate(["- [one] useful"], "[one] body", {"one"}) == []
    )

    records = recorder.read_all()
    reserved = next(event for event in adapter.events() if event.kind == "reserved")
    reconciled = [event for event in adapter.events() if event.kind == "reconciled"]
    assert probe.complete_calls == 1
    assert len(records) == 1
    assert records[0].call_kind == "memory_consolidate"
    assert records[0].total_tokens == meter.snapshot().tokens
    assert meter.snapshot().reserved_tokens == 0
    assert reserved.reservation.purpose == SpendPurpose.MEMORY_CONSOLIDATE
    assert len(reconciled) == 1
    assert reconciled[0].estimated is True
    assert reconciled[0].amount == reserved.reservation.estimate


@pytest.mark.asyncio
async def test_shared_parent_cap_defers_safety_judge_to_human_approval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_probe = _BackendProbe(
        _result('{"ids": ["one"]}', LLMUsage(prompt_tokens=9, completion_tokens=2))
    )
    safety_probe = _BackendProbe(
        _result(
            '{"safe": true, "reason": "read-only"}',
            LLMUsage(prompt_tokens=20, completion_tokens=5),
        )
    )
    monkeypatch.setattr(
        "vibe.core.memory._llm_client.BACKEND_FACTORY",
        {Backend.GENERIC: memory_probe.backend_class()},
    )
    monkeypatch.setattr(
        "vibe.core.tools.safety_judge.BACKEND_FACTORY",
        {Backend.GENERIC: safety_probe.backend_class()},
    )
    adapter = _adapter(tmp_path, max_calls=1)
    meter, recorder = _meter(tmp_path)
    selector = MemorySelector(
        model=_model(),
        provider=_provider(),
        max_selected=1,
        usage_meter=meter,
        spend_adapter=adapter,
    )
    assert await selector.select(["- [one] useful"], "question", {"one"}) == ["one"]

    judge = SafetyJudge(
        model=_model(),
        provider=_provider(),
        config=SafetyJudgeConfig(enabled=True, model="aux-model"),
        usage_meter=meter,
        spend_adapter=adapter,
    )
    loop = build_test_agent_loop()
    monkeypatch.setattr(loop, "_resolve_safety_judge", lambda: judge)
    approval = _RecordingApproval()
    loop.approval_callback = approval
    bash = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState())

    decision = await loop._should_execute_tool(
        bash, BashArgs(command="npm install"), "call-1"
    )

    assert decision.verdict.value == "skip"
    assert approval.called is True
    assert safety_probe.complete_calls == 0
    assert meter.snapshot().calls == 1
    assert meter.snapshot().reserved_tokens == 0
    assert len(recorder.read_all()) == 1
    snapshot = adapter.snapshot()
    assert snapshot.spent_calls == 1
    assert snapshot.rejected_calls == 1
    rejected = [event for event in adapter.events() if event.kind == "rejected"]
    assert rejected[-1].rejection.purpose == SpendPurpose.SAFETY_JUDGE
