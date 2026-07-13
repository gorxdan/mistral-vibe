from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict
import pytest

from tests.conftest import build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import BUILTIN_AGENTS, BuiltinAgentName
from vibe.core.loop import LoopManager
from vibe.core.output_formatters import TextOutputFormatter
from vibe.core.programmatic import (
    ProgrammaticOptions,
    _drive_programmatic_turn,
    _new_programmatic_loop,
    _wire_isolated_approval,
)
from vibe.core.types import ApprovalResponse, AssistantEvent, ToolResultEvent


class _NoScheduledLoops:
    def __init__(self) -> None:
        self.loops: list[object] = []


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ApprovalLoop:
    def __init__(self, agent_name: str) -> None:
        self.callback: Any = None
        self.agent_profile = BUILTIN_AGENTS[agent_name]

    def set_approval_callback(self, callback: Any) -> None:
        self.callback = callback


class _SkippedThenPassLoop:
    def __init__(self) -> None:
        self.pass_emitted = False

    async def act(self, prompt: str):
        yield ToolResultEvent(
            tool_name="bash",
            tool_class=None,
            tool_call_id="denied-cleanup",
            skipped=True,
            skip_reason="policy denied",
        )
        self.pass_emitted = True
        yield AssistantEvent(content="VERDICT: PASS")


def test_programmatic_host_profile_permission_does_not_mark_loop_as_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIBE_ISOLATED_AUTO_APPROVE", raising=False)
    monkeypatch.delenv("VIBE_TEAMMATE_NAME", raising=False)
    config = build_test_vibe_config(effort_mode="le-chaton")

    loop = _new_programmatic_loop(
        config,
        ProgrammaticOptions(allow_subagent=True, headless=True),
        TextOutputFormatter(),
        backend=FakeBackend(),
    )

    assert loop._is_subagent is False
    assert "work_strategy" in loop.tool_manager.available_tools


def test_isolated_programmatic_loop_is_scoped_as_subagent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_ISOLATED_AUTO_APPROVE", "1")
    config = build_test_vibe_config(effort_mode="le-chaton")

    loop = _new_programmatic_loop(
        config,
        ProgrammaticOptions(
            agent_name=BuiltinAgentName.WORKER, allow_subagent=True, headless=True
        ),
        TextOutputFormatter(),
        backend=FakeBackend(),
    )

    assert loop._is_subagent is True
    assert "work_strategy" not in loop.tool_manager.available_tools


@pytest.mark.asyncio
async def test_programmatic_verifier_fails_before_pass_after_skipped_tool() -> None:
    loop = _SkippedThenPassLoop()
    formatter = TextOutputFormatter()

    with pytest.raises(RuntimeError, match="Verifier tool call 'bash'.*policy denied"):
        await _drive_programmatic_turn(
            cast(AgentLoop, loop),
            formatter,
            ProgrammaticOptions(agent_name=BuiltinAgentName.VERIFIER),
            "verify",
            cast(LoopManager, _NoScheduledLoops()),
        )

    assert not loop.pass_emitted
    assert formatter.finalize() is None


@pytest.mark.asyncio
async def test_programmatic_nonverifier_keeps_existing_skipped_tool_behavior() -> None:
    loop = _SkippedThenPassLoop()
    formatter = TextOutputFormatter()

    await _drive_programmatic_turn(
        cast(AgentLoop, loop),
        formatter,
        ProgrammaticOptions(agent_name=BuiltinAgentName.REVIEWER),
        "review",
        cast(LoopManager, _NoScheduledLoops()),
    )

    assert loop.pass_emitted
    assert formatter.finalize() == "VERDICT: PASS"


@pytest.mark.asyncio
async def test_isolated_verifier_rejects_ask_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_ISOLATED_AUTO_APPROVE", "1")
    loop = _ApprovalLoop(BuiltinAgentName.VERIFIER)

    _wire_isolated_approval(cast(AgentLoop, loop))

    assert loop.callback is not None
    response, _, _ = await loop.callback("bash", _Args(), "call", None, None)
    assert response is ApprovalResponse.NO


@pytest.mark.asyncio
async def test_isolated_worker_still_approves_ask_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_ISOLATED_AUTO_APPROVE", "1")
    loop = _ApprovalLoop(BuiltinAgentName.WORKER)

    _wire_isolated_approval(cast(AgentLoop, loop))

    assert loop.callback is not None
    response, _, _ = await loop.callback("edit", _Args(), "call", None, None)
    assert response is ApprovalResponse.YES
