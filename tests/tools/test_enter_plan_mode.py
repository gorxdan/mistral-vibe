from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel
import pytest

from tests.mock.utils import collect_result
from vibe.core.agents.models import AgentProfile, AgentSafety, BuiltinAgentName
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.ask_user_question import Answer, AskUserQuestionResult
from vibe.core.tools.builtins.enter_plan_mode import (
    EnterPlanMode,
    EnterPlanModeArgs,
    EnterPlanModeConfig,
)


@dataclass
class MockAgentManager:
    active_profile: AgentProfile
    _switched_to: list[str] = field(default_factory=list)

    def switch_profile(self, name: str) -> None:
        self._switched_to.append(name)
        self.active_profile = AgentProfile(
            name=name,
            display_name=name.title(),
            description="",
            safety=AgentSafety.SAFE,
        )


def _plan_profile() -> AgentProfile:
    return AgentProfile(
        name=BuiltinAgentName.PLAN,
        display_name="Plan",
        description="Plan mode",
        safety=AgentSafety.SAFE,
    )


def _default_profile() -> AgentProfile:
    return AgentProfile(
        name=BuiltinAgentName.DEFAULT,
        display_name="Default",
        description="Default mode",
        safety=AgentSafety.SAFE,
    )


@pytest.fixture
def tool() -> EnterPlanMode:
    return EnterPlanMode(
        config_getter=lambda: EnterPlanModeConfig(), state=BaseToolState()
    )


@pytest.fixture
def default_manager() -> MockAgentManager:
    return MockAgentManager(active_profile=_default_profile())


class MockCallback:
    def __init__(self, result: AskUserQuestionResult) -> None:
        self._result = result
        self.received_args: BaseModel | None = None

    async def __call__(self, args: BaseModel) -> BaseModel:
        self.received_args = args
        return self._result


class TestErrorCases:
    @pytest.mark.asyncio
    async def test_requires_agent_manager(self, tool: EnterPlanMode) -> None:
        ctx = InvokeContext(
            tool_call_id="t1",
            user_input_callback=MockCallback(
                AskUserQuestionResult(answers=[], cancelled=True)
            ),
        )
        with pytest.raises(ToolError, match="agent manager"):
            await collect_result(tool.run(EnterPlanModeArgs(), ctx))

    @pytest.mark.asyncio
    async def test_rejects_when_already_in_plan_mode(self, tool: EnterPlanMode) -> None:
        manager = MockAgentManager(active_profile=_plan_profile())
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=manager,  # type: ignore[arg-type]
            user_input_callback=MockCallback(
                AskUserQuestionResult(answers=[], cancelled=True)
            ),
        )
        with pytest.raises(ToolError, match="plan mode"):
            await collect_result(tool.run(EnterPlanModeArgs(), ctx))

    @pytest.mark.asyncio
    async def test_requires_interactive_ui(
        self, tool: EnterPlanMode, default_manager: MockAgentManager
    ) -> None:
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=default_manager,  # type: ignore[arg-type]
        )
        with pytest.raises(ToolError, match="interactive UI"):
            await collect_result(tool.run(EnterPlanModeArgs(), ctx))


class MockSwitchAgentCallback:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, name: str) -> None:
        self.calls.append(name)


class TestAnswerHandling:
    @pytest.mark.asyncio
    async def test_yes_uses_switch_agent_callback(
        self, tool: EnterPlanMode, default_manager: MockAgentManager
    ) -> None:
        switch_cb = MockSwitchAgentCallback()
        cb = MockCallback(
            AskUserQuestionResult(
                answers=[
                    Answer(
                        question="q", answer="Yes, switch to plan mode", is_other=False
                    )
                ],
                cancelled=False,
            )
        )
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=default_manager,  # type: ignore[arg-type]
            user_input_callback=cb,
            switch_agent_callback=switch_cb,
        )
        result = await collect_result(tool.run(EnterPlanModeArgs(), ctx))
        assert result.switched is True
        assert switch_cb.calls == [BuiltinAgentName.PLAN]
        assert default_manager._switched_to == []

    @pytest.mark.asyncio
    async def test_yes_falls_back_to_switch_profile(
        self, tool: EnterPlanMode, default_manager: MockAgentManager
    ) -> None:
        cb = MockCallback(
            AskUserQuestionResult(
                answers=[
                    Answer(
                        question="q", answer="Yes, switch to plan mode", is_other=False
                    )
                ],
                cancelled=False,
            )
        )
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=default_manager,  # type: ignore[arg-type]
            user_input_callback=cb,
        )
        result = await collect_result(tool.run(EnterPlanModeArgs(), ctx))
        assert result.switched is True
        assert default_manager._switched_to == [BuiltinAgentName.PLAN]

    @pytest.mark.asyncio
    async def test_no_stays_in_current_mode(
        self, tool: EnterPlanMode, default_manager: MockAgentManager
    ) -> None:
        cb = MockCallback(
            AskUserQuestionResult(
                answers=[Answer(question="q", answer="No", is_other=False)],
                cancelled=False,
            )
        )
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=default_manager,  # type: ignore[arg-type]
            user_input_callback=cb,
        )
        result = await collect_result(tool.run(EnterPlanModeArgs(), ctx))
        assert result.switched is False
        assert default_manager._switched_to == []

    @pytest.mark.asyncio
    async def test_cancelled_stays(
        self, tool: EnterPlanMode, default_manager: MockAgentManager
    ) -> None:
        cb = MockCallback(AskUserQuestionResult(answers=[], cancelled=True))
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=default_manager,  # type: ignore[arg-type]
            user_input_callback=cb,
        )
        result = await collect_result(tool.run(EnterPlanModeArgs(), ctx))
        assert result.switched is False
        assert default_manager._switched_to == []

    @pytest.mark.asyncio
    async def test_other_includes_feedback(
        self, tool: EnterPlanMode, default_manager: MockAgentManager
    ) -> None:
        cb = MockCallback(
            AskUserQuestionResult(
                answers=[
                    Answer(question="q", answer="Just do it directly", is_other=True)
                ],
                cancelled=False,
            )
        )
        ctx = InvokeContext(
            tool_call_id="t1",
            agent_manager=default_manager,  # type: ignore[arg-type]
            user_input_callback=cb,
        )
        result = await collect_result(tool.run(EnterPlanModeArgs(), ctx))
        assert result.switched is False
        assert "Just do it directly" in result.message
