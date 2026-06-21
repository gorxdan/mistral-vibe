from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar, cast

from pydantic import BaseModel

from vibe.core.agents.models import BuiltinAgentName
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.builtins.ask_user_question import (
    AskUserQuestionArgs,
    AskUserQuestionResult,
    Choice,
    Question,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData


class EnterPlanModeArgs(BaseModel):
    pass


class EnterPlanModeResult(BaseModel):
    switched: bool
    message: str


class EnterPlanModeConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class EnterPlanMode(
    BaseTool[
        EnterPlanModeArgs, EnterPlanModeResult, EnterPlanModeConfig, BaseToolState
    ],
    ToolUIData[EnterPlanModeArgs, EnterPlanModeResult],
):
    description: ClassVar[str] = (
        "Switch into plan mode to research the codebase and draft an implementation "
        "plan before making any changes. Call this for long or multi-step tasks that "
        "warrant a reviewed plan first -- multi-file refactors, new features, "
        "architecturally significant changes, or anything where you should explore and "
        "confirm the approach with the user before editing. Do not call it for simple "
        "single-file edits, quick lookups, or tasks you can complete confidently in a "
        "turn or two. This asks the user to confirm the switch to read-only plan mode."
    )

    @classmethod
    def format_call_display(cls, args: EnterPlanModeArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary="Request to enter plan mode")

    @classmethod
    def format_result_display(cls, result: EnterPlanModeResult) -> ToolResultDisplay:
        return ToolResultDisplay(success=result.switched, message=result.message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Waiting for user confirmation"

    async def run(
        self, args: EnterPlanModeArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[EnterPlanModeResult, None]:
        if ctx is None or ctx.agent_manager is None:
            raise ToolError("EnterPlanMode requires an agent manager context.")

        if ctx.agent_manager.active_profile.name == BuiltinAgentName.PLAN:
            raise ToolError("EnterPlanMode cannot be used while already in plan mode.")

        if ctx.user_input_callback is None:
            raise ToolError("EnterPlanMode requires an interactive UI.")

        confirmation = AskUserQuestionArgs(
            questions=[
                Question(
                    question="Switch to plan mode to draft an implementation plan first?",
                    header="Plan mode",
                    options=[
                        Choice(
                            label="Yes, switch to plan mode",
                            description="Enter read-only plan mode to research and draft a plan",
                        ),
                        Choice(label="No", description="Continue in the current mode"),
                    ],
                )
            ]
        )

        result = await ctx.user_input_callback(confirmation)
        result = cast(AskUserQuestionResult, result)

        if result.cancelled or not result.answers:
            yield EnterPlanModeResult(
                switched=False, message="User cancelled. Staying in the current mode."
            )
            return

        answer = result.answers[0]
        if answer.answer.lower() == "yes, switch to plan mode":
            if ctx.switch_agent_callback:
                await ctx.switch_agent_callback(BuiltinAgentName.PLAN)
            else:
                ctx.agent_manager.switch_profile(BuiltinAgentName.PLAN)
            yield EnterPlanModeResult(
                switched=True,
                message="Switched to plan mode. Research the task and write your plan to the plan file, then call exit_plan_mode when ready.",
            )
        elif answer.is_other:
            yield EnterPlanModeResult(
                switched=False,
                message=f"Staying in the current mode. User feedback: {answer.answer}",
            )
        else:
            yield EnterPlanModeResult(
                switched=False, message="Staying in the current mode."
            )
