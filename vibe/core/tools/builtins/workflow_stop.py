from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.types import ToolResultEvent, ToolStreamEvent

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


class WorkflowStopArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    run_id: str | None = Field(
        default=None,
        description=(
            "The workflow run id to stop (e.g. 'wf-1'). Ignored when `all` is "
            "true. Exactly one of `run_id` and `all` must be provided."
        ),
    )
    all: bool = Field(default=False, description="Stop every active workflow run.")

    @model_validator(mode="after")
    def _require_target(self) -> WorkflowStopArgs:
        if not self.all and not self.run_id:
            raise ValueError("workflow_stop requires either a run_id or all=true.")
        return self


class WorkflowStopResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stopped: bool = Field(description="True if at least one run was stopped.")
    stopped_run_ids: list[str] = Field(
        default_factory=list, description="The run ids that were stopped."
    )
    message: str = Field(description="Human-readable summary of the outcome.")


class WorkflowStopConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class WorkflowStop(
    BaseTool[WorkflowStopArgs, WorkflowStopResult, WorkflowStopConfig, BaseToolState],
    ToolUIData[WorkflowStopArgs, WorkflowStopResult],
):
    description: ClassVar[str] = (
        "Stop one or all background workflow runs. Cancels the run's asyncio "
        "task, halting any in-flight agents immediately. Use this to recover "
        "from a runaway or misbehaving workflow instead of waiting for it to "
        "finish or exceed its budget. Pass a `run_id` for one run, or `all` "
        "for every active run. Already-finished runs are reported as not stopped."
    )

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        # Detail lives in the on-demand `workflow-authoring` skill; the schema
        # description is enough for the always-on baseline.
        return None

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        # Mirrors launch_workflow / workflow_status: hidden when workflows are
        # disabled, since there is nothing to stop.
        if config is None:
            return True
        return not getattr(config, "disable_workflows", False)

    @classmethod
    def get_call_display(cls, event: Any) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, WorkflowStopArgs):
            if args.all:
                return ToolCallDisplay(summary="Stopping all workflows")
            return ToolCallDisplay(summary=f"Stopping workflow: {args.run_id}")
        return ToolCallDisplay(summary="Stopping workflow")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, WorkflowStopResult):
            return ToolResultDisplay(success=result.stopped, message=result.message)
        return ToolResultDisplay(success=True, message="Workflow stop")

    @classmethod
    def get_status_text(cls) -> str:
        return "Stopping workflow"

    def resolve_permission(self, args: WorkflowStopArgs) -> PermissionContext | None:
        return PermissionContext(permission=ToolPermission.ASK)

    async def run(
        self, args: WorkflowStopArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WorkflowStopResult, None]:
        if ctx is None:
            raise ToolError("Workflow stop tool requires context")
        callback: Callable[[str | None, bool], Awaitable[dict[str, Any]]] | None = (
            ctx.workflow_stop_callback
        )
        if callback is None:
            raise ToolError(
                "Workflow stop is not available in this context "
                "(no stop callback wired). Use /workflows stop instead."
            )
        data = await callback(args.run_id, args.all)
        yield WorkflowStopResult(**data)
