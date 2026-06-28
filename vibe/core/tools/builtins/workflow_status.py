from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

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


class WorkflowStatusArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    run_id: str | None = Field(
        default=None,
        description=(
            "Optional workflow run id (e.g. 'wf-1'). If omitted, return the "
            "live status of every run (running and finished)."
        ),
    )


class WorkflowStatusResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    runs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One status dict per matching run, newest-relevant first.",
    )


class WorkflowStatusConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class WorkflowStatus(
    BaseTool[
        WorkflowStatusArgs, WorkflowStatusResult, WorkflowStatusConfig, BaseToolState
    ],
    ToolUIData[WorkflowStatusArgs, WorkflowStatusResult],
):
    read_only: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Diagnostic snapshot of background workflow runs: per-run agent count, "
        "phase breakdown, in-flight agents with their running token totals, and "
        "budget. This is NOT for routine progress checks — workflow completion "
        "is auto-delivered to your context, so do not poll this waiting for a "
        "run to finish. Call it ONCE when you suspect a run is stuck or "
        "runaway (before deciding to workflow_stop it), or to inspect a run "
        "the auto-delivery never reached. To revisit a long run later without "
        "blocking, arm a `schedule` timer rather than polling. Pass a run_id "
        "for one run, or omit it for all runs."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        # Mirrors launch_workflow: hidden when workflows are disabled, since
        # there is nothing to report on.
        if config is None:
            return True
        return not getattr(config, "disable_workflows", False)

    @classmethod
    def get_call_display(cls, event: Any) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, WorkflowStatusArgs) and args.run_id:
            return ToolCallDisplay(summary=f"Workflow status: {args.run_id}")
        return ToolCallDisplay(summary="Workflow status (all runs)")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, WorkflowStatusResult):
            n = len(result.runs)
            return ToolResultDisplay(
                success=True, message=f"Workflow status: {n} run(s)."
            )
        return ToolResultDisplay(success=True, message="Workflow status")

    @classmethod
    def get_status_text(cls) -> str:
        return "Querying workflow status"

    def resolve_permission(self, args: WorkflowStatusArgs) -> PermissionContext | None:
        return PermissionContext(permission=ToolPermission.ALWAYS)

    async def run(
        self, args: WorkflowStatusArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WorkflowStatusResult, None]:
        if ctx is None:
            raise ToolError("Workflow status tool requires context")
        if ctx.workflow_status_callback is None:
            raise ToolError(
                "Workflow status is not available in this context "
                "(no status callback wired)."
            )
        runs = ctx.workflow_status_callback(args.run_id)
        yield WorkflowStatusResult(runs=runs)
