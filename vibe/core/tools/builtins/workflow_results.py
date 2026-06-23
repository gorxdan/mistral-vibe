from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

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


class WorkflowResultsArgs(BaseModel):
    run_id: str = Field(
        description=(
            "Workflow run id (e.g. 'wf-1'). Required — there is no 'all runs' "
            "form for results."
        )
    )
    phase: str | None = Field(
        default=None,
        description=(
            "Optional phase name to filter to. When omitted, every phase's "
            "agents are returned."
        ),
    )
    raw: bool = Field(
        default=False,
        description=(
            "Include the full response text for each agent. Default (false) "
            "truncates each agent's response to 4KB so a large run doesn't "
            "flood context. Pass true when you need the complete output."
        ),
    )


class WorkflowResultsResult(BaseModel):
    run_id: str
    status: str = Field(
        description=(
            "Run-level status: running, paused, completed, "
            "completed_with_failures, failed, or stopped."
        )
    )
    phases: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-phase summary: {name, agents, completed, failed}.",
    )
    agent_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "One dict per finalized agent: {label, agent, phase, completed, "
            "response, error, schema_errors, tokens_in, tokens_out}. response "
            "may be truncated (pass raw=true for full text). schema_errors "
            "carries field-level reasons when an agent failed JSON-schema "
            "validation (empty otherwise)."
        ),
    )
    return_value: Any = Field(
        default=None,
        description=(
            "The workflow script's return value (what main() returned), or None "
            "while the run is still in flight. This is the canonical pull path "
            "for a run's result — use it whenever the auto-delivered completion "
            "summary was missed, truncated, or you need the structured output. "
            "Structured values (dict/list) pass through when they fit the cap; "
            "larger values come back as a truncated string unless raw=true."
        ),
    )


class WorkflowResultsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class WorkflowResults(
    BaseTool[
        WorkflowResultsArgs, WorkflowResultsResult, WorkflowResultsConfig, BaseToolState
    ],
    ToolUIData[WorkflowResultsArgs, WorkflowResultsResult],
):
    read_only: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Retrieve a workflow run's outputs: the script's return_value plus "
        "per-agent {label, response, error, schema_errors, tokens}. Use this "
        "as the canonical pull path for a run's result — especially when the "
        "auto-delivered completion summary was missed/truncated, when you need "
        "the structured return value, or to recover why agents failed "
        "(schema_errors carries field-level JSON-validation reasons; failed "
        "agents' raw responses are included). Pass raw=true for untruncated "
        "responses and the full return_value. Prefer this over workflow_status "
        "when you need outputs, not live progress."
    )

    # Per-agent response cap when raw=false. Chosen so a 16-agent batch fits in
    # ~64KB rather than flooding the host's context. raw=true lifts it entirely.
    _DEFAULT_PER_AGENT_CHAR_CAP: ClassVar[int] = 4000
    # return_value cap when raw=false. Larger than the per-agent cap because the
    # return value is the synthesized point of the run; raw=true lifts it.
    _DEFAULT_RETURN_VALUE_CHAR_CAP: ClassVar[int] = 16_000

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        # Mirrors launch_workflow / workflow_status: hidden when workflows are
        # disabled, since there is nothing to retrieve.
        if config is None:
            return True
        return not getattr(config, "disable_workflows", False)

    @classmethod
    def get_call_display(cls, event: Any) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, WorkflowResultsArgs):
            suffix = f", phase={args.phase}" if args.phase else ""
            return ToolCallDisplay(summary=f"Workflow results: {args.run_id}{suffix}")
        return ToolCallDisplay(summary="Workflow results")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, WorkflowResultsResult):
            n = len(result.agent_results)
            failed = sum(1 for r in result.agent_results if not r.get("completed"))
            msg = f"Workflow results: {n} agent(s)"
            if failed:
                msg += f" ({failed} failed)"
            return ToolResultDisplay(success=True, message=msg)
        return ToolResultDisplay(success=True, message="Workflow results")

    @classmethod
    def get_status_text(cls) -> str:
        return "Retrieving workflow results"

    def resolve_permission(self, args: WorkflowResultsArgs) -> PermissionContext | None:
        # Read-only retrieval of run data the host already owns. Low blast
        # radius — no reason to gate behind ASK.
        return PermissionContext(permission=ToolPermission.ALWAYS)

    async def run(
        self, args: WorkflowResultsArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WorkflowResultsResult, None]:
        if ctx is None:
            raise ToolError("Workflow results tool requires context")
        if ctx.workflow_results_callback is None:
            raise ToolError(
                "Workflow results are not available in this context "
                "(no results callback wired)."
            )
        data = ctx.workflow_results_callback(
            args.run_id, phase=args.phase, raw=args.raw
        )
        yield WorkflowResultsResult(**data)
