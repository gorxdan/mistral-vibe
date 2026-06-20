from __future__ import annotations

import ast
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, ClassVar

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
from vibe.core.types import ToolCallEvent, ToolResultEvent, ToolStreamEvent
from vibe.core.workflows.security import validate_script

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


def _is_phase_call(node: ast.AST) -> str | None:
    """Return the literal phase name for a ``phase("...")`` call, else None.

    Only literal first-argument strings are recognized, so a script that
    computes phase names dynamically contributes nothing misleading.
    """
    if not isinstance(node, ast.Call):
        return None
    if not (isinstance(node.func, ast.Name) and node.func.id == "phase"):
        return None
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _extract_planned_phases(script: str) -> list[str]:
    """Parse phase(...) names from a workflow script for the approval preview."""
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return []
    phases: list[str] = []
    for node in ast.walk(tree):
        name = _is_phase_call(node)
        if name is not None and name not in phases:
            phases.append(name)
    return phases


class LaunchWorkflowArgs(BaseModel):
    script: str = Field(
        description="The workflow script source code (Python with async def main())"
    )
    name: str | None = Field(
        default=None, description="Optional name for the workflow run"
    )


class LaunchWorkflowResult(BaseModel):
    run_id: str = Field(description="The ID of the launched workflow run")
    launched: bool = Field(description="Whether the workflow was successfully launched")


class LaunchWorkflowConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class LaunchWorkflow(
    BaseTool[
        LaunchWorkflowArgs, LaunchWorkflowResult, LaunchWorkflowConfig, BaseToolState
    ],
    ToolUIData[LaunchWorkflowArgs, LaunchWorkflowResult],
):
    description: ClassVar[str] = (
        "Launch a workflow script that orchestrates parallel agents. "
        "The script must define an `async def main()` function. "
        "The runtime injects: agent, parallel, pipeline, phase, log, budget, args. "
        "Use this when a task needs multiple independent agents, adversarial "
        "verification, or dynamic loops. The workflow runs in the background."
    )

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, LaunchWorkflowArgs):
            name = args.name or "workflow"
            phases = _extract_planned_phases(args.script)
            if phases:
                preview = f"Launching workflow: {name}\nPlanned phases: {' \u2192 '.join(phases)}"
            else:
                preview = f"Launching workflow: {name}"
            return ToolCallDisplay(summary=preview)
        return ToolCallDisplay(summary="Launching workflow")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, LaunchWorkflowResult):
            if result.launched:
                return ToolResultDisplay(
                    success=True, message=f"Workflow launched: {result.run_id}"
                )
            return ToolResultDisplay(success=False, message="Workflow launch failed")
        return ToolResultDisplay(success=True, message="Workflow launched")

    @classmethod
    def get_status_text(cls) -> str:
        return "Launching workflow"

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        if config is None:
            return True
        return not getattr(config, "disable_workflows", False)

    def resolve_permission(self, args: LaunchWorkflowArgs) -> PermissionContext | None:
        return PermissionContext(permission=ToolPermission.ASK)

    async def run(
        self, args: LaunchWorkflowArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | LaunchWorkflowResult, None]:
        if not ctx:
            raise ToolError("Launch workflow tool requires context")

        violations = validate_script(args.script)
        if violations:
            raise ToolError(
                "Script validation failed:\n" + "\n".join(f"  {v}" for v in violations)
            )

        if not ctx.launch_workflow_callback:
            raise ToolError(
                "Workflow launching is not available in this context "
                "(no launch callback wired). Run the script manually via "
                "/workflows or the WorkflowRuntime API."
            )

        run_id = ctx.launch_workflow_callback(args.script, args.name)

        yield LaunchWorkflowResult(run_id=run_id, launched=True)
