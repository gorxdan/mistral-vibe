from __future__ import annotations

import ast
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, ClassVar

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
from vibe.core.types import ToolCallEvent, ToolResultEvent, ToolStreamEvent
from vibe.core.workflows.security import check_script

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


def _looks_like_path(script: str) -> bool:
    """Detect a common mistake: passing a file path instead of source text.

    ``launch_workflow`` takes the script's *source* inline; it does not read
    files. When a model writes the script to a scratchpad file and then passes
    the path, ``validate_script`` accepts it (a bare ``foo.py`` parses as an
    attribute access) and the run fails later with a confusing "no main()".
    This catches that case up front so the error names the real problem.
    """
    s = script.strip()
    if "\n" in s:
        return False
    return s.endswith(".py") and "def " not in s


class LaunchWorkflowArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    script: str = Field(
        description="The workflow script source code (Python with async def main())"
    )
    name: str | None = Field(
        default=None, description="Optional name for the workflow run"
    )


class LaunchWorkflowResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    run_id: str = Field(description="The ID of the launched workflow run")
    launched: bool = Field(description="Whether the workflow was successfully launched")
    delivery: str = Field(
        description=(
            "How results arrive: the run executes in the background, so the "
            "script's return_value and per-agent outputs are NOT in this "
            "result. They are auto-delivered as a message on completion, and "
            "you can re-read them at any time — including after a missed or "
            "truncated delivery — with workflow_results(run_id=<run_id>)."
        )
    )


class LaunchWorkflowConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class LaunchWorkflow(
    BaseTool[
        LaunchWorkflowArgs, LaunchWorkflowResult, LaunchWorkflowConfig, BaseToolState
    ],
    ToolUIData[LaunchWorkflowArgs, LaunchWorkflowResult],
):
    host_only: ClassVar[bool] = True
    is_subagent_spawner: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Launch a workflow script that orchestrates parallel agents. "
        "Pass the script's SOURCE TEXT in the `script` argument inline (not a "
        "file path). The script must define an `async def main()` function. "
        "The runtime injects: agent, parallel, pipeline, phase, log, budget, "
        "args, plus the synthesis helpers flatten/dedup_by/merge_by. "
        "`parallel`/`pipeline` accept `max_concurrency=N` to cap in-flight "
        "agents. The run is BACKGROUND and fire-and-forget from this tool: the "
        "return_value is NOT in the result — it is auto-delivered on completion "
        "and re-readable via workflow_results(run_id). Note: the script runs in "
        "a sandbox — imports are allowlisted (no `asyncio`; the injected helpers "
        "are already awaitable) and `str.format()` is forbidden (use f-strings "
        "or `%`)."
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
    def get_tool_prompt(cls) -> str | None:
        # The full ~3.2k authoring guide (prompts/launch_workflow.md) is loaded
        # on demand via the `workflow-authoring` skill, not injected into every
        # system prompt. Keep only a concise pointer here so the always-on
        # baseline stays small while the tool remains discoverable + callable.
        return (
            "To author a workflow script, load the `workflow-authoring` skill "
            "first — it is the single source of truth for the script API "
            "(agent/parallel/pipeline/phase/log/budget/args + synthesis "
            "helpers), the sandbox rules (allowlisted imports, no asyncio, no "
            "str.format), and concurrency/rate-limit recovery. Do not write a "
            "script from memory. Pass the script SOURCE inline in `script`; the "
            "run is background and results return via `workflow_results(run_id)`."
        )

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

        if _looks_like_path(args.script):
            raise ToolError(
                "`script` must be the workflow SOURCE TEXT, not a file path "
                f"(got {args.script.strip()!r}). The tool does not read files; "
                "read the file's contents and pass them inline in `script`."
            )

        # Full pre-flight gate: safety AND correctness (undefined names,
        # coroutine-as-thunk). These authoring mistakes pass the safety AST check
        # but crash or silently produce zero agents at exec time — catching them
        # here fails the launch at no cost instead of after spawning.
        violations = check_script(args.script)
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

        yield LaunchWorkflowResult(
            run_id=run_id,
            launched=True,
            delivery=(
                f"Run {run_id} is executing in the background. Its return_value "
                f"and per-agent outputs are auto-delivered on completion; re-read "
                f"them any time with workflow_results(run_id='{run_id}')."
            ),
        )
