from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from vibe.core.teams._task_checks import (
    TaskCheckEvidence,
    run_guarded_task_checks,
    task_check_diagnostics,
)
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData


class TaskChecksArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskChecksResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: tuple[TaskCheckEvidence, ...]
    diagnostics: tuple[str, ...]


class TaskChecksConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class TaskChecks(
    BaseTool[TaskChecksArgs, TaskChecksResult, TaskChecksConfig, BaseToolState],
    ToolUIData[TaskChecksArgs, TaskChecksResult],
):
    description: ClassVar[str] = (
        "Run the immutable acceptance checks bound to this structured task. "
        "The harness chooses every command and returns exact bounded evidence."
    )
    runtime_scoped: ClassVar[bool] = True

    @classmethod
    def format_call_display(cls, args: TaskChecksArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary="Run bound task checks")

    @classmethod
    def format_result_display(cls, result: TaskChecksResult) -> ToolResultDisplay:
        message = (
            "Bound task checks passed" if result.passed else "Bound task checks failed"
        )
        return ToolResultDisplay(success=result.passed, message=message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Running bound task checks"

    async def run(
        self, args: TaskChecksArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[TaskChecksResult, None]:
        if ctx is None:
            raise ToolError("task_checks requires invocation context")
        contract = ctx.task_contract
        if contract is None:
            raise ToolError("task_checks requires a host-bound structured task")
        checks, mutation = await asyncio.to_thread(
            run_guarded_task_checks, contract.trusted_checks, contract.workspace_root
        )
        diagnostics = list(task_check_diagnostics(checks))
        if mutation is not None:
            diagnostics.append(mutation)
        yield TaskChecksResult(
            passed=mutation is None and bool(checks) and all(c.passed for c in checks),
            checks=checks,
            diagnostics=tuple(diagnostics),
        )


__all__ = ["TaskChecks", "TaskChecksArgs", "TaskChecksConfig", "TaskChecksResult"]
