from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar

from vibe.core.config import VibeConfig
from vibe.core.orchestration import (
    OrchestrationDecision,
    OrchestrationState,
    StrategyReceipt,
)
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.types import ToolStreamEvent


class WorkStrategyArgs(OrchestrationDecision):
    pass


class WorkStrategyConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class WorkStrategy(
    BaseTool[WorkStrategyArgs, StrategyReceipt, WorkStrategyConfig, BaseToolState]
):
    host_only: ClassVar[bool] = True
    host_policy_control: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Record the adaptive execution strategy for the current Le Chaton work. "
        "Choose direct for localized or sequentially coupled host work; task for "
        "productive subagent lanes; workflow for staged fan-out or adversarial "
        "cross-checking; team for long-running coordination. The host keeps its "
        "normal tools after declaring a route and must reassess if scope drifts. "
        "Declare the highest plausible risk; risk cannot be downgraded during the "
        "active lifecycle. Start at most two agent-owned evidence lanes and wait "
        "for terminal evidence before replacing an active strategy. A rejected "
        "redeclaration leaves the accepted route and its debt active. "
        "The receipt supplies structural lane bindings: task/team prompts use "
        "[lane:<id>] markers, while workflow agent() calls use literal label='<id>'."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        return config is not None and config.is_le_chaton()

    def resolve_permission(self, args: WorkStrategyArgs) -> PermissionContext | None:
        return PermissionContext(permission=ToolPermission.ALWAYS)

    async def run(
        self, args: WorkStrategyArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | StrategyReceipt, None]:
        if ctx is None or ctx.work_strategy_callback is None:
            raise ToolError("Work strategy recording is not available in this context")
        decision = OrchestrationDecision.model_validate(args.model_dump())
        try:
            yield ctx.work_strategy_callback(decision)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc


__all__ = [
    "OrchestrationDecision",
    "OrchestrationState",
    "StrategyReceipt",
    "WorkStrategy",
    "WorkStrategyArgs",
    "WorkStrategyConfig",
]
