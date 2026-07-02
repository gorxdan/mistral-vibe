from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from vibe.core.loop import (
    MIN_INTERVAL_SECONDS,
    LoopError,
    format_duration,
    parse_interval,
)
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.types import ToolStreamEvent


class ScheduleArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: Literal["create", "list", "cancel"]
    interval: str | None = Field(
        default=None,
        description="For create: <number><unit> — e.g. 30s, 1m, 5m, 10m, 2h, 1d.",
    )
    prompt: str | None = Field(
        default=None,
        description="For create: the instruction to run when the timer fires.",
    )
    recurring: bool = Field(
        default=True,
        description="For create: True repeats every interval; False fires once.",
    )
    target: str | None = Field(
        default=None, description="For cancel: a loop id, or 'all'."
    )


class ScheduleResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str
    id: str | None = None
    message: str
    scheduled: list[str] = Field(default_factory=list)


class ScheduleConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class Schedule(BaseTool[ScheduleArgs, ScheduleResult, ScheduleConfig, BaseToolState]):
    manifest_deferrable: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Schedule a future turn instead of blocking on `sleep`. `create` arms a "
        "timer that re-prompts you with `prompt` after `interval` (recurring, or "
        "once); `list` shows armed timers; `cancel` removes one (or 'all'). The "
        "harness fires the turn at the interval — you never sleep or block. Use "
        "this for 'check again in 5m', to revisit a long-running workflow later "
        "(instead of polling workflow_status), or any wait — NOT `sleep`. Min "
        f"interval {MIN_INTERVAL_SECONDS}s."
    )

    async def run(
        self, args: ScheduleArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ScheduleResult, None]:
        scheduler = ctx.scheduler if ctx is not None else None
        if scheduler is None:
            yield ScheduleResult(
                action=args.action,
                message=(
                    "Scheduling is unavailable in this context (it needs an "
                    "interactive session). Do not use `sleep` to wait — continue "
                    "without blocking, or ask the user to schedule a follow-up."
                ),
            )
            return

        if args.action == "list":
            loops = scheduler.loops
            yield ScheduleResult(
                action="list",
                message=f"{len(loops)} scheduled timer(s)",
                scheduled=[
                    f"{lp.id}: {'every' if lp.recurring else 'once in'} "
                    f"{format_duration(lp.interval_seconds)} — {lp.prompt}"
                    for lp in loops
                ],
            )
            return

        if args.action == "cancel":
            if not args.target:
                raise ToolError("cancel requires 'target' (a loop id or 'all').")
            count = await scheduler.cancel(args.target)
            yield ScheduleResult(
                action="cancel",
                message=f"cancelled {count} timer(s)" if count else "no match",
            )
            return

        # create
        if not args.interval or not args.prompt:
            raise ToolError("create requires 'interval' and 'prompt'.")
        try:
            seconds = parse_interval(args.interval)
            loop = await scheduler.add_loop(
                seconds, args.prompt, recurring=args.recurring
            )
        except LoopError as e:
            raise ToolError(str(e)) from e
        kind = "every" if args.recurring else "once in"
        yield ScheduleResult(
            action="create",
            id=loop.id,
            message=(
                f"scheduled `{loop.id}` {kind} {format_duration(seconds)}: "
                f"{loop.prompt}"
            ),
        )
