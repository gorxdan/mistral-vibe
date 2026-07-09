from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, ClassVar

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


class TeamSpawnArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Name for the teammate to spawn.")
    prompt: str = Field(description="Initial task prompt for the teammate.")
    agent: str = Field(
        default="auto-approve", description="Agent profile for the teammate subprocess."
    )
    max_turns: int = Field(
        default=20,
        description=(
            "Maximum turns per task (worker mode) or for the whole one-shot run."
        ),
        ge=1,
    )
    worker: bool = Field(
        default=False,
        description=(
            "If true, spawn a long-lived queue worker (VIBE_TEAM_WORKER=1) that "
            "claims tasks from the shared TaskStore until stopped. If false "
            "(default), run a single -p prompt and exit."
        ),
    )


class TeamSpawnResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    team_dir: str
    message: str
    worker: bool = False


class TeamSpawnConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class TeamSpawn(
    BaseTool[TeamSpawnArgs, TeamSpawnResult, TeamSpawnConfig, BaseToolState],
    ToolUIData[TeamSpawnArgs, TeamSpawnResult],
):
    description: ClassVar[str] = (
        "Spawn a teammate subprocess for team-based work. This creates a shared "
        "team workspace so the host can coordinate via team_message and inspect "
        "the teammate through background. Pass worker=true for a long-lived "
        "queue worker that claims TaskStore tasks until stopped."
    )

    @classmethod
    def get_call_display(cls, event: Any) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, TeamSpawnArgs):
            return ToolCallDisplay(summary=f"Spawning teammate: {args.name}")
        return ToolCallDisplay(summary="Spawning teammate")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, TeamSpawnResult):
            return ToolResultDisplay(success=True, message=result.message)
        return ToolResultDisplay(success=True, message="Teammate spawned")

    @classmethod
    def get_status_text(cls) -> str:
        return "Spawning teammate"

    def resolve_permission(self, args: TeamSpawnArgs) -> PermissionContext | None:
        if self.config.permission in {ToolPermission.ALWAYS, ToolPermission.NEVER}:
            return PermissionContext(permission=self.config.permission)
        return None

    async def run(
        self, args: TeamSpawnArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TeamSpawnResult, None]:
        if ctx is None:
            raise ToolError("Team spawn tool requires context")
        if ctx.team_spawn_callback is None:
            raise ToolError(
                "Team spawning is not available in this context "
                "(no team spawn callback wired)."
            )
        result = await ctx.team_spawn_callback(
            args.name, args.prompt, args.agent, args.max_turns, args.worker
        )
        yield TeamSpawnResult(
            name=str(result["name"]),
            team_dir=str(result["team_dir"]),
            message=str(result["message"]),
            worker=bool(result.get("worker", args.worker)),
        )
