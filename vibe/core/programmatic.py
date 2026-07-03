from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any

import orjson
from pydantic import BaseModel

# Sentinel-prefixed stderr line carrying real token stats, emitted when
# VIBE_WORKFLOW_EMIT_STATS=1 (set by the workflow isolated-agent executor so it
# can charge real tokens instead of an estimate). Kept off stdout so normal
# programmatic output is unaffected.
_STATS_SENTINEL = "__VIBE_WORKFLOW_STATS__"

from vibe import __version__
from vibe.core.agent_loop import AgentLoop, AgentLoopParams, TeleportError
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import SandboxConfig, VibeConfig
from vibe.core.hooks.models import HookConfigResult, HookSessionContext
from vibe.core.logger import logger
from vibe.core.loop import LoopManager
from vibe.core.lsp._lifecycle import setup_lsp_for_config, teardown_lsp_async
from vibe.core.output_formatters import OutputFormatter, create_formatter
from vibe.core.schedule_driver import ScheduleDriver
from vibe.core.teams import TeamManager
from vibe.core.telemetry.build_metadata import build_launch_context
from vibe.core.telemetry.types import ClientMetadata
from vibe.core.teleport.types import (
    TeleportPushRequiredEvent,
    TeleportPushResponseEvent,
)
from vibe.core.tools.background import BackgroundRegistry
from vibe.core.tools.permissions import RequiredPermission
from vibe.core.tools.utils import isolated_worktree_root
from vibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    LLMMessage,
    OutputFormat,
    Role,
    ScheduledLoop,
)
from vibe.core.utils import ConversationLimitException
from vibe.core.worktree.manager import (
    WorktreeHandle,
    worktree_enabled,
    worktree_manager,
)

__all__ = ["ProgrammaticOptions", "TeleportError", "run_programmatic"]

_DEFAULT_CLIENT_METADATA = ClientMetadata(name="vibe_programmatic", version=__version__)


async def _drive_scheduled_loops(
    agent_loop: AgentLoop,
    scheduler: LoopManager,
    formatter: OutputFormatter,
    keep_alive_seconds: int,
) -> None:
    """Fire due scheduled loops as further turns until they drain (one-shots)
    or the deadline passes (so recurring loops don't run forever in -p).
    """

    async def _fire(due: ScheduledLoop) -> None:
        logger.info("Firing scheduled loop %s: %s", due.id, due.prompt)
        async with aclosing(agent_loop.act(due.prompt)) as events:
            async for event in events:
                formatter.on_event(event)

    driver = ScheduleDriver(scheduler, can_fire=lambda: True, fire=_fire)
    deadline = asyncio.get_running_loop().time() + keep_alive_seconds
    await driver.run_until_idle(deadline=deadline)


async def _teardown_lsp_and_loop(agent_loop: AgentLoop) -> None:
    await teardown_lsp_async()
    await agent_loop.aclose()
    await agent_loop.telemetry_client.aclose()


async def _isolated_auto_approve(
    tool_name: str,
    args: BaseModel,
    tool_call_id: str,
    required_permissions: list[RequiredPermission] | None,
    judge_deferral: str | None,
) -> tuple[ApprovalResponse, str | None, dict[str, Any] | None]:
    """Auto-yes every ASK-gated tool in an isolated subprocess.

    The spawn was pre-judged (task._judge_isolated_spawn / workflow runtime) and
    write/edit/read are confined to the worktree (enforce_isolated_confine), so
    the host's unreachable per-tool gate can be bypassed. Without this the
    isolated worker/editor/grunt SKIPs every write/edit/bash and silently produces no
    work — see programmatic.run_programmatic for the env-flag handshake.
    """
    return ApprovalResponse.YES, None, None


def _wire_isolated_approval(agent_loop: AgentLoop) -> None:
    """Wire the auto-yes callback when running as an isolated subprocess."""
    if os.environ.get("VIBE_ISOLATED_AUTO_APPROVE") == "1":
        agent_loop.set_approval_callback(_isolated_auto_approve)


@dataclass(frozen=True, slots=True)
class ProgrammaticOptions:
    max_turns: int | None = None
    max_price: float | None = None
    max_session_tokens: int | None = None
    output_format: OutputFormat = OutputFormat.TEXT
    previous_messages: list[LLMMessage] | None = None
    agent_name: str = BuiltinAgentName.DEFAULT
    client_metadata: ClientMetadata = field(
        default_factory=lambda: _DEFAULT_CLIENT_METADATA
    )
    teleport: bool = False
    headless: bool = False
    hook_config_result: HookConfigResult | None = None
    allow_subagent: bool = False
    keep_alive_seconds: int | None = None
    no_worktree: bool = False


async def _run_session(
    config: VibeConfig,
    agent_loop: AgentLoop,
    scheduler: LoopManager,
    formatter: OutputFormatter,
    opts: ProgrammaticOptions,
    prompt: str,
    background_registry: BackgroundRegistry,
    team_cleanup: Callable[[], Awaitable[None]] | None = None,
) -> str | None:
    setup_lsp_for_config(config, lambda: config, Path.cwd(), warmup=True)
    try:
        if opts.previous_messages:
            non_system_messages = [
                msg for msg in opts.previous_messages if not (msg.role == Role.SYSTEM)
            ]
            agent_loop.messages.extend(non_system_messages)
            logger.info(
                "Loaded %d messages from previous session", len(non_system_messages)
            )
            metadata = agent_loop.session_logger.session_metadata
            if metadata is not None:
                scheduler.restore(list(metadata.loops))
        else:
            await agent_loop.initialize_experiments()
            agent_loop.emit_new_session_telemetry()

        if opts.teleport and config.vibe_code_enabled:
            gen = agent_loop.teleport_to_vibe_code(prompt or None)
            async for event in gen:
                formatter.on_event(event)
                if isinstance(event, TeleportPushRequiredEvent):
                    next_event = await gen.asend(
                        TeleportPushResponseEvent(approved=True)
                    )
                    formatter.on_event(next_event)
        else:
            async with aclosing(agent_loop.act(prompt)) as events:
                async for event in events:
                    formatter.on_event(event)
                    if (
                        isinstance(event, AssistantEvent)
                        and event.stopped_by_middleware
                    ):
                        raise ConversationLimitException(event.content)

        # Keep-alive drive: fire scheduled loops as further turns until they
        # drain (one-shots) or the deadline passes (caps recurring loops).
        if opts.keep_alive_seconds and scheduler.loops:
            await _drive_scheduled_loops(
                agent_loop, scheduler, formatter, opts.keep_alive_seconds
            )

        if os.environ.get("VIBE_WORKFLOW_EMIT_STATS") == "1":
            stats_line = _STATS_SENTINEL + orjson.dumps({
                "prompt_tokens": agent_loop.stats.session_prompt_tokens,
                "completion_tokens": agent_loop.stats.session_completion_tokens,
            }).decode("utf-8")
            sys.stderr.write("\n" + stats_line + "\n")
            sys.stderr.flush()
        return formatter.finalize()
    finally:
        agent_loop.emit_session_closed_telemetry()
        if team_cleanup is not None:
            await team_cleanup()
        # Reap backgrounded processes so a `vibe -p` that started a server
        # does not orphan it to init on exit. Aggregated categories own
        # their own shutdown; this only reaps registry-owned processes.
        await background_registry.shutdown()
        await _teardown_lsp_and_loop(agent_loop)


def _emit_headless_sandbox_nudge(sandbox: SandboxConfig | None) -> None:
    # Headless has no TUI toast, so surface the unshare-only nudge on stderr.
    if sandbox is None:
        return
    from vibe.core.tools.sandbox import unshare_confinement_nudge

    nudge = unshare_confinement_nudge(
        sandbox_enabled=sandbox.enabled, backend_override=sandbox.backend
    )
    if nudge:
        sys.stderr.write(nudge + "\n")
        sys.stderr.flush()


def _exit_if_orphaned_isolated_child() -> bool:
    # F7: child lease — if this isolated spawn's parent has died, exit cleanly
    # rather than running orphaned in a worktree whose owner no longer exists.
    parent_pid_str = os.environ.get("VIBE_ISO_PARENT_PID")
    if parent_pid_str is None:
        return False
    try:
        parent_pid = int(parent_pid_str)
        if parent_pid > 0:
            os.kill(parent_pid, 0)
    except (ValueError, ProcessLookupError):
        logger.warning(
            "Orphaned isolated spawn: parent pid %s is gone; exiting.", parent_pid_str
        )
        return True
    except OSError:
        pass  # exists but not signalable — assume alive
    return False


def run_programmatic(
    config: VibeConfig, prompt: str, *, options: ProgrammaticOptions | None = None
) -> str | None:
    opts = options or ProgrammaticOptions()
    formatter = create_formatter(opts.output_format)

    if _exit_if_orphaned_isolated_child():
        return None

    # Worktree isolation (on by default for programmatic, worktree.mode="on"):
    # enter before AgentLoop so Path.cwd() consumers see the worktree.
    worktree_handle: WorktreeHandle | None = None
    # An isolated spawn already runs inside a parent-created ephemeral worktree;
    # entering another moves cwd outside enforce_isolated_confine's root (every
    # file tool errors).
    if (
        not opts.no_worktree
        and isolated_worktree_root() is None
        and worktree_enabled(config, programmatic=True)
    ):
        worktree_handle = worktree_manager.enter("programmatic", config.worktree)
        if worktree_handle is not None and not config.displayed_workdir:
            config.displayed_workdir = str(worktree_handle.original_repo_root)

    agent_loop = AgentLoop(
        config,
        params=AgentLoopParams(
            agent_name=opts.agent_name,
            message_observer=formatter.on_message_added,
            max_turns=opts.max_turns,
            max_price=opts.max_price,
            max_session_tokens=opts.max_session_tokens,
            enable_streaming=False,
            headless=opts.headless,
            is_subagent=opts.allow_subagent,
            launch_context=build_launch_context(
                agent_entrypoint="programmatic",
                agent_version=__version__,
                client_name=opts.client_metadata.name,
                client_version=opts.client_metadata.version,
            ),
            hook_config_result=opts.hook_config_result,
        ),
    )
    _wire_isolated_approval(agent_loop)
    # Wire the scheduler so the `schedule` tool works headless (create/list/
    # cancel persist to session metadata). They only FIRE in this run when
    # keep_alive_seconds is set (see the drive phase below); otherwise they
    # persist and fire on a later interactive/ACP resume of the session.
    scheduler = LoopManager(agent_loop.session_logger)
    agent_loop.set_scheduler(scheduler)
    # Wire a background registry so the bash tool's background=True works
    # headless. Without this, ctx.background_registry is None and the bash
    # tool refuses to spawn (ToolError). Owns processes outright; aggregated
    # categories (workflows/teams/loops) stay empty — no Tasks pane here, but
    # the `background` tool and async subagent completions still work.
    background_registry = BackgroundRegistry()
    agent_loop.background_registry = background_registry
    team_manager: TeamManager | None = None

    def hook_context() -> HookSessionContext:
        transcript = ""
        if agent_loop.session_logger.enabled and agent_loop.session_logger.session_dir:
            transcript = str(agent_loop.session_logger.messages_filepath.resolve())
        return HookSessionContext(
            session_id=agent_loop.session_id,
            transcript_path=transcript,
            cwd=str(Path.cwd().resolve()),
            parent_session_id=agent_loop.parent_session_id,
        )

    def ensure_team_manager() -> TeamManager:
        nonlocal team_manager
        if team_manager is None:
            team_manager = TeamManager(
                agent_loop.session_id,
                hooks_manager=agent_loop.hooks_manager,
                hook_context=hook_context,
            )
            background_registry.attach_team_manager(lambda: team_manager)
        return team_manager

    async def spawn_team(
        name: str, prompt: str, agent: str, max_turns: int
    ) -> dict[str, Any]:
        manager = ensure_team_manager()
        await manager.spawn_teammate(name, prompt, agent=agent, max_turns=max_turns)
        return {
            "name": name,
            "team_dir": str(manager.team_dir),
            "message": f"Spawned teammate `{name}`.",
        }

    async def cleanup_team() -> None:
        if team_manager is None:
            return
        await team_manager.stop_all()
        await asyncio.to_thread(team_manager.cleanup)

    agent_loop.team_dir_callback = lambda: (
        str(team_manager.team_dir) if team_manager is not None else None
    )
    agent_loop.team_spawn_callback = spawn_team
    if opts.headless:
        bash_cfg = agent_loop.tool_manager.get_tool_config("bash")
        _emit_headless_sandbox_nudge(getattr(bash_cfg, "sandbox", None))
    logger.info("USER: %s", prompt)

    try:
        return asyncio.run(
            _run_session(
                config,
                agent_loop,
                scheduler,
                formatter,
                opts,
                prompt,
                background_registry,
                cleanup_team,
            )
        )
    finally:
        if worktree_handle is not None:
            worktree_manager.exit(worktree_handle)
