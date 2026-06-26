from __future__ import annotations

import asyncio
from contextlib import aclosing
import os
from pathlib import Path
import sys

import orjson
from pydantic import BaseModel

# Sentinel-prefixed stderr line carrying real token stats, emitted when
# VIBE_WORKFLOW_EMIT_STATS=1 (set by the workflow isolated-agent executor so it
# can charge real tokens instead of an estimate). Kept off stdout so normal
# programmatic output is unaffected.
_STATS_SENTINEL = "__VIBE_WORKFLOW_STATS__"

from vibe import __version__
from vibe.core.agent_loop import AgentLoop, TeleportError
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import VibeConfig
from vibe.core.hooks.models import HookConfigResult
from vibe.core.logger import logger
from vibe.core.loop import LoopManager
from vibe.core.lsp._lifecycle import setup_lsp_for_config, teardown_lsp_async
from vibe.core.output_formatters import OutputFormatter, create_formatter
from vibe.core.schedule_driver import ScheduleDriver
from vibe.core.telemetry.build_metadata import build_entrypoint_metadata
from vibe.core.telemetry.types import ClientMetadata
from vibe.core.teleport.types import (
    TeleportPushRequiredEvent,
    TeleportPushResponseEvent,
)
from vibe.core.tools.permissions import RequiredPermission
from vibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    LLMMessage,
    OutputFormat,
    Role,
    ScheduledLoop,
)
from vibe.core.utils import ConversationLimitException
from vibe.core.worktree.manager import worktree_enabled, worktree_manager

__all__ = ["TeleportError", "run_programmatic"]

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
) -> tuple[ApprovalResponse, str | None]:
    """Auto-yes every ASK-gated tool in an isolated subprocess.

    The spawn was pre-judged (task._judge_isolated_spawn / workflow runtime) and
    write/edit/read are confined to the worktree (enforce_isolated_confine), so
    the host's unreachable per-tool gate can be bypassed. Without this the
    isolated worker/editor SKIPs every write/edit/bash and silently produces no
    work — see programmatic.run_programmatic for the env-flag handshake.
    """
    return ApprovalResponse.YES, None


def _wire_isolated_approval(agent_loop: AgentLoop) -> None:
    """Wire the auto-yes callback when running as an isolated subprocess."""
    if os.environ.get("VIBE_ISOLATED_AUTO_APPROVE") == "1":
        agent_loop.set_approval_callback(_isolated_auto_approve)


def run_programmatic(  # noqa: PLR0913, PLR0917
    config: VibeConfig,
    prompt: str,
    max_turns: int | None = None,
    max_price: float | None = None,
    max_session_tokens: int | None = None,
    output_format: OutputFormat = OutputFormat.TEXT,
    previous_messages: list[LLMMessage] | None = None,
    agent_name: str = BuiltinAgentName.DEFAULT,
    client_metadata: ClientMetadata = _DEFAULT_CLIENT_METADATA,
    teleport: bool = False,
    headless: bool = False,
    hook_config_result: HookConfigResult | None = None,
    allow_subagent: bool = False,
    keep_alive_seconds: int | None = None,
) -> str | None:
    formatter = create_formatter(output_format)

    # Worktree isolation: enter before AgentLoop so all Path.cwd() consumers
    # see the worktree. Auto-ON for programmatic (mode=auto-by-entrypoint).
    worktree_handle = None
    if worktree_enabled(config, programmatic=True):
        worktree_handle = worktree_manager.enter("programmatic", config.worktree)
        if worktree_handle is not None and not config.displayed_workdir:
            config.displayed_workdir = str(worktree_handle.original_repo_root)

    agent_loop = AgentLoop(
        config,
        agent_name=agent_name,
        message_observer=formatter.on_message_added,
        max_turns=max_turns,
        max_price=max_price,
        max_session_tokens=max_session_tokens,
        enable_streaming=False,
        headless=headless,
        is_subagent=allow_subagent,
        entrypoint_metadata=build_entrypoint_metadata(
            agent_entrypoint="programmatic",
            agent_version=__version__,
            client_name=client_metadata.name,
            client_version=client_metadata.version,
        ),
        hook_config_result=hook_config_result,
    )
    _wire_isolated_approval(agent_loop)
    # Wire the scheduler so the `schedule` tool works headless (create/list/
    # cancel persist to session metadata). They only FIRE in this run when
    # keep_alive_seconds is set (see the drive phase below); otherwise they
    # persist and fire on a later interactive/ACP resume of the session.
    scheduler = LoopManager(agent_loop.session_logger)
    agent_loop.set_scheduler(scheduler)
    logger.info("USER: %s", prompt)

    async def _async_run() -> str | None:
        setup_lsp_for_config(config, lambda: config, Path.cwd(), warmup=True)
        try:
            if previous_messages:
                non_system_messages = [
                    msg for msg in previous_messages if not (msg.role == Role.system)
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

            if teleport and config.vibe_code_enabled:
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
            if keep_alive_seconds and scheduler.loops:
                await _drive_scheduled_loops(
                    agent_loop, scheduler, formatter, keep_alive_seconds
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
            await _teardown_lsp_and_loop(agent_loop)

    try:
        return asyncio.run(_async_run())
    finally:
        if worktree_handle is not None:
            worktree_manager.exit(worktree_handle)
