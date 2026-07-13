from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import partial
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import orjson
from pydantic import BaseModel

# Sentinel-prefixed stderr line carrying real token stats, emitted when
# VIBE_WORKFLOW_EMIT_STATS=1 (set by the workflow isolated-agent executor so it
# can charge real tokens instead of an estimate). Kept off stdout so normal
# programmatic output is unaffected.
_STATS_SENTINEL = "__VIBE_WORKFLOW_STATS__"

from vibe import __version__
from vibe.core.agent_loop import AgentLoop, AgentLoopParams, TeleportError
from vibe.core.agents.models import (
    AgentProfile,
    BuiltinAgentName,
    profile_requires_isolation,
)
from vibe.core.config import SandboxConfig, VibeConfig
from vibe.core.hooks.models import HookConfigResult, HookSessionContext
from vibe.core.logger import logger
from vibe.core.loop import LoopManager
from vibe.core.lsp._lifecycle import setup_lsp_for_config, teardown_lsp_async
from vibe.core.output_formatters import OutputFormatter, create_formatter
from vibe.core.schedule_driver import ScheduleDriver
from vibe.core.tasking._policy import (
    BoundTaskContract,
    TaskContractAuthority,
    TaskContractError,
)
from vibe.core.tasking._runtime_context import bind_process_runtime_context
from vibe.core.teams import TeamManager, TeamSafetyMode
from vibe.core.telemetry.build_metadata import build_launch_context
from vibe.core.telemetry.types import ClientMetadata
from vibe.core.teleport.types import (
    TeleportPushRequiredEvent,
    TeleportPushResponseEvent,
)
from vibe.core.tools._task_manifest import (
    TaskManifestError,
    validate_task_manifest_for_agent,
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
    ToolResultEvent,
)
from vibe.core.usage._session import SessionSpendAdapter
from vibe.core.utils import ConversationLimitException
from vibe.core.verification_state import VerificationState
from vibe.core.worktree.manager import (
    WorktreeHandle,
    worktree_enabled,
    worktree_manager,
)

__all__ = ["ProgrammaticOptions", "TeleportError", "run_programmatic"]

_DEFAULT_CLIENT_METADATA = ClientMetadata(name="vibe_programmatic", version=__version__)

if TYPE_CHECKING:
    from vibe.core.llm.types import BackendLike


async def _drive_scheduled_loops(
    agent_loop: AgentLoop,
    scheduler: LoopManager,
    formatter: OutputFormatter,
    keep_alive_seconds: int,
    *,
    fail_on_skipped_tool: bool = False,
) -> None:
    """Fire due scheduled loops as further turns until they drain (one-shots)
    or the deadline passes (so recurring loops don't run forever in -p).
    """

    async def _fire(due: ScheduledLoop) -> None:
        logger.info("Firing scheduled loop %s: %s", due.id, due.prompt)
        async with aclosing(agent_loop.act(due.prompt)) as events:
            async for event in events:
                formatter.on_event(event)
                _raise_for_skipped_verifier_tool(event, enabled=fail_on_skipped_tool)

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
    """Auto-yes ASK-gated tools for a write-capable isolated subprocess.

    The spawn was pre-judged (task._judge_isolated_spawn / workflow runtime) and
    write/edit/read are confined to the worktree (enforce_isolated_confine), so
    the host's unreachable per-tool gate can be bypassed. Without this the
    isolated worker/editor/grunt SKIPs every write/edit/bash and silently produces no
    work. Read-only profiles use a rejecting callback so ASK cannot bypass their
    jailed allowlist — see programmatic.run_programmatic for the env handshake.
    """
    return ApprovalResponse.YES, None, None


async def _isolated_reject_approval(
    tool_name: str,
    args: BaseModel,
    tool_call_id: str,
    required_permissions: list[RequiredPermission] | None,
    judge_deferral: str | None,
) -> tuple[ApprovalResponse, str | None, dict[str, Any] | None]:
    return ApprovalResponse.NO, None, None


def _wire_isolated_approval(agent_loop: AgentLoop) -> None:
    """Wire an isolation-aware callback when running as a child process."""
    if os.environ.get("VIBE_ISOLATED_AUTO_APPROVE") != "1":
        return
    callback = (
        _isolated_reject_approval
        if not profile_requires_isolation(agent_loop.agent_profile)
        else _isolated_auto_approve
    )
    agent_loop.set_approval_callback(callback)


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
    is_subagent: bool = False
    keep_alive_seconds: int | None = None
    no_worktree: bool = False


def _new_programmatic_loop(
    config: VibeConfig,
    opts: ProgrammaticOptions,
    formatter: OutputFormatter,
    *,
    backend: BackendLike | None = None,
    spend_adapter: SessionSpendAdapter | None = None,
    task_contract: BoundTaskContract | None = None,
) -> AgentLoop:
    return AgentLoop(
        config,
        backend=backend,
        params=AgentLoopParams(
            agent_name=opts.agent_name,
            message_observer=formatter.on_message_added,
            max_turns=opts.max_turns,
            max_price=opts.max_price,
            max_session_tokens=opts.max_session_tokens,
            enable_streaming=False,
            headless=opts.headless,
            is_subagent=(
                opts.is_subagent
                or os.environ.get("VIBE_ISOLATED_AUTO_APPROVE") == "1"
                or bool(os.environ.get("VIBE_TEAMMATE_NAME"))
            ),
            allow_subagent_profile=opts.allow_subagent,
            launch_context=build_launch_context(
                agent_entrypoint="programmatic",
                agent_version=__version__,
                client_name=opts.client_metadata.name,
                client_version=opts.client_metadata.version,
            ),
            hook_config_result=opts.hook_config_result,
            spend_adapter=spend_adapter,
            task_contract=task_contract,
        ),
    )


async def _run_team_worker_session(
    agent_loop: AgentLoop, bootstrap_prompt: str, opts: ProgrammaticOptions
) -> None:
    """Claim-loop path for VIBE_TEAM_WORKER=1 (long-lived queue worker).

    Each claimed task is one ``agent_loop.act`` turn. Bootstrap prompt is only
    logged — the harness injects per-task prompts from TaskStore.
    """
    logger.info("TEAM_WORKER bootstrap: %s", bootstrap_prompt[:500])

    from vibe.core.teams._same_worker_repair import run_same_worker_repair
    from vibe.core.teams._structured_attempt import evaluate_structured_attempt
    from vibe.core.teams.models import Task
    from vibe.core.teams.worker_loop import (
        WorkerTaskAttempt,
        run_team_worker_loop,
        worker_task_prompt,
    )

    async def run_task(task: Task) -> WorkerTaskAttempt:
        contract = None
        if task.brief is not None:
            contract = BoundTaskContract.bind(
                task.brief,
                authority=TaskContractAuthority.LEAD,
                workspace_root=Path.cwd(),
                verification_state=VerificationState.from_recipe(
                    agent_loop.base_config.trusted_verification_recipe
                ),
            )
            _validate_team_task_contract(contract, agent_loop.agent_profile)
        if contract is None:
            task_spend = agent_loop.spend_adapter.child_agent(
                purpose=agent_loop.spend_adapter.default_purpose
            )
        else:
            task_spend = agent_loop.spend_adapter.child_task(
                task_id=task.id,
                purpose=agent_loop.spend_adapter.default_purpose,
                limits=contract.spend_limits(),
                task_brief_hash=contract.brief_hash,
            )
        task_formatter = create_formatter(opts.output_format)
        task_loop = _new_programmatic_loop(
            agent_loop.base_config.model_copy(deep=True),
            opts,
            task_formatter,
            spend_adapter=task_spend,
            task_contract=contract,
        )
        task_loop.parent_session_id = agent_loop.session_id
        _wire_isolated_approval(task_loop)
        task_background = BackgroundRegistry()
        task_loop.background_registry = task_background
        try:
            await task_loop.initialize_experiments()
            task_loop.emit_new_session_telemetry()
            async with aclosing(task_loop.act(worker_task_prompt(task))) as events:
                async for event in events:
                    task_formatter.on_event(event)
                    _raise_for_skipped_verifier_tool(
                        event,
                        enabled=task_loop.agent_profile.name
                        == BuiltinAgentName.VERIFIER,
                    )
                    if (
                        isinstance(event, AssistantEvent)
                        and event.stopped_by_middleware
                    ):
                        raise ConversationLimitException(event.content)
            summary = task_formatter.finalize()
            outcome = None
            if contract is not None and task.brief is not None:
                outcome = await evaluate_structured_attempt(
                    task.brief,
                    contract,
                    summary,
                    repair=partial(run_same_worker_repair, task_loop),
                )
            return WorkerTaskAttempt(summary, contract, outcome)
        finally:
            task_loop.emit_session_closed_telemetry()
            await task_background.shutdown()
            await task_loop.aclose()
            await task_loop.telemetry_client.aclose()

    await run_team_worker_loop(run_task)


def _validate_team_task_contract(
    contract: BoundTaskContract, agent_profile: AgentProfile
) -> None:
    if agent_profile.name == BuiltinAgentName.VERIFIER:
        raise TaskContractError(
            "structured verifier assignments are not supported by the team worker "
            "TASK_OUTCOME protocol; use the task or workflow verifier path"
        )
    try:
        validate_task_manifest_for_agent(
            contract.manifest,
            agent=agent_profile.name,
            read_only=not profile_requires_isolation(agent_profile),
        )
    except TaskManifestError as exc:
        raise TaskContractError(str(exc)) from exc


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

        await _drive_programmatic_turn(agent_loop, formatter, opts, prompt, scheduler)

        if os.environ.get("VIBE_WORKFLOW_EMIT_STATS") == "1":
            stats_line = _STATS_SENTINEL + orjson.dumps({
                "prompt_tokens": agent_loop.stats.session_prompt_tokens,
                "completion_tokens": agent_loop.stats.session_completion_tokens,
                "cached_tokens": agent_loop.stats.session_cached_tokens,
                "cache_write_tokens": agent_loop.stats.session_cache_write_tokens,
                "reasoning_tokens": agent_loop.stats.session_reasoning_tokens,
                "cost_usd": agent_loop.stats.session_cost,
                "cost_initialized": agent_loop.stats.accumulated_cost_initialized,
                "cost_estimated": agent_loop.stats.cost_is_estimated,
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


async def _drive_programmatic_turn(
    agent_loop: AgentLoop,
    formatter: OutputFormatter,
    opts: ProgrammaticOptions,
    prompt: str,
    scheduler: LoopManager,
) -> None:
    """Run teleport, team-worker claim loop, or a single act turn + keep-alive."""
    if opts.teleport and agent_loop.base_config.vibe_code_enabled:
        gen = agent_loop.teleport_to_vibe_code(prompt or None)
        async for event in gen:
            formatter.on_event(event)
            if isinstance(event, TeleportPushRequiredEvent):
                next_event = await gen.asend(TeleportPushResponseEvent(approved=True))
                formatter.on_event(next_event)
    else:
        from vibe.core.teams.worker_loop import is_team_worker

        if is_team_worker():
            await _run_team_worker_session(agent_loop, prompt, opts)
        else:
            await _act_once(
                agent_loop,
                formatter,
                prompt,
                fail_on_skipped_tool=opts.agent_name == BuiltinAgentName.VERIFIER,
            )

    if opts.keep_alive_seconds and scheduler.loops:
        await _drive_scheduled_loops(
            agent_loop,
            scheduler,
            formatter,
            opts.keep_alive_seconds,
            fail_on_skipped_tool=opts.agent_name == BuiltinAgentName.VERIFIER,
        )


def _raise_for_skipped_verifier_tool(event: object, *, enabled: bool) -> None:
    if not enabled or not isinstance(event, ToolResultEvent) or not event.skipped:
        return
    reason = event.skip_reason or event.error or "tool call was skipped"
    raise RuntimeError(f"Verifier tool call {event.tool_name!r} did not run: {reason}")


async def _act_once(
    agent_loop: AgentLoop,
    formatter: OutputFormatter,
    prompt: str,
    *,
    fail_on_skipped_tool: bool = False,
) -> None:
    async with aclosing(agent_loop.act(prompt)) as events:
        async for event in events:
            formatter.on_event(event)
            _raise_for_skipped_verifier_tool(event, enabled=fail_on_skipped_tool)
            if isinstance(event, AssistantEvent) and event.stopped_by_middleware:
                raise ConversationLimitException(event.content)


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

    task_contract, spend_adapter = bind_process_runtime_context(config, Path.cwd())

    agent_loop = _new_programmatic_loop(
        config,
        opts,
        formatter,
        spend_adapter=spend_adapter,
        task_contract=task_contract,
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
    # the `background` tool still works. Async subagent requests fall back to
    # blocking because this surface has no completion wake callback.
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
                spend_adapter=agent_loop.spend_adapter,
                terminal_callback=agent_loop.observe_team_completion,
            )
            background_registry.attach_team_manager(lambda: team_manager)
        return team_manager

    async def spawn_team(
        name: str,
        prompt: str,
        agent: str,
        max_turns: int,
        worker: bool = False,
        safety_mode: TeamSafetyMode = TeamSafetyMode.SHARED,
    ) -> dict[str, Any]:
        manager = ensure_team_manager()
        await manager.spawn_teammate(
            name,
            prompt,
            agent=agent,
            max_turns=max_turns,
            worker=worker,
            safety_mode=safety_mode,
        )
        launch_id = manager.launch_id_for(name)
        if launch_id is None:
            raise RuntimeError(f"Missing launch id for teammate '{name}'")
        kind = "worker" if worker else "teammate"
        return {
            "launch_id": launch_id,
            "name": name,
            "team_dir": str(manager.team_dir),
            "message": f"Spawned {kind} `{name}`.",
            "worker": worker,
            "safety_mode": safety_mode.value,
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
