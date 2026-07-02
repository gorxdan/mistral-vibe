from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
import contextlib
import copy
import dataclasses
import functools
import hashlib
import os
from pathlib import Path
import shutil
import threading
from threading import Thread
import time
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from opentelemetry import trace

from vibe.core import loop_tracer, profiler, stream_tracer
from vibe.core.agent_loop._errors import (
    AgentLoopError,
    ImagesNotSupportedError,
    TeleportError,
)
from vibe.core.agent_loop._init_guard import requires_init
from vibe.core.agent_loop._limits import (
    AGGREGATE_TOOL_RESULT_CHARS,
    MAX_CONCURRENT_SUBAGENTS,
    MAX_TOOL_RESULT_CHARS,
    TOOL_RESULT_PREVIEW_CHARS,
    tool_result_hard_cap,
)
from vibe.core.agent_loop.session_mixin import AgentLoopSessionMixin
from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import AgentProfile, BuiltinAgentName
from vibe.core.baseline_scaling import BaselineTier, baseline_tier_for
from vibe.core.cache_store import InMemoryVibeCodeCacheStore, VibeCodeCacheStore
from vibe.core.config import ModelConfig, ProviderConfig, VibeConfig, resolve_api_key
from vibe.core.experiments import ExperimentManager
from vibe.core.experiments.client import RemoteEvalClient
from vibe.core.experiments.session import (
    hydrate_experiments_from_session as session_hydrate_experiments_from_session,
    initialize_experiments as session_initialize_experiments,
)
from vibe.core.hooks.manager import HooksManager
from vibe.core.hooks.models import HookConfigResult, HookEvent
from vibe.core.llm.backend.factory import create_backend
from vibe.core.llm.format import APIToolFormatHandler
from vibe.core.llm.models import FailedToolCall, ResolvedMessage, ResolvedToolCall
from vibe.core.llm.types import BackendLike
from vibe.core.logger import logger
from vibe.core.lsp._integration import drain_diagnostics_into
from vibe.core.middleware import (
    CHAT_AGENT_EXIT,
    CHAT_AGENT_REMINDER,
    PLAN_AGENT_EXIT,
    AutoCompactMiddleware,
    ContextWarningMiddleware,
    ConversationContext,
    LoopDetectionMiddleware,
    MicrocompactMiddleware,
    MiddlewareAction,
    MiddlewarePipeline,
    MiddlewareResult,
    PriceLimitMiddleware,
    ReadOnlyAgentMiddleware,
    SnipMiddleware,
    TokenLimitMiddleware,
    ToolResultBudgetMiddleware,
    TurnLimitMiddleware,
    make_plan_agent_reminder,
)
from vibe.core.plan_session import PlanSession
from vibe.core.resource_monitor import ResourceMonitor
from vibe.core.rewind import RewindManager
from vibe.core.scratchpad import init_scratchpad
from vibe.core.session.session_id import generate_session_id
from vibe.core.session.session_logger import SessionLogger
from vibe.core.session.session_migration import migrate_sessions_entrypoint
from vibe.core.skills.manager import SkillManager
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.telemetry.build_metadata import build_request_metadata
from vibe.core.telemetry.send import TelemetryClient
from vibe.core.telemetry.types import (
    EntrypointMetadata,
    TelemetryCallType,
    TelemetryRequestMetadata,
    TerminalEmulator,
)
from vibe.core.teleport.errors import ServiceTeleportError
from vibe.core.teleport.telemetry import TeleportTelemetryTracker
from vibe.core.teleport.types import TeleportCompleteEvent
from vibe.core.tools.base import (
    BaseTool,
    CancellableToolResult,
    InvokeContext,
    ToolError,
    ToolPermission,
    ToolPermissionError,
)
from vibe.core.tools.manager import ToolManager
from vibe.core.tools.mcp_sampling import MCPSamplingHandler
from vibe.core.tools.permissions import (
    ApprovedRule,
    PermissionStore,
    RequiredPermission,
)
from vibe.core.tools.tool_result_store import ToolResultStore
from vibe.core.tracing import (
    agent_span,
    context_shaping_span,
    set_agent_usage,
    set_context_shaping_result,
    set_tool_error,
    set_tool_exec_duration,
    set_tool_result,
    set_tool_user_wait,
    tool_span,
)
from vibe.core.trusted_folders import has_agents_md_file
from vibe.core.types import (
    AgentProfileChangedEvent,
    AgentStats,
    ApprovalCallback,
    AssistantEvent,
    BackgroundTaskCompletedEvent,
    BaseEvent,
    CompactEndEvent,
    CompactionOrigin,
    CompactStartEvent,
    ContentFilterError,
    ContextTooLongError,
    ImageAttachment,
    InjectedMessageKind,
    LLMMessage,
    MessageList,
    PlanReviewEndedEvent,
    PlanReviewRequestedEvent,
    RateLimitCallback,
    RateLimitError,
    ReasoningEvent,
    ResponseTooLongError,
    Role,
    ServerError,
    SessionTitleUpdatedEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    TransportError,
    UserInputCallback,
    UserMessageEvent,
)
from vibe.core.usage import RateLimitStore, get_usage_recorder, rate_limit_from_headers
from vibe.core.utils import (
    TOOL_ERROR_TAG,
    VIBE_STOP_EVENT_TAG,
    VIBE_WARNING_TAG,
    CancellationReason,
    get_server_url_from_api_base,
    get_user_agent,
    get_user_cancellation_message,
    is_user_cancellation_event,
)


def _git_executable_present() -> bool:
    # GitPython is imported lazily (perf), so teleport availability can no longer
    # piggyback on an eager `import git` failing. Detect the executable cheaply,
    # honoring GIT_PYTHON_GIT_EXECUTABLE the same way GitPython would, without
    # paying the GitPython import cost on the startup path.
    return shutil.which(os.environ.get("GIT_PYTHON_GIT_EXECUTABLE", "git")) is not None


_TELEPORT_AVAILABLE = _git_executable_present()


@functools.cache
def _teleport_service_cls() -> type[TeleportService] | None:
    try:
        from vibe.core.teleport.teleport import TeleportService
    except ImportError:
        return None
    return TeleportService


if TYPE_CHECKING:
    from vibe.core.loop import Scheduler
    from vibe.core.memory.store import MemoryStore
    from vibe.core.teleport.teleport import TeleportService
    from vibe.core.teleport.types import TeleportPushResponseEvent, TeleportYieldEvent
    from vibe.core.tools.background import BackgroundRegistry
    from vibe.core.tools.connectors import ConnectorRegistry
    from vibe.core.tools.mcp import MCPRegistry
    from vibe.core.tools.safety_judge import JudgeVerdict
from vibe.core.agent_loop._models import ToolDecision, ToolExecutionResponse


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentLoopParams:
    agent_name: str = BuiltinAgentName.DEFAULT
    message_observer: Callable[[LLMMessage], None] | None = None
    max_turns: int | None = None
    max_price: float | None = None
    max_session_tokens: int | None = None
    enable_streaming: bool = False
    entrypoint_metadata: EntrypointMetadata | None = None
    terminal_emulator: TerminalEmulator | None = None
    is_subagent: bool = False
    defer_heavy_init: bool = False
    headless: bool = False
    hook_config_result: HookConfigResult | None = None
    permission_store: PermissionStore | None = None
    mcp_registry: MCPRegistry | None = None
    cache_store: VibeCodeCacheStore | None = None


class AgentLoop(AgentLoopSessionMixin):
    def __init__(
        self,
        config: VibeConfig,
        *,
        backend: BackendLike | None = None,
        params: AgentLoopParams | None = None,
    ) -> None:
        p = params or AgentLoopParams()
        self._init_base_state(config, p.cache_store, p.headless, p.defer_heavy_init)
        self._init_registries(
            p.permission_store,
            p.mcp_registry,
            p.agent_name,
            p.is_subagent,
            p.defer_heavy_init,
            p.message_observer,
            p.max_turns,
            p.max_price,
            p.max_session_tokens,
        )
        self._init_backend(backend, p.enable_streaming)
        self._init_session_identity(p.is_subagent)
        self._init_messages(p.defer_heavy_init, p.message_observer)
        self._init_session_state(
            p.is_subagent, p.entrypoint_metadata, p.terminal_emulator, config
        )
        self._init_telemetry(config, p.is_subagent)
        self._init_hooks(p.hook_config_result)
        self._init_rewind()

        Thread(
            target=migrate_sessions_entrypoint,
            args=(config.session_logging,),
            daemon=True,
            name="migrate_sessions",
        ).start()

        if p.defer_heavy_init:
            self._start_deferred_init()

    def _init_base_state(
        self,
        config: VibeConfig,
        cache_store: VibeCodeCacheStore | None,
        headless: bool,
        defer_heavy_init: bool,
    ) -> None:
        self.cache_store = cache_store or InMemoryVibeCodeCacheStore()
        self._base_config = config
        self._headless = headless

        self._defer_heavy_init = defer_heavy_init
        self._deferred_init_thread: threading.Thread | None = None
        self._deferred_init_lock = threading.Lock()
        self._init_error: Exception | None = None
        self._init_start_time = time.monotonic()
        self._experiments_task: asyncio.Task[None] | None = None
        self._pending_new_session_telemetry: bool = False
        self._ready_telemetry_pending: bool = defer_heavy_init

    def _init_registries(
        self,
        permission_store: PermissionStore | None,
        mcp_registry: MCPRegistry | None,
        agent_name: str,
        is_subagent: bool,
        defer_heavy_init: bool,
        message_observer: Callable[[LLMMessage], None] | None,
        max_turns: int | None,
        max_price: float | None,
        max_session_tokens: int | None,
    ) -> None:
        self._permission_store = permission_store or PermissionStore()

        self.mcp_registry: MCPRegistry | None = (
            None if defer_heavy_init else mcp_registry or self._create_mcp_registry()
        )
        self.connector_registry: ConnectorRegistry | None = (
            None if defer_heavy_init else self._create_connector_registry()
        )
        self.agent_manager = AgentManager(
            lambda: self._base_config,
            initial_agent=agent_name,
            allow_subagent=is_subagent,
        )
        self.tool_manager = ToolManager(
            lambda: self.config,
            mcp_registry=self.mcp_registry,
            connector_registry=self.connector_registry,
            defer_mcp=defer_heavy_init,
            permission_getter=self._permission_store.get_tool_permission,
        )
        self.skill_manager = SkillManager(lambda: self.config)
        self.message_observer = message_observer
        self._max_turns = max_turns
        self._max_price = max_price
        self._max_session_tokens = max_session_tokens
        self._plan_session = PlanSession()

        self.format_handler = APIToolFormatHandler()

    def _init_backend(
        self, backend: BackendLike | None, enable_streaming: bool
    ) -> None:
        self.backend_factory = lambda: backend or self._select_backend()
        self.backend = self.backend_factory()
        self._sampling_handler = MCPSamplingHandler(
            backend_getter=lambda: self.backend,
            config_getter=lambda: self.config,
            metadata_getter=lambda: self._build_backend_metadata(
                call_type="secondary_call"
            ).model_dump(exclude_none=True),
            extra_headers_getter=self._get_extra_headers,
        )

        self.enable_streaming = enable_streaming
        self.middleware_pipeline = MiddlewarePipeline()
        self.tool_result_store = ToolResultStore(
            session_dir_getter=lambda: self.session_logger.session_dir
        )
        self._setup_middleware()

    def _init_session_identity(self, is_subagent: bool) -> None:
        self.session_id = generate_session_id()
        self.parent_session_id: str | None = None
        # codex (openai-chatgpt) sticky-routing token: captured from the
        # `x-codex-turn-state` response header and replayed on subsequent
        # requests within the same turn so they pin to one backend partition and
        # keep the prompt cache warm. Reset at each user turn (codex forbids
        # replaying it across turns). See _chat_streaming / _get_extra_headers.
        self._codex_turn_state: str | None = None
        self.scratchpad_dir = (
            init_scratchpad(self.session_id) if not is_subagent else None
        )
        self._files_read: dict[str, str] = {}
        self._files_read_reconstructed: bool = False
        self._agents_md_fingerprint: str | None = None

        self._system_prompt_tier = self._current_baseline_tier()

    def _init_messages(
        self,
        defer_heavy_init: bool,
        message_observer: Callable[[LLMMessage], None] | None,
    ) -> None:
        system_prompt = get_universal_system_prompt(
            self.tool_manager,
            self.config,
            self.skill_manager,
            self.agent_manager,
            include_git_status=not defer_heavy_init,
            scratchpad_dir=self.scratchpad_dir,
            headless=self._headless,
            tier=self._system_prompt_tier,
        )
        system_message = LLMMessage(role=Role.SYSTEM, content=system_prompt)
        self.messages = MessageList(initial=[system_message], observer=message_observer)

        self.stats = AgentStats()

    def _init_session_state(
        self,
        is_subagent: bool,
        entrypoint_metadata: EntrypointMetadata | None,
        terminal_emulator: TerminalEmulator | None,
        config: VibeConfig,
    ) -> None:
        self.approval_callback: ApprovalCallback | None = None
        self.scheduler: Scheduler | None = None
        # Reason the safety judge deferred the current tool call to the user, if
        # any; read by the approval UI. Set per-decision in _judge_tool_safety.
        self.pending_judge_deferral: str | None = None
        # Session-scoped LRU of judge verdicts, keyed on the exact call signature
        # (tool_name, args_repr, flagged_reasons). Repeated identical ASK-gated
        # calls reuse a verdict instead of re-querying the judge model. Only real
        # verdicts are cached; fail-closed ones (timeout/error) are retried.
        self._judge_verdict_cache: OrderedDict[
            tuple[str, str, tuple[str, ...], str], JudgeVerdict
        ] = OrderedDict()
        self._judge_verdict_cache_maxsize: int = config.safety_judge.verdict_cache_size
        # Judge model alias the cached verdicts were produced under. When the
        # configured judge model changes mid-session, the cache is cleared so a
        # verdict from one model is never reused under another.
        self._judge_model_alias_for_cache: str | None = None
        # When the active model is rate-limited/overloaded, the loop switches to
        # a configured fallback model for the rest of the session. Tracks the
        # override + which fallback aliases have already been tried.
        self._fallback_model_override: ModelConfig | None = None
        self._tried_fallback_aliases: set[str] = set()
        # When the model truncates output (ResponseTooLongError), the loop retries
        # the turn with a larger max_tokens. Per-turn override + attempt counter.
        self._max_output_override: int | None = None
        self._response_too_long_attempts: int = 0
        # True while inside a Stop-hook-induced continuation (passed to the next
        # Stop invocation so a hook can avoid an infinite continue loop).
        self._stop_hook_active: bool = False
        # SessionStart fires once on the first act() of a session.
        self._session_started: bool = False
        self._is_subagent = is_subagent
        self._memory_store: MemoryStore | None = None
        self._memory_applied = False
        self._mem_surfaced: set[str] = set()
        self._mem_extract_cursor: int = 0
        self._late_memory_section: str = ""
        self._mem_extract_writes: int = 0
        self._mem_extract_task: asyncio.Task[None] | None = None
        self._mem_prefetch_task: asyncio.Task[list[str]] | None = None
        self._mem_consolidate_task: asyncio.Task[None] | None = None
        self._mem_verify_task: asyncio.Task[None] | None = None
        self._memory_trash_swept: bool = False
        self.user_input_callback: UserInputCallback | None = None
        # Asked when a turn is rate-limited and no automatic fallback is
        # available, to let the user pick a model to switch to (the rate-limit
        # model-switch dialog). None in headless/ACP runs → surface the error.
        self.rate_limit_callback: RateLimitCallback | None = None
        self.entrypoint_metadata = entrypoint_metadata
        self.terminal_emulator = terminal_emulator

    def _init_telemetry(self, config: VibeConfig, is_subagent: bool) -> None:
        try:
            active_model = config.get_active_model()
            self.stats.input_price_per_million = active_model.input_price
            self.stats.output_price_per_million = active_model.output_price
        except ValueError:
            pass

        self._usage_recorder = get_usage_recorder()
        self._rate_limit_store = RateLimitStore()

        self._current_user_message_id: str | None = None
        self._is_user_prompt_call: bool = False
        self._pending_injected_messages: list[LLMMessage] = []
        self._subagent_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUBAGENTS)
        self._response_format: dict[str, Any] | None = None
        self.launch_workflow_callback: Callable[[str, str | None], str] | None = None
        self.workflow_status_callback: (
            Callable[[str | None], list[dict[str, Any]]] | None
        ) = None
        self.workflow_results_callback: Callable[..., dict[str, Any]] | None = None
        self.workflow_stop_callback: (
            Callable[[str | None, bool], Awaitable[dict[str, Any]]] | None
        ) = None
        self.team_dir_callback: Callable[[], str | None] | None = None
        # Unified background-task registry — owns bash-backgrounded processes
        # and aggregates workflows/teams/loops. None until wired by the entry
        # point (TUI VibeApp, run_programmatic, or ACP _create_agent_loop); the
        # bash tool refuses background=True without it.
        self.background_registry: BackgroundRegistry | None = None

        self.experiment_manager = ExperimentManager(
            client=RemoteEvalClient.from_settings(
                api_host=config.experiments.api_host,
                client_key=config.experiments.client_key,
            ),
            overrides=dict(config.experiment_overrides),
        )
        self.telemetry_client = TelemetryClient(
            config_getter=lambda: self.config,
            session_id_getter=lambda: self.session_id,
            parent_session_id_getter=lambda: self.parent_session_id,
            entrypoint_metadata_getter=lambda: self.entrypoint_metadata,
            experiments_getter=lambda: self.experiment_manager.assignments(),
        )
        self.session_logger = SessionLogger(config.session_logging, self.session_id)
        self.resource_monitor = ResourceMonitor(
            enabled=not is_subagent, label_getter=lambda: self.session_id
        )

    def _init_hooks(self, hook_config_result: HookConfigResult | None) -> None:
        self._hook_config_result = hook_config_result
        self._hooks_manager = (
            HooksManager(hook_config_result.hooks) if hook_config_result else None
        )
        self.hook_config_issues = (
            hook_config_result.issues if hook_config_result else []
        )
        self.hooks_count = len(hook_config_result.hooks) if hook_config_result else 0

    def _init_rewind(self) -> None:
        self.rewind_manager = RewindManager(
            messages=self.messages,
            save_messages=self._save_messages,
            reset_session=self._reset_session,
        )
        self._teleport_service: TeleportService | None = None

    @property
    def hooks_manager(self) -> HooksManager | None:
        return self._hooks_manager

    @property
    def hook_config_result(self) -> HookConfigResult | None:
        return self._hook_config_result

    def _start_deferred_init(self) -> threading.Thread:
        with self._deferred_init_lock:
            if self._deferred_init_thread is not None:
                return self._deferred_init_thread

            thread = threading.Thread(
                target=self._complete_init, daemon=True, name="agent_loop_init"
            )
            self._deferred_init_thread = thread
            thread.start()
            return thread

    @property
    def is_initialized(self) -> bool:
        if not self._defer_heavy_init:
            return True
        thread = self._deferred_init_thread
        return thread is not None and not thread.is_alive()

    def _complete_init(self) -> None:
        try:
            self._ensure_remote_registries()
            self.tool_manager.integrate_all(raise_on_mcp_failure=True)
            self._system_prompt_tier = self._current_baseline_tier()
            system_prompt = get_universal_system_prompt(
                self.tool_manager,
                self.config,
                self.skill_manager,
                self.agent_manager,
                scratchpad_dir=self.scratchpad_dir,
                headless=self._headless,
                experiment_manager=self.experiment_manager,
                tier=self._system_prompt_tier,
            )
            self.messages.update_system_prompt(system_prompt)
        except Exception as exc:
            self._init_error = exc

    async def wait_until_ready(self) -> None:
        if self._defer_heavy_init:
            thread = self._start_deferred_init()
            await asyncio.to_thread(thread.join)
            if err := self._init_error:
                raise copy.copy(err).with_traceback(err.__traceback__)
        if (task := self._experiments_task) is not None:
            if task is asyncio.current_task():
                return
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._ready_telemetry_pending:
            self._ready_telemetry_pending = False
            duration = int((time.monotonic() - self._init_start_time) * 1000)
            self.emit_ready_telemetry(duration)
        if self._pending_new_session_telemetry:
            self._pending_new_session_telemetry = False
            self.emit_new_session_telemetry()

    @property
    def agent_profile(self) -> AgentProfile:
        return self.agent_manager.active_profile

    @property
    def base_config(self) -> VibeConfig:
        return self._base_config

    @property
    def config(self) -> VibeConfig:
        return self.agent_manager.config

    @property
    def bypass_tool_permissions(self) -> bool:
        return self.config.bypass_tool_permissions

    def refresh_config(self) -> None:
        self._base_config = VibeConfig.load()
        self.agent_manager.invalidate_config()

    def _drain_pending_injections(self) -> bool:
        staged = False
        if self._pending_injected_messages:
            for injected in self._pending_injected_messages:
                self.messages.append(injected)
            self._pending_injected_messages.clear()
            staged = True
        if drain_diagnostics_into(self.stage_injected_message):
            for injected in self._pending_injected_messages:
                self.messages.append(injected)
            self._pending_injected_messages.clear()
            staged = True
        return staged

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        self.approval_callback = callback

    def set_scheduler(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def set_user_input_callback(self, callback: UserInputCallback) -> None:
        self.user_input_callback = callback

    def set_rate_limit_callback(self, callback: RateLimitCallback) -> None:
        self.rate_limit_callback = callback

    def set_tool_permission(
        self, tool_name: str, permission: ToolPermission, save_permanently: bool = False
    ) -> None:
        if save_permanently:
            VibeConfig.save_updates({
                "tools": {tool_name: {"permission": permission.value}}
            })

        self._permission_store.set_tool_permission(tool_name, permission)

    def approve_always(
        self,
        tool_name: str,
        required_permissions: list[RequiredPermission] | None,
        save_permanently: bool = False,
    ) -> None:
        if required_permissions:
            for rp in required_permissions:
                self._permission_store.add_rule(
                    ApprovedRule(
                        tool_name=tool_name,
                        scope=rp.scope,
                        session_pattern=rp.session_pattern,
                    )
                )
            if save_permanently:
                self.config.add_tool_allowlist_patterns(
                    tool_name, [rp.session_pattern for rp in required_permissions]
                )
        else:
            self.set_tool_permission(
                tool_name, ToolPermission.ALWAYS, save_permanently=save_permanently
            )

    def start_initialize_experiments(self) -> None:
        if self._experiments_task is not None:
            return
        self._pending_new_session_telemetry = True
        self._ready_telemetry_pending = True
        self._experiments_task = asyncio.create_task(self.initialize_experiments())

    async def initialize_experiments(self) -> None:
        updated = await session_initialize_experiments(
            config=self.config,
            manager=self.experiment_manager,
            session_logger=self.session_logger,
            entrypoint_metadata=self.entrypoint_metadata,
            terminal_emulator=self.terminal_emulator,
        )
        if updated:
            with contextlib.suppress(Exception):
                await self.refresh_system_prompt()

    async def hydrate_experiments_from_session(self) -> None:
        hydrated = await session_hydrate_experiments_from_session(
            config=self.config,
            manager=self.experiment_manager,
            session_logger=self.session_logger,
        )
        if hydrated:
            with contextlib.suppress(Exception):
                await self.refresh_system_prompt()

    def emit_new_session_telemetry(self) -> None:
        entrypoint = (
            self.entrypoint_metadata.agent_entrypoint
            if self.entrypoint_metadata
            else "unknown"
        )
        client_name = (
            self.entrypoint_metadata.client_name if self.entrypoint_metadata else None
        )
        client_version = (
            self.entrypoint_metadata.client_version
            if self.entrypoint_metadata
            else None
        )
        has_agents_md = has_agents_md_file(Path.cwd())
        nb_skills = len(self.skill_manager.available_skills)
        nb_mcp_servers = len(self.config.mcp_servers)
        nb_models = len(self.config.models)

        self.telemetry_client.send_new_session(
            has_agents_md=has_agents_md,
            nb_skills=nb_skills,
            nb_mcp_servers=nb_mcp_servers,
            nb_models=nb_models,
            entrypoint=entrypoint,
            client_name=client_name,
            client_version=client_version,
            terminal_emulator=self.terminal_emulator,
        )

    def emit_ready_telemetry(self, init_duration_ms: int) -> None:
        self.telemetry_client.send_ready(init_duration_ms=init_duration_ms)

    def emit_session_closed_telemetry(self) -> None:
        self.telemetry_client.send_session_closed()

    async def aclose(self) -> None:
        await self.resource_monitor.aclose()
        if (task := self._experiments_task) is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        # Reap the fire-and-forget memory tasks so a session ending mid-flight
        # can't leave a dangling, state-mutating consolidation. Order matters:
        # consolidation writes first (merge/trash), then verification (re-tag +
        # stamp), then extraction (upsert), then the short-lived prefetch. Each
        # is cancelled then awaited so no task outlives the loop.
        for attr in (
            "_mem_consolidate_task",
            "_mem_verify_task",
            "_mem_extract_task",
            "_mem_prefetch_task",
        ):
            task = getattr(self, attr)
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
            setattr(self, attr, None)
        with contextlib.suppress(Exception):
            await self.backend.__aexit__(None, None, None)
        with contextlib.suppress(Exception):
            await self.experiment_manager.aclose()
        # Tear down pooled MCP connections (subprocess/SSE) so no server is
        # left alive after the agent exits.
        if self.mcp_registry is not None:
            with contextlib.suppress(Exception):
                await self.mcp_registry.close()

    def _create_connector_registry(self) -> ConnectorRegistry | None:
        if not self._base_config.enable_connectors:
            return None

        # Connectors are a Mistral-cloud feature. Gate on the ACTIVE provider
        # being Mistral — get_mistral_provider() returns any Mistral provider
        # even when it isn't active, which spawned a registry (and a 401 fetch)
        # for users whose active provider is non-Mistral but who keep a leftover
        # Mistral provider with a set-but-unauthorized key.
        if not self._base_config.is_active_model_mistral():
            return None

        provider = self._base_config.get_active_provider()
        api_key_env = provider.api_key_env_var or "MISTRAL_API_KEY"
        api_key = resolve_api_key(api_key_env) or ""
        if not api_key:
            return None

        server_url = get_server_url_from_api_base(provider.api_base)
        from vibe.core.tools.connectors import ConnectorRegistry

        return ConnectorRegistry(api_key=api_key, server_url=server_url)

    @staticmethod
    def _create_mcp_registry() -> MCPRegistry:
        # Local import: keeps the external MCP SDK (pulled transitively by the
        # registry) off the CLI import path so cold start stays lazy.
        from vibe.core.tools.mcp import MCPRegistry

        return MCPRegistry()

    def _ensure_remote_registries(self) -> None:
        if self.mcp_registry is None and self.config.mcp_servers:
            self.mcp_registry = self._create_mcp_registry()
            self.tool_manager.set_mcp_registry(self.mcp_registry)

        if self.connector_registry is None:
            self.connector_registry = self._create_connector_registry()
            self.tool_manager.set_connector_registry(self.connector_registry)

    # No @requires_init (async-only guard): runs during __init__ and inside
    # _complete_init itself, where waiting on init would deadlock.
    def _current_baseline_tier(self) -> BaselineTier:
        try:
            model = self.effective_model()
        except Exception:
            try:
                model = self.config.get_active_model()
            except Exception:
                return BaselineTier.LARGE
        return baseline_tier_for(model, self.config)

    def _sync_baseline_tier(self) -> None:
        # Rebuild messages[0] when the effective model's baseline tier drifts
        # (e.g. a failover from a small-window to a large-window model), so the
        # system prompt, tool schemas, and tier all describe the same model.
        tier = self._current_baseline_tier()
        if tier == self._system_prompt_tier:
            return
        self._system_prompt_tier = tier
        system_prompt = get_universal_system_prompt(
            self.tool_manager,
            self.config,
            self.skill_manager,
            self.agent_manager,
            scratchpad_dir=self.scratchpad_dir,
            headless=self._headless,
            experiment_manager=self.experiment_manager,
            tier=tier,
        )
        self.messages.update_system_prompt(system_prompt)

    async def refresh_system_prompt(self) -> None:
        self._system_prompt_tier = self._current_baseline_tier()
        system_prompt = get_universal_system_prompt(
            self.tool_manager,
            self.config,
            self.skill_manager,
            self.agent_manager,
            scratchpad_dir=self.scratchpad_dir,
            headless=self._headless,
            experiment_manager=self.experiment_manager,
            tier=self._system_prompt_tier,
        )
        self.messages.update_system_prompt(system_prompt)

    def _select_backend(self) -> BackendLike:
        provider = self.config.get_active_provider()
        timeout = self.config.api_timeout
        return create_backend(provider=provider, timeout=timeout)

    async def _save_messages(self) -> None:
        await self.session_logger.save_interaction(
            self.messages,
            self.stats,
            self._base_config,
            self.tool_manager,
            self.agent_profile,
        )

    @requires_init
    async def inject_user_context(
        self,
        content: str,
        *,
        as_message: bool = False,
        images: list[ImageAttachment] | None = None,
        client_message_id: str | None = None,
    ) -> None:
        if as_message:
            self.messages.append(
                LLMMessage(
                    role=Role.USER,
                    content=content,
                    message_id=client_message_id or str(uuid4()),
                    images=images or None,
                )
            )
        else:
            self.messages.append(
                LLMMessage(
                    role=Role.USER,
                    content=content,
                    injected=True,
                    injected_kind=InjectedMessageKind.USER_CONTEXT,
                    images=images or None,
                )
            )
        await self._save_messages()

    def stage_injected_message(
        self,
        content: str,
        *,
        images: list[ImageAttachment] | None = None,
        client_message_id: str | None = None,
    ) -> None:
        self._pending_injected_messages.append(
            LLMMessage(
                role=Role.USER,
                content=content,
                injected=True,
                injected_kind=InjectedMessageKind.STAGED,
                images=images or None,
                message_id=client_message_id,
            )
        )

    @requires_init
    async def act(
        self,
        msg: str,
        client_message_id: str | None = None,
        *,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        self._response_format = response_format
        self.resource_monitor.start()
        # No-op unless VIBE_TRACE_LOOP is set; covers the shared Textual loop.
        loop_tracer.install()
        # No-op unless VIBE_TRACE_STREAM is set; turn id matches profiler.section.
        stream_tracer.turn_started(
            self,
            f"{self.session_id[:8]}-{self.stats.steps}",
            is_subagent=self._is_subagent,
        )
        try:
            try:
                active_model = self.effective_model()
                model_name = active_model.name
            except ValueError:
                active_model = None
                model_name = None
            if images and active_model is not None and not active_model.supports_images:
                raise ImagesNotSupportedError(active_model.alias)
            self._clean_message_history()
            self.rewind_manager.create_checkpoint()
            if not self._session_started:
                self._session_started = True
                source = "resume" if self.parent_session_id else "startup"
                ss_injected, ss_events = await self._dispatch_session_start_hooks(
                    source
                )
                for ev in ss_events:
                    yield ev
                # Append session-start context BEFORE the user prompt (which
                # _conversation_loop appends next) so the first turn sees it as
                # a session preamble.
                for ctx_text in ss_injected:
                    self.messages.append(
                        LLMMessage(
                            role=Role.USER,
                            content=ctx_text,
                            injected=True,
                            injected_kind=InjectedMessageKind.SESSION_START,
                        )
                    )
            agent_provider: str | None = None
            if active_model is not None:
                try:
                    agent_provider = self.config.get_provider_for_model(
                        active_model
                    ).name
                except Exception:
                    agent_provider = None
            async with agent_span(
                model=model_name,
                session_id=self.session_id,
                provider=agent_provider,
                agent_profile=self.agent_profile.name,
            ) as agent_turn_span:
                prompt0 = self.stats.session_prompt_tokens
                completion0 = self.stats.session_completion_tokens
                cached0 = self.stats.session_cached_tokens
                try:
                    with (
                        profiler.section(
                            f"turn-{self.session_id[:8]}-{self.stats.steps}",
                            turn=self.stats.steps,
                        ),
                        self.resource_monitor.turn(),
                    ):
                        async for event in self._conversation_loop(
                            msg,
                            client_message_id=client_message_id,
                            auto_title=auto_title,
                            images=images,
                        ):
                            yield event
                finally:
                    set_agent_usage(
                        agent_turn_span,
                        input_tokens=self.stats.session_prompt_tokens - prompt0,
                        output_tokens=self.stats.session_completion_tokens
                        - completion0,
                        cached_tokens=self.stats.session_cached_tokens - cached0,
                    )
        finally:
            self._response_format = None
            stream_tracer.turn_finished(self)

    @property
    def teleport_service(self) -> TeleportService:
        if not _TELEPORT_AVAILABLE:
            raise TeleportError(
                "Teleport requires git to be installed. "
                "Please install git and try again."
            )
        service_cls = _teleport_service_cls()
        if service_cls is None:
            raise TeleportError("_TeleportService is unexpectedly None")

        if self._teleport_service is None:
            self._teleport_service = service_cls(
                session_logger=self.session_logger,
                vibe_code_sessions_base_url=self.config.vibe_code_sessions_base_url,
                vibe_code_api_key=self.config.vibe_code_api_key,
                vibe_config=self._base_config,
            )
        return self._teleport_service

    @requires_init
    async def teleport_to_vibe_code(
        self, prompt: str | None
    ) -> AsyncGenerator[TeleportYieldEvent, TeleportPushResponseEvent | None]:
        nb_session_messages = max(len(self.messages) - 1, 0)
        if prompt:
            resolved_prompt = prompt
        else:
            last = self._last_user_message()
            content = last.content if last else None
            resolved_prompt = (
                f"{content} (continue)" if isinstance(content, str) and content else ""
            )
        telemetry_tracker = TeleportTelemetryTracker(
            telemetry_client=self.telemetry_client,
            nb_session_messages=nb_session_messages,
            stage="no_history" if not resolved_prompt else "git_check",
        )
        try:
            async with self.teleport_service:
                gen = self.teleport_service.execute(prompt=resolved_prompt)
                response: TeleportPushResponseEvent | None = None
                while True:
                    try:
                        event = await gen.asend(response)
                        telemetry_tracker.record_event(event)
                        if isinstance(event, TeleportCompleteEvent):
                            telemetry_tracker.send_success()
                        response = yield event
                    except StopAsyncIteration:
                        break
        except ServiceTeleportError as e:
            telemetry_tracker.record_service_error(e)
            raise TeleportError(str(e)) from e
        except (asyncio.CancelledError, GeneratorExit):
            telemetry_tracker.record_cancelled()
            raise
        except Exception as e:
            telemetry_tracker.record_unexpected_error(e)
            raise
        finally:
            telemetry_tracker.send_failure_if_needed()
            self._teleport_service = None

    def _last_user_message(self) -> LLMMessage | None:
        return next(
            (
                m
                for m in reversed(self.messages)
                if m.role == Role.USER and not m.injected
            ),
            None,
        )

    def set_max_turns(self, max_turns: int) -> None:
        self._max_turns = max_turns
        self._setup_middleware()

    def set_max_tokens(self, max_tokens: int) -> None:
        self._max_output_override = max_tokens

    def _setup_middleware(self) -> None:
        self.middleware_pipeline.clear()

        if self._max_turns is not None:
            self.middleware_pipeline.add(TurnLimitMiddleware(self._max_turns))

        if self._max_price is not None:
            self.middleware_pipeline.add(PriceLimitMiddleware(self._max_price))

        if self._max_session_tokens is not None:
            self.middleware_pipeline.add(TokenLimitMiddleware(self._max_session_tokens))

        # Heuristic: detect an agent stuck repeating the same tool call, beyond
        # the hard turn cap. Cheap (history-only), no extra model calls.
        self.middleware_pipeline.add(LoopDetectionMiddleware())

        # Cheap, local context shapers run before the LLM-summary fallback.
        # Both no-op when disabled (read their config live), so registration is
        # unconditional and a mid-session config edit takes effect immediately.
        self.middleware_pipeline.add(
            ToolResultBudgetMiddleware(
                self.tool_result_store,
                AGGREGATE_TOOL_RESULT_CHARS,
                keep_recent_messages=self.config.context_shaping.snip.keep_recent_turns,
            )
        )
        self.middleware_pipeline.add(SnipMiddleware())
        self.middleware_pipeline.add(MicrocompactMiddleware())
        self.middleware_pipeline.add(AutoCompactMiddleware())
        if self.config.context_warnings:
            self.middleware_pipeline.add(ContextWarningMiddleware(0.5))

        self.middleware_pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: self.agent_profile,
                BuiltinAgentName.PLAN,
                lambda: make_plan_agent_reminder(
                    self._plan_session.plan_file_path_str,
                    has_ask_user_question="ask_user_question"
                    in self.tool_manager.available_tools,
                    has_exit_plan_mode="exit_plan_mode"
                    in self.tool_manager.available_tools,
                ),
                PLAN_AGENT_EXIT,
            )
        )
        self.middleware_pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: self.agent_profile,
                BuiltinAgentName.CHAT,
                CHAT_AGENT_REMINDER,
                CHAT_AGENT_EXIT,
            )
        )

    async def _handle_middleware_result(
        self, result: MiddlewareResult
    ) -> AsyncGenerator[BaseEvent]:
        match result.action:
            case MiddlewareAction.STOP:
                yield AssistantEvent(
                    content=f"<{VIBE_STOP_EVENT_TAG}>{result.reason}</{VIBE_STOP_EVENT_TAG}>",
                    stopped_by_middleware=True,
                )

            case MiddlewareAction.INJECT_MESSAGE:
                if result.message:
                    injected_message = LLMMessage(
                        role=Role.USER,
                        content=result.message,
                        injected=True,
                        injected_kind=InjectedMessageKind.MIDDLEWARE,
                    )
                    self.messages.append(injected_message)

            case MiddlewareAction.COMPACT:
                old_tokens = result.metadata.get(
                    "old_tokens", self.stats.context_tokens
                )
                threshold = result.metadata.get(
                    "threshold", self.effective_model().auto_compact_threshold
                )
                async for ev in self._run_compaction(old_tokens, threshold):
                    yield ev

            case MiddlewareAction.CONTINUE:
                pass

    async def _run_compaction(
        self, old_tokens: int, threshold: int, *, trigger: str = "auto"
    ) -> AsyncGenerator[BaseEvent, None]:
        old_session_id = self.session_id
        old_parent_session_id = self.parent_session_id
        old_cached = self.stats.last_turn_cached_tokens
        tool_call_id = str(uuid4())

        logger.debug(
            "compaction started (trigger=%s): context=%d threshold=%d cached=%d. "
            "Expect a cache-read drop on the next turn — this is expected, not a "
            "provider cache failure.",
            trigger,
            old_tokens,
            threshold,
            old_cached,
        )

        yield CompactStartEvent(
            tool_call_id=tool_call_id,
            current_context_tokens=old_tokens,
            threshold=threshold,
        )

        # Notify pre-compaction hooks (observe-only; never blocks compaction).
        try:
            async for ev in self._run_pre_compact_hooks(trigger, old_tokens, threshold):
                yield ev
        except Exception as e:
            logger.warning("pre_compact hook failed (%s); compacting anyway", e)

        compact_status: Literal["success", "failure", "cancelled"] = "success"
        async with context_shaping_span(op="compact", trigger=trigger) as span:
            try:
                summary = await self.compact()
            except asyncio.CancelledError:
                compact_status = "cancelled"
                raise
            except Exception:
                compact_status = "failure"
                raise
            finally:
                self.telemetry_client.send_auto_compact_triggered(
                    nb_context_tokens_before=old_tokens,
                    auto_compact_threshold=threshold,
                    status=compact_status,
                    session_id=old_session_id,
                    parent_session_id=old_parent_session_id,
                )
                from vibe.core.utils.tokens import approx_token_count

                after = sum(approx_token_count(m.content or "") for m in self.messages)
                set_context_shaping_result(
                    span,
                    tokens_before=old_tokens,
                    tokens_after=after,
                    threshold=threshold,
                    status=compact_status,
                )

        # Snapshot what shaped this conversation for downstream observability.
        # Pure record — never consulted by the turn loop. Hash the system prompt
        # (messages[0]) so a recompaction under a different prompt is detectable
        # without retaining the full prompt text on the event.
        system_prompt_text = self.messages[0].content or ""
        yield CompactEndEvent(
            tool_call_id=tool_call_id,
            summary_length=len(summary),
            old_session_id=old_session_id,
            new_session_id=self.session_id,
            origin=CompactionOrigin(
                model_alias=self.effective_model().alias,
                agent_profile=self.agent_profile.name,
                system_prompt_hash=hashlib.sha256(
                    system_prompt_text.encode("utf-8")
                ).hexdigest()[:16],
            ),
        )

    def effective_model(self) -> ModelConfig:
        return self._fallback_model_override or self.config.get_active_model()

    def _get_context(self) -> ConversationContext:
        return ConversationContext(
            messages=self.messages,
            stats=self.stats,
            config=self.config,
            active_model=self.effective_model(),
        )

    async def _try_reactive_shaping(self) -> bool:
        threshold = self.effective_model().auto_compact_threshold
        if threshold <= 0:
            return False
        ctx = self._get_context()
        snip = SnipMiddleware()
        microcompact = MicrocompactMiddleware()
        start = snip.estimated_tokens(ctx)
        for _ in range(12):  # bounded; microcompact does ~1 block/call
            before = snip.estimated_tokens(ctx)
            await snip.before_turn(ctx)
            await microcompact.before_turn(ctx)
            after = snip.estimated_tokens(ctx)
            if after >= before:
                break
        return snip.estimated_tokens(ctx) < start

    def _build_backend_metadata(
        self, call_type: TelemetryCallType | None = None
    ) -> TelemetryRequestMetadata:
        return build_request_metadata(
            entrypoint_metadata=self.entrypoint_metadata,
            session_id=self.session_id,
            parent_session_id=self.parent_session_id,
            call_type=(
                call_type
                if call_type is not None
                else ("main_call" if self._is_user_prompt_call else "secondary_call")
            ),
            message_id=self._current_user_message_id,
        )

    def _get_extra_headers(
        self, provider: ProviderConfig | None = None
    ) -> dict[str, str]:
        provider = self.config.get_active_provider() if provider is None else provider
        headers: dict[str, str] = {**provider.extra_headers}
        if not any(k.lower() == "user-agent" for k in headers):
            headers["user-agent"] = get_user_agent(provider.backend)
        headers["x-affinity"] = self.session_id
        # The codex backend pins cache routing on per-conversation identity
        # headers (it does not route on the body prompt_cache_key alone). Send
        # the session id as both, matching codex and the body prompt_cache_key.
        # Scoped to openai-chatgpt so other providers are untouched. The volatile
        # x-codex-turn-state token is added per-call by the chat methods, not here.
        if getattr(provider, "api_style", "") == "openai-chatgpt":
            headers["session-id"] = self.session_id
            headers["thread-id"] = self.session_id
        return headers

    def _codex_routing(
        self, provider: ProviderConfig
    ) -> tuple[dict[str, str], dict[str, str] | None]:
        headers = self._get_extra_headers(provider)
        if getattr(provider, "api_style", "") != "openai-chatgpt":
            return headers, {}
        if self._is_user_prompt_call:
            self._codex_turn_state = None
        elif self._codex_turn_state:
            headers["x-codex-turn-state"] = self._codex_turn_state
        return headers, {}

    def _capture_codex_turn_state(self, sink: dict[str, str] | None) -> None:
        if sink and (ts := sink.get("x-codex-turn-state")):
            self._codex_turn_state = ts

    def _capture_rate_limits(
        self, provider: ProviderConfig, sink: dict[str, str] | None
    ) -> None:
        if not sink:
            return
        snapshot = rate_limit_from_headers(provider.name, sink, captured_at=time.time())
        if snapshot is not None:
            self._rate_limit_store.update(snapshot)

    @staticmethod
    def _wire_temperature(
        active_model: ModelConfig, provider: ProviderConfig
    ) -> float | None:
        # Temperature as actually sent, for the trace: the Responses API
        # (gpt-5.x/codex/fugu) omits it for non gpt-4/3.5 models, so recording the
        # config value would over-report a temperature that never hits the wire.
        api_style = getattr(provider, "api_style", "openai")
        if api_style in {"openai-responses", "openai-chatgpt"}:
            from vibe.core.llm.backend.openai_responses import (
                responses_temperature_supported,
            )

            if not responses_temperature_supported(active_model.name):
                return None
        return active_model.temperature

    def _trace_recovery(self, *, error_type: str, action: str, **extra: Any) -> None:
        # Record a self-heal (failover / escalation / compaction) on the active
        # span, so a trace shows why a turn retried instead of just failing.
        attrs: dict[str, Any] = {
            "error.type": error_type,
            "vibe.recovery.action": action,
        }
        for key, value in extra.items():
            if value is not None:
                attrs[f"vibe.recovery.{key}"] = value
        try:
            trace.get_current_span().add_event("llm_recovery", attrs)
        except Exception:
            pass

    def _escalate_max_output(self) -> int | None:
        from vibe.core.llm.backend.generic import adapter_supports_max_output_escalation

        esc = self.config.max_output_escalation
        if not esc.enabled:
            return None
        # Gate and clamp against the model actually on the wire: the failover
        # override wins (matches _resolve_active_model in the backend mixin).
        model = self.effective_model()
        provider = self.config.get_provider_for_model(model)
        # Codex strips max_output_tokens, so an escalated retry would be
        # byte-identical: go straight to the terminal path instead.
        if not adapter_supports_max_output_escalation(provider.api_style):
            return None
        self._response_too_long_attempts += 1
        if self._response_too_long_attempts > esc.max_attempts:
            return None
        cap = model.max_output_tokens or esc.cap
        current = self._max_output_override or esc.base
        next_val = min(int(current * esc.factor), cap)
        if next_val <= (self._max_output_override or 0):
            # Already at cap; a further retry would request the same size.
            return None
        self._max_output_override = next_val
        return next_val

    async def _conversation_loop(
        self,
        user_msg: str,
        client_message_id: str | None = None,
        *,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> AsyncGenerator[BaseEvent]:
        if not self._files_read_reconstructed:
            self._reconstruct_files_read()
        await self._check_agents_md_changed()

        user_message = LLMMessage(
            role=Role.USER,
            content=user_msg,
            message_id=client_message_id,
            images=images or None,
        )
        self.messages.append(user_message)
        self.stats.steps += 1
        self._current_user_message_id = user_message.message_id

        if user_message.message_id is None:
            raise AgentLoopError("User message must have a message_id")

        yield UserMessageEvent(content=user_msg, message_id=user_message.message_id)

        (
            block_reason,
            injected_ctx,
            hook_events,
        ) = await self._dispatch_user_prompt_submit_hooks(
            user_msg, user_message.message_id, bool(images)
        )
        for hook_event in hook_events:
            yield hook_event
        if block_reason is not None:
            # Redact the stored user prompt: a hook denied it (often because it
            # held a secret or an injection), so the raw content must not be
            # retained in the transcript nor re-sent to the model on later turns.
            # The message slot stays (same id) for transcript coherence; the UI
            # already showed the user what they typed via UserMessageEvent above.
            user_message.content = (
                "[blocked by user_prompt_submit hook; content redacted] "
                f"reason: {block_reason}"
            )
            blocked = LLMMessage(
                role=Role.ASSISTANT, content=f"Prompt blocked by hook: {block_reason}"
            )
            self.messages.append(blocked)
            yield AssistantEvent(
                content=blocked.content or "", message_id=blocked.message_id
            )
            # This early return sits above the try/finally that persists the
            # transcript on every other exit; mirror it so the blocked turn is
            # saved and any pending injections drained on the same boundary.
            self._drain_pending_injections()
            await self._save_messages()
            return
        for ctx_text in injected_ctx:
            self.messages.append(
                LLMMessage(
                    role=Role.USER,
                    content=ctx_text,
                    injected=True,
                    injected_kind=InjectedMessageKind.USER_PROMPT_HOOK,
                )
            )

        if self.config.memory.prefetch:
            self._kick_memory_prefetch(user_msg)
        else:
            await self._apply_memory_selection(user_msg)

        if auto_title is not None and self.session_logger.set_initial_auto_title(
            auto_title
        ):
            yield SessionTitleUpdatedEvent(title=auto_title)

        if self._hooks_manager:
            self._hooks_manager.reset_retry_count()

        try:
            should_break_loop = False
            first_llm_turn = True
            emergency_compacted = False
            shaping_attempted = False
            # Output-escalation state is scoped to this user turn.
            self._max_output_override = None
            self._response_too_long_attempts = 0
            while not should_break_loop:
                self._is_user_prompt_call = False
                # Re-tier the system prompt BEFORE the shapers measure context,
                # so a mid-turn failover's baseline is reflected this turn.
                self._sync_baseline_tier()
                result = await self.middleware_pipeline.run_before_turn(
                    self._get_context()
                )
                async for event in self._handle_middleware_result(result):
                    yield event

                if result.action == MiddlewareAction.STOP:
                    return

                self.stats.steps += 1
                user_cancelled = False
                async for ev in self._drain_async_agent_completions():
                    yield ev
                if first_llm_turn:
                    self._is_user_prompt_call = True
                    first_llm_turn = False
                    # Fold in deep recall only if it settled while middleware
                    # yielded to the event loop; never blocks the first call.
                    self._consume_memory_prefetch()
                try:
                    async for event in self._perform_llm_turn():
                        if is_user_cancellation_event(event):
                            user_cancelled = True
                        yield event
                except ContextTooLongError:
                    # Self-heal: first try the cheap shapers aggressively (no LLM
                    # call) to chip away at old history, and only fall back to the
                    # nuclear LLM-summary compaction if they can't recover. If it
                    # recurs after compaction, surface the error as before.
                    if emergency_compacted:
                        raise
                    if not shaping_attempted:
                        shaping_attempted = True
                        logger.warning(
                            "Context overflow; trying reactive shaping before "
                            "compaction"
                        )
                        if await self._try_reactive_shaping():
                            self._trace_recovery(
                                error_type="context_too_long", action="reactive_shaping"
                            )
                            continue
                    emergency_compacted = True
                    threshold = self.effective_model().auto_compact_threshold
                    async for ev in self._run_compaction(
                        self.stats.context_tokens, threshold, trigger="emergency"
                    ):
                        yield ev
                    self._trace_recovery(
                        error_type="context_too_long", action="emergency_compact"
                    )
                    continue
                except RateLimitError as e:
                    fallback = self._switch_to_fallback_model()
                    if fallback is None:
                        if self.rate_limit_callback is not None:
                            fallback = await self._prompt_model_switch_on_rate_limit(e)
                        else:
                            fallback = self._auto_fallback_headless()
                    self._apply_failover(
                        e,
                        fallback,
                        error_type="rate_limit",
                        unavailable_reason="Active model rate-limited",
                        log_template="Active model rate-limited; switching to %r",
                    )
                    continue
                except ContentFilterError as e:
                    self._apply_failover(
                        e,
                        self._switch_to_fallback_model(),
                        error_type="content_filter",
                        unavailable_reason=f"Request blocked by {e.provider!r} content filter",
                        log_template="Request blocked by %r content filter; falling back to %r",
                        log_prefix_args=(e.provider,),
                    )
                    continue
                except ServerError as e:
                    self._apply_failover(
                        e,
                        self._switch_to_fallback_model(),
                        error_type="server_error",
                        unavailable_reason=f"{e.provider!r} backend server error",
                        log_template="%r backend server error; falling back to %r",
                        log_prefix_args=(e.provider,),
                    )
                    continue
                except TransportError as e:
                    # A dropped connection (no HTTP response) reaches here after
                    # the backend's own connection-error retries are exhausted
                    # (mistral.py RetryConfig, retry_connection_errors=True). The
                    # loop-level response is failover to a different backend's
                    # connection, not another in-place retry on the same dead
                    # socket. Without this clause the bare RuntimeError fallthrough
                    # in _raise_for_backend_error terminated the turn.
                    self._apply_failover(
                        e,
                        self._switch_to_fallback_model(),
                        error_type="transport",
                        unavailable_reason=f"{e.provider!r} backend dropped the connection",
                        log_template="%r backend dropped the connection; falling back to %r",
                        log_prefix_args=(e.provider,),
                    )
                    continue
                except ResponseTooLongError:
                    nxt = self._escalate_max_output()
                    if nxt is None:
                        raise
                    logger.warning(
                        "Response truncated; retrying turn with max_tokens=%d", nxt
                    )
                    self._trace_recovery(
                        error_type="response_too_long",
                        action="escalate_max_output",
                        new_max_tokens=nxt,
                    )
                    continue
                # Per-turn save so the on-disk log stays fresh; after the
                # inner loop so before_tool rewrites land in the snapshot.
                await self._save_messages()
                self._is_user_prompt_call = False

                last_message = self.messages[-1]
                should_break_loop = last_message.role != Role.TOOL

                if self._drain_pending_injections():
                    should_break_loop = False

                if user_cancelled:
                    return

                if should_break_loop:
                    retry_msg, hook_events = await self._dispatch_post_turn_hooks()
                    for hook_event in hook_events:
                        yield hook_event
                    if retry_msg is not None:
                        self.messages.append(retry_msg)
                        should_break_loop = False

                if should_break_loop:
                    # Stop hooks get the last word: a deny injects a continuation
                    # and re-enters the loop (capped by HookRetryState; guarded by
                    # stop_hook_active to prevent runaway continues).
                    stop_msg, stop_events = await self._dispatch_stop_hooks(
                        self._stop_hook_active
                    )
                    for hook_event in stop_events:
                        yield hook_event
                    if stop_msg is not None:
                        self.messages.append(stop_msg)
                        should_break_loop = False
                        self._stop_hook_active = True
                    else:
                        self._stop_hook_active = False

            self._maybe_schedule_memory_extraction()
            # Periodic consolidation runs only after extraction has settled (_mem_extract_task
            # in-flight guard) so the two never mutate the store concurrently.
            self._maybe_schedule_consolidation()
            # Periodic verification runs last: it reads the store and re-tags,
            # so it must wait on consolidation's writes and extraction's upserts.
            self._maybe_schedule_verification()
        finally:
            # Abandon any prefetch that never settled so it can't leak across
            # turns or race the next kick.
            self._cancel_memory_prefetch()
            # Fold in any messages staged after the loop's last drain (e.g. a
            # double-enter inject that landed during the post-turn hooks) so
            # they become injected context for the next turn instead of being
            # lost. The save runs on every exit (including the middleware-STOP
            # and user-cancelled return paths above, which skip the per-turn
            # save at the top of the loop).
            self._drain_pending_injections()
            await self._save_messages()

    def _handle_plan_review_ended(self) -> None:
        if not self._plan_session.has_content_changed():
            return None

        content = self._plan_session.read()
        if content is None:
            return None

        msg = LLMMessage(
            role=Role.USER,
            content=(
                f"<{VIBE_WARNING_TAG}>The user has manually updated the plan file. "
                f"Here is the updated version -- use this as the source of truth "
                f"for implementation:\n\n{content}</{VIBE_WARNING_TAG}>"
            ),
            injected=True,
            injected_kind=InjectedMessageKind.PLAN_UPDATE,
        )
        self._pending_injected_messages.append(msg)

    def _handle_session_plan_events(self, event: BaseEvent) -> BaseEvent | None:
        if isinstance(event, ToolCallEvent) and event.tool_name == "exit_plan_mode":
            self._plan_session.snapshot_content_hash()
            return PlanReviewRequestedEvent(file_path=self._plan_session.plan_file_path)

        if isinstance(event, ToolResultEvent) and event.tool_name == "exit_plan_mode":
            self._handle_plan_review_ended()
            return PlanReviewEndedEvent()

        return None

    async def _drain_async_agent_completions(
        self,
    ) -> AsyncGenerator[BackgroundTaskCompletedEvent]:
        registry = self.background_registry
        if registry is None:
            return
        for rec in registry.pop_async_completions():
            summary = (
                rec.response if rec.response else f"[{rec.agent} produced no output]"
            )
            if rec.error:
                summary += f"\n[error: {rec.error}]"
            label = getattr(rec, "label", None) or rec.agent
            message = (
                f"[background subagent {rec.task_id} ({label}) "
                f"{'completed' if rec.completed else 'failed'}]\n{summary}"
            )
            self.messages.append(
                LLMMessage(
                    role=Role.USER,
                    content=message,
                    injected=True,
                    injected_kind=InjectedMessageKind.BACKGROUND_TASK,
                )
            )
            yield BackgroundTaskCompletedEvent(
                task_id=rec.task_id,
                agent=rec.agent,
                response=rec.response,
                completed=rec.completed,
                worktree_path=rec.worktree_path,
                branch=rec.branch,
                error=rec.error,
            )

    async def _perform_llm_turn(self) -> AsyncGenerator[BaseEvent, None]:
        if self.enable_streaming:
            async for event in self._stream_assistant_events():
                yield event
        else:
            assistant_event = await self._get_assistant_event()
            if assistant_event.content:
                yield assistant_event

        last_message = self.messages[-1]

        parsed = self.format_handler.parse_message(last_message)
        resolved = self.format_handler.resolve_tool_calls(parsed, self.tool_manager)

        if not resolved.tool_calls and not resolved.failed_calls:
            return

        profile_before = self.agent_profile.name
        async for event in self._handle_tool_calls(resolved):
            yield event

            if session_plan_event := self._handle_session_plan_events(event):
                yield session_plan_event

        if self.agent_profile.name != profile_before:
            yield AgentProfileChangedEvent(agent_name=self.agent_profile.name)

    def _build_tool_call_events(
        self, tool_calls: list[ToolCall] | None, emitted_ids: set[str]
    ) -> Generator[ToolCallEvent, None, None]:
        for tc in tool_calls or []:
            if tc.id is None or not tc.function.name:
                continue
            if tc.id in emitted_ids:
                continue

            tool_class = self.tool_manager.manifest_tools.get(tc.function.name)
            if tool_class is None:
                continue

            yield ToolCallEvent(
                tool_call_id=tc.id,
                tool_call_index=tc.index,
                tool_name=tc.function.name,
                tool_class=tool_class,
            )

    async def _stream_assistant_events(
        self,
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | ToolCallEvent]:
        message_id: str | None = None
        reasoning_message_id: str | None = None
        emitted_tool_call_ids = set[str]()

        async for chunk in self._chat_streaming():
            if message_id is None:
                message_id = chunk.message.message_id
            if reasoning_message_id is None:
                reasoning_message_id = chunk.message.reasoning_message_id

            for event in self._build_tool_call_events(
                chunk.message.tool_calls, emitted_tool_call_ids
            ):
                emitted_tool_call_ids.add(event.tool_call_id)
                yield event

            if chunk.message.reasoning_content:
                yield ReasoningEvent(
                    content=chunk.message.reasoning_content,
                    message_id=reasoning_message_id,
                )

            if chunk.message.content:
                yield AssistantEvent(
                    content=chunk.message.content, message_id=message_id
                )

    async def _get_assistant_event(self) -> AssistantEvent:
        llm_result = await self._chat()
        return AssistantEvent(
            content=llm_result.message.content or "",
            message_id=llm_result.message.message_id,
        )

    async def _handle_tool_calls(
        self, resolved: ResolvedMessage
    ) -> AsyncGenerator[ToolCallEvent | ToolResultEvent | ToolStreamEvent | HookEvent]:
        async for event in self._emit_failed_tool_events(resolved.failed_calls):
            yield event
        if not resolved.tool_calls:
            return

        for tool_call in resolved.tool_calls:
            yield ToolCallEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                args=tool_call.validated_args,
                tool_call_id=tool_call.call_id,
            )

        async for event in self._run_tools_concurrently(resolved.tool_calls):
            yield event

    async def _emit_failed_tool_events(
        self, failed_calls: list[FailedToolCall]
    ) -> AsyncGenerator[ToolResultEvent]:
        for failed in failed_calls:
            error_msg = f"<{TOOL_ERROR_TAG}>{failed.tool_name}: {failed.error}</{TOOL_ERROR_TAG}>"
            yield ToolResultEvent(
                tool_name=failed.tool_name,
                tool_class=None,
                error=error_msg,
                tool_call_id=failed.call_id,
            )
            self.stats.tool_calls_failed += 1
            self.messages.append(
                self.format_handler.create_failed_tool_response_message(
                    failed, error_msg
                )
            )

    async def _run_tools_concurrently(
        self, tool_calls: list[ResolvedToolCall]
    ) -> AsyncGenerator[ToolCallEvent | ToolResultEvent | ToolStreamEvent | HookEvent]:
        queue: asyncio.Queue[
            ToolCallEvent | ToolResultEvent | ToolStreamEvent | HookEvent | None
        ] = asyncio.Queue()

        def _concurrent_safe(tc: ResolvedToolCall) -> bool:
            # Per-call, not just the static read_only flag: a `task` spawning a
            # read-only in-process subagent is safe to fan out, while one
            # spawning a write-capable subagent must serialize.
            return tc.tool_class.call_is_read_only(
                tc.validated_args, agent_manager=self.agent_manager
            )

        readers = [tc for tc in tool_calls if _concurrent_safe(tc)]
        writers = [tc for tc in tool_calls if not _concurrent_safe(tc)]

        async def _run_writers_sequentially() -> None:
            for tc in writers:
                await self._execute_tool_to_queue(tc, queue)

        tasks = [
            asyncio.create_task(self._execute_tool_to_queue(tc, queue))
            for tc in readers
        ]
        if writers:
            tasks.append(asyncio.create_task(_run_writers_sequentially()))

        async def _signal_when_all_done() -> None:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await queue.put(None)

        monitor = asyncio.create_task(_signal_when_all_done())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        except GeneratorExit:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise
        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if not monitor.done():
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor

    async def _execute_tool_to_queue(
        self,
        tc: ResolvedToolCall,
        queue: asyncio.Queue[
            ToolCallEvent | ToolResultEvent | ToolStreamEvent | HookEvent | None
        ],
    ) -> None:
        # Cap concurrent subagent fan-out so a batch of independent task calls
        # doesn't overwhelm the backend; other tools run uncapped.
        if tc.tool_class.is_subagent_spawner:
            async with self._subagent_semaphore:
                async for event in self._process_one_tool_call(tc):
                    await queue.put(event)
        else:
            async for event in self._process_one_tool_call(tc):
                await queue.put(event)

    async def _process_one_tool_call(
        self, tool_call: ResolvedToolCall
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        async with tool_span(
            tool_name=tool_call.tool_name,
            call_id=tool_call.call_id,
            arguments=tool_call.validated_args.model_dump_json(),
        ) as span:
            async for event in self._execute_tool_call(span, tool_call):
                yield event

    async def _execute_tool_call(
        self, span: trace.Span, tool_call: ResolvedToolCall
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        try:
            tool_instance = self.tool_manager.get(tool_call.tool_name)
        except Exception as exc:
            error_msg = f"Error getting tool '{tool_call.tool_name}': {exc}"
            yield self._tool_failure_event(tool_call, error_msg, span=span)
            return

        try:
            tool_input = self._serialize_tool_input(tool_call)
        except Exception as exc:
            error_msg = (
                f"<{TOOL_ERROR_TAG}>Failed to serialize tool input for "
                f"'{tool_call.tool_name}': {exc}</{TOOL_ERROR_TAG}>"
            )
            self.stats.tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=error_msg,
                tool_call_id=tool_call.call_id,
            )
            self._handle_tool_response(tool_call, error_msg, "failure", span=span)
            return

        events, resolution = await self._run_before_tool_pipeline(
            tool_call, tool_input, span=span
        )
        for ev in events:
            yield ev
        if resolution.denial_event is not None:
            yield resolution.denial_event
            return
        tool_call = resolution.tool_call
        tool_input = resolution.tool_input

        decision: ToolDecision | None = None
        tool_started = False
        try:
            decision = await self._should_execute_tool(
                tool_instance, tool_call.validated_args, tool_call.call_id
            )

            # Apply a user MODIFY (re-validate edited args); a validation
            # failure comes back as a feedback SKIP decision handled below.
            tool_call, tool_input, decision = self._resolve_modification(
                tool_call, tool_input, decision
            )

            if decision.verdict == ToolExecutionResponse.SKIP:
                async for ev in self._handle_tool_skip(tool_call, decision, span=span):
                    yield ev
                return

            if tool_call.tool_name == "ask_user_question":
                # Fire a question notification before the tool blocks on input.
                # Fired BEFORE tool_started=True: a cancellation during the
                # notification must not finalize as a started-then-cancelled
                # tool (which would spuriously fire after-tool hooks).
                await self._fire_notification_hooks(
                    "question", "Waiting for user input", tool_call.tool_name
                )
            tool_started = True
            async for ev in self._invoke_tool(
                tool_call, tool_instance, tool_input, decision, span=span
            ):
                yield ev

        except asyncio.CancelledError:
            cancel = str(
                get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
            )
            self.stats.tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=cancel,
                cancelled=True,
                tool_call_id=tool_call.call_id,
            )
            async for ev in self._finalize_cancelled_tool(
                tool_call,
                tool_input,
                decision,
                cancel,
                span=span,
                tool_started=tool_started,
            ):
                yield ev
            raise

        except Exception as exc:
            error_msg = f"<{TOOL_ERROR_TAG}>{tool_instance.get_name()} failed: {exc}</{TOOL_ERROR_TAG}>"
            if isinstance(exc, ToolPermissionError):
                self.stats.tool_calls_agreed -= 1
                self.stats.tool_calls_rejected += 1
            else:
                self.stats.tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=error_msg,
                tool_call_id=tool_call.call_id,
            )
            async for ev in self._run_after_tool_and_finalize(
                tool_call,
                tool_input=tool_input,
                tool_status="failure",
                response_status="failure",
                decision=decision,
                span=span,
                tool_error=str(exc),
                initial_text=error_msg,
            ):
                yield ev

    async def _invoke_tool(
        self,
        tool_call: ResolvedToolCall,
        tool_instance: BaseTool,
        tool_input: dict[str, Any],
        decision: ToolDecision,
        *,
        span: trace.Span,
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        self.stats.tool_calls_agreed += 1

        # Snapshot read (rewind/undo) does blocking file I/O; run it off the
        # event loop so a write tool on a large file doesn't stall concurrent
        # readers in the same turn.
        snapshot = await asyncio.to_thread(
            tool_instance.get_file_snapshot, tool_call.validated_args
        )
        if snapshot is not None:
            self.rewind_manager.add_snapshot(snapshot)

        # Interactive tools (ask_user_question, in-tool approval) block on a
        # human inside invoke(). Time those awaits so they can be subtracted
        # from exec_duration, which is meant to be exec-only — otherwise a
        # multi-hour human answer is misread as tool runtime.
        human_wait_s = 0.0

        def _timed(cb: Any) -> Any:
            if cb is None:
                return None

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                nonlocal human_wait_s
                wait_start = time.perf_counter()
                try:
                    return await cb(*args, **kwargs)
                finally:
                    human_wait_s += time.perf_counter() - wait_start

            return _wrapped

        start_time = time.perf_counter()
        result_model = None
        duration = 0.0
        try:
            async for item in tool_instance.invoke(
                ctx=InvokeContext(
                    tool_call_id=tool_call.call_id,
                    scheduler=self.scheduler,
                    agent_manager=self.agent_manager,
                    active_model=self.effective_model().alias,
                    session_dir=self.session_logger.session_dir,
                    entrypoint_metadata=self.entrypoint_metadata,
                    approval_callback=_timed(self.approval_callback),
                    user_input_callback=_timed(self.user_input_callback),
                    sampling_callback=self._sampling_handler,
                    plan_file_path=self._plan_session.plan_file_path,
                    switch_agent_callback=self.switch_agent,
                    skill_manager=self.skill_manager,
                    scratchpad_dir=self.scratchpad_dir,
                    permission_store=self._permission_store,
                    hook_config_result=self._hook_config_result,
                    session_id=self.session_id,
                    terminal_emulator=self.terminal_emulator,
                    launch_workflow_callback=self.launch_workflow_callback,
                    workflow_status_callback=self.workflow_status_callback,
                    workflow_results_callback=self.workflow_results_callback,
                    workflow_stop_callback=self.workflow_stop_callback,
                    team_dir_callback=self.team_dir_callback,
                    background_registry=self.background_registry,
                    files_read=self._files_read,
                    tool_manager=self.tool_manager,
                ),
                **tool_call.args_dict,
            ):
                if isinstance(item, ToolStreamEvent):
                    yield item
                else:
                    result_model = item
        finally:
            # Stamp exec duration on EVERY exit — success, ToolError (nonzero
            # exit / size cap / not-found), and cancellation. invoke() raises
            # past this point on failure, which previously skipped the success-
            # only call below and left failure/timeout latency uninstrumented.
            # Subtract human-wait so exec_duration stays exec-only (recorded
            # separately as user_wait_s); duration flows to ToolResultEvent too.
            duration = max(0.0, time.perf_counter() - start_time - human_wait_s)
            set_tool_exec_duration(span, duration)
            if human_wait_s > 0:
                set_tool_user_wait(span, human_wait_s)
        if result_model is None:
            raise ToolError("Tool did not yield a result")

        result_dict = result_model.model_dump()
        text = "\n".join(f"{k}: {v}" for k, v in result_dict.items())
        extra = tool_instance.get_result_extra(result_model)
        if extra:
            text += "\n\n" + extra

        result_cancelled = (
            isinstance(result_model, CancellableToolResult) and result_model.cancelled
        )
        yield ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            result=result_model,
            cancelled=result_cancelled,
            duration=duration,
            tool_call_id=tool_call.call_id,
            approval_note=decision.feedback if decision.judge_approved else None,
        )
        async for ev in self._run_after_tool_and_finalize(
            tool_call,
            tool_input=tool_input,
            tool_status="cancelled" if result_cancelled else "success",
            response_status="success",
            decision=decision,
            span=span,
            tool_output=result_dict,
            duration_ms=duration * 1000.0,
            initial_text=text,
        ):
            yield ev
        self.stats.tool_calls_succeeded += 1

    def _apply_tool_result_budget(self, tool_call: ResolvedToolCall, text: str) -> str:
        return self.tool_result_store.shape(
            tool_call.call_id,
            text,
            preview_chars=TOOL_RESULT_PREVIEW_CHARS,
            hard_cap=self._tool_result_hard_cap(),
        )

    def _tool_result_hard_cap(self) -> int:
        try:
            threshold = self.effective_model().auto_compact_threshold
        except Exception:
            return MAX_TOOL_RESULT_CHARS
        return tool_result_hard_cap(threshold)

    def _handle_tool_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        status: Literal["success", "failure", "skipped"],
        decision: ToolDecision | None = None,
        result: dict[str, Any] | None = None,
        span: trace.Span | None = None,
    ) -> None:
        text = self._apply_tool_result_budget(tool_call, text)
        self.messages.append(
            LLMMessage.model_validate(
                self.format_handler.create_tool_response_message(tool_call, text)
            )
        )

        if span is not None:
            set_tool_result(span, text)
            if status == "failure":
                set_tool_error(span, text)
        self.telemetry_client.send_tool_call_finished(
            tool_call=tool_call,
            agent_profile_name=self.agent_profile.name,
            model=self.config.active_model,
            status=status,
            decision=decision,
            result=result,
            message_id=self._current_user_message_id,
        )

    def _tool_failure_event(
        self,
        tool_call: ResolvedToolCall,
        error_msg: str,
        decision: ToolDecision | None = None,
        cancelled: bool = False,
        span: trace.Span | None = None,
    ) -> ToolResultEvent:
        self._handle_tool_response(tool_call, error_msg, "failure", decision, span=span)
        return ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            error=error_msg,
            cancelled=cancelled,
            tool_call_id=tool_call.call_id,
        )
