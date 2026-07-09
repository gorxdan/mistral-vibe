"""Agent conversation loop.

Single-file module (upstream layout). Fork-only subsystem mixins live in
sibling modules (agent_loop_memory/_failover/_safety_judge/_hooks) and are
composed onto AgentLoop; ToolDecision/limits stay in agent_loop_models/_limits
to avoid an import cycle through the safety-judge base.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
import contextlib
import copy
import dataclasses
import functools
from functools import wraps
import hashlib
from http import HTTPStatus
import inspect
import os
from pathlib import Path
import shutil
import threading
from threading import Thread
import time
from typing import TYPE_CHECKING, Any, Literal, NoReturn
from uuid import uuid4

from opentelemetry import trace
import orjson
from pydantic import BaseModel

from vibe.core import loop_tracer, profiler, stream_tracer
from vibe.core.agent_loop_failover import AgentLoopFailoverMixin
from vibe.core.agent_loop_hooks import AgentLoopHooksMixin
from vibe.core.agent_loop_limits import (
    AGGREGATE_TOOL_RESULT_CHARS,
    MAX_CONCURRENT_SUBAGENTS,
    MAX_TOOL_RESULT_CHARS,
    TOOL_RESULT_PREVIEW_CHARS,
    tool_result_hard_cap,
)
from vibe.core.agent_loop_memory import AgentLoopMemoryMixin
from vibe.core.agent_loop_models import ToolDecision, ToolExecutionResponse
from vibe.core.agent_loop_safety_judge import AgentLoopSafetyJudgeMixin
from vibe.core.agents.manager import AgentManager
from vibe.core.agents.models import AgentProfile, BuiltinAgentName
from vibe.core.baseline_scaling import (
    BaselineTier,
    baseline_tier_for,
    trim_tool_descriptions,
)
from vibe.core.cache_store import InMemoryVibeCodeCacheStore, VibeCodeCacheStore
from vibe.core.compaction import (
    build_extractive_summary,
    build_summary_input,
    collect_leading_injected_context,
    collect_persisted_tool_outputs,
    collect_prior_user_messages,
    render_compaction_context,
    truncate_compaction_context_for_backend,
)
from vibe.core.config import ModelConfig, ProviderConfig, VibeConfig, resolve_api_key
from vibe.core.config.fingerprint import file_fingerprint
from vibe.core.experiments import ExperimentManager
from vibe.core.experiments.client import RemoteEvalClient
from vibe.core.experiments.session import (
    hydrate_experiments_from_session as session_hydrate_experiments_from_session,
    initialize_experiments as session_initialize_experiments,
)
from vibe.core.hooks.manager import HooksManager
from vibe.core.hooks.models import HookConfigResult, HookEvent
from vibe.core.llm.backend.factory import create_backend
from vibe.core.llm.exceptions import BackendError
from vibe.core.llm.format import APIToolFormatHandler
from vibe.core.llm.models import FailedToolCall, ResolvedMessage, ResolvedToolCall
from vibe.core.llm.types import BackendLike, CompletionRequest
from vibe.core.logger import logger
from vibe.core.lsp._integration import drain_diagnostics_into
from vibe.core.middleware import (
    CHAT_AGENT_EXIT,
    CHAT_AGENT_REMINDER,
    PLAN_AGENT_EXIT,
    AutoCompactMiddleware,
    ContextShaperMiddleware,
    ContextWarningMiddleware,
    ConversationContext,
    LoopDetectionMiddleware,
    MicrocompactMiddleware,
    MiddlewareAction,
    MiddlewarePipeline,
    MiddlewareResult,
    PriceLimitMiddleware,
    ReadOnlyAgentMiddleware,
    ResetReason,
    SnipMiddleware,
    TokenLimitMiddleware,
    ToolResultBudgetMiddleware,
    TurnLimitMiddleware,
    make_plan_agent_reminder,
)
from vibe.core.paths import safe_cwd
from vibe.core.plan_session import PlanSession
from vibe.core.prompts import UtilityPrompt
from vibe.core.resource_monitor import ResourceMonitor, resource_monitor_opt_in
from vibe.core.rewind import RewindManager
from vibe.core.scratchpad import init_scratchpad
from vibe.core.session.session_id import extract_suffix, generate_session_id
from vibe.core.session.session_logger import SessionLogger
from vibe.core.session.session_migration import migrate_sessions_entrypoint
from vibe.core.skills.manager import SkillManager
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.telemetry.build_metadata import (
    build_attachment_counts,
    build_request_metadata,
)
from vibe.core.telemetry.send import TelemetryClient
from vibe.core.telemetry.types import (
    LaunchContext,
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
    PermissionContext,
    PermissionStore,
    RequiredPermission,
)
from vibe.core.tools.tool_result_store import ToolResultStore
from vibe.core.tracing import (
    add_message_content_events,
    agent_span,
    chat_span,
    context_shaping_span,
    set_agent_usage,
    set_context_shaping_result,
    set_finish_reason,
    set_tool_error,
    set_tool_exec_duration,
    set_tool_result,
    set_tool_user_wait,
    set_usage,
    tool_span,
)
from vibe.core.trusted_folders import has_agents_md_file
from vibe.core.types import (
    AgentProfileChangedEvent,
    AgentStats,
    ApprovalCallback,
    ApprovalResponse,
    AssistantEvent,
    AvailableTool,
    BackgroundTaskCompletedEvent,
    BaseEvent,
    CompactEndEvent,
    CompactionOrigin,
    CompactStartEvent,
    ContentFilterError,
    ContextTooLongError,
    ImageAttachment,
    InjectedMessageKind,
    LLMChunk,
    LLMChunkAccumulator,
    LLMMessage,
    LLMUsage,
    MessageList,
    PlanReviewEndedEvent,
    PlanReviewRequestedEvent,
    RateLimitCallback,
    RateLimitError,
    ReasoningEvent,
    RefusalError,
    ResponseTooLongError,
    Role,
    ServerError,
    SessionTitleUpdatedEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    TransportError,
    UnclassifiedBackendError,
    UserInputCallback,
    UserMessageEvent,
)
from vibe.core.usage import (
    RateLimitStore,
    UsageRecord,
    compute_cost,
    get_usage_recorder,
    lookup_pricing,
    rate_limit_from_headers,
)
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
from vibe.core.utils.tokens import truncate_middle_to_tokens

if TYPE_CHECKING:
    from vibe.core.loop import Scheduler
    from vibe.core.memory.store import MemoryStore
    from vibe.core.teleport.teleport import TeleportService
    from vibe.core.teleport.types import TeleportPushResponseEvent, TeleportYieldEvent
    from vibe.core.tools.background import BackgroundRegistry
    from vibe.core.tools.connectors import ConnectorRegistry
    from vibe.core.tools.mcp import MCPRegistry
    from vibe.core.tools.safety_judge import JudgeVerdict


class AgentLoopError(Exception): ...


class AgentLoopStateError(AgentLoopError): ...


class AgentLoopLLMResponseError(AgentLoopError): ...


_STREAM_DEGENERATE_RETRIES = 2

_RATE_LIMIT_GRACE_RETRIES = 2
_RATE_LIMIT_GRACE_BASE_DELAY = 5.0


class InvalidStreamError(AgentLoopLLMResponseError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Invalid streamed response: {reason}")


class CompactionFailedError(AgentLoopError):
    def __init__(self, reason: str) -> None:
        self.reason = reason  # "tool_call" | "empty_summary"
        super().__init__(f"Compaction did not produce a summary (reason={reason}).")


class ImagesNotSupportedError(AgentLoopError): ...


class TeleportError(AgentLoopError): ...


def _refusal_error(provider: str, model: str, chunk: LLMChunk) -> RefusalError:
    stop = chunk.stop
    return RefusalError(
        provider,
        model,
        category=stop.category if stop else None,
        explanation=stop.explanation if stop else None,
    )


def _degenerate_response_reason(chunk: LLMChunk) -> str | None:
    msg = chunk.message
    has_content = bool((msg.content or "").strip())
    has_tool_calls = bool(msg.tool_calls)
    has_reasoning = bool((msg.reasoning_content or "").strip())
    if has_content or has_tool_calls or has_reasoning:
        return None
    if chunk.usage is not None:
        return None
    return "empty response (no content, tool calls, or reasoning) and no usage"


def _raise_for_backend_error(
    e: Exception, provider_name: str, model_name: str
) -> NoReturn:
    if isinstance(e, RefusalError | ResponseTooLongError):
        raise
    if _should_raise_rate_limit_error(e):
        raise RateLimitError(provider_name, model_name) from e
    if _is_context_too_long_error(e):
        raise ContextTooLongError(provider_name, model_name) from e
    if _is_response_too_long_error(e):
        raise ResponseTooLongError(provider_name, model_name) from e
    if _is_content_filter_error(e):
        raise ContentFilterError(provider_name, model_name) from e
    if _is_non_retryable_error(e):
        raise
    if _is_transport_error(e):
        raise TransportError(provider_name, model_name) from e
    if _is_server_error(e):
        raise ServerError(provider_name, model_name) from e
    # Typed residual so the loop can failover (bare RuntimeError aborted the turn).
    raise UnclassifiedBackendError(provider_name, model_name, str(e)) from e


def _should_raise_rate_limit_error(e: Exception) -> bool:
    return isinstance(e, BackendError) and e.status == HTTPStatus.TOO_MANY_REQUESTS


_MAX_SERVER_STATUS = 599


def _is_server_error(e: Exception) -> bool:
    backend = e if isinstance(e, BackendError) else getattr(e, "__cause__", None)
    return (
        isinstance(backend, BackendError)
        and backend.status is not None
        and HTTPStatus.INTERNAL_SERVER_ERROR <= backend.status <= _MAX_SERVER_STATUS
    )


def _is_transport_error(e: Exception) -> bool:
    # A dropped connection / transport failure reaches the loop as
    # BackendError(status=None): build_request_error (exceptions.py) sets no
    # status because there is no HTTP response to classify. status=None is the
    # unique signal — any HTTP status means a response arrived and is routed by
    # the server / rate-limit / context classifiers above. Mutually exclusive
    # with _is_server_error (which requires a 5xx status).
    backend = e if isinstance(e, BackendError) else getattr(e, "__cause__", None)
    return isinstance(backend, BackendError) and backend.status is None


def _is_context_too_long_error(e: Exception) -> bool:
    if isinstance(e, BackendError):
        return e.is_context_too_long
    if isinstance(e, RuntimeError) and isinstance(e.__cause__, BackendError):
        return e.__cause__.is_context_too_long
    return False


def _is_response_too_long_error(e: Exception) -> bool:
    if isinstance(e, BackendError):
        return e.is_response_too_long
    if isinstance(e, RuntimeError) and isinstance(e.__cause__, BackendError):
        return e.__cause__.is_response_too_long
    return False


def _is_content_filter_error(e: Exception) -> bool:
    if isinstance(e, BackendError):
        return e.is_content_filtered
    if isinstance(e, RuntimeError) and isinstance(e.__cause__, BackendError):
        return e.__cause__.is_content_filtered
    return False


def _is_non_retryable_error(e: BaseException) -> bool:
    # Detect Temporal-style ``non_retryable`` flag without importing temporalio.
    # Walks ``__cause__`` so an ``ActivityError`` whose cause is a non-retryable
    # ``ApplicationError`` is detected too — that's what callers driving the
    # agent loop from a Temporal activity will see when a sub-activity has
    # already failed terminally.
    seen: set[int] = set()
    current: BaseException | None = e
    while current is not None and id(current) not in seen:
        if getattr(current, "non_retryable", False):
            return True
        seen.add(id(current))
        current = current.__cause__
    return False


def requires_init(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.isasyncgenfunction(fn):

        @wraps(fn)
        async def gen_wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
            await self.wait_until_ready()
            agen = fn(self, *args, **kwargs)
            sent: Any = None
            try:
                while True:
                    sent = yield await agen.asend(sent)
            except StopAsyncIteration:
                return
            finally:
                await agen.aclose()

        return gen_wrapper

    if not inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"@requires_init requires an async method; {fn.__qualname__} is sync "
            "and cannot await wait_until_ready()"
        )

    @wraps(fn)
    async def wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
        await self.wait_until_ready()
        return await fn(self, *args, **kwargs)

    return wrapper


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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AgentLoopParams:
    agent_name: str = BuiltinAgentName.DEFAULT
    message_observer: Callable[[LLMMessage], None] | None = None
    max_turns: int | None = None
    max_price: float | None = None
    max_session_tokens: int | None = None
    enable_streaming: bool = False
    launch_context: LaunchContext | None = None
    terminal_emulator: TerminalEmulator | None = None
    is_subagent: bool = False
    defer_heavy_init: bool = False
    headless: bool = False
    hook_config_result: HookConfigResult | None = None
    permission_store: PermissionStore | None = None
    mcp_registry: MCPRegistry | None = None
    cache_store: VibeCodeCacheStore | None = None


class AgentLoop(
    AgentLoopMemoryMixin,
    AgentLoopFailoverMixin,
    AgentLoopSafetyJudgeMixin,
    AgentLoopHooksMixin,
):
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
            p.is_subagent, p.launch_context, p.terminal_emulator, config
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

    def set_user_input_callback(self, callback: UserInputCallback) -> None:
        self.user_input_callback = callback

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
            launch_context=self.launch_context,
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
        has_agents_md = has_agents_md_file(safe_cwd())
        nb_skills = len(self.skill_manager.available_skills)
        nb_mcp_servers = len(self.config.mcp_servers)
        nb_models = len(self.config.models)

        self.telemetry_client.send_new_session(
            has_agents_md=has_agents_md,
            nb_skills=nb_skills,
            nb_mcp_servers=nb_mcp_servers,
            nb_models=nb_models,
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
                    async with contextlib.AsyncExitStack() as turn_stack:
                        turn_stack.enter_context(
                            profiler.section(
                                f"turn-{self.session_id[:8]}-{self.stats.steps}",
                                turn=self.stats.steps,
                            )
                        )
                        await turn_stack.enter_async_context(
                            self.resource_monitor.turn_async()
                        )
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

    def _get_context(self) -> ConversationContext:
        return ConversationContext(
            messages=self.messages,
            stats=self.stats,
            config=self.config,
            active_model=self.effective_model(),
        )

    def _build_backend_metadata(
        self, call_type: TelemetryCallType | None = None
    ) -> TelemetryRequestMetadata:
        return build_request_metadata(
            launch_context=self.launch_context,
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
            rate_limit_retries = 0
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
                    if rate_limit_retries < _RATE_LIMIT_GRACE_RETRIES:
                        rate_limit_retries += 1
                        delay = _RATE_LIMIT_GRACE_BASE_DELAY * rate_limit_retries
                        logger.warning(
                            "Rate-limited on %s; backing off for %.1fs before "
                            "retrying (grace %d/%d)",
                            e.model,
                            delay,
                            rate_limit_retries,
                            _RATE_LIMIT_GRACE_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
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
                except UnclassifiedBackendError as e:
                    # Residual API error: failover when pool configured (else re-raise).
                    self._apply_failover(
                        e,
                        self._switch_to_fallback_model(),
                        error_type="unclassified_backend",
                        unavailable_reason=(
                            f"{e.provider!r} backend returned an unclassified error"
                        ),
                        log_template=(
                            "%r backend unclassified error; falling back to %r"
                        ),
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
                    launch_context=self.launch_context,
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
                    team_spawn_callback=self.team_spawn_callback,
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

    async def _should_execute_tool(
        self, tool: BaseTool, args: BaseModel, tool_call_id: str
    ) -> ToolDecision:
        if self.bypass_tool_permissions:
            return ToolDecision(
                verdict=ToolExecutionResponse.EXECUTE,
                approval_type=ToolPermission.ALWAYS,
            )

        async with self._permission_store.lock:
            tool_name = tool.get_name()
            ctx = tool.resolve_permission(args)

            if ctx is None:
                config_perm = self.tool_manager.get_tool_config(tool_name).permission
                ctx = PermissionContext(permission=config_perm)

            if ctx.permission == ToolPermission.ALWAYS:
                return ToolDecision(
                    verdict=ToolExecutionResponse.EXECUTE,
                    approval_type=ToolPermission.ALWAYS,
                )
            if ctx.permission == ToolPermission.NEVER:
                return ToolDecision(
                    verdict=ToolExecutionResponse.SKIP,
                    approval_type=ToolPermission.NEVER,
                    feedback=ctx.reason
                    or f"Tool '{tool_name}' is permanently disabled",
                )
            uncovered = [
                rp
                for rp in ctx.required_permissions
                if not self._permission_store.covers(tool_name, rp)
            ]
            if ctx.required_permissions and not uncovered:
                return ToolDecision(
                    verdict=ToolExecutionResponse.EXECUTE,
                    approval_type=ToolPermission.ALWAYS,
                )

        # Lock released: the safety-judge LLM call and human approval are slow;
        # holding the permission lock across them would serialize every parallel
        # ASK-gated tool. The rule-store reads above happened under the lock.
        judged, judge_deferral = await self._judge_tool_safety(
            tool_name, args, uncovered
        )
        if judged is not None:
            return judged
        return await self._ask_approval(
            tool_name, args, tool_call_id, uncovered, judge_deferral=judge_deferral
        )

    async def _ask_approval(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission],
        judge_deferral: str | None = None,
    ) -> ToolDecision:
        if not self.approval_callback:
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                approval_type=ToolPermission.ASK,
                feedback="Tool execution not permitted.",
            )
        await self._fire_notification_hooks(
            "permission_required", f"Approval needed for {tool_name}", tool_name
        )
        response, feedback, modified_args = await self.approval_callback(
            tool_name,
            args,
            tool_call_id,
            required_permissions,
            # Carry the judge's deferral reason (set in _judge_tool_safety) so
            # the host prompt can show WHY approval is needed even when the
            # call originated from a workflow/task subagent — the subagent's
            # loop-local pending_judge_deferral is invisible to the host, so the
            # note must travel with the callback itself.
            judge_deferral,
        )

        match response:
            case ApprovalResponse.YES:
                verdict = ToolExecutionResponse.EXECUTE
            case ApprovalResponse.MODIFY:
                verdict = ToolExecutionResponse.EXECUTE
            case _:
                verdict = ToolExecutionResponse.SKIP

        return ToolDecision(
            verdict=verdict,
            approval_type=ToolPermission.ASK,
            feedback=feedback,
            modified_args=modified_args
            if response == ApprovalResponse.MODIFY
            else None,
        )

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

    def _messages_for_backend(self, active_model: ModelConfig) -> Sequence[LLMMessage]:
        msgs = self._cap_injected_messages_for_backend(self._with_late_memory())
        if active_model.supports_images:
            return msgs
        if not any(m.images for m in msgs):
            return msgs
        return [m.model_copy(update={"images": None}) if m.images else m for m in msgs]

    def count_history_images_unsupported_by_active_model(self) -> int:
        try:
            active_model = self.config.get_active_model()
        except ValueError:
            return 0
        if active_model.supports_images:
            return 0
        return sum(1 for m in self.messages if m.images)

    async def _chat(
        self,
        max_tokens: int | None = None,
        model_override: ModelConfig | None = None,
        *,
        harness: bool = False,
    ) -> LLMChunk:
        # Apply the output-escalation override only to main-turn calls: callers
        # that set model_override (e.g. compaction summary) must not inherit it.
        if max_tokens is None and model_override is None:
            max_tokens = self._max_output_override
        active_model, provider = self._resolve_active_model(model_override)
        # self.backend always serves effective_model()'s provider (init, failover,
        # and reload keep them in lockstep). A model_override (e.g. compaction)
        # may target a different provider than the current failover backend —
        # reuse self.backend only when providers match, otherwise build a one-off
        # backend so the model name + temperature reach the right endpoint
        # (gpt-5.5 reaching a kimi backend -> "invalid temperature").
        backend = self.backend
        if (
            model_override is not None
            and provider.name
            != self.config.get_provider_for_model(self.effective_model()).name
        ):
            backend = create_backend(provider=provider, timeout=self.config.api_timeout)
        backend_metadata = self._build_backend_metadata()

        available_tools = self._available_tools(active_model)
        tool_choice = self.format_handler.get_tool_choice()

        last_user_message = self._last_user_message()
        self.telemetry_client.send_request_sent(
            model=active_model.alias,
            nb_context_chars=sum(len(m.content or "") for m in self.messages),
            nb_context_messages=len(self.messages),
            nb_prompt_chars=len(last_user_message.content or "")
            if last_user_message
            else 0,
            call_type=backend_metadata.call_type,
            message_id=backend_metadata.message_id,
            attachment_counts=build_attachment_counts(
                last_user_message, supports_images=active_model.supports_images
            ),
        )

        try:
            async with chat_span(
                model=active_model.name,
                provider=provider.name,
                temperature=self._wire_temperature(active_model, provider),
                max_tokens=max_tokens,
                thinking=active_model.thinking,
            ) as _span:
                start_time = time.perf_counter()
                extra_headers, turn_state_sink = self._codex_routing(provider)
                result = await backend.complete(
                    CompletionRequest(
                        model=active_model,
                        messages=self._messages_for_backend(active_model),
                        temperature=active_model.temperature,
                        tools=available_tools,
                        tool_choice=tool_choice,
                        extra_headers=extra_headers,
                        max_tokens=max_tokens,
                        metadata=backend_metadata.model_dump(exclude_none=True),
                        response_format=self._response_format,
                    ),
                    response_headers_sink=turn_state_sink,
                )
                end_time = time.perf_counter()
                self._capture_codex_turn_state(turn_state_sink)
                self._capture_rate_limits(provider, turn_state_sink)

                if result.usage is None:
                    raise AgentLoopLLMResponseError(
                        "Usage data missing in non-streaming completion response"
                    )
                self._update_stats(
                    usage=result.usage,
                    time_seconds=end_time - start_time,
                    provider=provider,
                    model=active_model,
                    harness=harness,
                )
                set_usage(_span, result.usage)
                set_finish_reason(_span, result.stop.reason if result.stop else None)
                self._capture_chat_content(_span, result.message, last_user_message)

            if result.correlation_id:
                self.telemetry_client.last_correlation_id = result.correlation_id

            processed_message = self.format_handler.process_api_response_message(
                result.message
            )
            # Raise before committing the truncated turn to history so the
            # escalation retry (larger max_tokens) starts from a clean message list.
            if result.stop and result.stop.is_truncated:
                raise ResponseTooLongError(provider.name, active_model.name)
            self.messages.append(processed_message)
            if result.stop and result.stop.is_refusal:
                raise _refusal_error(provider.name, active_model.name, result)
            return LLMChunk(
                message=processed_message, usage=result.usage, stop=result.stop
            )

        except Exception as e:
            _raise_for_backend_error(e, provider.name, active_model.name)

    async def _chat_streaming(
        self, max_tokens: int | None = None
    ) -> AsyncGenerator[LLMChunk]:
        if max_tokens is None:
            max_tokens = self._max_output_override
        active_model, provider = self._resolve_active_model()
        backend_metadata = self._build_backend_metadata()

        available_tools = self._available_tools(active_model)
        tool_choice = self.format_handler.get_tool_choice()

        last_user_message = self._last_user_message()
        self.telemetry_client.send_request_sent(
            model=active_model.alias,
            nb_context_chars=sum(len(m.content or "") for m in self.messages),
            nb_context_messages=len(self.messages),
            nb_prompt_chars=len(last_user_message.content or "")
            if last_user_message
            else 0,
            call_type=backend_metadata.call_type,
            message_id=backend_metadata.message_id,
            attachment_counts=build_attachment_counts(
                last_user_message, supports_images=active_model.supports_images
            ),
        )

        for attempt in range(_STREAM_DEGENERATE_RETRIES):
            try:
                async with chat_span(
                    model=active_model.name,
                    provider=provider.name,
                    temperature=self._wire_temperature(active_model, provider),
                    max_tokens=max_tokens,
                    thinking=active_model.thinking,
                ) as _span:
                    start_time = time.perf_counter()
                    # Accumulate streamed deltas in O(n) instead of folding with
                    # LLMChunk.__add__ per chunk (which re-concatenates the whole
                    # message every delta -> O(n^2) over a response).
                    chunk_acc = LLMChunkAccumulator()
                    extra_headers, turn_state_sink = self._codex_routing(provider)
                    stream_tracer.stream_started(self)
                    async for chunk in self.backend.complete_streaming(
                        CompletionRequest(
                            model=active_model,
                            messages=self._messages_for_backend(active_model),
                            temperature=active_model.temperature,
                            tools=available_tools,
                            tool_choice=tool_choice,
                            extra_headers=extra_headers,
                            max_tokens=max_tokens,
                            metadata=backend_metadata.model_dump(exclude_none=True),
                            response_format=self._response_format,
                        ),
                        response_headers_sink=turn_state_sink,
                    ):
                        stream_tracer.chunk_received(self)
                        if chunk.correlation_id:
                            self.telemetry_client.last_correlation_id = (
                                chunk.correlation_id
                            )
                        processed_chunk = LLMChunk(
                            message=self.format_handler.process_api_response_message(
                                chunk.message
                            ),
                            usage=chunk.usage,
                            stop=chunk.stop,
                        )
                        chunk_acc.add(processed_chunk)
                        yield processed_chunk
                    end_time = time.perf_counter()
                    self._capture_codex_turn_state(turn_state_sink)
                    self._capture_rate_limits(provider, turn_state_sink)

                    chunk_agg = chunk_acc.build()
                    if chunk_agg is None or chunk_agg.usage is None:
                        raise AgentLoopLLMResponseError(
                            "Usage data missing in final chunk of streamed completion"
                        )
                    # Reject a degenerate no-op response (no content, tool calls,
                    # or reasoning) so it is re-requested below rather than
                    # silently ending the turn producing nothing. A degenerate
                    # response yields inert empty chunks upstream, so the retry
                    # with a fresh accumulator is clean.
                    degenerate_reason = _degenerate_response_reason(chunk_agg)
                    if degenerate_reason is not None:
                        raise InvalidStreamError(degenerate_reason)
                    self._update_stats(
                        usage=chunk_acc.usage,
                        time_seconds=end_time - start_time,
                        provider=provider,
                        model=active_model,
                    )
                    set_usage(_span, chunk_acc.usage)
                    set_finish_reason(
                        _span, chunk_agg.stop.reason if chunk_agg.stop else None
                    )
                    self._capture_chat_content(
                        _span, chunk_agg.message, last_user_message
                    )

                # Raise before committing the truncated turn so the escalation
                # retry re-streams from a clean message list (mirrors _chat).
                if chunk_agg.stop and chunk_agg.stop.is_truncated:
                    raise ResponseTooLongError(provider.name, active_model.name)
                self.messages.append(chunk_agg.message)
                if chunk_agg.stop and chunk_agg.stop.is_refusal:
                    raise _refusal_error(provider.name, active_model.name, chunk_agg)
                return

            except InvalidStreamError as e:
                if attempt < _STREAM_DEGENERATE_RETRIES - 1:
                    logger.warning(
                        "Degenerate streamed response (%s); re-requesting stream "
                        "attempt %d/%d",
                        e.reason,
                        attempt + 1,
                        _STREAM_DEGENERATE_RETRIES,
                    )
                    continue
                raise
            except Exception as e:
                _raise_for_backend_error(e, provider.name, active_model.name)

    def _update_stats(
        self,
        usage: LLMUsage,
        time_seconds: float,
        *,
        provider: ProviderConfig,
        model: ModelConfig,
        harness: bool = False,
    ) -> None:
        self.stats.last_turn_duration = time_seconds
        self.stats.last_turn_prompt_tokens = usage.prompt_tokens
        self.stats.last_turn_completion_tokens = usage.completion_tokens
        self.stats.session_prompt_tokens += usage.prompt_tokens
        self.stats.session_completion_tokens += usage.completion_tokens
        self.stats.last_turn_cached_tokens = usage.cached_tokens
        self.stats.session_cached_tokens += usage.cached_tokens
        self.stats.context_tokens = usage.prompt_tokens + usage.completion_tokens
        if time_seconds > 0 and usage.completion_tokens > 0:
            self.stats.tokens_per_second = usage.completion_tokens / time_seconds

        # Persist the call for cross-session usage windows (/status). Best-effort:
        # a recorder failure never affects the turn. Cost precedence: a user's
        # explicit per-model config prices win; otherwise the built-in pricing
        # table supplies verified rates; both absent → cost_usd=0 (card shows —).
        if model.input_price > 0 or model.output_price > 0:
            cost = (
                usage.prompt_tokens * model.input_price
                + usage.completion_tokens * model.output_price
            ) / 1_000_000
        else:
            pricing = lookup_pricing(model.name)
            if pricing is not None:
                cost = compute_cost(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cached_tokens=usage.cached_tokens,
                    pricing=pricing,
                )
            else:
                cost = 0.0
        # Accrue the exact per-call cost so session_cost survives a mid-session
        # model switch without repricing the whole session against the new model.
        self.stats.accumulated_cost_usd += cost
        self._usage_recorder.record(
            UsageRecord.from_usage(
                timestamp=time.time(),
                provider=provider.name,
                model=model.name,
                usage=usage,
                cost_usd=cost,
                duration_s=time_seconds,
                session_id=self.session_id,
                harness=harness,
            )
        )

    def _clean_message_history(self) -> None:
        ACCEPTABLE_HISTORY_SIZE = 2
        if len(self.messages) < ACCEPTABLE_HISTORY_SIZE:
            return
        self._fill_missing_tool_responses()

    def _fill_missing_tool_responses(self) -> None:
        i = 1
        while i < len(self.messages):
            msg = self.messages[i]

            if not (msg.role == "assistant" and msg.tool_calls):
                i += 1
                continue

            expected = len(msg.tool_calls)
            if expected == 0:
                i += 1
                continue

            responded, end = self._collect_responded_ids(self.messages, i + 1)
            next_i = end

            if len(responded) >= expected:
                i = next_i
                continue

            insertion_point = next_i
            for tc in msg.tool_calls:
                if (tc.id or "") in responded:
                    continue
                self.messages.insert(
                    insertion_point,
                    LLMMessage(
                        role=Role.TOOL,
                        tool_call_id=tc.id or "",
                        name=(tc.function.name or "") if tc.function else "",
                        content=str(
                            get_user_cancellation_message(
                                CancellationReason.TOOL_NO_RESPONSE
                            )
                        ),
                    ),
                )
                insertion_point += 1

            i = next_i

    async def _reset_session(
        self, keep_parent: bool = True, *, lifecycle_reason: str | None = None
    ) -> None:
        # lifecycle_reason is set only for real session boundaries (e.g. /clear),
        # NOT compaction's internal reset — so SessionEnd/Start don't double-fire
        # during a compaction (PreCompact already covers that).
        if lifecycle_reason is not None:
            await self._fire_session_end_hooks(lifecycle_reason)
            self._session_started = False  # next act() re-fires SessionStart
        old_session_id = self.session_id
        self.emit_session_closed_telemetry()
        suffix = extract_suffix(self.session_id)
        self.session_id = generate_session_id(suffix=suffix)
        parent_session_id = old_session_id if keep_parent else None
        self.parent_session_id = parent_session_id
        self.session_logger.reset_session(
            self.session_id, parent_session_id=parent_session_id
        )
        self._files_read.clear()
        self._files_read_reconstructed = False
        self._agents_md_fingerprint = None
        await self.initialize_experiments()
        self.emit_new_session_telemetry()

    async def fork(self, message_id: str | None = None) -> AgentLoop:
        messages = self._messages_for_fork(message_id)
        forked = AgentLoop(
            self.base_config.model_copy(deep=True),
            params=AgentLoopParams(
                agent_name=self.agent_profile.name,
                max_turns=self._max_turns,
                max_price=self._max_price,
                max_session_tokens=self._max_session_tokens,
                enable_streaming=self.enable_streaming,
                launch_context=self.launch_context,
                terminal_emulator=self.terminal_emulator,
                defer_heavy_init=True,
                hook_config_result=self._hook_config_result,
            ),
        )
        forked._max_output_override = self._max_output_override
        forked.session_id = generate_session_id(suffix=extract_suffix(self.session_id))
        forked.parent_session_id = self.session_id
        # A forked session gets its OWN fresh background registry — not the
        # parent's reference. _close_agent_loop calls registry.shutdown() on
        # every session close, so sharing the parent's reference would reap
        # the parent's running processes when the fork closes. A fresh instance
        # lets the fork use bash background=True (the original gap) while
        # keeping ownership boundaries clean: closing the fork reaps only the
        # fork's own processes, and the parent's registry is untouched.
        # Local import: BackgroundRegistry is TYPE_CHECKING-only at module
        # scope (used in annotations), but fork() needs it at runtime.
        from vibe.core.tools.background import BackgroundRegistry

        forked.background_registry = BackgroundRegistry()
        forked.session_logger.reset_session(
            forked.session_id, parent_session_id=self.session_id
        )
        forked.messages.extend(messages)
        await forked.session_logger.save_interaction(
            forked.messages,
            forked.stats,
            forked.base_config,
            forked.tool_manager,
            forked.agent_profile,
        )
        return forked

    def _messages_for_fork(self, message_id: str | None) -> list[LLMMessage]:
        source_messages = [m for m in self.messages if m.role != Role.SYSTEM]
        if message_id is None:
            return [m.model_copy(deep=True) for m in source_messages]

        anchor_index = next(
            (i for i, m in enumerate(source_messages) if message_id == m.message_id),
            None,
        )
        if anchor_index is None:
            raise ValueError(f"Cannot fork from unknown message_id: {message_id}")

        if source_messages[anchor_index].role != Role.USER:
            raise ValueError("Fork from message_id is only supported for user messages")

        next_turn_index = next(
            (
                i
                for i, m in enumerate(
                    source_messages[anchor_index + 1 :], start=anchor_index + 1
                )
                if m.role == Role.USER
            ),
            len(source_messages),
        )
        return [m.model_copy(deep=True) for m in source_messages[:next_turn_index]]

    @requires_init
    async def clear_history(self) -> None:
        await self.session_logger.save_interaction(
            self.messages,
            self.stats,
            self._base_config,
            self.tool_manager,
            self.agent_profile,
        )
        self.messages.reset(self.messages[:1])

        self.stats = AgentStats.create_fresh(self.stats)
        self.stats.trigger_listeners()

        try:
            # /clear keeps a live failover override -> price the effective model.
            active_model = self.effective_model()
            self.stats.update_pricing(
                active_model.input_price, active_model.output_price
            )
            self.stats.update_model_bounds(active_model.auto_compact_threshold)
        except ValueError:
            pass

        self.middleware_pipeline.reset()
        self.tool_manager.reset_all()
        await self._reset_session(keep_parent=False, lifecycle_reason="clear")

    @requires_init
    async def compact(self, extra_instructions: str = "") -> str:
        try:
            self._clean_message_history()
            await self.session_logger.save_interaction(
                self.messages,
                self.stats,
                self._base_config,
                self.tool_manager,
                self.agent_profile,
            )

            summary_prefix = UtilityPrompt.COMPACT_SUMMARY_PREFIX.read()
            history_snapshot = list(self.messages)
            prior_user_messages = collect_prior_user_messages(
                history_snapshot, summary_prefix
            )

            summary_request = self.config.compaction_prompt
            if extra_instructions:
                summary_request += (
                    f"\n\n## Additional Instructions\n{extra_instructions}"
                )
            self.stats.steps += 1

            compaction_model = self.config.get_compaction_model()
            # Bound the summarizer input: feeding the full, already over-window
            # history to the summary call overflows the summarizer itself (the very
            # case compaction handles). Run the call against the bounded copy so it
            # never overflows; the real history is restored before reset() below.
            summary_messages = MessageList(
                build_summary_input(
                    history_snapshot,
                    summary_request,
                    compaction_model.auto_compact_threshold,
                )
            )
            summary_content = ""
            try:
                original_messages = self.messages
                self.messages = summary_messages
                try:
                    summary_result = await self._chat(
                        model_override=compaction_model, harness=True
                    )
                finally:
                    self.messages = original_messages

                if summary_result.usage is None:
                    raise AgentLoopLLMResponseError(
                        "Usage data missing in compaction summary response"
                    )
                summary_content = (summary_result.message.content or "").strip()
                has_tool_calls = bool(summary_result.message.tool_calls)
                if has_tool_calls or not summary_content:
                    reason = "tool_call" if has_tool_calls else "empty_summary"
                    self.telemetry_client.send_compaction_failed(
                        reason=reason,
                        session_id=self.session_id,
                        parent_session_id=self.parent_session_id,
                    )
                    if self.config.raise_on_compaction_failure:
                        raise CompactionFailedError(reason)
                    summary_content = ""
            except Exception:
                if self.config.raise_on_compaction_failure:
                    raise
                logger.warning(
                    "compaction summary call failed; using extractive fallback"
                )
                summary_content = ""

            if not summary_content:
                summary_content = build_extractive_summary(history_snapshot)

            system_message = self.messages[0]
            # Preserve the leading injected environment context (file-tree,
            # AGENTS.md, deep-memory) across compaction. collect_prior_user_messages
            # skips every injected message, so without this the model's grounding
            # vanishes after every reset.
            leading_context = collect_leading_injected_context(history_snapshot)
            persisted_tool_outputs = collect_persisted_tool_outputs(history_snapshot)
            compaction_context = render_compaction_context(
                prior_user_messages, summary_content, persisted_tool_outputs
            )
            compaction_context_message = LLMMessage(
                role=Role.USER,
                content=compaction_context,
                injected=True,
                injected_kind=InjectedMessageKind.COMPACTION_CONTEXT,
            )
            self.messages.reset([
                system_message,
                *leading_context,
                compaction_context_message,
            ])

            await self._reset_session()

            # Context size is unknown without an API call; reset to 0. The next
            # LLM turn recomputes it accurately from real usage (_update_stats).
            self.stats.context_tokens = 0
            await self.session_logger.save_interaction(
                self.messages,
                self.stats,
                self._base_config,
                self.tool_manager,
                self.agent_profile,
            )

            self.middleware_pipeline.reset(reset_reason=ResetReason.COMPACT)

            return summary_content

        except Exception:
            await self.session_logger.save_interaction(
                self.messages,
                self.stats,
                self._base_config,
                self.tool_manager,
                self.agent_profile,
            )
            raise

    @requires_init
    async def switch_agent(self, agent_name: str) -> None:
        if agent_name == self.agent_profile.name:
            return
        self.agent_manager.switch_profile(agent_name)
        await self.reload_with_initial_messages(reset_middleware=False)

    @requires_init
    async def reload_with_initial_messages(
        self,
        base_config: VibeConfig | None = None,
        max_turns: int | None = None,
        max_price: float | None = None,
        reset_middleware: bool = True,
    ) -> None:
        # Force an immediate yield to allow the UI to update before heavy sync work.
        # When there are no messages, save_interaction returns early without any await,
        # so the coroutine would run synchronously through ToolManager, SkillManager,
        # and system prompt generation without yielding control to the event loop.
        await asyncio.sleep(0)

        await self.session_logger.save_interaction(
            self.messages,
            self.stats,
            self._base_config,
            self.tool_manager,
            self.agent_profile,
        )

        model_changed = False
        if base_config is not None:
            model_changed = base_config.active_model != self._base_config.active_model
            self._base_config = base_config
            self.agent_manager.invalidate_config()

        # Drop the rate-limit/fallback override ONLY when the reload makes a
        # different configured model authoritative — otherwise the stale override
        # would force the old model onto the freshly-rebuilt backend (e.g. a
        # switched-away glm-5.2 reaching a kimi backend -> "invalid temperature").
        # A reload that doesn't change the active model (config edit, agent
        # switch) must PRESERVE the user's explicit rate-limit switch, or the loop
        # reverts to the rate-limited model and re-prompts every turn.
        if model_changed:
            self._fallback_model_override = None
            self._tried_fallback_aliases.clear()

        # An agent switch rebinds self.config to the new agent's overrides, which
        # can drop the surviving override's provider (e.g. a rate-limit switch to
        # gpt-5.5/openai-chatgpt, then cycling to an agent whose providers omit
        # openai-chatgpt). get_provider_for_model would raise and crash the reload
        # worker; drop the orphaned override and fall back to the configured
        # backend instead. Same reset semantics as the model_changed branch above.
        if self._fallback_model_override is not None and (
            not self.config.is_model_available(self._fallback_model_override)
        ):
            self._fallback_model_override = None
            self._tried_fallback_aliases.clear()

        # Build the backend for whatever model is now effective: a surviving
        # override keeps its own provider, so build for it rather than letting the
        # config-default backend mismatch the overridden model.
        if self._fallback_model_override is not None:
            self.backend = create_backend(
                provider=self.config.get_provider_for_model(
                    self._fallback_model_override
                ),
                timeout=self.config.api_timeout,
            )
        else:
            self.backend = self.backend_factory()

        if max_turns is not None:
            self._max_turns = max_turns
        if max_price is not None:
            self._max_price = max_price

        self._ensure_remote_registries()
        self.tool_manager = ToolManager(
            lambda: self.config,
            mcp_registry=self.mcp_registry,
            connector_registry=self.connector_registry,
            permission_getter=self._permission_store.get_tool_permission,
        )
        self.skill_manager = SkillManager(lambda: self.config)

        self._system_prompt_tier = self._current_baseline_tier()
        new_system_prompt = get_universal_system_prompt(
            self.tool_manager,
            self.config,
            self.skill_manager,
            self.agent_manager,
            scratchpad_dir=self.scratchpad_dir,
            headless=self._headless,
            experiment_manager=self.experiment_manager,
            tier=self._system_prompt_tier,
        )

        self.messages.update_system_prompt(new_system_prompt)

        if len(self.messages) == 1:
            self.stats.reset_context_state()

        try:
            # Effective-model rule: a surviving failover override, not the
            # configured primary, is what requests route to.
            active_model = self.effective_model()
            self.stats.update_pricing(
                active_model.input_price, active_model.output_price
            )
            self.stats.update_model_bounds(active_model.auto_compact_threshold)
        except ValueError:
            pass

        if reset_middleware:
            self._setup_middleware()

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

        # No servers -> no registry: MCPRegistry pulls the mcp SDK (~75ms),
        # which must stay off cold start; reload -> _ensure_remote_registries.
        self.mcp_registry: MCPRegistry | None = None
        if not defer_heavy_init:
            if mcp_registry is not None:
                self.mcp_registry = mcp_registry
            elif self._base_config.mcp_servers:
                self.mcp_registry = self._create_mcp_registry()
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
        launch_context: LaunchContext | None,
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
        self.launch_context = (
            launch_context.model_copy(
                update={
                    "terminal_emulator": terminal_emulator
                    or launch_context.terminal_emulator
                }
            )
            if launch_context is not None
            else None
        )
        self.terminal_emulator = terminal_emulator or (
            launch_context.terminal_emulator if launch_context else None
        )

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
        self.team_spawn_callback: (
            Callable[[str, str, str, int, bool], Awaitable[dict[str, Any]]] | None
        ) = None
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
            launch_context=self.launch_context,
            experiments_getter=lambda: self.experiment_manager.assignments(),
        )
        self.session_logger = SessionLogger(config.session_logging, self.session_id)
        self.resource_monitor = ResourceMonitor(
            enabled=not is_subagent and resource_monitor_opt_in(),
            label_getter=lambda: self.session_id,
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

    def set_scheduler(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler

    def set_rate_limit_callback(self, callback: RateLimitCallback) -> None:
        self.rate_limit_callback = callback

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

    async def _try_reactive_shaping(self) -> bool:
        threshold = self.effective_model().auto_compact_threshold
        if threshold <= 0:
            return False
        ctx = self._get_context()
        # emergency: hard-overflow shedding must never be floor/cooldown-blocked.
        snip = SnipMiddleware()
        microcompact = MicrocompactMiddleware(emergency=True)
        start = snip.estimated_tokens(ctx)
        for _ in range(12):  # bounded; a batch pass converges in ~1 iteration
            before = snip.estimated_tokens(ctx)
            await snip.before_turn(ctx)
            await microcompact.before_turn(ctx)
            after = snip.estimated_tokens(ctx)
            if after >= before:
                break
        if snip.estimated_tokens(ctx) >= start:
            return False
        # The persistent shapers' cooldown was armed near overflow-level est;
        # left in place it suppresses proactive passes for the whole regrowth.
        for mw in self.middleware_pipeline.middlewares:
            if isinstance(mw, ContextShaperMiddleware):
                mw.reset()
        return True

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

    def _cap_injected_messages_for_backend(
        self, messages: Sequence[LLMMessage]
    ) -> Sequence[LLMMessage]:
        max_tokens = self.config.context_shaping.max_injected_message_tokens
        if max_tokens <= 0:
            return messages
        capped: list[LLMMessage] | None = None
        for idx, message in enumerate(messages):
            if not message.injected or not isinstance(message.content, str):
                continue
            if message.injected_kind == InjectedMessageKind.COMPACTION_CONTEXT:
                content = truncate_compaction_context_for_backend(
                    message.content, max_tokens
                )
            else:
                content = truncate_middle_to_tokens(message.content, max_tokens)
            if content == message.content:
                continue
            if capped is None:
                capped = list(messages)
            capped[idx] = message.model_copy(update={"content": content})
        return messages if capped is None else capped

    def _with_late_memory(self) -> Sequence[LLMMessage]:
        section = self._late_memory_section
        if self.config.memory.inject_mode != "late" or not section:
            return self.messages
        mem_msg = LLMMessage(
            role=Role.USER,
            content=self._wrap_memories(section),
            injected=True,
            injected_kind=InjectedMessageKind.MEMORY,
        )
        msgs = list(self.messages)
        if self.config.memory.late_anchor == "tail":
            # Absolute tail: adapters rely on trailing MEMORY message(s) being the
            # request's suffix when placing the history cache breakpoint.
            msgs.append(mem_msg)
            return msgs
        insert_at = next(
            (i for i in range(len(msgs) - 1, -1, -1) if msgs[i].role == Role.USER),
            len(msgs),
        )
        msgs.insert(insert_at, mem_msg)
        return msgs

    def _resolve_active_model(
        self, model_override: ModelConfig | None = None
    ) -> tuple[ModelConfig, ProviderConfig]:
        active_model = (
            model_override
            or self._fallback_model_override
            or self.config.get_active_model()
        )
        return active_model, self.config.get_provider_for_model(active_model)

    def _capture_chat_content(
        self, span: trace.Span, response_msg: LLMMessage, user_msg: LLMMessage | None
    ) -> None:
        """Attach turn prose to the chat span when content capture is opted in.

        Off by default; the only way a long session's recall behaviour (model
        restating/contradicting itself) becomes visible in a trace, since spans
        otherwise carry token counts but no message text.
        """
        if not self.config.otel_capture_content:
            return
        add_message_content_events(
            span,
            user_text=user_msg.content if user_msg else None,
            assistant_text=response_msg.content,
            reasoning_text=response_msg.reasoning_content,
            tool_call_names=[
                tc.function.name
                for tc in (response_msg.tool_calls or ())
                if tc.function.name
            ],
        )

    def _available_tools(self, active_model: ModelConfig) -> list[AvailableTool]:
        # Tool subset is tier-invariant (no tool removed); only the schema-text
        # description is trimmed on a small-window tier. Keyed on the per-turn
        # effective model so a failover to a small window trims consistently.
        tier = baseline_tier_for(active_model, self.config)
        return self.format_handler.get_available_tools(
            self.tool_manager,
            trim_descriptions=trim_tool_descriptions(tier, self.config),
            description_max_chars=self.config.baseline_scaling.tool_description_max_chars,
        )

    @staticmethod
    def _collect_responded_ids(
        messages: Sequence[LLMMessage], start: int
    ) -> tuple[set[str], int]:
        # Only the contiguous block is scanned — not every later tool message —
        # so a placeholder always lands inside the current turn, not a later one.
        responded: set[str] = set()
        j = start
        while j < len(messages) and messages[j].role == "tool":
            msg = messages[j]
            if msg.tool_call_id is not None:
                responded.add(msg.tool_call_id)
            j += 1
        return responded, j

    def _reconstruct_files_read(self) -> None:
        if self._files_read_reconstructed:
            return
        self._files_read_reconstructed = True
        for msg in self.messages:
            if msg.role != Role.ASSISTANT or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.function.name not in {"read", "write_file"}:
                    continue
                try:
                    args = orjson.loads(tc.function.arguments or "{}")
                except (orjson.JSONDecodeError, TypeError):
                    continue
                raw_path = args.get("file_path") or args.get("path")
                if not raw_path:
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = Path.cwd() / path
                path = path.resolve()
                if not path.exists():
                    continue
                try:
                    self._files_read[str(path)] = file_fingerprint(path)
                except OSError:
                    continue

    async def _check_agents_md_changed(self) -> None:
        from vibe.core.config.harness_files._harness_manager import (
            get_harness_files_manager,
        )

        try:
            mgr = get_harness_files_manager()
        except RuntimeError:
            return
        parts: list[str] = []
        for path in mgr.agents_md_file_paths():
            try:
                parts.append(file_fingerprint(path))
            except OSError:
                pass
        current = "|".join(parts)
        if self._agents_md_fingerprint is None:
            self._agents_md_fingerprint = current
        elif current != self._agents_md_fingerprint:
            self._agents_md_fingerprint = current
            await self.refresh_system_prompt()
