"""Session-lifecycle mixin for AgentLoop.

Provides history hygiene, session reset/fork/clear/compact/switch/reload, and the
agents.md-change check. Extracted from the loop module.

Implicit dependencies on the host class (AgentLoop):

Attributes (set by AgentLoop.__init__):
    agent_manager          (AgentManager)
    backend_factory        (Callable)
    connector_registry     (ConnectorRegistry | None)
    experiment_manager     (ExperimentManager)
    mcp_registry           (MCPRegistry | None)
    middleware_pipeline    (MiddlewarePipeline)
    parent_session_id      (str | None)
    session_logger         (SessionLogger)
    skill_manager          (SkillManager)
    scratchpad_dir         (Path)
    stats                  (AgentStats)
    tool_manager           (ToolManager)
    entrypoint_metadata    (EntrypointMetadata | None)
    terminal_emulator      (TerminalEmulator | None)
    _base_config           (VibeConfig)
    _fallback_model_override (ModelConfig | None)
    _files_read            (set[str])
    _files_read_reconstructed (bool)
    _headless              (bool)
    _hook_config_result    (HookConfigResult | None)
    _max_output_override   (int | None)
    _max_price             (float | None)
    _max_session_tokens    (int | None)
    _max_turns             (int | None)
    _permission_store      (PermissionStore)
    _session_started       (bool)
    _tried_fallback_aliases (set[str])
    _agents_md_fingerprint (str | None)

Properties (defined on AgentLoop):
    config                 (VibeConfig)
    base_config            (VibeConfig)
    agent_profile          (AgentProfile)
    backend                (BackendLike)
    enable_streaming       (bool)
    session_id             (str)

Methods (defined elsewhere on AgentLoop / sibling mixins):
    _chat(...) / _setup_middleware() / _ensure_remote_registries()
    initialize_experiments() / refresh_system_prompt()
    emit_new_session_telemetry() / emit_session_closed_telemetry()
    _fire_session_end_hooks()
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from vibe.core.agent_loop._errors import (
    AgentLoopLLMResponseError,
    CompactionFailedError,
)
from vibe.core.agent_loop._init_guard import requires_init
from vibe.core.agent_loop.backend_mixin import AgentLoopBackendMixin
from vibe.core.baseline_scaling import BaselineTier
from vibe.core.compaction import (
    build_extractive_summary,
    build_summary_input,
    collect_leading_injected_context,
    collect_persisted_tool_outputs,
    collect_prior_user_messages,
    render_compaction_context,
)
from vibe.core.config.fingerprint import file_fingerprint
from vibe.core.llm.backend.factory import create_backend
from vibe.core.logger import logger
from vibe.core.middleware import ResetReason
from vibe.core.prompts import UtilityPrompt
from vibe.core.session.session_id import extract_suffix, generate_session_id
from vibe.core.skills.manager import SkillManager
from vibe.core.system_prompt import get_universal_system_prompt
from vibe.core.tools.manager import ToolManager
from vibe.core.types import (
    AgentStats,
    InjectedMessageKind,
    LLMMessage,
    MessageList,
    Role,
)
from vibe.core.utils import CancellationReason, get_user_cancellation_message

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibe.core.agent_loop._loop import AgentLoop
    from vibe.core.agents.manager import AgentManager
    from vibe.core.config import VibeConfig
    from vibe.core.experiments import ExperimentManager
    from vibe.core.hooks.models import HookConfigResult
    from vibe.core.middleware import MiddlewarePipeline
    from vibe.core.telemetry.types import EntrypointMetadata, TerminalEmulator
    from vibe.core.tools.connectors import ConnectorRegistry


class AgentLoopSessionMixin(AgentLoopBackendMixin):
    """Mixin that adds session lifecycle (reset/fork/clear/compact/switch) to AgentLoop.

    Inherits AgentLoopBackendMixin (top of the mixin chain: Backend → Failover →
    Memory → Safety → Hooks) so all shared attrs/methods resolve via inheritance
    without redeclaration.
    """

    # Declared for type-checking only; set by AgentLoop.__init__. Plain
    # annotations (no values) — they do not shadow instance attrs set in __init__.
    agent_manager: AgentManager
    backend_factory: Callable[..., Any]
    connector_registry: ConnectorRegistry | None
    enable_streaming: bool
    entrypoint_metadata: EntrypointMetadata | None
    experiment_manager: ExperimentManager
    mcp_registry: Any
    middleware_pipeline: MiddlewarePipeline
    parent_session_id: str | None
    scratchpad_dir: Path | None
    session_logger: Any
    skill_manager: SkillManager
    terminal_emulator: TerminalEmulator | None
    tool_manager: ToolManager
    _base_config: VibeConfig
    _files_read: dict[str, str]
    _files_read_reconstructed: bool
    _headless: bool
    _hook_config_result: HookConfigResult | None
    _max_price: float | None
    _max_session_tokens: int | None
    _max_turns: int | None
    _session_started: bool
    _agents_md_fingerprint: str | None

    # ``_fallback_model_override`` and ``_tried_fallback_aliases`` are declared
    # on AgentLoopFailoverMixin; omitted here to avoid an incompatible-override
    # error. They resolve via the MRO (Failover is a base of this mixin).

    @property
    def base_config(self) -> VibeConfig: ...

    async def initialize_experiments(self) -> Any: ...

    async def refresh_system_prompt(self) -> None: ...

    def _current_baseline_tier(self) -> BaselineTier: ...

    _system_prompt_tier: BaselineTier

    def emit_new_session_telemetry(self) -> None: ...

    def emit_session_closed_telemetry(self) -> None: ...

    def _setup_middleware(self) -> None: ...

    def _ensure_remote_registries(self) -> None: ...

    # ``_fire_session_end_hooks`` is provided by AgentLoopHooksMixin via MRO;
    # redeclaring it here as a stub would shadow the real implementation.

    def _clean_message_history(self) -> None:
        ACCEPTABLE_HISTORY_SIZE = 2
        if len(self.messages) < ACCEPTABLE_HISTORY_SIZE:
            return
        self._fill_missing_tool_responses()

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
        from vibe.core.agent_loop._loop import AgentLoop as _AgentLoop, AgentLoopParams

        messages = self._messages_for_fork(message_id)
        forked = _AgentLoop(
            self.base_config.model_copy(deep=True),
            params=AgentLoopParams(
                agent_name=self.agent_profile.name,
                max_turns=self._max_turns,
                max_price=self._max_price,
                max_session_tokens=self._max_session_tokens,
                enable_streaming=self.enable_streaming,
                entrypoint_metadata=self.entrypoint_metadata,
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
                    if self.config.raise_on_compaction_failure:
                        reason = "tool_call" if has_tool_calls else "empty_summary"
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
