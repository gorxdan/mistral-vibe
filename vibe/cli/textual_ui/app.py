from __future__ import annotations

import asyncio
import codecs
import collections
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from enum import StrEnum, auto
import functools
import gc
import os
from pathlib import Path
import signal
import time
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid4
from weakref import WeakKeyDictionary

import orjson
from pydantic import BaseModel
from rich import print as rprint
from textual.app import WINDOWS, App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalGroup, VerticalScroll
from textual.css.query import NoMatches
from textual.driver import Driver
from textual.events import AppBlur, AppFocus, MouseUp, Paste
from textual.theme import BUILTIN_THEMES
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Input, Static, TextArea
from textual.worker import Worker, WorkerFailed, WorkerState

from vibe import __version__ as CORE_VERSION
from vibe.cli.clipboard import (
    copy_selection_to_clipboard,
    copy_text_to_clipboard,
    read_clipboard,
)
from vibe.cli.commands import CommandAvailabilityContext, CommandRegistry
from vibe.cli.narrator_manager import NarratorManagerPort, NarratorState
from vibe.cli.plan_offer.adapters.http_whoami_gateway import HttpWhoAmIGateway
from vibe.cli.plan_offer.decide_plan_offer import (
    PlanInfo,
    decide_plan_offer,
    plan_offer_cta,
    plan_title,
    resolve_api_key_for_plan,
)
from vibe.cli.plan_offer.ports.whoami_gateway import WhoAmIGateway, WhoAmIPlanType
from vibe.cli.terminal_detect import Terminal, detect_terminal
from vibe.cli.textual_ui.handlers.event_handler import EventHandler
from vibe.cli.textual_ui.message_queue import MessageQueue, QueueController, QueuePorts
from vibe.cli.textual_ui.notifications import (
    NotificationContext,
    NotificationPort,
    TextualNotificationAdapter,
)
from vibe.cli.textual_ui.quit_manager import QuitManager
from vibe.cli.textual_ui.scheduled_loop_runner import ScheduledLoopRunner
from vibe.cli.textual_ui.session_exit import print_session_resume_message
from vibe.cli.textual_ui.widgets.agent_badge import ModelStatusBadge, SubModelBadge
from vibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from vibe.cli.textual_ui.widgets.banner.banner import Banner
from vibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from vibe.cli.textual_ui.widgets.chat_input.input_kinds import (
    Bash,
    EmptyBash,
    LeChatonPrompt,
    Prompt,
    Skill,
    SlashCommand,
    Teleport,
    classify,
)
from vibe.cli.textual_ui.widgets.chat_input.paste_image import (
    handle_clipboard_image_paste,
)
from vibe.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from vibe.cli.textual_ui.widgets.collapsible import CollapsibleSection
from vibe.cli.textual_ui.widgets.compact import CompactMessage
from vibe.cli.textual_ui.widgets.config_app import ConfigApp
from vibe.cli.textual_ui.widgets.context_progress import ContextProgress, TokenState
from vibe.cli.textual_ui.widgets.debug_console import DebugConsole
from vibe.cli.textual_ui.widgets.effort_picker import EffortPickerApp
from vibe.cli.textual_ui.widgets.feedback_bar import FeedbackBar
from vibe.cli.textual_ui.widgets.feedback_bar_manager import FeedbackBarManager
from vibe.cli.textual_ui.widgets.load_more import HistoryLoadMoreRequested
from vibe.cli.textual_ui.widgets.loading import (
    DEFAULT_LOADING_STATUS,
    LoadingWidget,
    paused_timer,
)
from vibe.cli.textual_ui.widgets.mcp_add_app import MCPAddApp
from vibe.cli.textual_ui.widgets.messages import (
    VSCODE_EXTENSION_PROMO_WHATS_NEW_SUFFIX,
    AssistantMessage,
    BashOutputMessage,
    ErrorMessage,
    InterruptMessage,
    LspInstallCallout,
    LspInstallHintCallout,
    SlashCommandMessage,
    StreamingMessageBase,
    TeleportUserMessage,
    UserCommandMessage,
    UserMessage,
    VscodeExtensionPromoMessage,
    WarningMessage,
    WhatsNewMessage,
)
from vibe.cli.textual_ui.widgets.model_picker import ModelPickerApp
from vibe.cli.textual_ui.widgets.narrator_status import NarratorStatus
from vibe.cli.textual_ui.widgets.no_markup_static import (
    NoMarkupStatic,
    NonSelectableStatic,
)
from vibe.cli.textual_ui.widgets.path_display import PathDisplay
from vibe.cli.textual_ui.widgets.provider_login_app import ProviderLoginApp
from vibe.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp
from vibe.cli.textual_ui.widgets.question_app import QuestionApp
from vibe.cli.textual_ui.widgets.rewind_app import RewindApp
from vibe.cli.textual_ui.widgets.session_picker import SessionPickerApp
from vibe.cli.textual_ui.widgets.subagents_badge import SubagentsBadge
from vibe.cli.textual_ui.widgets.tasks_app import TasksApp
from vibe.cli.textual_ui.widgets.teleport_message import TeleportMessage
from vibe.cli.textual_ui.widgets.theme_picker import ThemePickerApp, sorted_theme_names
from vibe.cli.textual_ui.widgets.thinking_picker import ThinkingPickerApp
from vibe.cli.textual_ui.widgets.tool_widgets import (
    EditApprovalWidget,
    EditResultWidget,
)
from vibe.cli.textual_ui.widgets.voice_app import VoiceApp
from vibe.cli.textual_ui.widgets.workflow_save_app import WorkflowSaveApp
from vibe.cli.textual_ui.windowing import (
    HISTORY_RESUME_TAIL_MESSAGES,
    LOAD_MORE_BATCH_SIZE,
    HistoryLoadMoreManager,
    SessionWindowing,
    build_history_widgets,
    create_resume_plan,
    non_system_history_messages,
    should_resume_history,
    sync_backfill_state,
)
from vibe.cli.textual_ui.workflow_runner import WorkflowRunner
from vibe.cli.update_notifier import (
    GitHubUpdateGateway,
    UpdateCacheRepository,
    UpdateError,
    UpdateGateway,
    get_update_if_available,
    load_whats_new_content,
    mark_version_as_seen,
    should_show_whats_new,
)
from vibe.cli.voice_manager import VoiceManagerPort
from vibe.cli.voice_manager.voice_manager_port import TranscribeState
from vibe.cli.vscode_extension_promo import (
    FileSystemVscodeExtensionPromoRepository,
    VscodeExtensionPromo,
    VscodeExtensionPromoState,
    should_show_promo,
)
from vibe.core.agents import AgentProfile
from vibe.core.autocompletion.path_prompt import (
    PathPromptPayload,
    PathResource,
    build_path_prompt_payload,
    build_title_segments,
)
from vibe.core.autocompletion.path_prompt_adapter import (
    extract_image_resources,
    render_path_prompt_from_payload,
)
from vibe.core.config import (
    DEFAULT_THEME,
    MCPHttp,
    MCPStreamableHttp,
    ModelConfig,
    VibeConfig,
)
from vibe.core.data_retention import DATA_RETENTION_MESSAGE
from vibe.core.hooks.models import HookStartEvent
from vibe.core.log_reader import LogReader
from vibe.core.logger import logger
from vibe.core.lsp._lifecycle import setup_lsp_for_config, teardown_lsp_async
from vibe.core.paths import CACHE_FILE, HISTORY_FILE, safe_cwd
from vibe.core.rewind import RewindError
from vibe.core.search import (
    SearxngSettings,
    begin_autostart,
    ensure_running,
    signal_autostart_done,
    stop_all_started,
)
from vibe.core.sentry import capture_sentry_exception
from vibe.core.session.image_snapshot import ImageSnapshotError, snapshot_image
from vibe.core.session.resume_sessions import (
    ResumeSessionInfo,
    list_local_resume_sessions,
    session_latest_messages,
    short_session_id,
)
from vibe.core.session.saved_sessions import (
    delete_saved_session,
    update_saved_session_title_at_path,
)
from vibe.core.session.session_loader import SessionLoader
from vibe.core.session.title_format import format_session_title
from vibe.core.skills.manager import SkillManager
from vibe.core.teams.manager import TeamManager
from vibe.core.teams.models import TeamSafetyMode
from vibe.core.teleport.telemetry import send_teleport_early_failure_telemetry
from vibe.core.teleport.types import (
    TeleportCheckingGitEvent,
    TeleportCompleteEvent,
    TeleportPushingEvent,
    TeleportPushRequiredEvent,
    TeleportPushResponseEvent,
    TeleportStartingWorkflowEvent,
)
from vibe.core.tools.background import BackgroundRegistry, TaskCategory
from vibe.core.tools.base import InvokeContext
from vibe.core.tools.builtins.ask_user_question import (
    AskUserQuestionArgs,
    AskUserQuestionResult,
    Choice,
    Question,
)
from vibe.core.tools.builtins.websearch import resolve_searxng_settings
from vibe.core.tools.connectors import compute_connector_counts
from vibe.core.tools.mcp_settings import persist_mcp_toggle
from vibe.core.tools.permissions import RequiredPermission
from vibe.core.transcribe import make_transcribe_client
from vibe.core.types import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_MESSAGE,
    AgentStats,
    ApprovalResponse,
    Backend,
    BaseEvent,
    ContentFilterError,
    ContextTooLongError,
    ImageAttachment,
    LLMMessage,
    RateLimitError,
    RefusalError,
    Role,
    WaitingForInputEvent,
)
from vibe.core.utils import (
    CancellationReason,
    get_user_cancellation_message,
    is_dangerous_directory,
)
from vibe.core.workflows.manager import WorkflowManager
from vibe.core.workflows.runtime import WorkflowError, WorkflowRuntime

if TYPE_CHECKING:
    from vibe.cli.narrator_manager import NarratorManager
    from vibe.cli.textual_ui.widgets.connector_auth_app import ConnectorAuthApp
    from vibe.cli.textual_ui.widgets.mcp_app import MCPApp
    from vibe.cli.voice_manager import VoiceManager
    from vibe.core.agent_loop import AgentLoop

_VSCODE_FAMILY_TERMINALS = {Terminal.VSCODE, Terminal.VSCODE_INSIDERS, Terminal.CURSOR}


def _is_vscode_family_terminal() -> bool:
    return detect_terminal() in _VSCODE_FAMILY_TERMINALS


class BottomApp(StrEnum):
    Approval = auto()
    Config = auto()
    ConnectorAuth = auto()
    EffortPicker = auto()
    Input = auto()
    MCP = auto()
    MCPAdd = auto()
    ModelPicker = auto()
    ProviderLogin = auto()
    ProxySetup = auto()
    Question = auto()
    ThemePicker = auto()
    ThinkingPicker = auto()
    Rewind = auto()
    SessionPicker = auto()
    Voice = auto()
    Tasks = auto()


class ChatScroll(VerticalScroll):
    @property
    def is_at_bottom(self) -> bool:
        return self.scroll_target_y >= self.max_scroll_y

    _reanchor_pending: bool = False
    _scrolling_down: bool = False

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._scrolling_down = new_value >= old_value

    def release_anchor(self) -> None:
        super().release_anchor()
        # Textual's MRO dispatch calls Widget._on_mouse_scroll_down AFTER
        # our override, so any re-anchor we do gets immediately undone.
        # Defer the re-check until all handlers for this event have finished.
        if not self._reanchor_pending:
            self._reanchor_pending = True
            self.call_later(self._maybe_reanchor)

    def _maybe_reanchor(self) -> None:
        self._reanchor_pending = False
        if (
            self._anchored
            and self._anchor_released
            and self.is_at_bottom
            and self._scrolling_down
        ):
            self.anchor()

    def update_node_styles(self, animate: bool = True) -> None:
        pass


PRUNE_LOW_MARK = 1000
PRUNE_HIGH_MARK = 1500
DOUBLE_ESC_DELAY = 0.2
_SUBAGENTS_BADGE_REFRESH_S = 1.0
_MAX_AUTO_CONTINUES = 100
_AUTO_CONTINUE_PROMPT = (
    "A background subagent finished; its result is above. Continue the task, "
    "or stop if nothing remains."
)
_WORKFLOW_CONTINUE_PROMPT = (
    "A background workflow finished; its result is above. Continue the task, "
    "or stop if nothing remains."
)

_DEFAULT_TYPING_DEBOUNCE_MS = 1000
_TYPING_DEBOUNCE_ENV_VAR = "VIBE_TYPING_GRACE_PERIOD_MS"


def _resolve_typing_debounce_s() -> float:
    try:
        ms = int(os.environ[_TYPING_DEBOUNCE_ENV_VAR])
        if ms < 0:
            raise ValueError
    except (KeyError, ValueError):
        ms = _DEFAULT_TYPING_DEBOUNCE_MS
    return ms / 1000


async def prune_oldest_children(
    messages_area: Widget, low_mark: int, high_mark: int
) -> bool:
    total_height = messages_area.virtual_size.height
    if total_height <= high_mark:
        return False

    children = messages_area.children
    if not children:
        return False

    accumulated = 0
    cut = len(children)

    for child in reversed(children):
        if not child.display:
            cut -= 1
            continue
        accumulated += child.outer_size.height
        cut -= 1
        if accumulated >= low_mark:
            break

    to_remove = list(children[:cut])
    if not to_remove:
        return False

    await messages_area.remove_children(to_remove)
    return True


@dataclass(frozen=True, slots=True)
class StartupOptions:
    initial_prompt: str | None = None
    teleport_on_start: bool = False
    show_resume_picker: bool = False
    is_resuming_session: bool = False


_REJECT_HINT_BUSY = "wait for the current job to finish."
_REJECT_HINT_PAUSED = "clear the queue first or remove this input."
_SUBAGENT_MODEL_HINT = (
    "Subagent model — used by the task tool and workflow agents when no "
    "per-spawn model is set. Empty inherits the host session's model."
)
_GRUNT_MODEL_HINT = (
    "Grunt model — default model for the 'grunt' subagent (bulk/grunt work). "
    "Empty falls back to subagent_model, then the host."
)


@dataclass(frozen=True, slots=True)
class _ImageAttachmentRejection:
    message: str
    no_vision: bool = False


class VibeApp(App):
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "app.tcss"
    PAUSE_GC_ON_SCROLL: ClassVar[bool] = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "interrupt_or_quit", "Quit", show=False),
        Binding("ctrl+d", "delete_right_or_quit", "Quit", show=False, priority=True),
        Binding("ctrl+z", "suspend_with_message", "Suspend", show=False, priority=True),
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+o", "toggle_tool", "Toggle Tool", show=False),
        Binding("ctrl+y", "copy_selection", "Copy", show=False, priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy", show=False, priority=True),
        Binding("shift+tab", "cycle_mode", "Cycle Mode", show=False, priority=True),
        Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False, priority=True),
        Binding(
            "shift+down", "scroll_chat_down", "Scroll Down", show=False, priority=True
        ),
        Binding(
            "ctrl+g", "open_plan_in_editor", "Edit Plan", show=False, priority=False
        ),
        Binding("ctrl+backslash", "toggle_debug_console", "Debug Console", show=False),
        Binding("alt+up", "rewind_prev", "Rewind Previous", show=False, priority=True),
        Binding("ctrl+p", "rewind_prev", "Rewind Previous", show=False, priority=True),
        Binding("alt+down", "rewind_next", "Rewind Next", show=False, priority=True),
        Binding("ctrl+n", "rewind_next", "Rewind Next", show=False, priority=True),
        Binding("ctrl+w", "toggle_tasks", "Tasks", show=False, priority=True),
    ]

    def get_driver_class(self) -> type[Driver]:
        from vibe.cli.textual_ui.terminal_input_filter import patch_driver_parser

        driver_class = super().get_driver_class()
        patch_driver_parser(driver_class)
        return driver_class

    def __init__(
        self,
        agent_loop: AgentLoop,
        startup: StartupOptions | None = None,
        update_notifier: UpdateGateway | None = None,
        update_cache_repository: UpdateCacheRepository | None = None,
        current_version: str = CORE_VERSION,
        plan_offer_gateway: WhoAmIGateway | None = None,
        terminal_notifier: NotificationPort | None = None,
        voice_manager: VoiceManagerPort | None = None,
        narrator_manager: NarratorManagerPort | None = None,
        vscode_extension_promo: VscodeExtensionPromo | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent_loop = agent_loop
        self._init_core_state(
            update_notifier,
            update_cache_repository,
            current_version,
            plan_offer_gateway,
            vscode_extension_promo,
        )
        self._init_managers(voice_manager, narrator_manager, terminal_notifier)
        self._init_ui_state()
        self._init_workflow_and_commands(startup)

    def _init_core_state(
        self,
        update_notifier: UpdateGateway | None,
        update_cache_repository: UpdateCacheRepository | None,
        current_version: str,
        plan_offer_gateway: WhoAmIGateway | None,
        vscode_extension_promo: VscodeExtensionPromo | None,
    ) -> None:
        self._plan_info: PlanInfo | None = None
        self._agent_running = False
        self._interrupt_requested = False
        self._agent_task: asyncio.Task | None = None
        self._workflow_flush_tasks: set[asyncio.Task] = set()
        self._auto_continue_active = False
        self._consecutive_auto_continues = 0
        self._bash_task: asyncio.Task | None = None
        self._update_notifier = update_notifier
        self._update_cache_repository = update_cache_repository
        self._current_version = current_version
        self._plan_offer_gateway = plan_offer_gateway
        self._vscode_extension_promo = vscode_extension_promo
        self._show_vscode_extension_promo = (
            vscode_extension_promo is not None
            and _is_vscode_family_terminal()
            and should_show_promo(vscode_extension_promo.initial_state)
        )

    def _init_managers(
        self,
        voice_manager: VoiceManagerPort | None,
        narrator_manager: NarratorManagerPort | None,
        terminal_notifier: NotificationPort | None,
    ) -> None:
        self._voice_manager: VoiceManagerPort = (
            voice_manager or self._make_default_voice_manager()
        )
        self._terminal_notifier = terminal_notifier or TextualNotificationAdapter(
            self,
            get_enabled=lambda: self.config.enable_notifications,
            default_title="Mistral Vibe",
        )
        self._narrator_manager: NarratorManagerPort = (
            narrator_manager or self._make_default_narrator_manager()
        )

    def _init_ui_state(self) -> None:
        self._queue = QueueController(self._build_queue_ports())
        self._loading_widget: LoadingWidget | None = None
        self._pending_approval: asyncio.Future | None = None
        self._pending_question: asyncio.Future | None = None
        # Set while a rate-limit model-switch dialog is open mid-turn; the model
        # picker resolves it with the chosen alias (or None on cancel) so the
        # agent loop can switch model and retry instead of surfacing the 429.
        self._pending_model_switch: asyncio.Future | None = None
        self._user_interaction_lock = asyncio.Lock()
        self.event_handler: EventHandler | None = None
        self._chat_input_container: ChatInputContainer | None = None
        self._current_bottom_app: BottomApp = BottomApp.Input
        self.history_file = HISTORY_FILE.path
        self._tools_collapsed = True
        self._lsp_nudge_shown_this_session = False
        self._recent_edited_exts: collections.deque[str] = collections.deque(maxlen=16)
        self._windowing = SessionWindowing(load_more_batch_size=LOAD_MORE_BATCH_SIZE)
        self._load_more = HistoryLoadMoreManager()
        self._tool_call_map: dict[str, str] | None = None
        self._history_widget_indices: WeakKeyDictionary[Widget, int] = (
            WeakKeyDictionary()
        )
        self._last_escape_time: float | None = None
        self._quit_manager = QuitManager(self)
        self._banner: Banner | None = None
        self._whats_new_message: WhatsNewMessage | None = None
        self._cached_messages_area: Widget | None = None
        self._cached_chat: ChatScroll | None = None
        self._cached_loading_area: Widget | None = None
        self._log_reader = LogReader()
        self._debug_console: DebugConsole | None = None
        self._switch_agent_generation = 0
        self._rewind_mode = False
        self._rewind_highlighted_widget: UserMessage | None = None
        self._fatal_init_error = False
        self._force_quit_task: asyncio.Task[None] | None = None

    def _init_workflow_and_commands(self, startup: StartupOptions | None) -> None:
        self.commands = self._build_command_registry()
        self._loop_runner = ScheduledLoopRunner(
            self.agent_loop.session_logger,
            can_fire=lambda: (
                not self._agent_running and self._current_bottom_app == BottomApp.Input
            ),
            fire=self._handle_user_message,
            mount=self._mount_and_scroll,
            tools_collapsed=lambda: self._tools_collapsed,
        )
        self.agent_loop.set_scheduler(self._loop_runner.manager)
        self._workflow_runner = WorkflowRunner(
            mount=self._mount_and_scroll,
            on_complete=self._on_workflow_complete,
            persist_callback=self._persist_workflow_snapshots,
            snapshot_loader=self._load_workflow_snapshots,
            resume_runtime_factory=self._build_resume_runtime,
        )
        self._workflow_manager = WorkflowManager(lambda: self.agent_loop.config)
        self._register_workflow_commands()
        self.agent_loop.launch_workflow_callback = self._launch_workflow_from_tool
        self.agent_loop.workflow_status_callback = self._workflow_status_for_tool
        self.agent_loop.workflow_results_callback = self._workflow_results_for_tool
        self.agent_loop.workflow_stop_callback = self._workflow_stop_for_tool
        self.agent_loop.team_dir_callback = self._team_dir_for_tool
        self.agent_loop.team_spawn_callback = self._team_spawn_for_tool
        self._team_manager: TeamManager | None = None
        self._background_registry = BackgroundRegistry()
        self._background_registry.attach_workflow_runner(lambda: self._workflow_runner)
        self._background_registry.attach_team_manager(lambda: self._team_manager)
        self._background_registry.attach_loop_manager(lambda: self._loop_runner.manager)
        self.agent_loop.background_registry = self._background_registry
        self._background_registry.attach_completion_callback(
            self._on_async_completion_wake
        )
        self._configure_startup_options(startup)

    def _configure_startup_options(self, startup: StartupOptions | None) -> None:
        opts = startup or StartupOptions()
        self._initial_prompt = opts.initial_prompt
        self._teleport_on_start = (
            opts.teleport_on_start and self.agent_loop.base_config.vibe_code_enabled
        )
        self._show_resume_picker = opts.show_resume_picker
        self._is_resuming_session = opts.is_resuming_session

    @property
    def config(self) -> VibeConfig:
        return self.agent_loop.config

    @property
    def _input_queue(self) -> MessageQueue:
        return self._queue.queue

    def _build_queue_ports(self) -> QueuePorts:
        return QueuePorts(
            mount_and_scroll=self._mount_and_scroll,
            agent_running=lambda: self._agent_running,
            bash_task=lambda: self._bash_task,
            active_model=self._active_model_or_none,
            remove_loading_widget=self._remove_loading_widget,
            set_loading_queue_count=self._set_loading_queue_count,
            inject_user_context=self.agent_loop.inject_user_context,
            stage_injected_message=self.agent_loop.stage_injected_message,
            next_message_index=lambda: len(self.agent_loop.messages),
            start_agent_turn=self._start_queued_agent_turn,
            await_agent_turn=self._await_agent_turn,
            run_bash=self._start_queued_bash,
            maybe_show_feedback_bar=self._maybe_show_feedback_bar,
            send_skill_telemetry=self._send_skill_telemetry,
            send_at_mention_telemetry=self._send_at_mention_telemetry,
            render_payload=lambda payload: render_path_prompt_from_payload(
                payload, skip_images=True
            ),
        )

    def _active_model_or_none(self) -> ModelConfig | None:
        try:
            return self.agent_loop.config.get_active_model()
        except ValueError:
            return None

    def _set_loading_queue_count(self, count: int) -> None:
        if self._loading_widget is not None:
            self._loading_widget.set_queue_count(count)

    def _maybe_show_feedback_bar(self) -> None:
        if self._feedback_bar_manager.should_show(self.agent_loop):
            self._feedback_bar.show()
            self._feedback_bar_manager.record_feedback_asked(self.agent_loop)

    def _start_queued_agent_turn(
        self,
        content: str,
        *,
        prebuilt_images: list[ImageAttachment] | None = None,
        prebuilt_payload: PathPromptPayload | None = None,
    ) -> asyncio.Task:
        self._agent_task = asyncio.create_task(
            self._handle_agent_loop_turn(
                content,
                prebuilt_images=prebuilt_images,
                prebuilt_payload=prebuilt_payload,
            )
        )
        return self._agent_task

    async def _await_agent_turn(self) -> None:
        agent_task = self._agent_task
        if agent_task is None:
            return
        await agent_task

    async def _on_async_completion_wake(self) -> None:
        # A background subagent finished: if idle (and the user isn't driving),
        # auto-continue one turn to drain it. Capped to avoid an unattended loop.
        if self._is_busy() or self._auto_continue_active:
            return
        if self._input_queue.paused or bool(self._input_queue):
            return
        if self._consecutive_auto_continues >= _MAX_AUTO_CONTINUES:
            return
        self._auto_continue_active = True
        self._consecutive_auto_continues += 1
        self._start_queued_agent_turn(_AUTO_CONTINUE_PROMPT)

    def _start_queued_bash(
        self, command: str, *, existing_widget: BashOutputMessage | None = None
    ) -> asyncio.Task:
        self._bash_task = asyncio.create_task(
            self._handle_bash_command(
                command, existing_widget=existing_widget, start_drain_on_finish=False
            )
        )
        return self._bash_task

    @property
    def _connectors_enabled(self) -> bool:
        return self.agent_loop.connector_registry is not None

    def _get_command_availability_context(self) -> CommandAvailabilityContext:
        return CommandAvailabilityContext(
            vibe_code_enabled=self.agent_loop.base_config.vibe_code_enabled,
            is_active_model_mistral=self.config.is_active_model_mistral(),
            plan_info=self._plan_info,
        )

    def _build_command_registry(self) -> CommandRegistry:
        return CommandRegistry(
            availability_context=self._get_command_availability_context()
        )

    def _refresh_command_registry(self) -> None:
        self.commands.refresh(self._get_command_availability_context())

    def compose(self) -> ComposeResult:
        with ChatScroll(id="chat"):
            connectors_connected, connectors_total = compute_connector_counts(
                self.config, self.agent_loop.connector_registry
            )
            self._banner = Banner(
                config=self.config,
                skill_manager=self.agent_loop.skill_manager,
                connectors_connected=connectors_connected,
                connectors_total=connectors_total,
                hooks_count=self.agent_loop.hooks_count,
            )
            yield self._banner
            yield VerticalGroup(id="messages")

        with Horizontal(id="loading-area"):
            yield NarratorStatus(self._narrator_manager)
            yield Static(id="loading-area-content")
            self._clipboard_notice = NonSelectableStatic(id="clipboard-notice")
            self._clipboard_notice.display = False
            self._clipboard_hide_timer: Timer | None = None
            yield self._clipboard_notice
            yield FeedbackBar()

        with Static(id="bottom-app-container"):
            yield ChatInputContainer(
                history_file=self.history_file,
                command_registry=self.commands,
                id="input-container",
                safety=self.agent_loop.agent_profile.safety,
                agent_name=self.agent_loop.agent_profile.display_name.lower(),
                skill_entries_getter=self._get_skill_entries,
                file_watcher_for_autocomplete_getter=self._is_file_watcher_enabled,
                voice_manager=self._voice_manager,
            )

        with Horizontal(id="bottom-bar"):
            yield PathDisplay(self.config.displayed_workdir or safe_cwd())
            yield ModelStatusBadge()
            yield SubModelBadge()
            yield SubagentsBadge()
            yield NoMarkupStatic(id="spacer")
            yield ContextProgress()

    @property
    def _messages_area(self) -> Widget:
        if self._cached_messages_area is None:
            self._cached_messages_area = self.query_one("#messages")
        return self._cached_messages_area

    @property
    def _chat_widget(self) -> ChatScroll:
        if self._cached_chat is None:
            self._cached_chat = self.query_one("#chat", ChatScroll)
        return self._cached_chat

    @property
    def _loading_area(self) -> Widget:
        if self._cached_loading_area is None:
            self._cached_loading_area = self.query_one("#loading-area-content")
        return self._cached_loading_area

    async def on_mount(self) -> None:
        setup_lsp_for_config(
            self.agent_loop.base_config,
            lambda: self.agent_loop.base_config,
            safe_cwd(),
            warmup=True,
        )
        self._apply_theme(self.config.theme)
        self._terminal_notifier.restore()
        self._feedback_bar = self.query_one(FeedbackBar)
        self._feedback_bar_manager = FeedbackBarManager()

        self.event_handler = EventHandler(
            mount_callback=self._mount_and_scroll,
            get_tools_collapsed=lambda: self._tools_collapsed,
            on_profile_changed=self._on_profile_changed,
            on_code_file_edited=self._maybe_nudge_lsp,
        )

        self._chat_input_container = self.query_one(ChatInputContainer)
        context_progress = self.query_one(ContextProgress)

        def update_context_progress(stats: AgentStats) -> None:
            context_progress.tokens = TokenState(
                max_tokens=self.config.get_active_model().auto_compact_threshold,
                current_tokens=stats.context_tokens,
                cached_tokens=stats.last_turn_cached_tokens,
            )

        self.agent_loop.stats.add_listener("context_tokens", update_context_progress)
        self.agent_loop.stats.trigger_listeners()

        subagents_badge = self.query_one(SubagentsBadge)

        def update_subagents_badge() -> None:
            subagents_badge.running = tuple(
                entry.label.split(":", 1)[0].strip()
                for entry in self._background_registry.list_tasks(
                    category=TaskCategory.ASYNC_AGENT
                )
                if entry.status == "running"
            )

        update_subagents_badge()
        self.set_interval(_SUBAGENTS_BADGE_REFRESH_S, update_subagents_badge)

        self.agent_loop.set_approval_callback(self._approval_callback)
        self.agent_loop.set_user_input_callback(self._user_input_callback)
        self.agent_loop.set_rate_limit_callback(self._rate_limit_callback)
        self._refresh_profile_widgets()

        chat_input_container = self.query_one(ChatInputContainer)
        chat_input_container.focus_input()
        await self._resolve_plan()
        await self._show_dangerous_directory_warning()
        await self._resume_history_from_messages()
        self._loop_runner.restore_from_session()
        self._loop_runner.start()
        await self._check_and_show_whats_new()
        self._schedule_update_notification()
        if self._is_resuming_session:
            await self.agent_loop.hydrate_experiments_from_session()
        else:
            self.agent_loop.start_initialize_experiments()

        self.call_after_refresh(self._refresh_banner)
        self._show_config_issues()

        self.run_worker(self._watch_init_completion(), exclusive=False)
        self.run_worker(self._searxng_autostart(), exclusive=False)

        if self._show_resume_picker:
            self.run_worker(self._show_session_picker(), exclusive=False)
        elif self._initial_prompt or self._teleport_on_start:
            self.call_after_refresh(self._process_initial_prompt)

        gc.collect()
        gc.freeze()

    def _show_config_issues(self) -> None:
        for issue in (
            *self.agent_loop.hook_config_issues,
            *self.agent_loop.skill_manager.config_issues,
        ):
            self.notify(
                f"{issue.file}\n{issue.message}",
                severity="warning",
                markup=False,
                timeout=10,
            )
        # One-time nudge: sandbox enabled but only the unshare backend is
        # available (namespace isolation, no filesystem write confinement).
        try:
            from vibe.core.tools.sandbox import unshare_confinement_nudge

            bash_cfg = self.agent_loop.tool_manager.get_tool_config("bash")
            sb = getattr(bash_cfg, "sandbox", None)
            if sb is not None:
                nudge = unshare_confinement_nudge(
                    sandbox_enabled=sb.enabled, backend_override=sb.backend
                )
                if nudge:
                    self.notify(nudge, severity="warning", markup=False, timeout=15)
        except Exception:
            logger.debug("bubblewrap install nudge skipped", exc_info=True)
        # One-time session-start nudge: a code project is detected but LSP is
        # off. The edit-triggered nudge only fires on file edit, so a read-only
        # session never learns LSP exists. This closes that gap at startup.
        try:
            from vibe.core.lsp._nudge import session_start_lsp_nudge

            lsp_msg = session_start_lsp_nudge(
                self.agent_loop.base_config, CACHE_FILE.path
            )
            if lsp_msg:
                self.notify(lsp_msg, markup=False, timeout=12)
        except Exception:
            logger.debug("lsp session-start nudge skipped", exc_info=True)

    async def _watch_init_completion(self) -> None:
        init_widget = None
        try:
            if not self.agent_loop.is_initialized:
                await self._ensure_loading_widget("Initializing", show_hint=False)
                init_widget = self._loading_widget
            await self.agent_loop.wait_until_ready()
            await self._show_mcp_auth_required_notice()
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Background initialization failed: {e}",
                    collapsed=self._tools_collapsed,
                )
            )
            await self._mount_and_scroll(
                Static("Press any key to exit...", classes="error-hint")
            )
            if self._chat_input_container:
                self._chat_input_container.disabled = True
                self._chat_input_container.display = False
            self._fatal_init_error = True
        finally:
            if self._loading_widget is init_widget:
                await self._remove_loading_widget()
            self._refresh_banner()
            try:
                from vibe.cli.textual_ui.widgets.mcp_app import MCPApp

                self.query_one(MCPApp).refresh_index()
            except Exception:
                pass

    async def _show_mcp_auth_required_notice(self) -> None:
        """Show a notice if any enabled MCP servers require OAuth authentication."""
        registry = self.agent_loop.mcp_registry
        if registry is None:
            return
        from vibe.core.tools.mcp import AuthStatus

        statuses = registry.status()
        disabled = registry.disabled_aliases()
        aliases = sorted(
            alias
            for alias, status in statuses.items()
            if status is AuthStatus.NEEDS_AUTH and alias not in disabled
        )
        if not aliases:
            return
        command = f"/mcp login {aliases[0]}"
        if len(aliases) > 1:
            detail = ", ".join(aliases)
            message = (
                "MCP servers need OAuth authentication: "
                f"{detail}. Run `{command}` to start with {aliases[0]!r}."
            )
        else:
            message = (
                f"MCP server {aliases[0]!r} needs OAuth authentication. "
                f"Run `{command}` to authenticate."
            )
        await self._mount_and_scroll(UserCommandMessage(message))

    def _process_initial_prompt(self) -> None:
        if self._teleport_on_start and self.commands.has_command("teleport"):
            self.run_worker(
                self._handle_teleport_command(self._initial_prompt), exclusive=False
            )
        elif self._initial_prompt:
            self.run_worker(
                self._handle_user_message(self._initial_prompt), exclusive=False
            )

    def _is_file_watcher_enabled(self) -> bool:
        return self.config.file_watcher_for_autocomplete

    def on_key(self) -> None:
        if self._fatal_init_error:
            self.exit()

    async def on_chat_input_container_submitted(
        self, event: ChatInputContainer.Submitted
    ) -> None:
        value = event.value.strip()
        input_widget = self.query_one(ChatInputContainer)

        # Double-enter: an empty submit while an agent turn is running with
        # queued prompts folds those prompts into the running turn instead of
        # waiting for the whole turn to finish.
        inject_now = (
            not value
            and self._agent_running
            and not self._input_queue.paused
            and bool(self._input_queue)
        )
        if not value and not self._input_queue.paused and not inject_now:
            return

        if self._banner:
            self._banner.freeze_animation()

        if self._whats_new_message:
            await self._whats_new_message.remove()
            self._whats_new_message = None

        if inject_now:
            await self._inject_queued_now()
            return

        if (
            self._input_queue.paused or self._is_busy()
        ) and self._is_busy_allowed_command(value):
            await self._handle_command(value)
            return

        if self._input_queue.paused:
            if not await self._handle_paused_submit(value):
                self._restore_input_if_empty(input_widget, value)
            return

        if self._is_busy():
            if not await self._handle_queue_submit(
                value, reject_hint=_REJECT_HINT_BUSY
            ):
                self._restore_input_if_empty(input_widget, value)
            return

        await self._dispatch_idle_input(value)

    @staticmethod
    def _restore_input_if_empty(input_widget: ChatInputContainer, value: str) -> None:
        if not input_widget.value:
            input_widget.value = value

    async def _empty_bash_error(self) -> None:
        await self._mount_and_scroll(
            ErrorMessage(
                "No command provided after '!'", collapsed=self._tools_collapsed
            )
        )

    def _warn_not_queueable(self, message: str) -> None:
        self.notify(message, severity="warning", markup=False)

    async def _dispatch_idle_input(self, value: str) -> None:
        match classify(value, commands=self.commands, expand_skill=self._expand_skill):
            case Teleport(target=target):
                await self._handle_teleport_command(target)
            case SlashCommand():
                await self._handle_command(value)
            case Skill(expanded_prompt=expanded):
                await self._handle_user_message(expanded, title_source=value)
            case Bash(command=command):
                self._bash_task = asyncio.create_task(
                    self._handle_bash_command(command)
                )
                self._queue.notify_busy_changed()
            case EmptyBash():
                await self._empty_bash_error()
            case Prompt(text=text):
                await self._handle_user_message(text)
            case LeChatonPrompt(text=text):
                await self._handle_le_chaton_prompt(text)

    async def _handle_paused_submit(self, value: str) -> bool:
        if value and not await self._handle_queue_submit(
            value, reject_hint=_REJECT_HINT_PAUSED
        ):
            return False
        self._queue.set_paused(False)
        self._queue.start_drain_if_needed()
        return True

    async def _handle_queue_submit(self, value: str, *, reject_hint: str) -> bool:
        match classify(value, commands=self.commands, expand_skill=self._expand_skill):
            case Teleport():
                self._warn_not_queueable(f"Teleport cannot be queued — {reject_hint}")
                return False
            case SlashCommand():
                self._warn_not_queueable(
                    f"Slash commands cannot be queued — {reject_hint}"
                )
                return False
            case Skill(expanded_prompt=expanded, name=name):
                return await self._enqueue_prompt_with_resources(
                    expanded, skill_name=name
                )
            case Bash(command=command):
                await self._queue.enqueue_bash(command)
            case EmptyBash():
                await self._empty_bash_error()
            case Prompt(text=text):
                return await self._enqueue_prompt_with_resources(text)
            case LeChatonPrompt(text=text):
                return await self._enqueue_prompt_with_resources(text)
        return True

    async def _enqueue_prompt_with_resources(
        self, content: str, *, skill_name: str | None = None
    ) -> bool:
        payload = build_path_prompt_payload(content, base_dir=safe_cwd())
        images = await self._prepare_images_or_abort(payload)
        if images is None:
            return False
        await self._queue.enqueue_prompt(
            content, skill_name=skill_name, images=images, payload=payload
        )
        return True

    async def _inject_queued_now(self) -> None:
        if await self._queue.inject_now():
            self.notify(
                "Injected — the agent will pick this up at the next step.", timeout=3
            )

    def _is_busy(self) -> bool:
        if self._agent_running:
            return True
        if self._bash_task is not None and not self._bash_task.done():
            return True
        if self._queue.draining:
            return True
        return False

    def _is_busy_allowed_command(self, value: str) -> bool:
        if not value.startswith("/"):
            return False
        resolved = self.commands.parse_command(value)
        return resolved is not None and resolved[1].safe_while_busy

    async def on_approval_app_approval_granted(
        self, message: ApprovalApp.ApprovalGranted
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None, None))

    async def on_approval_app_approval_granted_always_tool(
        self, message: ApprovalApp.ApprovalGrantedAlwaysTool
    ) -> None:
        self.agent_loop.approve_always(message.tool_name, message.required_permissions)

        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None, None))

    async def on_approval_app_approval_granted_always_permanent(
        self, message: ApprovalApp.ApprovalGrantedAlwaysPermanent
    ) -> None:
        self.agent_loop.approve_always(
            message.tool_name, message.required_permissions, save_permanently=True
        )

        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None, None))

    async def on_approval_app_approval_rejected(
        self, message: ApprovalApp.ApprovalRejected
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            feedback = str(
                get_user_cancellation_message(CancellationReason.OPERATION_CANCELLED)
            )
            self._pending_approval.set_result((ApprovalResponse.NO, feedback, None))

        if self._loading_widget and self._loading_widget.parent:
            await self._remove_loading_widget()

    async def on_approval_app_approval_modify(
        self, message: ApprovalApp.ApprovalModify
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            # Engine re-validates and re-dispatches with the edited args; the
            # user already approved the modified form, so no re-prompt.
            self._pending_approval.set_result((
                ApprovalResponse.MODIFY,
                None,
                message.modified_args,
            ))

        if self._loading_widget and self._loading_widget.parent:
            await self._remove_loading_widget()

    async def on_question_app_answered(self, message: QuestionApp.Answered) -> None:
        if self._pending_question and not self._pending_question.done():
            result = AskUserQuestionResult(answers=message.answers, cancelled=False)
            self._pending_question.set_result(result)

    async def on_question_app_cancelled(self, message: QuestionApp.Cancelled) -> None:
        if self._pending_question and not self._pending_question.done():
            result = AskUserQuestionResult(answers=[], cancelled=True)
            self._pending_question.set_result(result)

    def on_chat_text_area_feedback_key_pressed(
        self, message: ChatTextArea.FeedbackKeyPressed
    ) -> None:
        self._feedback_bar.handle_feedback_key(message.rating)

    def on_chat_text_area_non_feedback_key_pressed(
        self, message: ChatTextArea.NonFeedbackKeyPressed
    ) -> None:
        self._feedback_bar.hide()

    def on_feedback_bar_feedback_given(
        self, message: FeedbackBar.FeedbackGiven
    ) -> None:
        self.agent_loop.telemetry_client.send_user_rating_feedback(
            rating=message.rating, model=self.config.active_model
        )

    async def _remove_loading_widget(self) -> None:
        if self._loading_widget and self._loading_widget.parent:
            await self._loading_widget.remove()
            self._loading_widget = None

    async def _resolve_turn_images(
        self, payload: PathPromptPayload, prebuilt: list[ImageAttachment] | None
    ) -> list[ImageAttachment] | None:
        if prebuilt is not None:
            return prebuilt
        return await self._prepare_images_or_abort(payload)

    async def _prepare_images_or_abort(
        self, payload: PathPromptPayload
    ) -> list[ImageAttachment] | None:
        result = await self._build_image_attachments(payload)
        if isinstance(result, _ImageAttachmentRejection):
            await self._remove_loading_widget()
            if result.no_vision:
                await self._mount_and_scroll(
                    ErrorMessage(result.message, show_border=False)
                )
            else:
                await self._mount_and_scroll(
                    ErrorMessage(result.message, collapsed=self._tools_collapsed)
                )
            return None
        return result

    async def _build_image_attachments(
        self, payload: PathPromptPayload
    ) -> list[ImageAttachment] | _ImageAttachmentRejection:
        image_resources = extract_image_resources(payload)
        if not image_resources:
            return []

        if len(image_resources) > MAX_IMAGES_PER_MESSAGE:
            return _ImageAttachmentRejection(
                f"Too many image attachments (got {len(image_resources)}, "
                f"max {MAX_IMAGES_PER_MESSAGE})."
            )

        try:
            active_model = self.agent_loop.config.get_active_model()
        except ValueError:
            active_model = None
        if active_model is not None and not active_model.supports_images:
            return _ImageAttachmentRejection(
                f"Model `{active_model.alias}` does not support images. "
                f"Switch with /model, remove the attachment, or ask me to enable the support for this model.",
                no_vision=True,
            )

        attachments: list[ImageAttachment] = []
        session_dir = self.agent_loop.session_logger.session_dir
        for resource in image_resources:
            result = self._snapshot_single_image(resource, session_dir)
            if isinstance(result, str):
                return _ImageAttachmentRejection(result)
            attachments.append(result)
        return attachments

    def _snapshot_single_image(
        self, resource: PathResource, session_dir: Path | None
    ) -> ImageAttachment | str:
        try:
            size = resource.path.stat().st_size
        except OSError as e:
            return f"Cannot read image {resource.alias}: {e}"
        if size > MAX_IMAGE_BYTES:
            return (
                f"Image `{resource.alias}` is "
                f"{size / (1024 * 1024):.1f} MB; max is "
                f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB."
            )
        try:
            return snapshot_image(
                resource.path, alias=resource.alias, session_dir=session_dir
            )
        except ImageSnapshotError as e:
            return f"Failed to attach image {resource.alias}: {e}"

    async def on_config_app_open_model_picker(
        self, _message: ConfigApp.OpenModelPicker
    ) -> None:
        config_app = self.query_one(ConfigApp)
        changes = config_app.convert_changes_for_save()
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        await self._switch_to_input_app()
        await self._switch_to_model_picker_app()

    async def on_config_app_open_judge_model_picker(
        self, _message: ConfigApp.OpenJudgeModelPicker
    ) -> None:
        config_app = self.query_one(ConfigApp)
        changes = config_app.convert_changes_for_save()
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        await self._switch_to_input_app()
        await self._switch_to_model_picker_app(target="judge")

    async def on_config_app_open_subagent_model_picker(
        self, _message: ConfigApp.OpenSubagentModelPicker
    ) -> None:
        config_app = self.query_one(ConfigApp)
        changes = config_app.convert_changes_for_save()
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        await self._switch_to_input_app()
        await self._switch_to_model_picker_app(target="subagent")

    async def on_config_app_open_grunt_model_picker(
        self, _message: ConfigApp.OpenGruntModelPicker
    ) -> None:
        config_app = self.query_one(ConfigApp)
        changes = config_app.convert_changes_for_save()
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        await self._switch_to_input_app()
        await self._switch_to_model_picker_app(target="grunt")

    async def on_config_app_open_thinking_picker(
        self, _message: ConfigApp.OpenThinkingPicker
    ) -> None:
        config_app = self.query_one(ConfigApp)
        changes = config_app.convert_changes_for_save()
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        await self._switch_to_input_app()
        await self._switch_to_thinking_picker_app()

    async def _ensure_loading_widget(
        self, status: str = DEFAULT_LOADING_STATUS, *, show_hint: bool = True
    ) -> None:
        if self._loading_widget and self._loading_widget.parent:
            self._loading_widget.set_status(status)
            return

        try:
            loading_area = self._loading_area
        except Exception:
            return
        loading = LoadingWidget(status=status, show_hint=show_hint)
        self._loading_widget = loading
        await loading_area.mount(loading)

    async def on_config_app_config_closed(
        self, message: ConfigApp.ConfigClosed
    ) -> None:
        await self._handle_config_settings_closed(message.changes)
        await self._switch_to_input_app()

    async def on_voice_app_config_closed(self, message: VoiceApp.ConfigClosed) -> None:
        await self._handle_voice_settings_closed(message.changes)
        await self._switch_to_input_app()

    async def _handle_config_settings_closed(self, changes: dict[str, Any]) -> None:
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        else:
            await self._mount_and_scroll(
                UserCommandMessage("Configuration closed (no changes saved).")
            )

    async def _handle_voice_settings_closed(
        self, changes: dict[str, str | bool]
    ) -> None:
        if not changes:
            await self._mount_and_scroll(
                UserCommandMessage("Voice settings closed (no changes saved).")
            )
            return

        if "voice_mode_enabled" in changes:
            current = self._voice_manager.is_enabled
            desired = changes["voice_mode_enabled"]
            if current != desired:
                self._voice_manager.toggle_voice_mode()
                self.agent_loop.telemetry_client.send_telemetry_event(
                    "vibe.voice_mode_toggled", {"enabled": desired}
                )
                self.agent_loop.refresh_config()
                if desired:
                    await self._mount_and_scroll(
                        UserCommandMessage(
                            "Voice mode enabled. Press **Ctrl+R** to start recording."
                        )
                    )
                else:
                    await self._mount_and_scroll(
                        UserCommandMessage("Voice mode disabled.")
                    )

        non_voice_changes = {
            k: v for k, v in changes.items() if k != "voice_mode_enabled"
        }
        if non_voice_changes:
            VibeConfig.save_updates(non_voice_changes)
            self.agent_loop.refresh_config()
            self._narrator_manager.sync()

    async def on_model_picker_app_model_selected(
        self, message: ModelPickerApp.ModelSelected
    ) -> None:
        # Mid-turn rate-limit switch: hand the chosen alias back to the agent
        # loop (which rebuilds the backend and retries). Transient override —
        # do NOT persist active_model; _rate_limit_callback restores the UI.
        if (
            self._pending_model_switch is not None
            and not self._pending_model_switch.done()
        ):
            self._pending_model_switch.set_result(message.alias)
            subagent_model = str(self.config.subagent_model or message.alias)
            self._set_model_badges(message.alias, subagent_model)
            return
        target = getattr(self, "_model_picker_target", "active")
        self._model_picker_target = "active"
        discovered = getattr(self, "_discovered_models", {})
        updates: dict[str, Any] = {}
        if message.alias in discovered:
            # A live-discovered model has no config block yet — persist it (and
            # its auto-detected provider) so it stays resolvable after reload.
            from vibe.core.llm.model_discovery import build_persisted_updates

            updates = build_persisted_updates(self.config, discovered[message.alias])
        if target == "judge":
            updates["safety_judge"] = {"model": message.alias}
        elif target == "subagent":
            updates["subagent_model"] = message.alias
        elif target == "grunt":
            updates["grunt_model"] = message.alias
        else:
            updates["active_model"] = message.alias
        self._discovered_models = {}
        try:
            # dump_config validates the merged config before writing, so a
            # failure here leaves the on-disk config untouched.
            VibeConfig.save_updates(updates)
        except Exception as exc:
            self.notify(
                f"Could not switch model: {exc}", severity="error", markup=False
            )
            await self._switch_to_input_app()
            return
        await self._reload_config()
        await self._switch_to_input_app()

    async def on_model_picker_app_cancelled(
        self, _event: ModelPickerApp.Cancelled
    ) -> None:
        # Cancel during a rate-limit switch: resolve with None so the agent loop
        # surfaces the error; _rate_limit_callback restores the UI.
        if (
            self._pending_model_switch is not None
            and not self._pending_model_switch.done()
        ):
            self._pending_model_switch.set_result(None)
            return
        self._model_picker_target = "active"
        await self._switch_to_input_app()

    async def on_thinking_picker_app_thinking_selected(
        self, message: ThinkingPickerApp.ThinkingSelected
    ) -> None:
        self.config.set_thinking(message.level)
        await self._reload_config()
        await self._switch_to_input_app()

    async def on_thinking_picker_app_cancelled(
        self, _event: ThinkingPickerApp.Cancelled
    ) -> None:
        await self._switch_to_input_app()

    async def on_effort_picker_app_effort_selected(
        self, message: EffortPickerApp.EffortSelected
    ) -> None:
        self.config.set_effort_mode(message.level)
        await self._reload_config()
        await self._switch_to_input_app()

    async def on_effort_picker_app_cancelled(
        self, _event: EffortPickerApp.Cancelled
    ) -> None:
        await self._switch_to_input_app()

    async def on_theme_picker_app_theme_previewed(
        self, message: ThemePickerApp.ThemePreviewed
    ) -> None:
        self._apply_theme(message.theme)
        await self._restyle_diff_widgets()

    async def on_theme_picker_app_theme_selected(
        self, message: ThemePickerApp.ThemeSelected
    ) -> None:
        self._apply_theme(message.theme)
        self.config.theme = message.theme
        VibeConfig.save_updates({"theme": message.theme})
        await self._restyle_diff_widgets()
        await self._switch_to_input_app()

    async def on_theme_picker_app_cancelled(
        self, message: ThemePickerApp.Cancelled
    ) -> None:
        self._apply_theme(message.original_theme)
        await self._restyle_diff_widgets()
        await self._switch_to_input_app()

    async def _restyle_diff_widgets(self) -> None:
        # Diff content bakes in ANSI-vs-truecolor styling, so it must be rebuilt.
        for widget in self.query(EditResultWidget):
            await widget.recompose()
        for widget in self.query(EditApprovalWidget):
            await widget.recompose()

    async def on_mcpapp_mcpclosed(self, _message: MCPApp.MCPClosed) -> None:
        await self._mount_and_scroll(UserCommandMessage("MCP servers closed."))
        await self._switch_to_input_app()

    async def on_mcpapp_mcptoggled(self, message: MCPApp.MCPToggled) -> None:
        from vibe.cli.textual_ui.widgets.mcp_app import MCPApp, MCPSourceKind

        persist_mcp_toggle(
            self.agent_loop.config,
            name=message.name,
            is_connector=message.kind == MCPSourceKind.CONNECTOR,
            disabled=message.disabled,
            tool_name=message.tool_name,
        )
        self.agent_loop.refresh_config()
        self.query_one(MCPApp).refresh_index()
        self._refresh_banner()

    async def on_mcpapp_connector_auth_requested(
        self, message: MCPApp.ConnectorAuthRequested
    ) -> None:
        from vibe.cli.textual_ui.widgets.connector_auth_app import ConnectorAuthApp

        await self._switch_to_input_app()
        await self._switch_from_input(
            ConnectorAuthApp(
                connector_name=message.connector_name,
                connector_registry=message.connector_registry,
                tool_manager=message.tool_manager,
            )
        )

    async def on_mcpapp_mcp_oauth_login_requested(
        self, message: MCPApp.MCPOAuthLoginRequested
    ) -> None:
        await self._switch_to_input_app()
        await self._mcp_login(alias=message.server_name)

    async def on_mcpaddapp_mcpaddclosed(self, message: MCPAddApp.MCPAddClosed) -> None:
        await self._switch_to_input_app()
        if message.error:
            await self._mount_and_scroll(
                ErrorMessage(f"Failed to add MCP server: {message.error}")
            )
            return
        if message.added:
            await self.agent_loop.refresh_system_prompt()
            self._refresh_banner()
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"Added MCP server {message.name!r}. Run /mcp to browse, "
                    f"or /mcp login {message.name} for OAuth."
                )
            )

    async def on_connector_auth_app_connector_auth_closed(
        self, message: ConnectorAuthApp.ConnectorAuthClosed
    ) -> None:
        if message.refreshed:
            await self.agent_loop.refresh_system_prompt()
            self._refresh_banner()
        await self._switch_to_input_app()
        await self._show_mcp(cmd_args=message.connector_name)

    async def on_proxy_setup_app_proxy_setup_closed(
        self, message: ProxySetupApp.ProxySetupClosed
    ) -> None:
        if message.error:
            await self._mount_and_scroll(
                ErrorMessage(f"Failed to save proxy settings: {message.error}")
            )
        elif message.saved:
            await self._mount_and_scroll(
                UserCommandMessage(
                    "Proxy settings saved. Restart the CLI for changes to take effect."
                )
            )
        else:
            await self._mount_and_scroll(UserCommandMessage("Proxy setup cancelled."))

        await self._switch_to_input_app()

    async def on_provider_login_app_provider_login_closed(
        self, message: ProviderLoginApp.ProviderLoginClosed
    ) -> None:
        await self._switch_to_input_app()
        if message.error:
            await self._mount_and_scroll(
                ErrorMessage(message.error, collapsed=self._tools_collapsed)
            )
            return
        if not message.authenticated:
            await self._mount_and_scroll(
                UserCommandMessage("Provider login cancelled.")
            )
            return

        self.agent_loop.refresh_config()
        await self.agent_loop.refresh_system_prompt()
        self._refresh_banner()
        await self._mount_and_scroll(
            UserCommandMessage(f"Logged in to {message.provider_name}.")
        )

    async def on_compact_message_completed(
        self, message: CompactMessage.Completed
    ) -> None:
        children = list(self._messages_area.children)

        try:
            compact_index = children.index(message.compact_widget)
        except ValueError:
            return

        if compact_index == 0:
            return

        with self.batch_update():
            for widget in children[:compact_index]:
                await widget.remove()

    async def _handle_command(self, user_input: str) -> bool:
        if resolved := self.commands.parse_command(user_input):
            cmd_name, command, cmd_args = resolved
            self.agent_loop.telemetry_client.send_slash_command_used(
                cmd_name, "builtin"
            )
            command_text = user_input.strip()
            display = (
                command_text.removeprefix("/")
                if command_text.startswith("/")
                else cmd_name
            )
            await self._mount_and_scroll(SlashCommandMessage(display))
            handler = getattr(self, command.handler)
            if asyncio.iscoroutinefunction(handler):
                await handler(cmd_args=cmd_args, _cmd_name=cmd_name)
            else:
                handler(cmd_args=cmd_args, _cmd_name=cmd_name)
            return True
        return False

    def _get_skill_entries(self) -> list[tuple[str, str]]:
        if not self.agent_loop:
            return []
        return [
            (f"/{name}", info.description)
            for name, info in self.agent_loop.skill_manager.available_skills.items()
            if info.user_invocable
        ]

    def _expand_skill(self, user_input: str) -> Skill | None:
        if not self.agent_loop:
            return None
        skill = self.agent_loop.skill_manager.parse_skill_command(user_input)
        if skill is None:
            return None
        return Skill(
            expanded_prompt=SkillManager.build_skill_prompt(user_input, skill),
            name=skill.name,
        )

    def _send_skill_telemetry(self, name: str | None) -> None:
        if name is None:
            return
        self.agent_loop.telemetry_client.send_slash_command_used(name, "skill")

    def _send_at_mention_telemetry(
        self, payload: PathPromptPayload, message_id: str
    ) -> None:
        if not payload.all_resources:
            return
        context_types: dict[str, int] = {}
        for r in payload.all_resources:
            context_types[r.kind] = context_types.get(r.kind, 0) + 1
        file_ext_counts: dict[str, int] = {}
        for r in payload.all_resources:
            if r.kind == "file" and r.path.suffix:
                file_ext_counts[r.path.suffix] = (
                    file_ext_counts.get(r.path.suffix, 0) + 1
                )
        self.agent_loop.telemetry_client.send_at_mention_inserted(
            nb_mentions=len(payload.all_resources),
            context_types=context_types,
            file_extensions=file_ext_counts or None,
            message_id=message_id,
        )

    @staticmethod
    async def _bash_read_stream(
        stream: asyncio.StreamReader | None,
        parts: list[str],
        bash_msg: BashOutputMessage,
    ) -> None:
        if not stream:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if not text:
                continue
            parts.append(text)
            await bash_msg.append_output(text)
        final_text = decoder.decode(b"", final=True)
        if not final_text:
            return
        parts.append(final_text)
        await bash_msg.append_output(final_text)

    @staticmethod
    async def _kill_running_process(proc: asyncio.subprocess.Process | None) -> None:
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()

    async def _handle_bash_command(
        self,
        command: str,
        *,
        existing_widget: BashOutputMessage | None = None,
        start_drain_on_finish: bool = True,
    ) -> None:
        try:
            await self._handle_bash_command_inner(
                command, existing_widget=existing_widget
            )
        finally:
            current = asyncio.current_task()
            if self._bash_task is current:
                self._bash_task = None
            self._queue.notify_busy_changed()
            if start_drain_on_finish:
                self._queue.start_drain_if_needed()

    async def _handle_bash_command_inner(
        self, command: str, *, existing_widget: BashOutputMessage | None = None
    ) -> None:
        if not command:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No command provided after '!'", collapsed=self._tools_collapsed
                )
            )
            return

        if existing_widget is not None:
            bash_msg = existing_widget
        else:
            bash_msg = BashOutputMessage(command, str(safe_cwd()), pending=True)
            await self._mount_and_scroll(bash_msg)
        await self._ensure_loading_widget("Running command")
        bash_loading_widget = self._loading_widget

        proc: asyncio.subprocess.Process | None = None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        try:
            proc = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        self._bash_read_stream(proc.stdout, stdout_parts, bash_msg),
                        self._bash_read_stream(proc.stderr, stderr_parts, bash_msg),
                        proc.wait(),
                    ),
                    timeout=30,
                )
            except TimeoutError:
                await self._kill_running_process(proc)
                stdout = "".join(stdout_parts)
                stderr = "".join(stderr_parts)
                await bash_msg.finish(1)
                await self._mount_and_scroll(
                    ErrorMessage(
                        "Command timed out after 30 seconds",
                        collapsed=self._tools_collapsed,
                    )
                )
                await self.agent_loop.inject_user_context(
                    self._format_manual_command_context(
                        command=command,
                        cwd=str(safe_cwd()),
                        stdout=stdout,
                        stderr=stderr,
                        status="timed out after 30 seconds",
                    )
                )
                return

            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            exit_code = proc.returncode or 0
            await bash_msg.finish(exit_code)
            await self.agent_loop.inject_user_context(
                self._format_manual_command_context(
                    command=command,
                    cwd=str(safe_cwd()),
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                )
            )
        except asyncio.CancelledError:
            await self._kill_running_process(proc)
            await bash_msg.finish(1, interrupted=True)
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            await self.agent_loop.inject_user_context(
                self._format_manual_command_context(
                    command=command,
                    cwd=str(safe_cwd()),
                    stdout=stdout,
                    stderr=stderr,
                    status="interrupted by user",
                )
            )
        except Exception as e:
            await self._kill_running_process(proc)
            await bash_msg.finish(1)
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            await self._mount_and_scroll(
                ErrorMessage(f"Command failed: {e}", collapsed=self._tools_collapsed)
            )
            await self.agent_loop.inject_user_context(
                self._format_manual_command_context(
                    command=command,
                    cwd=str(safe_cwd()),
                    stdout=stdout,
                    stderr=stderr,
                    status=f"failed before completion: {e}",
                )
            )
        finally:
            if self._loading_widget is bash_loading_widget:
                await self._remove_loading_widget()

    def _get_bash_max_output_bytes(self) -> int:
        from vibe.core.tools.builtins.bash import BashToolConfig

        config = self.agent_loop.tool_manager.get_tool_config("bash")
        if isinstance(config, BashToolConfig):
            return config.max_output_bytes
        return BashToolConfig().max_output_bytes

    @staticmethod
    def _cap_output(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n... [truncated]"

    def _format_manual_command_context(
        self,
        *,
        command: str,
        cwd: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
        status: str | None = None,
    ) -> str:
        limit = self._get_bash_max_output_bytes()
        stdout = self._cap_output(stdout, limit)
        stderr = self._cap_output(stderr, limit)

        sections = [
            "Manual `!` command result from the user. Use this as context only.",
            f"Command: `{command}`",
            f"Working directory: `{cwd}`",
        ]

        if status is not None:
            sections.append(f"Status: {status}")

        if exit_code is not None:
            sections.append(f"Exit code: {exit_code}")

        if stdout:
            sections.append(f"Stdout:\n```text\n{stdout.rstrip()}\n```")

        if stderr:
            sections.append(f"Stderr:\n```text\n{stderr.rstrip()}\n```")

        if not stdout and not stderr:
            sections.append("Output:\n```text\n(no output)\n```")

        return "\n\n".join(sections)

    async def _handle_user_message(
        self, message: str, *, title_source: str | None = None
    ) -> None:
        self._consecutive_auto_continues = 0
        prompt_payload = build_path_prompt_payload(message, base_dir=safe_cwd())
        images = await self._prepare_images_or_abort(prompt_payload)
        if images is None:
            input_widget = self.query_one(ChatInputContainer)
            if not input_widget.value:
                input_widget.value = message
            return

        # message_index is where the user message will land in agent_loop.messages
        # (checkpoint is created in agent_loop.act())
        message_index = len(self.agent_loop.messages)
        user_message = UserMessage(
            message, message_index=message_index, images=images or None
        )

        messages_area = self._cached_messages_area or self.query_one("#messages")
        last_child = messages_area.children[-1] if messages_area.children else None
        if isinstance(last_child, UserMessage):
            last_child.set_show_separator(False)
            user_message.set_follows_previous(True)

        await self._mount_and_scroll(user_message)
        if self._feedback_bar_manager.should_show(self.agent_loop):
            self._feedback_bar.show()
            self._feedback_bar_manager.record_feedback_asked(self.agent_loop)

        if not self._agent_running:
            await self._remove_loading_widget()
            self._agent_task = asyncio.create_task(
                self._handle_agent_loop_turn(
                    message,
                    title_source=title_source,
                    prebuilt_images=images,
                    prebuilt_payload=prompt_payload,
                )
            )
            self._queue.notify_busy_changed()

    def _reset_ui_state(self) -> None:
        self._windowing.reset()
        self._tool_call_map = None
        self._history_widget_indices = WeakKeyDictionary()

    async def _resume_history_from_messages(self) -> None:
        messages_area = self._messages_area
        if not should_resume_history(list(messages_area.children)):
            return

        history_messages = non_system_history_messages(self.agent_loop.messages)
        if (
            plan := create_resume_plan(history_messages, HISTORY_RESUME_TAIL_MESSAGES)
        ) is None:
            return
        await self._mount_history_batch(
            plan.tail_messages,
            messages_area,
            plan.tool_call_map,
            start_index=plan.tail_start_index,
        )
        self.call_after_refresh(self._chat_widget.anchor)
        self._tool_call_map = plan.tool_call_map
        self._windowing.set_backfill(plan.backfill_messages)
        await self._load_more.set_visible(
            messages_area,
            visible=self._windowing.has_backfill,
            remaining=self._windowing.remaining,
        )

    async def _mount_history_batch(
        self,
        batch: list[LLMMessage],
        messages_area: Widget,
        tool_call_map: dict[str, str],
        *,
        start_index: int,
        before: Widget | int | None = None,
        after: Widget | None = None,
    ) -> None:
        widgets = build_history_widgets(
            batch=batch,
            tool_call_map=tool_call_map,
            start_index=start_index,
            history_widget_indices=self._history_widget_indices,
        )

        with self.batch_update():
            if not widgets:
                return
            if before is not None:
                await messages_area.mount_all(widgets, before=before)
            elif after is not None:
                await messages_area.mount_all(widgets, after=after)
            else:
                await messages_area.mount_all(widgets)

        for widget in widgets:
            if isinstance(widget, StreamingMessageBase):
                await widget.write_initial_content()

    def _is_tool_enabled_in_main_agent(self, tool: str) -> bool:
        return tool in self.agent_loop.tool_manager.available_tools

    async def _wait_for_typing_pause(self) -> None:
        try:
            text_area = self.query_one(ChatTextArea)
        except Exception:
            return

        debounce_s = _resolve_typing_debounce_s()
        if text_area.time_since_last_keystroke() >= debounce_s:
            return

        if self._loading_widget:
            self._loading_widget.show_debounce_hint()

        try:
            while True:
                elapsed = text_area.time_since_last_keystroke()
                if elapsed >= debounce_s:
                    return
                await asyncio.sleep(debounce_s - elapsed)
        finally:
            if self._loading_widget:
                self._loading_widget.hide_debounce_hint()

    async def _approval_callback(
        self,
        tool: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
        judge_note: str | None = None,
    ) -> tuple[ApprovalResponse, str | None, dict[str, Any] | None]:
        if self.agent_loop and self.agent_loop.config.bypass_tool_permissions:
            if self._is_tool_enabled_in_main_agent(tool):
                return (ApprovalResponse.YES, None, None)

        # The judge note arrives via the callback argument (threaded from
        # _ask_approval) rather than the loop-local pending_judge_deferral
        # attribute: that attribute lives on whichever loop ran the judge
        # (often a workflow/task subagent's loop), which is invisible to the
        # host. The callback argument is the only channel that crosses loops.
        # Fall back to the host's own attribute only for the host's direct
        # calls where the note wasn't passed.
        if judge_note is None and self.agent_loop:
            judge_note = self.agent_loop.pending_judge_deferral
        async with self._user_interaction_lock:
            await self._wait_for_typing_pause()
            self._pending_approval = asyncio.Future()
            self._terminal_notifier.notify(NotificationContext.ACTION_REQUIRED)
            try:
                with paused_timer(self._loading_widget):
                    await self._switch_to_approval_app(
                        tool, args, required_permissions, judge_note=judge_note
                    )
                    result = await self._pending_approval
                return result
            finally:
                self._pending_approval = None
                await self._switch_to_input_app()

    async def _user_input_callback(self, args: BaseModel) -> BaseModel:
        question_args = cast(AskUserQuestionArgs, args)

        async with self._user_interaction_lock:
            await self._wait_for_typing_pause()
            self._pending_question = asyncio.Future()
            self._terminal_notifier.notify(NotificationContext.ACTION_REQUIRED)
            try:
                with paused_timer(self._loading_widget):
                    await self._switch_to_question_app(question_args)
                    result = await self._pending_question
                return result
            finally:
                self._pending_question = None
                await self._switch_to_input_app()

    async def _rate_limit_callback(
        self, provider: str, model: str, candidates: list[str]
    ) -> str | None:
        async with self._user_interaction_lock:
            await self._wait_for_typing_pause()
            self._pending_model_switch = asyncio.Future()
            self._terminal_notifier.notify(NotificationContext.ACTION_REQUIRED)
            try:
                with paused_timer(self._loading_widget):
                    await self._mount_and_scroll(
                        WarningMessage(
                            f"Rate limited on {model!r} ({provider}). "
                            "Pick a model to switch to, or press Esc to stop."
                        )
                    )
                    await self._switch_to_rate_limit_picker_app(model, candidates)
                    chosen = await self._pending_model_switch
                return chosen
            finally:
                self._pending_model_switch = None
                await self._switch_to_input_app()

    async def _switch_to_rate_limit_picker_app(
        self, current_model: str, candidates: list[str]
    ) -> None:
        display_names = {m.alias: m.name for m in self.config.available_models}
        providers = {m.alias: m.provider for m in self.config.available_models}
        provider_order = {
            provider.name: index for index, provider in enumerate(self.config.providers)
        }
        candidate_index = {alias: index for index, alias in enumerate(candidates)}
        ordered_candidates = sorted(
            candidates,
            key=lambda alias: (
                provider_order.get(providers.get(alias, ""), len(provider_order)),
                candidate_index[alias],
            ),
        )
        await self._switch_from_input(
            ModelPickerApp(
                model_aliases=ordered_candidates,
                current_model=current_model,
                display_names=display_names,
                providers=providers,
            )
        )

    async def _handle_turn_error(self, *, cancelled: bool = False) -> None:
        if self._loading_widget and self._loading_widget.parent:
            await self._loading_widget.remove()
        if self.event_handler:
            self.event_handler.stop_current_tool_call(
                success=False, cancelled=cancelled
            )

    async def _handle_agent_loop_init(self) -> None:
        show_init_spinner = not self.agent_loop.is_initialized
        if show_init_spinner:
            await self._ensure_loading_widget("Initializing", show_hint=False)
        await self.agent_loop.wait_until_ready()
        if show_init_spinner:
            await self._remove_loading_widget()
            self._refresh_banner()

    async def _handle_agent_loop_events(
        self, events: AsyncGenerator[BaseEvent]
    ) -> None:
        async for event in events:
            self._narrator_manager.on_turn_event(event)
            if isinstance(event, WaitingForInputEvent):
                await self._remove_loading_widget()
            elif isinstance(event, HookStartEvent):
                await self._ensure_loading_widget(f"Running hook {event.hook_name}")
            if self.event_handler:
                await self.event_handler.handle_event(
                    event, loading_widget=self._loading_widget
                )

    async def _handle_agent_loop_turn(
        self,
        prompt: str,
        *,
        title_source: str | None = None,
        prebuilt_images: list[ImageAttachment] | None = None,
        prebuilt_payload: PathPromptPayload | None = None,
    ) -> None:
        self._agent_running = True

        await self._remove_loading_widget()

        try:
            await self._handle_agent_loop_init()
            await self._ensure_loading_widget()
            message_id = str(uuid4())
            prompt_payload = prebuilt_payload or build_path_prompt_payload(
                prompt, base_dir=safe_cwd()
            )
            self._send_at_mention_telemetry(prompt_payload, message_id)
            images = await self._resolve_turn_images(prompt_payload, prebuilt_images)
            if images is None:
                return
            rendered_prompt = render_path_prompt_from_payload(
                prompt_payload, skip_images=True
            )
            auto_title: str | None = None
            if self.agent_loop.session_logger.needs_initial_auto_title():
                auto_title = (
                    format_session_title(
                        build_title_segments(
                            title_source or prompt, base_dir=safe_cwd()
                        )
                    )
                    or None
                )
            self._narrator_manager.cancel()
            self._narrator_manager.on_turn_start(rendered_prompt)
            async with aclosing(
                self.agent_loop.act(
                    rendered_prompt,
                    client_message_id=message_id,
                    auto_title=auto_title,
                    images=images or None,
                )
            ) as events:
                await self._handle_agent_loop_events(events)
        except asyncio.CancelledError:
            await self._handle_turn_error(cancelled=True)
            self._narrator_manager.on_turn_cancel()
            raise
        except Exception as e:
            await self._handle_turn_error()

            # _watch_init_completion already rendered the fatal startup error
            # and told the user to exit -- don't duplicate the message.
            if self._fatal_init_error:
                return

            message = self._resolve_turn_error_message(e)
            self._narrator_manager.on_turn_error(message)

            await self._mount_and_scroll(
                ErrorMessage(message, collapsed=self._tools_collapsed)
            )
        finally:
            self._narrator_manager.on_turn_end()
            self._agent_running = False
            self._auto_continue_active = False
            self._interrupt_requested = False
            self._agent_task = None
            if self._loading_widget:
                await self._loading_widget.remove()
            self._loading_widget = None
            if self.event_handler:
                await self.event_handler.finalize_streaming()
                self.event_handler.escalate_unresolved_errors()
            self._queue.notify_busy_changed()
            self._queue.start_drain_if_needed()
            await self._refresh_windowing_from_history()
            self._terminal_notifier.notify(NotificationContext.COMPLETE)

    def _resolve_turn_error_message(self, e: Exception) -> str:
        if isinstance(e, RateLimitError):
            return self._rate_limit_message(e)
        if isinstance(e, ContentFilterError):
            return self._content_filter_message(e)
        if isinstance(e, ContextTooLongError):
            return self._context_too_long_message()
        if isinstance(e, RefusalError):
            return self._refusal_message(e)
        return str(e)

    def _rate_limit_message(self, e: RateLimitError) -> str:
        target = f"{e.model} ({e.provider})"
        # Upsell only on a Mistral 429; match by backend, not the name "mistral".
        mistral_provider_names = {
            p.name for p in self.config.providers if p.backend == Backend.MISTRAL
        }
        upgrade_to_pro = (
            e.provider in mistral_provider_names
            and self._plan_info
            and (
                self._plan_info.plan_type
                in {WhoAmIPlanType.API, WhoAmIPlanType.UNAUTHORIZED}
                or self._plan_info.is_free_mistral_code_plan()
            )
        )
        if upgrade_to_pro:
            base = (
                f"Rate limits exceeded for {target}. Please wait a moment before "
                "trying again, or upgrade to Pro for higher rate limits and "
                "uninterrupted access."
            )
        else:
            base = (
                f"Rate limits exceeded for {target}. "
                "Please wait a moment before trying again."
            )
        if e.failover_hint:
            base = f"{base}\n\n{e.failover_hint}"
        return base

    def _content_filter_message(self, e: ContentFilterError) -> str:
        base = (
            f"The request to {e.model} ({e.provider}) was blocked by the "
            "content filter."
        )
        if e.failover_hint:
            base = f"{base}\n\n{e.failover_hint}"
        return base

    def _context_too_long_message(self) -> str:
        return (
            "The conversation context exceeds the model's maximum limit. "
            "The last messages and output of agent actions went above the allowed size.\n\n"
            "To recover:\n"
            "1. Use /rewind to undo recent messages and tool outputs\n"
            "2. Then use /compact to summarize the remaining conversation\n\n"
            "This will free up context space so you can continue working."
        )

    def _refusal_message(self, e: RefusalError) -> str:
        lead = "The model declined to respond and stopped early (refusal)."
        if e.category:
            lead += f"\nCategory: {e.category}."
        detail = e.explanation or (
            "This can happen with certain prompts or content. "
            "Try rephrasing your request or starting a new conversation."
        )
        return f"{lead}\n\n{detail}"

    async def _teleport_command(self, **kwargs: Any) -> None:
        await self._handle_teleport_command(show_message=False)

    async def _handle_teleport_command(
        self, value: str | None = None, show_message: bool = True
    ) -> None:
        has_history = any(msg.role != Role.SYSTEM for msg in self.agent_loop.messages)
        if not value:
            if show_message:
                await self._mount_and_scroll(SlashCommandMessage("teleport"))
            if not has_history:
                send_teleport_early_failure_telemetry(
                    self.agent_loop.telemetry_client,
                    stage="no_history",
                    error_class="TeleportNoHistoryError",
                    nb_session_messages=len(self.agent_loop.messages[1:]),
                )
                await self._mount_and_scroll(
                    ErrorMessage(
                        "No conversation history to teleport.",
                        collapsed=self._tools_collapsed,
                    )
                )
                return
        elif show_message:
            await self._mount_and_scroll(TeleportUserMessage(value))
        self.run_worker(self._teleport(value), exclusive=False)

    async def _teleport(self, prompt: str | None = None) -> None:
        loading = LoadingWidget()
        await self._loading_area.mount(loading)

        teleport_msg = TeleportMessage()
        await self._mount_and_scroll(teleport_msg)

        from vibe.core.agent_loop import TeleportError

        try:
            gen = self.agent_loop.teleport_to_vibe_code(prompt)
            async for event in gen:
                match event:
                    case TeleportCheckingGitEvent():
                        teleport_msg.set_status("Preparing workspace...")
                    case TeleportPushRequiredEvent(
                        unpushed_count=count, branch_not_pushed=branch_not_pushed
                    ):
                        await loading.remove()
                        response = await self._ask_push_approval(
                            count, branch_not_pushed
                        )
                        await self._loading_area.mount(loading)
                        teleport_msg.set_status("Teleporting...")
                        next_event = await gen.asend(response)
                        if isinstance(next_event, TeleportPushingEvent):
                            teleport_msg.set_status("Syncing with remote...")
                    case TeleportPushingEvent():
                        teleport_msg.set_status("Syncing with remote...")
                    case TeleportStartingWorkflowEvent():
                        teleport_msg.set_status("Teleporting...")
                    case TeleportCompleteEvent(url=url):
                        teleport_msg.set_complete(url)
        except TeleportError as e:
            await teleport_msg.remove()
            await self._mount_and_scroll(
                ErrorMessage(str(e), collapsed=self._tools_collapsed)
            )
        finally:
            if loading.parent:
                await loading.remove()

    async def _ask_push_approval(
        self, count: int, branch_not_pushed: bool
    ) -> TeleportPushResponseEvent:
        if branch_not_pushed:
            question = "Your branch doesn't exist on remote. Push to continue?"
        else:
            word = f"commit{'s' if count != 1 else ''}"
            question = f"You have {count} unpushed {word}. Push to continue?"
        push_label = "Push and continue"
        result = await self._user_input_callback(
            AskUserQuestionArgs(
                questions=[
                    Question(
                        question=question,
                        header="Push",
                        options=[Choice(label=push_label), Choice(label="Cancel")],
                        hide_other=True,
                    )
                ]
            )
        )
        ok = (
            isinstance(result, AskUserQuestionResult)
            and not result.cancelled
            and bool(result.answers)
            and result.answers[0].answer == push_label
        )
        return TeleportPushResponseEvent(approved=ok)

    async def _interrupt_agent_loop(self) -> None:
        if not self._agent_running or self._interrupt_requested:
            return

        self._interrupt_requested = True

        if self._pending_approval and not self._pending_approval.done():
            feedback = str(
                get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
            )
            self._pending_approval.set_result((ApprovalResponse.NO, feedback, None))
        if self._pending_question and not self._pending_question.done():
            self._pending_question.set_result(
                AskUserQuestionResult(answers=[], cancelled=True)
            )

        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass

        if self.event_handler:
            self.event_handler.stop_current_tool_call(cancelled=True)
            self.event_handler.stop_current_compact()
            await self.event_handler.finalize_streaming()

        self._agent_running = False
        await self._loading_area.remove_children()
        self._loading_widget = None

        await self._mount_and_scroll(InterruptMessage())

        self._interrupt_requested = False

    async def _show_help(self, **kwargs: Any) -> None:
        help_text = self.commands.get_help_text()
        await self._mount_and_scroll(UserCommandMessage(help_text))

    def _get_last_assistant_message_text(self) -> str | None:
        for child in reversed(self._messages_area.children):
            if not isinstance(child, AssistantMessage):
                continue
            if not (content := child.get_content().strip()):
                continue
            return content
        return None

    async def _copy_last_agent_message(self, **kwargs: Any) -> None:
        if (content := self._get_last_assistant_message_text()) is None:
            self.notify(
                "No agent message available to copy", severity="warning", timeout=3
            )
            return

        copied_text = copy_text_to_clipboard(
            self, content, success_message="Last agent message copied to clipboard"
        )
        if copied_text is not None:
            self.agent_loop.telemetry_client.send_user_copied_text(copied_text)

    async def _paste_clipboard_image_command(self, **_kwargs: Any) -> None:
        await handle_clipboard_image_paste(self, notify_when_empty=True)

    async def _refresh_mcp_browser(self) -> str:
        await self.agent_loop.tool_manager.refresh_remote_tools_async()
        await self.agent_loop.refresh_system_prompt()
        self._refresh_banner()
        return "Refreshed."

    async def _dispatch_mcp_subcommand(self, cmd_args: str) -> bool:
        subcommand, _, remainder = cmd_args.strip().partition(" ")
        match subcommand:
            case "login":
                await self._mcp_login(alias=remainder.strip())
            case "logout":
                await self._mcp_logout(alias=remainder.strip())
            case "refresh":
                await self._mcp_refresh()
            case "add":
                await self._mcp_add()
            case "status":
                await self._show_mcp_status()
            case _:
                return False
        return True

    async def _show_mcp_status(self) -> None:
        await self.agent_loop.wait_until_ready()
        registry = self.agent_loop.mcp_registry
        statuses = registry.status() if registry is not None else {}
        if not statuses:
            await self._mount_and_scroll(
                UserCommandMessage("No MCP servers configured.")
            )
            return
        lines = ["### MCP auth status", ""]
        for alias, status in sorted(statuses.items()):
            lines.append(f"- `{alias}`: `{status.value}`")
        await self._mount_and_scroll(UserCommandMessage("\n".join(lines)))

    async def _show_mcp(self, cmd_args: str = "", **kwargs: Any) -> None:
        if await self._dispatch_mcp_subcommand(cmd_args):
            return

        mcp_servers = self.config.mcp_servers
        connector_registry = (
            self.agent_loop.connector_registry if self._connectors_enabled else None
        )
        has_connectors = (
            connector_registry is not None and connector_registry.connector_count > 0
        )
        if not mcp_servers and not has_connectors:
            await self._mount_and_scroll(
                UserCommandMessage("No MCP servers or connectors configured.")
            )
            return

        if self._current_bottom_app == BottomApp.MCP:
            return
        name = cmd_args.strip()
        connector_names = (
            connector_registry.get_connector_names() if connector_registry else []
        )
        if (
            name
            and not any(s.name == name for s in mcp_servers)
            and name not in connector_names
        ):
            all_names = [s.name for s in mcp_servers] + connector_names
            entity = "MCP server or connector" if has_connectors else "MCP server"
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Unknown {entity}: {name}. Known: " + ", ".join(all_names),
                    collapsed=self._tools_collapsed,
                )
            )
            return
        await self._mount_and_scroll(UserCommandMessage("MCP servers opened..."))
        from vibe.cli.textual_ui.widgets.mcp_app import MCPApp

        await self._switch_from_input(
            MCPApp(
                mcp_servers=mcp_servers,
                tool_manager=self.agent_loop.tool_manager,
                initial_server=name,
                connector_registry=connector_registry,
                get_vibe_config=lambda: self.agent_loop.config,
                refresh_callback=self._refresh_mcp_browser,
            )
        )

    def _find_oauth_http_server(self, alias: str) -> MCPHttp | MCPStreamableHttp | None:
        for srv in self.config.mcp_servers:
            if (
                isinstance(srv, (MCPHttp, MCPStreamableHttp))
                and srv.name == alias
                and srv.auth.type == "oauth"
            ):
                return srv
        return None

    async def _mcp_login(self, *, alias: str) -> None:
        from vibe.core.auth import (
            MCPOAuthError,
            MCPOAuthHeadlessError,
            MCPOAuthPortInUse,
            perform_oauth_login,
        )

        if not alias:
            await self._mount_and_scroll(ErrorMessage("Usage: /mcp login <name>"))
            return
        srv = self._find_oauth_http_server(alias)
        if srv is None:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"No OAuth-configured HTTP MCP server named {alias!r}. "
                    "OAuth login applies to http/streamable-http servers with "
                    'auth.type = "oauth".'
                )
            )
            return

        await self._mount_and_scroll(UserCommandMessage(f"Logging in to {alias}..."))
        import webbrowser

        async def _open_browser(url: str) -> None:
            webbrowser.open(url)

        try:
            await perform_oauth_login(srv, on_url=_open_browser)
        except MCPOAuthPortInUse as exc:
            await self._mount_and_scroll(ErrorMessage(str(exc)))
            return
        except MCPOAuthHeadlessError as exc:
            await self._mount_and_scroll(ErrorMessage(str(exc)))
            return
        except MCPOAuthError as exc:
            await self._mount_and_scroll(ErrorMessage(str(exc)))
            return

        await self.agent_loop.tool_manager.refresh_remote_tools_async()
        await self.agent_loop.refresh_system_prompt()
        self._refresh_banner()
        await self._mount_and_scroll(
            UserCommandMessage(f"Logged in to {alias}. Tools refreshed.")
        )

    async def _mcp_logout(self, *, alias: str) -> None:
        from vibe.core.auth import clear_stored_credentials

        if not alias:
            await self._mount_and_scroll(ErrorMessage("Usage: /mcp logout <name>"))
            return
        await clear_stored_credentials(alias)
        await self.agent_loop.tool_manager.refresh_remote_tools_async()
        await self.agent_loop.refresh_system_prompt()
        self._refresh_banner()
        await self._mount_and_scroll(
            UserCommandMessage(f"Cleared OAuth credentials for {alias}.")
        )

    async def _mcp_refresh(self) -> None:
        result = await self._refresh_mcp_browser()
        await self._mount_and_scroll(UserCommandMessage(result))

    async def _mcp_add(self) -> None:
        await self._switch_from_input(MCPAddApp())

    async def _show_status(self, **kwargs: Any) -> None:
        from vibe.cli.textual_ui.widgets._status_render import (
            StatusCardData,
            status_context_window,
        )
        from vibe.cli.textual_ui.widgets.status_card import StatusCard
        from vibe.core.usage import fetch_codex_quota, get_usage_recorder, summarize

        stats = self.agent_loop.stats
        try:
            active_model = self.config.get_active_model()
            model_name = active_model.name
            context_window = status_context_window(active_model)
        except ValueError:
            model_name = "<none>"
            context_window = None
        try:
            provider_name = self.config.get_active_provider().name
        except ValueError:
            provider_name = "<none>"

        records = get_usage_recorder().read_all()
        summary = summarize(records)

        # Best-effort Codex/ChatGPT plan quota fetch. Only the openai-chatgpt
        # provider exposes this; skip (silently) for everyone else. A failed
        # or slow fetch never blocks the status render — fetch returns None.
        codex_quota = None
        chatgpt_provider = next(
            (
                p
                for p in self.config.providers
                if getattr(p, "api_style", "") == "openai-chatgpt"
            ),
            None,
        )
        if chatgpt_provider is not None:
            codex_quota = await fetch_codex_quota(chatgpt_provider.api_base)

        await self._mount_and_scroll(
            StatusCard(
                StatusCardData(
                    stats=stats,
                    summary=summary,
                    version=CORE_VERSION,
                    model_name=model_name,
                    provider_name=provider_name,
                    workdir=Path(self.config.displayed_workdir or safe_cwd()),
                    session_id=self.agent_loop.session_id,
                    context_window=context_window,
                    rate_limits=self.agent_loop._rate_limit_store.all(),
                    codex_quota=codex_quota,
                )
            )
        )

    async def _spend_command(self, cmd_args: str = "", **kwargs: Any) -> None:
        cmd_args = cmd_args.strip().lower()
        if cmd_args == "reset":
            new_session_id = self.agent_loop.reset_spend()
            await self._mount_and_scroll(
                UserCommandMessage(
                    "Spend ledger reset. "
                    f"New spend session: {new_session_id}. "
                    "Cumulative call/cost/token counters start fresh; "
                    "conversation history is preserved."
                )
            )
            return

        if cmd_args and cmd_args != "status":
            await self._mount_and_scroll(
                ErrorMessage(
                    "Unknown `/spend` subcommand. "
                    "Use `/spend` (status) or `/spend reset`."
                )
            )
            return

        snapshot = self.agent_loop.spend_adapter.snapshot()
        env = snapshot.envelope
        lines = [f"### Spend budget — scope `{env.scope_id}`", ""]

        def pct(remaining: float | int | None, used: float | int) -> str:
            if remaining is None:
                return "no limit"
            total = remaining + used
            if total <= 0:
                return "—"
            return f"{(used / total) * 100:.0f}%"

        spent_calls = snapshot.spent_calls
        remaining_calls = snapshot.remaining_calls
        lines.append(
            f"- **Calls**: {spent_calls} used"
            + (
                f" of {spent_calls + remaining_calls} ({pct(remaining_calls, spent_calls)})"
                if remaining_calls is not None
                else ""
            )
        )
        spent_cost = snapshot.spent.cost_usd
        remaining_cost = snapshot.remaining_cost_usd
        lines.append(
            f"- **Cost**: ${spent_cost:.4f} used"
            + (
                f" of ${spent_cost + remaining_cost:.4f}"
                if remaining_cost is not None
                else ""
            )
        )
        spent_total = snapshot.spent.total_tokens
        remaining_total = snapshot.remaining_total_tokens
        lines.append(
            f"- **Tokens**: {spent_total:,} used"
            + (
                f" of {spent_total + remaining_total:,}"
                if remaining_total is not None
                else ""
            )
        )
        lines.append("")
        lines.append(
            "Run `/spend reset` to start a fresh ledger (keeps conversation history)."
        )
        await self._mount_and_scroll(UserCommandMessage("\n".join(lines)))

    async def _show_config(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.Config:
            return
        await self._switch_to_config_app()

    async def _show_model(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.ModelPicker:
            return
        await self._switch_to_model_picker_app()

    async def _show_provider_login(self, cmd_args: str = "", **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.ProviderLogin:
            return
        args = cmd_args.strip().split()
        if len(args) > 1:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Usage: /login [provider]", collapsed=self._tools_collapsed
                )
            )
            return
        provider_name = args[0] if args else None
        await self._mount_and_scroll(UserCommandMessage("Provider login opened..."))
        await self._switch_from_input(
            ProviderLoginApp(config=self.config, provider_name=provider_name)
        )

    async def _show_thinking(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.ThinkingPicker:
            return
        await self._switch_to_thinking_picker_app()

    async def _show_effort(self, **kwargs: Any) -> None:
        if self.config.disable_workflows:
            from vibe.cli.textual_ui.widgets.messages import ErrorMessage

            await self._mount_and_scroll(
                ErrorMessage("Workflows are disabled. Effort mode unavailable.")
            )
            return
        if self._current_bottom_app == BottomApp.EffortPicker:
            return
        await self._switch_to_effort_picker_app()

    async def _show_theme(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.ThemePicker:
            return
        await self._switch_to_theme_picker_app()

    async def _show_proxy_setup(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.ProxySetup:
            return
        await self._switch_to_proxy_setup_app()

    async def _show_data_retention(self, **kwargs: Any) -> None:
        await self._mount_and_scroll(UserCommandMessage(DATA_RETENTION_MESSAGE))

    async def _rename_local_session(self, title: str) -> str:
        session_logger = self.agent_loop.session_logger
        if not session_logger.enabled or session_logger.session_metadata is None:
            raise ValueError("Session logging is disabled in configuration.")

        if (
            session_logger.session_dir is not None
            and session_logger.metadata_filepath.exists()
        ):
            await update_saved_session_title_at_path(session_logger.session_dir, title)

        session_logger.set_title(title)
        renamed_title = session_logger.session_metadata.title
        assert renamed_title is not None
        return renamed_title

    async def _rename_session(self, cmd_args: str = "", **kwargs: Any) -> None:
        title = cmd_args.strip()
        if not title:
            await self._mount_and_scroll(
                ErrorMessage("Usage: /rename <title>", collapsed=self._tools_collapsed)
            )
            return

        try:
            renamed_title = await self._rename_local_session(title)
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to rename session: {e}", collapsed=self._tools_collapsed
                )
            )
            return

        await self._mount_and_scroll(
            UserCommandMessage(f'Session renamed to "{renamed_title}".')
        )

    def _build_picker(self, sessions: list[ResumeSessionInfo]) -> SessionPickerApp:
        sessions = sorted(sessions, key=lambda s: s.end_time or "", reverse=True)
        return SessionPickerApp(
            sessions=sessions,
            latest_messages=session_latest_messages(sessions, self.config),
            current_session_id=self.agent_loop.session_id,
            cwd=str(safe_cwd()),
        )

    async def _show_session_picker(self, **kwargs: Any) -> None:
        from vibe.core.worktree.manager import original_working_directory

        # Match how sessions are recorded: session_logger stores
        # original_working_directory(), not the worktree path that
        # safe_cwd() resolves to under worktree isolation.
        if not self.config.session_logging.enabled or not (
            local_sessions := list_local_resume_sessions(
                self.config, original_working_directory()
            )
        ):
            await self._mount_and_scroll(
                UserCommandMessage("No sessions found for this directory.")
            )
            if self._show_resume_picker:
                self._show_resume_picker = False
                self._process_initial_prompt()
            return

        await self._switch_from_input(self._build_picker(local_sessions))

    async def on_session_picker_app_session_selected(
        self, event: SessionPickerApp.SessionSelected
    ) -> None:
        await self._switch_to_input_app()
        session = ResumeSessionInfo(
            session_id=event.session_id, cwd="", title=None, end_time=None
        )
        try:
            await self._resume_local_session(session)
        except Exception as e:
            if self._show_resume_picker:
                self._show_resume_picker = False
                self._startup_prompt_processed = True
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to load session: {e}", collapsed=self._tools_collapsed
                )
            )
            return

        if self._show_resume_picker:
            self._show_resume_picker = False
            self._process_initial_prompt()

    async def on_session_picker_app_session_delete_requested(
        self, event: SessionPickerApp.SessionDeleteRequested
    ) -> None:
        if event.session_id == self.agent_loop.session_id:
            self._clear_pending_session_delete(event.option_id)
            await self._mount_and_scroll(
                ErrorMessage(
                    "Deleting the current session is not supported.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        try:
            await delete_saved_session(event.session_id, self.config.session_logging)
        except Exception as e:
            self._clear_pending_session_delete(event.option_id)
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to delete session: {e}", collapsed=self._tools_collapsed
                )
            )
            return

        try:
            picker = self.query_one(SessionPickerApp)
        except Exception:
            picker = None

        if picker is not None:
            picker.remove_session(event.option_id)

        await self._mount_and_scroll(
            UserCommandMessage(
                f"Deleted session `{short_session_id(event.session_id)}`."
            )
        )

        if picker is not None and not picker.has_sessions:
            await self._switch_to_input_app()
            await self._mount_and_scroll(
                UserCommandMessage("No saved sessions left for this directory.")
            )

    def _clear_pending_session_delete(self, option_id: str) -> None:
        try:
            self.query_one(SessionPickerApp).clear_pending_delete(option_id)
        except Exception:
            pass

    async def on_session_picker_app_cancelled(
        self, event: SessionPickerApp.Cancelled
    ) -> None:
        await self._switch_to_input_app()

        await self._mount_and_scroll(UserCommandMessage("Resume cancelled."))

    async def _resume_local_session(self, session: ResumeSessionInfo) -> None:
        session_config = self.config.session_logging
        session_path = SessionLoader.find_session_by_id(
            session.session_id, session_config
        )

        if not session_path:
            raise ValueError(
                f"Session `{short_session_id(session.session_id)}` not found."
            )

        self._emit_session_closed_for_active_session()

        loaded_messages, metadata = SessionLoader.load_session(session_path)
        if self._chat_input_container:
            self._chat_input_container.set_custom_border(None)

        non_system_messages = [
            msg for msg in loaded_messages if msg.role != Role.SYSTEM
        ]

        self.agent_loop.resume_existing_session(
            session.session_id, metadata.get("parent_session_id"), session_path
        )
        await self.agent_loop.hydrate_experiments_from_session()
        current_system_messages = [
            msg for msg in self.agent_loop.messages if msg.role == Role.SYSTEM
        ]
        self.agent_loop.messages.reset(current_system_messages + non_system_messages)
        self._refresh_profile_widgets()

        self._reset_ui_state()
        await self._load_more.hide()

        await self._messages_area.remove_children()

        if self.event_handler:
            self.event_handler.is_remote = False
        await self._resume_history_from_messages()
        self._loop_runner.restore_from_session()
        await self._mount_and_scroll(
            UserCommandMessage(
                f"Resumed session `{short_session_id(session.session_id)}`"
            )
        )

    async def remove_loading(self) -> None:
        await self._remove_loading_widget()

    async def ensure_loading(self, status: str = DEFAULT_LOADING_STATUS) -> None:
        await self._ensure_loading_widget(status)

    @property
    def loading_widget(self) -> LoadingWidget | None:
        return self._loading_widget

    async def _reload_config(self, **kwargs: Any) -> None:
        try:
            self._reset_ui_state()
            await self._load_more.hide()
            base_config = VibeConfig.load()

            await self.agent_loop.reload_with_initial_messages(base_config=base_config)
            await self._resolve_plan()
            self._narrator_manager.sync()
            self._refresh_model_status_badge()

            setup_lsp_for_config(
                base_config, lambda: self.agent_loop.base_config, safe_cwd()
            )

            # Re-discover workflows so new/changed/removed scripts (and the
            # disable_workflows flag) take effect without a restart.
            self._workflow_manager = WorkflowManager(lambda: self.agent_loop.config)
            self.commands.clear_dynamic()
            self._register_workflow_commands()

            if self._banner:
                cc, ct = compute_connector_counts(
                    base_config, self.agent_loop.connector_registry
                )
                self._banner.set_state(
                    base_config,
                    self.agent_loop.skill_manager,
                    connectors_connected=cc,
                    connectors_total=ct,
                    hooks_count=self.agent_loop.hooks_count,
                    plan_description=plan_title(self._plan_info),
                )
            self._show_config_issues()
            await self._mount_and_scroll(
                UserCommandMessage(
                    "Configuration reloaded (includes agent instructions and skills)."
                )
            )
            stripped_count = (
                self.agent_loop.count_history_images_unsupported_by_active_model()
            )
            if stripped_count > 0:
                try:
                    model_alias = self.agent_loop.config.get_active_model().alias
                except ValueError:
                    model_alias = "the active model"
                noun = "image" if stripped_count == 1 else "images"
                await self._mount_and_scroll(
                    WarningMessage(
                        f"{stripped_count} {noun} from earlier turns will be omitted "
                        f"when sending to {model_alias} (no vision support)."
                    )
                )
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to reload config: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _install_lean(self, **kwargs: Any) -> None:
        current = list(self.agent_loop.base_config.installed_agents)
        if "lean" in current:
            await self._mount_and_scroll(
                UserCommandMessage("Lean agent is already installed.")
            )
            return
        VibeConfig.save_updates({"installed_agents": sorted([*current, "lean"])})
        await self._reload_config()

    async def _uninstall_lean(self, **kwargs: Any) -> None:
        current = list(self.agent_loop.base_config.installed_agents)
        if "lean" not in current:
            await self._mount_and_scroll(
                UserCommandMessage("Lean agent is not installed.")
            )
            return
        VibeConfig.save_updates({
            "installed_agents": [a for a in current if a != "lean"]
        })
        await self._reload_config()

    async def _install_lsp(self, **kwargs: Any) -> None:
        current = list(self.agent_loop.base_config.installed_components)
        if "lsp" not in current:
            VibeConfig.save_updates({"installed_components": sorted([*current, "lsp"])})
        # Always reload so the manager re-syncs from the latest installed
        # package code + PATH-discovered presets. Without this, a process
        # that started before a code/package update keeps a stale manager.
        await self._reload_config()

        from vibe.core.lsp import get_lsp_manager

        manager = get_lsp_manager()
        if manager and manager.servers:
            lines = ["LSP enabled. Detected language servers:", ""]
            for server in manager.servers.values():
                exts = ", ".join(sorted(server.config.languages.keys()))
                lines.append(f"  - {server.config.name} ({exts})")
            lines.append("")
            lines.append(
                "The lsp tool is now available; diagnostics surface "
                "automatically after edits. Run /lsp to check status."
            )
        else:
            lines = self._lsp_empty_server_hint_lines(
                footer="Run /lspstall again after changing directory or config."
            )
        await self._mount_and_scroll(UserCommandMessage("\n".join(lines)))

    def _recent_preset_keys(self) -> list[str]:
        from vibe.core.lsp._defaults import preset_for_extension

        seen: set[str] = set()
        ordered: list[str] = []
        for ext in self._recent_edited_exts:
            preset = preset_for_extension(ext)
            if preset is None or preset.key in seen:
                continue
            seen.add(preset.key)
            ordered.append(preset.key)
        return ordered

    def _lsp_empty_server_hint_lines(self, *, footer: str = "") -> list[str]:
        """Lines for the no-active-servers case, naming the real cause.

        Distinguishes "servers installed on PATH but no project manifest marker
        matched here" from "no language-server binaries on PATH at all". The old
        code reported both as "not on your PATH", sending users to reinstall
        servers that were already working.
        """
        from vibe.core.lsp._defaults import PRESETS, available_presets, broken_presets

        on_path = available_presets(None)
        if on_path:
            names = ", ".join(p.display_name for p in on_path)
            lines = [
                f"Language servers are installed ({names}), but none matched "
                "this project: no manifest marker (pyproject.toml, "
                "package.json, Cargo.toml, go.mod, …) was found at or above "
                "the current directory.",
                "",
                "To use them here, either:",
                "  - launch vibe from the project root,",
                "  - declare the server with a [[lsp_servers]] block in "
                "config.toml, or",
                "  - set lsp_auto_discover = false and add it manually.",
            ]
        else:
            recent = self._recent_preset_keys()
            pinned = [PRESETS[k] for k in recent if k in PRESETS]
            rest = [p for k, p in PRESETS.items() if k not in recent]
            ordered = pinned + rest
            lines = [
                "No language server binaries were found on your PATH.",
                "",
                "Install one to get started:",
                "",
            ]
            for preset in ordered:
                lines.append(f"  - {preset.display_name}: {preset.install_hint}")
        broken = broken_presets()
        if broken:
            lines.append("")
            lines.append("Installed but not working (probe failed):")
            for probe in broken:
                detail = probe.stderr or f"exit {probe.returncode}"
                lines.append(
                    f"  - {probe.preset.display_name}: reinstall "
                    f"({probe.preset.install_hint}) — {detail}"
                )
        if footer:
            lines.append("")
            lines.append(footer)
        return lines

    async def _uninstall_lsp(self, **kwargs: Any) -> None:
        current = list(self.agent_loop.base_config.installed_components)
        if "lsp" not in current:
            await self._mount_and_scroll(
                UserCommandMessage("LSP feature is not installed.")
            )
            return
        VibeConfig.save_updates({
            "installed_components": [c for c in current if c != "lsp"]
        })
        await self._reload_config()

    async def _show_lsp_status(self, **kwargs: Any) -> None:
        installed = "lsp" in self.agent_loop.base_config.installed_components
        if not installed:
            await self._mount_and_scroll(
                UserCommandMessage(
                    "LSP feature is not installed. Run /lspstall to enable it."
                )
            )
            return
        from vibe.core.lsp import get_lsp_manager

        manager = get_lsp_manager()
        if manager is None or not manager.servers:
            hint = "\n".join(self._lsp_empty_server_hint_lines())
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"LSP is installed but no servers are active.\n\n{hint}"
                )
            )
            return
        lines = ["## LSP servers", ""]
        for name, server in manager.servers.items():
            state = server.state.value
            exts = ", ".join(sorted(server.config.languages.keys()))
            line = f"- **{name}** ({state}) — {exts}"
            if server.last_error:
                line += f"\n  error: {server.last_error}"
            lines.append(line)
        await self._mount_and_scroll(UserCommandMessage("\n".join(lines)))

    def _maybe_nudge_lsp(self, file_path: str) -> None:
        ext = Path(file_path).suffix.lower()
        if ext:
            self._recent_edited_exts.appendleft(ext)
        if self._lsp_nudge_shown_this_session:
            return
        from vibe.core.lsp._nudge import (
            evaluate_nudge,
            record_first_prompted,
            record_install_hint_shown,
        )

        decision = evaluate_nudge(
            file_path, self.agent_loop.base_config, CACHE_FILE.path
        )
        if decision.kind == "first_prompt":
            record_first_prompted(CACHE_FILE.path)
            self._lsp_nudge_shown_this_session = True
            self.call_after_refresh(
                lambda: self._mount_lsp_callout(decision.preset_display_name)
            )
        elif decision.kind == "reminder":
            self._lsp_nudge_shown_this_session = True
            self.notify(
                f"LSP is available for {decision.preset_display_name}. "
                "Run /lspstall to enable.",
                timeout=10,
            )
        elif decision.kind == "install_hint":
            preset_key = self._preset_key_for_display_name(decision.preset_display_name)
            if preset_key is None:
                return
            record_install_hint_shown(preset_key, CACHE_FILE.path)
            self._lsp_nudge_shown_this_session = True
            self.call_after_refresh(
                lambda: self._mount_lsp_install_hint_callout(
                    decision.preset_display_name, decision.install_hint, preset_key
                )
            )

    @staticmethod
    def _preset_key_for_display_name(display_name: str) -> str | None:
        from vibe.core.lsp._defaults import PRESETS

        return next(
            (p.key for p in PRESETS.values() if p.display_name == display_name), None
        )

    async def _mount_lsp_install_hint_callout(
        self, language_display_name: str, install_hint: str, preset_key: str
    ) -> None:
        await self._mount_and_scroll(
            LspInstallHintCallout(language_display_name, install_hint, preset_key)
        )

    async def _mount_lsp_callout(self, language_display_name: str) -> None:
        await self._mount_and_scroll(LspInstallCallout(language_display_name))

    def on_lsp_install_callout_accepted(
        self, event: LspInstallCallout.Accepted
    ) -> None:
        current = list(self.agent_loop.base_config.installed_components)
        if "lsp" not in current:
            VibeConfig.save_updates({"installed_components": sorted([*current, "lsp"])})
            asyncio.create_task(self._reload_config())
        self.notify("LSP enabled.", timeout=4)

    def on_lsp_install_callout_declined(
        self, event: LspInstallCallout.Declined
    ) -> None:
        from vibe.core.lsp._nudge import record_declined

        record_declined(CACHE_FILE.path)
        self._lsp_nudge_shown_this_session = True
        self.notify("You can enable LSP later with /lspstall.", timeout=6)

    def on_lsp_install_hint_callout_dismissed(
        self, event: LspInstallHintCallout.Dismissed
    ) -> None:
        from vibe.core.lsp._nudge import record_install_hint_declined

        record_install_hint_declined(event.preset_key, CACHE_FILE.path)
        self._lsp_nudge_shown_this_session = True

    async def _clear_history(self, **kwargs: Any) -> None:
        try:
            self._reset_ui_state()
            if self._chat_input_container:
                self._chat_input_container.set_custom_border(None)
            await self.agent_loop.clear_history()
            if self.event_handler:
                await self.event_handler.finalize_streaming()
            await self._messages_area.remove_children()

            await self._messages_area.mount(SlashCommandMessage("clear"))
            await self._mount_and_scroll(
                UserCommandMessage("Conversation history cleared!")
            )
            self._chat_widget.scroll_home(animate=False)

        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to clear history: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _show_log_path(self, **kwargs: Any) -> None:
        if not self.agent_loop.session_logger.enabled:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Session logging is disabled in configuration.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        try:
            log_path = str(self.agent_loop.session_logger.session_dir)
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"## Current Log Directory\n\n`{log_path}`\n\nYou can send this directory to share your interaction."
                )
            )
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to get log path: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _loop_command(self, cmd_args: str = "", **kwargs: Any) -> None:
        widget = await self._loop_runner.handle_command(cmd_args)
        await self._mount_and_scroll(widget)

    async def _tasks_command(self, cmd_args: str = "", **kwargs: Any) -> None:
        from vibe.cli.textual_ui.widgets.messages import (
            ErrorMessage,
            UserCommandMessage,
        )

        cmd_args = cmd_args.strip()
        if not cmd_args:
            if self._current_bottom_app == BottomApp.Tasks:
                return
            await self._switch_to_tasks_app()
            return

        parts = cmd_args.split(None, 1)
        if parts[0].lower() in {"stop", "cancel", "kill"} and len(parts) > 1:
            target = parts[1].strip()
            if target == "all":
                stopped_any = False
                for entry in self._background_registry.list_tasks():
                    if await self._background_registry.stop(entry.task_id):
                        stopped_any = True
                await self._mount_and_scroll(
                    UserCommandMessage(
                        "Stopped all background tasks."
                        if stopped_any
                        else "No running tasks to stop."
                    )
                )
                return
            stopped = await self._background_registry.stop(target)
            if stopped:
                await self._mount_and_scroll(UserCommandMessage(f"Stopped `{target}`."))
            else:
                await self._mount_and_scroll(
                    ErrorMessage(
                        f"Could not stop `{target}` — not found or already finished."
                    )
                )
            return

        # Fall back to workflow-runner subcommands (snapshot/resume/list) for
        # /workflows parity. Workflows disabled only blocks these, not the pane.
        if self.config.disable_workflows:
            await self._mount_and_scroll(
                ErrorMessage("Workflows are disabled in this configuration.")
            )
            return
        widget = await self._workflow_runner.handle_command(cmd_args)
        await self._mount_and_scroll(widget)

    def _build_team_manager(self) -> TeamManager:
        loop = self.agent_loop

        def hook_context() -> Any:
            # Mirror AgentLoop._hook_session_context so team hooks see the same
            # session id, transcript path, and cwd as agent hooks.
            from vibe.core.hooks.models import HookSessionContext

            transcript = ""
            if (
                loop.session_logger.enabled
                and loop.session_logger.session_dir is not None
            ):
                transcript = str(loop.session_logger.messages_filepath.resolve())
            return HookSessionContext(
                session_id=loop.session_id,
                transcript_path=transcript,
                cwd=str(safe_cwd().resolve()),
                parent_session_id=loop.parent_session_id,
            )

        # Auto-activate for the deferred manifest (lead reads teammate escalations
        # via team_message); no-op when defer_builtin_tools is off.
        loop.tool_manager.pin_manifest_tools(["team_message"])
        return TeamManager(
            loop.session_id,
            hooks_manager=loop.hooks_manager,
            hook_context=hook_context,
            spend_adapter=loop.spend_adapter,
        )

    def _render_team_list_rows(self, members: list[Any]) -> list[str]:
        rows = [
            "| Name | Status | PID | Mode | Task | Age |",
            "|------|--------|-----|------|------|-----|",
        ]
        active_by_assignee: dict[str, Any] = {}
        if self._team_manager is not None:
            try:
                for task in self._team_manager.task_store.get_all_tasks():
                    if getattr(task, "assignee", None):
                        active_by_assignee[str(task.assignee)] = task
            except Exception:
                pass
        now = time.time()
        for m in members:
            mode = getattr(m, "safety_mode", None)
            mode_val = getattr(mode, "value", mode) or "shared"
            if mode_val == "shared":
                mode_val = "-"
            task_id = getattr(m, "last_task_id", None) or "-"
            age = "-"
            claimed = getattr(m, "last_claimed_at", None)
            active = active_by_assignee.get(m.name)
            if active is not None:
                if claimed is None:
                    claimed = getattr(active, "claimed_at", None)
                if task_id == "-":
                    task_id = getattr(active, "id", "-")
            if claimed is not None:
                try:
                    age = f"{int(now - claimed)}s"
                except (TypeError, ValueError):
                    age = "?"
            if getattr(m, "worker", False):
                mode_val = "worker" if mode_val == "-" else f"{mode_val}+worker"
            rows.append(
                f"| {m.name} | {m.status} | {m.pid or '-'} "
                f"| {mode_val} | {task_id} | {age} |"
            )
        return rows

    async def _team_command(self, cmd_args: str = "", **kwargs: Any) -> None:
        from vibe.cli.textual_ui.widgets.messages import (
            ErrorMessage,
            UserCommandMessage,
        )

        cmd_args = cmd_args.strip()
        if not cmd_args or cmd_args in {"list", "ls"}:
            if self._team_manager is None:
                await self._mount_and_scroll(UserCommandMessage("No team active."))
                return
            members = self._team_manager.get_members()
            if not members:
                await self._mount_and_scroll(UserCommandMessage("No teammates."))
                return
            rows = self._render_team_list_rows(members)
            await self._mount_and_scroll(UserCommandMessage("\n".join(rows)))
            return

        parts = cmd_args.split(None, 2)
        verb = parts[0].lower()

        match verb:
            case "spawn":
                await self._team_spawn(parts, ErrorMessage, UserCommandMessage)
            case "stop" | "cancel" | "kill":
                await self._team_stop(parts, ErrorMessage, UserCommandMessage)
            case "cleanup":
                await self._team_cleanup(UserCommandMessage)
            case "task":
                await self._team_task(parts, ErrorMessage, UserCommandMessage)
            case _:
                await self._mount_and_scroll(
                    ErrorMessage(
                        f"Unknown /team subcommand: `{verb}`.\n"
                        "Usage: /team [list|spawn <name> <prompt>|stop <name|all>|"
                        "task <add|done|list>|cleanup]"
                    )
                )

    async def _team_spawn(
        self, parts: list[str], ErrorMessage: type, UserCommandMessage: type
    ) -> None:
        _MIN_PARTS_FOR_SPAWN = 3
        if len(parts) < _MIN_PARTS_FOR_SPAWN:
            await self._mount_and_scroll(
                ErrorMessage("Usage: /team spawn <name> <prompt>")
            )
            return
        name = parts[1]
        prompt = parts[2]
        if self._team_manager is None:
            self._team_manager = self._build_team_manager()
        await self._team_manager.spawn_teammate(name, prompt)
        await self._mount_and_scroll(UserCommandMessage(f"Spawned teammate `{name}`."))

    async def _team_stop(
        self, parts: list[str], ErrorMessage: type, UserCommandMessage: type
    ) -> None:
        _MIN_PARTS_FOR_STOP = 2
        if self._team_manager is None:
            await self._mount_and_scroll(UserCommandMessage("No team active."))
            return
        if len(parts) < _MIN_PARTS_FOR_STOP:
            await self._mount_and_scroll(ErrorMessage("Usage: /team stop <name|all>"))
            return
        target = parts[1]
        if target == "all":
            await self._team_manager.stop_all()
            await self._mount_and_scroll(UserCommandMessage("Stopped all teammates."))
        else:
            stopped = await self._team_manager.stop_teammate(target)
            if stopped:
                await self._mount_and_scroll(
                    UserCommandMessage(f"Stopped teammate `{target}`.")
                )
            else:
                await self._mount_and_scroll(
                    ErrorMessage(f"Could not stop `{target}`.")
                )

    async def _team_cleanup(self, UserCommandMessage: type) -> None:
        if self._team_manager is not None:
            self._team_manager.cleanup()
            self._team_manager = None
            await self._mount_and_scroll(UserCommandMessage("Team cleaned up."))

    async def _team_task(
        self, parts: list[str], ErrorMessage: type, UserCommandMessage: type
    ) -> None:
        _MIN_PARTS_FOR_SUBCOMMAND = 2
        _MIN_PARTS_FOR_REST = 3
        if self._team_manager is None:
            self._team_manager = self._build_team_manager()
        sub = parts[1].lower() if len(parts) >= _MIN_PARTS_FOR_SUBCOMMAND else "list"
        rest = parts[2] if len(parts) >= _MIN_PARTS_FOR_REST else ""
        if sub == "add":
            if not rest.strip():
                await self._mount_and_scroll(
                    ErrorMessage("Usage: /team task add <description>")
                )
                return
            task = await self._team_manager.add_team_task(rest.strip())
            await self._mount_and_scroll(
                UserCommandMessage(f"Created task `{task.id}`: {task.description}")
            )
        elif sub == "done":
            id_and_result = rest.split(None, 1) if rest else []
            if not id_and_result:
                await self._mount_and_scroll(
                    ErrorMessage("Usage: /team task done <id> [result]")
                )
                return
            task_id = id_and_result[0]
            task_result = id_and_result[1] if len(id_and_result) > 1 else None
            task = await self._team_manager.complete_team_task(task_id, task_result)
            if task is None:
                await self._mount_and_scroll(ErrorMessage(f"No such task `{task_id}`."))
            else:
                await self._mount_and_scroll(
                    UserCommandMessage(f"Completed task `{task.id}`.")
                )
        elif sub in {"list", "ls"}:
            store = self._team_manager.task_store
            store.reload()
            tasks = store.get_all_tasks()
            if not tasks:
                await self._mount_and_scroll(UserCommandMessage("No tasks."))
                return
            rows = [
                "| ID | Status | Assignee | Description |",
                "|----|--------|----------|-------------|",
            ]
            for t in tasks:
                rows.append(
                    f"| {t.id} | {t.status.value} | {t.assignee or '-'} | "
                    f"{t.description} |"
                )
            await self._mount_and_scroll(UserCommandMessage("\n".join(rows)))
        else:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Unknown /team task subcommand: `{sub}`.\n"
                    "Usage: /team task [add <desc>|done <id> [result]|list]"
                )
            )

    async def _switch_to_tasks_app(self) -> None:
        await self._switch_from_input(
            TasksApp(
                registry=self._background_registry,
                workflow_runner=self._workflow_runner,
            )
        )

    async def on_tasks_app_closed(self, _event: TasksApp.Closed) -> None:
        await self._switch_to_input_app()

    async def on_tasks_app_task_stop_requested(
        self, message: TasksApp.TaskStopRequested
    ) -> None:
        stopped = await self._background_registry.stop(message.task_id)
        if stopped:
            self.notify(f"Stopped {message.task_id}", markup=False)
        else:
            self.notify(
                f"{message.task_id} not found or already finished",
                severity="warning",
                markup=False,
            )

    async def on_tasks_app_task_pause_requested(
        self, message: TasksApp.TaskPauseRequested
    ) -> None:
        # Pause/resume toggle for workflow runs only (registry.pause returns
        # False for any other category). Both branches must await — pause() is
        # async and unpause happens via the same toggle when is_paused is True.
        entry = self._workflow_runner.find_run(message.task_id)
        if entry is None:
            return
        if entry.is_paused:
            await self._background_registry.pause(message.task_id)
            self.notify(f"Resumed workflow {message.task_id}", markup=False)
        else:
            await self._background_registry.pause(message.task_id)
            self.notify(
                f"Paused workflow {message.task_id} (in-flight agents finish)",
                markup=False,
            )

    async def on_tasks_app_save_requested(
        self, message: TasksApp.SaveRequested
    ) -> None:
        if self.config.disable_workflows:
            self.notify("Workflows are disabled.", severity="warning")
            return
        # Open the name + location dialog instead of saving immediately, so the
        # user can name the command and choose project vs personal. The dialog
        # posts SaveConfirmed/Cancelled back.
        await self._switch_from_input(
            WorkflowSaveApp(
                run_id=message.run_id,
                script_source=message.script_source,
                default_name=message.name,
            )
        )

    async def on_workflow_save_app_save_confirmed(
        self, message: WorkflowSaveApp.SaveConfirmed
    ) -> None:
        try:
            path = self._workflow_manager.save_workflow_source(
                message.name, message.script_source, location=message.location
            )
            self._workflow_manager.reload()
            self._register_workflow_commands()
        except Exception as e:  # surface any save failure to the user
            self.notify(f"Failed to save workflow: {e}", severity="error")
            await self._switch_to_tasks_app()
            return
        self.notify(f"Saved /{message.name} to {path}", markup=False)
        await self._switch_to_tasks_app()

    async def on_workflow_save_app_cancelled(
        self, _message: WorkflowSaveApp.Cancelled
    ) -> None:
        await self._switch_to_tasks_app()

    def _register_workflow_commands(self) -> None:
        if self.config.disable_workflows:
            return

        from vibe.cli.commands import Command

        for name, info in self._workflow_manager.workflows.items():
            self.commands.register_dynamic(
                name,
                Command(
                    aliases=frozenset([f"/{name}"]),
                    description=info.description or f"Run workflow: {name}",
                    handler="_run_workflow_command",
                ),
            )

    def _resolve_workflow_source(self, name: str) -> str | None:
        info = self._workflow_manager.get_workflow(name)
        return info.source if info is not None else None

    def _build_workflow_parent_context(self, tool_call_id: str) -> InvokeContext:
        loop = self.agent_loop
        return InvokeContext(
            tool_call_id=tool_call_id,
            agent_manager=loop.agent_manager,
            active_model=loop.effective_model().alias,
            session_dir=loop.session_logger.session_dir,
            launch_context=loop.launch_context,
            approval_callback=loop.approval_callback,
            sampling_callback=loop._sampling_handler,
            skill_manager=loop.skill_manager,
            scratchpad_dir=loop.scratchpad_dir,
            permission_store=loop._permission_store,
            hook_config_result=loop._hook_config_result,
            session_id=loop.session_id,
            terminal_emulator=loop.terminal_emulator,
            # Hand the runtime the host's judge resolver so each isolated
            # agent's prompt is judged at spawn (the subprocess itself runs
            # auto-approved and can't prompt the host per-tool).
            safety_judge_factory=loop._resolve_safety_judge,
        )

    def _launch_workflow_from_tool(self, script: str, name: str | None = None) -> str:
        if self.config.disable_workflows:
            raise WorkflowError("Workflows are disabled in this configuration.")
        parent_context = self._build_workflow_parent_context(
            f"workflow-tool-{name or 'run'}"
        )
        runtime = WorkflowRuntime(
            parent_context=parent_context,
            workflow_source_resolver=self._resolve_workflow_source,
        )
        run_id = self._workflow_runner.launch(script, runtime=runtime)
        return run_id

    def _workflow_status_for_tool(self, run_id: str | None = None) -> list[dict]:
        runs = self._workflow_runner.runs
        if run_id is not None:
            runs = [r for r in runs if r.run_id == run_id]
        out: list[dict] = []
        for entry in runs:
            status = entry.runtime.live_status()
            out.append({
                "run_id": entry.run_id,
                "status": entry.status.value,
                "elapsed_s": round(entry.elapsed, 1),
                "name": None,
                **status,
            })
        return out

    def _workflow_results_for_tool(
        self, run_id: str, *, phase: str | None = None, raw: bool = False
    ) -> dict[str, Any]:
        from vibe.core.tools.builtins.workflow_results import WorkflowResults

        cap = None if raw else WorkflowResults._DEFAULT_PER_AGENT_CHAR_CAP

        entry = next(
            (r for r in self._workflow_runner.runs if r.run_id == run_id), None
        )
        if entry is None:
            return {
                "run_id": run_id,
                "status": "unknown",
                "phases": [],
                "agent_results": [],
            }

        if entry.result is not None:
            run_phases = entry.result.run.phases
            status = entry.result.run.status.value
        else:
            run_phases = list(entry.runtime._phases.values())
            status = entry.status.value

        if phase is not None:
            run_phases = [p for p in run_phases if p.name == phase]

        phase_summaries: list[dict[str, Any]] = []
        agent_results: list[dict[str, Any]] = []
        for p in run_phases:
            completed = sum(1 for r in p.agent_results if r.completed)
            phase_summaries.append({
                "name": p.name,
                "agents": len(p.agent_results),
                "completed": completed,
                "failed": len(p.agent_results) - completed,
            })
            for r in p.agent_results:
                response: Any = r.response
                if (
                    cap is not None
                    and isinstance(response, str)
                    and len(response) > cap
                ):
                    response = (
                        response[:cap] + "\n…(truncated; pass raw=true for full text)"
                    )
                agent_results.append({
                    "label": r.label,
                    "agent": r.agent,
                    "phase": p.name,
                    "completed": r.completed,
                    "response": response,
                    "error": r.error,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "schema_errors": list(r.schema_errors),
                })

        return {
            "run_id": run_id,
            "status": status,
            "phases": phase_summaries,
            "agent_results": agent_results,
            "return_value": self._return_value_for_tool(entry, raw=raw),
        }

    @staticmethod
    def _return_value_for_tool(entry: Any, *, raw: bool) -> Any:
        result = getattr(entry, "result", None)
        if result is None:
            return None
        value = getattr(result, "return_value", None)
        if value is None:
            return None
        rendered = VibeApp._stringify_workflow_value(value)
        cap = None if raw else VibeApp._WORKFLOW_DELIVERY_CHAR_CAP
        if cap is not None and len(rendered) > cap:
            return rendered[:cap] + "\n…(truncated; pass raw=true for full value)"
        # Small/structured values pass through unchanged so the model keeps the
        # dict/list shape rather than a JSON string.
        return value

    async def _workflow_stop_for_tool(
        self, run_id: str | None, all_runs: bool
    ) -> dict[str, Any]:
        runner = self._workflow_runner
        if all_runs:
            active = [
                e.run_id
                for e in runner.runs
                if e.task is not None and not e.task.done()
            ]
            if not active:
                return {
                    "stopped": False,
                    "stopped_run_ids": [],
                    "message": "No active workflow runs to stop.",
                }
            await runner.stop_all()
            return {
                "stopped": True,
                "stopped_run_ids": active,
                "message": f"Stopped {len(active)} workflow run(s): "
                f"{', '.join(active)}.",
            }
        assert run_id is not None  # enforced by the tool's arg validator
        stopped = await runner.stop(run_id)
        if stopped:
            return {
                "stopped": True,
                "stopped_run_ids": [run_id],
                "message": f"Stopped workflow `{run_id}`.",
            }
        return {
            "stopped": False,
            "stopped_run_ids": [],
            "message": (f"Could not stop `{run_id}` — not found or already finished."),
        }

    def _team_dir_for_tool(self) -> str | None:
        if self._team_manager is None:
            return None
        return str(self._team_manager.team_dir)

    async def _team_spawn_for_tool(
        self,
        name: str,
        prompt: str,
        agent: str,
        max_turns: int,
        worker: bool = False,
        safety_mode: TeamSafetyMode = TeamSafetyMode.SHARED,
    ) -> dict[str, Any]:
        if self._team_manager is None:
            self._team_manager = self._build_team_manager()
        await self._team_manager.spawn_teammate(
            name,
            prompt,
            agent=agent,
            max_turns=max_turns,
            worker=worker,
            safety_mode=safety_mode,
        )
        kind = "worker" if worker else "teammate"
        return {
            "name": name,
            "team_dir": str(self._team_manager.team_dir),
            "message": f"Spawned {kind} `{name}`.",
            "worker": worker,
            "safety_mode": safety_mode.value,
        }

    async def _run_workflow_command(self, cmd_args: str = "", **kwargs: Any) -> None:
        from vibe.cli.textual_ui.widgets.messages import (
            ErrorMessage,
            UserCommandMessage,
        )

        if self.config.disable_workflows:
            await self._mount_and_scroll(
                ErrorMessage("Workflows are disabled in this configuration.")
            )
            return

        cmd_name = kwargs.get("_cmd_name", "")
        if not cmd_name:
            await self._mount_and_scroll(
                ErrorMessage("Could not determine workflow name.")
            )
            return

        info = self._workflow_manager.get_workflow(cmd_name)
        if info is None:
            await self._mount_and_scroll(
                ErrorMessage(f"Workflow '{cmd_name}' not found.")
            )
            return

        parent_context = self._build_workflow_parent_context(f"workflow-{cmd_name}")
        runtime = WorkflowRuntime(
            parent_context=parent_context,
            workflow_source_resolver=self._resolve_workflow_source,
        )
        run_id = self._workflow_runner.launch(
            info.source, runtime=runtime, args=cmd_args or None
        )
        await self._mount_and_scroll(
            UserCommandMessage(
                f"Launched workflow `{cmd_name}` as `{run_id}`. "
                f"Use /workflows to check progress."
            )
        )

    async def _on_workflow_complete(self, result: Any) -> None:
        from vibe.cli.textual_ui.widgets.messages import SubagentResponseMessage
        from vibe.core.workflows.models import WorkflowStatus

        summary = result.summary
        # Failed and budget-blocked runs are not successful completions. A STOPPED
        # run (cancelled by the user) is neither success nor error.
        status = getattr(result.run, "status", None)
        run_id = getattr(result.run, "run_id", "workflow")
        if isinstance(status, WorkflowStatus) and status in {
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
        }:
            label = f"Workflow Result — {run_id} ({status.value})"
            await self._mount_and_scroll(
                SubagentResponseMessage(summary, label=label, collapsed=False)
            )
        else:
            label = f"Workflow Result — {run_id} (completed)"
            await self._mount_and_scroll(SubagentResponseMessage(summary, label=label))

        # Deliver the outcome + return value to the host agent's context. The
        # UserCommandMessage above is a UI-only note for the human; without this
        # injection the agent that launched the workflow never sees the actual
        # return_value, making background workflows useless to it. Failures are
        # delivered too (the summary now carries failed-agent counts/errors).
        payload = self._format_workflow_delivery(result)
        if payload:
            try:
                if self._agent_running:
                    # A turn is in flight (typically the one that launched this
                    # run, which is still going when a fast run — e.g. a script
                    # that errors at 0 agents — completes). Fold the outcome into
                    # it via the pending-injection path so the loop keeps going
                    # and the agent acts on the result. inject_user_context only
                    # appends to history, which a running turn never re-reads, so
                    # the agent would otherwise appear to stop on a failed run.
                    self.agent_loop.stage_injected_message(payload)
                    # Teardown race: the run can complete in the window between
                    # the loop's last pending-injection drain and the turn
                    # actually ending (_agent_running flips false only after an
                    # awaited save). There the staged result is folded into
                    # history without any LLM turn acting on it, stranding it
                    # until the next human message. Arm a post-turn flush that
                    # resumes once if that strand is detected.
                    launching = self._agent_task
                    if launching is not None:
                        flush = asyncio.create_task(
                            self._flush_stranded_workflow_delivery(launching)
                        )
                        self._workflow_flush_tasks.add(flush)
                        flush.add_done_callback(self._workflow_flush_tasks.discard)
                else:
                    # Idle: the launching turn already ended, so there is no live
                    # loop to fold into. Auto-resume — drive a continuation turn
                    # with the delivery as its prompt so the agent acts on the
                    # outcome instead of stalling until the next human message.
                    # The event consumer ignores UserMessageEvent, so this adds
                    # no redundant user bubble on top of the summary shown above.
                    # Set _agent_running synchronously (before the task is
                    # scheduled) so a user submit racing on the same loop tick
                    # cannot also start a turn and clobber _agent_task.
                    self._agent_running = True
                    self._agent_task = asyncio.create_task(
                        self._handle_agent_loop_turn(payload)
                    )
                    self._queue.notify_busy_changed()
            except Exception:
                logger.warning(
                    "Failed to deliver workflow result to agent loop", exc_info=True
                )

    async def _flush_stranded_workflow_delivery(
        self, launching_task: asyncio.Task
    ) -> None:
        # Safety net for the busy-branch teardown race in _on_workflow_complete.
        # Once the launching turn settles, drive one continuation turn iff the
        # staged delivery was stranded: the agent is fully idle and history ends
        # with an injected user message that no assistant reply followed (a
        # normally-consumed delivery ends with the agent's assistant response,
        # so this never double-fires). Gated by the same auto-continue cap as
        # the subagent wake so an unattended chain stays bounded.
        try:
            await launching_task
        except Exception:
            return
        if self._is_busy() or self._auto_continue_active:
            return
        if self._input_queue.paused or bool(self._input_queue):
            return
        msgs = self.agent_loop.messages
        if not msgs or msgs[-1].role != Role.USER or not msgs[-1].injected:
            return
        if self._consecutive_auto_continues >= _MAX_AUTO_CONTINUES:
            return
        self._auto_continue_active = True
        self._consecutive_auto_continues += 1
        self._agent_running = True
        self._start_queued_agent_turn(_WORKFLOW_CONTINUE_PROMPT)
        self._queue.notify_busy_changed()

    _WORKFLOW_DELIVERY_CHAR_CAP = 16_000

    @staticmethod
    def _stringify_workflow_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return orjson.dumps(value, option=orjson.OPT_INDENT_2, default=str).decode(
                "utf-8"
            )
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _format_workflow_delivery(cls, result: Any) -> str:
        summary = getattr(result, "summary", "") or ""
        return_value = getattr(result, "return_value", None)
        parts: list[str] = []
        if summary:
            parts.append(summary)
        if return_value is not None:
            rendered = cls._stringify_workflow_value(return_value)
            if rendered and rendered.strip():
                if len(rendered) > cls._WORKFLOW_DELIVERY_CHAR_CAP:
                    rendered = (
                        rendered[: cls._WORKFLOW_DELIVERY_CHAR_CAP] + "\n…(truncated)"
                    )
                parts.append(f"Result:\n{rendered}")

        # Recover outputs from agents that did not complete cleanly. The script's
        # return_value cannot reach these (they raised and degraded to None),
        # but the run records them; surface them so partial work is never lost.
        failed_outputs = cls._collect_recoverable_outputs(getattr(result, "run", None))
        if failed_outputs:
            used = sum(len(p) for p in parts)
            per = max(
                800,
                (cls._WORKFLOW_DELIVERY_CHAR_CAP - used) // max(1, len(failed_outputs)),
            )
            chunks: list[str] = []
            for label, response, error, schema_errors in failed_outputs:
                blob = cls._stringify_workflow_value(response)
                if len(blob) > per:
                    blob = blob[:per] + "\n…(truncated)"
                tag = f"[{label or 'agent'}]"
                if error:
                    tag += f" failed: {error.splitlines()[0][:120]}"
                # Surface the first field-level schema error so a systemic
                # schema mismatch is named in the push, not just the generic
                # "Schema validation failed after N attempts". Full per-agent
                # detail is on the workflow_results pull path.
                if schema_errors:
                    tag += f" [{schema_errors[0][:120]}]"
                chunks.append(f"{tag}\n{blob}")
            parts.append(
                "Recovered outputs from agents that did not complete cleanly "
                "(not included in the result above):\n\n" + "\n\n".join(chunks)
            )
        return "\n\n".join(parts)

    @staticmethod
    def _collect_recoverable_outputs(
        run: Any,
    ) -> list[tuple[str | None, Any, str | None, list[str]]]:
        if run is None:
            return []
        out: list[tuple[str | None, Any, str | None, list[str]]] = []
        for phase in getattr(run, "phases", []) or []:
            for ar in getattr(phase, "agent_results", []) or []:
                if getattr(ar, "completed", True):
                    continue
                resp = getattr(ar, "response", None)
                if resp in (None, "", []):
                    continue
                out.append((
                    getattr(ar, "label", None),
                    resp,
                    getattr(ar, "error", None),
                    list(getattr(ar, "schema_errors", []) or []),
                ))
        return out

    async def _persist_workflow_snapshots(self) -> None:
        snapshots: list[dict[str, Any]] = []
        for entry in self._workflow_runner.runs:
            # Snapshot every run, including in-flight and cancelled ones, so
            # interrupted runs are captured for inspection/resume. Previously
            # only runs with a result (natural completion) were snapshotted.
            try:
                snap = entry.runtime.snapshot(
                    run_id=entry.run_id,
                    script_source=entry.script_source,
                    args=entry.args,
                    return_value=(
                        entry.result.return_value if entry.result is not None else None
                    ),
                )
            except Exception:
                logger.warning(
                    "Failed to snapshot workflow %s", entry.run_id, exc_info=True
                )
                continue
            snapshots.append(snap.model_dump(mode="json"))
        await self.agent_loop.session_logger.persist_workflow_snapshots(snapshots)

    def _load_workflow_snapshots(self) -> list[dict[str, Any]]:
        return self.agent_loop.session_logger.load_workflow_snapshots()

    def _build_resume_runtime(self) -> WorkflowRuntime | None:
        if self.config.disable_workflows:
            return None
        parent_context = self._build_workflow_parent_context("workflow-resume")
        return WorkflowRuntime(
            parent_context=parent_context,
            workflow_source_resolver=self._resolve_workflow_source,
        )

    async def _handle_le_chaton_prompt(self, text: str) -> None:
        if self.config.disable_workflows:
            await self._handle_user_message(text)
            return
        previous_mode = self.config.effort_mode
        # set_effort_mode("le-chaton") bumps the active model's thinking to
        # "max"; capture the prior level so the turn-scoped switch is fully
        # reversible. Docs say the keyword triggers le chaton "for that turn"
        # -- without restoration it persisted permanently across sessions.
        previous_thinking = self.config.get_active_model().thinking
        if previous_mode != "le-chaton":
            self.config.set_effort_mode("le-chaton")
            await self._reload_config()
        try:
            await self._handle_user_message(text)
            # _handle_user_message spawns the turn as a background task and
            # returns immediately; the turn reads the thinking level live when it
            # builds the LLM request. Wait for that turn to finish before
            # restoring, otherwise the boost is reverted before it is ever used
            # (the le-chaton turn would run at the prior level). asyncio.wait
            # does not propagate the turn's own exception/cancellation, but still
            # surfaces cancellation of this coroutine.
            task = self._agent_task
            if task is not None and not task.done():
                await asyncio.wait({task})
        finally:
            if previous_mode != "le-chaton":
                self.config.set_effort_mode(previous_mode)
                if self.config.get_active_model().thinking != previous_thinking:
                    self.config.set_thinking(previous_thinking)
                await self._reload_config()

    async def _compact_history(self, cmd_args: str = "", **kwargs: Any) -> None:
        if self._agent_running:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Cannot compact while agent loop is processing. Please wait.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if len(self.agent_loop.messages) <= 1:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to compact yet.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not self.event_handler:
            return

        old_session_id = self.agent_loop.session_id
        compact_msg = CompactMessage()
        self.event_handler.current_compact = compact_msg
        await self._mount_and_scroll(compact_msg)

        self._agent_task = asyncio.create_task(
            self._run_compact(compact_msg, old_session_id, cmd_args.strip())
        )

    async def _run_compact(
        self,
        compact_msg: CompactMessage,
        old_session_id: str,
        extra_instructions: str = "",
    ) -> None:
        self._agent_running = True
        try:
            await self.agent_loop.compact(extra_instructions=extra_instructions)
            compact_msg.set_complete(
                old_session_id=old_session_id, new_session_id=self.agent_loop.session_id
            )

        except asyncio.CancelledError:
            compact_msg.set_error("Compaction interrupted")
            raise
        except Exception as e:
            compact_msg.set_error(str(e))
        finally:
            self._agent_running = False
            self._agent_task = None
            if self.event_handler:
                self.event_handler.current_compact = None

    def _get_session_resume_info(self) -> str | None:
        if not self.agent_loop.session_logger.enabled:
            return None
        if not self.agent_loop.session_logger.session_id:
            return None
        session_config = self.agent_loop.session_logger.session_config
        session_path = SessionLoader.does_session_exist(
            self.agent_loop.session_logger.session_id, session_config
        )
        if session_path is None:
            return None
        return short_session_id(self.agent_loop.session_logger.session_id)

    async def _worktree_command(self, cmd_args: str = "", **kwargs: Any) -> None:
        from vibe.cli.textual_ui.widgets.messages import (
            ErrorMessage,
            UserCommandMessage,
        )
        from vibe.core.worktree.manager import worktree_manager

        wt = worktree_manager.active
        if wt is None:
            await self._mount_and_scroll(
                UserCommandMessage(
                    "No worktree is active. Isolation is on by default — "
                    "relaunch without `--no-worktree` to enable it, or set "
                    '`worktree.mode = "on"` in config.'
                )
            )
            return

        sub = cmd_args.strip().lower()
        if sub in {"", "status"}:
            from git import Repo

            repo = Repo(str(wt.worktree_path))
            dirty = repo.is_dirty(untracked_files=True)
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"**Worktree isolation active**\n"
                    f"- Branch: `{wt.branch}`\n"
                    f"- Worktree: `{wt.worktree_path}`\n"
                    f"- Original root: `{wt.original_repo_root}`\n"
                    f"- Dirty: {'yes' if dirty else 'no'}\n"
                    f"- Created at HEAD: `{wt.create_head_sha[:8]}`"
                )
            )
        elif sub == "diff":
            from git import Repo

            root_repo = Repo(str(wt.original_repo_root))
            diff_output = root_repo.git.diff(wt.create_head_sha, wt.branch, "--stat")
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"**Diff** (original HEAD..worktree branch)\n```\n{diff_output or '(no changes)'}\n```"
                )
            )
        elif sub == "merge":
            # F10: 6740fdb removed exit-time auto-merge; the branch is kept
            # and must be landed or discarded explicitly after the session.
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"**Worktree branch:** `{wt.branch}`\n"
                    f"The exit-time auto-merge was removed — the branch is kept "
                    f"for explicit review.\n"
                    f"```\n"
                    f"# land it (rebase-then-ff, under the per-repo merge lock):\n"
                    f"cd {wt.original_repo_root}\n"
                    f"vibe worktree merge {wt.branch}\n"
                    f"# or discard:\n"
                    f"vibe worktree discard {wt.branch}\n"
                    f"```"
                )
            )
        else:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Unknown /worktree subcommand: `{sub}`. "
                    "Usage: /worktree [status|diff|merge]"
                )
            )

    async def _stop_teams(self) -> None:
        if self._team_manager is not None:
            try:
                await self._team_manager.stop_all()
                self._team_manager.cleanup()
            except Exception as exc:
                logger.error("Failed to stop teams during shutdown", exc_info=exc)
            finally:
                self._team_manager = None

    async def _exit_app(self, **kwargs: Any) -> None:
        self._emit_session_closed_for_active_session()
        await self._loop_runner.stop()
        await self._workflow_runner.stop_all()
        await self._stop_teams()
        await self._searxng_teardown()
        # Reap backgrounded processes so a forgotten dev server doesn't orphan
        # to init when vibe exits. Teams/workflows are already reaped above.
        try:
            await self._background_registry.shutdown()
        except Exception as exc:
            logger.error("Failed to reap background processes on exit", exc_info=exc)
        await teardown_lsp_async()
        self._log_reader.shutdown()
        await self._voice_manager.close()
        await self._narrator_manager.close()
        await self.agent_loop.aclose()
        try:
            await self.agent_loop.telemetry_client.aclose()
        except Exception as exc:
            logger.error("Failed to close telemetry client during exit", exc_info=exc)
        finally:
            self.exit(result=self._get_session_resume_info())

    def _searxng_settings(self) -> SearxngSettings:
        return resolve_searxng_settings(self.config.tools)

    async def _searxng_autostart(self) -> None:
        try:
            settings = self._searxng_settings()
            # No-op for the common case (no searxng_url), so Mistral-only users
            # pay nothing; failures are logged, never fatal to the session.
            if not (settings.url and settings.manage and settings.autostart):
                return
            # Gate early searches: a request issued while we (re)start the
            # container for health or engine reconciliation would otherwise race
            # it and surface a spurious "SearXNG is down" prompt.
            begin_autostart()
            try:
                outcome = await ensure_running(settings)
                if outcome.started:
                    self.notify(
                        f"Started local SearXNG ({settings.effective_url})",
                        markup=False,
                    )
                elif outcome.attempted and not outcome.ok:
                    self.notify(
                        f"SearXNG unavailable: {outcome.detail}",
                        severity="warning",
                        markup=False,
                    )
            finally:
                signal_autostart_done()
        except Exception as exc:
            logger.warning("SearXNG autostart failed", exc_info=exc)

    async def _searxng_teardown(self) -> None:
        try:
            settings = self._searxng_settings()
            # stop_all_started stops only containers vibe itself launched.
            await stop_all_started(enabled=settings.stop_on_exit)
        except Exception as exc:
            logger.warning("SearXNG teardown failed", exc_info=exc)

    def _make_default_voice_manager(self) -> VoiceManager:
        from vibe.cli.voice_manager import VoiceManager
        from vibe.core.audio_recorder import AudioRecorder

        try:
            model = self.config.get_active_transcribe_model()
            provider = self.config.get_transcribe_provider_for_model(model)
            transcribe_client = make_transcribe_client(provider, model)
        except (ValueError, KeyError) as exc:
            logger.error(
                "Failed to initialize transcription, check transcribe model configuration",
                exc_info=exc,
            )
            transcribe_client = None

        return VoiceManager(
            lambda: self.config,
            audio_recorder=AudioRecorder(),
            transcribe_client=transcribe_client,
            telemetry_client=self.agent_loop.telemetry_client,
        )

    async def _show_voice_settings(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.Voice:
            return
        await self._switch_to_voice_app()

    async def _switch_from_input(self, widget: Widget, scroll: bool = False) -> None:
        bottom_container = self.query_one("#bottom-app-container")
        chat = self._chat_widget
        should_scroll = scroll and chat.is_at_bottom

        with self.batch_update():
            if self._chat_input_container:
                self._chat_input_container.display = False
                self._chat_input_container.disabled = True

            self._feedback_bar.hide()

            self._current_bottom_app = self._bottom_app_by_widget().get(
                type(widget), BottomApp.Input
            )
            await bottom_container.mount(widget)

        self.call_after_refresh(widget.focus)
        if should_scroll:
            self.call_after_refresh(chat.anchor)

    async def _switch_to_config_app(self) -> None:
        if self._current_bottom_app == BottomApp.Config:
            return

        await self._mount_and_scroll(UserCommandMessage("Configuration opened..."))
        await self._switch_from_input(ConfigApp(self.config))

    async def _switch_to_voice_app(self) -> None:
        if self._current_bottom_app == BottomApp.Voice:
            return

        await self._mount_and_scroll(UserCommandMessage("Voice settings opened..."))
        await self._switch_from_input(VoiceApp(self.config))

    async def _switch_to_model_picker_app(self, target: str = "active") -> None:
        if self._current_bottom_app == BottomApp.ModelPicker:
            return

        # Remember whether this picker sets the active model or the safety-judge
        # model so on_model_picker_app_model_selected persists to the right key.
        self._model_picker_target = target
        from vibe.core.llm.model_discovery import discover_extra_models

        discovered = await discover_extra_models(self.config)
        self._discovered_models = {dm.model.alias: dm for dm in discovered}
        model_aliases = [m.alias for m in self.config.available_models]
        model_aliases += [
            alias for alias in self._discovered_models if alias not in model_aliases
        ]
        # Show the provider's real API model name as each entry's primary label
        # (the friendly alias is what gets persisted as active_model). Discovered
        # models use the server's friendly display_name when advertised (e.g.
        # "Tencent: Hy3 (free)") and fall back to the raw API id otherwise.
        display_names = {m.alias: m.name for m in self.config.available_models}
        display_names.update({
            alias: dm.display_name or dm.model.name
            for alias, dm in self._discovered_models.items()
        })
        providers = {m.alias: m.provider for m in self.config.available_models}
        providers.update({
            alias: dm.provider.name for alias, dm in self._discovered_models.items()
        })
        provider_order = {
            provider.name: index for index, provider in enumerate(self.config.providers)
        }
        model_index = {alias: index for index, alias in enumerate(model_aliases)}
        model_aliases.sort(
            key=lambda alias: (
                provider_order.get(providers.get(alias, ""), len(provider_order)),
                model_index[alias],
            )
        )
        if target == "judge":
            current_model = str(self.config.safety_judge.model or "")
        elif target == "subagent":
            current_model = str(self.config.subagent_model or "")
        elif target == "grunt":
            current_model = str(self.config.grunt_model or "")
        else:
            current_model = str(self.config.active_model)
        await self._switch_from_input(
            ModelPickerApp(
                model_aliases=model_aliases,
                current_model=current_model,
                display_names=display_names,
                footer_hint=(
                    _SUBAGENT_MODEL_HINT
                    if target == "subagent"
                    else _GRUNT_MODEL_HINT
                    if target == "grunt"
                    else None
                ),
                providers=providers,
            )
        )

    async def _switch_to_thinking_picker_app(self) -> None:
        if self._current_bottom_app == BottomApp.ThinkingPicker:
            return

        from vibe.core.config import THINKING_LEVELS

        current_thinking = self.config.get_active_model().thinking
        await self._switch_from_input(
            ThinkingPickerApp(
                thinking_levels=THINKING_LEVELS, current_thinking=current_thinking
            )
        )

    async def _switch_to_effort_picker_app(self) -> None:
        if self._current_bottom_app == BottomApp.EffortPicker:
            return

        from vibe.cli.textual_ui.widgets.effort_picker import EffortPickerApp
        from vibe.core.config import EFFORT_LEVELS

        current_effort = self.config.effort_mode
        await self._switch_from_input(
            EffortPickerApp(effort_levels=EFFORT_LEVELS, current_effort=current_effort)
        )

    async def _switch_to_theme_picker_app(self) -> None:
        if self._current_bottom_app == BottomApp.ThemePicker:
            return

        await self._switch_from_input(
            ThemePickerApp(
                theme_names=sorted_theme_names(), current_theme=self.config.theme
            )
        )

    def _apply_theme(self, theme: str) -> None:
        if theme not in BUILTIN_THEMES:
            logger.warning("Unknown theme=%s; falling back to %s", theme, DEFAULT_THEME)
            self.theme = DEFAULT_THEME
            return
        self.theme = theme

    async def _switch_to_proxy_setup_app(self) -> None:
        if self._current_bottom_app == BottomApp.ProxySetup:
            return

        await self._mount_and_scroll(UserCommandMessage("Proxy setup opened..."))
        await self._switch_from_input(ProxySetupApp())

    async def _switch_to_approval_app(
        self,
        tool_name: str,
        tool_args: BaseModel,
        required_permissions: list[RequiredPermission] | None = None,
        judge_note: str | None = None,
    ) -> None:
        approval_app = ApprovalApp(
            tool_name=tool_name,
            tool_args=tool_args,
            config=self.config,
            required_permissions=required_permissions,
            judge_note=judge_note,
        )
        await self._switch_from_input(approval_app, scroll=True)

    async def _switch_to_question_app(self, args: AskUserQuestionArgs) -> None:
        await self._switch_from_input(QuestionApp(args=args), scroll=True)

    async def _switch_to_input_app(self) -> None:
        if self._chat_input_container:
            self._chat_input_container.disabled = False
            self._chat_input_container.display = True
            self._current_bottom_app = BottomApp.Input
            self._refresh_profile_widgets()

        for app in BottomApp:
            if app != BottomApp.Input:
                try:
                    await self.query_one(f"#{app.value}-app").remove()
                except Exception:
                    pass

        if self._chat_input_container:
            self.call_after_refresh(self._chat_input_container.focus_input)
            if self._chat_widget.is_at_bottom:
                self.call_after_refresh(self._chat_widget.anchor)

    @classmethod
    @functools.cache
    def _bottom_app_widget(cls) -> dict[BottomApp, type[Widget]]:
        from vibe.cli.textual_ui.widgets.connector_auth_app import ConnectorAuthApp
        from vibe.cli.textual_ui.widgets.mcp_app import MCPApp

        return {
            BottomApp.Input: ChatInputContainer,
            BottomApp.Config: ConfigApp,
            BottomApp.ModelPicker: ModelPickerApp,
            BottomApp.ProviderLogin: ProviderLoginApp,
            BottomApp.ThemePicker: ThemePickerApp,
            BottomApp.ThinkingPicker: ThinkingPickerApp,
            BottomApp.EffortPicker: EffortPickerApp,
            BottomApp.ProxySetup: ProxySetupApp,
            BottomApp.Approval: ApprovalApp,
            BottomApp.Question: QuestionApp,
            BottomApp.SessionPicker: SessionPickerApp,
            BottomApp.MCP: MCPApp,
            BottomApp.MCPAdd: MCPAddApp,
            BottomApp.ConnectorAuth: ConnectorAuthApp,
            BottomApp.Rewind: RewindApp,
            BottomApp.Voice: VoiceApp,
            BottomApp.Tasks: TasksApp,
        }

    @classmethod
    @functools.cache
    def _bottom_app_by_widget(cls) -> dict[type[Widget], BottomApp]:
        return {cls_obj: app for app, cls_obj in cls._bottom_app_widget().items()}

    def _close_bottom_panel(
        self, source: str, action: Callable[[], None], *, clear_timestamp: bool = True
    ) -> None:
        try:
            action()
        except NoMatches:
            pass
        except Exception:
            logger.warning("bottom-panel %s action failed", source, exc_info=True)
        if clear_timestamp:
            self._last_escape_time = None

    def _focus_current_bottom_app(self) -> None:
        def _focus() -> None:
            app = self._current_bottom_app
            if app == BottomApp.Input:
                self.query_one(ChatInputContainer).focus_input()
            else:
                widget_cls = self._bottom_app_widget().get(app)
                if widget_cls is not None:
                    self.query_one(widget_cls).focus()

        self._close_bottom_panel("focus", _focus, clear_timestamp=False)

    def _handle_config_app_escape(self) -> None:
        def _close() -> None:
            self.query_one(ConfigApp).action_close()

        self._close_bottom_panel("config", _close)

    def _handle_voice_app_escape(self) -> None:
        def _close() -> None:
            self.query_one(VoiceApp).action_close()

        self._close_bottom_panel("voice", _close)

    def _handle_approval_app_escape(self) -> None:
        def _close() -> None:
            approval_app = self.query_one(ApprovalApp)
            if not approval_app.is_within_grace_period():
                approval_app.action_reject()
                self.agent_loop.telemetry_client.send_user_cancelled_action(
                    "reject_approval"
                )

        self._close_bottom_panel("approval", _close)

    def _handle_question_app_escape(self) -> None:
        def _close() -> None:
            question_app = self.query_one(QuestionApp)
            if not question_app.is_within_grace_period():
                question_app.action_cancel()
                self.agent_loop.telemetry_client.send_user_cancelled_action(
                    "cancel_question"
                )

        self._close_bottom_panel("question", _close)

    def _handle_model_picker_app_escape(self) -> None:
        def _close() -> None:
            picker = self.query_one(ModelPickerApp)
            # First escape clears a typed filter (mistype recovery); a second
            # escape (or escape with no filter) cancels the picker.
            if picker.clear_filter():
                return
            picker.post_message(ModelPickerApp.Cancelled())

        self._close_bottom_panel("model-picker", _close)

    def _handle_theme_picker_app_escape(self) -> None:
        def _close() -> None:
            self.query_one(ThemePickerApp).post_message(
                ThemePickerApp.Cancelled(original_theme=self.config.theme)
            )

        self._close_bottom_panel("theme-picker", _close)

    def _handle_thinking_picker_app_escape(self) -> None:
        def _close() -> None:
            self.query_one(ThinkingPickerApp).post_message(
                ThinkingPickerApp.Cancelled()
            )

        self._close_bottom_panel("thinking-picker", _close)

    def _handle_session_picker_app_escape(self) -> None:
        def _close() -> None:
            self.query_one(SessionPickerApp).action_cancel()

        self._close_bottom_panel("session-picker", _close)

    def _handle_tasks_app_escape(self) -> None:
        def _close() -> None:
            self.query_one(TasksApp).action_back()

        self._close_bottom_panel("tasks", _close)

    def _get_user_message_widgets(self) -> list[UserMessage]:
        return [
            child
            for child in self._messages_area.children
            if isinstance(child, UserMessage) and child.message_index is not None
        ]

    def _start_rewind_mode(self, **kwargs: Any) -> None:
        self.action_rewind_prev()

    def action_rewind_prev(self) -> None:
        if self._agent_running:
            return
        # ctrl+p/alt+up are app-priority bindings — don't hijack rewind while a
        # picker or other bottom app owns the keyboard.
        if self._current_bottom_app not in {BottomApp.Input, BottomApp.Rewind}:
            return

        user_widgets = self._get_user_message_widgets()
        if not user_widgets:
            return

        if not self._rewind_mode:
            self._rewind_mode = True
            target = user_widgets[-1]
        elif self._rewind_highlighted_widget is not None:
            try:
                idx = user_widgets.index(self._rewind_highlighted_widget)
            except ValueError:
                idx = len(user_widgets)
            if idx <= 0:
                self.run_worker(self._rewind_prev_at_top(), exclusive=False)
                return
            target = user_widgets[idx - 1]
        else:
            target = user_widgets[-1]

        self.run_worker(self._select_rewind_widget(target), exclusive=False)

    async def _rewind_prev_at_top(self) -> None:
        if self._load_more.widget is not None and self._windowing.has_backfill:
            await self.on_history_load_more_requested(HistoryLoadMoreRequested())
            user_widgets = self._get_user_message_widgets()
            if user_widgets and self._rewind_highlighted_widget is not None:
                try:
                    idx = user_widgets.index(self._rewind_highlighted_widget)
                except ValueError:
                    idx = 0
                if idx > 0:
                    await self._select_rewind_widget(user_widgets[idx - 1])
                    return
        self.call_after_refresh(self._chat_widget.scroll_home, animate=False)

    def action_rewind_next(self) -> None:
        if not self._rewind_mode:
            return
        if self._current_bottom_app not in {BottomApp.Input, BottomApp.Rewind}:
            return

        if self._rewind_highlighted_widget is None:
            return

        user_widgets = self._get_user_message_widgets()
        try:
            idx = user_widgets.index(self._rewind_highlighted_widget)
        except ValueError:
            return
        if idx >= len(user_widgets) - 1:
            return

        self.run_worker(
            self._select_rewind_widget(user_widgets[idx + 1]), exclusive=False
        )

    async def _select_rewind_widget(self, widget: UserMessage) -> None:
        if self._rewind_highlighted_widget is not None:
            self._rewind_highlighted_widget.remove_class("rewind-selected")

        widget.add_class("rewind-selected")
        self._rewind_highlighted_widget = widget

        msg_index = widget.message_index
        has_file_changes = (
            msg_index is not None
            and self.agent_loop.rewind_manager.has_file_changes_at(msg_index)
        )

        await self._switch_to_rewind_app(
            widget.get_content(), has_file_changes=has_file_changes
        )

        chat = self._chat_widget
        self.call_after_refresh(chat.scroll_to_widget, widget, animate=False, top=True)

    async def _switch_to_rewind_app(
        self, message_preview: str, *, has_file_changes: bool
    ) -> None:
        if self._current_bottom_app == BottomApp.Rewind:
            # Reuse existing widget if the option set hasn't changed
            try:
                existing = self.query_one(RewindApp)
                if existing.has_file_changes == has_file_changes:
                    existing.update_preview(message_preview)
                    return
                await existing.remove()
            except Exception:
                pass

            rewind_app = RewindApp(
                message_preview=message_preview, has_file_changes=has_file_changes
            )
            bottom_container = self.query_one("#bottom-app-container")
            self._current_bottom_app = BottomApp.Rewind
            await bottom_container.mount(rewind_app)
            self.call_after_refresh(rewind_app.focus)
        else:
            rewind_app = RewindApp(
                message_preview=message_preview, has_file_changes=has_file_changes
            )
            await self._switch_from_input(rewind_app)

    def _clear_rewind_state(self) -> None:
        if self._rewind_highlighted_widget is not None:
            self._rewind_highlighted_widget.remove_class("rewind-selected")
            self._rewind_highlighted_widget = None
        self._rewind_mode = False

    async def _exit_rewind_mode(self) -> None:
        self._clear_rewind_state()
        await self._switch_to_input_app()

    async def on_rewind_app_rewind_with_restore(
        self, message: RewindApp.RewindWithRestore
    ) -> None:
        await self._execute_rewind(restore_files=True)

    async def on_rewind_app_rewind_without_restore(
        self, message: RewindApp.RewindWithoutRestore
    ) -> None:
        await self._execute_rewind(restore_files=False)

    async def _execute_rewind(self, *, restore_files: bool) -> None:
        if not self._rewind_mode or self._rewind_highlighted_widget is None:
            return

        target_widget = self._rewind_highlighted_widget
        msg_index = target_widget.message_index

        if msg_index is None:
            return

        if msg_index < len(self.agent_loop.messages):
            try:
                (
                    message_content,
                    restore_errors,
                ) = await self.agent_loop.rewind_manager.rewind_to_message(
                    msg_index, restore_files=restore_files
                )
            except RewindError as exc:
                self.notify(str(exc), severity="error")
                return
        else:
            message_content = target_widget.get_content()
            restore_errors = []

        for error in restore_errors:
            self.notify(error, severity="warning")

        children = list(self._messages_area.children)
        try:
            target_idx = children.index(target_widget)
        except ValueError:
            target_idx = len(children)
        to_remove = children[target_idx:]
        if to_remove:
            await self._messages_area.remove_children(to_remove)

        self._clear_rewind_state()

        await self._switch_to_input_app()
        if self._chat_input_container:
            self._chat_input_container.value = message_content

    def _handle_input_app_escape(self) -> None:
        def _close() -> None:
            self.query_one(ChatInputContainer).value = ""

        self._close_bottom_panel("input", _close)

    def _handle_agent_running_escape(self) -> None:
        self.agent_loop.telemetry_client.send_user_cancelled_action("interrupt_agent")
        self.run_worker(self._interrupt_agent_loop(), exclusive=False)

    def _handle_bottom_app_close_escape(
        self,
        widget_type: (
            type[MCPApp]
            | type[MCPAddApp]
            | type[ProxySetupApp]
            | type[ConnectorAuthApp]
            | type[ProviderLoginApp]
        ),
    ) -> None:
        def _close() -> None:
            self.query_one(widget_type).action_close()

        self._close_bottom_panel(widget_type.__name__, _close)

    def _try_interrupt_bottom_app_escape(self) -> bool:
        from vibe.cli.textual_ui.widgets.connector_auth_app import ConnectorAuthApp
        from vibe.cli.textual_ui.widgets.mcp_app import MCPApp

        app = self._current_bottom_app
        handlers: dict[BottomApp, Callable[[], None]] = {
            BottomApp.Config: self._handle_config_app_escape,
            BottomApp.Voice: self._handle_voice_app_escape,
            BottomApp.MCP: lambda: self._handle_bottom_app_close_escape(MCPApp),
            BottomApp.MCPAdd: lambda: self._handle_bottom_app_close_escape(MCPAddApp),
            BottomApp.ConnectorAuth: lambda: self._handle_bottom_app_close_escape(
                ConnectorAuthApp
            ),
            BottomApp.ProxySetup: lambda: self._handle_bottom_app_close_escape(
                ProxySetupApp
            ),
            BottomApp.Approval: self._handle_approval_app_escape,
            BottomApp.Question: self._handle_question_app_escape,
            BottomApp.ModelPicker: self._handle_model_picker_app_escape,
            BottomApp.ProviderLogin: lambda: self._handle_bottom_app_close_escape(
                ProviderLoginApp
            ),
            BottomApp.ThemePicker: self._handle_theme_picker_app_escape,
            BottomApp.ThinkingPicker: self._handle_thinking_picker_app_escape,
            BottomApp.SessionPicker: self._handle_session_picker_app_escape,
            BottomApp.Tasks: self._handle_tasks_app_escape,
        }
        handler = handlers.get(app)
        if handler is not None:
            handler()
            return True
        if app == BottomApp.Rewind:
            self.run_worker(self._exit_rewind_mode(), exclusive=False)
            self._last_escape_time = None
            return True
        if (
            app == BottomApp.Input
            and self._last_escape_time is not None
            and (time.monotonic() - self._last_escape_time) < DOUBLE_ESC_DELAY
        ):
            self._handle_input_app_escape()
            return True
        return False

    def _try_interrupt_no_job_steps(self) -> bool:
        if self._voice_manager.transcribe_state != TranscribeState.IDLE:
            self._voice_manager.cancel_recording()
            return True

        if (
            self._chat_input_container
            and self._chat_input_container.dismiss_completion()
        ):
            if self._chat_input_container.value.startswith("/"):
                self._chat_input_container.value = ""
            self._last_escape_time = None
            return True

        if self._try_interrupt_bottom_app_escape():
            return True

        if (
            self._narrator_manager.is_playing
            or self._narrator_manager.state != NarratorState.IDLE
        ):
            self._narrator_manager.cancel()
            return True

        return False

    def _try_interrupt_running_job(self) -> bool:
        interrupted = False
        if self._bash_task and not self._bash_task.done():
            self._bash_task.cancel()
            interrupted = True
        if self._agent_running:
            self._handle_agent_running_escape()
            interrupted = True
        return interrupted

    def _try_interrupt(self) -> bool:
        if self._try_interrupt_no_job_steps():
            return True

        interrupted = self._try_interrupt_running_job()
        if interrupted and self._input_queue:
            self._queue.set_paused(True)

        if not interrupted and self._input_queue:
            self._queue.set_paused(True)
            interrupted = True

        self._last_escape_time = time.monotonic()
        if self._chat_widget.is_at_bottom:
            self.call_after_refresh(self._chat_widget.anchor)
        self._focus_current_bottom_app()
        return interrupted

    def action_interrupt(self) -> None:
        self._try_interrupt()

    async def on_history_load_more_requested(self, _: HistoryLoadMoreRequested) -> None:
        self._load_more.set_enabled(False)
        try:
            if not self._windowing.has_backfill:
                await self._load_more.hide()
                return
            if (batch := self._windowing.next_load_more_batch()) is None:
                await self._load_more.hide()
                return
            messages_area = self._messages_area
            if self._tool_call_map is None:
                self._tool_call_map = {}
            if self._load_more.widget:
                before: Widget | int | None = None
                after: Widget | None = self._load_more.widget
            else:
                before = 0
                after = None
            await self._mount_history_batch(
                batch.messages,
                messages_area,
                self._tool_call_map,
                start_index=batch.start_index,
                before=before,
                after=after,
            )
            if not self._windowing.has_backfill:
                await self._load_more.hide()
            else:
                await self._load_more.show(messages_area, self._windowing.remaining)
        finally:
            self._load_more.set_enabled(True)

    async def action_toggle_tool(self) -> None:
        self._tools_collapsed = not self._tools_collapsed
        for section in self.query(CollapsibleSection):
            section.set_collapsed(self._tools_collapsed)

    async def action_toggle_tasks(self) -> None:
        # The Tasks pane covers processes, teams, and loops too — it stays
        # available even when workflows are disabled (those rows just hide).
        if self._current_bottom_app == BottomApp.Tasks:
            await self._switch_to_input_app()
        elif self._current_bottom_app == BottomApp.Input:
            await self._switch_to_tasks_app()
        # Any other bottom app (a pending Approval/Question modal, a picker,
        # etc.) is left untouched: switching away would orphan its pending
        # interaction Future and hang the agent holding _user_interaction_lock.

    def action_cycle_mode(self) -> None:
        if self._current_bottom_app != BottomApp.Input:
            return
        self._refresh_profile_widgets()
        self._focus_current_bottom_app()
        self.run_worker(self._cycle_agent(), group="mode_switch", exclusive=True)

    def _refresh_profile_widgets(self) -> None:
        self._update_profile_widgets(self.agent_loop.agent_profile)
        self._refresh_model_status_badge()

    def _on_profile_changed(self) -> None:
        self._refresh_profile_widgets()
        self._refresh_banner()

    def _refresh_banner(self) -> None:
        if self._banner:
            cc, ct = compute_connector_counts(
                self.config, self.agent_loop.connector_registry
            )
            self._banner.set_state(
                self.config,
                self.agent_loop.skill_manager,
                connectors_connected=cc,
                connectors_total=ct,
                hooks_count=self.agent_loop.hooks_count,
                plan_description=plan_title(self._plan_info),
            )

    def _update_profile_widgets(self, profile: AgentProfile) -> None:
        if self._chat_input_container:
            self._chat_input_container.set_safety(profile.safety)
            self._chat_input_container.set_agent_name(profile.display_name.lower())
            self._chat_input_container.set_custom_border(None)

    def _set_model_badges(self, active_model: str, subagent_model: str) -> None:
        try:
            self.query_one(ModelStatusBadge).set_model(active_model)
            self.query_one(SubModelBadge).set_model(subagent_model, active_model)
        except NoMatches:
            pass

    def _refresh_model_status_badge(self) -> None:
        try:
            active_model = self.agent_loop.effective_model().alias
        except ValueError:
            active_model = str(self.config.active_model or "")
        subagent_model = str(self.config.subagent_model or active_model)
        self._set_model_badges(active_model, subagent_model)

    async def _cycle_agent(self) -> None:
        new_profile = self.agent_loop.agent_manager.next_agent(
            self.agent_loop.agent_profile
        )
        self._update_profile_widgets(new_profile)
        if self._chat_input_container:
            self._chat_input_container.switching_mode = True

        loop = asyncio.get_running_loop()

        def schedule_switch() -> None:
            self._switch_agent_generation += 1
            my_gen = self._switch_agent_generation

            def switch_agent_sync() -> None:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.agent_loop.switch_agent(new_profile.name), loop
                    )
                    future.result()
                    self.agent_loop.set_approval_callback(self._approval_callback)
                    self.agent_loop.set_user_input_callback(self._user_input_callback)
                    self.agent_loop.set_rate_limit_callback(self._rate_limit_callback)
                finally:
                    if (
                        self._chat_input_container
                        and self._switch_agent_generation == my_gen
                    ):
                        self.call_from_thread(self._refresh_banner)
                        self.call_from_thread(
                            setattr, self._chat_input_container, "switching_mode", False
                        )

            self.run_worker(
                switch_agent_sync, group="switch_agent", exclusive=True, thread=True
            )

        self.call_after_refresh(schedule_switch)

    async def action_toggle_debug_console(self, **kwargs: Any) -> None:
        if self._debug_console is not None:
            await self._debug_console.remove()
            self._debug_console = None
        else:
            self._debug_console = DebugConsole(log_reader=self._log_reader)
            await self.mount(self._debug_console)

    def _get_chat_input(self) -> ChatInputContainer | None:
        input_widgets = self.query(ChatInputContainer)
        if input_widgets:
            return input_widgets.first()
        return None

    def action_interrupt_or_quit(self) -> None:
        # Ctrl+C priority ladder: clear input → second-press quit → bottom-app/voice/etc
        # no-op steps → pop last queued item (LIFO) → cancel running job → request quit.
        if (container := self._get_chat_input()) and container.value:
            container.value = ""
            return
        if self._quit_manager.is_confirmed("Ctrl+C"):
            self._force_quit()
            return
        if self._try_interrupt_no_job_steps():
            return
        if self._input_queue:
            self.run_worker(self._queue.pop_last(), exclusive=False)
            return
        if self._try_interrupt_running_job():
            return
        self._quit_manager.request_confirmation(
            "Ctrl+C", self._queue.quit_warning_extra()
        )

    def action_delete_right_or_quit(self) -> None:
        # Bottom app open = chat input unmounted; ctrl+d must never reach the
        # quit path — forward delete-right to the focused input instead.
        if self._current_bottom_app != BottomApp.Input:
            if isinstance(focused := self.focused, Input):
                focused.action_delete_right()
            return

        if (container := self._get_chat_input()) and container.value:
            if container.input_widget:
                container.input_widget.action_delete_right()
            return

        if not self.config.ask_confirmation_on_exit:
            self._force_quit()
            return

        if self._quit_manager.is_confirmed("Ctrl+D"):
            self._force_quit()
            return
        self._quit_manager.request_confirmation(
            "Ctrl+D", self._queue.quit_warning_extra()
        )

    def _emit_session_closed_for_active_session(self) -> None:
        self.agent_loop.emit_session_closed_telemetry()

    def _force_quit(self) -> None:
        if self._force_quit_task is not None and not self._force_quit_task.done():
            return
        self._force_quit_task = asyncio.create_task(self._force_quit_async())

    async def _force_quit_async(self) -> None:
        self._emit_session_closed_for_active_session()
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        if self._bash_task and not self._bash_task.done():
            self._bash_task.cancel()

        # Reap trusted teammate subprocesses on force-quit too — otherwise the
        # dominant TUI exit path orphans them (only graceful /exit cleaned up).
        await self._stop_teams()
        await self._searxng_teardown()

        self._log_reader.shutdown()
        self._narrator_manager.cancel()
        await self.agent_loop.aclose()
        try:
            await self.agent_loop.telemetry_client.aclose()
        except Exception as exc:
            logger.error(
                "Failed to close telemetry client during force quit", exc_info=exc
            )
        finally:
            self.exit(result=self._get_session_resume_info())

    def action_scroll_chat_up(self) -> None:
        try:
            self._chat_widget.scroll_relative(y=-5, animate=False)
        except Exception:
            pass

    def action_scroll_chat_down(self) -> None:
        try:
            self._chat_widget.scroll_relative(y=5, animate=False)
        except Exception:
            pass

    async def _show_dangerous_directory_warning(self) -> None:
        is_dangerous, reason = is_dangerous_directory()
        if is_dangerous:
            warning = (
                f"⚠ WARNING: {reason}\n\nRunning in this location is not recommended."
            )
            await self._mount_and_scroll(WarningMessage(warning, show_border=False))

    async def _record_vscode_extension_promo_shown(self) -> None:
        if self._vscode_extension_promo is None:
            return
        previous_count = (
            self._vscode_extension_promo.initial_state.shown_count
            if self._vscode_extension_promo.initial_state is not None
            else 0
        )
        try:
            await self._vscode_extension_promo.repository.set(
                VscodeExtensionPromoState(shown_count=previous_count + 1)
            )
        except Exception:
            logger.warning(
                "Failed to persist VSCode extension promo shown count", exc_info=True
            )

    async def _check_and_show_whats_new(self) -> None:
        if self._update_cache_repository is None:
            await self._maybe_show_vscode_extension_promo()
            return

        if not await should_show_whats_new(
            self._current_version, self._update_cache_repository
        ):
            await self._maybe_show_vscode_extension_promo()
            return

        content = load_whats_new_content()
        if content is not None:
            body = content
            plan_offer = plan_offer_cta(
                self._plan_info, vibe_base_url=self.config.vibe_base_url
            )
            if plan_offer is not None:
                body = f"{body}\n\n{plan_offer}"
            if self._show_vscode_extension_promo:
                body = f"{body}{VSCODE_EXTENSION_PROMO_WHATS_NEW_SUFFIX}"
            whats_new_message = WhatsNewMessage(body)
            if self._history_widget_indices:
                whats_new_message.add_class("after-history")
            chat = self._chat_widget
            should_anchor = chat.is_at_bottom
            await chat.mount(whats_new_message, after=self._messages_area)
            self._whats_new_message = whats_new_message
            if should_anchor:
                chat.anchor()
            if self._show_vscode_extension_promo:
                self.run_worker(
                    self._record_vscode_extension_promo_shown(), exclusive=False
                )
        else:
            await self._maybe_show_vscode_extension_promo()
        await mark_version_as_seen(self._current_version, self._update_cache_repository)

    async def _maybe_show_vscode_extension_promo(self) -> None:
        if not self._show_vscode_extension_promo:
            return
        promo_message = VscodeExtensionPromoMessage()
        chat = self._chat_widget
        should_anchor = chat.is_at_bottom
        await chat.mount(promo_message, before=self._messages_area)
        if should_anchor:
            chat.anchor()
        self.run_worker(self._record_vscode_extension_promo_shown(), exclusive=False)

    async def _resolve_plan(self) -> None:
        if self._plan_offer_gateway is None:
            self._plan_info = None
            self._refresh_command_registry()
            return

        try:
            if not self.config.is_active_model_mistral():
                self._plan_info = None
                return

            provider = self.config.get_active_provider()
            api_key = resolve_api_key_for_plan(provider)
            self._plan_info = await decide_plan_offer(api_key, self._plan_offer_gateway)
        except Exception as exc:
            logger.warning(
                "Plan-offer check failed (%s).", type(exc).__name__, exc_info=True
            )
            self._plan_info = None
        finally:
            self._refresh_command_registry()

    async def _mount_and_scroll(
        self, widget: Widget, after: Widget | None = None, before: Widget | None = None
    ) -> None:
        messages_area = self._messages_area
        is_user_initiated = isinstance(widget, (UserMessage, UserCommandMessage))
        should_anchor = is_user_initiated or self._chat_widget.is_at_bottom

        pin_anchor: Widget | None = None
        if after is None:
            pin_anchor = self._queue.pin_target(messages_area)

        with self.batch_update():
            if before is not None and before.parent is messages_area:
                await messages_area.mount(widget, before=before)
            elif after is not None and after.parent is messages_area:
                await messages_area.mount(widget, after=after)
            elif pin_anchor is not None:
                await messages_area.mount(widget, before=pin_anchor)
            else:
                await messages_area.mount(widget)
            if isinstance(widget, StreamingMessageBase):
                await widget.write_initial_content()

        self.call_after_refresh(self._try_prune)
        if should_anchor:
            self._chat_widget.anchor()

    async def _try_prune(self) -> None:
        pruned = await prune_oldest_children(
            self._messages_area, PRUNE_LOW_MARK, PRUNE_HIGH_MARK
        )
        if self._load_more.widget and not self._load_more.widget.parent:
            self._load_more.widget = None
        if pruned:
            if self._chat_widget.is_at_bottom:
                self.call_later(self._chat_widget.anchor)

    async def _refresh_windowing_from_history(self) -> None:
        if self._load_more.widget is None:
            return
        messages_area = self._messages_area
        has_backfill, tool_call_map = sync_backfill_state(
            history_messages=non_system_history_messages(self.agent_loop.messages),
            messages_children=list(messages_area.children),
            history_widget_indices=self._history_widget_indices,
            windowing=self._windowing,
        )
        self._tool_call_map = tool_call_map
        await self._load_more.set_visible(
            messages_area, visible=has_backfill, remaining=self._windowing.remaining
        )

    def _schedule_update_notification(self) -> None:
        if self._update_notifier is None or not self.config.enable_update_checks:
            return

        asyncio.create_task(self._check_update(), name="version-update-check")

    async def _check_update(self) -> None:
        if self._update_notifier is None or self._update_cache_repository is None:
            return

        try:
            await get_update_if_available(
                update_notifier=self._update_notifier,
                current_version=self._current_version,
                update_cache_repository=self._update_cache_repository,
            )
        except UpdateError as exc:
            logger.warning("Update check failed", exc_info=exc)
        except Exception as exc:
            logger.debug("Update check failed", exc_info=exc)

    def action_copy_selection(self) -> None:
        copied_text = copy_selection_to_clipboard(self, show_toast=False)
        if copied_text is not None:
            self.agent_loop.telemetry_client.send_user_copied_text(copied_text)

    _RIGHT_MOUSE_BUTTON = 3

    def on_mouse_up(self, event: MouseUp) -> None:
        if event.button == self._RIGHT_MOUSE_BUTTON:
            # Mouse tracking steals right-click from the terminal, so its
            # native context-menu paste never fires. Read the OS clipboard and
            # route it through the focused input's Paste handler (which also
            # rewrites bare image paths to @<path> tokens).
            self._paste_from_clipboard_into_focus()
            return
        if self.config.autocopy_to_clipboard:
            copied_text = copy_selection_to_clipboard(self, show_toast=False)
            if copied_text is not None:
                self._clipboard_notice.update("Selection copied to clipboard")
                self._clipboard_notice.display = True
                if self._clipboard_hide_timer is not None:
                    self._clipboard_hide_timer.stop()
                self._clipboard_hide_timer = self.set_timer(
                    2.0, lambda: setattr(self._clipboard_notice, "display", False)
                )
                self.agent_loop.telemetry_client.send_user_copied_text(copied_text)

    def _paste_from_clipboard_into_focus(self) -> None:
        self.run_worker(self._paste_from_clipboard_worker(), exclusive=False)

    async def _paste_from_clipboard_worker(self) -> None:
        target = self.focused
        if not isinstance(target, (TextArea, Input)):
            return
        text = await asyncio.to_thread(read_clipboard)
        if not text:
            return
        target.post_message(Paste(text))

    def on_app_blur(self, event: AppBlur) -> None:
        self._terminal_notifier.on_blur()
        if self._chat_input_container and self._chat_input_container.input_widget:
            self._chat_input_container.input_widget.set_app_focus(False)

    def on_app_focus(self, event: AppFocus) -> None:
        self._terminal_notifier.on_focus()
        if self._chat_input_container and self._chat_input_container.input_widget:
            self._chat_input_container.input_widget.set_app_focus(True)

    def action_open_plan_in_editor(self) -> None:
        if self.event_handler is None:
            return

        if plan_file_message := self.event_handler.plan_file_message:
            plan_file_message.open_in_editor()

    def action_suspend_with_message(self) -> None:
        if WINDOWS or self._driver is None or not self._driver.can_suspend:
            return
        with self.suspend():
            rprint(
                "Mistral Vibe has been suspended. Run [bold cyan]fg[/bold cyan] to bring Mistral Vibe back."
            )
            os.kill(os.getpid(), signal.SIGTSTP)

    def _on_driver_signal_resume(self, event: Driver.SignalResume) -> None:
        # Textual doesn't repaint after resuming from Ctrl+Z (SIGTSTP);
        # force a full layout refresh so the UI isn't garbled.
        self.refresh(layout=True)

    def _make_default_narrator_manager(self) -> NarratorManager:
        from vibe.cli.narrator_manager import NarratorManager
        from vibe.core.audio_player.audio_player import AudioPlayer

        return NarratorManager(
            config_getter=lambda: self.config,
            audio_player=AudioPlayer(),
            telemetry_client=self.agent_loop.telemetry_client,
            spend_adapter_getter=lambda: self.agent_loop.spend_adapter,
        )

    def _handle_exception(self, error: Exception) -> None:
        if not isinstance(error, WorkerFailed):
            capture_sentry_exception(
                error, fatal=True, tags={"vibe_boundary": "textual_app"}
            )
        return super()._handle_exception(error)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        error = event.worker.error
        if event.state == WorkerState.ERROR and error:
            capture_sentry_exception(
                error,
                fatal=False,
                tags={
                    "vibe_boundary": "textual_worker",
                    "worker_name": event.worker.name or "",
                },
            )


def run_textual_ui(
    agent_loop: AgentLoop,
    update_cache_repository: UpdateCacheRepository,
    startup: StartupOptions | None = None,
) -> None:
    from vibe.cli.stderr_guard import stderr_guard

    update_notifier = GitHubUpdateGateway(owner="gorxdan", repository="mistral-vibe")
    plan_offer_gateway = HttpWhoAmIGateway(base_url=agent_loop.config.console_base_url)
    vscode_extension_promo_repository = FileSystemVscodeExtensionPromoRepository()
    vscode_extension_promo = VscodeExtensionPromo(
        repository=vscode_extension_promo_repository,
        initial_state=asyncio.run(vscode_extension_promo_repository.get()),
    )

    with stderr_guard():
        app = VibeApp(
            agent_loop=agent_loop,
            startup=startup,
            update_notifier=update_notifier,
            update_cache_repository=update_cache_repository,
            plan_offer_gateway=plan_offer_gateway,
            vscode_extension_promo=vscode_extension_promo,
        )
        session_id = app.run()

    print_session_resume_message(
        session_id, agent_loop.stats, agent_loop.config.session_logging
    )
