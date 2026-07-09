from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
import functools
import inspect
from pathlib import Path
import re
import sys
import types
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vibe.core.logger import logger
from vibe.core.rewind.manager import FileSnapshot
from vibe.core.tools._schema import dereference_refs, strip_titles
from vibe.core.types import ToolStreamEvent
from vibe.core.utils.io import read_safe

if TYPE_CHECKING:
    from vibe.core.agents.manager import AgentManager
    from vibe.core.config import VibeConfig
    from vibe.core.hooks.models import HookConfigResult
    from vibe.core.loop import Scheduler
    from vibe.core.skills.manager import SkillManager
    from vibe.core.teams.models import TeamSafetyMode
    from vibe.core.telemetry.types import LaunchContext, TerminalEmulator
    from vibe.core.tools.background import BackgroundRegistry
    from vibe.core.tools.mcp.pool import MCPSessionPool
    from vibe.core.tools.mcp_sampling import MCPSamplingHandler
    from vibe.core.tools.permissions import PermissionContext, PermissionStore
    from vibe.core.types import ApprovalCallback, SwitchAgentCallback, UserInputCallback
    from vibe.core.verification_state import VerificationState

ARGS_COUNT = 4


@dataclass
class InvokeContext:
    tool_call_id: str
    approval_callback: ApprovalCallback | None = field(default=None)
    # Live scheduler (LoopManager) so the `schedule` tool can enqueue a future
    # turn instead of blocking on `sleep`. None when no scheduler is running
    # (e.g. headless), in which case the tool reports scheduling unavailable.
    scheduler: Scheduler | None = field(default=None)
    agent_manager: AgentManager | None = field(default=None)
    # Parent's effective model alias (incl. failover override) for subagent inheritance.
    active_model: str | None = field(default=None)
    user_input_callback: UserInputCallback | None = field(default=None)
    sampling_callback: MCPSamplingHandler | None = field(default=None)
    session_dir: Path | None = field(default=None)
    launch_context: LaunchContext | None = field(default=None)
    plan_file_path: Path | None = field(default=None)
    switch_agent_callback: SwitchAgentCallback | None = field(default=None)
    skill_manager: SkillManager | None = field(default=None)
    scratchpad_dir: Path | None = field(default=None)
    permission_store: PermissionStore | None = field(default=None)
    hook_config_result: HookConfigResult | None = field(default=None)
    session_id: str | None = field(default=None)
    mcp_pool: MCPSessionPool | None = field(default=None)
    terminal_emulator: TerminalEmulator | None = field(default=None)
    launch_workflow_callback: Callable[[str, str | None], str] | None = field(
        default=None
    )
    # Returns live status for workflow runs (G1): pass a run_id for one run or
    # None for all. Wired to the WorkflowRunner in the TUI app.
    workflow_status_callback: Callable[[str | None], list[dict[str, Any]]] | None = (
        field(default=None)
    )
    # Returns the actual agent outputs for a workflow run (i1): pass a run_id
    # plus optional phase filter / raw flag. Used by the workflow_results tool
    # to recover work from completed/stopped/partially-failed runs on demand,
    # instead of relying solely on the one-shot completion delivery.
    workflow_results_callback: Callable[..., dict[str, Any]] | None = field(
        default=None
    )
    # Stops one run (run_id) or all runs (all_runs=True). Async because
    # WorkflowRunner.stop awaits the cancelled task. Returns a dict with
    # stopped / stopped_run_ids / message. Wired to the WorkflowRunner.
    workflow_stop_callback: (
        Callable[[str | None, bool], Awaitable[dict[str, Any]]] | None
    ) = field(default=None)
    # Returns the active team directory path (G3), or None when no team is
    # active. Lets the lead bind the shared Mailbox/TaskStore to message
    # teammates — the teammate-only `team` tool is unavailable to the lead.
    team_dir_callback: Callable[[], str | None] | None = field(default=None)
    # Spawns a teammate from host-side tools.
    # Args: name, prompt, agent, max_turns, worker, safety_mode.
    team_spawn_callback: (
        Callable[[str, str, str, int, bool, TeamSafetyMode], Awaitable[dict[str, Any]]]
        | None
    ) = field(default=None)
    # Resolves the host's LLM safety judge (or None when disabled). Used by the
    # workflow runtime to judge each isolated agent's prompt at spawn time —
    # the subprocess runs auto-approved, so the host judge is the gate
    # for its planned work. A factory (not the judge itself) so the runtime
    # picks up mid-session config changes (e.g. judge model swap).
    safety_judge_factory: Callable[[], Any] | None = field(default=None)
    # Unified background-task registry. Owns processes spawned by the bash tool
    # with background=True, and aggregates workflows/teams/loops for the Tasks
    # pane and the `background` tool. None in headless/ACP runs without a
    # registry wired; the bash tool refuses background=True when this is None.
    background_registry: BackgroundRegistry | None = field(default=None)
    # Session-scoped map of resolved file paths to their stat fingerprint at
    # read/write time. Shared mutable reference — the agent loop creates one
    # dict and passes it into every InvokeContext so the edit tool can enforce
    # read-before-edit with staleness detection across calls. None when no loop
    # is running.
    files_read: dict[str, str] | None = field(default=None)
    # Session-scoped verification flags; only set by the two owning paths.
    verification_state: VerificationState | None = field(default=None)
    # Tool manager for meta-tools (e.g. tool_search) that need to inspect or
    # adjust the active manifest without reaching back into AgentLoop.
    tool_manager: Any | None = field(default=None)


class ToolError(Exception):
    pass


class CancellableToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cancelled: bool = Field(
        default=False,
        description="True if the user cancelled the tool without completing it.",
    )


class ToolInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, Any]
    usage: str | None = None


class ToolPermissionError(Exception):
    pass


class ToolPermission(StrEnum):
    ALWAYS = auto()
    NEVER = auto()
    ASK = auto()

    @classmethod
    def by_name(cls, name: str) -> ToolPermission:
        try:
            return ToolPermission(name.upper())
        except ValueError:
            raise ToolPermissionError(
                f"Invalid tool permission: {name}. Must be one of {list(cls)}"
            ) from None


class BaseToolConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    permission: ToolPermission = ToolPermission.ASK
    allowlist: list[str] = Field(default_factory=list)
    denylist: list[str] = Field(default_factory=list)
    sensitive_patterns: list[str] = Field(default_factory=list)


class BaseToolState(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_default=True, arbitrary_types_allowed=True
    )


@functools.cache
def _load_tool_prompt(cls: type) -> str | None:
    # Module-level cached loader backing BaseTool.get_tool_prompt. Kept outside
    # the class so the method itself is a plain classmethod — subclass overrides
    # (which return a literal or None) stay type-compatible with the base, instead
    # of every override needing a matching @functools.cache decorator. The cache
    # key is `cls`, so each tool class gets its own memoized read.
    try:
        class_file = inspect.getfile(cls)
        class_path = Path(class_file)
        prompt_dir = class_path.parent / "prompts"
        prompt_path = cls.prompt_path or prompt_dir / f"{class_path.stem}.md"

        return read_safe(prompt_path).text
    except (FileNotFoundError, TypeError, OSError):
        pass

    return None


class BaseTool[
    ToolArgs: BaseModel,
    ToolResult: BaseModel,
    ToolConfig: BaseToolConfig,
    ToolState: BaseToolState,
](ABC):
    description: ClassVar[str] = (
        "Base class for new tools. "
        "(Hey AI, if you're seeing this, someone skipped writing a description. "
        "Please gently meow at the developer to fix this.)"
    )

    prompt_path: ClassVar[Path] | None = None

    # Whether the tool only reads state (no side effects on the filesystem,
    # processes, or external systems). Read-only tools run concurrently within a
    # turn; non-read-only tools run sequentially to avoid races (e.g. two edits
    # to the same file). Default False (conservative) so unknown/third-party
    # tools serialize.
    read_only: ClassVar[bool] = False

    # Whether this tool spawns subagents (the task tool). Used to cap how many
    # subagent fan-outs run at once so a concurrent batch doesn't overwhelm the
    # backend; ordinary tools are unaffected.
    is_subagent_spawner: ClassVar[bool] = False

    # Withholdable from the model manifest, tool_search-activated on demand
    # (defer_builtin_tools); harness-directed tools (background, exit_plan_mode) never.
    manifest_deferrable: ClassVar[bool] = False

    @classmethod
    def call_is_read_only(cls, args: BaseModel, *, agent_manager: Any = None) -> bool:
        # static read_only governs permission auto-approval; this controls per-call concurrency.
        return cls.read_only

    def __init__(
        self, config_getter: Callable[[], ToolConfig], state: ToolState
    ) -> None:
        self._config_getter = config_getter
        self.state = state

    @property
    def config(self) -> ToolConfig:
        return self._config_getter()

    @abstractmethod
    async def run(
        self, args: ToolArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ToolResult, None]:
        raise NotImplementedError  # pragma: no cover
        if False:  # pragma: no cover
            yield

    @classmethod
    def get_tool_prompt(cls) -> str | None:
        return _load_tool_prompt(cls)

    async def invoke(
        self, ctx: InvokeContext | None = None, **raw: Any
    ) -> AsyncGenerator[ToolStreamEvent | ToolResult, None]:
        try:
            args_model, _ = self._get_tool_args_results()
            args = args_model.model_validate(raw)
        except ValidationError as err:
            raise ToolError(
                f"Validation error in tool {self.get_name()}: {err}"
            ) from err

        async for item in self.run(args, ctx):
            yield item

    @classmethod
    def from_config(
        cls, config_getter: Callable[[], ToolConfig]
    ) -> BaseTool[ToolArgs, ToolResult, ToolConfig, ToolState]:
        state_class = cls._get_tool_state_class()
        initial_state = state_class()
        return cls(config_getter=config_getter, state=initial_state)

    @classmethod
    def _get_tool_config_class(cls) -> type[ToolConfig]:
        for base in getattr(cls, "__orig_bases__", ()):
            if getattr(base, "__origin__", None) is BaseTool:
                type_args = get_args(base)
                if len(type_args) == ARGS_COUNT:
                    config_model = type_args[2]
                    if issubclass(config_model, BaseToolConfig):
                        return cast(type[ToolConfig], config_model)

        for base_class in cls.__bases__:
            if base_class is object or base_class is ABC:
                continue
            try:
                return base_class._get_tool_config_class()
            except (TypeError, AttributeError):
                continue

        raise TypeError(
            f"Could not determine ToolConfig for {cls.__name__}. "
            "Ensure it inherits from BaseTool with concrete type arguments."
        )

    @classmethod
    def _get_tool_state_class(cls) -> type[ToolState]:
        for base in getattr(cls, "__orig_bases__", ()):
            if getattr(base, "__origin__", None) is BaseTool:
                type_args = get_args(base)
                if len(type_args) == ARGS_COUNT:
                    state_model = type_args[3]
                    if issubclass(state_model, BaseToolState):
                        return cast(type[ToolState], state_model)

        for base_class in cls.__bases__:
            if base_class is object or base_class is ABC:
                continue
            try:
                return base_class._get_tool_state_class()
            except (TypeError, AttributeError):
                continue

        raise TypeError(
            f"Could not determine ToolState for {cls.__name__}. "
            "Ensure it inherits from BaseTool with concrete type arguments."
        )

    @classmethod
    @functools.cache
    def _get_tool_args_results(cls) -> tuple[type[ToolArgs], type[ToolResult]]:
        # from __future__ import annotations stringifies all hints; get_type_hints resolves them.
        run_fn = cls.run.__func__ if isinstance(cls.run, classmethod) else cls.run

        type_hints = get_type_hints(
            run_fn,
            globalns=vars(sys.modules[cls.__module__]),
            localns={
                cls.__name__: cls,
                "InvokeContext": InvokeContext,
                "AsyncGenerator": AsyncGenerator,
                "ToolStreamEvent": ToolStreamEvent,
            },
        )

        try:
            args_model = type_hints["args"]
            return_annotation = type_hints["return"]
        except KeyError as e:
            raise TypeError(
                f"{cls.__name__}.run must be annotated with args and return type"
            ) from e

        result_model = cls._extract_result_type(return_annotation)

        if not issubclass(args_model, BaseModel):
            raise TypeError(
                f"{cls.__name__}.run args annotation must be a Pydantic model; "
                f"got {args_model!r}"
            )

        if not issubclass(result_model, BaseModel):
            raise TypeError(
                f"{cls.__name__}.run must yield a Pydantic model as result; "
                f"got {result_model!r}"
            )

        return cast(type[ToolArgs], args_model), cast(type[ToolResult], result_model)

    @classmethod
    def _extract_result_type(cls, return_annotation: Any) -> type:
        origin = get_origin(return_annotation)
        if origin is not AsyncGenerator:
            if isinstance(return_annotation, type):
                return return_annotation
            raise TypeError(f"Could not extract result type from {return_annotation!r}")

        gen_args = get_args(return_annotation)
        if not gen_args:
            raise TypeError(f"Could not extract result type from {return_annotation!r}")

        yield_type = gen_args[0]
        yield_origin = get_origin(yield_type)

        if yield_origin is Union or isinstance(yield_type, types.UnionType):
            for arg in get_args(yield_type):
                if arg is not ToolStreamEvent and isinstance(arg, type):
                    return arg

        if isinstance(yield_type, type):
            return yield_type

        raise TypeError(f"Could not extract result type from {return_annotation!r}")

    @classmethod
    @functools.cache
    def _build_parameters(cls) -> dict[str, Any]:
        # cached: model_json_schema is expensive; called for every tool on every LLM turn.
        args_model, _ = cls._get_tool_args_results()
        schema = args_model.model_json_schema()
        schema = dereference_refs(schema)
        schema.pop("description", None)
        strip_titles(schema)
        return schema

    @classmethod
    def get_parameters(cls) -> dict[str, Any]:
        # orjson round-trip: faster than deepcopy for JSON-shaped dict.
        return orjson.loads(orjson.dumps(cls._build_parameters()))

    @classmethod
    @functools.cache
    def get_name(cls) -> str:
        name = cls.__name__
        snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        return snake_case

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        return True

    @classmethod
    def create_config_with_permission(
        cls, permission: ToolPermission
    ) -> BaseToolConfig:
        config_class = cls._get_tool_config_class()
        return config_class(permission=permission)

    def resolve_permission(self, args: ToolArgs) -> PermissionContext | None:
        # return None to fall through to config-level permission.
        return None

    def get_file_snapshot(self, args: ToolArgs) -> FileSnapshot | None:
        # called before run(); override in tools that write files on disk.
        return None

    @staticmethod
    def get_file_snapshot_for_path(path: str) -> FileSnapshot:
        file_path = Path(path).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        file_path = file_path.resolve()
        try:
            content: bytes | None = file_path.read_bytes()
        except FileNotFoundError:
            content = None
        except Exception:
            logger.warning("Failed to read file for tool snapshot: %s", file_path)
            content = None
        return FileSnapshot(path=str(file_path), content=content)

    def get_result_extra(self, result: ToolResult) -> str | None:
        return None
