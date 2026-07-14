from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
import shlex
from typing import TYPE_CHECKING, Any, Literal

from vibe.core.orchestration import (
    OrchestrationCapabilities,
    OrchestrationController,
    OrchestrationDecision,
    OrchestrationRoute,
    OrchestrationTurnSummary,
    StrategyReceipt,
)
from vibe.core.paths import safe_cwd
from vibe.core.tools._background_delivery import (
    compact_background_completion,
    escape_background_body,
)

if TYPE_CHECKING:
    from vibe.core.agents.manager import AgentManager
    from vibe.core.config import ModelConfig, VibeConfig
    from vibe.core.llm.models import ResolvedToolCall
    from vibe.core.teams.models import TeamSafetyMode
    from vibe.core.tools.background import BackgroundRegistry
    from vibe.core.tools.manager import ToolManager
    from vibe.core.types import ImageAttachment, LLMMessage
    from vibe.core.workflows.models import (
        WorkflowLaneAttestation,
        WorkflowLaneExpectation,
    )


_SHELL_SEPARATORS = frozenset({"&&", "||", ";", "|"})
_SHELL_REDIRECTIONS = frozenset({"<", "<<", "<<<", ">", ">>", "<&", ">&"})
_OBSERVATIONAL_COMMANDS = frozenset({
    "basename",
    "cat",
    "comm",
    "cut",
    "date",
    "diff",
    "dirname",
    "du",
    "echo",
    "file",
    "fmt",
    "fold",
    "grep",
    "head",
    "join",
    "less",
    "ls",
    "md5sum",
    "more",
    "nl",
    "od",
    "paste",
    "ps",
    "pwd",
    "readlink",
    "rg",
    "sha1sum",
    "sha256sum",
    "shasum",
    "sort",
    "stat",
    "sum",
    "tac",
    "tail",
    "tr",
    "tree",
    "uname",
    "uniq",
    "wc",
    "which",
    "whoami",
})
_GIT_OBSERVATIONAL_SUBCOMMANDS = frozenset({
    "cat-file",
    "describe",
    "diff",
    "grep",
    "log",
    "ls-files",
    "ls-tree",
    "rev-list",
    "rev-parse",
    "show",
    "status",
})
_CHECK_COMMANDS = frozenset({"mypy", "pyright", "pytest"})
_DOTNET_OBSERVATIONAL_COMMANDS = frozenset({"build", "test"})
_MUTATING_FIND_FLAGS = frozenset({
    "-delete",
    "-exec",
    "-execdir",
    "-fls",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-ok",
    "-okdir",
})
_MUTATING_RG_FLAGS = frozenset({"--pre", "--pre-glob"})
_MUTATING_SORT_FLAGS = frozenset({"-o", "--output"})
_MUTATING_DATE_FLAGS = frozenset({"-s", "--set"})
_OBSERVATIONAL_ENV_VARIABLES = frozenset({
    "PYRIGHT_PYTHON_FORCE_VERSION",
    "UV_CACHE_DIR",
})
_EFFECTFUL_GIT_OPTIONS = frozenset({
    "--ext-diff",
    "--filters",
    "--open-files-in-pager",
    "--output",
    "--textconv",
})
_UV_RUN_MIN_TOKENS = 3


def is_observational_shell_command(command: str, *, background: bool = False) -> bool:
    if background or not command.strip() or "\n" in command or "\r" in command:
        return False
    tokens = _tokenize_shell(command)
    if tokens is None or _shell_tokens_are_effectful(tokens, command):
        return False
    segments = _split_shell_segments(tokens)
    return segments is not None and all(
        _shell_segment_is_observational(segment) for segment in segments
    )


def _tokenize_shell(command: str) -> list[str] | None:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>()`")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def _shell_tokens_are_effectful(tokens: list[str], command: str) -> bool:
    return (
        "$(" in command
        or "${" in command
        or any(
            token in _SHELL_REDIRECTIONS
            or "`" in token
            or token in {"(", ")", "&", "|&"}
            or token.startswith((">", "<"))
            for token in tokens
        )
    )


def _split_shell_segments(tokens: list[str]) -> list[list[str]] | None:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if not segments[-1]:
                return None
            segments.append([])
            continue
        segments[-1].append(token)
    if not segments[-1]:
        return None
    return segments


def _shell_segment_is_observational(tokens: list[str]) -> bool:
    tokens = _unwrap_shell_prefix(tokens)
    if not tokens:
        return False
    command = tokens[0].rsplit("/", maxsplit=1)[-1]
    args = tokens[1:]
    if command in _CHECK_COMMANDS:
        observational = True
    elif command == "dotnet":
        observational = bool(args) and args[0] in _DOTNET_OBSERVATIONAL_COMMANDS
    elif command == "ruff":
        observational = _ruff_is_observational(args)
    elif command == "git":
        observational = _git_is_observational(args)
    elif command == "find":
        observational = not any(flag in args for flag in _MUTATING_FIND_FLAGS)
    elif command == "rg":
        observational = not any(
            arg in _MUTATING_RG_FLAGS
            or any(arg.startswith(f"{flag}=") for flag in _MUTATING_RG_FLAGS)
            for arg in args
        )
    elif command == "sort":
        observational = not any(
            arg in _MUTATING_SORT_FLAGS
            or arg.startswith("--output=")
            or (arg.startswith("-o") and arg != "-o")
            for arg in args
        )
    elif command == "date":
        observational = not any(
            arg in _MUTATING_DATE_FLAGS
            or arg.startswith("-s")
            or arg.startswith("--set=")
            for arg in args
        )
    else:
        observational = command in _OBSERVATIONAL_COMMANDS
    return observational


def _unwrap_shell_prefix(tokens: list[str]) -> list[str]:
    remaining = list(tokens)
    while remaining:
        command = remaining[0].rsplit("/", maxsplit=1)[-1]
        if command == "env":
            remaining = remaining[1:]
            if remaining and remaining[0] == "--":
                remaining = remaining[1:]
            elif remaining and remaining[0].startswith("-"):
                return []
            continue
        if (
            command == "uv"
            and len(remaining) >= _UV_RUN_MIN_TOKENS
            and remaining[1] == "run"
        ):
            remaining = remaining[2:]
            if remaining and remaining[0] == "--":
                remaining = remaining[1:]
            continue
        if "=" in remaining[0] and not remaining[0].startswith("-"):
            variable, _, _ = remaining[0].partition("=")
            if variable not in _OBSERVATIONAL_ENV_VARIABLES:
                return []
            remaining = remaining[1:]
            continue
        return remaining
    return remaining


def _git_is_observational(args: list[str]) -> bool:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        option = args[index]
        if option in {"-C", "--git-dir", "--work-tree", "--namespace"}:
            index += 2
            continue
        if option.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
            index += 1
            continue
        return False
    if index >= len(args):
        return False
    subcommand = args[index]
    subcommand_args = args[index + 1 :]
    if subcommand not in _GIT_OBSERVATIONAL_SUBCOMMANDS:
        return False
    if any(
        arg in _EFFECTFUL_GIT_OPTIONS
        or arg.startswith(("--open-files-in-pager=", "--output="))
        for arg in subcommand_args
    ):
        return False
    return True


def _ruff_is_observational(args: list[str]) -> bool:
    mutating = {
        "--fix",
        "--fix-only",
        "--add-noqa",
        "--output-file",
        "-o",
        "--watch",
        "-w",
    }
    if any(
        arg in mutating
        or arg.startswith(("--fix=", "--fix-only=", "--add-noqa=", "--output-file="))
        for arg in args
    ):
        return False
    if "check" in args:
        return "--no-fix" in args or "--diff" in args
    if "format" in args:
        return "--check" in args or "--diff" in args
    return False


class AgentLoopOrchestrationMixin:
    _is_subagent: bool
    _headless: bool
    agent_manager: AgentManager
    background_registry: BackgroundRegistry | None
    launch_workflow_callback: (
        Callable[[str, str | None, tuple[WorkflowLaneExpectation, ...] | None], str]
        | None
    )
    team_spawn_callback: (
        Callable[[str, str, str, int, bool, TeamSafetyMode], Awaitable[dict[str, Any]]]
        | None
    )
    tool_manager: ToolManager
    scratchpad_dir: Path | None
    _pending_injected_messages: list[LLMMessage]

    @property
    def config(self) -> VibeConfig: ...

    def _init_orchestration(self) -> None:
        self._orchestration = OrchestrationController(workspace_root=safe_cwd())

    def _begin_orchestration_turn(
        self, user_prompt: str, *, continuation_id: str | None = None
    ) -> None:
        available = self.tool_manager.available_tools
        enabled = (
            self.config.is_le_chaton()
            and not self._is_subagent
            and "work_strategy" in available
        )
        self._orchestration.begin_turn(
            enabled=enabled,
            user_prompt=user_prompt,
            capabilities=OrchestrationCapabilities(
                task="task" in available,
                workflow=(
                    "launch_workflow" in available
                    and self.launch_workflow_callback is not None
                ),
                team=(
                    not self._headless
                    and "team_spawn" in available
                    and self.team_spawn_callback is not None
                    and self.background_registry is not None
                    and self.background_registry.supports_async_agent_delivery
                ),
                background_delivery=bool(
                    self.background_registry is not None
                    and self.background_registry.supports_async_agent_delivery
                ),
            ),
            continuation_id=continuation_id,
        )

    def issue_orchestration_continuation(
        self,
        *,
        route: Literal["task", "workflow", "team"] | None = None,
        launch_id: str | None = None,
    ) -> str | None:
        resolved_route = OrchestrationRoute(route) if route is not None else None
        return self._orchestration.issue_continuation(
            route=resolved_route, launch_id=launch_id
        )

    def _declare_orchestration_strategy(
        self, decision: OrchestrationDecision
    ) -> StrategyReceipt:
        return self._orchestration.declare(decision)

    def _orchestration_before_tool(self, tool_call: ResolvedToolCall) -> str | None:
        read_only = self._orchestration_call_is_observational(tool_call)
        return self._orchestration.before_tool(
            tool_call.tool_name,
            tool_call.args_dict,
            read_only=read_only,
            call_id=tool_call.call_id,
        )

    def _release_orchestration_reservation(self, tool_call: ResolvedToolCall) -> None:
        self._orchestration.release_reservation(tool_call.call_id)

    def _bound_workflow_launch_callback(
        self, tool_call: ResolvedToolCall
    ) -> Callable[[str, str | None], str] | None:
        callback = self.launch_workflow_callback
        if callback is None:
            return None
        expected: tuple[WorkflowLaneExpectation, ...] | None = None
        captured_script: str | None = None
        if tool_call.tool_name == "launch_workflow":
            captured_script = str(tool_call.args_dict.get("script", ""))
            expected = self._orchestration.workflow_lane_expectations(
                tool_call.call_id, captured_script
            )

        def launch(script: str, name: str | None) -> str:
            if captured_script is not None and script != captured_script:
                raise ValueError(
                    "Workflow script changed after its strategy lanes were bound"
                )
            return callback(script, name, expected)

        return launch

    def _record_orchestration_tool_result(
        self,
        tool_call: ResolvedToolCall,
        status: Literal["success", "failure", "skipped"],
        result: dict[str, Any] | None = None,
    ) -> None:
        read_only = self._orchestration_call_is_observational(tool_call)
        self._orchestration.record_tool_result(
            tool_call.tool_name,
            tool_call.args_dict,
            status,
            result=result,
            read_only=read_only,
            call_id=tool_call.call_id,
        )

    def _orchestration_call_is_observational(self, tool_call: ResolvedToolCall) -> bool:
        if tool_call.tool_name == "bash":
            return is_observational_shell_command(
                str(tool_call.args_dict.get("command", "")),
                background=bool(tool_call.args_dict.get("background", False)),
            )
        return tool_call.tool_class.call_is_read_only(
            tool_call.validated_args, agent_manager=self.agent_manager
        )

    def _orchestration_completion_nudge(self) -> str | None:
        return self._orchestration.completion_nudge()

    def _orchestration_completion_blocker(self) -> str | None:
        return self._orchestration.completion_blocker()

    def observe_workflow_completion(
        self,
        run_id: str,
        *,
        succeeded: bool,
        attestation: WorkflowLaneAttestation | None,
    ) -> None:
        self._orchestration.record_workflow_completion(
            run_id, succeeded=succeeded, attestation=attestation
        )

    def observe_team_completion(
        self, launch_id: str, succeeded: bool, output: str
    ) -> None:
        self._orchestration.record_team_completion(launch_id, succeeded=succeeded)
        status = "completed" if succeeded else "failed"
        body = output or "[teammate produced no output]"
        artifact_path = (
            self.scratchpad_dir / f"{launch_id}.team-result.txt"
            if self.scratchpad_dir is not None
            else None
        )
        preview = compact_background_completion(body, artifact_path)
        self.stage_injected_message(
            f"[team launch {launch_id} {status}]\n{escape_background_body(preview)}"
        )
        if self.background_registry is not None:
            self.background_registry.notify_external_completion()

    def stage_injected_message(
        self,
        content: str,
        *,
        images: list[ImageAttachment] | None = None,
        client_message_id: str | None = None,
    ) -> None: ...

    def _observe_task_completion(self, task_id: str, *, succeeded: bool) -> None:
        self._orchestration.record_task_completion(task_id, succeeded=succeeded)

    def _effective_request_model(
        self, model: ModelConfig, *, harness: bool = False
    ) -> ModelConfig:
        if harness or self._is_subagent or not self.config.is_le_chaton():
            return model
        if model.thinking == "max":
            return model
        return model.model_copy(update={"thinking": "max"})

    @property
    def orchestration_summary(self) -> OrchestrationTurnSummary:
        return self._orchestration.summary

    @property
    def orchestration_requires_le_chaton(self) -> bool:
        return self._orchestration.has_open_debt

    @property
    def has_pending_injected_messages(self) -> bool:
        return bool(self._pending_injected_messages)


__all__ = ["AgentLoopOrchestrationMixin", "is_observational_shell_command"]
