from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import sys

from vibe.cli.plan_offer.decide_plan_offer import PlanInfo
from vibe.core.logger import logger

ALT_KEY = "⌥" if sys.platform == "darwin" else "Alt"


@dataclass(frozen=True)
class CommandAvailabilityContext:
    vibe_code_enabled: bool = False
    is_active_model_mistral: bool = False
    plan_info: PlanInfo | None = None

    def is_teleport_available(self) -> bool:
        return (
            self.vibe_code_enabled
            and self.is_active_model_mistral
            and self.plan_info is not None
            and self.plan_info.is_teleport_eligible()
        )


CommandAvailability = Callable[[CommandAvailabilityContext], bool]


@dataclass
class Command:
    aliases: frozenset[str]
    description: str
    handler: str
    exits: bool = False
    is_available: CommandAvailability | None = None


class CommandRegistry:
    def __init__(
        self,
        excluded_commands: list[str] | None = None,
        availability_context: CommandAvailabilityContext | None = None,
    ) -> None:
        if excluded_commands is None:
            excluded_commands = []
        self._disabled_commands = set(excluded_commands)
        self._availability_context = CommandAvailabilityContext()
        self._commands: dict[str, Command] = {}
        self._dynamic_commands: dict[str, Command] = {}
        self.refresh(availability_context)

    def _build_commands(self) -> dict[str, Command]:
        return {
            "help": Command(
                aliases=frozenset(["/help"]),
                description="Show help message",
                handler="_show_help",
            ),
            "config": Command(
                aliases=frozenset(["/config"]),
                description="Edit config settings",
                handler="_show_config",
            ),
            "model": Command(
                aliases=frozenset(["/model"]),
                description="Select active model",
                handler="_show_model",
            ),
            "thinking": Command(
                aliases=frozenset(["/thinking"]),
                description="Select thinking level",
                handler="_show_thinking",
            ),
            "effort": Command(
                aliases=frozenset(["/effort"]),
                description="Select effort mode (normal or le chaton)",
                handler="_show_effort",
            ),
            "reload": Command(
                aliases=frozenset(["/reload"]),
                description="Reload configuration, agent instructions, and skills from disk",
                handler="_reload_config",
            ),
            "clear": Command(
                aliases=frozenset(["/clear"]),
                description="Clear conversation history",
                handler="_clear_history",
            ),
            "copy": Command(
                aliases=frozenset(["/copy"]),
                description="Copy the last agent message to the clipboard",
                handler="_copy_last_agent_message",
            ),
            "log": Command(
                aliases=frozenset(["/log"]),
                description="Show path to current interaction log file",
                handler="_show_log_path",
            ),
            "debug": Command(
                aliases=frozenset(["/debug"]),
                description="Toggle debug console",
                handler="action_toggle_debug_console",
            ),
            "compact": Command(
                aliases=frozenset(["/compact"]),
                description="Compact conversation history by summarizing. Optionally pass instructions to guide the summary",
                handler="_compact_history",
            ),
            "exit": Command(
                aliases=frozenset(["/exit", "exit", "quit", ":q", ":quit"]),
                description="Exit the application",
                handler="_exit_app",
                exits=True,
            ),
            "status": Command(
                aliases=frozenset(["/status"]),
                description="Display agent statistics",
                handler="_show_status",
            ),
            "teleport": Command(
                aliases=frozenset(["/teleport"]),
                description="Teleport session to Vibe Code Web",
                handler="_teleport_command",
                is_available=CommandAvailabilityContext.is_teleport_available,
            ),
            "proxy-setup": Command(
                aliases=frozenset(["/proxy-setup"]),
                description="Configure proxy and SSL certificate settings",
                handler="_show_proxy_setup",
            ),
            "resume": Command(
                aliases=frozenset(["/resume", "/continue"]),
                description="Browse, resume, or delete saved sessions",
                handler="_show_session_picker",
            ),
            "rename": Command(
                aliases=frozenset(["/rename"]),
                description="Rename the current session",
                handler="_rename_session",
            ),
            "mcp": Command(
                aliases=frozenset(["/mcp", "/connectors"]),
                description=(
                    "Display available MCP servers and connectors. "
                    "Pass a name to list its tools"
                ),
                handler="_show_mcp",
            ),
            "voice": Command(
                aliases=frozenset(["/voice"]),
                description="Configure voice settings",
                handler="_show_voice_settings",
            ),
            "leanstall": Command(
                aliases=frozenset(["/leanstall"]),
                description="Install the Lean 4 agent (leanstral)",
                handler="_install_lean",
            ),
            "unleanstall": Command(
                aliases=frozenset(["/unleanstall"]),
                description="Uninstall the Lean 4 agent",
                handler="_uninstall_lean",
            ),
            "rewind": Command(
                aliases=frozenset(["/rewind"]),
                description="Rewind to a previous message",
                handler="_start_rewind_mode",
            ),
            "loop": Command(
                aliases=frozenset(["/loop"]),
                description=(
                    "Schedule a recurring prompt. "
                    "Use `/loop <interval> <prompt>`, `/loop list`, or `/loop cancel <id|all>`"
                ),
                handler="_loop_command",
            ),
            "workflows": Command(
                aliases=frozenset(["/workflows"]),
                description=(
                    "Alias for /tasks — the unified background-task manager. "
                    "Use /tasks (or ctrl+w) to open the pane."
                ),
                handler="_tasks_command",
            ),
            "tasks": Command(
                aliases=frozenset(["/tasks"]),
                description=(
                    "Manage background tasks (processes, workflows, teams, loops). "
                    "Use `/tasks`, `/tasks stop <id|all>`, or `/tasks list`."
                ),
                handler="_tasks_command",
            ),
            "team": Command(
                aliases=frozenset(["/team"]),
                description=(
                    "Manage agent teams. "
                    "Use `/team list`, `/team spawn <name> <prompt>`, "
                    "or `/team stop <name|all>`"
                ),
                handler="_team_command",
            ),
            "worktree": Command(
                aliases=frozenset(["/worktree"]),
                description=(
                    "Show worktree isolation status, diff, or trigger merge. "
                    "Use `/worktree status`, `/worktree diff`, or `/worktree merge`"
                ),
                handler="_worktree_command",
            ),
            "data-retention": Command(
                aliases=frozenset(["/data-retention"]),
                description="Show data retention information",
                handler="_show_data_retention",
            ),
            "theme": Command(
                aliases=frozenset(["/theme"]),
                description="Select theme",
                handler="_show_theme",
            ),
        }

    @property
    def commands(self) -> dict[str, Command]:
        return self._commands

    def refresh(
        self, availability_context: CommandAvailabilityContext | None = None
    ) -> None:
        self._availability_context = (
            availability_context or CommandAvailabilityContext()
        )
        built = self._build_commands()
        # Dynamic (discovered workflow) commands must never override a builtin
        # slash command — a project-level workflow named e.g. "exit" or "clear"
        # would otherwise hijack it.
        for name, command in self._dynamic_commands.items():
            if name in built:
                logger.warning(
                    "Workflow command '/%s' collides with a builtin; skipping", name
                )
                continue
            built[name] = command
        self._commands = {
            name: command
            for name, command in built.items()
            if name not in self._disabled_commands
            and self._is_command_available(command)
        }

    def register_dynamic(self, name: str, command: Command) -> bool:
        """Register a discovered command. Refuses to shadow a builtin slash
        command; returns True if registered, False if skipped.
        """
        if name in self._build_commands():
            logger.warning(
                "Workflow command '/%s' collides with a builtin command; skipping", name
            )
            return False
        self._dynamic_commands[name] = command
        self._commands[name] = command
        return True

    def clear_dynamic(self) -> None:
        for name in self._dynamic_commands:
            self._commands.pop(name, None)
        self._dynamic_commands.clear()
        self.refresh()

    def _is_command_available(self, command: Command) -> bool:
        if command.is_available is None:
            return True
        return command.is_available(self._availability_context)

    def _alias_map(self) -> dict[str, str]:
        return {
            alias: cmd_name
            for cmd_name, cmd in self.commands.items()
            for alias in cmd.aliases
        }

    def get(self, name: str) -> Command | None:
        return self.commands.get(name)

    def has_command(self, name: str) -> bool:
        return name in self.commands

    def get_command_name(self, user_input: str) -> str | None:
        return self._alias_map().get(user_input.lower().strip())

    def parse_command(self, user_input: str) -> tuple[str, Command, str] | None:
        parts = user_input.strip().split(None, 1)
        if not parts:
            return None

        cmd_word = parts[0]
        cmd_args = parts[1] if len(parts) > 1 else ""
        cmd_name = self.get_command_name(cmd_word)
        if cmd_name is None:
            return None

        # Bare aliases (e.g. `exit`) match only as the whole input, else a
        # message starting with one would be swallowed instead of sent.
        if not cmd_word.startswith("/") and cmd_args:
            return None

        command = self.commands[cmd_name]
        return cmd_name, command, cmd_args

    def get_help_text(self) -> str:
        lines: list[str] = [
            "### Keyboard Shortcuts",
            "",
            "- `Enter` Submit message",
            "- `Ctrl+J` / `Shift+Enter` Insert newline",
            "- `Escape` Interrupt agent or close dialogs",
            "- `Ctrl+C` Quit (or clear input if text present)",
            "- `Ctrl+G` Edit input in external editor",
            "- `Ctrl+O` Toggle tool output view",
            "- `Shift+Tab` Cycle through agents (default, plan, ...)",
            f"- `{ALT_KEY}+↑↓` / `Ctrl+P/N` Rewind to previous/next message",
            "",
            "### Special Features",
            "",
            "- `!<command>` Execute bash command directly",
            "- `@path/to/file/` Autocompletes file paths",
            "",
            "### Commands",
            "",
        ]

        for cmd in self.commands.values():
            aliases = ", ".join(f"`{alias}`" for alias in sorted(cmd.aliases))
            lines.append(f"- {aliases}: {cmd.description}")
        return "\n".join(lines)
