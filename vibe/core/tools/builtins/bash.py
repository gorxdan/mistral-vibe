from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from functools import lru_cache
import logging
import os
from pathlib import Path
import re
from typing import ClassVar, Literal, final

from pydantic import BaseModel, Field
from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from vibe.core.config import SandboxConfig
from vibe.core.scratchpad import is_scratchpad_path
from vibe.core.tools.arity import build_session_pattern
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)
from vibe.core.tools.sandbox import (
    SandboxSpec,
    build_sandbox_command,
    detect_backend,
    scrub_env,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.tools.utils import is_path_within_workdir
from vibe.core.types import ToolResultEvent, ToolStreamEvent
from vibe.core.utils import is_windows, kill_async_subprocess
from vibe.core.utils.io import decode_safe


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


logger = logging.getLogger(__name__)
_sandbox_unavailable_warned = False


def _build_sandbox_env(config: SandboxConfig) -> dict[str, str]:
    base = _get_base_env()
    if not config.scrub_env:
        return base
    return scrub_env(base, config.env_passthrough)


def _extract_commands(command: str) -> list[str]:
    parser = _get_parser()
    tree = parser.parse(command.encode("utf-8"))

    commands: list[str] = []

    def find_commands(node: Node) -> None:
        if node.type == "command":
            parts = []
            for child in node.children:
                if (
                    child.type
                    in {"command_name", "word", "string", "raw_string", "concatenation"}
                    and child.text is not None
                ):
                    parts.append(child.text.decode("utf-8"))
            if parts:
                commands.append(" ".join(parts))

        for child in node.children:
            find_commands(child)

    find_commands(tree.root_node)
    return commands


def _get_shell_executable() -> str | None:
    if is_windows():
        return None
    return os.environ.get("SHELL")


def _get_base_env() -> dict[str, str]:
    base_env = {**os.environ, "CI": "true", "NONINTERACTIVE": "1", "NO_TTY": "1"}

    if is_windows():
        base_env["GIT_PAGER"] = "more"
        base_env["PAGER"] = "more"
    else:
        base_env["TERM"] = "dumb"
        base_env["DEBIAN_FRONTEND"] = "noninteractive"
        base_env["GIT_PAGER"] = "cat"
        base_env["PAGER"] = "cat"
        base_env["LESS"] = "-FX"
        base_env["LC_ALL"] = "en_US.UTF-8"

    return base_env


_READ_ONLY_COMMANDS_WINDOWS = ["dir", "findstr", "more", "type", "ver", "where"]
_READ_ONLY_COMMANDS_POSIX = [
    "basename",
    "cat",
    "comm",
    "cut",
    "date",
    "diff",
    "dirname",
    "du",
    "file",
    "find",
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
    "pwd",
    "readlink",
    "sha1sum",
    "sha256sum",
    "shasum",
    "sort",
    "stat",
    "sum",
    "tac",
    "tail",
    "tr",
    "uname",
    "uniq",
    "wc",
    "which",
]


def default_read_only_commands() -> list[str]:
    return list(
        _READ_ONLY_COMMANDS_WINDOWS if is_windows() else _READ_ONLY_COMMANDS_POSIX
    )


def _get_default_allowlist() -> list[str]:
    common = ["cd", "echo", "git diff", "git log", "git status", "tree", "whoami"]
    return common + default_read_only_commands()


def _get_default_denylist() -> list[str]:
    common = ["gdb", "pdb", "passwd"]

    if is_windows():
        return common + ["cmd /k", "powershell -NoExit", "pwsh -NoExit", "notepad"]
    else:
        return common + [
            "nano",
            "vim",
            "vi",
            "emacs",
            "bash -i",
            "sh -i",
            "zsh -i",
            "fish -i",
            "dash -i",
            "screen",
            "tmux",
        ]


def _get_default_denylist_standalone() -> list[str]:
    common = ["python", "python3", "ipython"]

    if is_windows():
        return common + ["cmd", "powershell", "pwsh", "notepad"]
    else:
        return common + ["bash", "sh", "nohup", "vi", "vim", "emacs", "nano", "su"]


_PATH_COMMANDS = {
    "cat",
    "cd",
    "chmod",
    "chown",
    "cp",
    "head",
    "ls",
    "mkdir",
    "mv",
    "rm",
    "stat",
    "tail",
    "touch",
    "wc",
}

_FIND_EXECUTION_PREDICATES = {"-exec", "-execdir", "-ok", "-okdir"}


def _collect_outside_dirs(command_parts: list[str]) -> set[str]:
    """Collect parent directories referenced outside the workdir.

    Iterates file-manipulating commands (see _PATH_COMMANDS) and inspects
    their arguments as candidate paths. Skips flags (-r, --recursive) and
    chmod mode strings (+x). For any argument that resolves outside the current
    working directory, adds the parent directory (or the path itself when it is
    a directory) to the result set — suitable for building an OUTSIDE_DIRECTORY
    RequiredPermission.
    """
    dirs: set[str] = set()
    for part in command_parts:
        tokens = part.split()
        command = tokens[0] if tokens else None
        if not command or command not in _PATH_COMMANDS:
            continue
        for token in tokens[1:]:
            # Skip CLI flags like -r, --recursive
            if token.startswith("-"):
                continue
            # Skip chmod mode strings like +x, +rwx — they are not file paths
            if command == "chmod" and token.startswith("+"):
                continue
            # Only consider tokens that look like paths
            if not (
                token.startswith(os.sep)
                or token.startswith("~")
                or token.startswith(".")
                or os.sep in token
            ):
                continue
            if is_path_within_workdir(token):
                continue
            if is_scratchpad_path(token):
                continue
            # Resolve relative / home-relative paths, then collect parent dir
            resolved = Path(token).expanduser()
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            resolved = resolved.resolve()
            # For a directory target use the dir itself; for a file use its parent
            parent = str(resolved) if resolved.is_dir() else str(resolved.parent)
            dirs.add(parent)
    return dirs


def _matches_pattern(command: str, pattern: str) -> bool:
    return command == pattern or command.startswith(pattern + " ")


# A `sleep` of this many seconds or more is treated as a blocking wait — the
# agent should schedule a future turn instead of tying up the session.
_SLEEP_BLOCK_THRESHOLD_S = 10.0
_SLEEP_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# Match `sleep <duration>` anywhere in the raw command (any position in a
# compound command). Scans the RAW command, not the AST parts, because the
# bash parser drops bare numeric args (`sleep 300` -> `sleep`).
_SLEEP_RE = re.compile(r"(?:^|[\s;&|()`])sleep\s+(\d[\d.]*[smhd]?)")


def _sleep_token_seconds(token: str) -> float:
    if token and token[-1] in _SLEEP_UNIT_SECONDS:
        return float(token[:-1]) * _SLEEP_UNIT_SECONDS[token[-1]]
    return float(token)  # bare number = seconds; raises ValueError if non-numeric


def _blocking_sleep_reason(command: str) -> str | None:
    """Reason to deny a long blocking `sleep` in *command*, or None to allow it.

    Short sleeps (a couple seconds, e.g. waiting for a service) are fine; a long
    sleep blocks the turn, hits the command timeout, and wastes the session —
    denied with a pointer to the `schedule` tool. A non-numeric duration
    (`sleep $X`) doesn't match and is left to the normal flow.
    """
    longest = 0.0
    for match in _SLEEP_RE.finditer(command):
        try:
            longest = max(longest, _sleep_token_seconds(match.group(1)))
        except ValueError:
            continue
    if longest < _SLEEP_BLOCK_THRESHOLD_S:
        return None
    return (
        f"Blocking `sleep` (~{int(longest)}s) is not allowed — it ties up the "
        "session and hits the command timeout. Don't sleep to wait, poll, or "
        "track an interval. Use the `schedule` tool instead (e.g. "
        "action='create', interval='5m', recurring=false) to get a future turn "
        "without blocking."
    )


class BashToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    max_output_bytes: int = Field(
        default=16_000, description="Maximum bytes to capture from stdout and stderr."
    )
    default_timeout: int = Field(
        default=300, description="Default timeout for commands in seconds."
    )
    allowlist: list[str] = Field(
        default_factory=_get_default_allowlist,
        description="Command prefixes that are automatically allowed",
    )
    denylist: list[str] = Field(
        default_factory=_get_default_denylist,
        description="Command prefixes that are automatically denied",
    )
    denylist_standalone: list[str] = Field(
        default_factory=_get_default_denylist_standalone,
        description="Commands that are denied only when run without arguments",
    )
    sensitive_patterns: list[str] = Field(
        default=["sudo"],
        description="Command prefixes that always ASK regardless of arity approval.",
    )
    sandbox: SandboxConfig = Field(
        default_factory=SandboxConfig,
        description="OS-level sandbox for spawned commands (opt-in; default off).",
    )


class BashArgs(BaseModel):
    command: str
    timeout: int | None = Field(
        default=None, description="Override the default command timeout."
    )


class BashResult(BaseModel):
    command: str
    stdout: str
    stderr: str
    returncode: int


class Bash(
    BaseTool[BashArgs, BashResult, BashToolConfig, BaseToolState],
    ToolUIData[BashArgs, BashResult],
):
    description: ClassVar[str] = "Run a one-off bash command and capture its output."

    @classmethod
    def format_call_display(cls, args: BashArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"bash: {args.command}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, BashResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        return ToolResultDisplay(success=True, message=f"Ran {event.result.command}")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running command"

    @staticmethod
    def _has_find_execution_predicate(command: str) -> bool:
        """Defensive check for find -exec, -execdir, -ok, -okdir predicates."""
        if not _matches_pattern(command, "find"):
            return False
        return any(predicate in command for predicate in _FIND_EXECUTION_PREDICATES)

    @staticmethod
    def _build_command_required_permission(
        invocation_pattern: str, session_pattern: str, label: str
    ) -> RequiredPermission:
        return RequiredPermission(
            scope=PermissionScope.COMMAND_PATTERN,
            invocation_pattern=invocation_pattern,
            session_pattern=session_pattern,
            label=label,
        )

    @staticmethod
    def _build_outside_directory_permission(glob: str) -> RequiredPermission:
        return RequiredPermission(
            scope=PermissionScope.OUTSIDE_DIRECTORY,
            invocation_pattern=glob,
            session_pattern=glob,
            label=f"outside workdir ({glob})",
        )

    def _find_denylist_match(self, command: str) -> str | None:
        return next(
            (p for p in self.config.denylist if _matches_pattern(command, p)), None
        )

    def _is_standalone_denylisted(self, command: str) -> bool:
        parts = command.split()
        if not parts:
            return False
        base_command = parts[0]
        if len(parts) == 1:
            command_name = os.path.basename(base_command)
            if command_name in self.config.denylist_standalone:
                return True
            if base_command in self.config.denylist_standalone:
                return True
        return False

    def _is_allowlisted(self, command: str) -> bool:
        return any(
            _matches_pattern(command, pattern) for pattern in self.config.allowlist
        )

    def _is_sensitive(self, command: str) -> bool:
        tokens = command.split()
        if not tokens:
            return False
        return tokens[0] in self.config.sensitive_patterns

    def _resolve_guardrail_permission(
        self, command_parts: list[str]
    ) -> PermissionContext | None:
        find_execution_required: list[RequiredPermission] = []
        seen_find_execution: set[str] = set()

        for part in command_parts:
            if matched := self._find_denylist_match(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' matches denylist pattern '{matched}'. Do not attempt to run this command.",
                )
            if self._is_standalone_denylisted(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' is not allowed as a standalone command. Do not attempt to run this command.",
                )
            if not self._has_find_execution_predicate(part):
                continue
            if part in seen_find_execution:
                continue
            seen_find_execution.add(part)
            find_execution_required.append(
                self._build_command_required_permission(
                    invocation_pattern=part, session_pattern=part, label=part
                )
            )

        if not find_execution_required:
            return None
        return PermissionContext(
            permission=ToolPermission.ASK, required_permissions=find_execution_required
        )

    def _is_unconditionally_allowed(
        self, command_parts: list[str], outside_dirs: set[str]
    ) -> bool:
        if any(self._is_sensitive(part) for part in command_parts):
            return False

        if self.config.permission == ToolPermission.ALWAYS:
            return True

        return all(self._is_allowlisted(part) for part in command_parts) and (
            not outside_dirs
        )

    def _build_required_permissions(
        self, command_parts: list[str], outside_dirs: set[str]
    ) -> list[RequiredPermission]:
        required: list[RequiredPermission] = []
        seen_session: set[str] = set()

        for part in command_parts:
            if not part:
                continue
            tokens = part.split()
            if not tokens:
                continue

            is_sensitive = self._is_sensitive(part)
            if not is_sensitive and self._is_allowlisted(part):
                continue

            if is_sensitive:
                required.append(
                    self._build_command_required_permission(
                        invocation_pattern=part, session_pattern=part, label=part
                    )
                )
                continue

            session_pat = build_session_pattern(tokens)
            if session_pat in seen_session:
                continue
            seen_session.add(session_pat)
            required.append(
                self._build_command_required_permission(
                    invocation_pattern=part,
                    session_pattern=session_pat,
                    label=session_pat,
                )
            )

        for glob in sorted(str(Path(d) / "*") for d in outside_dirs):
            required.append(self._build_outside_directory_permission(glob))

        return required

    def resolve_permission(self, args: BashArgs) -> PermissionContext | None:
        if is_windows():
            return None

        if blocking_sleep := _blocking_sleep_reason(args.command):
            return PermissionContext(
                permission=ToolPermission.NEVER, reason=blocking_sleep
            )

        command_parts = _extract_commands(args.command)
        if not command_parts:
            return None

        guardrail_permission = self._resolve_guardrail_permission(command_parts)
        if (
            guardrail_permission
            and guardrail_permission.permission == ToolPermission.NEVER
        ):
            return guardrail_permission
        outside_dirs = _collect_outside_dirs(command_parts)
        if (
            self._is_unconditionally_allowed(command_parts, outside_dirs)
            and not guardrail_permission
        ):
            return PermissionContext(permission=ToolPermission.ALWAYS)

        required = self._build_required_permissions(command_parts, outside_dirs)
        if guardrail_permission:
            required.extend(guardrail_permission.required_permissions)
        if not required:
            return None

        return PermissionContext(
            permission=ToolPermission.ASK, required_permissions=required
        )

    @final
    def _build_timeout_error(self, command: str, timeout: int) -> ToolError:
        return ToolError(f"Command timed out after {timeout}s: {command!r}")

    @final
    def _build_result(
        self, *, command: str, stdout: str, stderr: str, returncode: int
    ) -> BashResult:
        if returncode != 0:
            error_msg = f"Command failed: {command!r}\n"
            error_msg += f"Return code: {returncode}"
            if stderr:
                error_msg += f"\nStderr: {stderr}"
            if stdout:
                error_msg += f"\nStdout: {stdout}"
            raise ToolError(error_msg.strip())

        return BashResult(
            command=command, stdout=stdout, stderr=stderr, returncode=returncode
        )

    def _resolve_sandbox(
        self, ctx: InvokeContext | None, command: str
    ) -> tuple[list[str] | None, Path | None, dict[str, str]]:
        """Resolve the sandbox wrapper for a command.

        Returns (argv_prefix, seatbelt_profile_path, env). argv_prefix is None
        when sandboxing is disabled or no backend is available (run as today).
        """
        sb = self.config.sandbox
        if not sb.enabled:
            return None, None, _get_base_env()

        write_roots: list[Path] = [Path.cwd()]
        if ctx is not None and ctx.scratchpad_dir is not None:
            write_roots.append(Path(ctx.scratchpad_dir))
        write_roots += [Path(d) for d in sb.write_dirs]
        # Widen writes to any out-of-tree dir the command references — those were
        # already surfaced to (and approved by) the permission gate.
        for d in _collect_outside_dirs(_extract_commands(command)):
            write_roots.append(Path(d))

        backend = detect_backend(sb.backend)
        if backend == "none":
            if sb.require_backend:
                raise ToolError(
                    "Sandbox required (require_backend=true) but no sandbox "
                    "backend is available on this platform."
                )
            global _sandbox_unavailable_warned
            if not _sandbox_unavailable_warned:
                logger.warning(
                    "bash sandbox enabled but no backend available; "
                    "running unsandboxed"
                )
                _sandbox_unavailable_warned = True
            return None, None, _get_base_env()

        env = _build_sandbox_env(sb)
        spec = SandboxSpec(
            write_roots=write_roots,
            allow_network=sb.allow_network,
            env=env,
            extra_args=sb.extra_args,
        )
        argv, _name, profile = build_sandbox_command(spec, backend)
        if argv is None:
            return None, None, _get_base_env()
        return argv, profile, env

    async def run(
        self, args: BashArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | BashResult, None]:
        timeout = args.timeout or self.config.default_timeout
        max_bytes = self.config.max_output_bytes

        proc = None
        profile_path: Path | None = None
        try:
            # start_new_session is Unix-only, on Windows it's ignored
            kwargs: dict[Literal["start_new_session"], bool] = (
                {} if is_windows() else {"start_new_session": True}
            )

            sandbox_argv, profile_path, run_env = self._resolve_sandbox(
                ctx, args.command
            )
            if sandbox_argv is not None:
                shell_exe = _get_shell_executable() or "/bin/sh"
                argv = [*sandbox_argv, shell_exe, "-c", args.command]
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *argv,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        stdin=asyncio.subprocess.DEVNULL,
                        env=run_env,
                        **kwargs,
                    )
                except (FileNotFoundError, OSError) as exc:
                    # Wrapper binary missing / namespace creation refused. Fail
                    # closed only if the user demanded a sandbox.
                    if self.config.sandbox.require_backend:
                        raise ToolError(
                            f"Sandbox wrapper failed to start: {exc}"
                        ) from exc
                    logger.warning(
                        "sandbox wrapper failed to start (%s); "
                        "falling back unsandboxed",
                        exc,
                    )
                    proc = await asyncio.create_subprocess_shell(
                        args.command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        stdin=asyncio.subprocess.DEVNULL,
                        env=_get_base_env(),
                        executable=_get_shell_executable(),
                        **kwargs,
                    )
            else:
                proc = await asyncio.create_subprocess_shell(
                    args.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=run_env,
                    executable=_get_shell_executable(),
                    **kwargs,
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except TimeoutError:
                await kill_async_subprocess(proc)
                raise self._build_timeout_error(args.command, timeout)

            stdout = (
                decode_safe(stdout_bytes, from_subprocess=True).text[:max_bytes]
                if stdout_bytes
                else ""
            )
            stderr = (
                decode_safe(stderr_bytes, from_subprocess=True).text[:max_bytes]
                if stderr_bytes
                else ""
            )

            returncode = proc.returncode or 0

            yield self._build_result(
                command=args.command,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
            )

        except (ToolError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise ToolError(f"Error running command {args.command!r}: {exc}") from exc
        finally:
            if proc is not None:
                await kill_async_subprocess(proc)
            if profile_path is not None:
                try:
                    profile_path.unlink()
                except OSError:
                    pass
