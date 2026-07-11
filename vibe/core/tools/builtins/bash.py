from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from functools import lru_cache
import os
from pathlib import Path
import re
import shlex
import time
from typing import ClassVar, Literal, final

from pydantic import BaseModel, ConfigDict, Field
from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from vibe.core.config import SandboxConfig
from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME
from vibe.core.scratchpad import is_scratchpad_path
from vibe.core.tools._team_safety import enforce_shared_ask
from vibe.core.tools.arity import build_session_pattern
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.command_safety import (
    allowlisted_argument_is_unsafe,
    destructive_command_reason,
    unwrapped_command,
)
from vibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)
from vibe.core.tools.sandbox import (
    HOST_GIT_ENV_PASSTHROUGH,
    SandboxSpec,
    build_sandbox_command,
    detect_backend,
    scrub_env,
)
from vibe.core.tools.sandbox_seccomp import build_seccomp_bpf, open_seccomp_fd
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.tools.utils import (
    is_path_within_workdir,
    isolated_scratchpad_root,
    isolated_worktree_root,
)
from vibe.core.types import ToolResultEvent, ToolStreamEvent
from vibe.core.utils import is_windows, kill_async_subprocess
from vibe.core.utils.io import decode_safe


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


_sandbox_unavailable_warned = False


def _close_fd_quietly(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _sandbox_toolchain_cache_root() -> Path:
    # Writable, persistent (stays warm), private (host ~/.cache is read-only in
    # the sandbox; a sandboxed agent must not poison the user's real caches).
    root = VIBE_HOME.path / "sandbox-cache"
    (root / "uv").mkdir(parents=True, exist_ok=True)
    (root / "pre-commit").mkdir(parents=True, exist_ok=True)
    return root


def _build_sandbox_env(config: SandboxConfig, *, host_session: bool) -> dict[str, str]:
    base = _get_base_env()
    if not config.scrub_env:
        return base
    passthrough = list(config.env_passthrough)
    if host_session:
        # Keep authenticated git/gh working for the user's own session; the
        # worker (isolated) branch passes host_session=False to stay strict.
        passthrough += sorted(HOST_GIT_ENV_PASSTHROUGH)
    return scrub_env(base, passthrough)


@lru_cache(maxsize=64)
def _extract_commands_cached(command: str) -> tuple[str, ...]:
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
            # When a command has a heredoc (or other redirect), tree-sitter
            # wraps it in a redirected_statement and the redirect is a sibling
            # of the command node, not a child.  Without this check,
            # `python3 << 'EOF'` is extracted as bare `python3` and
            # incorrectly blocked by the standalone denylist.
            if parts and node.parent and node.parent.type == "redirected_statement":
                parts.append("<redirect>")
            if parts:
                commands.append(" ".join(parts))

        for child in node.children:
            find_commands(child)

    find_commands(tree.root_node)
    return tuple(commands)


def _extract_commands(command: str) -> list[str]:
    return list(_extract_commands_cached(command))


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
    dirs: set[str] = set()
    for part in command_parts:
        tokens = part.split()
        command = tokens[0] if tokens else None
        if not command or command not in _PATH_COMMANDS:
            continue
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            if command == "chmod" and token.startswith("+"):
                continue
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
            resolved = Path(token).expanduser()
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            resolved = resolved.resolve()
            parent = str(resolved) if resolved.is_dir() else str(resolved.parent)
            dirs.add(parent)
    return dirs


def _matches_pattern(command: str, pattern: str) -> bool:
    return command == pattern or command.startswith(pattern + " ")


_SLEEP_BLOCK_THRESHOLD_S = 10.0
_SLEEP_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
# Match `sleep <duration>` anywhere in the raw command (any position in a
# compound command). Scans the RAW command, not the AST parts, because the
# bash parser drops bare numeric args (`sleep 300` -> `sleep`).
_SLEEP_RE = re.compile(r"(?:^|[\s;&|()`])sleep\s+(\d[\d.]*[smhd]?)")


def _sleep_token_seconds(token: str) -> float:
    if token and token[-1] in _SLEEP_UNIT_SECONDS:
        return float(token[:-1]) * _SLEEP_UNIT_SECONDS[token[-1]]
    return float(token)


def _blocking_sleep_reason(command: str) -> str | None:
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


# C0 control chars (minus \t=\x09 and \n=\x0a, which are legitimate whitespace)
# plus DEL. \r is the CR differential: bash treats it as a token boundary in
# some configs while tree-sitter swallows it, so the validator and the shell
# disagree on what runs. NUL and the rest have no valid use in a command.
_FORBIDDEN_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0d\x0e-\x1f\x7f]")

# Shell operators that either tree-sitter drops from its extracted command text
# (redirections) or that compose commands in ways the allowlist prefix check is
# not sound for (pipes, lists, substitution). Presence forces ASK even when the
# leading command word is allowlisted. Longer/compound forms first so the
# reported operator is the most specific.
_SIDE_EFFECTING_OPERATORS = (">>", "||", "&&", ">", "|", ";", "$(", "`")


def _forbidden_control_char_reason(command: str) -> str | None:
    match = _FORBIDDEN_CONTROL_RE.search(command)
    if match is None:
        return None
    char = match.group(0)
    label = {"\r": "carriage return (\\r)", "\x00": "NUL", "\x7f": "DEL"}.get(
        char, f"control char U+{ord(char):04X}"
    )
    return (
        f"Command contains {label}, which has no legitimate use in a single "
        "command string and can make the security validator disagree with the "
        "shell on tokenization. Rewrite the command without it."
    )


# stderr fragments that mark a failure the OS sandbox likely caused (bwrap's own
# error prefix, or the errno strings a confined write/exec produces).
_SANDBOX_FAILURE_MARKERS = (
    "bwrap:",
    "read-only file system",
    "permission denied",
    "operation not permitted",
)
_SANDBOX_BLOCKED_HINT = (
    "the OS sandbox may have blocked this. The bash tool runs sandboxed with a "
    "read-only filesystem, but the file tools (read, edit, write_file) and "
    "write_durable bypass the sandbox — switch to one of those instead of "
    "retrying bash. To grant bash write access, add the path to "
    "sandbox.write_dirs or set sandbox.enabled=false"
)


def _sandbox_failure_hint(stderr: str) -> str | None:
    low = stderr.lower()
    if any(marker in low for marker in _SANDBOX_FAILURE_MARKERS):
        return _SANDBOX_BLOCKED_HINT
    return None


def _auto_approval_blocker(command: str) -> str | None:
    for op in _SIDE_EFFECTING_OPERATORS:
        if op in command:
            return (
                f"Command uses shell operator '{op}'. The allowlist inspects "
                "only the leading command word, so it cannot soundly "
                "auto-approve composition or redirection."
            )
    try:
        shlex.split(command, posix=True)
    except ValueError:
        return (
            "Command could not be tokenized (unbalanced quotes); the "
            "validator's view may not match what the shell executes."
        )
    return None


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
        description=(
            "OS-level sandbox for spawned commands (default on where a backend "
            "exists): confines writes to the workspace and keeps .git hooks and "
            "config read-only. Set enabled=false to opt out."
        ),
    )


class BashArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    timeout: int | None = Field(
        default=None, description="Override the default command timeout."
    )
    background: bool = Field(
        default=False,
        description=(
            "Run the command in the background and return immediately instead of "
            "blocking until it finishes. Use for long-lived processes (dev servers, "
            "watchers). The process is registered in the background registry; tail "
            "its output via the `background` tool or the Tasks pane, and stop it with "
            "background action='stop'. Output goes to a log file under the scratchpad."
        ),
    )


class BashResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    command: str
    stdout: str
    stderr: str
    returncode: int
    background_task_id: str | None = None
    pid: int | None = None


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

        if event.result.background_task_id is not None:
            return ToolResultDisplay(
                success=True,
                message=(
                    f"Backgrounded {event.result.command} "
                    f"({event.result.background_task_id}, pid {event.result.pid})"
                ),
            )

        return ToolResultDisplay(success=True, message=f"Ran {event.result.command}")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running command"

    @staticmethod
    def _has_find_execution_predicate(command: str) -> bool:
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
        candidates = [command]
        if (unwrapped := unwrapped_command(command)) and unwrapped != command:
            candidates.append(unwrapped)
        return next(
            (
                pattern
                for pattern in self.config.denylist
                if any(_matches_pattern(candidate, pattern) for candidate in candidates)
            ),
            None,
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

    def _is_auto_approved(self, command: str) -> bool:
        return self._is_allowlisted(command) and (
            allowlisted_argument_is_unsafe(command) is None
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

        return all(self._is_auto_approved(part) for part in command_parts) and (
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
            if not is_sensitive and self._is_auto_approved(part):
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

        if control_char := _forbidden_control_char_reason(args.command):
            return PermissionContext(
                permission=ToolPermission.NEVER, reason=control_char
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
        blocker = _auto_approval_blocker(args.command)
        return self._resolve_auto_or_ask(
            args.command, command_parts, outside_dirs, blocker, guardrail_permission
        )

    def _resolve_auto_or_ask(
        self,
        raw_command: str,
        command_parts: list[str],
        outside_dirs: set[str],
        blocker: str | None,
        guardrail_permission: PermissionContext | None,
    ) -> PermissionContext | None:
        destructive = destructive_command_reason([*command_parts, raw_command])
        arg_reason: str | None = None
        for part in command_parts:
            if self._is_allowlisted(part):
                arg_reason = allowlisted_argument_is_unsafe(part)
                if arg_reason:
                    break
        if (
            blocker is None
            and destructive is None
            and arg_reason is None
            and self._is_unconditionally_allowed(command_parts, outside_dirs)
            and not guardrail_permission
        ):
            return PermissionContext(permission=ToolPermission.ALWAYS)

        required = self._build_required_permissions(command_parts, outside_dirs)
        if guardrail_permission:
            required.extend(guardrail_permission.required_permissions)
        reason = destructive or arg_reason or blocker
        if not required and reason is None:
            return None

        return PermissionContext(
            permission=ToolPermission.ASK, required_permissions=required, reason=reason
        )

    @final
    def _build_timeout_error(self, command: str, timeout: int) -> ToolError:
        return ToolError(f"Command timed out after {timeout}s: {command!r}")

    @final
    def _build_result(
        self,
        *,
        command: str,
        stdout: str,
        stderr: str,
        returncode: int,
        sandbox_active: bool = False,
    ) -> BashResult:
        if returncode != 0:
            error_msg = f"Command failed: {command!r}\n"
            error_msg += f"Return code: {returncode}"
            if stderr:
                error_msg += f"\nStderr: {stderr}"
            if stdout:
                error_msg += f"\nStdout: {stdout}"
            if sandbox_active and (hint := _sandbox_failure_hint(stderr)):
                error_msg += f"\nHint: {hint}"
            raise ToolError(error_msg.strip())

        return BashResult(
            command=command, stdout=stdout, stderr=stderr, returncode=returncode
        )

    def _resolve_sandbox(
        self, ctx: InvokeContext | None, command: str
    ) -> tuple[list[str] | None, Path | None, dict[str, str], int | None]:
        sb = self.config.sandbox
        iso_root = isolated_worktree_root()
        if not sb.enabled and iso_root is None:
            return None, None, _get_base_env(), None

        if iso_root is not None:
            # An isolated subagent must OS-confine bash to its worktree — the same
            # boundary the file tools enforce via enforce_isolated_confine — even
            # when the user never enabled the global sandbox. Scope writes to the
            # worktree only (no _collect_outside_dirs widening: confinement is the
            # point). Scrub the env only when the user actually opted into the
            # sandbox; a bare isolation confine adds FS bounds without touching
            # command env (git/gh creds keep working).
            write_roots: list[Path] = [iso_root]
            if (iso_scratch := isolated_scratchpad_root()) is not None:
                write_roots.append(iso_scratch)
            env = (
                _build_sandbox_env(sb, host_session=False)
                if sb.enabled
                else _get_base_env()
            )
        else:
            write_roots = [Path.cwd()]
            write_roots += [Path(d) for d in sb.write_dirs]
            # Widen writes to any out-of-tree dir the command references — those
            # were already surfaced to (and approved by) the permission gate.
            for d in _collect_outside_dirs(_extract_commands(command)):
                write_roots.append(Path(d))
            env = _build_sandbox_env(sb, host_session=True)
            # Host session only (isolated subagents stay worktree-strict): let
            # `git commit`'s pre-commit/uv gates write their cache.
            cache_root = _sandbox_toolchain_cache_root()
            write_roots.append(cache_root)
            env["UV_CACHE_DIR"] = str(cache_root / "uv")
            env["PRE_COMMIT_HOME"] = str(cache_root / "pre-commit")

        if ctx is not None and ctx.scratchpad_dir is not None:
            write_roots.append(Path(ctx.scratchpad_dir))

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
                    "bash sandbox enabled but no backend available; running unsandboxed"
                )
                _sandbox_unavailable_warned = True
            return None, None, env, None

        spec = SandboxSpec(
            write_roots=write_roots,
            allow_network=sb.allow_network,
            env=env,
            extra_args=sb.extra_args,
        )
        argv, _name, profile = build_sandbox_command(spec, backend)
        if argv is None:
            return None, None, env, None
        seccomp_fd = self._maybe_seccomp_fd(sb, backend, argv)
        return argv, profile, env, seccomp_fd

    def _maybe_seccomp_fd(
        self, sb: SandboxConfig, backend: str, argv: list[str]
    ) -> int | None:
        """Load a seccomp filter into a bwrap argv, mutating it in place.

        Inserts ``--seccomp <fd>`` before the trailing ``--`` and returns the fd
        (the caller lists it in ``pass_fds`` and closes it after spawn). Returns
        None when seccomp is off, the backend isn't bwrap, or the arch is
        unsupported — bwrap then runs with namespace isolation only.
        """
        if backend != "bwrap" or not sb.seccomp:
            return None
        bpf = build_seccomp_bpf()
        if bpf is None:
            return None
        try:
            fd = open_seccomp_fd(bpf)
        except (OSError, AttributeError) as exc:
            # AttributeError: os.memfd_create missing (non-Linux); OSError: memfd
            # unsupported by the kernel. Either way, run without the filter.
            logger.warning(
                "seccomp filter unavailable (%s); bwrap sandbox running without "
                "a syscall filter",
                exc,
            )
            return None
        # _bwrap_argv always terminates with "--"; keep --seccomp before it.
        insert_at = argv.index("--") if "--" in argv else len(argv)
        argv[insert_at:insert_at] = ["--seccomp", str(fd)]
        return fd

    async def _run_background(
        self, args: BashArgs, ctx: InvokeContext | None
    ) -> AsyncGenerator[ToolStreamEvent | BashResult, None]:
        if ctx is None:
            raise ToolError(
                "background execution is not available in this context "
                "(no background registry)"
            )
        # Logs go under the scratchpad (already a sandbox write-root) so a
        # backgrounded server stays writable even when the OS sandbox is on.
        log_root = ctx.scratchpad_dir or ctx.session_dir
        if log_root is None:
            raise ToolError(
                "background execution requires a scratchpad or session directory"
            )
        registry = ctx.background_registry
        if registry is None:
            raise ToolError(
                "background execution is not available in this context "
                "(no background registry)"
            )
        bg_dir = Path(log_root) / "bg"
        bg_dir.mkdir(parents=True, exist_ok=True)

        # Unique log name (monotonic ns avoids pid-reuse collisions). Created
        # eagerly so the Tasks pane's log tail works before the first write.
        log_path = bg_dir / f"bg-{time.monotonic_ns()}.log"
        log_path.touch()
        log_handle = log_path.open("ab", buffering=0)

        kwargs: dict[Literal["start_new_session"], bool] = (
            {} if is_windows() else {"start_new_session": True}
        )
        sandbox_argv, _profile_path, run_env, seccomp_fd = self._resolve_sandbox(
            ctx, args.command
        )
        shell_exe = _get_shell_executable()
        try:
            if sandbox_argv is not None:
                argv = [*sandbox_argv, shell_exe or "/bin/sh", "-c", args.command]
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=run_env,
                    pass_fds=(seccomp_fd,) if seccomp_fd is not None else (),
                    **kwargs,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    args.command,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=run_env,
                    executable=shell_exe,
                    **kwargs,
                )
        except (FileNotFoundError, OSError) as exc:
            log_handle.close()
            _close_fd_quietly(seccomp_fd)
            raise ToolError(
                f"Failed to start background command {args.command!r}: {exc}"
            ) from exc
        # Child has inherited the seccomp fd during spawn; the parent copy is done.
        _close_fd_quietly(seccomp_fd)

        # NOTE: on macOS with sandbox enabled, _profile_path is a temp SBPL file
        # that sandbox-exec must keep for the process's lifetime, so it is NOT
        # unlinked here. It is a small file under the (session-scoped) scratchpad
        # and is cleaned when the scratchpad is; an acceptable v1 leak for the
        # rare macOS+sandbox+background combination.
        try:
            task_id = await registry.register_process(
                proc,
                command=args.command,
                cwd=Path.cwd(),
                log_path=log_path,
                log_handle=log_handle,
            )
        except Exception:
            # Cap exceeded or other registry error. The proc was never recorded
            # (register_process's cap check runs before insertion), so the
            # registry cannot reach it — force-kill the process group (the child
            # runs in its own session via start_new_session) and close the handle.
            # Without this the orphan survives even app exit.
            log_handle.close()
            await kill_async_subprocess(proc)
            raise

        yield BashResult(
            command=args.command,
            stdout="",
            stderr="",
            returncode=-1,  # sentinel: still running (finalized async by registry)
            background_task_id=task_id,
            pid=proc.pid,
        )

    async def run(
        self, args: BashArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | BashResult, None]:
        timeout = args.timeout or self.config.default_timeout
        max_bytes = self.config.max_output_bytes
        await enforce_shared_ask(
            self.get_name(),
            args.command,
            self.resolve_permission(args),
            self.config.permission,
        )

        # Returns BEFORE foreground try/finally below — finally never kills backgrounds.
        if args.background:
            async for item in self._run_background(args, ctx):
                yield item
            return

        proc = None
        profile_path: Path | None = None
        seccomp_fd: int | None = None
        ran_sandboxed = False
        try:
            kwargs: dict[Literal["start_new_session"], bool] = (
                {} if is_windows() else {"start_new_session": True}
            )

            sandbox_argv, profile_path, run_env, seccomp_fd = self._resolve_sandbox(
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
                        pass_fds=(seccomp_fd,) if seccomp_fd is not None else (),
                        **kwargs,
                    )
                    ran_sandboxed = True
                except (FileNotFoundError, OSError) as exc:
                    if self.config.sandbox.require_backend:
                        raise ToolError(
                            f"Sandbox wrapper failed to start: {exc}"
                        ) from exc
                    logger.warning(
                        "sandbox wrapper failed to start (%s); falling back "
                        "unsandboxed. Filesystem containment is lost but the "
                        "scrubbed environment is preserved (no secrets re-injected).",
                        exc,
                    )
                    # Keep the already-scrubbed run_env: a user who enabled the
                    # sandbox/scrub_env to drop secrets must not lose that
                    # protection just because the containment backend failed.
                    proc = await asyncio.create_subprocess_shell(
                        args.command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        stdin=asyncio.subprocess.DEVNULL,
                        env=run_env,
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
                raise self._build_timeout_error(args.command, timeout) from None

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

            yield self._build_result(
                command=args.command,
                stdout=stdout,
                stderr=stderr,
                returncode=proc.returncode or 0,
                sandbox_active=ran_sandboxed,
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
            _close_fd_quietly(seccomp_fd)
