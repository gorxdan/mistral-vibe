from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import ClassVar, Literal, NamedTuple, final

from pydantic import BaseModel, ConfigDict, Field
from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from vibe.core.config import SandboxConfig
from vibe.core.config.harness_files import get_harness_files_manager
from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME
from vibe.core.tools._command_tokens import split_bash_tokens
from vibe.core.tools._model_write_policy import (
    hard_control_plane_command_reason,
    model_protected_roots,
    verification_protected_roots,
)
from vibe.core.tools._shell import (
    get_autoapproved_shell_env,
    get_base_shell_env,
    get_bash_executable,
)
from vibe.core.tools._team_safety import enforce_shared_ask
from vibe.core.tools.arity import build_session_pattern
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolAuthorizationSource,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.builtins._bash_command_policy import (
    CommandPolicyAnalysis,
    analyze_command_policy,
    auto_approval_blocker,
    command_analysis_preflight_denial,
    command_uses_unmanaged_background,
    execution_match_candidates,
    harden_automated_command,
    masked_verification_status_reason,
    shell_input_payloads,
)
from vibe.core.tools.builtins._bash_path_policy import (
    collect_outside_dirs as _collect_outside_dirs,
)
from vibe.core.tools.command_safety import (
    allowlisted_argument_is_unsafe,
    destructive_command_reason,
)
from vibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
    authorization_context_fingerprint,
)
from vibe.core.tools.sandbox import (
    HOST_GIT_ENV_PASSTHROUGH,
    SandboxSpec,
    build_sandbox_command,
    resolve_backend,
    scrub_env,
    strict_read_hidden_roots,
)
from vibe.core.tools.sandbox_seccomp import build_seccomp_bpf, open_seccomp_fd
from vibe.core.tools.tool_result_store import ToolResultStore
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.tools.utils import isolated_scratchpad_root, isolated_worktree_root
from vibe.core.types import ToolResultEvent, ToolStreamEvent
from vibe.core.utils import is_windows, kill_async_subprocess
from vibe.core.utils.io import decode_safe, read_safe


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


_sandbox_unavailable_warned = False
_BASH_EXECUTABLE_UNAVAILABLE = (
    "A compatible Bash executable is unavailable; command execution is disabled."
)
_AUTOMATED_AUTHORIZATION_SOURCES = frozenset({
    ToolAuthorizationSource.POLICY,
    ToolAuthorizationSource.SAFETY_JUDGE,
    ToolAuthorizationSource.BYPASS,
})


def _close_fd_quietly(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _unlink_quietly(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
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


class _ModelSandboxControl(NamedTuple):
    protected_roots: tuple[Path, ...]
    topology_state: str | None
    strict: bool
    protect_git_metadata: bool
    candidate_readonly: bool


def _model_sandbox_control(
    ctx: InvokeContext | None, iso_root: Path | None
) -> _ModelSandboxControl:
    verification_state = ctx.verification_state if ctx is not None else None
    protected_roots = verification_protected_roots(verification_state)
    recipe = (
        verification_state.trusted_recipe if verification_state is not None else None
    )
    topology = recipe.config.execution_topology if recipe is not None else None
    task_contract = ctx.task_contract if ctx is not None else None
    autoapprove = bool(
        ctx is not None
        and ctx.agent_manager is not None
        and ctx.agent_manager.config.bypass_tool_permissions
    )
    verifier = bool(
        ctx is not None
        and ctx.agent_manager is not None
        and getattr(ctx.agent_manager.config, "system_prompt_id", None) == "verifier"
    )
    bypass_authorization = bool(
        ctx is not None and ctx.authorization_source is ToolAuthorizationSource.BYPASS
    )
    return _ModelSandboxControl(
        protected_roots=protected_roots,
        topology_state=topology.state if topology is not None else None,
        strict=bool(
            iso_root is not None
            or task_contract
            or protected_roots
            or autoapprove
            or verifier
            or bypass_authorization
        ),
        protect_git_metadata=bool(
            iso_root is not None or task_contract or protected_roots or verifier
        ),
        candidate_readonly=verifier,
    )


def _strict_model_env(
    ctx: InvokeContext | None, *, restrict_home_tools: bool
) -> dict[str, str]:
    env = scrub_env(_get_base_env(), [])
    path_value = env.get("PATH", "")
    resolved_path = (
        _managed_path_entries(path_value)
        if restrict_home_tools
        else _resolved_path_entries(path_value)
    )
    env.update({
        "VIBE_STRICT_MODEL_CONTROL": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "UV_CACHE_DIR": "/tmp/vibe-toolchain-cache/uv",
        "PRE_COMMIT_HOME": "/tmp/vibe-toolchain-cache/pre-commit",
        "XDG_CACHE_HOME": "/tmp/vibe-toolchain-cache/xdg",
        "HOME": (
            str(ctx.scratchpad_dir)
            if ctx is not None and ctx.scratchpad_dir is not None
            else "/tmp/vibe-model-home"
        ),
    })
    env["PATH"] = os.pathsep.join(resolved_path)
    return env


def _resolved_path_entries(value: str) -> list[str]:
    entries: list[str] = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        try:
            resolved = Path(entry).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if resolved.is_dir():
            entries.append(str(resolved))
    return entries


def _managed_path_entries(value: str) -> list[str]:
    home = Path.home().resolve()
    repository_bin = _repository_root(Path.cwd()) / ".venv" / "bin"
    uv = shutil.which("uv", path=value)
    allowed_home_directories = {repository_bin.resolve()}
    if uv is not None:
        allowed_home_directories.add(Path(uv).resolve().parent)
    return [
        entry
        for entry in _resolved_path_entries(value)
        if not Path(entry).is_relative_to(home)
        or Path(entry) in allowed_home_directories
    ]


def _strict_read_roots(
    write_roots: list[Path],
    control: _ModelSandboxControl,
    env: dict[str, str],
    iso_root: Path | None,
) -> list[Path]:
    home = Path.home().resolve()
    repository = (iso_root or _repository_root(Path.cwd())).resolve()
    roots = {repository, *write_roots, *control.protected_roots}
    if control.topology_state is None:
        try:
            roots.update(get_harness_files_manager().project_roots)
        except RuntimeError:
            pass
    if git_common := _linked_git_common(repository):
        roots.add(git_common)
    for entry in env.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        directory = Path(entry).resolve()
        if directory != home:
            roots.add(directory)
    python_runtime = _candidate_python_runtime(repository, home)
    if python_runtime is not None:
        roots.add(python_runtime)
    unsafe = next(
        (
            root
            for root in roots
            if home == root.resolve() or home.is_relative_to(root.resolve())
        ),
        None,
    )
    if unsafe is not None:
        raise ToolError(
            "Strict model control cannot expose a filesystem root that contains "
            f"the host home: {unsafe}"
        )
    return sorted(roots)


def _candidate_python_runtime(repository: Path, home: Path) -> Path | None:
    python = repository / ".venv" / "bin" / "python"
    try:
        resolved = python.resolve(strict=True)
        uv_python_root = (home / ".local" / "share" / "uv" / "python").resolve()
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(uv_python_root):
        return None
    return resolved.parent.parent


def _repository_root(directory: Path) -> Path:
    resolved = directory.resolve()
    return next(
        (
            parent
            for parent in (resolved, *resolved.parents)
            if (parent / ".git").exists()
        ),
        resolved,
    )


def _linked_git_common(repository: Path) -> Path | None:
    dotgit = repository / ".git"
    if not dotgit.is_file() or dotgit.is_symlink():
        return None
    try:
        line = next(
            item
            for item in read_safe(dotgit).text.splitlines()
            if item.startswith("gitdir:")
        )
    except (OSError, StopIteration):
        return None
    gitdir = Path(line.partition(":")[2].strip())
    if not gitdir.is_absolute():
        gitdir = repository / gitdir
    resolved = gitdir.resolve()
    return resolved.parent.parent if resolved.parent.name == "worktrees" else resolved


def _sandbox_write_scope(
    config: SandboxConfig,
    ctx: InvokeContext | None,
    command: str,
    iso_root: Path | None,
    control: _ModelSandboxControl,
) -> tuple[list[Path], dict[str, str]]:
    if iso_root is not None:
        write_roots = [iso_root]
        if (iso_scratch := isolated_scratchpad_root()) is not None:
            write_roots.append(iso_scratch)
        env = _build_sandbox_env(config, host_session=False)
    else:
        # A topology-bound shell is read-only against the candidate.  File
        # mutations use Edit/Write so allowed-path checks remain structural.
        readonly_candidate = (
            control.topology_state is not None or control.candidate_readonly
        )
        write_roots = [] if readonly_candidate else [Path.cwd()]
        if not readonly_candidate:
            write_roots += [Path(directory) for directory in config.write_dirs]
        if not control.strict:
            write_roots += [
                Path(directory)
                for directory in _collect_outside_dirs(_extract_commands(command))
            ]
        env = _build_sandbox_env(config, host_session=True)
        if not control.strict:
            cache_root = _sandbox_toolchain_cache_root()
            write_roots.append(cache_root)
            env["UV_CACHE_DIR"] = str(cache_root / "uv")
            env["PRE_COMMIT_HOME"] = str(cache_root / "pre-commit")

    if control.strict:
        env = _strict_model_env(ctx, restrict_home_tools=True)
    if ctx is not None and ctx.scratchpad_dir is not None:
        write_roots.append(Path(ctx.scratchpad_dir))
    return write_roots, env


@lru_cache(maxsize=64)
def _extract_commands_cached(command: str) -> tuple[str, ...]:
    parser = _get_parser()
    tree = parser.parse(command.encode("utf-8"))

    commands: list[str] = []

    def find_commands(root: Node) -> None:
        pending = [root]
        while pending:
            node = pending.pop()
            if node.type == "command":
                parts = []
                for child in node.children:
                    if (
                        child.type
                        in {
                            "command_name",
                            "variable_assignment",
                            "ansi_c_string",
                            "expansion",
                            "simple_expansion",
                            "word",
                            "number",
                            "string",
                            "raw_string",
                            "concatenation",
                        }
                        and child.text is not None
                    ):
                        parts.append(child.text.decode("utf-8"))
                if parts and node.parent and node.parent.type == "redirected_statement":
                    parts.append("<redirect>")
                if parts:
                    commands.append(" ".join(parts))
            pending.extend(reversed(node.children))

    find_commands(tree.root_node)
    for body in shell_input_payloads(command):
        find_commands(parser.parse(body.encode("utf-8")).root_node)
    return tuple(dict.fromkeys(commands))


def _extract_commands(command: str) -> list[str]:
    return list(_extract_commands_cached(command))


def _analyze_command_policy(command: str) -> CommandPolicyAnalysis:
    if denial := command_analysis_preflight_denial(command):
        return CommandPolicyAnalysis(denial=denial)
    command_parts = _extract_commands(command)
    analysis = analyze_command_policy(
        command, command_parts, extract_commands=_extract_commands
    )
    if analysis.denial is not None:
        return analysis
    denial = masked_verification_status_reason(
        command, command_parts, extract_commands=_extract_commands
    )
    return CommandPolicyAnalysis(denial=denial, deferral=analysis.deferral)


_get_shell_executable = get_bash_executable


def _runtime_shell_executable() -> str | None:
    executable = _get_shell_executable()
    if executable is None and not is_windows():
        raise ToolError(_BASH_EXECUTABLE_UNAVAILABLE)
    return executable


def _autoapproved_shell_env(ctx: InvokeContext | None) -> dict[str, str]:
    tmpdir = (
        str(Path(ctx.scratchpad_dir).resolve())
        if ctx is not None and ctx.scratchpad_dir is not None
        else "/tmp"
    )
    return get_autoapproved_shell_env(tmpdir=tmpdir)


def _automated_authorization_is_current(
    source: ToolAuthorizationSource,
    permission: PermissionContext | None,
    configured: ToolPermission,
) -> bool:
    effective = permission.permission if permission is not None else configured
    if source is ToolAuthorizationSource.POLICY:
        return effective is ToolPermission.ALWAYS
    return effective is not ToolPermission.NEVER and not (
        permission is not None and permission.requires_explicit_user_approval
    )


def _stored_authorization_is_current(
    ctx: InvokeContext, permission: PermissionContext | None, configured: ToolPermission
) -> bool:
    effective = permission.permission if permission is not None else configured
    if effective is ToolPermission.ALWAYS:
        return True
    if (
        effective is not ToolPermission.ASK
        or permission is None
        or permission.requires_explicit_user_approval
        or not permission.required_permissions
        or ctx.permission_store is None
    ):
        return False
    return all(
        ctx.permission_store.covers("bash", required)
        for required in permission.required_permissions
    )


def _trusted_execution_required(
    args: BashArgs,
    ctx: InvokeContext | None,
    permission: PermissionContext | None,
    configured: ToolPermission,
) -> bool:
    source = ctx.authorization_source if ctx is not None else None
    if source is not None:
        current = permission or PermissionContext(permission=configured)
        if (
            ctx is None
            or ctx.authorization_fingerprint is None
            or ctx.authorization_fingerprint
            != authorization_context_fingerprint("bash", args, current)
        ):
            raise ToolError(
                "Bash authorization context changed after approval; submit the "
                "command again so it can be re-evaluated."
            )
    if source is ToolAuthorizationSource.BYPASS and (
        ctx is None
        or ctx.agent_manager is None
        or not getattr(ctx.agent_manager.config, "bypass_tool_permissions", False)
    ):
        raise ToolError(
            "Bash auto-approve authority changed before execution; submit the "
            "command again so it can be re-evaluated."
        )
    if source is not None and source in _AUTOMATED_AUTHORIZATION_SOURCES:
        if not _automated_authorization_is_current(source, permission, configured):
            raise ToolError(
                "Bash authorization changed after approval; submit the command "
                "again so it can be re-evaluated."
            )
        return True
    if source is ToolAuthorizationSource.STORED_USER:
        if ctx is None or not _stored_authorization_is_current(
            ctx, permission, configured
        ):
            raise ToolError(
                "Stored Bash authorization no longer covers this command; "
                "submit it again for approval."
            )
        return False
    effective = permission.permission if permission is not None else configured
    if source is not None:
        if effective is ToolPermission.NEVER:
            raise ToolError(
                "Bash was disabled after approval; the command was not started."
            )
        return False
    if effective is ToolPermission.NEVER:
        reason = permission.reason if permission is not None else None
        raise ToolError(reason or "Bash execution is disabled by configuration.")
    return effective is ToolPermission.ALWAYS


def _hard_guardrail_reason(command_parts: list[str], raw_command: str) -> str | None:
    if not is_windows() and _get_shell_executable() is None:
        return _BASH_EXECUTABLE_UNAVAILABLE
    return hard_control_plane_command_reason(command_parts, raw_command)


_get_base_env = get_base_shell_env


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
    common = ["echo", "git log", "tree", "whoami"]
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


_FIND_EXECUTION_PREDICATES = {"-exec", "-execdir", "-ok", "-okdir"}


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


def _format_full_output(
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
    stdout_bytes: int,
    stderr_bytes: int,
) -> str:
    return (
        f"command: {command}\n"
        f"returncode: {returncode}\n"
        "format: decoded text (newline-normalized; not byte-exact)\n"
        f"stdout_bytes: {stdout_bytes}\n"
        f"stdout_chars: {len(stdout)}\n"
        f"stderr_bytes: {stderr_bytes}\n"
        f"stderr_chars: {len(stderr)}\n"
        f"--- stdout ---\n{stdout}\n"
        f"--- stderr ---\n{stderr}"
    )


def _persist_full_output(
    ctx: InvokeContext | None,
    *,
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
    stdout_bytes: int,
    stderr_bytes: int,
) -> Path | None:
    if ctx is None:
        return None
    content = _format_full_output(
        command, returncode, stdout, stderr, stdout_bytes, stderr_bytes
    )
    isolated_root = isolated_worktree_root()
    isolated_scratch = isolated_scratchpad_root()
    roots = (
        (ctx.scratchpad_dir, ctx.session_dir)
        if isolated_root is not None
        else (ctx.session_dir, ctx.scratchpad_dir)
    )
    previous_root: Path | None = None
    for root in roots:
        if root is None or root == previous_root:
            continue
        previous_root = root
        if isolated_root is not None:
            resolved = root.resolve()
            if not resolved.is_relative_to(isolated_root) and (
                isolated_scratch is None
                or not resolved.is_relative_to(isolated_scratch)
            ):
                continue
        store = ToolResultStore(lambda root=root: root)
        if path := store.persist(f"{ctx.tool_call_id}-bash-full", content):
            return path
    return None


def _tiny_truncation_preview(stream: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if max_chars == 1:
        return "…"
    preview_chars = max_chars - 1
    head_chars = (preview_chars * 3 + 3) // 4
    tail_chars = preview_chars - head_chars
    tail = stream[-tail_chars:] if tail_chars else ""
    return f"{stream[:head_chars]}…{tail}"


def _truncate_stream(
    stream: str,
    *,
    label: str,
    max_chars: int,
    total_bytes: int,
    artifact_path: Path | None,
) -> str:
    if len(stream) <= max_chars:
        return stream

    prefix = (
        f"…[{label} truncated: {len(stream):,} characters / "
        f"{total_bytes:,} bytes total; "
    )
    recovery = "full decoded-text artifact unavailable"
    if artifact_path is not None:
        recovery = (
            f"full decoded text persisted to {artifact_path}; use the `read` tool "
            "to retrieve it"
        )
    marker = f"{prefix}{recovery}]…"
    if artifact_path is not None and len(marker) > max_chars:
        marker = f"{prefix}full decoded text saved; see full_output_path]…"
    if len(marker) > max_chars:
        return _tiny_truncation_preview(stream, max_chars)

    preview_chars = max_chars - len(marker) - 4
    if preview_chars <= 0:
        return marker
    head_chars = preview_chars * 3 // 4
    tail_chars = preview_chars - head_chars
    head = stream[:head_chars]
    tail = stream[-tail_chars:] if tail_chars else ""
    return "\n\n".join(part for part in (head, marker, tail) if part)


class _ShapedOutput(NamedTuple):
    stdout: str
    stderr: str
    full_output_path: Path | None


class _OutputCaptureLimitExceeded(RuntimeError):
    pass


async def _communicate_limited(
    proc: asyncio.subprocess.Process, *, max_capture_bytes: int
) -> tuple[bytes, bytes]:
    stdout = bytearray()
    stderr = bytearray()
    captured = 0

    async def drain(
        stream: asyncio.StreamReader | None, destination: bytearray
    ) -> None:
        nonlocal captured
        if stream is None:
            return
        while chunk := await stream.read(64 * 1024):
            if captured + len(chunk) > max_capture_bytes:
                raise _OutputCaptureLimitExceeded
            captured += len(chunk)
            destination.extend(chunk)

    tasks = [
        asyncio.create_task(drain(proc.stdout, stdout)),
        asyncio.create_task(drain(proc.stderr, stderr)),
    ]
    try:
        await asyncio.gather(*tasks)
        await proc.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return bytes(stdout), bytes(stderr)


def _shape_output(
    ctx: InvokeContext | None,
    *,
    command: str,
    returncode: int,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    max_chars: int,
) -> _ShapedOutput:
    stdout_full = (
        decode_safe(stdout_bytes, from_subprocess=True).text if stdout_bytes else ""
    )
    stderr_full = (
        decode_safe(stderr_bytes, from_subprocess=True).text if stderr_bytes else ""
    )
    artifact_path = None
    if len(stdout_full) > max_chars or len(stderr_full) > max_chars:
        artifact_path = _persist_full_output(
            ctx,
            command=command,
            returncode=returncode,
            stdout=stdout_full,
            stderr=stderr_full,
            stdout_bytes=len(stdout_bytes),
            stderr_bytes=len(stderr_bytes),
        )
    stdout = _truncate_stream(
        stdout_full,
        label="stdout",
        max_chars=max_chars,
        total_bytes=len(stdout_bytes),
        artifact_path=artifact_path,
    )
    stderr = _truncate_stream(
        stderr_full,
        label="stderr",
        max_chars=max_chars,
        total_bytes=len(stderr_bytes),
        artifact_path=artifact_path,
    )
    return _ShapedOutput(stdout, stderr, artifact_path)


def _command_syntax_denial(command: str) -> PermissionContext | None:
    reason = _blocking_sleep_reason(command) or _forbidden_control_char_reason(command)
    if reason is None:
        return None
    return PermissionContext(permission=ToolPermission.NEVER, reason=reason)


class BashToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    max_output_bytes: int = Field(
        default=16_000,
        ge=0,
        description=(
            "Maximum decoded-text characters from each stream to show inline. "
            "This is a character cap retained under its legacy config key, not a "
            "raw-byte capture limit."
        ),
    )
    max_capture_bytes: int = Field(
        default=16_000_000,
        ge=1_024,
        le=256_000_000,
        description=(
            "Maximum combined raw stdout/stderr bytes retained for a foreground "
            "command. The process tree is terminated when this hard bound is "
            "exceeded."
        ),
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
    full_output_path: str | None = None
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
    def model_approval_deferral_reason(args: BashArgs) -> str | None:
        return _analyze_command_policy(args.command).deferral

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
        candidates = execution_match_candidates(
            command, extract_commands=_extract_commands
        )
        return next(
            (
                pattern
                for pattern in self.config.denylist
                if any(_matches_pattern(candidate, pattern) for candidate in candidates)
            ),
            None,
        )

    def _is_standalone_denylisted(self, command: str, raw_command: str) -> bool:
        pending = [_get_parser().parse(raw_command.encode("utf-8")).root_node]
        while pending:
            node = pending.pop()
            if node.type in {
                "file_redirect",
                "heredoc_redirect",
                "herestring_redirect",
            }:
                return False
            pending.extend(reversed(node.named_children))
        for candidate in execution_match_candidates(
            command, extract_commands=_extract_commands
        ):
            try:
                parts = split_bash_tokens(candidate)
            except ValueError:
                continue
            if len(parts) != 1:
                continue
            base_command = parts[0]
            normalized = os.path.basename(base_command)
            if (
                normalized in self.config.denylist_standalone
                or base_command in self.config.denylist_standalone
            ):
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
        return any(
            _matches_pattern(candidate, pattern)
            for candidate in execution_match_candidates(
                command, extract_commands=_extract_commands
            )
            for pattern in self.config.sensitive_patterns
        )

    def _resolve_guardrail_permission(
        self,
        command_parts: list[str],
        raw_command: str,
        policy_analysis: CommandPolicyAnalysis,
    ) -> PermissionContext | None:
        if policy_denial := policy_analysis.denial:
            return PermissionContext(
                permission=ToolPermission.NEVER, reason=policy_denial
            )
        find_execution_required: list[RequiredPermission] = []
        seen_find_execution: set[str] = set()

        for part in command_parts:
            if matched := self._find_denylist_match(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' matches denylist pattern '{matched}'. Do not attempt to run this command.",
                )
            if self._is_standalone_denylisted(part, raw_command):
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

    def _resolve_permission_from_parts(
        self,
        args: BashArgs,
        command_parts: list[str],
        policy_analysis: CommandPolicyAnalysis | None,
    ) -> PermissionContext | None:
        if hard_denial := _hard_guardrail_reason(command_parts, args.command):
            return PermissionContext(
                permission=ToolPermission.NEVER, reason=hard_denial
            )
        if policy_analysis is None:
            policy_analysis = _analyze_command_policy(args.command)
        guardrail_permission = self._resolve_guardrail_permission(
            command_parts, args.command, policy_analysis
        )
        if (
            guardrail_permission
            and guardrail_permission.permission == ToolPermission.NEVER
        ):
            return guardrail_permission
        outside_dirs = (
            set()
            if is_windows()
            else _collect_outside_dirs(command_parts, args.command)
        )
        forced_permission: PermissionContext | None = None
        if self.config.permission is ToolPermission.NEVER:
            forced_permission = PermissionContext(
                permission=ToolPermission.NEVER,
                reason="The bash tool is disabled by configuration.",
            )
        elif authority_reason := policy_analysis.deferral:
            required_permissions: list[RequiredPermission] = []
            if guardrail_permission is not None:
                required_permissions.extend(
                    self._build_required_permissions(command_parts, outside_dirs)
                )
                required_permissions.extend(guardrail_permission.required_permissions)
            forced_permission = PermissionContext(
                permission=ToolPermission.ASK,
                required_permissions=required_permissions,
                reason=authority_reason,
                requires_explicit_user_approval=True,
            )
        if forced_permission is not None:
            return forced_permission
        if not command_parts:
            blocker = auto_approval_blocker(args.command)
            return (
                PermissionContext(permission=ToolPermission.ASK, reason=blocker)
                if blocker is not None
                else None
            )
        if is_windows():
            return None
        blocker = auto_approval_blocker(args.command)
        return self._resolve_auto_or_ask(
            args.command, command_parts, outside_dirs, blocker, guardrail_permission
        )

    def resolve_permission(self, args: BashArgs) -> PermissionContext | None:
        if preflight_denial := command_analysis_preflight_denial(args.command):
            return PermissionContext(
                permission=ToolPermission.NEVER, reason=preflight_denial
            )
        if syntax_denial := _command_syntax_denial(args.command):
            return syntax_denial

        command_parts = _extract_commands(args.command)
        return self._resolve_permission_from_parts(args, command_parts, None)

    def _resolve_execution_permission(
        self,
        args: BashArgs,
        command_parts: list[str],
        policy_analysis: CommandPolicyAnalysis,
    ) -> PermissionContext | None:
        resolver = self.resolve_permission
        if getattr(resolver, "__func__", None) is Bash.resolve_permission:
            return self._resolve_permission_from_parts(
                args, command_parts, policy_analysis
            )
        return resolver(args)

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
        full_output_path: Path | None = None,
        sandbox_active: bool = False,
    ) -> BashResult:
        if returncode != 0:
            error_msg = f"Command failed: {command!r}\n"
            error_msg += f"Return code: {returncode}"
            if stderr:
                error_msg += f"\nStderr: {stderr}"
            if stdout:
                error_msg += f"\nStdout: {stdout}"
            if full_output_path is not None:
                error_msg += f"\nFull decoded-text output: {full_output_path}"
            if sandbox_active and (hint := _sandbox_failure_hint(stderr)):
                error_msg += f"\nHint: {hint}"
            raise ToolError(error_msg.strip())

        return BashResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            full_output_path=str(full_output_path) if full_output_path else None,
        )

    def _resolve_sandbox(
        self,
        ctx: InvokeContext | None,
        command: str,
        *,
        trusted_system_path_only: bool = False,
    ) -> tuple[list[str] | None, Path | None, dict[str, str], int | None]:
        sb = self.config.sandbox
        iso_root = isolated_worktree_root()
        control = _model_sandbox_control(ctx, iso_root)
        if not sb.enabled and iso_root is None and not control.strict:
            env = (
                _autoapproved_shell_env(ctx)
                if trusted_system_path_only
                else _get_base_env()
            )
            return None, None, env, None
        write_roots, env = _sandbox_write_scope(sb, ctx, command, iso_root, control)
        if trusted_system_path_only:
            env = _autoapproved_shell_env(ctx)
            if control.strict:
                env["VIBE_STRICT_MODEL_CONTROL"] = "1"

        backend = resolve_backend(sb.backend)
        strict_backend = (
            backend.name == "bwrap" and sys.platform.startswith("linux")
        ) or (backend.name == "sandbox-exec" and sys.platform == "darwin")
        if control.strict and not strict_backend:
            raise ToolError(
                "Strict model control requires a filesystem-confining sandbox "
                "backend for this platform; refusing unconfined bash"
            )
        if backend.name == "none":
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
            read_roots=(
                _strict_read_roots(write_roots, control, env, iso_root)
                if control.strict
                else []
            ),
            hidden_roots=(strict_read_hidden_roots() if control.strict else []),
            protected_roots=(
                list(model_protected_roots(control.protected_roots))
                if control.strict
                else []
            ),
            protect_git_metadata=control.protect_git_metadata,
            allow_network=False if control.strict else sb.allow_network,
            env=env,
            extra_args=sb.extra_args,
        )
        argv, _name, profile = build_sandbox_command(spec, backend)
        if argv is None:
            if control.strict:
                raise ToolError(
                    "Strict model control could not construct its sandbox wrapper"
                )
            return None, None, env, None
        seccomp_fd = self._maybe_seccomp_fd(sb, backend.name, argv)
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
        self,
        args: BashArgs,
        ctx: InvokeContext | None,
        *,
        trusted_system_path_only: bool = False,
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
        shell_exe = _runtime_shell_executable()
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
        if trusted_system_path_only:
            sandbox = self._resolve_sandbox(
                ctx, args.command, trusted_system_path_only=True
            )
        else:
            sandbox = self._resolve_sandbox(ctx, args.command)
        sandbox_argv, _profile_path, run_env, seccomp_fd = sandbox
        try:
            if sandbox_argv is not None:
                if shell_exe is None:
                    raise ToolError(_BASH_EXECUTABLE_UNAVAILABLE)
                argv = [*sandbox_argv, shell_exe, "-c", args.command]
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

    async def _start_foreground(
        self,
        command: str,
        ctx: InvokeContext | None,
        *,
        trusted_system_path_only: bool = False,
    ) -> tuple[asyncio.subprocess.Process, Path | None, int | None, bool]:
        shell_exe = _runtime_shell_executable()
        kwargs: dict[Literal["start_new_session"], bool] = (
            {} if is_windows() else {"start_new_session": True}
        )
        if trusted_system_path_only:
            sandbox = self._resolve_sandbox(ctx, command, trusted_system_path_only=True)
        else:
            sandbox = self._resolve_sandbox(ctx, command)
        sandbox_argv, profile_path, run_env, seccomp_fd = sandbox
        if sandbox_argv is None:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=run_env,
                executable=shell_exe,
                **kwargs,
            )
            return proc, profile_path, seccomp_fd, False

        if shell_exe is None:
            raise ToolError(_BASH_EXECUTABLE_UNAVAILABLE)
        argv = [*sandbox_argv, shell_exe, "-c", command]
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
        except (FileNotFoundError, OSError) as exc:
            if self.config.sandbox.require_backend or (
                run_env.get("VIBE_STRICT_MODEL_CONTROL") == "1"
            ):
                _unlink_quietly(profile_path)
                _close_fd_quietly(seccomp_fd)
                raise ToolError(f"Sandbox wrapper failed to start: {exc}") from exc
            logger.warning(
                "sandbox wrapper failed to start (%s); falling back unsandboxed. "
                "Filesystem containment is lost but the scrubbed environment is "
                "preserved (no secrets re-injected).",
                exc,
            )
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=run_env,
                    executable=shell_exe,
                    **kwargs,
                )
            except BaseException:
                _unlink_quietly(profile_path)
                _close_fd_quietly(seccomp_fd)
                raise
            return proc, profile_path, seccomp_fd, False
        except BaseException:
            _unlink_quietly(profile_path)
            _close_fd_quietly(seccomp_fd)
            raise
        return proc, profile_path, seccomp_fd, True

    async def _collect_foreground(
        self, proc: asyncio.subprocess.Process, command: str, timeout: int
    ) -> tuple[bytes, bytes]:
        try:
            return await asyncio.wait_for(
                _communicate_limited(
                    proc, max_capture_bytes=self.config.max_capture_bytes
                ),
                timeout=timeout,
            )
        except TimeoutError:
            await kill_async_subprocess(proc)
            raise self._build_timeout_error(command, timeout) from None
        except _OutputCaptureLimitExceeded:
            await kill_async_subprocess(proc)
            raise ToolError(
                "Command output exceeded the configured combined capture "
                f"limit of {self.config.max_capture_bytes:,} bytes; the "
                "process tree was terminated"
            ) from None

    async def run(
        self, args: BashArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | BashResult, None]:
        timeout = args.timeout or self.config.default_timeout
        max_bytes = self.config.max_output_bytes
        trusted_system_path_only = await self._validate_execution_authorization(
            args, ctx, require_local_shell=True
        )
        execution_command = (
            harden_automated_command(args.command)
            if trusted_system_path_only
            else args.command
        )

        # Returns BEFORE foreground try/finally below — finally never kills backgrounds.
        if args.background:
            execution_args = args.model_copy(update={"command": execution_command})
            async for item in self._run_background(
                execution_args, ctx, trusted_system_path_only=trusted_system_path_only
            ):
                yield item
            return

        proc = None
        profile_path: Path | None = None
        seccomp_fd: int | None = None
        ran_sandboxed = False
        try:
            (
                proc,
                profile_path,
                seccomp_fd,
                ran_sandboxed,
            ) = await self._start_foreground(
                execution_command,
                ctx,
                trusted_system_path_only=trusted_system_path_only,
            )
            stdout_bytes, stderr_bytes = await self._collect_foreground(
                proc, args.command, timeout
            )

            shaped_output = _shape_output(
                ctx,
                command=args.command,
                returncode=proc.returncode or 0,
                stdout_bytes=stdout_bytes,
                stderr_bytes=stderr_bytes,
                max_chars=max_bytes,
            )

            yield self._build_result(
                command=args.command,
                stdout=shaped_output.stdout,
                stderr=shaped_output.stderr,
                returncode=proc.returncode or 0,
                full_output_path=shaped_output.full_output_path,
                sandbox_active=ran_sandboxed,
            )

        except (ToolError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise ToolError(f"Error running command {args.command!r}: {exc}") from exc
        finally:
            if proc is not None:
                await kill_async_subprocess(proc)
            _unlink_quietly(profile_path)
            _close_fd_quietly(seccomp_fd)

    async def _validate_execution_authorization(
        self, args: BashArgs, ctx: InvokeContext | None, *, require_local_shell: bool
    ) -> bool:
        verification_roots = verification_protected_roots(
            ctx.verification_state if ctx is not None else None
        )
        if preflight_denial := command_analysis_preflight_denial(args.command):
            raise ToolError(preflight_denial)
        if require_local_shell:
            _runtime_shell_executable()
        command_parts = _extract_commands(args.command)
        policy_analysis = _analyze_command_policy(args.command)
        if policy_denial := policy_analysis.denial:
            raise ToolError(policy_denial)
        if control_plane := hard_control_plane_command_reason(
            command_parts, args.command, extra_roots=verification_roots
        ):
            raise ToolError(control_plane)
        if _model_sandbox_control(ctx, isolated_worktree_root()).strict and (
            args.background or command_uses_unmanaged_background(args.command)
        ):
            raise ToolError(
                "Strict model control does not permit background Bash commands"
            )
        permission = _command_syntax_denial(args.command)
        if permission is None:
            permission = self._resolve_execution_permission(
                args, command_parts, policy_analysis
            )
        await enforce_shared_ask(
            self.get_name(), args.command, permission, self.config.permission
        )
        return _trusted_execution_required(
            args, ctx, permission, self.config.permission
        )
