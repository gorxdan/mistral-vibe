from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.tools.utils import resolve_file_tool_permission
from vibe.core.types import ToolStreamEvent
from vibe.core.utils import kill_async_subprocess
from vibe.core.utils.io import decode_safe, read_safe

if TYPE_CHECKING:
    from vibe.core.types import ToolResultEvent


class GrepBackend(StrEnum):
    RIPGREP = auto()
    GNU_GREP = auto()


class GrepOutputMode(StrEnum):
    CONTENT = auto()
    FILES_WITH_MATCHES = auto()
    COUNT = auto()


class GrepToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )

    max_output_bytes: int = Field(
        default=64_000, description="Hard cap for the total size of matched lines."
    )
    default_max_matches: int = Field(
        default=100, description="Default maximum number of matches to return."
    )
    default_timeout: int = Field(
        default=60, description="Default timeout for the search command in seconds."
    )
    exclude_patterns: list[str] = Field(
        default=[
            ".venv/",
            "venv/",
            ".env/",
            "env/",
            "node_modules/",
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".tox/",
            ".nox/",
            ".coverage/",
            "htmlcov/",
            "dist/",
            "build/",
            ".idea/",
            ".vscode/",
            "*.egg-info",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            "Thumbs.db",
        ],
        description="List of glob patterns to exclude from search (dirs should end with /).",
    )
    codeignore_file: str = Field(
        default=".vibeignore",
        description="Name of the file to read for additional exclusion patterns.",
    )


class GrepArgs(BaseModel):
    pattern: str
    path: str = "."
    output_mode: GrepOutputMode = Field(
        default=GrepOutputMode.CONTENT,
        description=(
            "content = file:line:text; files_with_matches = matching filenames; "
            "count = file:count."
        ),
    )
    glob: str | None = Field(
        default=None,
        description="Only search files matching this glob, e.g. '*.py' or '*.{ts,tsx}'.",
    )
    type: str | None = Field(
        default=None,
        description="Only search files of this ripgrep type, e.g. 'py' (ripgrep only).",
    )
    case_insensitive: bool = Field(
        default=False, description="Case-insensitive search (-i)."
    )
    context: int = Field(
        default=0,
        ge=0,
        description="Lines of context around each match (-C). Content mode only.",
    )
    context_before: int = Field(
        default=0,
        ge=0,
        description="Lines of context before each match (-B). Content mode only.",
    )
    context_after: int = Field(
        default=0,
        ge=0,
        description="Lines of context after each match (-A). Content mode only.",
    )
    multiline: bool = Field(
        default=False, description="Allow patterns to span lines (ripgrep only)."
    )
    head_limit: int | None = Field(
        default=None, ge=1, description="Cap output to the first N lines/files."
    )
    max_matches: int | None = Field(
        default=None, description="Override the default maximum number of matches."
    )
    use_default_ignore: bool = Field(
        default=True, description="Whether to respect .gitignore and .ignore files."
    )


class GrepMatch(BaseModel):
    path: str
    line: int | None = None

    @classmethod
    def from_output_line(cls, raw: str) -> GrepMatch | None:
        """Parse a single grep/rg output line in `file:line:content` format.

        Handles Windows drive-letter paths like ``C:\\repo\\file.py:10:match``
        by skipping a single-letter first segment.
        """
        parts = raw.split(":", 3)
        MIN_MATCH_PARTS = 2
        if len(parts) < MIN_MATCH_PARTS:
            return None

        # Windows drive letter: first part is a single letter (e.g. "C")
        MIN_WINDOWS_PARTS = 3
        is_windows_path = (
            len(parts[0]) == 1
            and parts[0].isalpha()
            and len(parts) >= MIN_WINDOWS_PARTS
        )
        if is_windows_path:
            file_path = f"{parts[0]}:{parts[1]}"
            line_str = parts[2]
        else:
            file_path = parts[0]
            line_str = parts[1]

        try:
            line_num = int(line_str) if line_str else None
        except (ValueError, TypeError):
            line_num = None
        return cls(path=str(Path(file_path).resolve()), line=line_num)


class GrepResult(BaseModel):
    matches: str
    match_count: int
    was_truncated: bool = Field(
        description="True if output was cut short by max_matches or max_output_bytes."
    )
    output_mode: GrepOutputMode = GrepOutputMode.CONTENT

    @property
    def parsed_matches(self) -> list[GrepMatch]:
        if self.output_mode is not GrepOutputMode.CONTENT:
            return []
        results: list[GrepMatch] = []
        for line in self.matches.splitlines():
            if match := GrepMatch.from_output_line(line):
                results.append(match)
        return results


def _is_zero_count(line: str) -> bool:
    return line.rsplit(":", 1)[-1] == "0"


def _result_noun(output_mode: GrepOutputMode, count: int) -> str:
    if output_mode is GrepOutputMode.CONTENT:
        return "matches" if count != 1 else "match"
    return "files" if count != 1 else "file"


class Grep(
    BaseTool[GrepArgs, GrepResult, GrepToolConfig, BaseToolState],
    ToolUIData[GrepArgs, GrepResult],
):
    read_only: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Recursively search file contents for a regex pattern (ripgrep-backed, "
        ".gitignore-aware). ALWAYS use this to search contents — never shell "
        "out to `grep`/`rg` via bash. Use for text: error messages, log lines, "
        "string literals, config values, regex. Find files by name with `glob`; "
        "resolve symbols (definitions, references, types) with `lsp` when "
        "available — grep misses re-exports and imports that lsp resolves."
    )

    def resolve_permission(self, args: GrepArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
        )

    def _detect_backend(self) -> GrepBackend:
        if shutil.which("rg"):
            return GrepBackend.RIPGREP
        if shutil.which("grep"):
            return GrepBackend.GNU_GREP
        raise ToolError(
            "Neither ripgrep (rg) nor grep is installed. "
            "Please install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
        )

    async def run(
        self, args: GrepArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | GrepResult, None]:
        backend = self._detect_backend()
        self._validate_args(args)

        exclude_patterns = self._collect_exclude_patterns()
        cmd = self._build_command(args, exclude_patterns, backend)
        stdout = await self._execute_search(cmd)

        yield self._parse_output(
            stdout,
            args.max_matches or self.config.default_max_matches,
            output_mode=args.output_mode,
            head_limit=args.head_limit,
        )

    def _validate_args(self, args: GrepArgs) -> None:
        if not args.pattern.strip():
            raise ToolError("Empty search pattern provided.")

        if args.output_mode is not GrepOutputMode.CONTENT and (
            args.context or args.context_before or args.context_after
        ):
            raise ToolError("Context lines are only supported in content output mode.")

        path_obj = Path(args.path).expanduser()
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj

        if not path_obj.exists():
            raise ToolError(f"Path does not exist: {args.path}")

    def _collect_exclude_patterns(self) -> list[str]:
        patterns = list(self.config.exclude_patterns)

        codeignore_path = Path.cwd() / self.config.codeignore_file
        if codeignore_path.is_file():
            patterns.extend(self._load_codeignore_patterns(codeignore_path))

        return patterns

    def _load_codeignore_patterns(self, codeignore_path: Path) -> list[str]:
        patterns = []
        try:
            content = read_safe(codeignore_path).text
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        except OSError:
            pass

        return patterns

    def _build_command(
        self, args: GrepArgs, exclude_patterns: list[str], backend: GrepBackend
    ) -> list[str]:
        if backend == GrepBackend.RIPGREP:
            return self._build_ripgrep_command(args, exclude_patterns)
        return self._build_gnu_grep_command(args, exclude_patterns)

    def _build_ripgrep_command(
        self, args: GrepArgs, exclude_patterns: list[str]
    ) -> list[str]:
        max_matches = args.max_matches or self.config.default_max_matches

        cmd = ["rg", "--no-heading", "--with-filename", "--no-binary"]
        cmd.append("--ignore-case" if args.case_insensitive else "--smart-case")

        match args.output_mode:
            case GrepOutputMode.CONTENT:
                cmd.append("--line-number")
                cmd.extend(self._ripgrep_context_flags(args))
                # Request one extra to detect truncation
                cmd.extend(["--max-count", str(max_matches + 1)])
            case GrepOutputMode.FILES_WITH_MATCHES:
                cmd.append("--files-with-matches")
            case GrepOutputMode.COUNT:
                cmd.append("--count")

        if args.multiline:
            cmd.extend(["--multiline", "--multiline-dotall"])
        if args.type is not None:
            cmd.extend(["--type", args.type])
        if args.glob is not None:
            cmd.extend(["--glob", args.glob])
        if not args.use_default_ignore:
            cmd.append("--no-ignore")

        for pattern in exclude_patterns:
            cmd.extend(["--glob", f"!{pattern}"])

        cmd.extend(["-e", args.pattern, args.path])

        return cmd

    def _ripgrep_context_flags(self, args: GrepArgs) -> list[str]:
        if args.context:
            return ["--context", str(args.context)]
        flags: list[str] = []
        if args.context_before:
            flags.extend(["--before-context", str(args.context_before)])
        if args.context_after:
            flags.extend(["--after-context", str(args.context_after)])
        return flags

    def _build_gnu_grep_command(
        self, args: GrepArgs, exclude_patterns: list[str]
    ) -> list[str]:
        if args.type is not None:
            raise ToolError(
                "The `type` filter requires ripgrep (rg); the GNU grep fallback "
                "cannot filter by file type. Use `glob` instead."
            )
        if args.multiline:
            raise ToolError(
                "`multiline` requires ripgrep (rg); the GNU grep fallback is "
                "line-oriented."
            )

        max_matches = args.max_matches or self.config.default_max_matches

        cmd = ["grep", "-r", "-H", "-I", "-E", f"--max-count={max_matches + 1}"]

        if args.case_insensitive or args.pattern.islower():
            cmd.append("-i")

        match args.output_mode:
            case GrepOutputMode.CONTENT:
                cmd.append("-n")
                cmd.extend(self._gnu_context_flags(args))
            case GrepOutputMode.FILES_WITH_MATCHES:
                cmd.append("-l")
            case GrepOutputMode.COUNT:
                cmd.append("-c")

        if args.glob is not None:
            cmd.append(f"--include={args.glob}")

        for pattern in exclude_patterns:
            if pattern.endswith("/"):
                dir_pattern = pattern.rstrip("/")
                cmd.append(f"--exclude-dir={dir_pattern}")
            else:
                cmd.append(f"--exclude={pattern}")

        cmd.extend(["-e", args.pattern, args.path])

        return cmd

    def _gnu_context_flags(self, args: GrepArgs) -> list[str]:
        if args.context:
            return [f"-C{args.context}"]
        flags: list[str] = []
        if args.context_before:
            flags.append(f"-B{args.context_before}")
        if args.context_after:
            flags.append(f"-A{args.context_after}")
        return flags

    async def _execute_search(self, cmd: list[str]) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.config.default_timeout
                )
            except TimeoutError:
                await kill_async_subprocess(proc, kill_process_group=False)
                raise ToolError(
                    f"Search timed out after {self.config.default_timeout}s"
                )

            stdout = (
                decode_safe(stdout_bytes, from_subprocess=True).text
                if stdout_bytes
                else ""
            )
            stderr = (
                decode_safe(stderr_bytes, from_subprocess=True).text
                if stderr_bytes
                else ""
            )

            if proc.returncode not in {0, 1}:
                error_msg = stderr or f"Process exited with code {proc.returncode}"
                raise ToolError(f"grep error: {error_msg}")

            return stdout

        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Error running grep: {exc}") from exc

    def _parse_output(
        self,
        stdout: str,
        max_matches: int,
        *,
        output_mode: GrepOutputMode,
        head_limit: int | None,
    ) -> GrepResult:
        output_lines = stdout.splitlines() if stdout else []
        if output_mode is GrepOutputMode.COUNT:
            output_lines = [line for line in output_lines if not _is_zero_count(line)]

        cap = max_matches if head_limit is None else min(max_matches, head_limit)

        truncated_lines = output_lines[:cap]
        truncated_output = "\n".join(truncated_lines)

        was_truncated = (
            len(output_lines) > cap
            or len(truncated_output) > self.config.max_output_bytes
        )

        final_output = truncated_output[: self.config.max_output_bytes]

        return GrepResult(
            matches=final_output,
            match_count=len(truncated_lines),
            was_truncated=was_truncated,
            output_mode=output_mode,
        )

    @classmethod
    def format_call_display(cls, args: GrepArgs) -> ToolCallDisplay:
        summary = f"Grepping '{args.pattern}'"
        if args.path != ".":
            summary += f" in {args.path}"
        if args.glob is not None:
            summary += f" matching {args.glob}"
        if args.type is not None:
            summary += f" [type: {args.type}]"
        if args.output_mode is GrepOutputMode.FILES_WITH_MATCHES:
            summary += " [files]"
        elif args.output_mode is GrepOutputMode.COUNT:
            summary += " [count]"
        if args.max_matches:
            summary += f" (max {args.max_matches})"
        if not args.use_default_ignore:
            summary += " [no-ignore]"
        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, GrepResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        noun = _result_noun(event.result.output_mode, event.result.match_count)
        message = f"Found {event.result.match_count} {noun}"
        suffix = "(truncated)" if event.result.was_truncated else ""

        return ToolResultDisplay(success=True, message=message, suffix=suffix)

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching files"
