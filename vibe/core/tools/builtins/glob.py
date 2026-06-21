from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from enum import StrEnum, auto
import fnmatch
import os
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, Field

from vibe.core.autocompletion.file_indexer.ignore_rules import IgnoreRules
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


class GlobBackend(StrEnum):
    RIPGREP = auto()
    WALK = auto()


class GlobToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )
    default_max_results: int = Field(
        default=1000, description="Default cap on the number of returned paths."
    )
    default_timeout: int = Field(
        default=30, description="Timeout for the file scan in seconds."
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
        description="List of glob patterns to exclude (dirs should end with /).",
    )
    codeignore_file: str = Field(
        default=".vibeignore",
        description="Name of the file to read for additional exclusion patterns.",
    )


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'.")
    path: str = Field(default=".", description="Root directory to search from.")
    max_results: int | None = Field(
        default=None, description="Override the default cap on returned paths."
    )
    use_default_ignore: bool = Field(
        default=True,
        description="Whether to respect .gitignore/.vibeignore and default excludes.",
    )


class GlobResult(BaseModel):
    paths: list[str]
    match_count: int
    was_truncated: bool = Field(
        description="True if results were capped by max_results."
    )


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _compile_glob(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    length = len(pattern)
    while i < length:
        char = pattern[i]
        if char == "*" and pattern[i : i + 2] == "**":
            if pattern[i : i + 3] == "**/":
                out.append("(?:[^/]+/)*")
                i += 3
            else:
                out.append(".*")
                i += 2
        elif char == "*":
            out.append("[^/]*")
            i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile(f"(?s:{''.join(out)})\\Z")


def _is_excluded(rel_str: str, name: str, is_dir: bool, exclude: list[str]) -> bool:
    for pattern in exclude:
        if pattern.endswith("/"):
            if is_dir and fnmatch.fnmatch(name, pattern[:-1]):
                return True
            continue
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_str, pattern):
            return True
    return False


class Glob(
    BaseTool[GlobArgs, GlobResult, GlobToolConfig, BaseToolState],
    ToolUIData[GlobArgs, GlobResult],
):
    read_only: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Find files by glob pattern (e.g. '**/*.py'), most recently modified first. "
        "Respects .gitignore and .vibeignore. Use this instead of bash find/ls."
    )

    def resolve_permission(self, args: GlobArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
        )

    def _detect_backend(self) -> GlobBackend:
        return GlobBackend.RIPGREP if shutil.which("rg") else GlobBackend.WALK

    async def run(
        self, args: GlobArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | GlobResult, None]:
        self._validate_args(args)
        root = self._resolve_root(args.path)
        exclude = self._collect_exclude_patterns()
        backend = self._detect_backend()

        if backend is GlobBackend.RIPGREP:
            paths = await self._run_ripgrep(args, root, exclude)
        else:
            paths = await asyncio.to_thread(self._walk_files, args, root, exclude)

        cap = args.max_results or self.config.default_max_results
        ordered, was_truncated = self._sort_and_cap(paths, cap)

        yield GlobResult(
            paths=ordered, match_count=len(ordered), was_truncated=was_truncated
        )

    def _validate_args(self, args: GlobArgs) -> None:
        if not args.pattern.strip():
            raise ToolError("Empty glob pattern provided.")

    def _resolve_root(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.exists():
            raise ToolError(f"Path does not exist: {raw_path}")
        if not path.is_dir():
            raise ToolError(f"Path is not a directory: {raw_path}")
        return path

    def _collect_exclude_patterns(self) -> list[str]:
        patterns = list(self.config.exclude_patterns)
        codeignore_path = Path.cwd() / self.config.codeignore_file
        if codeignore_path.is_file():
            patterns.extend(self._load_codeignore_patterns(codeignore_path))
        return patterns

    def _load_codeignore_patterns(self, codeignore_path: Path) -> list[str]:
        patterns: list[str] = []
        try:
            content = read_safe(codeignore_path).text
        except OSError:
            return patterns
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)
        return patterns

    async def _run_ripgrep(
        self, args: GlobArgs, root: Path, exclude: list[str]
    ) -> list[Path]:
        cmd = ["rg", "--files", "--no-messages"]
        if not args.use_default_ignore:
            cmd.append("--no-ignore")
        cmd.extend(["--glob", args.pattern])
        for pattern in exclude:
            cmd.extend(["--glob", f"!{pattern}"])
        cmd.append(str(root))

        stdout = await self._execute(cmd)
        return [Path(line) for line in stdout.splitlines() if line]

    def _walk_files(self, args: GlobArgs, root: Path, exclude: list[str]) -> list[Path]:
        matcher = _compile_glob(args.pattern)
        basename_only = "/" not in args.pattern
        ignore: IgnoreRules | None = None
        if args.use_default_ignore:
            ignore = IgnoreRules()
            ignore.ensure_for_root(root)
        results: list[Path] = []
        seen: set[tuple[int, int]] = set()

        def visit(current: Path) -> None:
            try:
                stat = current.stat()
            except OSError:
                return
            key = (stat.st_dev, stat.st_ino)
            if key in seen:
                return
            seen.add(key)
            try:
                entries = list(os.scandir(current))
            except OSError:
                return
            for entry in entries:
                name = entry.name
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                rel_str = Path(entry.path).relative_to(root).as_posix()
                if ignore is not None and ignore.should_ignore(rel_str, name, is_dir):
                    continue
                if _is_excluded(rel_str, name, is_dir, exclude):
                    continue
                if is_dir:
                    visit(Path(entry.path))
                    continue
                target = name if basename_only else rel_str
                if matcher.match(target):
                    results.append(Path(entry.path))

        visit(root)
        return results

    def _sort_and_cap(self, paths: list[Path], cap: int) -> tuple[list[str], bool]:
        ordered = sorted(paths, key=_safe_mtime, reverse=True)
        was_truncated = len(ordered) > cap
        return [str(path) for path in ordered[:cap]], was_truncated

    async def _execute(self, cmd: list[str]) -> str:
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
                    f"Glob search timed out after {self.config.default_timeout}s"
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
                raise ToolError(f"glob error: {error_msg}")

            return stdout

        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Error running glob: {exc}") from exc

    @classmethod
    def format_call_display(cls, args: GlobArgs) -> ToolCallDisplay:
        summary = f"Finding files matching '{args.pattern}'"
        if args.path != ".":
            summary += f" in {args.path}"
        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, GlobResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        plural = "file" if event.result.match_count == 1 else "files"
        message = f"Found {event.result.match_count} {plural}"
        suffix = "(truncated)" if event.result.was_truncated else ""

        return ToolResultDisplay(success=True, message=message, suffix=suffix)

    @classmethod
    def get_status_text(cls) -> str:
        return "Finding files"
