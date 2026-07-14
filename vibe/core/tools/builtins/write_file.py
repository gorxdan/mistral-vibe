from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import partial
from pathlib import Path
from typing import ClassVar, final

import anyio
from anyio.to_thread import run_sync as run_sync_in_thread
from pydantic import BaseModel, ConfigDict, Field

from vibe.core.config.fingerprint import file_fingerprint
from vibe.core.lsp._integration import notify_file_changed
from vibe.core.rewind.manager import FileSnapshot
from vibe.core.scratchpad import is_scratchpad_path
from vibe.core.tools._managed_write import ManagedWriteError, ManagedWriteTarget
from vibe.core.tools._model_write_policy import (
    protected_model_write_reason,
    verification_protected_roots,
)
from vibe.core.tools._team_safety import enforce_shared_ask
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
from vibe.core.tools.utils import (
    enforce_isolated_confine,
    enforce_team_metadata_confine,
    resolve_file_tool_permission,
)
from vibe.core.types import ToolResultEvent, ToolStreamEvent


class WriteFileArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    content: str


class WriteFileResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    bytes_written: int
    content: str


class WriteFileConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )
    max_write_bytes: int = 64_000
    create_parent_dirs: bool = True


class WriteFile(
    BaseTool[WriteFileArgs, WriteFileResult, WriteFileConfig, BaseToolState],
    ToolUIData[WriteFileArgs, WriteFileResult],
):
    description: ClassVar[str] = (
        "Create a UTF-8 file. Fails if the file already exists; use edit to modify."
    )

    @classmethod
    def format_call_display(cls, args: WriteFileArgs) -> ToolCallDisplay:
        suffix = "(scratchpad)" if is_scratchpad_path(args.path) else ""
        return ToolCallDisplay(
            summary=f"Writing {args.path}", content=args.content, suffix=suffix
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, WriteFileResult):
            suffix = "(scratchpad)" if is_scratchpad_path(event.result.path) else ""
            return ToolResultDisplay(
                success=True,
                message=f"Created {Path(event.result.path).name}",
                suffix=suffix,
            )

        return ToolResultDisplay(success=True, message="File written")

    @classmethod
    def get_status_text(cls) -> str:
        return "Writing file"

    def get_file_snapshot(self, args: WriteFileArgs) -> FileSnapshot | None:
        return self.get_file_snapshot_for_path(args.path)

    def resolve_permission(self, args: WriteFileArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
        )

    @final
    async def run(
        self, args: WriteFileArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WriteFileResult, None]:
        file_path, content_bytes = self._prepare_and_validate_path(
            args,
            protected_roots=verification_protected_roots(
                ctx.verification_state if ctx is not None else None
            ),
        )
        managed_target = self._capture_managed_target(file_path, ctx)
        try:
            permission = self.resolve_permission(args)
            await enforce_shared_ask(
                self.get_name(), str(file_path), permission, self.config.permission
            )
            if managed_target is None:
                self._ensure_parent_dir(file_path)
                await self._write_file(args, file_path)
            else:
                try:
                    await run_sync_in_thread(
                        partial(
                            managed_target.create_text,
                            args.content,
                            create_parent_dirs=self.config.create_parent_dirs,
                        )
                    )
                except FileExistsError as e:
                    raise ToolError(
                        f"File '{file_path}' already exists. Use edit to modify it."
                    ) from e
                except ManagedWriteError as e:
                    raise ToolError(str(e)) from e
                except OSError as e:
                    raise ToolError(f"Error writing {file_path}: {e}") from e
        finally:
            if managed_target is not None:
                managed_target.close()
        await notify_file_changed(file_path, args.content)

        if ctx and ctx.files_read is not None:
            if managed_target is not None and managed_target.published_fingerprint:
                ctx.files_read[str(file_path)] = managed_target.published_fingerprint
            else:
                try:
                    ctx.files_read[str(file_path)] = file_fingerprint(file_path)
                except OSError:
                    pass

        yield WriteFileResult(
            path=str(file_path), bytes_written=content_bytes, content=args.content
        )

    def _ensure_parent_dir(self, file_path: Path) -> None:
        if self.config.create_parent_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _capture_managed_target(
        file_path: Path, ctx: InvokeContext | None
    ) -> ManagedWriteTarget | None:
        try:
            return ManagedWriteTarget.capture(
                file_path,
                ctx.verification_state if ctx is not None else None,
                scratchpad_dir=ctx.scratchpad_dir if ctx is not None else None,
                require_existing=False,
            )
        except ManagedWriteError as e:
            raise ToolError(str(e)) from e
        except OSError as e:
            raise ToolError(f"Error authorizing managed write {file_path}: {e}") from e

    def _prepare_and_validate_path(
        self, args: WriteFileArgs, *, protected_roots: tuple[Path, ...] = ()
    ) -> tuple[Path, int]:
        if not args.path.strip():
            raise ToolError("Path cannot be empty")

        content_bytes = len(args.content.encode("utf-8"))
        if content_bytes > self.config.max_write_bytes:
            raise ToolError(
                f"Content exceeds {self.config.max_write_bytes} bytes limit"
            )

        file_path = Path(args.path).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        file_path = file_path.resolve()
        if protected := protected_model_write_reason(
            file_path, extra_roots=protected_roots
        ):
            raise ToolError(protected)
        enforce_team_metadata_confine(file_path)
        enforce_isolated_confine(file_path)

        if file_path.exists():
            raise ToolError(
                f"File '{file_path}' already exists. Use edit to modify it."
            )
        if not self.config.create_parent_dirs and not file_path.parent.exists():
            raise ToolError(f"Parent directory does not exist: {file_path.parent}")

        return file_path, content_bytes

    async def _write_file(self, args: WriteFileArgs, file_path: Path) -> None:
        try:
            async with await anyio.Path(file_path).open(
                mode="x", encoding="utf-8"
            ) as f:
                await f.write(args.content)
        except FileExistsError as e:
            raise ToolError(
                f"File '{file_path}' already exists. Use edit to modify it."
            ) from e
        except Exception as e:
            raise ToolError(f"Error writing {file_path}: {e}") from e
