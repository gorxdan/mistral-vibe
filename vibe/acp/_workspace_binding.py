"""Single-workspace binding for the process-global ACP runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import os
from pathlib import Path

from vibe.acp.exceptions import InvalidRequestError
from vibe.core.config.harness_files import add_session_dirs, get_harness_files_manager
from vibe.core.trusted_folders import WorkspaceTrustDecision, trusted_folders_manager
from vibe.core.utils import is_dangerous_directory


@dataclass(frozen=True)
class AcpWorkspace:
    cwd: Path
    requested_additional_dirs: tuple[Path, ...] = ()
    additional_dirs: tuple[Path, ...] = ()


class AcpWorkspaceBinding:
    def __init__(
        self, resolve_trust: Callable[[Path], Awaitable[WorkspaceTrustDecision | None]]
    ) -> None:
        self._resolve_trust = resolve_trust
        self._workspace: AcpWorkspace | None = None
        self._lock = asyncio.Lock()

    @property
    def workspace(self) -> AcpWorkspace | None:
        return self._workspace

    @staticmethod
    def normalize_request(
        cwd: str, additional_directories: list[str] | None
    ) -> tuple[Path, tuple[Path, ...]]:
        workspace = Path(cwd).expanduser().resolve()
        if not workspace.is_dir():
            raise InvalidRequestError(
                f"ACP workspace does not exist or is not a directory: {cwd}"
            )
        resolved: list[Path] = []
        for directory in additional_directories or []:
            path = Path(directory).expanduser().resolve()
            if not path.is_dir():
                raise InvalidRequestError(
                    "additional_directories path does not exist or is not a "
                    f"directory: {directory}"
                )
            is_dangerous, reason = is_dangerous_directory(path)
            if is_dangerous:
                raise InvalidRequestError(
                    f"additional_directories path is not allowed: {path} ({reason})"
                )
            if path not in resolved:
                resolved.append(path)
        return workspace, tuple(sorted(resolved, key=str))

    async def bind(
        self, cwd: str, additional_directories: list[str] | None
    ) -> AcpWorkspace:
        requested_cwd, requested_dirs = self.normalize_request(
            cwd, additional_directories
        )
        async with self._lock:
            if self._workspace is not None:
                self._assert_same_request(requested_cwd, requested_dirs)
                self._assert_process(self._workspace)
                return self._workspace

            await self._resolve_trust(requested_cwd)
            approved_dirs = await self._approve_additional_directories(requested_dirs)
            previous_cwd = Path.cwd().resolve()
            previous_dirs = get_harness_files_manager().additional_dirs
            try:
                os.chdir(requested_cwd)
                add_session_dirs(list(approved_dirs))
                self._workspace = AcpWorkspace(
                    cwd=requested_cwd,
                    requested_additional_dirs=requested_dirs,
                    additional_dirs=approved_dirs,
                )
                return self._workspace
            except Exception:
                os.chdir(previous_cwd)
                add_session_dirs(list(previous_dirs))
                raise

    async def bind_for_listing(
        self, cwd: str | None, additional_directories: list[str] | None
    ) -> None:
        if self._workspace is None:
            if additional_directories:
                await self.bind(cwd or str(Path.cwd()), additional_directories)
            return
        if cwd is not None or additional_directories is not None:
            requested_cwd, requested_dirs = self.normalize_request(
                cwd or str(self._workspace.cwd),
                additional_directories
                if additional_directories is not None
                else [str(path) for path in self._workspace.requested_additional_dirs],
            )
            self._assert_same_request(requested_cwd, requested_dirs)
        self._assert_process(self._workspace)

    def assert_session(self, workspace: AcpWorkspace) -> None:
        if self._workspace is None or workspace != self._workspace:
            raise InvalidRequestError(
                "The ACP session is not bound to this process workspace."
            )
        self._assert_process(workspace)

    async def _approve_additional_directories(
        self, directories: tuple[Path, ...]
    ) -> tuple[Path, ...]:
        approved: list[Path] = []
        for path in directories:
            decision = await self._resolve_trust(path)
            if decision == WorkspaceTrustDecision.DECLINE:
                continue
            trusted_folders_manager.trust_for_session(path)
            approved.append(path)
        return tuple(approved)

    def _assert_same_request(self, cwd: Path, requested_dirs: tuple[Path, ...]) -> None:
        if self._workspace is None:
            raise InvalidRequestError("The ACP process workspace is not bound.")
        if (
            cwd != self._workspace.cwd
            or requested_dirs != self._workspace.requested_additional_dirs
        ):
            raise InvalidRequestError(
                "This ACP process is already bound to a different workspace. Use "
                "a separate ACP process for another cwd or additional-directory set."
            )

    @staticmethod
    def _assert_process(workspace: AcpWorkspace) -> None:
        try:
            current = Path.cwd().resolve()
        except OSError as exc:
            raise InvalidRequestError(
                "The bound ACP workspace is no longer accessible."
            ) from exc
        if current != workspace.cwd:
            raise InvalidRequestError(
                "The ACP process working directory drifted from its bound "
                "workspace; restart the ACP server before continuing."
            )
        if get_harness_files_manager().additional_dirs != workspace.additional_dirs:
            raise InvalidRequestError(
                "The ACP harness roots drifted from the bound workspace; restart "
                "the ACP server before continuing."
            )
