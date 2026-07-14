from __future__ import annotations

import contextlib
import errno
import os
from pathlib import Path
import stat
import time
from typing import TYPE_CHECKING

from vibe.core.tools._model_write_policy import (
    ManagedWritePolicyError,
    managed_candidate_write_scope,
)
from vibe.core.utils.io import ReadSafeResult, decode_safe

if TYPE_CHECKING:
    from vibe.core.verification_state import VerificationState


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


class ManagedWriteError(OSError):
    pass


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _fingerprint(value: os.stat_result) -> str:
    return f"{value.st_dev}:{value.st_ino}:{value.st_mtime_ns}:{value.st_size}"


def _open_directory(parent_fd: int, name: str) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ManagedWriteError(
                f"managed write path has a symlink or non-directory ancestor: {name}"
            ) from exc
        raise


def _open_absolute_directory(path: Path) -> int:
    if os.name != "posix" or not getattr(os, "O_NOFOLLOW", 0):
        raise ManagedWriteError(
            "managed writes require descriptor-relative no-follow filesystem support"
        )
    if not path.is_absolute():
        raise ManagedWriteError("managed write root must be absolute")

    current = os.open(path.anchor, _DIRECTORY_FLAGS)
    try:
        for part in path.parts[1:]:
            if part in {"", ".", ".."}:
                raise ManagedWriteError("managed write root is not canonical")
            child = _open_directory(current, part)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


class ManagedWriteTarget:
    def __init__(
        self,
        *,
        root_path: Path,
        relative_parts: tuple[str, ...],
        root_fd: int,
        parent_fd: int,
        parent_prefix: tuple[str, ...],
        missing_parent_parts: tuple[str, ...],
        target_fd: int | None,
    ) -> None:
        self._root_path = root_path
        self._relative_parts = relative_parts
        self._root_fd = root_fd
        self._parent_fd = parent_fd
        self._parent_prefix = parent_prefix
        self._missing_parent_parts = missing_parent_parts
        self._target_fd = target_fd
        self._root_identity = _identity(os.fstat(root_fd))
        self._target_fingerprint: str | None = None
        self._published_fingerprint: str | None = None
        self._closed = False

    @classmethod
    def capture(
        cls,
        path: Path,
        state: VerificationState | None,
        *,
        scratchpad_dir: Path | None,
        require_existing: bool,
    ) -> ManagedWriteTarget | None:
        try:
            scope = managed_candidate_write_scope(
                path, state, scratchpad_dir=scratchpad_dir
            )
        except ManagedWritePolicyError as exc:
            raise ManagedWriteError(str(exc)) from exc
        if scope is None:
            return None

        parts = scope.relative.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ManagedWriteError(
                "managed write target must name a file below its root"
            )

        root_fd = _open_absolute_directory(scope.root)
        if _identity(os.fstat(root_fd)) != scope.root_identity:
            os.close(root_fd)
            raise ManagedWriteError("managed write root changed during authorization")

        parent_fd = os.dup(root_fd)
        prefix: list[str] = []
        missing: tuple[str, ...] = ()
        target_fd: int | None = None
        try:
            for index, part in enumerate(parts[:-1]):
                try:
                    child = _open_directory(parent_fd, part)
                except FileNotFoundError:
                    missing = tuple(parts[index:-1])
                    break
                os.close(parent_fd)
                parent_fd = child
                prefix.append(part)

            if require_existing:
                if missing:
                    raise ManagedWriteError(
                        f"managed edit parent does not exist: {path.parent}"
                    )
                try:
                    target_fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent_fd)
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ManagedWriteError(
                            f"managed edit target is a symlink or not a file: {path}"
                        ) from exc
                    raise
                if not stat.S_ISREG(os.fstat(target_fd).st_mode):
                    raise ManagedWriteError(
                        f"managed edit target is not a file: {path}"
                    )

            return cls(
                root_path=scope.root,
                relative_parts=tuple(parts),
                root_fd=root_fd,
                parent_fd=parent_fd,
                parent_prefix=tuple(prefix),
                missing_parent_parts=missing,
                target_fd=target_fd,
            )
        except BaseException:
            if target_fd is not None:
                os.close(target_fd)
            os.close(parent_fd)
            os.close(root_fd)
            raise

    @property
    def initial_fingerprint(self) -> str:
        if self._target_fd is None:
            raise ManagedWriteError(
                "managed write target did not exist at authorization"
            )
        return _fingerprint(os.fstat(self._target_fd))

    @property
    def published_fingerprint(self) -> str | None:
        return self._published_fingerprint

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._target_fd is not None:
            os.close(self._target_fd)
        os.close(self._parent_fd)
        os.close(self._root_fd)

    def _reopen_prefix(self, prefix: tuple[str, ...]) -> int:
        current = os.dup(self._root_fd)
        try:
            for part in prefix:
                child = _open_directory(current, part)
                os.close(current)
                current = child
            return current
        except BaseException:
            os.close(current)
            raise

    def _revalidate_root(self) -> None:
        current_fd = _open_absolute_directory(self._root_path)
        try:
            if _identity(os.fstat(current_fd)) != self._root_identity:
                raise ManagedWriteError(
                    "managed write root changed after authorization"
                )
        finally:
            os.close(current_fd)

    def _revalidate_directory(self, directory_fd: int, prefix: tuple[str, ...]) -> None:
        self._revalidate_root()
        current_fd = self._reopen_prefix(prefix)
        try:
            if _identity(os.fstat(current_fd)) != _identity(os.fstat(directory_fd)):
                raise ManagedWriteError(
                    "managed write ancestor changed after authorization"
                )
        finally:
            os.close(current_fd)

    def _revalidate_existing_target(self) -> os.stat_result:
        if self._target_fd is None:
            raise ManagedWriteError(
                "managed edit target did not exist at authorization"
            )
        self._revalidate_directory(self._parent_fd, self._parent_prefix)
        try:
            current = os.stat(
                self._relative_parts[-1], dir_fd=self._parent_fd, follow_symlinks=False
            )
        except FileNotFoundError as exc:
            raise ManagedWriteError(
                "managed edit target changed after authorization"
            ) from exc
        authorized = os.fstat(self._target_fd)
        if _identity(current) != _identity(authorized) or not stat.S_ISREG(
            current.st_mode
        ):
            raise ManagedWriteError("managed edit target changed after authorization")
        return authorized

    def _materialize_parent(self, *, create: bool) -> tuple[int, tuple[str, ...]]:
        self._revalidate_directory(self._parent_fd, self._parent_prefix)
        current = os.dup(self._parent_fd)
        prefix = list(self._parent_prefix)
        try:
            for part in self._missing_parent_parts:
                try:
                    child = _open_directory(current, part)
                except FileNotFoundError:
                    if not create:
                        raise ManagedWriteError(
                            f"managed write parent does not exist: {self._root_path.joinpath(*self._relative_parts[:-1])}"
                        ) from None
                    os.mkdir(part, dir_fd=current)
                    child = _open_directory(current, part)
                os.close(current)
                current = child
                prefix.append(part)
            full_prefix = tuple(prefix)
            self._revalidate_directory(current, full_prefix)
            return current, full_prefix
        except BaseException:
            os.close(current)
            raise

    def read_safe(self) -> ReadSafeResult:
        authorized = self._revalidate_existing_target()
        before = _fingerprint(authorized)
        if self._target_fd is None:
            raise ManagedWriteError(
                "managed edit target did not exist at authorization"
            )
        with os.fdopen(os.dup(self._target_fd), "rb") as file:
            raw = file.read()
        after = _fingerprint(os.fstat(self._target_fd))
        if after != before:
            raise ManagedWriteError("managed edit target changed while it was read")
        self._target_fingerprint = after
        return decode_safe(raw, raise_on_error=True)

    def create_text(self, content: str, *, create_parent_dirs: bool) -> None:
        parent_fd, parent_prefix = self._materialize_parent(create=create_parent_dirs)
        target_name = self._relative_parts[-1]
        temp_name = f".{target_name}.tmp.{os.getpid()}.{time.time_ns()}"
        temp_fd: int | None = None
        temp_identity: tuple[int, int] | None = None
        published = False
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            temp_fd = os.open(temp_name, flags, 0o666, dir_fd=parent_fd)
            temp_identity = _identity(os.fstat(temp_fd))
            with os.fdopen(temp_fd, "w", encoding="utf-8") as file:
                temp_fd = None
                file.write(content)
                file.flush()
            self._revalidate_directory(parent_fd, parent_prefix)
            os.link(
                temp_name,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            published = True
            self._revalidate_directory(parent_fd, parent_prefix)
            current = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
            if _identity(current) != temp_identity:
                raise ManagedWriteError(
                    "managed write target changed during atomic creation"
                )
            self._published_fingerprint = _fingerprint(current)
        except BaseException:
            if published and temp_identity is not None:
                self._unlink_if_identity(parent_fd, target_name, temp_identity)
            raise
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name, dir_fd=parent_fd)
            os.close(parent_fd)

    def replace_text(self, content: str, *, encoding: str, newline: str) -> None:
        authorized = self._revalidate_existing_target()
        if self._target_fingerprint is None:
            raise ManagedWriteError(
                "managed edit target was not read before replacement"
            )
        if _fingerprint(authorized) != self._target_fingerprint:
            raise ManagedWriteError("managed edit target changed after it was read")

        target_name = self._relative_parts[-1]
        temp_name = f".{target_name}.tmp.{os.getpid()}.{time.time_ns()}"
        temp_fd: int | None = None
        temp_identity: tuple[int, int] | None = None
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            temp_fd = os.open(temp_name, flags, 0o600, dir_fd=self._parent_fd)
            temp_identity = _identity(os.fstat(temp_fd))
            os.fchmod(temp_fd, stat.S_IMODE(authorized.st_mode))
            with os.fdopen(temp_fd, "w", encoding=encoding, newline=newline) as file:
                temp_fd = None
                file.write(content)
                file.flush()

            authorized = self._revalidate_existing_target()
            if _fingerprint(authorized) != self._target_fingerprint:
                raise ManagedWriteError("managed edit target changed after it was read")
            os.replace(
                temp_name,
                target_name,
                src_dir_fd=self._parent_fd,
                dst_dir_fd=self._parent_fd,
            )
            self._revalidate_directory(self._parent_fd, self._parent_prefix)
            current = os.stat(
                target_name, dir_fd=self._parent_fd, follow_symlinks=False
            )
            if _identity(current) != temp_identity:
                raise ManagedWriteError(
                    "managed edit target changed during atomic replacement"
                )
            self._published_fingerprint = _fingerprint(current)
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name, dir_fd=self._parent_fd)

    @staticmethod
    def _unlink_if_identity(
        parent_fd: int, name: str, expected: tuple[int, int]
    ) -> None:
        with contextlib.suppress(FileNotFoundError):
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if _identity(current) == expected:
                os.unlink(name, dir_fd=parent_fd)


__all__ = ["ManagedWriteError", "ManagedWriteTarget"]
