from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import os
from pathlib import Path, PurePosixPath
import stat
import uuid

_MAX_DIRECTORY_ENTRIES = 10_000


class ImmutableStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StableFile:
    payload: bytes
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


class ImmutableFileStore:
    def __init__(self, root: Path) -> None:
        expanded = root.expanduser()
        self.root = expanded if expanded.is_absolute() else expanded.absolute()
        _require_descriptor_support()

    def write(self, relative: PurePosixPath, payload: bytes) -> None:
        parts = _relative_parts(relative)
        with self._open_parent(parts, create=True) as (parent_fd, name):
            existing = _read_optional_at(parent_fd, name, max_bytes=len(payload))
            if existing is not None:
                if existing.payload != payload:
                    raise ImmutableStoreError(
                        "immutable file already exists with different content"
                    )
                _verify_entry_identity(parent_fd, name, existing)
                return
            temporary = f".{name}.{uuid.uuid4().hex}.tmp"
            descriptor = -1
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=parent_fd,
                )
                _write_all(descriptor, payload)
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = -1
                try:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    existing = _read_required_at(
                        parent_fd, name, max_bytes=len(payload)
                    )
                    if existing.payload != payload:
                        raise ImmutableStoreError(
                            "immutable file appeared with different content"
                        ) from None
                else:
                    os.fsync(parent_fd)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                else:
                    os.fsync(parent_fd)
            committed = _read_required_at(parent_fd, name, max_bytes=len(payload))
            if committed.payload != payload:
                raise ImmutableStoreError(
                    "immutable file changed before persistence completed"
                )
            _verify_entry_identity(parent_fd, name, committed)

    def read(self, relative: PurePosixPath, *, max_bytes: int) -> bytes:
        parts = _relative_parts(relative)
        with self._open_parent(parts, create=False) as (parent_fd, name):
            stable = _read_required_at(parent_fd, name, max_bytes=max_bytes)
            _verify_entry_identity(parent_fd, name, stable)
            return stable.payload

    def list_directory(self, relative: PurePosixPath) -> tuple[str, ...]:
        parts = _relative_parts(relative, allow_empty=True)
        with self._open_root(create=False) as root_fd:
            directory_fd = os.dup(root_fd)
            try:
                for part in parts:
                    next_fd = _open_directory_at(directory_fd, part, create=False)
                    os.close(directory_fd)
                    directory_fd = next_fd
                names: list[str] = []
                with os.scandir(directory_fd) as iterator:
                    for index, entry in enumerate(iterator, start=1):
                        if index > _MAX_DIRECTORY_ENTRIES:
                            raise ImmutableStoreError(
                                "immutable store directory contains too many entries"
                            )
                        names.append(entry.name)
                entries = tuple(sorted(names))
                self._verify_parent_identity(parts, directory_fd)
                return entries
            finally:
                os.close(directory_fd)

    @contextmanager
    def _open_parent(
        self, parts: tuple[str, ...], *, create: bool
    ) -> Iterator[tuple[int, str]]:
        with self._open_root(create=create) as root_fd:
            parent_fd = os.dup(root_fd)
            try:
                for part in parts[:-1]:
                    next_fd = _open_directory_at(parent_fd, part, create=create)
                    os.close(parent_fd)
                    parent_fd = next_fd
                yield parent_fd, parts[-1]
            finally:
                try:
                    self._verify_parent_identity(parts[:-1], parent_fd)
                finally:
                    os.close(parent_fd)

    @contextmanager
    def _open_root(self, *, create: bool) -> Iterator[int]:
        descriptor = self._open_root_descriptor(create=create)
        try:
            yield descriptor
        finally:
            try:
                current = self._open_root_descriptor(create=False)
                try:
                    _require_same_directory(descriptor, current)
                finally:
                    os.close(current)
            finally:
                os.close(descriptor)

    def _open_root_descriptor(self, *, create: bool) -> int:
        anchor, parts = _absolute_parts(self.root)
        descriptor = os.open(
            anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        try:
            for part in parts:
                next_fd = _open_directory_at(descriptor, part, create=create)
                os.close(descriptor)
                descriptor = next_fd
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _verify_parent_identity(
        self, parent_parts: tuple[str, ...], expected_fd: int
    ) -> None:
        current = self._open_root_descriptor(create=False)
        try:
            for part in parent_parts:
                next_fd = _open_directory_at(current, part, create=False)
                os.close(current)
                current = next_fd
            _require_same_directory(expected_fd, current)
        finally:
            os.close(current)


def read_stable_absolute_file(path: Path, *, max_bytes: int) -> StableFile:
    anchor, parts = _absolute_parts(path)
    if not parts:
        raise ImmutableStoreError("expected a regular file path")
    descriptor = os.open(
        anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        for part in parts[:-1]:
            next_fd = _open_directory_at(descriptor, part, create=False)
            os.close(descriptor)
            descriptor = next_fd
        stable = _read_required_at(descriptor, parts[-1], max_bytes=max_bytes)
        _verify_entry_identity(descriptor, parts[-1], stable)
        current = _open_absolute_directory(anchor, parts[:-1])
        try:
            _require_same_directory(descriptor, current)
        finally:
            os.close(current)
        return stable
    finally:
        os.close(descriptor)


def _read_optional_at(
    parent_fd: int, name: str, *, max_bytes: int
) -> StableFile | None:
    try:
        return _read_required_at(parent_fd, name, max_bytes=max_bytes)
    except FileNotFoundError:
        return None


def _read_required_at(parent_fd: int, name: str, *, max_bytes: int) -> StableFile:
    try:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ImmutableStoreError("refusing a symlinked immutable file") from exc
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ImmutableStoreError("immutable path is not a regular file")
        if before.st_nlink != 1:
            raise ImmutableStoreError("immutable file has an unexpected hard link")
        if before.st_size > max_bytes:
            raise ImmutableStoreError(
                f"immutable file exceeds the {max_bytes}-byte limit"
            )
        chunks: list[bytes] = []
        observed = 0
        while chunk := os.read(descriptor, min(64 * 1024, max_bytes + 1 - observed)):
            chunks.append(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise ImmutableStoreError(
                    f"immutable file exceeds the {max_bytes}-byte limit"
                )
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        )
        if before_identity != after_identity or observed != after.st_size:
            raise ImmutableStoreError("immutable file changed while it was read")
        return StableFile(
            payload=b"".join(chunks),
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            modified_ns=after.st_mtime_ns,
            changed_ns=after.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


def _verify_entry_identity(parent_fd: int, name: str, expected: StableFile) -> None:
    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ImmutableStoreError(
            "immutable file path changed after it was read"
        ) from exc
    identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    expected_identity = (
        expected.device,
        expected.inode,
        expected.size,
        expected.modified_ns,
        expected.changed_ns,
    )
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or identity != expected_identity
    ):
        raise ImmutableStoreError("immutable file path changed after it was read")


def _require_same_directory(expected_fd: int, current_fd: int) -> None:
    expected = os.fstat(expected_fd)
    current = os.fstat(current_fd)
    if (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
        raise ImmutableStoreError(
            "immutable store ancestor changed during the operation"
        )


def _open_directory_at(parent_fd: int, name: str, *, create: bool) -> int:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ImmutableStoreError(
                "refusing a symlinked or non-directory store ancestor"
            ) from exc
        raise
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise ImmutableStoreError("store ancestor is not a directory")
    return descriptor


def _open_absolute_directory(anchor: str, parts: tuple[str, ...]) -> int:
    descriptor = os.open(
        anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        for part in parts:
            next_fd = _open_directory_at(descriptor, part, create=False)
            os.close(descriptor)
            descriptor = next_fd
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise ImmutableStoreError("immutable file write did not make progress")
        offset += written


def _relative_parts(
    relative: PurePosixPath, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if relative.is_absolute() or ".." in relative.parts:
        raise ImmutableStoreError("immutable store path escapes its root")
    parts = tuple(part for part in relative.parts if part not in {"", "."})
    if not parts and not allow_empty:
        raise ImmutableStoreError("immutable store path is empty")
    return parts


def _absolute_parts(path: Path) -> tuple[str, tuple[str, ...]]:
    _require_descriptor_support()
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else expanded.absolute()
    if not absolute.anchor:
        raise ImmutableStoreError("immutable store path must be absolute")
    parts = tuple(part for part in absolute.parts if part != absolute.anchor)
    if any(part in {"", ".", ".."} for part in parts):
        raise ImmutableStoreError("immutable store path is not canonical")
    return absolute.anchor, parts


def _require_descriptor_support() -> None:
    if (
        os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or os.open not in os.supports_dir_fd
    ):
        raise ImmutableStoreError(
            "authority-bearing immutable storage requires POSIX no-follow descriptors"
        )


__all__ = [
    "ImmutableFileStore",
    "ImmutableStoreError",
    "StableFile",
    "read_stable_absolute_file",
]
