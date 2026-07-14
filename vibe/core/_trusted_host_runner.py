from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import BinaryIO, Protocol
import uuid

from vibe.core._immutable_store import (
    ImmutableStoreError,
    StableFile,
    read_stable_absolute_file,
)
from vibe.core._trusted_command import (
    TRUSTED_GIT_CONFIG_ARGS,
    TRUSTED_SYSTEM_PATH,
    minimal_trusted_git_environment,
    resolve_trusted_system_executable,
    validate_trusted_command_argv,
)
from vibe.core._verification_receipt import VerificationReceiptError
from vibe.core.utils._process_groups import signal_owned_process_group
from vibe.core.utils.io import decode_safe

MAX_COMBINED_OUTPUT_BYTES = 1024 * 1024

_HOST_COMMAND_TIMEOUT_SECONDS = 30
_HOST_DIAGNOSTIC_BYTES = 16 * 1024
_OUTPUT_CHUNK_BYTES = 64 * 1024
_PROCESS_STOP_SECONDS = 0.5
_MAX_SNAPSHOT_ENTRIES = 100_000
_MAX_SNAPSHOT_DEPTH = 128
_MAX_SNAPSHOT_FILE_BYTES = 512 * 1024 * 1024
_MAX_SNAPSHOT_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TREE_LISTING_BYTES = 128 * 1024 * 1024
_MAX_SYMLINK_BYTES = 1024 * 1024
_MAX_ENVIRONMENT_ATTESTATION_BYTES = 16 * 1024 * 1024
_MAX_TRUSTED_EXECUTABLE_BYTES = 512 * 1024 * 1024
_GIT_SYMLINK_MODE = 0o120000
_GIT_TREE_FIELD_COUNT = 3
_MAX_GIT_OBJECT_HEADER_BYTES = 1024


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


class _ByteReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


@dataclass(frozen=True, slots=True)
class FrozenSourceSnapshot:
    run_root: Path
    source_root: Path
    candidate_head: str
    candidate_tree: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class TrustedExecutable:
    lexical_path: Path
    resolved_path: Path
    materialization_root: Path
    materialized_path: Path
    sha256: str
    source_identity: tuple[int, int, int, int, int]
    materialized_identity: tuple[int, int, int, int, int]
    read_roots: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class TrustedEnvironmentAttestation:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class _GitTreeEntry:
    mode: int
    object_id: str
    relative: PurePosixPath


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    stdout: bytes
    stderr: bytes
    exit_code: int | None
    timed_out: bool
    output_limited: bool
    collector_error: str | None = None


class _CombinedOutput:
    def __init__(self, limit: int) -> None:
        self._remaining = limit
        self._lock = threading.Lock()
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.exceeded = threading.Event()
        self.reader_errors: list[str] = []
        self.completed_readers: set[str] = set()

    def append(self, stream: str, chunk: bytes) -> None:
        with self._lock:
            accepted = chunk[: self._remaining]
            target = self.stdout if stream == "stdout" else self.stderr
            target.extend(accepted)
            self._remaining -= len(accepted)
            if len(accepted) != len(chunk):
                self.exceeded.set()

    def record_reader_error(self, stream: str, exc: BaseException) -> None:
        with self._lock:
            self.reader_errors.append(f"{stream}: {type(exc).__name__}: {exc}")

    def record_reader_complete(self, stream: str) -> None:
        with self._lock:
            self.completed_readers.add(stream)

    def collector_diagnostic(self, *, readers_alive: bool) -> str | None:
        with self._lock:
            diagnostics = list(self.reader_errors)
            if readers_alive:
                diagnostics.append("one or more output readers did not terminate")
            missing = {"stdout", "stderr"} - self.completed_readers
            if missing:
                diagnostics.append(
                    f"output readers did not complete: {', '.join(sorted(missing))}"
                )
        return "; ".join(diagnostics) if diagnostics else None


def create_frozen_source_snapshot(
    repository_root: Path,
    *,
    candidate_head: str,
    candidate_tree: str,
    git_common_root: Path,
) -> FrozenSourceSnapshot:
    repository = repository_root.resolve()
    git_common = git_common_root.resolve()
    run_root = _create_run_root()
    try:
        if _paths_overlap(run_root, repository) or _paths_overlap(run_root, git_common):
            raise VerificationReceiptError(
                "trusted verification temporary directory overlaps candidate or Git metadata"
            )
        source_root = run_root / "source"
        host_home = run_root / "host-home"
        host_home.mkdir()
        git = resolve_trusted_system_executable("git")
        env = minimal_trusted_git_environment(host_home)
        observed_tree = _run_host_command(
            (
                str(git),
                *TRUSTED_GIT_CONFIG_ARGS,
                "-C",
                str(repository),
                "rev-parse",
                f"{candidate_head}^{{tree}}",
            ),
            env=env,
        )
        if observed_tree != candidate_tree:
            raise VerificationReceiptError(
                "candidate tree changed before the frozen snapshot was exported"
            )
        source_root.mkdir(mode=0o700)
        _export_git_tree(
            git=git,
            repository=repository,
            candidate_head=candidate_head,
            source_root=source_root,
            env=env,
        )
        _freeze_source_tree(source_root)
        snapshot = FrozenSourceSnapshot(
            run_root=run_root,
            source_root=source_root,
            candidate_head=candidate_head,
            candidate_tree=candidate_tree,
            content_sha256=_source_tree_sha256(source_root),
        )
        verify_frozen_source_snapshot(snapshot)
        return snapshot
    except Exception:
        _make_source_tree_writable(run_root / "source")
        shutil.rmtree(run_root, ignore_errors=True)
        raise


def verify_frozen_source_snapshot(snapshot: FrozenSourceSnapshot) -> None:
    dot_git = snapshot.source_root / ".git"
    if dot_git.exists() or dot_git.is_symlink():
        raise VerificationReceiptError(
            "frozen verification snapshot unexpectedly contains Git metadata"
        )
    observed = _source_tree_sha256(snapshot.source_root)
    if observed != snapshot.content_sha256:
        raise VerificationReceiptError(
            "frozen verification snapshot content changed during trusted checks"
        )


def cleanup_frozen_source_snapshot(snapshot: FrozenSourceSnapshot) -> None:
    _make_source_tree_writable(snapshot.source_root)
    shutil.rmtree(snapshot.run_root, ignore_errors=True)


def _export_git_tree(
    *,
    git: Path,
    repository: Path,
    candidate_head: str,
    source_root: Path,
    env: dict[str, str],
) -> None:
    entries = _read_git_tree_entries(git, repository, candidate_head, env)
    with tempfile.TemporaryFile() as error_file:
        process = subprocess.Popen(
            (
                str(git),
                *TRUSTED_GIT_CONFIG_ARGS,
                "-C",
                str(repository),
                "cat-file",
                "--batch",
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_file,
            env=env,
        )
        if process.stdin is None or process.stdout is None:
            process.kill()
            process.wait(timeout=_PROCESS_STOP_SECONDS)
            raise VerificationReceiptError(
                "trusted Git object reader pipes were not created"
            )
        total_size = 0
        try:
            for entry in entries:
                process.stdin.write(f"{entry.object_id}\n".encode("ascii"))
                process.stdin.flush()
                header = process.stdout.readline(1024)
                object_id, object_type, size = _parse_git_object_header(header)
                if object_id != entry.object_id or object_type != "blob":
                    raise VerificationReceiptError(
                        f"trusted Git returned the wrong object for {entry.relative}"
                    )
                if size > _MAX_SNAPSHOT_FILE_BYTES:
                    raise VerificationReceiptError(
                        f"frozen verification file exceeds the size limit: {entry.relative}"
                    )
                total_size += size
                if total_size > _MAX_SNAPSHOT_TOTAL_BYTES:
                    raise VerificationReceiptError(
                        "frozen verification snapshot exceeds the total size limit"
                    )
                destination = source_root.joinpath(*entry.relative.parts)
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                bounded = _BoundedReader(process.stdout, size)
                if entry.mode == _GIT_SYMLINK_MODE:
                    if size > _MAX_SYMLINK_BYTES:
                        raise VerificationReceiptError(
                            f"frozen verification symlink is too large: {entry.relative}"
                        )
                    target = os.fsdecode(bounded.read_all())
                    _validate_tree_symlink(entry.relative, target)
                    os.symlink(target, destination)
                else:
                    _write_snapshot_file(
                        destination, bounded, entry.mode, expected_size=size
                    )
                if bounded.remaining != 0 or process.stdout.read(1) != b"\n":
                    raise VerificationReceiptError(
                        f"trusted Git object was truncated: {entry.relative}"
                    )
            process.stdin.close()
            if process.wait(timeout=_HOST_COMMAND_TIMEOUT_SECONDS) != 0:
                raise VerificationReceiptError(
                    _git_object_reader_diagnostic(error_file)
                )
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=_PROCESS_STOP_SECONDS)
            raise
        finally:
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()


def _read_git_tree_entries(
    git: Path, repository: Path, candidate_head: str, env: dict[str, str]
) -> tuple[_GitTreeEntry, ...]:
    try:
        with (
            tempfile.TemporaryFile() as output_file,
            tempfile.TemporaryFile() as error_file,
        ):
            result = subprocess.run(
                (
                    str(git),
                    *TRUSTED_GIT_CONFIG_ARGS,
                    "-C",
                    str(repository),
                    "ls-tree",
                    "-r",
                    "-z",
                    "--full-tree",
                    candidate_head,
                ),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=output_file,
                stderr=error_file,
                env=env,
                timeout=_HOST_COMMAND_TIMEOUT_SECONDS,
            )
            output_size = os.fstat(output_file.fileno()).st_size
            if output_size > _MAX_TREE_LISTING_BYTES:
                raise VerificationReceiptError(
                    "trusted Git tree listing exceeds the size limit"
                )
            output_file.seek(0)
            error_file.seek(0)
            raw_output = output_file.read()
            raw_error = error_file.read(_HOST_DIAGNOSTIC_BYTES)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerificationReceiptError(
            f"trusted Git tree listing failed to run: {exc}"
        ) from exc
    if result.returncode != 0:
        diagnostic = (
            decode_safe(raw_error, from_subprocess=True).text.strip() or "no output"
        )
        raise VerificationReceiptError(f"trusted Git tree listing failed: {diagnostic}")
    records = tuple(record for record in raw_output.split(b"\0") if record)
    if len(records) > _MAX_SNAPSHOT_ENTRIES:
        raise VerificationReceiptError(
            "frozen verification snapshot contains too many entries"
        )
    entries = tuple(_parse_git_tree_entry(record) for record in records)
    if len({entry.relative for entry in entries}) != len(entries):
        raise VerificationReceiptError(
            "trusted Git tree listing contains duplicate paths"
        )
    return entries


def _parse_git_tree_entry(record: bytes) -> _GitTreeEntry:
    metadata, separator, raw_path = record.partition(b"\t")
    fields = metadata.split(b" ")
    if not separator or len(fields) != _GIT_TREE_FIELD_COUNT:
        raise VerificationReceiptError("trusted Git tree listing is malformed")
    raw_mode, object_type, raw_object_id = fields
    try:
        mode = int(raw_mode, 8)
        object_id = raw_object_id.decode("ascii")
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerificationReceiptError("trusted Git tree listing is malformed") from exc
    if object_type != b"blob" or mode not in {0o100644, 0o100755, _GIT_SYMLINK_MODE}:
        raise VerificationReceiptError(
            f"trusted verification does not support Git mode {raw_mode.decode(errors='replace')}"
        )
    if re.fullmatch(r"[0-9a-f]{40}", object_id) is None:
        raise VerificationReceiptError(
            "trusted Git tree listing contains an invalid object ID"
        )
    relative = _validated_tree_path(os.fsdecode(raw_path))
    return _GitTreeEntry(mode=mode, object_id=object_id, relative=relative)


def _parse_git_object_header(header: bytes) -> tuple[str, str, int]:
    if not header.endswith(b"\n") or len(header) >= _MAX_GIT_OBJECT_HEADER_BYTES:
        raise VerificationReceiptError("trusted Git object header is malformed")
    fields = header.rstrip(b"\n").split(b" ")
    if len(fields) != _GIT_TREE_FIELD_COUNT:
        raise VerificationReceiptError("trusted Git object header is malformed")
    try:
        object_id = fields[0].decode("ascii")
        object_type = fields[1].decode("ascii")
        size = int(fields[2])
    except (UnicodeDecodeError, ValueError) as exc:
        raise VerificationReceiptError(
            "trusted Git object header is malformed"
        ) from exc
    if size < 0:
        raise VerificationReceiptError("trusted Git object has a negative size")
    return object_id, object_type, size


def _git_object_reader_diagnostic(error_file: BinaryIO) -> str:
    error_file.seek(0)
    raw = error_file.read(_HOST_DIAGNOSTIC_BYTES)
    diagnostic = decode_safe(raw, from_subprocess=True).text.strip() or "no output"
    return f"trusted Git object reader failed: {diagnostic}"


class _BoundedReader:
    def __init__(self, source: _ByteReader, size: int) -> None:
        self.source = source
        self.remaining = size

    def read(self, size: int = -1) -> bytes:
        if self.remaining == 0:
            return b""
        requested = self.remaining if size < 0 else min(size, self.remaining)
        chunk = self.source.read(requested)
        self.remaining -= len(chunk)
        return chunk

    def read_all(self) -> bytes:
        chunks: list[bytes] = []
        while chunk := self.read(64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)


def _validated_tree_path(raw: str) -> PurePosixPath:
    relative = PurePosixPath(raw)
    if (
        not raw
        or relative.is_absolute()
        or ".." in relative.parts
        or any(part.casefold() == ".git" for part in relative.parts)
    ):
        raise VerificationReceiptError(
            f"frozen verification tree contains an unsafe path: {raw!r}"
        )
    normalized = tuple(part for part in relative.parts if part not in {"", "."})
    if not normalized:
        raise VerificationReceiptError(
            f"frozen verification tree contains an empty path: {raw!r}"
        )
    return PurePosixPath(*normalized)


def _validate_tree_symlink(relative: PurePosixPath, raw_target: str) -> None:
    target = PurePosixPath(raw_target)
    if not raw_target or target.is_absolute():
        raise VerificationReceiptError(
            f"frozen verification symlink escapes the snapshot: {relative}"
        )
    stack = list(relative.parent.parts)
    for part in target.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not stack:
                raise VerificationReceiptError(
                    f"frozen verification symlink escapes the snapshot: {relative}"
                )
            stack.pop()
            continue
        stack.append(part)


def _write_snapshot_file(
    destination: Path, source: _ByteReader, mode: int, *, expected_size: int
) -> None:
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o700 if mode & 0o111 else 0o600,
    )
    observed_size = 0
    try:
        while chunk := source.read(64 * 1024):
            observed_size += len(chunk)
            offset = 0
            while offset < len(chunk):
                written = os.write(descriptor, chunk[offset:])
                if written <= 0:
                    raise VerificationReceiptError(
                        "frozen verification file write did not make progress"
                    )
                offset += written
    finally:
        os.close(descriptor)
    if observed_size != expected_size:
        raise VerificationReceiptError(
            f"frozen verification file size changed while extracting {destination.name}"
        )


def _source_tree_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    entries = 0
    total_size = 0
    pending = [(source_root, PurePosixPath(), 0)]
    while pending:
        directory, relative, depth = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise VerificationReceiptError(
                f"frozen verification snapshot could not be inspected: {exc}"
            ) from exc
        descendant_directories: list[tuple[Path, PurePosixPath, int]] = []
        for child in children:
            entries += 1
            if entries > _MAX_SNAPSHOT_ENTRIES:
                raise VerificationReceiptError(
                    "frozen verification snapshot contains too many entries"
                )
            child_relative = relative / child.name
            name = os.fsencode(str(child_relative))
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise VerificationReceiptError(
                    f"frozen verification snapshot entry changed: {child_relative}"
                ) from exc
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            if stat.S_ISDIR(info.st_mode):
                digest.update(b"d")
                child_depth = depth + 1
                if child_depth > _MAX_SNAPSHOT_DEPTH:
                    raise VerificationReceiptError(
                        "frozen verification snapshot exceeds the depth limit"
                    )
                descendant_directories.append((
                    Path(child.path),
                    child_relative,
                    child_depth,
                ))
                continue
            if stat.S_ISLNK(info.st_mode):
                digest.update(b"l")
                target = os.fsencode(os.readlink(child.path))
                digest.update(len(target).to_bytes(8, "big"))
                digest.update(target)
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise VerificationReceiptError(
                    f"frozen verification snapshot has unsafe entry: {child_relative}"
                )
            if info.st_size > _MAX_SNAPSHOT_FILE_BYTES:
                raise VerificationReceiptError(
                    f"frozen verification file exceeds the size limit: {child_relative}"
                )
            total_size += info.st_size
            if total_size > _MAX_SNAPSHOT_TOTAL_BYTES:
                raise VerificationReceiptError(
                    "frozen verification snapshot exceeds the total size limit"
                )
            digest.update(b"x" if info.st_mode & 0o111 else b"f")
            digest.update(info.st_size.to_bytes(8, "big"))
            _hash_snapshot_file(Path(child.path), info, digest)
        pending.extend(reversed(descendant_directories))
    return digest.hexdigest()


def _hash_snapshot_file(path: Path, before: os.stat_result, digest: _Digest) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        while chunk := os.read(descriptor, 64 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise VerificationReceiptError(
            f"frozen verification snapshot changed while hashing {path.name}"
        )


def _freeze_source_tree(source_root: Path) -> None:
    entries = _walk_source_tree(source_root)
    for path, info in entries:
        if stat.S_ISREG(info.st_mode):
            path.chmod(0o555 if info.st_mode & 0o111 else 0o444)
    for path, info in reversed(entries):
        if stat.S_ISDIR(info.st_mode):
            path.chmod(0o555)
    source_root.chmod(0o555)


def _make_source_tree_writable(source_root: Path) -> None:
    try:
        source_root.chmod(0o700)
        entries = _walk_source_tree(source_root)
    except OSError:
        return
    for path, info in entries:
        if stat.S_ISDIR(info.st_mode):
            path.chmod(0o700)
        elif stat.S_ISREG(info.st_mode):
            path.chmod(0o600)


def _walk_source_tree(source_root: Path) -> list[tuple[Path, os.stat_result]]:
    entries: list[tuple[Path, os.stat_result]] = []
    pending = [(source_root, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                children = list(iterator)
        except OSError as exc:
            raise VerificationReceiptError(
                f"frozen verification snapshot could not be walked: {exc}"
            ) from exc
        descendant_directories: list[tuple[Path, int]] = []
        for child in children:
            if len(entries) >= _MAX_SNAPSHOT_ENTRIES:
                raise VerificationReceiptError(
                    "frozen verification snapshot contains too many entries"
                )
            info = child.stat(follow_symlinks=False)
            path = Path(child.path)
            entries.append((path, info))
            if stat.S_ISDIR(info.st_mode):
                child_depth = depth + 1
                if child_depth > _MAX_SNAPSHOT_DEPTH:
                    raise VerificationReceiptError(
                        "frozen verification snapshot exceeds the depth limit"
                    )
                descendant_directories.append((path, child_depth))
        pending.extend(reversed(descendant_directories))
    return entries


def resolve_trusted_executable(
    argv0: str,
    *,
    forbidden_roots: tuple[Path, ...],
    expected_sha256: str | None,
    materialization_root: Path,
) -> TrustedExecutable:
    requested = Path(argv0).expanduser()
    if requested.is_absolute():
        lexical = requested.absolute()
    elif requested.parent == Path("."):
        lexical = resolve_trusted_system_executable(argv0)
    else:
        raise VerificationReceiptError(
            "trusted check executable must be an absolute path or a bare name on "
            f"the sanitized system PATH: {argv0!r}"
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise VerificationReceiptError(
            f"trusted check executable could not be resolved: {argv0!r}"
        ) from exc
    if not os.access(resolved, os.X_OK):
        raise VerificationReceiptError(
            f"trusted check executable is not an executable regular file: {resolved}"
        )
    try:
        validate_trusted_command_argv((str(resolved),))
    except ValueError as exc:
        raise VerificationReceiptError(str(exc)) from exc
    for root in forbidden_roots:
        canonical = root.resolve()
        if lexical.is_relative_to(canonical) or resolved.is_relative_to(canonical):
            raise VerificationReceiptError(
                "trusted check executable cannot come from candidate, Git, or "
                f"temporary verification state: {lexical}"
            )
    stable = _read_trusted_executable(resolved)
    digest = hashlib.sha256(stable.payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise VerificationReceiptError(
            "trusted check executable SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {digest} for {resolved}"
        )
    roots = _executable_read_roots(lexical, resolved)
    materialized_path, materialized = _materialize_trusted_executable(
        stable.payload, materialization_root
    )
    roots.add(materialization_root)
    return TrustedExecutable(
        lexical_path=lexical,
        resolved_path=resolved,
        materialization_root=materialization_root,
        materialized_path=materialized_path,
        sha256=digest,
        source_identity=_stable_file_identity(stable),
        materialized_identity=_stable_file_identity(materialized),
        read_roots=tuple(sorted(roots)),
    )


def validate_trusted_executable(executable: TrustedExecutable) -> None:
    try:
        current_target = executable.lexical_path.resolve(strict=True)
    except OSError as exc:
        raise VerificationReceiptError(
            f"trusted check executable disappeared during execution: {executable.lexical_path}"
        ) from exc
    if current_target != executable.resolved_path:
        raise VerificationReceiptError(
            "trusted check executable target changed during execution: "
            f"{executable.lexical_path}"
        )
    current = _read_trusted_executable(current_target)
    if _stable_file_identity(current) != executable.source_identity:
        raise VerificationReceiptError(
            "trusted check executable identity changed during execution: "
            f"{executable.lexical_path}"
        )
    observed = hashlib.sha256(current.payload).hexdigest()
    if observed != executable.sha256:
        raise VerificationReceiptError(
            "trusted check executable content changed during execution: "
            f"{executable.lexical_path}"
        )
    materialized = _read_materialized_executable(executable.materialized_path)
    if _stable_file_identity(materialized) != executable.materialized_identity:
        raise VerificationReceiptError(
            "runner-owned trusted executable identity changed during execution"
        )
    materialized_digest = hashlib.sha256(materialized.payload).hexdigest()
    if materialized_digest != executable.sha256:
        raise VerificationReceiptError(
            "runner-owned trusted executable content changed during execution"
        )


def cleanup_trusted_executable(executable: TrustedExecutable) -> None:
    _remove_materialization_root(
        executable.materialization_root, executable.materialized_path
    )


def _remove_materialization_root(root: Path, executable: Path) -> None:
    try:
        root.chmod(0o700)
        executable.chmod(0o600)
    except OSError:
        pass
    shutil.rmtree(root, ignore_errors=True)


def _read_trusted_executable(path: Path) -> StableFile:
    try:
        stable = read_stable_absolute_file(
            path, max_bytes=_MAX_TRUSTED_EXECUTABLE_BYTES
        )
    except (ImmutableStoreError, OSError) as exc:
        raise VerificationReceiptError(
            f"trusted check executable could not be read safely: {path}: {exc}"
        ) from exc
    if stable.payload.startswith(b"#!"):
        raise VerificationReceiptError(
            "trusted verification does not execute shebang wrappers; configure "
            "the pinned native interpreter and pass the module or script as an argument"
        )
    return stable


def _read_materialized_executable(path: Path) -> StableFile:
    try:
        return read_stable_absolute_file(path, max_bytes=_MAX_TRUSTED_EXECUTABLE_BYTES)
    except (ImmutableStoreError, OSError) as exc:
        raise VerificationReceiptError(
            f"runner-owned trusted executable could not be read safely: {exc}"
        ) from exc


def _materialize_trusted_executable(
    payload: bytes, materialization_root: Path
) -> tuple[Path, StableFile]:
    if not materialization_root.is_absolute():
        raise VerificationReceiptError(
            "trusted executable materialization root must be absolute"
        )
    try:
        materialization_root.mkdir(mode=0o700)
    except OSError as exc:
        raise VerificationReceiptError(
            "trusted executable materialization root could not be created"
        ) from exc
    root_descriptor = -1
    try:
        root_descriptor = os.open(
            materialization_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        _write_materialized_executable(root_descriptor, payload)
        os.fsync(root_descriptor)
        os.fchmod(root_descriptor, 0o555)
        os.fsync(root_descriptor)
    except Exception:
        if root_descriptor >= 0:
            os.close(root_descriptor)
            root_descriptor = -1
        try:
            materialization_root.chmod(0o700)
        except OSError:
            pass
        shutil.rmtree(materialization_root, ignore_errors=True)
        raise
    finally:
        if root_descriptor >= 0:
            os.close(root_descriptor)
    path = materialization_root / "executable"
    try:
        stable = _read_materialized_executable(path)
    except Exception:
        _remove_materialization_root(materialization_root, path)
        raise
    if stable.payload != payload:
        _remove_materialization_root(materialization_root, path)
        raise VerificationReceiptError(
            "runner-owned trusted executable content changed during materialization"
        )
    return path, stable


def _write_materialized_executable(root_descriptor: int, payload: bytes) -> None:
    descriptor = os.open(
        "executable",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o700,
        dir_fd=root_descriptor,
    )
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise VerificationReceiptError(
                    "trusted executable materialization made no progress"
                )
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o555)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != len(payload)
        ):
            raise VerificationReceiptError(
                "runner-owned trusted executable was not materialized safely"
            )
    finally:
        os.close(descriptor)


def _stable_file_identity(stable: StableFile) -> tuple[int, int, int, int, int]:
    return (
        stable.device,
        stable.inode,
        stable.size,
        stable.modified_ns,
        stable.changed_ns,
    )


def resolve_environment_attestation(
    path: str | None, expected_sha256: str | None, *, forbidden_roots: tuple[Path, ...]
) -> TrustedEnvironmentAttestation | None:
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise VerificationReceiptError(
            "environment attestation path and SHA-256 must be configured together"
        )
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise VerificationReceiptError("environment attestation path must be absolute")
    lexical = requested.absolute()
    for root in forbidden_roots:
        canonical = root.resolve()
        if lexical.is_relative_to(canonical):
            raise VerificationReceiptError(
                "environment attestation cannot come from candidate, Git, or "
                f"temporary verification state: {lexical}"
            )
    observed = _environment_attestation_sha256(lexical)
    if observed != expected_sha256:
        raise VerificationReceiptError(
            "environment attestation SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed} for {lexical}"
        )
    return TrustedEnvironmentAttestation(path=lexical, sha256=observed)


def validate_environment_attestation(
    attestation: TrustedEnvironmentAttestation | None,
) -> None:
    if attestation is None:
        return
    observed = _environment_attestation_sha256(attestation.path)
    if observed != attestation.sha256:
        raise VerificationReceiptError(
            "environment attestation changed during trusted check execution: "
            f"{attestation.path}"
        )


def _environment_attestation_sha256(path: Path) -> str:
    try:
        stable = read_stable_absolute_file(
            path, max_bytes=_MAX_ENVIRONMENT_ATTESTATION_BYTES
        )
    except (ImmutableStoreError, OSError) as exc:
        raise VerificationReceiptError(
            f"environment attestation could not be read safely: {path}: {exc}"
        ) from exc
    return hashlib.sha256(stable.payload).hexdigest()


def stable_file_sha256(path: Path) -> str:
    try:
        resolved = path.resolve(strict=True)
        stable = read_stable_absolute_file(
            resolved, max_bytes=_MAX_TRUSTED_EXECUTABLE_BYTES
        )
        return hashlib.sha256(stable.payload).hexdigest()
    except (ImmutableStoreError, OSError) as exc:
        raise VerificationReceiptError(
            f"trusted executable could not be hashed safely: {path}"
        ) from exc


def minimal_check_environment(writable_root: Path) -> dict[str, str]:
    directories = {
        "HOME": writable_root / "home",
        "TMPDIR": writable_root / "tmp",
        "XDG_CACHE_HOME": writable_root / "cache",
        "XDG_CONFIG_HOME": writable_root / "config",
        "XDG_DATA_HOME": writable_root / "data",
        "UV_CACHE_DIR": writable_root / "uv-cache",
        "PRE_COMMIT_HOME": writable_root / "pre-commit",
        "PIP_CACHE_DIR": writable_root / "pip-cache",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    env = {name: str(path) for name, path in directories.items()}
    env.update({
        "PATH": TRUSTED_SYSTEM_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "USER": "vibe-verifier",
        "LOGNAME": "vibe-verifier",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "UV_OFFLINE": "1",
        "VIBE_TRUSTED_VERIFICATION": "1",
    })
    return env


def run_bounded_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float,
    max_output_bytes: int = MAX_COMBINED_OUTPUT_BYTES,
) -> BoundedProcessResult:
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        _terminate_process_group(process)
        raise VerificationReceiptError("trusted check output pipes were not created")
    output = _CombinedOutput(max_output_bytes)
    readers = (
        threading.Thread(
            target=_drain_pipe, args=(process.stdout, "stdout", output), daemon=True
        ),
        threading.Thread(
            target=_drain_pipe, args=(process.stderr, "stderr", output), daemon=True
        ),
    )
    started_readers: list[threading.Thread] = []
    try:
        for reader in readers:
            reader.start()
            started_readers.append(reader)
    except RuntimeError as exc:
        _terminate_process_group(process)
        process.stdout.close()
        process.stderr.close()
        for reader in started_readers:
            reader.join(timeout=_PROCESS_STOP_SECONDS)
        raise VerificationReceiptError(
            "trusted check output readers could not be started"
        ) from exc

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    output_limited = False
    while process.poll() is None:
        if output.exceeded.wait(timeout=0.01):
            output_limited = True
            _terminate_process_group(process)
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_process_group(process)
            break
    if process.poll() is None:
        _terminate_process_group(process)
    for reader in readers:
        reader.join(timeout=_PROCESS_STOP_SECONDS)
    readers_alive = any(reader.is_alive() for reader in readers)
    if readers_alive:
        process.stdout.close()
        process.stderr.close()
        for reader in readers:
            reader.join(timeout=_PROCESS_STOP_SECONDS)
        readers_alive = any(reader.is_alive() for reader in readers)
    output_limited = output_limited or output.exceeded.is_set()
    collector_error = output.collector_diagnostic(readers_alive=readers_alive)

    stderr = bytes(output.stderr)
    exit_code = process.returncode
    if timed_out:
        exit_code = None
    if output_limited:
        exit_code = None
        stderr += (
            "\ntrusted verification combined output exceeded "
            f"{max_output_bytes} bytes; process terminated\n"
        ).encode()
    if collector_error is not None:
        exit_code = None
        stderr += (
            "\ntrusted verification output collection was incomplete: "
            f"{collector_error}\n"
        ).encode()
    return BoundedProcessResult(
        stdout=bytes(output.stdout),
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        output_limited=output_limited,
        collector_error=collector_error,
    )


def _drain_pipe(pipe: BinaryIO, stream: str, output: _CombinedOutput) -> None:
    try:
        while chunk := pipe.read(_OUTPUT_CHUNK_BYTES):
            output.append(stream, chunk)
    except Exception as exc:
        output.record_reader_error(stream, exc)
    finally:
        output.record_reader_complete(stream)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    _signal_process(process, signal.SIGTERM)
    try:
        process.wait(timeout=_PROCESS_STOP_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process(process, signal.SIGKILL)
    try:
        process.wait(timeout=_PROCESS_STOP_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_PROCESS_STOP_SECONDS)


def _signal_process(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if signal_owned_process_group(process.pid, sig):
        return
    try:
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError):
        pass


def _executable_read_roots(lexical: Path, resolved: Path) -> set[Path]:
    roots = {lexical.parent.resolve(), resolved.parent.resolve()}
    for executable in (lexical, resolved):
        parent = executable.parent.resolve()
        environment_root = parent.parent
        if parent.name == "bin" and (environment_root / "pyvenv.cfg").is_file():
            roots.add(environment_root)
            python = parent / "python"
            if python.exists():
                roots.update(_python_install_roots(python))
        roots.update(_python_install_roots(executable))
    return {root for root in roots if root.is_dir()}


def _python_install_roots(executable: Path) -> set[Path]:
    try:
        resolved = executable.resolve(strict=True)
    except OSError:
        return set()
    if not resolved.name.lower().startswith("python"):
        return set()
    return {resolved.parent.parent}


def _run_host_command(argv: tuple[str, ...], *, env: dict[str, str]) -> str:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            env=env,
            text=False,
            timeout=_HOST_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerificationReceiptError(
            f"trusted host command failed to start: {Path(argv[0]).name}: {exc}"
        ) from exc
    if result.returncode != 0:
        raw = (result.stderr or result.stdout)[-_HOST_DIAGNOSTIC_BYTES:]
        diagnostic = decode_safe(raw, from_subprocess=True).text.strip() or "no output"
        raise VerificationReceiptError(
            f"trusted host command failed ({Path(argv[0]).name}): {diagnostic}"
        )
    return decode_safe(result.stdout, from_subprocess=True).text.strip()


def _paths_overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _create_run_root() -> Path:
    for _ in range(10):
        candidate = Path("/tmp") / f"vibe-trusted-verification-{uuid.uuid4().hex}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate.resolve()
    raise VerificationReceiptError(
        "could not allocate an isolated trusted verification directory"
    )


__all__ = [
    "MAX_COMBINED_OUTPUT_BYTES",
    "BoundedProcessResult",
    "FrozenSourceSnapshot",
    "TrustedEnvironmentAttestation",
    "TrustedExecutable",
    "cleanup_frozen_source_snapshot",
    "cleanup_trusted_executable",
    "create_frozen_source_snapshot",
    "minimal_check_environment",
    "resolve_environment_attestation",
    "resolve_trusted_executable",
    "run_bounded_process",
    "validate_environment_attestation",
    "validate_trusted_executable",
    "verify_frozen_source_snapshot",
]
