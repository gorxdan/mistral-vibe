from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
import getpass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Literal

import orjson

from vibe.core.logger import logger
from vibe.core.session.session_id import shorten_session_id
from vibe.core.session.session_loader import (
    MESSAGES_FILENAME,
    METADATA_FILENAME,
    SessionLoader,
)
from vibe.core.session.title_format import MAX_TITLE_LENGTH
from vibe.core.types import AgentStats, LLMMessage, Role, SessionMetadata
from vibe.core.utils import is_windows, utc_now
from vibe.core.utils.io import read_safe_async

if TYPE_CHECKING:
    from vibe.core.agents.models import AgentProfile
    from vibe.core.config import SessionLoggingConfig, VibeConfig
    from vibe.core.experiments.models import EvalResponse
    from vibe.core.tools.manager import ToolManager


TMP_CLEANUP_INTERVAL = timedelta(seconds=5)
# Static session context (tools schema, config dump, system prompt): ~100KB of
# analysis-only data no resume/picker reader consumes — kept out of the per-round meta.json.
CONTEXT_FILENAME = "context.json"
# Cap dirs archived per background run to bound startup-adjacent IO; any
# backlog drains over the next few sessions.
_ARCHIVE_MAX_PER_RUN = 25
_ARCHIVE_TMP_SUFFIX = ".tar.gz.tmp"
# Orphaned tmp tars (crashed archiver) are swept past this age; a live tar keeps
# its tmp mtime fresh, so age is a safe liveness proxy even for unlocked writers.
_ARCHIVE_TMP_MAX_AGE_SECONDS = 3600

# fcntl locking (POSIX) as in usage/_recorder.py; without it (Windows) concurrent
# archivers only duplicate work — unique tmps + the snapshot recheck carry safety.
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None


def _try_lock_archive_dir(archive_dir: Path) -> int | None:
    fd = os.open(archive_dir / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    if _fcntl is None:
        return fd
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _unlock_archive_dir(fd: int) -> None:
    try:
        if _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _sweep_stale_archive_tmps(archive_dir: Path) -> None:
    horizon = utc_now().timestamp() - _ARCHIVE_TMP_MAX_AGE_SECONDS
    for tmp in archive_dir.glob(f"*{_ARCHIVE_TMP_SUFFIX}"):
        try:
            if tmp.stat().st_mtime < horizon:
                tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _snapshot_dir_files(d: Path) -> tuple[int, dict[str, tuple[int, int]]] | None:
    # Dir mtime is blind to in-place appends (moves only on entry changes), so
    # staleness is per-file (mtime_ns, size) + dir mtime_ns for zero-write resume.
    try:
        files: dict[str, tuple[int, int]] = {}
        for p in d.rglob("*"):
            if p.is_file():
                st = p.stat()
                files[str(p.relative_to(d))] = (st.st_mtime_ns, st.st_size)
        return d.stat().st_mtime_ns, files
    except OSError:
        return None


def _prepare_archive_dir(archive_dir: Path) -> int | None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = _try_lock_archive_dir(archive_dir)
    if lock_fd is not None:
        _sweep_stale_archive_tmps(archive_dir)
    return lock_fd


def _archive_one_dir(
    d: Path, tarball: Path, archive_dir: Path, keep_dir: Callable[[], Path | None]
) -> bool:
    before = _snapshot_dir_files(d)
    if before is None:
        return False
    fd, tmp_name = tempfile.mkstemp(suffix=_ARCHIVE_TMP_SUFFIX, dir=archive_dir)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            tar.add(d, arcname=d.name)
        if _snapshot_dir_files(d) != before or d == keep_dir():
            return False  # concurrent writer; tar is stale
        tmp.replace(tarball)  # atomic; complete before dir removed
        shutil.rmtree(d, ignore_errors=True)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def archive_old_session_dirs(
    save_dir: Path,
    prefix: str,
    archive_after_days: int,
    keep: Path | Callable[[], Path | None] | None = None,
    max_per_run: int = _ARCHIVE_MAX_PER_RUN,
) -> None:
    """Tar+gzip session dirs untouched > *archive_after_days* into save_dir/archive/.

    Transcripts are valuable, so this never deletes data — it compresses a cold
    dir into ``archive/<name>.tar.gz`` (extractable to resume) and removes the
    now-redundant loose dir, which also speeds the live list/--continue scan. The
    tarball is written to a temp name and atomically renamed before the dir is
    removed, so a crash mid-archive never loses the dir. Best-effort; 0 disables.

    Concurrency: an exclusive flock on ``archive/.lock`` serializes archivers
    per save_dir (a loser skips the run), and each tar writes to a mkstemp-unique
    tmp so overlapping runs can never share an inode. A session resumed mid-run
    is protected by *keep* (a callable re-resolved around each dir, tracking the
    live session_dir) and by a per-file (mtime_ns, size) snapshot rechecked
    after tar — dir mtime alone is blind to in-place transcript appends.
    """
    if archive_after_days <= 0:
        return

    def keep_dir() -> Path | None:
        return keep() if callable(keep) else keep

    try:
        cutoff = utc_now().timestamp() - archive_after_days * 86400
        archive_dir = save_dir / "archive"
        done = 0
        lock_fd: int | None = None
        try:
            for d in sorted(save_dir.glob(f"{prefix}_*"), key=lambda p: p.name):
                if done >= max_per_run:
                    break
                try:
                    if not d.is_dir() or d == keep_dir():
                        continue
                    if d.stat().st_mtime >= cutoff:
                        continue
                    tarball = archive_dir / f"{d.name}.tar.gz"
                    if tarball.exists():
                        continue
                    if lock_fd is None:
                        lock_fd = _prepare_archive_dir(archive_dir)
                    if lock_fd is None:
                        return  # another archiver owns this save_dir
                    if _archive_one_dir(d, tarball, archive_dir, keep_dir):
                        done += 1
                except OSError:
                    pass
        finally:
            if lock_fd is not None:
                _unlock_archive_dir(lock_fd)
    except Exception:
        logger.debug("session dir archive skipped", exc_info=True)


class SessionLogger:
    def __init__(self, session_config: SessionLoggingConfig, session_id: str) -> None:
        self.session_config = session_config
        self.enabled = session_config.enabled
        self._last_tmp_cleanup_at: datetime | None = None
        self._tmp_cleanup_lock = Lock()
        self._static_context_fingerprint: str | None = None
        self._archive_thread: Thread | None = None

        if not self.enabled:
            self.save_dir: Path | None = None
            self.session_prefix: str | None = None
            self.session_id: str = "disabled"
            self.session_start_time: str = "N/A"
            self.session_dir: Path | None = None
            self.session_metadata: SessionMetadata | None = None
            return

        self.save_dir = Path(session_config.save_dir)
        self.session_prefix = session_config.session_prefix
        self.session_id = session_id
        self.session_start_time = utc_now().isoformat()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.save_folder
        # Off the critical init path; keep=lambda tracks resume_existing_session
        # retargeting session_dir after this thread starts.
        self._archive_thread = Thread(
            target=archive_old_session_dirs,
            args=(
                self.save_dir,
                self.session_prefix,
                session_config.archive_after_days,
            ),
            kwargs={"keep": lambda: self.session_dir},
            name="session-archiver",
            daemon=True,
        )
        self._archive_thread.start()
        self.session_metadata = self._initialize_session_metadata()

    @property
    def save_folder(self) -> Path:
        if self.save_dir is None or self.session_prefix is None:
            raise RuntimeError(
                "Cannot get session save folder when logging is disabled"
            )

        timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
        folder_name = (
            f"{self.session_prefix}_{timestamp}_{shorten_session_id(self.session_id)}"
        )
        return self.save_dir / folder_name

    def _get_session_info(self) -> tuple[Path, SessionMetadata] | None:
        if (
            not self.enabled
            or self.session_dir is None
            or self.session_metadata is None
        ):
            return None
        return (self.session_dir, self.session_metadata)

    @property
    def metadata_filepath(self) -> Path:
        if self.session_dir is None:
            raise RuntimeError(
                "Cannot get session metadata filepath when logging is disabled"
            )
        return self.session_dir / METADATA_FILENAME

    @property
    def messages_filepath(self) -> Path:
        if self.session_dir is None:
            raise RuntimeError(
                "Cannot get session messages filepath when logging is disabled"
            )
        return self.session_dir / MESSAGES_FILENAME

    def _fetch_git_metadata(self) -> tuple[str | None, str | None]:
        """Fetch git commit and branch in a single subprocess call."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD", "--abbrev-ref", "HEAD"],
                capture_output=True,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().splitlines()
                commit = lines[0] if len(lines) > 0 else None
                branch = lines[1] if len(lines) > 1 else None
                return commit, branch
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass
        return None, None

    @property
    def git_commit(self) -> str | None:
        return self._fetch_git_metadata()[0]

    @property
    def git_branch(self) -> str | None:
        return self._fetch_git_metadata()[1]

    @property
    def username(self) -> str:
        try:
            return getpass.getuser()
        except Exception:
            return "unknown"

    def _initialize_session_metadata(self) -> SessionMetadata:
        git_commit, git_branch = self._fetch_git_metadata()
        user_name = self.username

        # Use original_working_directory() so the recorded path is the user's
        # real checkout, not the worktree path. This keeps the exact-string
        # match in session_loader working after the worktree is removed.
        from vibe.core.worktree.manager import original_working_directory

        return SessionMetadata(
            session_id=self.session_id,
            start_time=self.session_start_time,
            end_time=None,
            git_commit=git_commit,
            git_branch=git_branch,
            username=user_name,
            environment={"working_directory": original_working_directory()},
            title=None,
            title_source="auto",
        )

    def _fallback_title_from_messages(self, messages: Sequence[LLMMessage]) -> str:
        first_user_message = None
        for message in messages:
            if message.role == Role.USER:
                first_user_message = message
                break

        if first_user_message is None:
            return "Untitled session"

        text = str(first_user_message.content)
        title = text[:MAX_TITLE_LENGTH]
        if len(text) > MAX_TITLE_LENGTH:
            title += "…"
        return title

    def _set_title_state(
        self, title: str | None, *, source: Literal["auto", "manual"]
    ) -> None:
        if self.session_metadata is None:
            return

        self.session_metadata.title = title
        self.session_metadata.title_source = source

    def set_title(self, title: str | None) -> None:
        if title is None:
            self._set_title_state(None, source="auto")
            return

        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Session title cannot be empty.")

        self._set_title_state(normalized_title, source="manual")

    def needs_initial_auto_title(self) -> bool:
        return self.session_metadata is not None and self.session_metadata.title is None

    def set_initial_auto_title(self, title: str) -> bool:
        if not self.needs_initial_auto_title():
            return False

        normalized_title = title.strip()
        if not normalized_title:
            return False

        self._set_title_state(normalized_title, source="auto")
        return True

    def _resolve_title(self, messages: Sequence[LLMMessage]) -> str | None:
        if self.session_metadata is None:
            return self._fallback_title_from_messages(messages)

        if self.session_metadata.title is not None:
            return self.session_metadata.title

        title = self._fallback_title_from_messages(messages)
        self._set_title_state(title, source="auto")
        return title

    # Both persist methods resolve only after write+fsync land (durability
    # contract, incl. act()'s finally-save on cancel); sequential awaits keep append order.
    @staticmethod
    async def persist_metadata(metadata: Any, session_dir: Path) -> None:
        await asyncio.to_thread(
            SessionLogger._persist_metadata_sync, metadata, session_dir
        )

    @staticmethod
    def _atomic_fsync_write_sync(payload: bytes, target: Path) -> None:
        temp_filepath = None
        try:
            # write_safe doesn't fit: the .json.tmp suffix (cleanup_tmp_files
            # contract) and fsync-before-rename are both required here.
            fd, tmp_name = tempfile.mkstemp(suffix=".json.tmp", dir=target.parent)
            temp_filepath = Path(tmp_name)
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_filepath, target)
        finally:
            if temp_filepath and temp_filepath.exists() and temp_filepath.is_file():
                temp_filepath.unlink()

    @staticmethod
    def _persist_metadata_sync(metadata: Any, session_dir: Path) -> None:
        metadata_filepath = session_dir / METADATA_FILENAME
        try:
            SessionLogger._atomic_fsync_write_sync(
                orjson.dumps(metadata, option=orjson.OPT_INDENT_2), metadata_filepath
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist session metadata to {metadata_filepath}: {e}"
            ) from e

    async def _persist_static_context(
        self, context: dict[str, Any], session_dir: Path
    ) -> None:
        payload = orjson.dumps(context, option=orjson.OPT_INDENT_2)
        fingerprint = hashlib.sha256(payload).hexdigest()
        context_filepath = session_dir / CONTEXT_FILENAME
        if (
            fingerprint == self._static_context_fingerprint
            and context_filepath.exists()
        ):
            return
        try:
            await asyncio.to_thread(
                SessionLogger._atomic_fsync_write_sync, payload, context_filepath
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist session context to {context_filepath}: {e}"
            ) from e
        self._static_context_fingerprint = fingerprint

    @staticmethod
    async def persist_messages(messages: list[dict], session_dir: Path) -> None:
        await asyncio.to_thread(
            SessionLogger._persist_messages_sync, messages, session_dir
        )

    @staticmethod
    def _persist_messages_sync(messages: list[dict], session_dir: Path) -> None:
        messages_filepath = session_dir / MESSAGES_FILENAME
        try:
            with open(messages_filepath, "ab") as f:
                for message in messages:
                    f.write(orjson.dumps(message) + b"\n")
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist session messages to {messages_filepath}: {e}"
            ) from e

    async def save_interaction(
        self,
        messages: Sequence[LLMMessage],
        stats: AgentStats,
        base_config: VibeConfig,
        tool_manager: ToolManager,
        agent_profile: AgentProfile,
    ) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, session_metadata = session_info
        metadata_path = session_dir / METADATA_FILENAME

        if not any(msg.role != Role.SYSTEM for msg in messages):
            return

        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Failed to create session directory at {session_dir}: {type(e).__name__}: {e}"
            ) from e

        try:
            if metadata_path.exists():
                raw = (await read_safe_async(metadata_path)).text
                old_metadata = orjson.loads(raw)
                old_total_messages = old_metadata["total_messages"]
            else:
                old_total_messages = 0
        except Exception as e:
            raise RuntimeError(
                f"Failed to read session metadata at {metadata_path}: {e}"
            ) from e

        try:
            non_system_messages = [m for m in messages if m.role != Role.SYSTEM]
            new_messages = non_system_messages[old_total_messages:]

            if len(new_messages) == 0:
                return

            messages_data = [
                m.model_dump(exclude_none=True, mode="json") for m in new_messages
            ]
            await SessionLogger.persist_messages(messages_data, session_dir)

            # If message update succeeded, write metadata
            tools_available = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_class.get_name(),
                        "description": tool_class.description,
                        "parameters": tool_class.get_parameters(),
                    },
                }
                for tool_class in tool_manager.available_tools.values()
            ]

            title = self._resolve_title(messages)
            system_prompt = (
                messages[0].model_dump()
                if len(messages) > 0 and messages[0].role == Role.SYSTEM
                else None
            )
            total_messages = len(non_system_messages)

            await self._persist_static_context(
                {
                    "tools_available": tools_available,
                    "config": base_config.model_dump(mode="json"),
                    "system_prompt": system_prompt,
                },
                session_dir,
            )

            metadata_dump = {
                **session_metadata.model_dump(),
                "end_time": utc_now().isoformat(),
                "stats": stats.model_dump(),
                "title": title,
                "total_messages": total_messages,
                "agent_profile": {
                    "name": agent_profile.name,
                    "overrides": agent_profile.overrides,
                },
            }

            await SessionLogger.persist_metadata(metadata_dump, session_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to save session to {session_dir}: {e}") from e
        finally:
            await asyncio.to_thread(self.maybe_cleanup_tmp_files)

    async def persist_loops(self) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, session_metadata = session_info
        metadata_path = session_dir / METADATA_FILENAME
        if not metadata_path.exists():
            return
        try:
            raw = (await read_safe_async(metadata_path)).text
            metadata = orjson.loads(raw)
        except (OSError, orjson.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to read session metadata at {metadata_path}: {e}"
            ) from e
        metadata["loops"] = [
            loop.model_dump(mode="json") for loop in session_metadata.loops
        ]
        await SessionLogger.persist_metadata(metadata, session_dir)

    async def persist_workflow_snapshots(self, snapshots: list[dict[str, Any]]) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, session_metadata = session_info
        metadata_path = session_dir / METADATA_FILENAME
        if not metadata_path.exists():
            return
        try:
            raw = (await read_safe_async(metadata_path)).text
            metadata = orjson.loads(raw)
        except (OSError, orjson.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to read session metadata at {metadata_path}: {e}"
            ) from e
        # Upsert by run_id rather than full-replace: on resume the runner starts
        # with an empty in-memory run list, so a plain replace would wipe
        # snapshots from a prior session that the current process never loaded.
        by_id: dict[Any, dict[str, Any]] = {
            s.get("run_id"): s
            for s in metadata.get("workflow_snapshots", [])
            if isinstance(s, dict)
        }
        for s in snapshots:
            by_id[s.get("run_id")] = s
        merged = list(by_id.values())
        session_metadata.workflow_snapshots = merged
        metadata["workflow_snapshots"] = merged
        await SessionLogger.persist_metadata(metadata, session_dir)

    def load_workflow_snapshots(self) -> list[dict[str, Any]]:
        """Return the workflow snapshots persisted for this session.

        Read-back counterpart to persist_workflow_snapshots, so a resume path
        can find a prior run's snapshot by run_id. Returns the in-memory list
        from the session metadata (kept fresh by persist_workflow_snapshots).
        """
        if self.session_metadata is None:
            return []
        return list(self.session_metadata.workflow_snapshots)

    async def persist_experiments(self, response: EvalResponse | None) -> None:
        session_info = self._get_session_info()
        if session_info is None:
            return
        session_dir, session_metadata = session_info
        session_metadata.experiments = response
        metadata_path = session_dir / METADATA_FILENAME
        if not metadata_path.exists():
            return
        try:
            raw = (await read_safe_async(metadata_path)).text
            metadata = orjson.loads(raw)
        except (OSError, orjson.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to read session metadata at {metadata_path}: {e}"
            ) from e
        metadata["experiments"] = (
            response.model_dump(mode="json") if response is not None else None
        )
        await SessionLogger.persist_metadata(metadata, session_dir)

    def reset_session(
        self, session_id: str, *, parent_session_id: str | None = None
    ) -> None:
        if not self.enabled:
            return

        self.session_id = session_id
        self.session_start_time = utc_now().isoformat()
        self.session_dir = self.save_folder
        self.session_metadata = self._initialize_session_metadata()
        self._static_context_fingerprint = None
        if parent_session_id is not None:
            self.session_metadata.parent_session_id = parent_session_id

    def resume_existing_session(self, session_id: str, session_dir: Path) -> None:
        if not self.enabled:
            return

        self.session_id = session_id
        self.session_dir = session_dir
        try:
            # Mark the dir live before the first append (which never moves dir
            # mtime), so a concurrent archiver's cutoff check sees the resume.
            os.utime(session_dir)
        except OSError:
            pass
        self.session_metadata = SessionLoader.load_metadata(session_dir)
        self._static_context_fingerprint = None

        if self.session_metadata.start_time:
            self.session_start_time = self.session_metadata.start_time

    def cleanup_tmp_files(self) -> None:
        """Delete this session's temporary files created more than 5 minutes ago.

        Scoped to the current session dir: temp files are only ever created
        beside the file they replace, and sweeping every sibling session dir
        made this a whole-tree walk on the save path. Strays in other dirs are
        each session's own to clean (and get tarred by the archiver anyway).
        """
        if not self.enabled or self.session_dir is None:
            return
        if not self.session_dir.is_dir():
            return

        now = utc_now()
        ago = now - timedelta(minutes=5)

        tmp_files = self.session_dir.glob("*.json.tmp")

        for file_path in tmp_files:
            if file_path.is_file():
                try:
                    file_mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=UTC
                    )
                    if file_mtime < ago:
                        file_path.unlink()
                except Exception:
                    continue

    def maybe_cleanup_tmp_files(self) -> None:
        if not self.enabled or not self.save_dir:
            return

        if not self._tmp_cleanup_lock.acquire(blocking=False):
            return
        try:
            now = utc_now()
            if (
                self._last_tmp_cleanup_at is not None
                and now - self._last_tmp_cleanup_at < TMP_CLEANUP_INTERVAL
            ):
                return

            self.cleanup_tmp_files()
            self._last_tmp_cleanup_at = now
        finally:
            self._tmp_cleanup_lock.release()
