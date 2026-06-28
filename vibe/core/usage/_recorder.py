from __future__ import annotations

import os
from pathlib import Path
import threading
import time

from pydantic import ValidationError

from vibe.core.logger import logger
from vibe.core.paths import VIBE_HOME
from vibe.core.usage.models import UsageRecord
from vibe.core.utils.io import read_safe

# Keep this many days of history. Older records are pruned opportunistically so
# the file stays bounded without a background sweeper.
_RETENTION_DAYS = 30
# Rewrite (compact) the file once it crosses this size, dropping records older
# than the retention window and any unparseable tail lines.
_TRIM_BYTES = 2 * 1024 * 1024
# Trim check cadence: avoid stat()ing the file on every write.
_TRIM_CHECK_EVERY = 256


class UsageRecorder:
    """Append-only JSONL sink for ``UsageRecord``s at ``~/.vibe/usage.jsonl``.

    Writes are a single ``O_APPEND`` line under 4 KB, which is atomic on POSIX,
    so concurrent Vibe processes never interleave or corrupt a record line. The
    read path tolerates a partially-written final line (skips unparseable
    lines). No file lock: contention is rare and each append is atomic.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else VIBE_HOME.path / "usage.jsonl"
        self._lock = threading.Lock()
        self._writes_since_trim = 0

    @property
    def path(self) -> Path:
        return self._path

    def record(self, record: UsageRecord) -> None:
        line = record.model_dump_json() + "\n"
        data = line.encode("utf-8")
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # "a" + write of one short line is the atomic-append pattern.
                fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
            except OSError as e:
                logger.error("Failed to append usage record: %s", e)
                return
            self._writes_since_trim += 1
            if self._writes_since_trim >= _TRIM_CHECK_EVERY:
                self._writes_since_trim = 0
                self._maybe_trim_locked()

    def read_all(self) -> list[UsageRecord]:
        """Read every parseable record, newest-last. Tolerates a torn final line."""
        if not self._path.exists():
            return []
        result = read_safe(self._path).text
        records: list[UsageRecord] = []
        for line in result.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(UsageRecord.model_validate_json(line))
            except ValidationError:
                continue
        return records

    def _maybe_trim_locked(self) -> None:
        try:
            if self._path.stat().st_size < _TRIM_BYTES:
                return
        except OSError:
            return
        cutoff = time.time() - _RETENTION_DAYS * 86400
        kept = [r for r in self.read_all() if r.timestamp >= cutoff]
        tmp = self._path.with_suffix(".jsonl.tmp")
        try:
            tmp.write_bytes(
                ("".join(r.model_dump_json() + "\n" for r in kept)).encode("utf-8")
            )
            os.replace(tmp, self._path)
        except OSError as e:
            logger.error("Failed to compact usage log: %s", e)


_default_recorder: UsageRecorder | None = None


def get_usage_recorder() -> UsageRecorder:
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = UsageRecorder()
    return _default_recorder


def reset_usage_recorder_for_tests(recorder: UsageRecorder | None = None) -> None:
    global _default_recorder
    _default_recorder = recorder
