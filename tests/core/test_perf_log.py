from __future__ import annotations

from collections.abc import Iterator
import os
import time

import pytest

from vibe.core import perf_log
from vibe.core.paths import LOG_DIR


@pytest.fixture(autouse=True)
def _reset_perf_log() -> Iterator[None]:
    yield
    if perf_log._HANDLER is not None:
        perf_log._HANDLER.close()
    perf_log._HANDLER = None


def test_perf_handler_creates_missing_log_dir() -> None:
    assert not LOG_DIR.path.exists()
    handler = perf_log.perf_handler()
    assert handler is not None
    assert LOG_DIR.path.is_dir()
    assert list(LOG_DIR.path.glob(f"vibe-perf-*-{os.getpid()}.log"))


def test_perf_handler_is_cached_per_process() -> None:
    first = perf_log.perf_handler()
    second = perf_log.perf_handler()
    assert first is not None
    assert first is second


def test_perf_handler_fail_soft_when_log_dir_is_a_file() -> None:
    LOG_DIR.path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.path.write_text("not a dir", encoding="utf-8")
    assert perf_log.perf_handler() is None
    assert perf_log._HANDLER is None


def test_perf_handler_retries_after_failure() -> None:
    LOG_DIR.path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.path.write_text("not a dir", encoding="utf-8")
    assert perf_log.perf_handler() is None
    LOG_DIR.path.unlink()
    assert perf_log.perf_handler() is not None


def test_perf_handler_gcs_stale_perf_logs() -> None:
    log_dir = LOG_DIR.path
    log_dir.mkdir(parents=True, exist_ok=True)
    stale_mtime = time.time() - 8 * 86400
    stale = log_dir / "vibe-perf-12345.log"
    stale_backup = log_dir / "vibe-perf-12345.log.1"
    other = log_dir / "vibe.log"
    for f in (stale, stale_backup, other):
        f.write_text("x", encoding="utf-8")
        os.utime(f, (stale_mtime, stale_mtime))
    fresh = log_dir / "vibe-perf-99999.log"
    fresh.write_text("x", encoding="utf-8")
    perf_log.perf_handler()
    assert not stale.exists()
    assert not stale_backup.exists()
    assert fresh.exists()
    assert other.exists()
