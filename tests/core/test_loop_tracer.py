from __future__ import annotations

import asyncio
from collections.abc import Iterator
import os
import time

import pytest

from vibe.core import loop_tracer, perf_log
from vibe.core.paths import LOG_DIR


def _reset_state() -> None:
    if loop_tracer._ORIG_RUN is not None:
        asyncio.Handle._run = loop_tracer._ORIG_RUN
    loop_tracer._ORIG_RUN = None
    loop_tracer._INSTALLED = False
    loop_tracer._THRESHOLD = 0.0
    loop_tracer._BLOCKERS.clear()
    for handler in list(loop_tracer._perf_log.handlers):
        loop_tracer._perf_log.removeHandler(handler)
        handler.close()
    if perf_log._HANDLER is not None:
        perf_log._HANDLER.close()
    perf_log._HANDLER = None


@pytest.fixture(autouse=True)
def _reset_tracer() -> Iterator[None]:
    _reset_state()
    yield
    _reset_state()


@pytest.mark.asyncio
async def test_install_creates_missing_log_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRACE_LOOP", "0.05")
    assert not LOG_DIR.path.exists()
    loop_tracer.install()
    assert loop_tracer._INSTALLED
    assert loop_tracer._perf_log.handlers
    assert LOG_DIR.path.is_dir()


@pytest.mark.asyncio
async def test_install_gcs_stale_perf_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRACE_LOOP", "0.05")
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
    loop_tracer.install()
    assert not stale.exists()
    assert not stale_backup.exists()
    assert fresh.exists()
    assert other.exists()


@pytest.mark.asyncio
async def test_perf_log_name_has_timestamp_and_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_TRACE_LOOP", "0.05")
    loop_tracer.install()
    assert list(LOG_DIR.path.glob(f"vibe-perf-*-{os.getpid()}.log"))


@pytest.mark.asyncio
async def test_install_fail_soft_when_handler_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_TRACE_LOOP", "0.05")
    LOG_DIR.path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.path.write_text("not a dir", encoding="utf-8")
    orig_run = asyncio.Handle._run
    loop_tracer.install()
    assert not loop_tracer._INSTALLED
    assert asyncio.Handle._run is orig_run
    assert not loop_tracer._perf_log.handlers
