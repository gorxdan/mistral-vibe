from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from vibe.core import loop_tracer
from vibe.core.paths import LOG_DIR


@pytest.fixture(autouse=True)
def _reset_tracer() -> Iterator[None]:
    yield
    if loop_tracer._ORIG_RUN is not None:
        asyncio.Handle._run = loop_tracer._ORIG_RUN
    loop_tracer._ORIG_RUN = None
    loop_tracer._INSTALLED = False
    loop_tracer._THRESHOLD = 0.0
    loop_tracer._BLOCKERS.clear()
    for handler in list(loop_tracer._perf_log.handlers):
        loop_tracer._perf_log.removeHandler(handler)
        handler.close()


@pytest.mark.asyncio
async def test_install_creates_missing_log_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRACE_LOOP", "0.05")
    assert not LOG_DIR.path.exists()
    loop_tracer.install()
    assert loop_tracer._INSTALLED
    assert loop_tracer._perf_log.handlers
    assert LOG_DIR.path.is_dir()


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
