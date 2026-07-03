from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
import time

import pytest

from vibe.core import profiler
from vibe.core.paths import LOG_DIR


@pytest.fixture(autouse=True)
def _reset_profiler() -> Iterator[None]:
    yield
    if profiler._state.profiler is not None:
        profiler._state.profiler.stop()
    profiler._state.profiler = None
    profiler._state.label = "default"


def _burn(seconds: float = 0.02) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def test_section_writes_reports_under_log_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_working_directory: Path
) -> None:
    pytest.importorskip("pyinstrument")
    monkeypatch.setenv("VIBE_PROFILE", "1")
    with profiler.section("turn-abc-0"):
        _burn()
    assert not list(tmp_working_directory.glob("*-profile.*"))
    assert (LOG_DIR.path / "turn-abc-0-profile.html").exists()
    assert (LOG_DIR.path / "turn-abc-0-profile.txt").exists()


def test_nested_section_keeps_outer_profile_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyinstrument")
    monkeypatch.setenv("VIBE_PROFILE", "1")
    with profiler.section("outer"):
        with pytest.warns(UserWarning), profiler.section("inner"):
            _burn()
        assert profiler._state.profiler is not None
        _burn()
    assert profiler._state.profiler is None
    assert (LOG_DIR.path / "outer-profile.html").exists()
    assert not (LOG_DIR.path / "inner-profile.html").exists()


def test_start_survives_stale_async_context(
    monkeypatch: pytest.MonkeyPatch, tmp_working_directory: Path
) -> None:
    pytest.importorskip("pyinstrument")
    monkeypatch.setenv("VIBE_PROFILE", "1")

    async def scenario() -> None:
        host_stopped = asyncio.Event()

        async def child() -> None:
            await host_stopped.wait()
            assert profiler._state.profiler is None
            with profiler.section("turn-child-0", turn=0):
                _burn()

        with profiler.section("turn-host-0", turn=0):
            # child snapshots the active-profiler context; parent then stops
            task = asyncio.create_task(child())
            await asyncio.sleep(0)
        host_stopped.set()
        await task

    asyncio.run(scenario())

    assert profiler._state.profiler is None
    assert (LOG_DIR.path / "turn-child-0-profile.html").exists()


def test_interval_env_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBE_PROFILE_INTERVAL", raising=False)
    assert profiler.interval() is None
    monkeypatch.setenv("VIBE_PROFILE_INTERVAL", "0.005")
    assert profiler.interval() == 0.005
    monkeypatch.setenv("VIBE_PROFILE_INTERVAL", "junk")
    assert profiler.interval() is None
    monkeypatch.setenv("VIBE_PROFILE_INTERVAL", "0")
    assert profiler.interval() == 0.0001  # clamped floor, never a 0s busy-loop


def test_stop_and_print_fail_soft_when_log_dir_unwritable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyinstrument")
    monkeypatch.setenv("VIBE_PROFILE", "1")
    LOG_DIR.path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.path.write_text("not a dir", encoding="utf-8")

    assert profiler.start("doomed")
    profiler.stop_and_print()

    assert profiler._state.profiler is None
    assert profiler._state.label == "default"
