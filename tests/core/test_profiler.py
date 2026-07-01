from __future__ import annotations

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
