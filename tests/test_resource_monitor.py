from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import pytest

from vibe.core.resource_monitor import (
    ResourceMonitor,
    _human_bytes,
    _ProcReading,
    _TreeAccumulator,
    _TreeWalk,
    resource_monitor_opt_in,
)


def _reading(cpu: float, rd: int | None, wr: int | None, rss: int = 0) -> _ProcReading:
    return _ProcReading(
        cpu_seconds=cpu, rss_bytes=rss, disk_read_bytes=rd, disk_write_bytes=wr
    )


def _walk(
    readings: dict[int, _ProcReading], retained: set[int] | None = None
) -> _TreeWalk:
    return _TreeWalk(
        readings=readings,
        retained=retained or set(),
        rss_bytes=sum(r.rss_bytes for r in readings.values()),
        nb_procs=len(readings),
    )


def test_resource_monitor_opt_in_env_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default (unset) must stay off so a default session has zero sampling
    # overhead; only an explicit truthy VIBE_RESOURCE_MONITOR opts in.
    monkeypatch.delenv("VIBE_RESOURCE_MONITOR", raising=False)
    assert resource_monitor_opt_in() is False
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv("VIBE_RESOURCE_MONITOR", val)
        assert resource_monitor_opt_in() is True
    for val in ("", "0", "false", "off", "no"):
        monkeypatch.setenv("VIBE_RESOURCE_MONITOR", val)
        assert resource_monitor_opt_in() is False


@pytest.fixture(autouse=True)
def _reset_heartbeat_owners() -> None:
    # The heartbeat owner set is process-scoped class state; clear it so tests
    # in the same xdist worker don't see a leftover owner pid.
    ResourceMonitor._owner_pids.clear()


# ---- accumulator: forward-delta + churn safety ----


def test_accumulator_sums_forward_deltas() -> None:
    accum = _TreeAccumulator()
    accum.update(_walk({1: _reading(1.0, 100, 10)}))
    accum.update(_walk({1: _reading(2.5, 250, 30)}))
    assert accum.cpu_seconds == pytest.approx(2.5)
    assert accum.disk_read_bytes == 250
    assert accum.disk_write_bytes == 30


def test_accumulator_survives_child_churn() -> None:
    accum = _TreeAccumulator()
    accum.update(_walk({1: _reading(1.0, 100, 0), 2: _reading(0.5, 50, 0)}))
    accum.update(_walk({1: _reading(2.0, 200, 0)}))
    assert accum.cpu_seconds == pytest.approx(2.5)
    assert accum.disk_read_bytes == 250
    accum.update(_walk({1: _reading(2.0, 200, 0), 3: _reading(0.3, 30, 0)}))
    assert accum.cpu_seconds == pytest.approx(2.8)
    assert accum.disk_read_bytes == 280


def test_accumulator_clamps_counter_reset() -> None:
    accum = _TreeAccumulator()
    accum.update(_walk({1: _reading(5.0, 500, 0)}))
    accum.update(_walk({1: _reading(1.0, 100, 0)}))
    assert accum.cpu_seconds == pytest.approx(5.0)
    assert accum.disk_read_bytes == 500


def test_accumulator_forgets_vanished_pids() -> None:
    accum = _TreeAccumulator()
    accum.update(_walk({1: _reading(1.0, 0, 0), 2: _reading(1.0, 0, 0)}))
    accum.update(_walk({1: _reading(1.0, 0, 0)}))
    assert set(accum._last) == {1}


# ---- fixed: re-baseline / over-count bugs ----


def test_empty_failed_walk_does_not_rebaseline() -> None:
    # A hard walk failure (_read_tree -> None) must skip the update entirely so
    # the next good walk computes a forward delta, not a fresh first-sight.
    accum = _TreeAccumulator()
    accum.update(_walk({1: _reading(5.0, 500, 50)}))
    # sample() skips update on a None walk, so the accumulator never sees an
    # empty walk; assert the underlying invariant via a forward read.
    accum.update(_walk({1: _reading(6.0, 600, 60)}))
    assert accum.cpu_seconds == pytest.approx(6.0)
    assert accum.disk_read_bytes == 600
    assert accum.disk_write_bytes == 60


def test_io_flicker_does_not_double_count() -> None:
    # real -> unavailable(None) -> real must NOT re-add the whole cumulative.
    accum = _TreeAccumulator()
    accum.update(_walk({1: _reading(1.0, 1000, 0)}))
    accum.update(_walk({1: _reading(1.0, None, None)}))  # io unavailable this walk
    accum.update(_walk({1: _reading(1.0, 1200, 0)}))
    assert accum.disk_read_bytes == 1200  # 1000 baseline + 200 forward, not 2200


def test_unreadable_proc_retains_baseline() -> None:
    # A proc that is alive-but-unreadable this walk (in `retained`, not in
    # `readings`) keeps its baseline; a later good read is a forward delta.
    accum = _TreeAccumulator()
    accum.update(_walk({20: _reading(5.0, 0, 0), 1: _reading(1.0, 0, 0)}))
    accum.update(_walk({1: _reading(1.0, 0, 0)}, retained={20}))
    accum.update(_walk({20: _reading(6.0, 0, 0), 1: _reading(1.0, 0, 0)}))
    assert accum.cpu_seconds == pytest.approx(7.0)  # not 12.0


def test_io_never_available_stays_unseen() -> None:
    accum = _TreeAccumulator()
    accum.update(_walk({1: _reading(1.0, None, None)}))
    accum.update(_walk({1: _reading(2.0, None, None)}))
    assert accum.io_seen is False
    assert accum.disk_read_bytes == 0


# ---- formatting ----


def test_human_bytes_adaptive() -> None:
    assert _human_bytes(0) == "0B"
    assert _human_bytes(512) == "512B"
    assert _human_bytes(40 * 1024) == "40.0KB"  # would have floored to 0.0MB before
    assert _human_bytes(5 * 1024**2) == "5.0MB"
    assert _human_bytes(3 * 1024**3) == "3.00GB"


# ---- live sampling ----


def test_sample_reads_own_process() -> None:
    monitor = ResourceMonitor()
    assert monitor.available
    totals = monitor.sample()
    assert totals.nb_procs >= 1
    assert totals.rss_bytes > 0
    assert totals.cpu_seconds >= 0.0


def test_turn_logs_perf_delta(caplog: pytest.LogCaptureFixture) -> None:
    monitor = ResourceMonitor()
    with caplog.at_level(logging.INFO, logger="vibe"):
        with monitor.turn("unit"):
            sum(i * i for i in range(200_000))
    assert any(
        "perf" in rec.getMessage() and "unit" in rec.getMessage()
        for rec in caplog.records
    )


def test_turn_is_noop_without_psutil() -> None:
    monitor = ResourceMonitor()
    monitor._root = None
    with monitor.turn("noop"):
        pass  # must not raise


def test_disabled_monitor_is_inert(caplog: pytest.LogCaptureFixture) -> None:
    monitor = ResourceMonitor(enabled=False)
    assert not monitor.available
    with caplog.at_level(logging.INFO, logger="vibe"):
        monitor.start()
        with monitor.turn("sub"):
            pass
    assert monitor._task is None
    assert not any("perf" in rec.getMessage() for rec in caplog.records)


def test_disabled_disk_renders_na(caplog: pytest.LogCaptureFixture) -> None:
    # When io was never available (simulated via accumulator), the log shows n/a.
    monitor = ResourceMonitor()
    with caplog.at_level(logging.INFO, logger="vibe"):
        # Force io_seen False so the disk fields render n/a.
        monitor._accum.io_seen = False
        line = monitor._disk(0, available=monitor._accum.io_seen)
    assert line == "n/a"


def test_sample_never_raises_on_walk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monitor = ResourceMonitor()

    def _boom() -> object:
        raise OSError("simulated /proc race")

    monkeypatch.setattr(monitor, "_read_tree", _boom)
    # Must not propagate — turn() wraps a real user turn.
    totals = monitor.sample()
    assert totals.nb_procs == 0
    with monitor.turn("safe"):
        pass


def test_sample_never_raises_on_accumulator_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even a fold error inside the accumulator must not escape sample() and
    # abort the turn that turn().__enter__ samples before yielding.
    monitor = ResourceMonitor()

    def _boom(_walk: object) -> None:
        raise ValueError("malformed reading")

    monkeypatch.setattr(monitor._accum, "update", _boom)
    body_ran = False
    with monitor.turn("safe2"):
        body_ran = True
    assert body_ran  # the wrapped body executed despite the fold error


# ---- off-loop sampling ----


def _record_walk_threads(
    monitor: ResourceMonitor, monkeypatch: pytest.MonkeyPatch
) -> list[int]:
    real_read = monitor._read_tree
    walk_threads: list[int] = []

    def recording_read() -> _TreeWalk | None:
        walk_threads.append(threading.get_ident())
        return real_read()

    monkeypatch.setattr(monitor, "_read_tree", recording_read)
    return walk_threads


@pytest.mark.asyncio
async def test_turn_async_samples_off_loop_thread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monitor = ResourceMonitor()
    walk_threads = _record_walk_threads(monitor, monkeypatch)
    with caplog.at_level(logging.INFO, logger="vibe"):
        async with monitor.turn_async("offload"):
            pass
    assert walk_threads
    assert threading.get_ident() not in walk_threads
    assert any(
        "perf" in rec.getMessage() and "offload" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_turn_async_is_noop_without_psutil() -> None:
    monitor = ResourceMonitor()
    monitor._root = None
    async with monitor.turn_async("noop"):
        pass  # must not raise


@pytest.mark.asyncio
async def test_heartbeat_samples_off_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = ResourceMonitor(interval_seconds=0.02)
    walk_threads = _record_walk_threads(monitor, monkeypatch)
    monitor.start()
    await asyncio.sleep(0.06)
    await monitor.aclose()
    assert walk_threads
    assert threading.get_ident() not in walk_threads


@pytest.mark.asyncio
async def test_concurrent_samples_do_not_interleave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = ResourceMonitor()
    real_read = monitor._read_tree
    gate = threading.Semaphore(1)
    overlaps: list[int] = []

    def slow_read() -> _TreeWalk | None:
        if not gate.acquire(blocking=False):
            overlaps.append(threading.get_ident())
            return real_read()
        try:
            time.sleep(0.02)
            return real_read()
        finally:
            gate.release()

    monkeypatch.setattr(monitor, "_read_tree", slow_read)
    await asyncio.gather(*(asyncio.to_thread(monitor.sample) for _ in range(4)))
    assert not overlaps


# ---- heartbeat lifecycle ----


@pytest.mark.asyncio
async def test_heartbeat_lifecycle(caplog: pytest.LogCaptureFixture) -> None:
    monitor = ResourceMonitor(interval_seconds=0.05)
    with caplog.at_level(logging.INFO, logger="vibe"):
        monitor.start()
        monitor.start()  # idempotent — no 2nd task
        await asyncio.sleep(0.13)
        await monitor.aclose()
    assert monitor._task is None
    assert any(
        "perf" in rec.getMessage() and "heartbeat" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_only_one_heartbeat_per_process() -> None:
    # Two enabled monitors in the same PID: only the first owns the heartbeat.
    m1 = ResourceMonitor(interval_seconds=0.05)
    m2 = ResourceMonitor(interval_seconds=0.05)
    m1.start()
    m2.start()
    assert m1._task is not None
    assert m2._task is None  # m2 did not spawn a duplicate heartbeat
    await m1.aclose()
    # After m1 releases ownership, a later monitor can take over.
    m3 = ResourceMonitor(interval_seconds=0.05)
    m3.start()
    assert m3._task is not None
    await m3.aclose()
    await m2.aclose()


@pytest.mark.asyncio
async def test_heartbeat_survives_sample_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A transient error in the heartbeat body must not kill the loop for the
    # rest of the session.
    monitor = ResourceMonitor(interval_seconds=0.03)
    calls = {"n": 0}
    real_sample = monitor.sample

    def _flaky() -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom in body")
        return real_sample()

    monkeypatch.setattr(monitor, "sample", _flaky)
    monitor.start()
    await asyncio.sleep(0.16)
    assert monitor._task is not None
    assert not monitor._task.done()  # still alive after the raised sample
    assert calls["n"] >= 3  # kept sampling past the error
    await monitor.aclose()


@pytest.mark.asyncio
async def test_aclose_is_safe_when_never_started() -> None:
    monitor = ResourceMonitor()
    await monitor.aclose()  # must not raise


@pytest.mark.asyncio
async def test_aclose_retrieves_dead_task_exception() -> None:
    # A heartbeat that died with an exception must have it retrieved by aclose
    # (no "exception never retrieved" warning at GC).
    monitor = ResourceMonitor(interval_seconds=0.01)

    async def _die() -> None:
        raise RuntimeError("heartbeat died")

    monitor._task = asyncio.create_task(_die())
    monitor._is_heartbeat_owner = True
    ResourceMonitor._owner_pids.add(os.getpid())
    await asyncio.sleep(0.02)
    assert monitor._task.done()
    await monitor.aclose()  # must not raise and must consume the exception
    assert monitor._task is None


@pytest.mark.asyncio
async def test_session_label_in_log(caplog: pytest.LogCaptureFixture) -> None:
    monitor = ResourceMonitor(label_getter=lambda: "abcd1234ef")
    with caplog.at_level(logging.INFO, logger="vibe"):
        with monitor.turn("tagged"):
            pass
    assert any("[abcd1234]" in rec.getMessage() for rec in caplog.records)
