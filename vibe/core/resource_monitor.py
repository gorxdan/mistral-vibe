from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass, replace
import os
import threading
import time
from typing import TYPE_CHECKING, ClassVar

from vibe.core.logger import logger

_KB = 1024
_MB = _KB * 1024
_GB = _MB * 1024

# TYPE_CHECKING split keeps psutil typed while frozen builds fall back to None.
if TYPE_CHECKING:
    import psutil
else:  # pragma: no cover - optional dep / frozen builds
    try:
        import psutil
    except ImportError:
        psutil = None

# Exception tuples hoisted so the tree walk can reference psutil's classes
# without pyright flagging member access on the optional (possibly-None) module.
# Empty tuples never match, which is correct: the walk is gated on psutil being
# present, so these clauses are unreachable when it is absent.
#
# OSError is folded into every per-proc catch on purpose: psutil re-raises a
# BARE FileNotFoundError/OSError for the documented /proc race (psutil #2418 —
# /proc/<pid>/stat exists but statm/io vanished mid-read) and for non-standard
# errnos, and psutil.Error is NOT an OSError subclass. Without this, a transient
# read race on a dying child would escape sample() and abort the user turn that
# turn() wraps. _PROC_GONE (proc genuinely died) evicts the pid; _PROC_UNREADABLE
# (alive but momentarily unreadable) retains its baseline so a later good walk
# computes a forward delta instead of re-counting the whole lifetime.
if psutil is not None:
    _CHILDREN_FAILED: tuple[type[BaseException], ...] = (psutil.Error, OSError)
    _PROC_GONE: tuple[type[BaseException], ...] = (psutil.NoSuchProcess,)
    _PROC_UNREADABLE: tuple[type[BaseException], ...] = (psutil.AccessDenied, OSError)
    _IO_UNAVAILABLE: tuple[type[BaseException], ...] = (
        psutil.AccessDenied,
        AttributeError,
        NotImplementedError,
        OSError,
    )
else:  # pragma: no cover - frozen builds without psutil
    _CHILDREN_FAILED = _PROC_GONE = _PROC_UNREADABLE = _IO_UNAVAILABLE = ()


# Per-process counters read once per tree walk. cpu_seconds and the disk byte
# counters are cumulative-since-process-start; rss is point-in-time. The disk
# fields are None when io_counters is unavailable for the proc (macOS has no
# per-proc io_counters; a child may transiently deny /proc/<pid>/io) — None
# means "unknown", distinct from a real 0, so the accumulator carries the prior
# baseline forward instead of re-baselining to 0.
@dataclass(frozen=True)
class _ProcReading:
    cpu_seconds: float
    rss_bytes: int
    disk_read_bytes: int | None
    disk_write_bytes: int | None


@dataclass(frozen=True)
class _TreeWalk:
    """One snapshot of the process tree.

    readings: pids successfully read this walk. retained: pids that are alive
    but momentarily unreadable (their accumulator baseline must be kept, not
    dropped). rss_bytes / nb_procs are over the readable procs only.
    """

    readings: dict[int, _ProcReading]
    retained: set[int]
    rss_bytes: int
    nb_procs: int


@dataclass(frozen=True)
class ResourceTotals:
    """Totals for the harness process tree.

    cpu_seconds / disk bytes are monotonic accumulations of forward deltas
    across a churning tree whose children (bash, LSP, MCP, tools) come and go,
    so they keep climbing even as the procs that earned them exit. rss_bytes /
    nb_procs are the most recent point-in-time readings.

    Caveats by design: rss_bytes is a naive sum of per-proc RSS, so pages
    shared across the tree (shared libs, COW) are counted once per proc — it is
    an upper bound on unique resident memory, not PSS. disk_available is False
    when no proc in the tree ever exposed io_counters (e.g. all of macOS), so
    callers can render "n/a" rather than a misleading 0.
    """

    cpu_seconds: float
    disk_read_bytes: int
    disk_write_bytes: int
    rss_bytes: int
    nb_procs: int
    disk_available: bool


def _human_bytes(num_bytes: int) -> str:
    # Adaptive unit so sub-MB per-turn IO and low rates don't floor to "0.0MB".
    if num_bytes < _KB:
        return f"{num_bytes}B"
    if num_bytes < _MB:
        return f"{num_bytes / _KB:.1f}KB"
    if num_bytes < _GB:
        return f"{num_bytes / _MB:.1f}MB"
    return f"{num_bytes / _GB:.2f}GB"


class _TreeAccumulator:
    """Sums forward deltas of monotonic per-process counters across a churning
    tree. Each pid's running cumulative is remembered; only forward movement is
    added, so a counter reset or pid reuse clamps to >= 0 rather than going
    negative.

    Scope of the churn-safety: a child observed by at least one walk keeps the
    work it had accrued as of its last reading. A child whose ENTIRE lifetime
    falls between two walks is never observed and contributes nothing — an
    inherent limit of sampling (the dead proc's counters are gone). Short tool
    subprocesses (ripgrep/grep) can therefore be under-counted in a per-turn
    delta; treat the disk/cpu deltas as a lower bound, not exact accounting.
    """

    def __init__(self) -> None:
        self._last: dict[int, _ProcReading] = {}
        self.cpu_seconds = 0.0
        self.disk_read_bytes = 0
        self.disk_write_bytes = 0
        self.io_seen = False

    def update(self, walk: _TreeWalk) -> None:
        for pid, cur in walk.readings.items():
            self._last[pid] = self._fold(self._last.get(pid), cur)
        # Keep baselines for every pid still in the tree (readable OR alive but
        # unreadable); only genuinely vanished pids are dropped. Retaining the
        # unreadable ones stops a later good read from re-counting their whole
        # lifetime as first-sight.
        keep = walk.readings.keys() | walk.retained
        self._last = {pid: r for pid, r in self._last.items() if pid in keep}

    def _fold(self, prev: _ProcReading | None, cur: _ProcReading) -> _ProcReading:
        if prev is None:
            self.cpu_seconds += cur.cpu_seconds
            read = self._first_io(cur.disk_read_bytes, is_read=True)
            write = self._first_io(cur.disk_write_bytes, is_read=False)
        else:
            self.cpu_seconds += max(0.0, cur.cpu_seconds - prev.cpu_seconds)
            read = self._fold_io(
                prev.disk_read_bytes, cur.disk_read_bytes, is_read=True
            )
            write = self._fold_io(
                prev.disk_write_bytes, cur.disk_write_bytes, is_read=False
            )
        return replace(cur, disk_read_bytes=read, disk_write_bytes=write)

    def _add_io(self, value: int, *, is_read: bool) -> None:
        if is_read:
            self.disk_read_bytes += value
        else:
            self.disk_write_bytes += value

    def _first_io(self, value: int | None, *, is_read: bool) -> int | None:
        if value is None:
            return None
        self.io_seen = True
        self._add_io(value, is_read=is_read)
        return value

    def _fold_io(
        self, prev: int | None, cur: int | None, *, is_read: bool
    ) -> int | None:
        if cur is None:
            # io unavailable this walk — carry the last known baseline forward
            # so recovery computes a forward delta, not a re-baseline to 0.
            return prev
        self.io_seen = True
        if prev is None:
            self._add_io(cur, is_read=is_read)
        else:
            self._add_io(max(0, cur - prev), is_read=is_read)
        return cur


class ResourceMonitor:
    """Local CPU / memory / disk-IO monitor for the harness process tree.

    Logs a periodic heartbeat (steady-state + idle leaks) and a per-turn delta
    (cost attributed to work). Output is logger-only — no network, no UI, and
    sampling can never raise into the harness. A no-op when psutil is
    unavailable or disabled (e.g. in-process subagents, which would otherwise
    double-sample the shared process tree).

    The heartbeat is process-scoped: only the first enabled monitor in a PID
    runs it, so an ACP server hosting several top-level sessions (or a forked
    loop) does not emit N duplicate heartbeat lines for one shared tree. Note
    that the tree is rooted at os.getpid(), so in such multi-session processes
    the cpu/disk/rss numbers reflect the WHOLE process, not one session.
    """

    _owner_lock: ClassVar[threading.Lock] = threading.Lock()
    _owner_pids: ClassVar[set[int]] = set()

    def __init__(
        self,
        *,
        interval_seconds: float = 30.0,
        enabled: bool = True,
        label_getter: Callable[[], str | None] | None = None,
    ) -> None:
        self._interval = interval_seconds
        self._accum = _TreeAccumulator()
        self._task: asyncio.Task[None] | None = None
        self._is_heartbeat_owner = False
        self._label_getter = label_getter
        self._last_rss = 0
        self._last_nb = 0
        self._root = (
            psutil.Process(os.getpid()) if enabled and psutil is not None else None
        )

    @property
    def available(self) -> bool:
        return self._root is not None

    def _tag(self) -> str:
        sid = self._label_getter() if self._label_getter is not None else None
        return f"[{sid[:8]}] " if sid else ""

    def _read_tree(self) -> _TreeWalk | None:
        """Walk the tree. Returns None on a hard failure (so the caller skips
        the accumulator update and keeps prior baselines) vs an empty-but-valid
        walk. The root is always self, so a truly empty readable set only ever
        means failure — never a legitimately empty tree.
        """
        root = self._root
        if root is None:
            return _TreeWalk(readings={}, retained=set(), rss_bytes=0, nb_procs=0)
        try:
            procs = [root, *root.children(recursive=True)]
        except _CHILDREN_FAILED:
            return None
        readings: dict[int, _ProcReading] = {}
        retained: set[int] = set()
        rss_total = 0
        for proc in procs:
            pid = proc.pid
            try:
                with proc.oneshot():
                    cpu = proc.cpu_times()
                    rss = proc.memory_info().rss
                    try:
                        io = proc.io_counters()
                        read_bytes: int | None = io.read_bytes
                        write_bytes: int | None = io.write_bytes
                    except _IO_UNAVAILABLE:
                        # Unknown (macOS has no per-proc io_counters; a child may
                        # deny /proc/<pid>/io) — None, NOT 0, so it isn't summed.
                        read_bytes = write_bytes = None
            except _PROC_GONE:
                # Genuinely dead — let it be pruned from the accumulator.
                continue
            except _PROC_UNREADABLE:
                # Alive but momentarily unreadable — keep its baseline.
                retained.add(pid)
                continue
            readings[pid] = _ProcReading(
                cpu_seconds=cpu.user + cpu.system,
                rss_bytes=rss,
                disk_read_bytes=read_bytes,
                disk_write_bytes=write_bytes,
            )
            rss_total += rss
        return _TreeWalk(
            readings=readings,
            retained=retained,
            rss_bytes=rss_total,
            nb_procs=len(readings),
        )

    def sample(self) -> ResourceTotals:
        # Absolute fail-soft: sampling must never propagate an exception into a
        # user turn (turn() calls this) or kill the heartbeat task. The whole
        # walk AND the accumulator fold are guarded so a malformed reading can't
        # escape either; on any error the prior totals are returned unchanged.
        try:
            walk = self._read_tree()
            if walk is not None:
                self._accum.update(walk)
                self._last_rss = walk.rss_bytes
                self._last_nb = walk.nb_procs
        except Exception:
            logger.debug("perf: sample failed", exc_info=True)
        return ResourceTotals(
            cpu_seconds=self._accum.cpu_seconds,
            disk_read_bytes=self._accum.disk_read_bytes,
            disk_write_bytes=self._accum.disk_write_bytes,
            rss_bytes=self._last_rss,
            nb_procs=self._last_nb,
            disk_available=self._accum.io_seen,
        )

    def start(self) -> None:
        if self._root is None or self._task is not None:
            return
        pid = os.getpid()
        with ResourceMonitor._owner_lock:
            if pid in ResourceMonitor._owner_pids:
                return  # another monitor in this process already owns the heartbeat
            ResourceMonitor._owner_pids.add(pid)
        try:
            self._task = asyncio.create_task(self._heartbeat_loop())
        except BaseException:
            # Release the slot if the task never started, so a sibling monitor
            # isn't permanently wedged out of running the heartbeat.
            with ResourceMonitor._owner_lock:
                ResourceMonitor._owner_pids.discard(pid)
            raise
        self._is_heartbeat_owner = True

    async def aclose(self) -> None:
        if self._is_heartbeat_owner:
            with ResourceMonitor._owner_lock:
                ResourceMonitor._owner_pids.discard(os.getpid())
            self._is_heartbeat_owner = False
        if self._task is not None:
            self._task.cancel()
            # Await unconditionally (even if already done) so a heartbeat that
            # died with an exception has it retrieved here rather than surfacing
            # as an "exception never retrieved" warning at GC.
            with contextlib.suppress(BaseException):
                await self._task
            self._task = None

    def _disk(self, num_bytes: int, *, available: bool, rate: bool = False) -> str:
        if not available:
            return "n/a"
        return _human_bytes(num_bytes) + ("/s" if rate else "")

    @contextlib.contextmanager
    def turn(self, label: str = "turn") -> Iterator[None]:
        if self._root is None:
            yield
            return
        before = self.sample()
        t0 = time.monotonic()
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                after = self.sample()
                elapsed = max(time.monotonic() - t0, 1e-6)
                cpu_delta = after.cpu_seconds - before.cpu_seconds
                read_delta = after.disk_read_bytes - before.disk_read_bytes
                write_delta = after.disk_write_bytes - before.disk_write_bytes
                # cpu% is cpu-seconds over wall-seconds: it can exceed 100% on a
                # multicore tree, and is diluted by any human-approval wait the
                # turn blocks on. The raw wall= and cpu= are logged alongside so
                # the ratio is always recoverable.
                logger.info(
                    "perf %s%s: wall=%.1fs cpu=%.2fs (%.0f%%) "
                    "disk_r=%s disk_w=%s rss=%s procs=%d",
                    self._tag(),
                    label,
                    elapsed,
                    cpu_delta,
                    100.0 * cpu_delta / elapsed,
                    self._disk(read_delta, available=after.disk_available),
                    self._disk(write_delta, available=after.disk_available),
                    _human_bytes(after.rss_bytes),
                    after.nb_procs,
                )

    async def _heartbeat_loop(self) -> None:
        prev = self.sample()
        t_prev = time.monotonic()
        while True:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
            # Guard the body so a stray error can't silently kill the heartbeat
            # for the rest of the session (sample() is already fail-soft, this
            # also covers the log/format path).
            try:
                cur = self.sample()
                now = time.monotonic()
                elapsed = max(now - t_prev, 1e-6)
                cpu_delta = cur.cpu_seconds - prev.cpu_seconds
                read_rate = int((cur.disk_read_bytes - prev.disk_read_bytes) / elapsed)
                write_rate = int(
                    (cur.disk_write_bytes - prev.disk_write_bytes) / elapsed
                )
                logger.info(
                    "perf %sheartbeat: cpu=%.0f%% rss=%s procs=%d disk_r=%s disk_w=%s",
                    self._tag(),
                    100.0 * cpu_delta / elapsed,
                    _human_bytes(cur.rss_bytes),
                    cur.nb_procs,
                    self._disk(read_rate, available=cur.disk_available, rate=True),
                    self._disk(write_rate, available=cur.disk_available, rate=True),
                )
                prev = cur
                t_prev = now
            except Exception:
                logger.debug("perf: heartbeat sample failed", exc_info=True)
