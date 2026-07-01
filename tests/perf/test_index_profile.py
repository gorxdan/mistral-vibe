"""CPU profiling harness for file/folder indexing (scandir + BFS walk).

Profiles the real ``FileIndexStore.rebuild(root, should_cancel)`` — the
recursive ``os.scandir`` walk plus per-entry ignore-rule / ascii-mask work that
``FileIndexer.get_index(root)`` drives on a background thread. This is the hot
path that was optimized historically (see CHANGELOG) but had no profiler.

Two profilers:

  * cProfile (default) — deterministic call counts/tottime. Best for the
    scandir/BFS churn this harness targets: shows exactly how many times
    ``scandir`` / ``_walk_directory`` / ``_create_entry`` fire per rebuild.
  * pyinstrument (VIBE_INDEX_TOOL=pyinstrument) — statistical wall-clock
    sampler; writes a flamegraph HTML and prints a self-time tree.

We warm up once via ``FileIndexer.get_index(root)`` (mirrors the real driver,
which blocks until the background rebuild completes), then profile
``VIBE_INDEX_RUNS`` direct ``FileIndexStore.rebuild`` calls. We call the store
directly in the loop because ``get_index`` caches by root and would not re-walk
the tree — ``rebuild`` always re-scans, so the profile reflects scandir/BFS.

Gated behind ``VIBE_INDEX_PROFILE`` so normal CI never pays the cost.

Run::

    VIBE_INDEX_PROFILE=1 VIBE_INDEX_RUNS=3 VIBE_INDEX_TOOL=cprofile \
        .venv/bin/python -m pytest tests/perf/test_index_profile.py -s -n0

Knobs (env vars):
    VIBE_INDEX_PROFILE   must be set to run at all
    VIBE_INDEX_ROOT      directory to index (default repo root)
    VIBE_INDEX_RUNS      rebuilds to profile after warmup (default 3)
    VIBE_INDEX_TOOL      "cprofile" (default) or "pyinstrument"
    VIBE_INDEX_TOP       rows to print (default 30)
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from vibe.core.autocompletion.file_indexer.ignore_rules import IgnoreRules
from vibe.core.autocompletion.file_indexer.indexer import FileIndexer
from vibe.core.autocompletion.file_indexer.store import FileIndexStats, FileIndexStore

_RUN = os.environ.get("VIBE_INDEX_PROFILE")
_ROOT = os.environ.get("VIBE_INDEX_ROOT")
_RUNS = int(os.environ.get("VIBE_INDEX_RUNS", "3"))
_TOOL = os.environ.get("VIBE_INDEX_TOOL", "cprofile").lower()
_TOP = int(os.environ.get("VIBE_INDEX_TOP", "30"))


def _resolve_root() -> Path:
    if _ROOT:
        return Path(_ROOT).resolve()
    # The test conftest chdir's into a temp dir (autouse), so cwd is useless as
    # a default. Derive the repo root from this file: tests/perf/<this>.
    return Path(__file__).resolve().parents[2]


def _rebuild_n(store: FileIndexStore, root: Path, runs: int) -> int:
    entries = 0
    for _ in range(runs):
        store.clear()  # force a real re-walk; never serve a cached snapshot
        store.rebuild(root)
        entries = len(store.snapshot())
    return entries


def _print_cprofile(pr, elapsed: float, runs: int, entries: int) -> None:
    import io
    import pstats

    print(
        f"\n[index] cProfile: {elapsed:.2f}s wall over {runs} rebuilds "
        f"({elapsed / runs * 1000:.1f} ms/run); entries indexed={entries}"
    )
    for sort_key in ("tottime", "cumulative"):
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(sort_key).print_stats(_TOP)
        print(f"\n[index] top {_TOP} by {sort_key}:\n{buf.getvalue()}")


@pytest.mark.timeout(0)
@pytest.mark.skipif(not _RUN, reason="set VIBE_INDEX_PROFILE=1 to run the harness")
def test_index_profile() -> None:
    root = _resolve_root()

    # Warm up via the real driver: get_index blocks until the background
    # rebuild completes, exercising the same path Mistral Vibe uses.
    indexer = FileIndexer()
    try:
        warm = indexer.get_index(root)
    finally:
        indexer.shutdown()

    # Profile direct store rebuilds: get_index caches by root, so we drive the
    # lowest-level call that always re-walks the tree (scandir + BFS).
    stats = FileIndexStats()
    store = FileIndexStore(IgnoreRules(), stats)

    print(f"\n[index] tool={_TOOL} runs={_RUNS} root={root} warmup_entries={len(warm)}")

    if _TOOL == "cprofile":
        import cProfile

        pr = cProfile.Profile()
        start = time.perf_counter()
        pr.enable()
        entries = _rebuild_n(store, root, _RUNS)
        pr.disable()
        _print_cprofile(pr, time.perf_counter() - start, _RUNS, entries)
        return

    from pyinstrument import Profiler

    profiler = Profiler()
    start = time.perf_counter()
    profiler.start()
    entries = _rebuild_n(store, root, _RUNS)
    profiler.stop()
    elapsed = time.perf_counter() - start

    out = "/tmp/index-profile.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(profiler.output_html())
    print(
        f"\n[index] pyinstrument: {elapsed:.2f}s wall over {_RUNS} rebuilds "
        f"({elapsed / _RUNS * 1000:.1f} ms/run); entries indexed={entries}; "
        f"flamegraph -> {out}"
    )
    print(profiler.output_text(unicode=True, color=False, show_all=False))
