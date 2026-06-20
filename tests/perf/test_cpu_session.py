"""CPU profiling harness for a chaton session (headless).

Drives the real ``VibeApp`` over many turns with an in-process fake backend and
profiles the per-turn work (event handling, Textual render/dispatch, tool
dispatch) — i.e. the CPU chaton spends around the model call, excluding network
and model time. Two profilers:

  * pyinstrument (default) — statistical wall-clock sampler; writes a flamegraph
    HTML and prints a self-time tree. Best for "where does the time go".
  * cProfile (VIBE_CPU_TOOL=cprofile) — deterministic call counts/tottime. Best
    for "what is called millions of times" (e.g. dispatch churn).

Gated behind ``VIBE_CPU_PROFILE`` so normal CI never pays the cost.

Run::

    VIBE_CPU_PROFILE=1 VIBE_CPU_TURNS=80 \
        .venv/bin/python -m pytest tests/perf/test_cpu_session.py -s -n0

Knobs (env vars):
    VIBE_CPU_PROFILE     must be set to run at all
    VIBE_CPU_TURNS       measured turns after warmup (default 60)
    VIBE_CPU_WARMUP      warmup turns before profiling (default 3)
    VIBE_CPU_REPLY_CHARS assistant reply size in chars (default 800)
    VIBE_CPU_TOOL_CALLS  if set, drive one bash-echo tool round-trip per turn
    VIBE_CPU_TOOL        "pyinstrument" (default) or "cprofile"
    VIBE_CPU_TOP         rows to print (default 30)
    VIBE_CPU_PRUNE_LOW/HIGH  override transcript prune marks
"""

from __future__ import annotations

import os
import time

import pytest

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_app,
    build_test_vibe_config,
)
from tests.perf.test_memory_session import _InfiniteFakeBackend, _ToolCallFakeBackend

_RUN = os.environ.get("VIBE_CPU_PROFILE")
_TURNS = int(os.environ.get("VIBE_CPU_TURNS", "60"))
_WARMUP = int(os.environ.get("VIBE_CPU_WARMUP", "3"))
_REPLY_CHARS = int(os.environ.get("VIBE_CPU_REPLY_CHARS", "800"))
_TOOL_CALLS = os.environ.get("VIBE_CPU_TOOL_CALLS")
_TOOL = os.environ.get("VIBE_CPU_TOOL", "pyinstrument").lower()
_TOP = int(os.environ.get("VIBE_CPU_TOP", "30"))
_PRUNE_LOW = os.environ.get("VIBE_CPU_PRUNE_LOW")
_PRUNE_HIGH = os.environ.get("VIBE_CPU_PRUNE_HIGH")


async def _drive(app, pilot, turns: int) -> None:
    image = ""
    for i in range(turns):
        await app._handle_user_message(f"{image}turn {i}")
        await pilot.pause()


def _print_cprofile(pr, elapsed: float, turns: int) -> None:
    import io
    import pstats

    print(
        f"\n[cpu] cProfile: {elapsed:.2f}s wall over {turns} turns "
        f"({elapsed / turns * 1000:.1f} ms/turn)"
    )
    for sort_key in ("tottime", "cumulative"):
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).strip_dirs().sort_stats(sort_key).print_stats(_TOP)
        print(f"\n[cpu] top {_TOP} by {sort_key}:\n{buf.getvalue()}")


@pytest.mark.timeout(0)
@pytest.mark.skipif(not _RUN, reason="set VIBE_CPU_PROFILE=1 to run the harness")
@pytest.mark.asyncio
async def test_cpu_session() -> None:
    import vibe.cli.textual_ui.app as _app_mod

    if _PRUNE_LOW:
        _app_mod.PRUNE_LOW_MARK = int(_PRUNE_LOW)
    if _PRUNE_HIGH:
        _app_mod.PRUNE_HIGH_MARK = int(_PRUNE_HIGH)

    reply = "x " * (_REPLY_CHARS // 2)
    if _TOOL_CALLS:
        cfg = build_test_vibe_config(bypass_tool_permissions=True)
        loop = build_test_agent_loop(
            config=cfg, backend=_ToolCallFakeBackend(reply, _REPLY_CHARS)
        )
    else:
        loop = build_test_agent_loop(backend=_InfiniteFakeBackend(reply))
    app = build_test_vibe_app(agent_loop=loop)

    print(
        f"\n[cpu] tool={_TOOL} turns={_TURNS} reply={_REPLY_CHARS} "
        f"tool_calls={bool(_TOOL_CALLS)} "
        f"prune={_app_mod.PRUNE_LOW_MARK}/{_app_mod.PRUNE_HIGH_MARK}"
    )

    async with app.run_test() as pilot:
        await _drive(app, pilot, _WARMUP)

        if _TOOL == "cprofile":
            import cProfile

            pr = cProfile.Profile()
            start = time.perf_counter()
            pr.enable()
            await _drive(app, pilot, _TURNS)
            pr.disable()
            _print_cprofile(pr, time.perf_counter() - start, _TURNS)
            return

        from pyinstrument import Profiler

        profiler = Profiler()
        start = time.perf_counter()
        profiler.start()
        await _drive(app, pilot, _TURNS)
        profiler.stop()
        elapsed = time.perf_counter() - start

    out = "/tmp/cpu-session.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(profiler.output_html())
    print(
        f"\n[cpu] pyinstrument: {elapsed:.2f}s wall over {_TURNS} turns "
        f"({elapsed / _TURNS * 1000:.1f} ms/turn); flamegraph -> {out}"
    )
    print(profiler.output_text(unicode=True, color=False, show_all=False))
