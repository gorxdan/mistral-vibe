"""Memory profiling harness for a long chaton session (headless).

Drives the real ``VibeApp`` (TUI widgets, agent loop, message history) over
many turns using an in-process fake backend, and reports per-turn memory growth
via ``tracemalloc``. The goal is to surface *retention* — structures that grow
unbounded with session length — for the four suspected areas:

  * transcript / message-widget retention (``#messages`` children)
  * agent-loop conversation history (``agent_loop.messages``)
  * image attachment buffers (see ``--with-images`` knob below)
  * workflow / agent event accumulation

It is gated behind ``VIBE_MEM_PROFILE`` so normal CI never pays the cost.

Run it::

    VIBE_MEM_PROFILE=1 VIBE_MEM_TURNS=300 \
        .venv/bin/python -m pytest tests/perf/test_memory_session.py -s -n0

Knobs (env vars):
    VIBE_MEM_PROFILE   must be set/truthy to run at all
    VIBE_MEM_TURNS     measured turns after warmup (default 200)
    VIBE_MEM_WARMUP    warmup turns before the baseline snapshot (default 5)
    VIBE_MEM_SNAP_EVERY  rows in the growth table cadence (default 25)
    VIBE_MEM_REPLY_CHARS assistant reply size in chars (default 800)
    VIBE_MEM_TOP       number of top allocation sites to print (default 30)
    VIBE_MEM_IMAGE_KB  if >0, attach a fake image of this size each turn
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import gc
import os
import tracemalloc

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.types import LLMChunk

_RUN = os.environ.get("VIBE_MEM_PROFILE")
_TURNS = int(os.environ.get("VIBE_MEM_TURNS", "200"))
_WARMUP = int(os.environ.get("VIBE_MEM_WARMUP", "5"))
_SNAP_EVERY = int(os.environ.get("VIBE_MEM_SNAP_EVERY", "25"))
_REPLY_CHARS = int(os.environ.get("VIBE_MEM_REPLY_CHARS", "800"))
_TOP = int(os.environ.get("VIBE_MEM_TOP", "30"))
_IMAGE_KB = int(os.environ.get("VIBE_MEM_IMAGE_KB", "0"))

_MB = 1024 * 1024


class _InfiniteFakeBackend(FakeBackend):
    """Returns a fixed-size assistant reply on every call, forever.

    Unlike ``FakeBackend`` (finite ``_streams``) this never exhausts, so a
    session of arbitrary length keeps producing real assistant messages and
    widgets to retain.
    """

    def __init__(self, reply: str) -> None:
        super().__init__()
        self._reply = reply

    def _chunk(self) -> LLMChunk:
        return mock_llm_chunk(content=self._reply)

    async def complete(self, **kwargs) -> LLMChunk:  # type: ignore[override]
        self._requests_messages.append(list(kwargs["messages"]))
        self._requests_extra_headers.append(kwargs.get("extra_headers"))
        self._requests_metadata.append(kwargs.get("metadata"))
        return self._chunk()

    async def complete_streaming(  # type: ignore[override]
        self, **kwargs
    ) -> AsyncGenerator[LLMChunk]:
        self._requests_messages.append(list(kwargs["messages"]))
        self._requests_extra_headers.append(kwargs.get("extra_headers"))
        self._requests_metadata.append(kwargs.get("metadata"))
        yield self._chunk()


def _fmt(snapshot_diff) -> str:
    lines = []
    for stat in snapshot_diff[:_TOP]:
        frame = stat.traceback[0]
        lines.append(
            f"  {stat.size_diff / 1024:9.1f} KB  "
            f"{stat.count_diff:+8d} objs  {frame.filename}:{frame.lineno}"
        )
    return "\n".join(lines)


@pytest.mark.timeout(0)
@pytest.mark.skipif(not _RUN, reason="set VIBE_MEM_PROFILE=1 to run the harness")
@pytest.mark.asyncio
async def test_memory_long_session() -> None:
    reply = "x " * (_REPLY_CHARS // 2)
    loop = build_test_agent_loop(backend=_InfiniteFakeBackend(reply))
    app = build_test_vibe_app(agent_loop=loop)

    image_arg = "@fake.png " if _IMAGE_KB > 0 else ""

    async with app.run_test() as pilot:
        for i in range(_WARMUP):
            await app._handle_user_message(f"warmup {i}")
            await pilot.pause()

        gc.collect()
        tracemalloc.start(25)
        base = tracemalloc.take_snapshot()
        base_cur, _ = tracemalloc.get_traced_memory()

        print(
            f"\n[mem] baseline after {_WARMUP} warmup turns: "
            f"{base_cur / _MB:.2f} MB traced"
        )
        print(
            f"[mem] driving {_TURNS} turns, reply={_REPLY_CHARS} chars, "
            f"image_kb={_IMAGE_KB}"
        )
        print(
            "\n turn |  traced MB |   peak MB | widgets | vheight | history | "
            "KB/turn(base)"
        )
        print("-" * 80)

        for i in range(_TURNS):
            await app._handle_user_message(f"{image_arg}turn {i}")
            # Let scheduled call_after_refresh callbacks run — notably
            # _try_prune, which is what bounds the transcript height.
            await pilot.pause()
            n = i + 1
            if n % _SNAP_EVERY == 0:
                gc.collect()
                cur, peak = tracemalloc.get_traced_memory()
                widgets = len(app._messages_area.children)
                vheight = app._messages_area.virtual_size.height
                hist = len(app.agent_loop.messages)
                per_turn = (cur - base_cur) / n / 1024
                print(
                    f" {n:5d} | {cur / _MB:9.2f} | {peak / _MB:8.2f} | "
                    f"{widgets:7d} | {vheight:7d} | {hist:7d} | {per_turn:8.1f} KB"
                )

        gc.collect()
        end = tracemalloc.take_snapshot()
        end_cur, end_peak = tracemalloc.get_traced_memory()

    diff = end.compare_to(base, "lineno")
    print(
        f"\n[mem] total growth over {_TURNS} turns: "
        f"{(end_cur - base_cur) / _MB:.2f} MB "
        f"({(end_cur - base_cur) / _TURNS / 1024:.1f} KB/turn), "
        f"peak {end_peak / _MB:.2f} MB"
    )
    print(f"\n[mem] top {_TOP} allocation sites by growth (size_diff):")
    print(_fmt(diff))

    tracemalloc.stop()
