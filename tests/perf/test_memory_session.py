"""Memory profiling harness for a long Mistral Vibe session (headless).

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
    VIBE_MEM_CPU       if truthy, also cProfile the measured turn loop (Textual
                       render/dispatch path), dump stats to /tmp/mem-render.pstats
                       and print tottime/cumulative tables; default off so the
                       default path is unchanged
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import gc
import json
import os
import tracemalloc

import pytest

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.llm.types import CompletionRequest
from vibe.core.types import FunctionCall, LLMChunk, Role, ToolCall

_RUN = os.environ.get("VIBE_MEM_PROFILE")
_TURNS = int(os.environ.get("VIBE_MEM_TURNS", "200"))
_WARMUP = int(os.environ.get("VIBE_MEM_WARMUP", "5"))
_SNAP_EVERY = int(os.environ.get("VIBE_MEM_SNAP_EVERY", "25"))
_REPLY_CHARS = int(os.environ.get("VIBE_MEM_REPLY_CHARS", "800"))
_TOP = int(os.environ.get("VIBE_MEM_TOP", "30"))
_IMAGE_KB = int(os.environ.get("VIBE_MEM_IMAGE_KB", "0"))
_CPU = os.environ.get("VIBE_MEM_CPU", "0") not in ("", "0", "false", "False")
_PRUNE_LOW = os.environ.get("VIBE_MEM_PRUNE_LOW")
_PRUNE_HIGH = os.environ.get("VIBE_MEM_PRUNE_HIGH")
_TOOL_CALLS = os.environ.get("VIBE_MEM_TOOL_CALLS")
_ECHO_CHARS = int(os.environ.get("VIBE_MEM_ECHO_CHARS", str(_REPLY_CHARS)))

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

    async def complete(  # type: ignore[override]
        self, request: CompletionRequest, *, response_headers_sink=None
    ) -> LLMChunk:
        self._requests_messages.append(list(request.messages))
        self._requests_extra_headers.append(request.extra_headers)
        self._requests_metadata.append(request.metadata)
        return self._chunk()

    async def complete_streaming(  # type: ignore[override]
        self, request: CompletionRequest, *, response_headers_sink=None
    ) -> AsyncGenerator[LLMChunk]:
        self._requests_messages.append(list(request.messages))
        self._requests_extra_headers.append(request.extra_headers)
        self._requests_metadata.append(request.metadata)
        yield self._chunk()


class _ToolCallFakeBackend(_InfiniteFakeBackend):
    """One bash-echo tool round-trip per turn, then a final text reply.

    Emits a tool call unless the last message is a tool result (the round-trip
    came back), in which case it returns the final assistant reply. Exercises
    the heavier tool-output widget path (BashOutputMessage) vs plain replies.
    Requires bypass_tool_permissions so execution does not block on approval.
    """

    def __init__(self, reply: str, echo_chars: int) -> None:
        super().__init__(reply)
        self._echo = "y" * echo_chars

    def _tool_chunk(self) -> LLMChunk:
        return mock_llm_chunk(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc-0",
                    function=FunctionCall(
                        name="bash",
                        arguments=json.dumps({"command": f"echo {self._echo}"}),
                    ),
                )
            ],
        )

    def _next(self, messages) -> LLMChunk:
        last_role = messages[-1].role if messages else None
        return self._chunk() if last_role == Role.TOOL else self._tool_chunk()

    async def complete(  # type: ignore[override]
        self, request: CompletionRequest, *, response_headers_sink=None
    ) -> LLMChunk:
        self._requests_messages.append(list(request.messages))
        self._requests_extra_headers.append(request.extra_headers)
        self._requests_metadata.append(request.metadata)
        return self._next(request.messages)

    async def complete_streaming(  # type: ignore[override]
        self, request: CompletionRequest, *, response_headers_sink=None
    ) -> AsyncGenerator[LLMChunk]:
        self._requests_messages.append(list(request.messages))
        self._requests_extra_headers.append(request.extra_headers)
        self._requests_metadata.append(request.metadata)
        yield self._next(request.messages)


class _FanOutFakeBackend(_InfiniteFakeBackend):
    """N tool calls in ONE assistant message per turn, then a final reply.

    Models the parallel tool fan-out the agent loop dispatches via
    ``_run_tools_concurrently``: each turn emits ``n`` tool calls in a single
    assistant message, and once the tool results come back returns the final
    text reply. The batch mixes read-only and writer tools so BOTH code paths
    in ``_run_tools_concurrently`` are exercised: ``bash`` has no ``read_only``
    override (base default ``False``) so each bash echo is a *writer* run in the
    sequential writer chain, while ``grep`` is ``read_only`` so it runs in the
    concurrent reader pool. The first call of every batch is a ``grep`` to
    guarantee at least one reader; the rest are bash echoes.

    Requires bypass_tool_permissions so execution does not block on approval.
    """

    def __init__(self, reply: str, echo_chars: int, fan_out: int) -> None:
        super().__init__(reply)
        self._echo = "y" * echo_chars
        self._fan_out = max(1, fan_out)

    def _one_call(self, idx: int) -> ToolCall:
        # idx 0 -> read-only grep (concurrent reader pool); rest -> bash echo
        # writers (sequential writer chain). Unique id per call in the batch.
        if idx == 0:
            return ToolCall(
                id=f"tc-{idx}",
                function=FunctionCall(
                    name="grep", arguments=json.dumps({"pattern": "def ", "path": "."})
                ),
            )
        return ToolCall(
            id=f"tc-{idx}",
            function=FunctionCall(
                name="bash", arguments=json.dumps({"command": f"echo {self._echo}"})
            ),
        )

    def _fan_out_chunk(self) -> LLMChunk:
        return mock_llm_chunk(
            content="", tool_calls=[self._one_call(i) for i in range(self._fan_out)]
        )

    def _next(self, messages) -> LLMChunk:
        last_role = messages[-1].role if messages else None
        return self._chunk() if last_role == Role.TOOL else self._fan_out_chunk()

    async def complete(  # type: ignore[override]
        self, request: CompletionRequest, *, response_headers_sink=None
    ) -> LLMChunk:
        self._requests_messages.append(list(request.messages))
        self._requests_extra_headers.append(request.extra_headers)
        self._requests_metadata.append(request.metadata)
        return self._next(request.messages)

    async def complete_streaming(  # type: ignore[override]
        self, request: CompletionRequest, *, response_headers_sink=None
    ) -> AsyncGenerator[LLMChunk]:
        self._requests_messages.append(list(request.messages))
        self._requests_extra_headers.append(request.extra_headers)
        self._requests_metadata.append(request.metadata)
        yield self._next(request.messages)


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
    # Optionally override the transcript prune marks (the memory/scrollback
    # knob) to measure their effect. _try_prune reads these module globals at
    # call time, so patching them here takes effect for the whole run.
    import vibe.cli.textual_ui.app as _app_mod

    if _PRUNE_LOW:
        _app_mod.PRUNE_LOW_MARK = int(_PRUNE_LOW)
    if _PRUNE_HIGH:
        _app_mod.PRUNE_HIGH_MARK = int(_PRUNE_HIGH)

    reply = "x " * (_REPLY_CHARS // 2)
    if _TOOL_CALLS:
        cfg = build_test_vibe_config(bypass_tool_permissions=True)
        loop = build_test_agent_loop(
            config=cfg, backend=_ToolCallFakeBackend(reply, _ECHO_CHARS)
        )
    else:
        loop = build_test_agent_loop(backend=_InfiniteFakeBackend(reply))
    app = build_test_vibe_app(agent_loop=loop)

    print(
        f"\n[mem] prune marks: low={_app_mod.PRUNE_LOW_MARK} "
        f"high={_app_mod.PRUNE_HIGH_MARK}  tool_calls={bool(_TOOL_CALLS)}"
    )
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

        # When VIBE_MEM_CPU is set, CPU-profile the measured turn loop with
        # pyinstrument so the Textual render/diff/message-pump cost is captured
        # as the transcript (widget count) grows. Imported locally so the
        # default path never depends on pyinstrument.
        _pr = None
        if _CPU:
            # cProfile, not pyinstrument: the sampler stops Textual ever going
            # idle, deadlocking pilot.pause(). Runs under tracemalloc (opt-in).
            import cProfile

            _pr = cProfile.Profile()
            _pr.enable()

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

        if _pr is not None:
            import io
            import pstats

            _pr.disable()
            _pr.dump_stats("/tmp/mem-render.pstats")
            print(
                "\n[mem] CPU stats written to /tmp/mem-render.pstats "
                "(open with snakeviz/pstats)"
            )
            print("[mem] render/dispatch profile over the measured turns:")
            for _sort in ("tottime", "cumulative"):
                _buf = io.StringIO()
                pstats.Stats(_pr, stream=_buf).strip_dirs().sort_stats(
                    _sort
                ).print_stats(_TOP)
                print(f"\n[mem] top {_TOP} by {_sort}:\n{_buf.getvalue()}")

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
