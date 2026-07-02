# Streaming-render CPU vs reply length (VIBE_STREAM_PROFILE=1 ... -s -n0).
# Knobs: VIBE_STREAM_SIZES/_DELTA/_CPROFILE/_TOP; one coalesced flush per delta.

from __future__ import annotations

import os
import time

import pytest

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.widgets.messages import AssistantMessage

_RUN = os.environ.get("VIBE_STREAM_PROFILE")
_SIZES_KB = [int(s) for s in os.environ.get("VIBE_STREAM_SIZES", "12,24,48").split(",")]
_DELTA_CHARS = int(os.environ.get("VIBE_STREAM_DELTA", "120"))
_CPROFILE = os.environ.get("VIBE_STREAM_CPROFILE", "0") not in ("", "0", "false")
_TOP = int(os.environ.get("VIBE_STREAM_TOP", "25"))

_WORDS = "the quick brown fox jumps over the lazy dog and runs far away ".split()


def make_reply(total_chars: int) -> str:
    # ~400-char paragraphs + periodic headings/fences: real block-mounting mix.
    parts: list[str] = []
    length = 0
    paragraph = 0
    word_index = 0
    while length < total_chars:
        paragraph += 1
        if paragraph % 8 == 0:
            block = f"## Section {paragraph}\n\n"
        elif paragraph % 11 == 0:
            block = f"```python\nvalue_{paragraph} = compute({paragraph})\n```\n\n"
        else:
            words: list[str] = []
            para_len = 0
            while para_len < 400:
                word = _WORDS[word_index % len(_WORDS)]
                word_index += 1
                words.append(word)
                para_len += len(word) + 1
            block = " ".join(words) + "\n\n"
        parts.append(block)
        length += len(block)
    return "".join(parts)[:total_chars]


async def stream_reply(app, pilot, reply: str, quarters: list[float]) -> None:
    message = AssistantMessage("")
    await app._mount_and_scroll(message)
    await pilot.pause()

    deltas = [reply[i : i + _DELTA_CHARS] for i in range(0, len(reply), _DELTA_CHARS)]
    quarter_size = max(1, len(deltas) // 4)
    start = time.process_time()
    for i, delta in enumerate(deltas):
        await message.append_content(delta)
        message._cancel_flush_timer()
        await message._flush_buffer()
        await pilot.pause()
        if (i + 1) % quarter_size == 0 and len(quarters) < 4:
            now = time.process_time()
            quarters.append(now - start)
            start = now
    await message.stop_stream()
    await pilot.pause()


@pytest.mark.timeout(0)
@pytest.mark.skipif(not _RUN, reason="set VIBE_STREAM_PROFILE=1 to run the harness")
@pytest.mark.asyncio
async def test_stream_render_cpu_vs_length() -> None:
    app = build_test_vibe_app()
    totals: dict[int, float] = {}
    quarter_map: dict[int, list[float]] = {}

    async with app.run_test() as pilot:
        for size_kb in _SIZES_KB:
            reply = make_reply(size_kb * 1024)
            await app._messages_area.remove_children()
            await pilot.pause()

            profiler = None
            if _CPROFILE and size_kb == max(_SIZES_KB):
                import cProfile

                profiler = cProfile.Profile()
                profiler.enable()

            quarters: list[float] = []
            begin = time.process_time()
            await stream_reply(app, pilot, reply, quarters)
            totals[size_kb] = time.process_time() - begin
            quarter_map[size_kb] = quarters

            if profiler is not None:
                import io
                import pstats

                profiler.disable()
                buf = io.StringIO()
                pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats(
                    "tottime"
                ).print_stats(_TOP)
                print(f"\n[stream] cProfile at {size_kb}KB:\n{buf.getvalue()}")

    print(f"\n[stream] delta={_DELTA_CHARS} chars, one flush per delta")
    print(" size |  total CPU | q1 .. q4 (s) | q4/q1")
    for size_kb in _SIZES_KB:
        quarters = quarter_map[size_kb]
        ratio = quarters[3] / quarters[0] if len(quarters) == 4 else float("nan")
        qs = " ".join(f"{q:6.3f}" for q in quarters)
        print(f" {size_kb:3d}KB | {totals[size_kb]:9.3f}s | {qs} | {ratio:5.2f}x")
    for smaller, larger in zip(_SIZES_KB, _SIZES_KB[1:], strict=False):
        if totals[smaller] > 0:
            print(
                f"[stream] {larger}KB / {smaller}KB total CPU: "
                f"{totals[larger] / totals[smaller]:.2f}x "
                f"(size ratio {larger / smaller:.1f}x)"
            )
