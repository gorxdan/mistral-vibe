from __future__ import annotations

from typing import Any

import pytest

from vibe.cli.textual_ui.widgets.messages import BashOutputMessage


class _PendingTimer:
    """Stand-in for a Textual Timer — only ``stop()`` is ever touched."""

    def stop(self) -> None: ...


class BashTestDouble(BashOutputMessage):
    """Minimal test double for BashOutputMessage that bypasses Textual internals.

    Mirrors the MessageTestDouble pattern in test_streaming_message_buffer.py: it
    sets only the fields the coalescing logic touches and stubs the Widget-state
    methods finish() reaches for, so the buffer/flush logic is unit-testable
    without a running app.
    """

    def __init__(self) -> None:
        # Initialise only the fields used by the buffer logic — no Textual setup.
        self._output = ""
        self._output_dirty = False
        self._flush_timer = None
        # Truthy sentinel short-circuits _ensure_output_container (no mount).
        self._output_container: Any = object()
        self._prompt_widget = None
        self._spinner_timer = None  # so SpinnerMixin.on_unmount via super() is safe
        self._timers_started = 0
        self._refresh_calls = 0

    # --- overrides used by the buffer logic ---

    async def _ensure_output_container(self) -> None:  # type: ignore[override]
        return  # bypass mount; _output_container sentinel keeps it a no-op anyway

    def set_timer(  # type: ignore[override]
        self,
        delay: float,
        callback: Any = None,
        *,
        name: str | None = None,
        pause: bool = False,
    ) -> _PendingTimer:
        # Seam for _schedule_output_flush: record that a flush timer was armed
        # (so the coalescing guard is testable) without running a real event loop.
        self._timers_started += 1
        return _PendingTimer()

    def _refresh_output_widgets(self) -> None:  # type: ignore[override]
        self._refresh_calls += 1

    # --- Widget-state stubs so finish() runs without Textual ---

    def stop_spinning(self, success: bool = True) -> None:  # type: ignore[override]
        pass

    def add_class(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        pass

    def remove_class(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        pass


def make_msg() -> BashTestDouble:
    return BashTestDouble()


class TestAppendOutput:
    @pytest.mark.asyncio
    async def test_chunks_coalesce_into_single_flush(self) -> None:
        msg = make_msg()

        for chunk in ("line one\n", "line two\n", "line three\n"):
            await msg.append_output(chunk)

        # Buffered, not yet rendered; exactly one flush timer armed.
        assert msg._output == "line one\nline two\nline three\n"
        assert msg._output_dirty is True
        assert msg._refresh_calls == 0
        assert msg._timers_started == 1

        msg._flush_output()

        # One refresh carries the whole coalesced buffer; timer cleared so the
        # next chunk can reschedule.
        assert msg._refresh_calls == 1
        assert msg._output_dirty is False
        assert msg._flush_timer is None

    @pytest.mark.asyncio
    async def test_flush_is_noop_when_not_dirty(self) -> None:
        msg = make_msg()

        msg._flush_output()

        assert msg._refresh_calls == 0
        assert msg._flush_timer is None

    @pytest.mark.asyncio
    async def test_flush_reschedules_for_next_chunk(self) -> None:
        msg = make_msg()
        await msg.append_output("a")
        msg._flush_output()

        assert msg._timers_started == 1
        await msg.append_output("b")
        # New chunk after a flush arms a fresh timer.
        assert msg._timers_started == 2

    @pytest.mark.asyncio
    async def test_empty_chunk_is_noop(self) -> None:
        # append_output guards on empty input (mirrors StreamingMessageBase): no
        # dirty flag, no timer armed, no output mutated.
        msg = make_msg()

        await msg.append_output("")

        assert msg._output == ""
        assert msg._output_dirty is False
        assert msg._timers_started == 0


class TestFinish:
    @pytest.mark.asyncio
    async def test_finish_flushes_remainder_exactly_once(self) -> None:
        msg = make_msg()
        await msg.append_output("pending\n")
        await msg.append_output("more\n")
        # Timer armed, but no coalesced flush has fired yet.
        assert msg._refresh_calls == 0

        await msg.finish(0)

        # The coalesced timer never fired separately; finish cancels it and does
        # the single terminal render. No double-write.
        assert msg._refresh_calls == 1
        assert msg._flush_timer is None
        assert msg._output_dirty is False
        assert "pending" in msg._output and "more" in msg._output

    @pytest.mark.asyncio
    async def test_finish_after_coalesced_flush_renders_terminal_once(self) -> None:
        msg = make_msg()
        await msg.append_output("chunk")
        msg._flush_output()  # coalesced flush during streaming
        assert msg._refresh_calls == 1

        await msg.finish(0)

        # finish always does one terminal render regardless of prior flushes.
        assert msg._refresh_calls == 2
        assert msg._flush_timer is None

    @pytest.mark.asyncio
    async def test_finish_no_output_renders_placeholder(self) -> None:
        msg = make_msg()

        await msg.finish(0)

        assert msg._output == "(no output)"
        assert msg._refresh_calls == 1

    @pytest.mark.asyncio
    async def test_finish_interrupted_appends_suffix(self) -> None:
        msg = make_msg()
        await msg.append_output("partial")

        await msg.finish(0, interrupted=True)

        assert "partial" in msg._output
        assert "(interrupted)" in msg._output


class TestNoDataLoss:
    @pytest.mark.asyncio
    async def test_all_streamed_content_present_after_finish(self) -> None:
        msg = make_msg()
        chunks = ["alpha\n", "beta\n", "gamma\n", "delta\n"]

        for chunk in chunks:
            await msg.append_output(chunk)
        await msg.finish(0)

        assert msg._output == "alpha\nbeta\ngamma\ndelta\n"

    @pytest.mark.asyncio
    async def test_finish_twice_no_double_render_of_remainder(self) -> None:
        msg = make_msg()
        await msg.append_output("data")

        await msg.finish(0)
        await msg.finish(0)

        # Second finish renders once more (terminal), but no content duplication.
        assert msg._output == "data"


class TestOnUnmount:
    @pytest.mark.asyncio
    async def test_on_unmount_cancels_pending_flush_timer(self) -> None:
        msg = make_msg()
        await msg.append_output("unflushed")
        assert msg._flush_timer is not None

        msg.on_unmount()

        assert msg._flush_timer is None
