from __future__ import annotations

from typing import Any

import pytest

from vibe.cli.textual_ui.widgets.messages import StreamingMessageBase


class FakeStream:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.stopped = False

    async def write(self, content: str) -> None:
        self.written.append(content)

    async def stop(self) -> None:
        self.stopped = True

    @property
    def all_written(self) -> str:
        return "".join(self.written)


class _PendingTimer:
    """Stand-in for a Textual Timer — only ``stop()`` is ever touched."""

    def stop(self) -> None: ...


class MessageTestDouble(StreamingMessageBase):
    """Minimal test double for StreamingMessageBase that bypasses Textual internals."""

    def __init__(self, at_bottom: bool = True, should_write: bool = True) -> None:
        # Initialise only the fields used by the buffer logic — no Textual setup.
        self._content = ""
        self._content_initialized = False
        self._to_write_buffer = ""
        self._stream = None
        self._markdown = None
        self._flush_timer = None
        self._timers_started = 0
        self._at_bottom = at_bottom
        self._should_write = should_write
        self._fake_stream: FakeStream = FakeStream()

    # --- overrides used by the buffer logic ---

    def _ensure_stream(self) -> FakeStream:  # type: ignore[override]
        if self._stream is None:
            self._stream = self._fake_stream  # type: ignore[assignment]
        return self._fake_stream

    def _is_chat_at_bottom(self) -> bool:
        return self._at_bottom

    def _should_write_content(self) -> bool:
        return self._should_write

    def set_timer(  # type: ignore[override]
        self,
        delay: float,
        callback: Any = None,
        *,
        name: str | None = None,
        pause: bool = False,
    ) -> _PendingTimer:
        # Seam for _schedule_flush: record that a flush timer was armed (so the
        # coalescing guard is testable) without running a real event loop.
        self._timers_started += 1
        return _PendingTimer()


def make_msg(*, at_bottom: bool = True, should_write: bool = True) -> MessageTestDouble:
    return MessageTestDouble(at_bottom=at_bottom, should_write=should_write)


class TestAppendContent:
    @pytest.mark.asyncio
    async def test_at_bottom_buffers_and_schedules_flush(self) -> None:
        msg = make_msg(at_bottom=True)

        await msg.append_content("hello")

        # Buffered, not written; one flush timer armed for the render frame.
        assert msg._fake_stream.all_written == ""
        assert msg._to_write_buffer == "hello"
        assert msg._timers_started == 1

    @pytest.mark.asyncio
    async def test_scrolled_away_buffers_without_writing_or_scheduling(self) -> None:
        msg = make_msg(at_bottom=False)

        await msg.append_content("hello")

        assert msg._fake_stream.all_written == ""
        assert msg._to_write_buffer == "hello"
        assert msg._timers_started == 0

    @pytest.mark.asyncio
    async def test_multiple_chunks_scrolled_away_accumulate(self) -> None:
        msg = make_msg(at_bottom=False)

        await msg.append_content("foo")
        await msg.append_content(" bar")

        assert msg._fake_stream.all_written == ""
        assert msg._to_write_buffer == "foo bar"

    @pytest.mark.asyncio
    async def test_at_bottom_chunks_coalesce_into_single_write(self) -> None:
        msg = make_msg(at_bottom=True)

        for chunk in ("a", "b", "c", "d", "e"):
            await msg.append_content(chunk)

        # All buffered; the guard arms the flush timer exactly once.
        assert msg._fake_stream.all_written == ""
        assert msg._to_write_buffer == "abcde"
        assert msg._timers_started == 1

        await msg._flush_buffer()

        # One write carries the whole coalesced buffer; timer cleared so the
        # next delta can reschedule.
        assert msg._fake_stream.written == ["abcde"]
        assert msg._to_write_buffer == ""
        assert msg._flush_timer is None

    @pytest.mark.asyncio
    async def test_flush_is_noop_when_buffer_empty(self) -> None:
        msg = make_msg(at_bottom=True)

        await msg._flush_buffer()

        assert msg._fake_stream.all_written == ""
        assert msg._flush_timer is None

    @pytest.mark.asyncio
    async def test_scroll_back_then_flush_writes_buffer_with_new_chunk(self) -> None:
        msg = make_msg(at_bottom=False)
        await msg.append_content("buffered")

        msg._at_bottom = True
        await msg.append_content(" new")

        # Nothing written yet — flush is scheduled, not immediate.
        assert msg._fake_stream.all_written == ""
        await msg._flush_buffer()

        # Buffered + new chunk written together in a single write call.
        assert msg._fake_stream.written == ["buffered new"]
        assert msg._to_write_buffer == ""

    @pytest.mark.asyncio
    async def test_empty_content_is_ignored(self) -> None:
        msg = make_msg()

        await msg.append_content("")

        assert msg._fake_stream.all_written == ""
        assert msg._to_write_buffer == ""

    @pytest.mark.asyncio
    async def test_should_write_false_skips_stream_and_buffer(self) -> None:
        msg = make_msg(should_write=False)

        await msg.append_content("invisible")

        assert msg._fake_stream.all_written == ""
        assert msg._to_write_buffer == ""
        # Content still accumulates in _content for later full-replay.
        assert msg._content == "invisible"


class TestStopStream:
    @pytest.mark.asyncio
    async def test_flushes_remaining_buffer(self) -> None:
        msg = make_msg(at_bottom=False)
        await msg.append_content("pending")

        await msg.stop_stream()

        assert msg._fake_stream.all_written == "pending"
        assert msg._to_write_buffer == ""

    @pytest.mark.asyncio
    async def test_stops_the_stream(self) -> None:
        msg = make_msg(at_bottom=False)
        await msg.append_content("x")
        await msg.stop_stream()

        assert msg._fake_stream.stopped is True

    @pytest.mark.asyncio
    async def test_idempotent_no_double_write(self) -> None:
        msg = make_msg(at_bottom=False)
        await msg.append_content("once")

        await msg.stop_stream()
        await msg.stop_stream()

        assert msg._fake_stream.all_written == "once"

    @pytest.mark.asyncio
    async def test_does_not_write_when_should_write_false(self) -> None:
        msg = make_msg(at_bottom=True, should_write=False)
        # Manually poke the buffer to verify the guard is respected.
        msg._to_write_buffer = "invisible"

        await msg.stop_stream()

        assert msg._fake_stream.all_written == ""
        assert msg._to_write_buffer == ""

    @pytest.mark.asyncio
    async def test_empty_buffer_no_extra_write(self) -> None:
        msg = make_msg()
        await msg.append_content("live")  # buffered (flush armed, not yet run)

        await msg.stop_stream()

        # stop_stream flushes the buffer once; no spurious second write.
        assert msg._fake_stream.written == ["live"]


class TestWriteInitialContent:
    @pytest.mark.asyncio
    async def test_writes_accumulated_content(self) -> None:
        msg = make_msg()
        msg._content = "full content"

        await msg.write_initial_content()

        assert msg._fake_stream.all_written == "full content"

    @pytest.mark.asyncio
    async def test_is_idempotent_second_call_writes_nothing(self) -> None:
        msg = make_msg()
        msg._content = "content"

        await msg.write_initial_content()
        await msg.write_initial_content()

        assert msg._fake_stream.all_written == "content"

    @pytest.mark.asyncio
    async def test_already_initialized_flag_is_noop(self) -> None:
        msg = make_msg()
        msg._content = "something"
        msg._content_initialized = True

        await msg.write_initial_content()

        assert msg._fake_stream.all_written == ""

    @pytest.mark.asyncio
    async def test_empty_content_writes_nothing(self) -> None:
        msg = make_msg()

        await msg.write_initial_content()

        assert msg._fake_stream.all_written == ""

    @pytest.mark.asyncio
    async def test_should_write_false_writes_nothing(self) -> None:
        msg = make_msg(should_write=False)
        msg._content = "hidden"

        await msg.write_initial_content()

        assert msg._fake_stream.all_written == ""

    @pytest.mark.asyncio
    async def test_clears_buffer_after_writing(self) -> None:
        msg = make_msg(at_bottom=False)
        await msg.append_content("buffered chunk")

        await msg.write_initial_content()

        assert msg._to_write_buffer == ""


class TestNoDoubleWrite:
    @pytest.mark.asyncio
    async def test_write_initial_then_stop_stream_no_duplication(self) -> None:
        """Regression: buffer must be cleared after write_initial_content so
        stop_stream does not re-write the same content.
        """
        msg = make_msg(at_bottom=False)
        await msg.append_content("part one ")
        await msg.append_content("part two")

        await msg.write_initial_content()
        await msg.stop_stream()

        written = msg._fake_stream.all_written
        assert written == "part one part two", (
            f"Expected content written exactly once, got: {written!r}"
        )

    @pytest.mark.asyncio
    async def test_write_initial_before_streaming_then_buffered_then_stop(self) -> None:
        """Reflects the real call order: write_initial_content is called once at
        mount time (stream is pristine, _content is empty), then streaming begins.
        Buffered content must be flushed exactly once by stop_stream.

        Note: calling write_initial_content AFTER live content has already been
        written to the stream is not a supported usage — it would duplicate the
        already-written portion because write_initial_content replays the full
        _content. In practice this never occurs because _mount_and_scroll calls
        write_initial_content immediately on mount, before streaming starts.
        """
        msg = make_msg(at_bottom=False)

        # At mount time content is empty — write_initial_content is a no-op
        # but arms _content_initialized so it can never write again.
        await msg.write_initial_content()

        await msg.append_content("buffered chunk one ")
        await msg.append_content("buffered chunk two")

        await msg.stop_stream()

        written = msg._fake_stream.all_written
        assert written == "buffered chunk one buffered chunk two", (
            f"Expected buffered content written exactly once, got: {written!r}"
        )

    @pytest.mark.asyncio
    async def test_write_initial_before_streaming_at_bottom_then_stop(self) -> None:
        """write_initial_content at mount (no-op), streaming at bottom writes
        directly, stop_stream has nothing extra to flush.
        """
        msg = make_msg(at_bottom=True)

        await msg.write_initial_content()

        await msg.append_content("live one ")
        await msg.append_content("live two")

        await msg.stop_stream()

        assert msg._fake_stream.all_written == "live one live two"

    @pytest.mark.asyncio
    async def test_stop_stream_twice_no_double_write(self) -> None:
        msg = make_msg(at_bottom=False)
        await msg.append_content("data")

        await msg.stop_stream()
        await msg.stop_stream()

        assert msg._fake_stream.all_written == "data"

    @pytest.mark.asyncio
    async def test_write_initial_twice_no_double_write(self) -> None:
        msg = make_msg()
        msg._content = "once"

        await msg.write_initial_content()
        await msg.write_initial_content()

        assert msg._fake_stream.all_written == "once"
