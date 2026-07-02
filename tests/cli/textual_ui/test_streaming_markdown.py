from __future__ import annotations

import pytest
from textual.widgets import Markdown
from textual.widgets._markdown import MarkdownBlock

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.widgets.messages import AssistantMessage, StreamingMarkdown

_REPLY = (
    "First paragraph with enough words to be a real block of prose here.\n\n"
    "## A heading\n\n"
    "Second paragraph, also a distinct top-level block in the document.\n\n"
    "```python\nvalue = compute(1)\n```\n\n"
    "Third and final paragraph rounding out the streamed reply body text.\n"
)


def _blocks(md: Markdown) -> list[MarkdownBlock]:
    return [c for c in md.children if isinstance(c, MarkdownBlock)]


async def _stream(pilot, message: AssistantMessage, reply: str, step: int = 24) -> None:
    for i in range(0, len(reply), step):
        await message.append_content(reply[i : i + step])
        message._cancel_flush_timer()
        await message._flush_buffer()
        await pilot.pause()
    await message.stop_stream()
    await pilot.pause()


class TestSettleFinalizedBlocks:
    @pytest.mark.asyncio
    async def test_assistant_message_uses_streaming_markdown(self) -> None:
        # Precondition for the rest: the assistant reply renders through our
        # subclass and streaming produced a multi-block document to settle.
        app = build_test_vibe_app()
        async with app.run_test() as pilot:
            message = AssistantMessage("")
            await app._mount_and_scroll(message)
            await pilot.pause()
            await _stream(pilot, message, _REPLY)
            md = message._get_markdown()
            assert isinstance(md, StreamingMarkdown)
            assert len(_blocks(md)) >= 4

    @pytest.mark.asyncio
    async def test_finalized_blocks_are_settled_tail_stays_styleable(self) -> None:
        app = build_test_vibe_app()
        async with app.run_test() as pilot:
            message = AssistantMessage("")
            await app._mount_and_scroll(message)
            await pilot.pause()
            await _stream(pilot, message, _REPLY)

            blocks = _blocks(message._get_markdown())
            # The tail is never settled (it may still be demoted by a future
            # append), so Textual keeps re-styling only it.
            assert blocks[-1]._has_order_style is True
            # Every earlier block (bar the just-demoted penultimate) is settled:
            # its order-dependent styles are final so the mount closure skips it.
            assert all(not b._has_order_style for b in blocks[:-2])

    @pytest.mark.asyncio
    async def test_streamed_document_matches_one_shot_render(self) -> None:
        # Byte-identity: small-delta streaming yields the same blocks (type, text,
        # margins) as a one-shot build; margins catch a stale settled :last-child.
        app = build_test_vibe_app()
        async with app.run_test() as pilot:
            streamed = AssistantMessage("")
            oneshot = AssistantMessage(_REPLY)
            await app._mount_and_scroll(streamed)
            await app._mount_and_scroll(oneshot)
            await oneshot.write_initial_content()
            await pilot.pause()
            await _stream(pilot, streamed, _REPLY)

            def shape(md: Markdown) -> list[tuple[str, str, tuple[int, int, int, int]]]:
                return [
                    (type(b).__name__, b._content.plain, b.styles.margin)
                    for b in _blocks(md)
                ]

            assert shape(streamed._get_markdown()) == shape(oneshot._get_markdown())
            # The one-shot render's tail carries the :last-child override, so the
            # equality above also pins the streamed tail to margin-bottom 0.
            assert _blocks(streamed._get_markdown())[-1].styles.margin.bottom == 0
