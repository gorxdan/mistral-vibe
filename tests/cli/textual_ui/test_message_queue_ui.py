from __future__ import annotations

import time

import pytest

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.app import VibeApp
from vibe.cli.textual_ui.message_queue import QueuedItemKind
from vibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from vibe.cli.textual_ui.widgets.messages import (
    BashOutputMessage,
    QueueHeaderMessage,
    UserMessage,
    WarningMessage,
)


@pytest.fixture
def vibe_app() -> VibeApp:
    return build_test_vibe_app()


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return False


@pytest.mark.asyncio
async def test_no_queue_header_when_empty(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test():
        headers = list(vibe_app.query(QueueHeaderMessage))
        assert headers == []


@pytest.mark.asyncio
async def test_bash_submitted_during_running_bash_is_queued(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 0.3"
        await pilot.press("enter")

        await _wait_until(pilot, lambda: vibe_app._bash_task is not None, timeout=1.0)

        chat_input.value = "!echo queued"
        await pilot.press("enter")

        assert len(vibe_app._input_queue) == 1
        assert vibe_app._input_queue.items[0].content == "echo queued"

        headers = list(vibe_app.query(QueueHeaderMessage))
        assert len(headers) == 1

        queued_bashes = [w for w in vibe_app.query(BashOutputMessage) if w._queued]
        assert len(queued_bashes) == 1


@pytest.mark.asyncio
async def test_slash_command_rejected_with_warning_when_busy(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 0.3"
        await pilot.press("enter")

        await _wait_until(pilot, lambda: vibe_app._bash_task is not None, timeout=1.0)

        chat_input.value = "/help"
        await pilot.press("enter")

        assert not list(vibe_app.query(WarningMessage))
        assert any(
            "Slash commands cannot be queued" in notification.message
            for notification in vibe_app._notifications
        )
        assert len(vibe_app._input_queue) == 0
        assert chat_input.value.startswith("/help")


@pytest.mark.asyncio
async def test_ctrl_c_pops_last_queued_item_lifo(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 2"
        await pilot.press("enter")

        await _wait_until(pilot, lambda: vibe_app._bash_task is not None, timeout=2.0)

        chat_input.value = "!echo first"
        await pilot.press("enter")
        chat_input.value = "!echo second"
        await pilot.press("enter")

        assert len(vibe_app._input_queue) == 2

        await pilot.press("ctrl+c")
        assert len(vibe_app._input_queue) == 1
        assert vibe_app._input_queue.items[0].content == "echo first"

        await pilot.press("escape")
        await _wait_until(pilot, lambda: vibe_app._bash_task is None, timeout=5.0)


@pytest.mark.asyncio
async def test_escape_pauses_queue_when_job_running(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 2"
        await pilot.press("enter")

        await _wait_until(pilot, lambda: vibe_app._bash_task is not None, timeout=2.0)

        chat_input.value = "!echo queued"
        await pilot.press("enter")
        assert len(vibe_app._input_queue) == 1

        await pilot.press("escape")
        assert vibe_app._input_queue.paused
        assert len(vibe_app._input_queue) == 1

        await _wait_until(pilot, lambda: vibe_app._bash_task is None, timeout=5.0)


@pytest.mark.asyncio
async def test_drain_runs_queued_bashes_in_fifo_order(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 0.2"
        await pilot.press("enter")

        await _wait_until(pilot, lambda: vibe_app._bash_task is not None, timeout=1.0)

        chat_input.value = "!echo first"
        await pilot.press("enter")
        chat_input.value = "!echo second"
        await pilot.press("enter")

        await _wait_until(
            pilot,
            lambda: (
                len(list(vibe_app.query(BashOutputMessage))) == 3
                and all(not m._pending for m in vibe_app.query(BashOutputMessage))
            ),
            timeout=5.0,
        )

        msgs = list(vibe_app.query(BashOutputMessage))
        assert len(msgs) == 3
        assert vibe_app._input_queue.paused is False
        assert len(vibe_app._input_queue) == 0


@pytest.mark.asyncio
async def test_enter_on_empty_input_flushes_paused_queue(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 2"
        await pilot.press("enter")

        await _wait_until(pilot, lambda: vibe_app._bash_task is not None, timeout=2.0)

        chat_input.value = "!echo queued"
        await pilot.press("enter")
        assert len(vibe_app._input_queue) == 1

        await pilot.press("escape")
        assert vibe_app._input_queue.paused

        await _wait_until(pilot, lambda: vibe_app._bash_task is None, timeout=10.0)

        chat_input.value = ""
        await pilot.press("enter")

        await _wait_until(
            pilot,
            lambda: (
                not vibe_app._input_queue.paused and len(vibe_app._input_queue) == 0
            ),
            timeout=10.0,
        )

        assert not vibe_app._input_queue.paused
        assert len(vibe_app._input_queue) == 0


@pytest.mark.asyncio
async def test_quit_warning_shows_queue_count(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test():
        vibe_app._input_queue.append_prompt("a")
        vibe_app._input_queue.append_prompt("b")
        warning = vibe_app._queue.quit_warning_extra()
        assert warning == "2 queued messages will be discarded"

        vibe_app._input_queue.pop_last()
        warning = vibe_app._queue.quit_warning_extra()
        assert warning == "1 queued message will be discarded"

        vibe_app._input_queue.pop_last()
        assert vibe_app._queue.quit_warning_extra() == ""


@pytest.mark.asyncio
async def test_double_enter_injects_queued_prompt_into_running_turn(
    vibe_app: VibeApp,
) -> None:
    async with vibe_app.run_test() as pilot:
        vibe_app._agent_running = True
        chat_input = vibe_app.query_one(ChatInputContainer)

        chat_input.value = "hey check this"
        await pilot.press("enter")
        assert len(vibe_app._input_queue) == 1
        assert vibe_app.agent_loop._pending_injected_messages == []

        # Second Enter on empty input folds the queued prompt into the turn.
        chat_input.value = ""
        await pilot.press("enter")
        await _wait_until(
            pilot, lambda: len(vibe_app._input_queue) == 0, timeout=2.0
        )

        assert len(vibe_app._input_queue) == 0
        staged = vibe_app.agent_loop._pending_injected_messages
        assert len(staged) == 1
        assert "check this" in (staged[0].content or "")
        assert staged[0].injected


@pytest.mark.asyncio
async def test_double_enter_stops_at_queued_bash(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        vibe_app._agent_running = True
        chat_input = vibe_app.query_one(ChatInputContainer)

        # Leading prompt is injectable; the bash after it is not (bash can't be
        # folded into an LLM turn), so double-enter injects the prompt and
        # leaves the bash queued.
        chat_input.value = "fold me now"
        await pilot.press("enter")
        chat_input.value = "!echo later"
        await pilot.press("enter")
        assert len(vibe_app._input_queue) == 2

        chat_input.value = ""
        await pilot.press("enter")
        await _wait_until(
            pilot, lambda: len(vibe_app._input_queue) == 1, timeout=2.0
        )

        assert len(vibe_app._input_queue) == 1
        assert vibe_app._input_queue.items[0].kind == QueuedItemKind.BASH
        assert vibe_app._input_queue.items[0].content == "echo later"

        staged = vibe_app.agent_loop._pending_injected_messages
        assert len(staged) == 1
        assert "fold me now" in (staged[0].content or "")


@pytest.mark.asyncio
async def test_double_enter_noop_without_queued_messages(
    vibe_app: VibeApp,
) -> None:
    async with vibe_app.run_test() as pilot:
        vibe_app._agent_running = True
        chat_input = vibe_app.query_one(ChatInputContainer)

        chat_input.value = ""
        await pilot.press("enter")
        await pilot.pause(0.05)

        assert len(vibe_app._input_queue) == 0
        assert vibe_app.agent_loop._pending_injected_messages == []


@pytest.mark.asyncio
async def test_double_enter_assigns_distinct_incrementing_widget_indices(
    vibe_app: VibeApp,
) -> None:
    """Multiple staged prompts must get distinct, incrementing message_index
    values matching the positions they will occupy in history when drained.

    Regression guard: capturing next_message_index() per item yielded the same
    stale index for every widget (staging defers the append), so rewind and
    at-mention telemetry resolved to the wrong message.
    """
    async with vibe_app.run_test() as pilot:
        vibe_app._agent_running = True
        chat_input = vibe_app.query_one(ChatInputContainer)

        chat_input.value = "first prompt"
        await pilot.press("enter")
        chat_input.value = "second prompt"
        await pilot.press("enter")
        assert len(vibe_app._input_queue) == 2

        chat_input.value = ""
        await pilot.press("enter")
        await _wait_until(
            pilot, lambda: len(vibe_app._input_queue) == 0, timeout=2.0
        )

        staged_widgets = [
            w for w in vibe_app.query(UserMessage)
            if w.message_index is not None and not w.pending
        ]
        indices = sorted(w.message_index for w in staged_widgets)
        assert len(indices) == 2, f"expected 2 staged widgets, got {len(indices)}"
        assert indices[1] == indices[0] + 1, (
            f"staged widget indices must be consecutive, got {indices}"
        )
        # The base index is the current history length, so both staged indices
        # must point at or beyond it (where the messages will actually land).
        base = len(vibe_app.agent_loop.messages)
        assert indices[0] >= base, (
            f"first staged index {indices[0]} below history length {base}"
        )


