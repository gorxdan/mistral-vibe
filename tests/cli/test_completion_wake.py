from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.app import _AUTO_CONTINUE_PROMPT, _MAX_AUTO_CONTINUES

pytestmark = pytest.mark.asyncio


def _install_fake_turn(app: Any, started: list[str]) -> None:
    async def fake_turn(prompt: str, **_kwargs: Any) -> None:
        started.append(prompt)
        app._agent_running = False
        app._auto_continue_active = False
        app._agent_task = None

    app._handle_agent_loop_turn = fake_turn


def _install_busy_turn(app: Any, gate: asyncio.Event) -> asyncio.Task[None]:
    async def busy_turn() -> None:
        await gate.wait()
        app._agent_running = False
        app._auto_continue_active = False
        app._agent_task = None

    task = asyncio.create_task(busy_turn())
    app._agent_running = True
    app._agent_task = task
    return task


async def test_completion_wake_latches_until_busy_turn_settles() -> None:
    app = build_test_vibe_app()
    started: list[str] = []
    _install_fake_turn(app, started)
    app.agent_loop.stage_injected_message("team result")
    gate = asyncio.Event()
    _install_busy_turn(app, gate)

    wake = asyncio.create_task(app._on_async_completion_wake())
    await asyncio.sleep(0)

    assert app._completion_wake_pending is True
    assert started == []

    gate.set()
    await wake
    await asyncio.sleep(0)

    assert started == [_AUTO_CONTINUE_PROMPT]
    assert app._consecutive_auto_continues == 1


async def test_completion_wakes_coalesce_while_auto_continue_is_active() -> None:
    app = build_test_vibe_app()
    started: list[str] = []
    _install_fake_turn(app, started)
    app.agent_loop.stage_injected_message("team result")
    gate = asyncio.Event()
    _install_busy_turn(app, gate)
    app._auto_continue_active = True

    wakes = [
        asyncio.create_task(app._on_async_completion_wake()),
        asyncio.create_task(app._on_async_completion_wake()),
    ]
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(*wakes)
    await asyncio.sleep(0)

    assert started == [_AUTO_CONTINUE_PROMPT]
    assert app._consecutive_auto_continues == 1


async def test_queued_human_turn_claims_latched_completion_wake() -> None:
    app = build_test_vibe_app()
    started: list[str] = []
    _install_fake_turn(app, started)
    app.agent_loop.stage_injected_message("team result")
    gate = asyncio.Event()
    _install_busy_turn(app, gate)

    wake = asyncio.create_task(app._on_async_completion_wake())
    await asyncio.sleep(0)
    app._input_queue.append_prompt("human request")
    app._input_queue.pause()
    gate.set()
    await wake

    assert app._completion_wake_pending is True
    assert started == []

    item = app._input_queue.pop_first()
    assert item is not None
    app._input_queue.resume()
    human_turn = app._start_queued_agent_turn(item.content)
    await human_turn

    assert app._completion_wake_pending is False
    assert app._consecutive_auto_continues == 0
    assert started == ["human request"]


async def test_completion_wake_respects_auto_continue_cap() -> None:
    app = build_test_vibe_app()
    started: list[str] = []
    _install_fake_turn(app, started)
    app.agent_loop.stage_injected_message("team result")
    app._consecutive_auto_continues = _MAX_AUTO_CONTINUES

    await app._on_async_completion_wake()

    assert app._completion_wake_pending is False
    assert app._agent_task is None
    assert started == []


async def test_busy_turn_that_consumes_completion_does_not_auto_continue() -> None:
    app = build_test_vibe_app()
    started: list[str] = []
    _install_fake_turn(app, started)
    gate = asyncio.Event()
    _install_busy_turn(app, gate)

    wake = asyncio.create_task(app._on_async_completion_wake())
    await asyncio.sleep(0)
    gate.set()
    await wake

    assert app._completion_wake_pending is False
    assert started == []


async def test_paused_bash_queue_rearms_latched_completion_on_resume() -> None:
    app = build_test_vibe_app()
    started: list[str] = []
    _install_fake_turn(app, started)
    app.agent_loop.stage_injected_message("team result")
    app._input_queue.append_bash("noop")
    app._input_queue.pause()

    def fake_bash(_command: str, **_kwargs: Any) -> asyncio.Task[None]:
        return asyncio.create_task(asyncio.sleep(0))

    app._queue._ports = replace(app._queue._ports, run_bash=fake_bash)

    await app._on_async_completion_wake()

    assert app._completion_wake_pending is True
    assert started == []

    await app._handle_paused_submit("")
    for _ in range(20):
        await asyncio.sleep(0)
        if started:
            break

    assert started == [_AUTO_CONTINUE_PROMPT]
    assert app._completion_wake_pending is False
