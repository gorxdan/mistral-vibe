from __future__ import annotations

from typing import Any

import pytest

from vibe.core.agent_loop._init_guard import requires_init


class _FakeLoop:
    def __init__(self) -> None:
        self.ready_waits = 0

    async def wait_until_ready(self) -> None:
        self.ready_waits += 1


def test_sync_function_rejected_at_decoration_time() -> None:
    with pytest.raises(TypeError, match="async"):

        @requires_init
        def sync_method(self: Any) -> str:
            return "value"


@pytest.mark.asyncio
async def test_async_function_waits_then_returns_value() -> None:
    @requires_init
    async def method(self: _FakeLoop) -> str:
        return "value"

    loop = _FakeLoop()
    assert await method(loop) == "value"
    assert loop.ready_waits == 1


@pytest.mark.asyncio
async def test_async_generator_waits_then_yields() -> None:
    @requires_init
    async def gen(self: _FakeLoop):
        yield 1
        yield 2

    loop = _FakeLoop()
    assert [item async for item in gen(loop)] == [1, 2]
    assert loop.ready_waits == 1
