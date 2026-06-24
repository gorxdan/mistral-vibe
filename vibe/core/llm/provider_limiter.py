"""Process-global, per-provider request limiter.

Secondary turn calls (judge, memory, narrator) and le-chaton/workflow fan-out
each build a fresh backend, so a per-instance bound would not be shared. State
is therefore keyed by running loop (pytest isolation) then provider name, so
every backend targeting one provider on one loop shares a semaphore.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
import weakref

if TYPE_CHECKING:
    from vibe.core.config import ProviderConfig

DEFAULT_MAX_CONCURRENT_REQUESTS = 4


class _ProviderState:
    def __init__(self, max_concurrency: int, requests_per_minute: float | None) -> None:
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._min_interval = (
            60.0 / requests_per_minute
            if requests_per_minute and requests_per_minute > 0
            else 0.0
        )
        self._pace_lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def _pace(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._pace_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_allowed = max(now, self._next_allowed) + self._min_interval


_registry: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _ProviderState]
] = weakref.WeakKeyDictionary()


def _state_for(provider: ProviderConfig) -> _ProviderState:
    loop = asyncio.get_running_loop()
    states = _registry.get(loop)
    if states is None:
        states = {}
        _registry[loop] = states
    state = states.get(provider.name)
    if state is None:
        max_concurrency = (
            provider.max_concurrent_requests or DEFAULT_MAX_CONCURRENT_REQUESTS
        )
        state = _ProviderState(max_concurrency, provider.requests_per_minute)
        states[provider.name] = state
    return state


@asynccontextmanager
async def provider_slot(provider: ProviderConfig) -> AsyncIterator[None]:
    state = _state_for(provider)
    async with state.semaphore:
        await state._pace()
        yield
