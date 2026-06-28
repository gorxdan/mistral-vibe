from __future__ import annotations

import asyncio

import pytest

from vibe.core.config import ProviderConfig
from vibe.core.llm.provider_limiter import provider_slot


def _provider(
    name: str,
    *,
    max_concurrent_requests: int | None = None,
    requests_per_minute: float | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        api_base="https://x.test/v1",
        max_concurrent_requests=max_concurrent_requests,
        requests_per_minute=requests_per_minute,
    )


@pytest.mark.asyncio
async def test_concurrency_capped_per_provider() -> None:
    provider = _provider("cap2", max_concurrent_requests=2)
    active = 0
    peak = 0

    async def worker() -> None:
        nonlocal active, peak
        async with provider_slot(provider):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(*(worker() for _ in range(8)))
    assert peak <= 2


@pytest.mark.asyncio
async def test_distinct_providers_do_not_share_a_slot() -> None:
    a = _provider("solo-a", max_concurrent_requests=1)
    b = _provider("solo-b", max_concurrent_requests=1)
    active = 0
    peak = 0

    async def worker(p: ProviderConfig) -> None:
        nonlocal active, peak
        async with provider_slot(p):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1

    await asyncio.gather(worker(a), worker(b))
    assert peak == 2


@pytest.mark.asyncio
async def test_requests_per_minute_paces_starts() -> None:
    provider = _provider("paced", max_concurrent_requests=10, requests_per_minute=600)
    loop = asyncio.get_running_loop()
    starts: list[float] = []

    async def worker() -> None:
        async with provider_slot(provider):
            starts.append(loop.time())

    await asyncio.gather(*(worker() for _ in range(3)))
    starts.sort()
    assert starts[-1] - starts[0] >= 0.18
