from __future__ import annotations

import gc
import warnings

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core import agent_loop as agent_loop_module
from vibe.core.baseline_scaling import BaselineTier


def _full_prompt_config(**kwargs):
    return build_test_vibe_config(
        include_prompt_detail=True, include_config_reference=True, **kwargs
    )


def test_current_baseline_tier_returns_tier_value() -> None:
    loop = build_test_agent_loop()

    tier = loop._current_baseline_tier()

    assert isinstance(tier, BaselineTier)
    assert loop._system_prompt_tier is BaselineTier.LARGE


def test_large_tier_prompt_contains_tier_gated_sections() -> None:
    loop = build_test_agent_loop(config=_full_prompt_config())

    prompt = loop.messages[0].content or ""

    assert "## Configuring Vibe (quick reference)" in prompt
    assert "## Verification contract" in prompt
    assert "## Investigation contract" in prompt


@pytest.mark.asyncio
async def test_prompt_not_rebuilt_when_tier_unchanged(monkeypatch) -> None:
    loop = build_test_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="one")],
            [mock_llm_chunk(content="two")],
        ])
    )
    rebuilds = 0
    real_builder = agent_loop_module.get_universal_system_prompt

    def counting_builder(*args, **kwargs):
        nonlocal rebuilds
        rebuilds += 1
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(
        agent_loop_module, "get_universal_system_prompt", counting_builder
    )

    [_ async for _ in loop.act("first")]
    [_ async for _ in loop.act("second")]

    assert rebuilds == 0


@pytest.mark.asyncio
async def test_no_unawaited_coroutine_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="hello"))
        )
        [_ async for _ in loop.act("go")]
        del loop
        gc.collect()

    unawaited = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "never awaited" in str(w.message)
    ]
    assert unawaited == []
