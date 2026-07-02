from __future__ import annotations

from tests.conftest import build_test_agent_loop, build_test_vibe_config
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
