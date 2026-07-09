from __future__ import annotations

from vibe.core.agent_loop_limits import MAX_CONCURRENT_SUBAGENTS


def test_paid_subagent_concurrency_default_is_bounded() -> None:
    assert MAX_CONCURRENT_SUBAGENTS == 2
