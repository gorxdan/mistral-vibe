from __future__ import annotations

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from vibe.core.agent_loop_limits import (
    MAX_TOOL_RESULT_CHARS,
    TOOL_RESULT_CHARS_PER_TOKEN,
    TOOL_RESULT_WINDOW_FRACTION,
    tool_result_hard_cap,
)


def _loop_with_threshold(threshold: int):
    models = make_test_models(auto_compact_threshold=threshold)
    config = build_test_vibe_config(models=models, active_model=models[0].alias)
    return build_test_agent_loop(config=config)


def test_hard_cap_large_windows_unchanged():
    # Large windows keep the original behaviour: the 100k floor still applies at
    # 200k (40k scaled < floor) and the scaled value wins at 880k.
    assert tool_result_hard_cap(200_000) == MAX_TOOL_RESULT_CHARS
    expected_big = int(
        880_000 * TOOL_RESULT_CHARS_PER_TOKEN * TOOL_RESULT_WINDOW_FRACTION
    )
    assert tool_result_hard_cap(880_000) == expected_big
    assert expected_big > MAX_TOOL_RESULT_CHARS


def test_hard_cap_scales_down_for_small_window():
    # A 27k-threshold local model (Ornith on a 32k window): the cap MUST fall
    # below the 25k-token MAX_TOOL_RESULT_CHARS floor, otherwise one tool result
    # can occupy ~76% of the window and overflow it (the regression that 400'd
    # the Qwen3 chat template via truncation).
    cap = tool_result_hard_cap(27_000)
    assert cap < MAX_TOOL_RESULT_CHARS
    # And it must scale with the budget, never below a usable minimum.
    assert cap <= int(27_000 * TOOL_RESULT_CHARS_PER_TOKEN * 0.20)
    assert cap > 0


def test_hard_cap_monotonic_in_threshold():
    caps = [tool_result_hard_cap(t) for t in (8_000, 27_000, 64_000, 200_000, 880_000)]
    assert caps == sorted(caps)


def test_hard_cap_method_matches_pure_fn():
    loop = _loop_with_threshold(27_000)
    assert loop._tool_result_hard_cap() == tool_result_hard_cap(27_000)
    assert loop._tool_result_hard_cap() < MAX_TOOL_RESULT_CHARS
