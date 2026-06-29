from __future__ import annotations

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from vibe.core.agent_loop._limits import (
    MAX_TOOL_RESULT_CHARS,
    TOOL_RESULT_CHARS_PER_TOKEN,
    TOOL_RESULT_WINDOW_FRACTION,
)


def _loop_with_threshold(threshold: int):
    models = make_test_models(auto_compact_threshold=threshold)
    config = build_test_vibe_config(models=models, active_model=models[0].alias)
    return build_test_agent_loop(config=config)


def test_hard_cap_floored_for_small_window():
    # Below ~500k-token windows the scaled value is under the constant, so the
    # floor keeps behaviour unchanged (no overflow risk for small models).
    loop = _loop_with_threshold(200_000)
    assert loop._tool_result_hard_cap() == MAX_TOOL_RESULT_CHARS


def test_hard_cap_scales_for_large_window():
    loop = _loop_with_threshold(880_000)
    expected = int(880_000 * TOOL_RESULT_CHARS_PER_TOKEN * TOOL_RESULT_WINDOW_FRACTION)
    assert loop._tool_result_hard_cap() == expected
    assert loop._tool_result_hard_cap() > MAX_TOOL_RESULT_CHARS
