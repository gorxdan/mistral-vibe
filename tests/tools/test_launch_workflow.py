from __future__ import annotations

import pytest

from vibe.core.tools.builtins.launch_workflow import (
    _extract_planned_phases,
    _looks_like_path,
)


def test_extracts_literal_phase_names_in_order() -> None:
    script = (
        "async def main():\n"
        "    phase('explore')\n"
        "    phase('verify')\n"
        "    phase('synthesize')\n"
        "    return {}\n"
    )
    assert _extract_planned_phases(script) == ["explore", "verify", "synthesize"]


def test_dedupes_repeated_phase_names() -> None:
    script = (
        "async def main():\n"
        "    phase('research')\n"
        "    phase('research')\n"
        "    return {}\n"
    )
    assert _extract_planned_phases(script) == ["research"]


def test_ignores_dynamically_computed_phase_names() -> None:
    # A non-literal first argument must not produce a misleading phase name.
    script = (
        "async def main():\n"
        "    name = 'computed'\n"
        "    phase(name)\n"
        "    phase('real')\n"
        "    return {}\n"
    )
    assert _extract_planned_phases(script) == ["real"]


def test_returns_empty_for_script_without_phases() -> None:
    assert _extract_planned_phases("async def main(): return {}") == []


def test_invalid_python_returns_empty() -> None:
    assert _extract_planned_phases("async def main(:") == []


@pytest.mark.parametrize(
    "script",
    [
        "review_commits.py",
        "./scripts/review_commits.py",
        "review_commits.py\n",  # trailing newline only
        "  .vibe/workflows/audit.py  ",  # surrounding whitespace
    ],
)
def test_looks_like_path_detects_file_paths(script: str) -> None:
    # Passing a path instead of source is the most common launch mistake; the
    # guard must catch bare paths, dotted/relative paths, and stray whitespace.
    assert _looks_like_path(script) is True


@pytest.mark.parametrize(
    "script",
    [
        "async def main():\n    return {}\n",
        "async def main(): return {}",  # one-liner still contains "def "
        "import json\n\nasync def main():\n    return json.dumps({})\n",
    ],
)
def test_looks_like_path_does_not_flag_real_source(script: str) -> None:
    assert _looks_like_path(script) is False
