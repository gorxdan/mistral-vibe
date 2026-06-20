from __future__ import annotations

from vibe.core.tools.builtins.launch_workflow import _extract_planned_phases


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
