from __future__ import annotations

from unittest.mock import MagicMock

from vibe.core.skills.manager import SkillManager
from vibe.core.skills.models import SkillInfo
from vibe.core.system_prompt import _get_available_skills_section


def _make_manager(skills: dict[str, SkillInfo]) -> SkillManager:
    manager = MagicMock(spec=SkillManager)
    manager.available_skills = skills
    return manager


def test_section_is_empty_when_no_skills() -> None:
    assert _get_available_skills_section(_make_manager({})) == ""


def test_truncates_description_to_first_sentence() -> None:
    manager = _make_manager({
        "stripe": SkillInfo(
            name="stripe",
            description=(
                "Implement Stripe payment processing. "
                "Covers checkout, subscriptions, and webhooks."
            ),
            prompt="body",
        )
    })

    section = _get_available_skills_section(manager)

    assert "- **stripe**: Implement Stripe payment processing." in section
    assert "subscriptions" not in section


def test_summary_overrides_description_in_index() -> None:
    manager = _make_manager({
        "expo": SkillInfo(
            name="expo",
            description="Build React Native apps with Expo. Long body omitted.",
            summary="Build Expo apps.",
            prompt="body",
        )
    })

    section = _get_available_skills_section(manager)

    assert "- **expo**: Build Expo apps." in section
    assert "React Native" not in section


def test_long_line_without_early_boundary_is_capped_with_ellipsis() -> None:
    run_on = "x" * 200  # no sentence terminator, exceeds the 160-char cap
    manager = _make_manager({
        "long": SkillInfo(name="long", description=run_on, prompt="body")
    })

    section = _get_available_skills_section(manager)

    line = next(l for l in section.splitlines() if l.startswith("- **long**"))
    # name prefix + cap + ellipsis; never the full 200-char description
    assert line.endswith("\u2026")
    assert run_on not in section


def test_skill_path_is_not_rendered_in_index() -> None:
    manager = _make_manager({
        "fs": SkillInfo(
            name="fs",
            description="A filesystem skill.",
            skill_path="/abs/path/SKILL.md",
            prompt="body",
        )
    })

    section = _get_available_skills_section(manager)

    assert "/abs/path" not in section
    assert "<path>" not in section
    assert "<description>" not in section


def test_name_and_summary_are_html_escaped() -> None:
    manager = _make_manager({
        "x": SkillInfo(
            name="x", description="A <b>bold</b> skill & more.", prompt="body"
        )
    })

    section = _get_available_skills_section(manager)

    assert "&lt;b&gt;bold&lt;/b&gt;" in section
    assert "&amp; more." in section
