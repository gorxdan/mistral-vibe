from __future__ import annotations

from pathlib import Path

from tests.conftest import build_test_vibe_config
from tests.skills.conftest import create_skill
from vibe.core.skills.manager import SkillManager
from vibe.core.system_prompt import _get_available_skills_section


def _manager_with_skills(skills_dir: Path) -> SkillManager:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        skill_paths=[skills_dir],
    )
    return SkillManager(lambda: config)


def test_summary_frontmatter_flows_through_real_pipeline_into_index(
    skills_dir: Path,
) -> None:
    create_skill(
        skills_dir,
        "stripe",
        "Implement Stripe payment processing. Covers checkout, subscriptions, webhooks.",
        summary="Integrate Stripe payments.",
    )

    manager = _manager_with_skills(skills_dir)
    section = _get_available_skills_section(manager)

    assert "- **stripe**: Integrate Stripe payments." in section
    assert "subscriptions" not in section


def test_first_sentence_falls_back_when_no_summary_in_real_pipeline(
    skills_dir: Path,
) -> None:
    create_skill(
        skills_dir,
        "postgres",
        "Design a PostgreSQL-specific schema. Covers indexes and constraints.",
    )

    manager = _manager_with_skills(skills_dir)
    section = _get_available_skills_section(manager)

    assert "- **postgres**: Design a PostgreSQL-specific schema." in section
    assert "indexes and constraints" not in section


def test_full_description_not_present_in_index_but_prompt_still_loaded_on_demand(
    skills_dir: Path,
) -> None:
    create_skill(
        skills_dir,
        "audit",
        "Audit websites for SEO and performance. Returns detailed reports.",
    )

    manager = _manager_with_skills(skills_dir)
    section = _get_available_skills_section(manager)
    skill_info = manager.get_skill("audit")

    assert skill_info is not None
    assert skill_info.prompt == "## Instructions\n\nTest instructions here."
    assert "Returns detailed reports" not in section


def test_skill_path_from_filesystem_not_leaked_into_index(skills_dir: Path) -> None:
    create_skill(skills_dir, "fs-skill", "A filesystem-backed skill.")

    manager = _manager_with_skills(skills_dir)
    section = _get_available_skills_section(manager)
    skill_info = manager.get_skill("fs-skill")

    assert skill_info is not None
    assert skill_info.skill_path is not None
    assert str(skill_info.skill_path) not in section
