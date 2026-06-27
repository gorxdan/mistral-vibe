from __future__ import annotations

from vibe import __version__
from vibe.core.skills.builtins.capsules import SkillDocCapsule
from vibe.core.skills.builtins.vibe import SKILL, VIBE_DOC_CAPSULE
from vibe.core.skills.models import SkillScope, SkillSource

_VERSION_PLACEHOLDER = "__VIBE_VERSION__"


def _make_capsule() -> SkillDocCapsule:
    return SkillDocCapsule(
        name="demo",
        description="A demo capsule.",
        summary="Demo summary.",
        prompt_template="Version is " + _VERSION_PLACEHOLDER + ".",
    )


class TestCapsuleRendering:
    def test_render_agent_prompt_substitutes_version(self) -> None:
        capsule = _make_capsule()
        assert _VERSION_PLACEHOLDER not in capsule.render_agent_prompt("1.2.3")
        assert capsule.render_agent_prompt("1.2.3") == "Version is 1.2.3."

    def test_render_human_markdown_matches_agent_prompt(self) -> None:
        capsule = _make_capsule()
        assert capsule.render_human_markdown("9.9.9") == capsule.render_agent_prompt(
            "9.9.9"
        )

    def test_custom_version_placeholder(self) -> None:
        capsule = SkillDocCapsule(
            name="demo",
            description="d",
            summary="s",
            prompt_template="v=__VER__",
            version_placeholder="__VER__",
        )
        assert capsule.render_agent_prompt("0.1") == "v=0.1"


class TestCapsuleToSkillInfo:
    def test_to_skill_info_carries_metadata(self) -> None:
        capsule = _make_capsule()
        info = capsule.to_skill_info("4.5.6")
        assert info.name == "demo"
        assert info.description == "A demo capsule."
        assert info.summary == "Demo summary."
        assert info.user_invocable is False
        assert info.prompt == "Version is 4.5.6."

    def test_to_skill_info_labels_builtin_source_and_scope(self) -> None:
        info = _make_capsule().to_skill_info("0.0.0")
        assert info.source is SkillSource.BUILTIN
        assert info.scope is SkillScope.BUILTIN

    def test_user_invocable_propagates(self) -> None:
        capsule = SkillDocCapsule(
            name="u",
            description="d",
            summary="s",
            prompt_template="p",
            user_invocable=True,
        )
        assert capsule.to_skill_info("0").user_invocable is True


class TestVibeCapsuleWiring:
    def test_skill_matches_capsule_at_running_version(self) -> None:
        assert SKILL.prompt == VIBE_DOC_CAPSULE.render_agent_prompt(__version__)
        assert _VERSION_PLACEHOLDER not in SKILL.prompt

    def test_capsule_name_matches_skill_name(self) -> None:
        assert VIBE_DOC_CAPSULE.name == SKILL.name == "vibe"

    def test_skill_is_builtin_scoped(self) -> None:
        assert SKILL.source is SkillSource.BUILTIN
        assert SKILL.scope is SkillScope.BUILTIN
