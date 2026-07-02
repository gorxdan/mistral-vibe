from __future__ import annotations

from vibe.core.skills.builtins.tool_guides import SKILL as TOOL_GUIDES_SKILL
from vibe.core.skills.builtins.vibe import SKILL as VIBE_SKILL
from vibe.core.skills.builtins.workflow import SKILL as WORKFLOW_SKILL
from vibe.core.skills.models import SkillInfo

BUILTIN_SKILLS: dict[str, SkillInfo] = {
    skill.name: skill for skill in [VIBE_SKILL, WORKFLOW_SKILL, TOOL_GUIDES_SKILL]
}

__all__ = ["BUILTIN_SKILLS"]
