from __future__ import annotations

from dataclasses import dataclass

from vibe.core.skills.models import SkillInfo, SkillScope, SkillSource


@dataclass(frozen=True)
class SkillDocCapsule:
    name: str
    description: str
    summary: str
    prompt_template: str
    user_invocable: bool = False
    version_placeholder: str = "__VIBE_VERSION__"

    def render_agent_prompt(self, version: str) -> str:
        return self.prompt_template.replace(self.version_placeholder, version)

    def render_human_markdown(self, version: str) -> str:
        return self.render_agent_prompt(version)

    def to_skill_info(self, version: str) -> SkillInfo:
        return SkillInfo(
            name=self.name,
            description=self.description,
            summary=self.summary,
            user_invocable=self.user_invocable,
            prompt=self.render_agent_prompt(version),
            source=SkillSource.BUILTIN,
            scope=SkillScope.BUILTIN,
        )
