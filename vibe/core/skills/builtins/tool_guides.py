from __future__ import annotations

from pathlib import Path

from vibe.core.skills.models import SkillInfo, SkillScope, SkillSource
from vibe.core.utils.io import read_safe

# Single source of truth: the extended guides live in prompts/*_guide.md next
# to each tool's slimmed inline prompt, loaded on demand via this skill.
_PROMPTS = Path(__file__).resolve().parents[2] / "tools" / "builtins" / "prompts"
_PROMPT = "\n\n---\n\n".join(
    read_safe(_PROMPTS / name).text
    for name in ("todo_guide.md", "ask_user_question_guide.md", "websearch_guide.md")
)

SKILL = SkillInfo(
    name="tool-guides",
    description=(
        "Extended usage guides and worked examples for the todo, "
        "ask_user_question, and web_search tools. The inline tool notes cover "
        "the essential rules; load this only when you need the detailed "
        "conventions or a full example."
    ),
    summary=(
        "Extended todo / ask_user_question / web_search conventions + examples "
        "(essentials are already inline)."
    ),
    user_invocable=False,
    prompt=_PROMPT,
    source=SkillSource.BUILTIN,
    scope=SkillScope.BUILTIN,
)
