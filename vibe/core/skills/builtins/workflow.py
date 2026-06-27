from __future__ import annotations

from pathlib import Path

from vibe.core.skills.models import SkillInfo, SkillScope, SkillSource

# Single source of truth: the workflow authoring guide lives in the
# launch_workflow tool's prompt file. It is loaded on demand via this skill
# instead of being injected into every system prompt (~3.2k tokens), since the
# overwhelming majority of turns never author a workflow.
_GUIDE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "builtins"
    / "prompts"
    / "launch_workflow.md"
)
_PROMPT = _GUIDE_PATH.read_text(encoding="utf-8")


SKILL = SkillInfo(
    name="workflow-authoring",
    description=(
        "Load BEFORE writing a workflow script for the `launch_workflow` tool. "
        "Single source of truth for the script API "
        "(agent/parallel/pipeline/phase/log/budget/args + synthesis helpers), "
        "the sandbox rules (allowlisted imports, no asyncio, no str.format), "
        "launch semantics, and concurrency/rate-limit recovery. Do not author a "
        "workflow from memory — load this, then write the script."
    ),
    summary=(
        "Workflow script API + sandbox rules — load before authoring a "
        "`launch_workflow` script (don't write one from memory)."
    ),
    user_invocable=False,
    prompt=_PROMPT,
    source=SkillSource.BUILTIN,
    scope=SkillScope.BUILTIN,
)
