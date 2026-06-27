from __future__ import annotations

import pytest

from vibe import VIBE_ROOT
from vibe.core.experiments.active import DEFAULT_VARIANTS, ExperimentName
from vibe.core.prompts import (
    UtilityPrompt,
    load_system_prompt,
    prompt_variant,
    set_prompt_variant,
)
from vibe.core.utils.io import read_safe

_SYS_VERBOSE = VIBE_ROOT / "core" / "prompts" / "_verbose"
_TOOL_VERBOSE = VIBE_ROOT / "core" / "tools" / "builtins" / "prompts" / "_verbose"


@pytest.fixture(autouse=True)
def _reset_variant():
    # The variant is process-global module state; reset after each test so the
    # default "compressed" arm never leaks into the rest of the suite.
    yield
    set_prompt_variant("compressed")


def test_prompt_compression_experiment_defaults_to_compressed() -> None:
    assert DEFAULT_VARIANTS[ExperimentName.PROMPT_COMPRESSION] == "compressed"
    assert prompt_variant() == "compressed"


def test_unknown_variant_falls_back_to_compressed() -> None:
    set_prompt_variant("nonsense")
    assert prompt_variant() == "compressed"


def test_verbose_arm_serves_precompression_across_all_load_paths() -> None:
    from vibe.core.skills.builtins import workflow as wf
    from vibe.core.tools.builtins.grep import Grep

    # Compressed (default, shipped) baseline.
    coord_c = load_system_prompt("coordinator")  # load_prompt path (profiles)
    agents_c = UtilityPrompt.AGENTS_DOC.read()  # Prompt.read path (utilities)
    grep_c = Grep.get_tool_prompt()  # get_tool_prompt path (tool descriptions)
    wf_c = wf.SKILL.prompt  # workflow-authoring skill loader path

    set_prompt_variant("verbose")
    coord_v = load_system_prompt("coordinator")
    agents_v = UtilityPrompt.AGENTS_DOC.read()
    grep_v = Grep.get_tool_prompt()
    wf_v = wf.SKILL.prompt

    assert grep_c is not None and grep_v is not None  # tool prompt always present

    # Each verbose arm equals the pre-compression copy under _verbose/.
    assert coord_v == read_safe(_SYS_VERBOSE / "coordinator.md").text.strip()
    assert agents_v == read_safe(_SYS_VERBOSE / "agents_doc.md").text.strip()
    assert grep_v == read_safe(_TOOL_VERBOSE / "grep.md").text
    assert wf_v == (_TOOL_VERBOSE / "launch_workflow.md").read_text(encoding="utf-8")

    # ...and it actually differs from the shipped compressed arm (longer).
    for c, v in ((coord_c, coord_v), (agents_c, agents_v), (grep_c, grep_v), (wf_c, wf_v)):
        assert v != c
        assert len(v) > len(c)


def test_reset_to_compressed_restores_shipped_prompts() -> None:
    from vibe.core.skills.builtins import workflow as wf
    from vibe.core.tools.builtins.grep import Grep

    coord_c, grep_c, wf_c = (
        load_system_prompt("coordinator"),
        Grep.get_tool_prompt(),
        wf.SKILL.prompt,
    )
    set_prompt_variant("verbose")
    set_prompt_variant("compressed")
    assert load_system_prompt("coordinator") == coord_c
    assert Grep.get_tool_prompt() == grep_c
    assert wf.SKILL.prompt == wf_c
