from __future__ import annotations

from tests.conftest import build_test_vibe_config
from vibe.core.config import ToolManifestConfig
from vibe.core.llm.format import APIToolFormatHandler, _trim_description
from vibe.core.tools.manager import ToolManager
from vibe.core.utils.text import first_sentence

DEFERRED = (
    "manage_memory",
    "schedule",
    "team_message",
    "workflow_status",
    "workflow_stop",
)
STUB_MARKER = "Hidden builtin tools you can activate here:"


def test_trim_short_description_unchanged():
    assert _trim_description("Read a file.", 220) == "Read a file."


def test_trim_long_description_to_first_sentence():
    desc = "Reads a file from disk. " + "Extra clause. " * 40
    out = _trim_description(desc, 220)
    assert out == "Reads a file from disk."
    assert len(out) <= 220


def test_trim_no_sentence_break_cuts_at_word_with_ellipsis():
    desc = "word " * 100  # no period
    out = _trim_description(desc, 50)
    assert out.endswith("…")
    assert len(out) <= 51
    assert " word" in out


def test_get_available_tools_trims_only_when_requested():
    config = build_test_vibe_config(system_prompt_id="tests")
    tm = ToolManager(lambda: config)
    handler = APIToolFormatHandler()

    full = handler.get_available_tools(tm)
    trimmed = handler.get_available_tools(
        tm, trim_descriptions=True, description_max_chars=120
    )

    by_name_full = {t.function.name: t.function.description for t in full}
    by_name_trim = {t.function.name: t.function.description for t in trimmed}
    assert by_name_full.keys() == by_name_trim.keys()
    # Every trimmed description is <= full and capped; at least one is shortened.
    assert any(len(by_name_trim[n]) < len(by_name_full[n]) for n in by_name_full)
    for n in by_name_full:
        assert len(by_name_trim[n]) <= max(len(by_name_full[n]), 121)
    # Param schemas are untouched by trimming.
    params_full = {t.function.name: t.function.parameters for t in full}
    params_trim = {t.function.name: t.function.parameters for t in trimmed}
    assert params_full == params_trim


def _deferral_manager() -> ToolManager:
    config = build_test_vibe_config(
        tool_manifest=ToolManifestConfig(defer_builtin_tools=True)
    )
    return ToolManager(lambda: config)


def test_tool_search_description_gains_sorted_stub_suffix_when_deferral_on():
    tm = _deferral_manager()
    descriptions = {
        t.function.name: t.function.description
        for t in APIToolFormatHandler().get_available_tools(tm)
    }

    desc = descriptions["tool_search"]
    assert STUB_MARKER in desc
    for name in DEFERRED:
        assert f"`{name}`" in desc
    assert "`background`" not in desc

    stubs = tm.deferred_builtin_stubs()
    assert [name for name, _ in stubs] == sorted(name for name, _ in stubs)
    assert all(len(summary) <= 91 for _, summary in stubs)


def test_stub_suffix_survives_description_trim():
    tm = _deferral_manager()
    descriptions = {
        t.function.name: t.function.description
        for t in APIToolFormatHandler().get_available_tools(
            tm, trim_descriptions=True, description_max_chars=120
        )
    }

    desc = descriptions["tool_search"]
    assert STUB_MARKER in desc
    for name in DEFERRED:
        assert f"`{name}`" in desc


def test_no_stub_suffix_when_deferral_off():
    config = build_test_vibe_config(system_prompt_id="tests")
    tm = ToolManager(lambda: config)

    for tool in APIToolFormatHandler().get_available_tools(tm):
        assert STUB_MARKER not in tool.function.description


def test_first_sentence_short_text_unchanged():
    assert first_sentence("Read a file.", 220) == "Read a file."


def test_first_sentence_cuts_to_first_sentence():
    desc = "Reads a file from disk. " + "Extra clause. " * 40
    assert first_sentence(desc, 220) == "Reads a file from disk."


def test_first_sentence_no_break_cuts_at_word_with_ellipsis():
    out = first_sentence("word " * 100, 50)
    assert out.endswith("…")
    assert len(out) <= 51
    assert " word" in out
