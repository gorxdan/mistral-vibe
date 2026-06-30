from __future__ import annotations

from tests.conftest import build_test_vibe_config
from vibe.core.llm.format import APIToolFormatHandler, _trim_description
from vibe.core.tools.manager import ToolManager


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
    assert any(
        len(by_name_trim[n]) < len(by_name_full[n]) for n in by_name_full
    )
    for n in by_name_full:
        assert len(by_name_trim[n]) <= max(len(by_name_full[n]), 121)
    # Param schemas are untouched by trimming.
    params_full = {t.function.name: t.function.parameters for t in full}
    params_trim = {t.function.name: t.function.parameters for t in trimmed}
    assert params_full == params_trim
