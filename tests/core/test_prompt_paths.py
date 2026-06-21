from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from vibe.core.prompts import load_prompt


@pytest.fixture(autouse=True)
def _harness():
    # load_prompt requires an initialized HarnessFilesManager.
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    yield
    reset_harness_files_manager()


def test_load_prompt_reads_from_extra_dirs_overriding_builtin(tmp_path: Path) -> None:
    # prompt_paths (extra_dirs) must be searched before builtins, so a
    # user/plugin can override a builtin prompt by stem without forking.
    d = tmp_path / "myprompts"
    d.mkdir()
    (d / "compact.md").write_text("CUSTOM COMPACT PROMPT BODY")

    out = load_prompt(
        "compact",
        setting_name="compaction_prompt_id",
        builtins={"compact": Path("/nonexistent/builtin/compact.md")},
        extra_dirs=[d],
    )
    assert out == "CUSTOM COMPACT PROMPT BODY"


def test_load_prompt_falls_back_to_builtin_when_extra_dirs_have_no_match(
    tmp_path: Path,
) -> None:
    builtin = tmp_path / "builtin.md"
    builtin.write_text("BUILTIN")
    d = tmp_path / "empty"
    d.mkdir()

    out = load_prompt(
        "anything",
        setting_name="x",
        builtins={"anything": builtin},
        extra_dirs=[d],
    )
    assert out == "BUILTIN"


def test_load_prompt_missing_raises_with_available_ids(tmp_path: Path) -> None:
    from vibe.core.prompts import MissingPromptFileError

    d = tmp_path / "prompts"
    d.mkdir()
    (d / "alpha.md").write_text("a")
    (d / "beta.md").write_text("b")

    with pytest.raises(MissingPromptFileError) as ei:
        load_prompt(
            "nope",
            setting_name="x",
            builtins={},
            extra_dirs=[d],
        )
    msg = str(ei.value)
    assert "alpha" in msg and "beta" in msg
