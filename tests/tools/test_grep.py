from __future__ import annotations

import shutil
import types
from typing import Any, cast

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.grep import (
    Grep,
    GrepArgs,
    GrepBackend,
    GrepOutputMode,
    GrepResult,
    GrepToolConfig,
    _is_symbol_shaped,
)


@pytest.fixture
def grep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig()
    return Grep(config_getter=lambda: config, state=BaseToolState())


@pytest.fixture
def grep_gnu_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_which = shutil.which

    def mock_which(cmd):
        if cmd == "rg":
            return None
        return original_which(cmd)

    monkeypatch.setattr("shutil.which", mock_which)
    config = GrepToolConfig()
    return Grep(config_getter=lambda: config, state=BaseToolState())


def test_vibe_data_excludes_anchored_when_root_is_vibe_home(
    grep, tmp_path, monkeypatch
):
    vibe_home = tmp_path / "dotvibe"
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep.VIBE_HOME", types.SimpleNamespace(path=vibe_home)
    )
    assert grep._vibe_data_excludes(str(vibe_home)) == ["!/worktrees/", "!/logs/"]


def test_vibe_data_excludes_relative_when_root_is_parent(grep, tmp_path, monkeypatch):
    vibe_home = tmp_path / "dotvibe"
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep.VIBE_HOME", types.SimpleNamespace(path=vibe_home)
    )
    assert grep._vibe_data_excludes(str(tmp_path)) == [
        "!/dotvibe/worktrees/",
        "!/dotvibe/logs/",
    ]


def test_vibe_data_excludes_empty_inside_worktree(grep, tmp_path, monkeypatch):
    vibe_home = tmp_path / "dotvibe"
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep.VIBE_HOME", types.SimpleNamespace(path=vibe_home)
    )
    inside = vibe_home / "worktrees" / "repo"
    assert grep._vibe_data_excludes(str(inside)) == []


def test_detects_ripgrep_when_available(grep):
    if shutil.which("rg"):
        assert grep._detect_backend() == GrepBackend.RIPGREP


def test_falls_back_to_gnu_grep(grep, monkeypatch):
    original_which = shutil.which

    def mock_which(cmd):
        if cmd == "rg":
            return None
        return original_which(cmd)

    monkeypatch.setattr("shutil.which", mock_which)

    if shutil.which("grep"):
        assert grep._detect_backend() == GrepBackend.GNU_GREP


def test_raises_error_if_no_grep_available(grep, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)

    with pytest.raises(ToolError) as err:
        grep._detect_backend()

    assert "Neither ripgrep (rg) nor grep is installed" in str(err.value)


@pytest.mark.asyncio
async def test_finds_pattern_in_file(grep, tmp_path):
    (tmp_path / "test.py").write_text("def hello():\n    print('world')\n")

    result = await collect_result(grep.run(GrepArgs(pattern="hello")))

    assert result.match_count == 1
    assert "hello" in result.matches
    assert "test.py" in result.matches
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_finds_multiple_matches(grep, tmp_path):
    (tmp_path / "test.py").write_text("foo\nbar\nfoo\nbaz\nfoo\n")

    result = await collect_result(grep.run(GrepArgs(pattern="foo")))

    assert result.match_count == 3
    assert result.matches.count("foo") == 3
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_returns_empty_on_no_matches(grep, tmp_path):
    (tmp_path / "test.py").write_text("def hello():\n    pass\n")

    result = await collect_result(grep.run(GrepArgs(pattern="nonexistent")))

    assert result.match_count == 0
    assert result.matches == ""
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_preserves_accents_when_matching_latin1_encoded_file(grep, tmp_path):
    (tmp_path / "menu.txt").write_bytes("café au lait\nthé glacé\n".encode("latin-1"))

    result = await collect_result(
        grep.run(GrepArgs(pattern="caf"))  # typos:disable-line
    )

    assert result.match_count == 1
    assert "\ufffd" not in result.matches
    assert "café au lait" in result.matches


@pytest.mark.asyncio
async def test_fails_with_empty_pattern(grep):
    with pytest.raises(ToolError) as err:
        await collect_result(grep.run(GrepArgs(pattern="")))

    assert "Empty search pattern" in str(err.value)


@pytest.mark.asyncio
async def test_fails_with_nonexistent_path(grep):
    with pytest.raises(ToolError) as err:
        await collect_result(grep.run(GrepArgs(pattern="test", path="nonexistent")))

    assert "Path does not exist" in str(err.value)


@pytest.mark.asyncio
async def test_searches_in_specific_path(grep, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "test.py").write_text("match here\n")
    (tmp_path / "other.py").write_text("match here too\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match", path="subdir")))

    assert result.match_count == 1
    assert "subdir" in result.matches and "test.py" in result.matches
    assert "other.py" not in result.matches


@pytest.mark.asyncio
async def test_truncates_to_max_matches(grep, tmp_path):
    (tmp_path / "test.py").write_text("\n".join(f"line {i}" for i in range(200)))

    result = await collect_result(grep.run(GrepArgs(pattern="line", max_matches=50)))

    assert result.match_count == 50
    assert result.was_truncated


@pytest.mark.asyncio
async def test_truncates_to_max_output_bytes(grep, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig(max_output_bytes=100)
    grep_tool = Grep(config_getter=lambda: config, state=BaseToolState())
    (tmp_path / "test.py").write_text("\n".join("x" * 100 for _ in range(10)))

    result = await collect_result(grep_tool.run(GrepArgs(pattern="x")))

    assert len(result.matches) <= 100
    assert result.was_truncated


@pytest.mark.asyncio
async def test_respects_default_ignore_patterns(grep, tmp_path):
    (tmp_path / "included.py").write_text("match\n")
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "excluded.js").write_text("match\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match")))

    assert "included.py" in result.matches
    assert "excluded.js" not in result.matches


@pytest.mark.asyncio
async def test_respects_vibeignore_file(grep, tmp_path):
    (tmp_path / ".vibeignore").write_text("custom_dir/\n*.tmp\n")
    custom_dir = tmp_path / "custom_dir"
    custom_dir.mkdir()
    (custom_dir / "excluded.py").write_text("match\n")
    (tmp_path / "excluded.tmp").write_text("match\n")
    (tmp_path / "included.py").write_text("match\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match")))

    assert "included.py" in result.matches
    assert "excluded.py" not in result.matches
    assert "excluded.tmp" not in result.matches


@pytest.mark.asyncio
async def test_ignores_comments_in_vibeignore(grep, tmp_path):
    (tmp_path / ".vibeignore").write_text("# comment\npattern/\n# another comment\n")
    (tmp_path / "file.py").write_text("match\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match")))

    assert result.match_count >= 1


@pytest.mark.asyncio
async def test_uses_effective_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig()
    grep_tool = Grep(config_getter=lambda: config, state=BaseToolState())
    (tmp_path / "test.py").write_text("match\n")

    result = await collect_result(grep_tool.run(GrepArgs(pattern="match", path=".")))

    assert result.match_count == 1
    assert "test.py" in result.matches


@pytest.mark.asyncio
async def test_single_file_match_includes_filename_in_output(grep, tmp_path):
    # Without --with-filename / -H, rg and grep omit the filename when
    # searching a single file, causing GrepMatch.from_output_line to
    # misinterpret the line number as a path. See VIBE-2772.
    (tmp_path / "only.py").write_text("hit one\nnope\nhit two\n")

    result = await collect_result(grep.run(GrepArgs(pattern="hit", path="only.py")))

    assert result.match_count == 2
    for parsed in result.parsed_matches:
        assert parsed.path.endswith("only.py")
        assert parsed.line is not None


@pytest.mark.skipif(not shutil.which("grep"), reason="GNU grep not available")
class TestGnuGrepBackend:
    @pytest.mark.asyncio
    async def test_finds_pattern_in_file(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("def hello():\n    print('world')\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="hello")))

        assert result.match_count == 1
        assert "hello" in result.matches
        assert "test.py" in result.matches

    @pytest.mark.asyncio
    async def test_finds_multiple_matches(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("foo\nbar\nfoo\nbaz\nfoo\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="foo")))

        assert result.match_count == 3
        assert result.matches.count("foo") == 3

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_matches(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("def hello():\n    pass\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="nonexistent"))
        )

        assert result.match_count == 0
        assert result.matches == ""

    @pytest.mark.asyncio
    async def test_case_insensitive_for_lowercase_pattern(
        self, grep_gnu_only, tmp_path
    ):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="hello")))

        assert result.match_count == 3

    @pytest.mark.asyncio
    async def test_case_sensitive_for_mixed_case_pattern(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="Hello")))

        assert result.match_count == 1

    @pytest.mark.asyncio
    async def test_respects_exclude_patterns(self, grep_gnu_only, tmp_path):
        (tmp_path / "included.py").write_text("match\n")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "excluded.js").write_text("match\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="match")))

        assert "included.py" in result.matches
        assert "excluded.js" not in result.matches

    @pytest.mark.asyncio
    async def test_searches_in_specific_path(self, grep_gnu_only, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "test.py").write_text("match here\n")
        (tmp_path / "other.py").write_text("match here too\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="match", path="subdir"))
        )

        assert result.match_count == 1
        assert "other.py" not in result.matches

    @pytest.mark.asyncio
    async def test_respects_vibeignore_file(self, grep_gnu_only, tmp_path):
        (tmp_path / ".vibeignore").write_text("custom_dir/\n*.tmp\n")
        custom_dir = tmp_path / "custom_dir"
        custom_dir.mkdir()
        (custom_dir / "excluded.py").write_text("match\n")
        (tmp_path / "excluded.tmp").write_text("match\n")
        (tmp_path / "included.py").write_text("match\n")

        result = await collect_result(grep_gnu_only.run(GrepArgs(pattern="match")))

        assert "included.py" in result.matches
        assert "excluded.py" not in result.matches
        assert "excluded.tmp" not in result.matches

    @pytest.mark.asyncio
    async def test_truncates_to_max_matches(self, grep_gnu_only, tmp_path):
        (tmp_path / "test.py").write_text("\n".join(f"line {i}" for i in range(200)))

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="line", max_matches=50))
        )

        assert result.match_count == 50
        assert result.was_truncated


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep not available")
class TestRipgrepBackend:
    @pytest.mark.asyncio
    async def test_smart_case_lowercase_pattern(self, grep, tmp_path):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep.run(GrepArgs(pattern="hello")))

        assert result.match_count == 3

    @pytest.mark.asyncio
    async def test_smart_case_mixed_case_pattern(self, grep, tmp_path):
        (tmp_path / "test.py").write_text("Hello\nHELLO\nhello\n")

        result = await collect_result(grep.run(GrepArgs(pattern="Hello")))

        assert result.match_count == 1

    @pytest.mark.asyncio
    async def test_searches_ignored_files_when_use_default_ignore_false(
        self, grep, tmp_path
    ):
        (tmp_path / ".ignore").write_text("ignored_by_rg/\n")

        ignored_dir = tmp_path / "ignored_by_rg"
        ignored_dir.mkdir()
        (ignored_dir / "file.py").write_text("match\n")
        (tmp_path / "included.py").write_text("match\n")

        result_with_ignore = await collect_result(grep.run(GrepArgs(pattern="match")))
        assert "included.py" in result_with_ignore.matches
        assert "ignored_by_rg" not in result_with_ignore.matches

        result_without_ignore = await collect_result(
            grep.run(GrepArgs(pattern="match", use_default_ignore=False))
        )
        assert "included.py" in result_without_ignore.matches
        assert "ignored_by_rg/file.py" in result_without_ignore.matches


@pytest.mark.asyncio
async def test_content_mode_is_default_and_back_compatible(grep, tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    pass\n")

    result = await collect_result(grep.run(GrepArgs(pattern="hello")))

    assert result.output_mode == GrepOutputMode.CONTENT
    assert result.match_count == 1
    assert result.parsed_matches and result.parsed_matches[0].line == 1


@pytest.mark.asyncio
async def test_output_mode_files_with_matches(grep, tmp_path):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "b.py").write_text("needle here too\n")
    (tmp_path / "c.py").write_text("nothing\n")

    result = await collect_result(
        grep.run(
            GrepArgs(pattern="needle", output_mode=GrepOutputMode.FILES_WITH_MATCHES)
        )
    )

    assert result.match_count == 2
    assert "a.py" in result.matches
    assert "b.py" in result.matches
    assert "c.py" not in result.matches
    assert result.parsed_matches == []


@pytest.mark.asyncio
async def test_output_mode_count(grep, tmp_path):
    (tmp_path / "a.py").write_text("hit\nhit\nhit\n")
    (tmp_path / "b.py").write_text("nope\n")

    result = await collect_result(
        grep.run(GrepArgs(pattern="hit", output_mode=GrepOutputMode.COUNT))
    )

    assert result.match_count == 1
    assert "a.py" in result.matches
    assert "3" in result.matches
    assert "b.py" not in result.matches


@pytest.mark.asyncio
async def test_context_after_includes_following_line(grep, tmp_path):
    (tmp_path / "a.py").write_text("alpha\nNEEDLE\nbravo\ncharlie\n")

    result = await collect_result(grep.run(GrepArgs(pattern="NEEDLE", context_after=1)))

    assert "NEEDLE" in result.matches
    assert "bravo" in result.matches
    assert "charlie" not in result.matches


@pytest.mark.asyncio
async def test_context_before_includes_preceding_line(grep, tmp_path):
    (tmp_path / "a.py").write_text("alpha\nbravo\nNEEDLE\ncharlie\n")

    result = await collect_result(
        grep.run(GrepArgs(pattern="NEEDLE", context_before=1))
    )

    assert "bravo" in result.matches
    assert "alpha" not in result.matches


@pytest.mark.asyncio
async def test_context_around_includes_both_sides(grep, tmp_path):
    (tmp_path / "a.py").write_text("alpha\nbravo\nNEEDLE\ncharlie\ndelta\n")

    result = await collect_result(grep.run(GrepArgs(pattern="NEEDLE", context=1)))

    assert "bravo" in result.matches
    assert "charlie" in result.matches
    assert "alpha" not in result.matches
    assert "delta" not in result.matches


@pytest.mark.asyncio
async def test_context_rejected_in_non_content_mode(grep):
    with pytest.raises(ToolError) as err:
        await collect_result(
            grep.run(
                GrepArgs(
                    pattern="x",
                    output_mode=GrepOutputMode.FILES_WITH_MATCHES,
                    context=2,
                )
            )
        )

    assert "content output mode" in str(err.value)


@pytest.mark.asyncio
async def test_head_limit_caps_output(grep, tmp_path):
    (tmp_path / "a.py").write_text("\n".join("match" for _ in range(20)) + "\n")

    result = await collect_result(grep.run(GrepArgs(pattern="match", head_limit=3)))

    assert result.match_count == 3
    assert result.was_truncated


@pytest.mark.asyncio
async def test_glob_filter_limits_to_matching_files(grep, tmp_path):
    (tmp_path / "a.py").write_text("target\n")
    (tmp_path / "b.js").write_text("target\n")

    result = await collect_result(grep.run(GrepArgs(pattern="target", glob="*.py")))

    assert "a.py" in result.matches
    assert "b.js" not in result.matches


@pytest.mark.asyncio
async def test_case_insensitive_overrides_smart_case(grep, tmp_path):
    (tmp_path / "a.py").write_text("Hello\nHELLO\nhello\n")

    result = await collect_result(
        grep.run(GrepArgs(pattern="Hello", case_insensitive=True))
    )

    assert result.match_count == 3


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep not available")
@pytest.mark.asyncio
async def test_type_filter_ripgrep(grep, tmp_path):
    (tmp_path / "a.py").write_text("target\n")
    (tmp_path / "a.md").write_text("target\n")

    result = await collect_result(grep.run(GrepArgs(pattern="target", type="py")))

    assert "a.py" in result.matches
    assert "a.md" not in result.matches


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep not available")
@pytest.mark.asyncio
async def test_multiline_matches_across_lines(grep, tmp_path):
    (tmp_path / "a.py").write_text("foo\nbar\n")

    without = await collect_result(grep.run(GrepArgs(pattern="foo.bar")))
    assert without.match_count == 0

    with_multiline = await collect_result(
        grep.run(GrepArgs(pattern="foo.bar", multiline=True))
    )
    assert with_multiline.match_count >= 1


@pytest.mark.skipif(not shutil.which("grep"), reason="GNU grep not available")
class TestGnuGrepEnrichment:
    @pytest.mark.asyncio
    async def test_type_raises_on_gnu_backend(self, grep_gnu_only, tmp_path):
        (tmp_path / "a.py").write_text("x\n")

        with pytest.raises(ToolError) as err:
            await collect_result(grep_gnu_only.run(GrepArgs(pattern="x", type="py")))

        assert "ripgrep" in str(err.value)

    @pytest.mark.asyncio
    async def test_multiline_raises_on_gnu_backend(self, grep_gnu_only, tmp_path):
        (tmp_path / "a.py").write_text("x\n")

        with pytest.raises(ToolError) as err:
            await collect_result(
                grep_gnu_only.run(GrepArgs(pattern="x", multiline=True))
            )

        assert "ripgrep" in str(err.value)

    @pytest.mark.asyncio
    async def test_glob_filter_on_gnu_backend(self, grep_gnu_only, tmp_path):
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.js").write_text("target\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="target", glob="*.py"))
        )

        assert "a.py" in result.matches
        assert "b.js" not in result.matches

    @pytest.mark.asyncio
    async def test_files_with_matches_on_gnu_backend(self, grep_gnu_only, tmp_path):
        (tmp_path / "a.py").write_text("needle\n")
        (tmp_path / "b.py").write_text("nope\n")

        result = await collect_result(
            grep_gnu_only.run(
                GrepArgs(
                    pattern="needle", output_mode=GrepOutputMode.FILES_WITH_MATCHES
                )
            )
        )

        assert result.match_count == 1
        assert "a.py" in result.matches
        assert "b.py" not in result.matches

    @pytest.mark.asyncio
    async def test_count_on_gnu_backend(self, grep_gnu_only, tmp_path):
        (tmp_path / "a.py").write_text("hit\nhit\n")
        (tmp_path / "b.py").write_text("nope\n")

        result = await collect_result(
            grep_gnu_only.run(GrepArgs(pattern="hit", output_mode=GrepOutputMode.COUNT))
        )

        assert result.match_count == 1
        assert "a.py" in result.matches
        assert "2" in result.matches
        assert "b.py" not in result.matches


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep not available")
@pytest.mark.asyncio
async def test_multiline_regex_error_hints_at_multiline_arg(grep):
    with pytest.raises(ToolError) as err:
        await collect_result(grep.run(GrepArgs(pattern=r"foo\nbar")))
    assert "`multiline` argument" in str(err.value)


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep not available")
@pytest.mark.asyncio
async def test_lookaround_regex_error_hints_unsupported(grep):
    with pytest.raises(ToolError) as err:
        await collect_result(grep.run(GrepArgs(pattern="(?<=foo)bar")))
    msg = str(err.value).lower()
    assert "look-around" in msg
    assert "rewrite" in msg or "unsupported" in msg


# --- lsp symbol-nudge on the success path ---


def test_is_symbol_shaped_classifies_patterns():
    # Bare identifiers (snake_case, camelCase, PascalCase, CONSTANTS) qualify.
    assert _is_symbol_shaped("FooBar")
    assert _is_symbol_shaped("find_references")
    assert _is_symbol_shaped("_private")
    assert _is_symbol_shaped("CONSTANT_VALUE")
    # Regex metacharacters, operators, dots, whitespace, or too short do not.
    assert not _is_symbol_shaped("foo.bar")
    assert not _is_symbol_shaped("foo bar")
    assert not _is_symbol_shaped("foo(")
    assert not _is_symbol_shaped("x")
    assert not _is_symbol_shaped("a.b.c")
    assert not _is_symbol_shaped("")


def test_hint_private_attr_excluded_from_model_dump():
    result = GrepResult(matches="", match_count=0, was_truncated=False)
    result._hint = "some nudge"
    dumped = result.model_dump()
    # Private attr stays out of the serialized fields — no "hint:" clutter.
    assert "_hint" not in dumped
    assert "hint" not in dumped


@pytest.mark.asyncio
async def test_symbol_grep_sets_hint_when_lsp_available(grep, tmp_path, monkeypatch):
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep._lsp_available", lambda *_args, **_kwargs: True
    )

    result = await collect_result(grep.run(GrepArgs(pattern="FooBar")))

    assert result._hint
    assert "FooBar" in result._hint
    assert "lsp" in result._hint
    assert "workspace_symbol" in result._hint
    # get_result_extra surfaces it for the model-visible text.
    assert grep is not None
    extra = grep.get_result_extra(result)
    assert extra is not None and "FooBar" in extra


@pytest.mark.asyncio
async def test_non_symbol_grep_sets_no_hint(grep, tmp_path, monkeypatch):
    (tmp_path / "test.py").write_text("error: boom\n")
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep._lsp_available", lambda *_args, **_kwargs: True
    )

    result = await collect_result(grep.run(GrepArgs(pattern="error: boom")))

    assert result._hint == ""
    assert grep.get_result_extra(result) is None


@pytest.mark.asyncio
async def test_symbol_grep_no_hint_when_lsp_unavailable(grep, tmp_path, monkeypatch):
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep._lsp_available", lambda *_args, **_kwargs: False
    )

    result = await collect_result(grep.run(GrepArgs(pattern="FooBar")))

    assert result._hint == ""
    assert grep.get_result_extra(result) is None


@pytest.mark.asyncio
async def test_symbol_grep_escalates_on_second_miss(grep, tmp_path, monkeypatch):
    from vibe.core.lsp import _adherence as adherence

    adherence.reset_for_test()
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep._lsp_available", lambda *_args, **_kwargs: True
    )

    first = await collect_result(grep.run(GrepArgs(pattern="FooBar")))
    second = await collect_result(grep.run(GrepArgs(pattern="FooBar")))

    assert first._hint.startswith("NOTE:")
    assert second._hint.startswith("ESCALATION:")
    assert "workspace_symbol" in second._hint
    assert adherence.snapshot()["consecutive_symbol_grep_miss"] == 2


def test_lsp_availability_prefers_active_manifest(monkeypatch):
    from vibe.core.config import VibeConfig
    from vibe.core.tools.builtins.grep import _lsp_available

    def fail_load(cls, **overrides):
        raise AssertionError("persisted config should not be consulted")

    monkeypatch.setattr(VibeConfig, "load", classmethod(fail_load))
    manager = types.SimpleNamespace(has_running_server_for=lambda **_kwargs: True)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep.get_lsp_manager", lambda: manager
    )

    unavailable = InvokeContext(
        tool_call_id="unavailable",
        tool_manager=types.SimpleNamespace(manifest_tools={"grep": object()}),
    )
    available = InvokeContext(
        tool_call_id="available",
        tool_manager=types.SimpleNamespace(
            manifest_tools={"grep": object(), "lsp": object()}
        ),
    )

    assert not _lsp_available(unavailable)
    assert _lsp_available(available)


def test_lsp_availability_requires_live_server_after_persisted_config(monkeypatch):
    from vibe.core.config import VibeConfig
    from vibe.core.tools.builtins.grep import _lsp_available

    config = types.SimpleNamespace(installed_components=["lsp"])
    monkeypatch.setattr(
        VibeConfig, "load", classmethod(lambda cls, **overrides: config)
    )

    monkeypatch.setattr("vibe.core.tools.builtins.grep.get_lsp_manager", lambda: None)
    assert not _lsp_available()
    manager = types.SimpleNamespace(has_running_server_for=lambda **_kwargs: True)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep.get_lsp_manager", lambda: manager
    )
    assert _lsp_available()
    config.installed_components = []
    assert not _lsp_available()


def test_lsp_availability_understands_brace_globs_and_ripgrep_types(monkeypatch):
    from vibe.core.tools.builtins.grep import _lsp_available

    calls: list[dict[str, object]] = []

    def has_running_server_for(**kwargs):
        calls.append(kwargs)
        return True

    manager = types.SimpleNamespace(has_running_server_for=has_running_server_for)
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep.get_lsp_manager", lambda: manager
    )
    ctx = InvokeContext(
        tool_call_id="available",
        tool_manager=types.SimpleNamespace(
            manifest_tools={"grep": object(), "lsp": object()}
        ),
    )

    assert _lsp_available(ctx, args=GrepArgs(pattern="Thing", glob="*.{ts,tsx}"))
    assert calls[-1]["extensions"] == (".ts", ".tsx")
    assert calls[-1]["operation"] == "workspace_symbol"

    assert _lsp_available(ctx, args=GrepArgs(pattern="Thing", type="py"))
    assert calls[-1]["language_id"] == "python"


@pytest.mark.asyncio
async def test_symbol_grep_streak_isolated_by_invoke_context(
    grep, tmp_path, monkeypatch
):
    from vibe.core.lsp import _adherence as adherence

    adherence.reset_for_test()
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep._lsp_available", lambda *_args, **_kwargs: True
    )
    first_ctx = InvokeContext(tool_call_id="first", session_id="first-session")
    second_ctx = InvokeContext(tool_call_id="second", session_id="second-session")

    first = await collect_result(grep.run(GrepArgs(pattern="FooBar"), first_ctx))
    second = await collect_result(grep.run(GrepArgs(pattern="FooBar"), second_ctx))

    assert first._hint.startswith("NOTE:")
    assert second._hint.startswith("NOTE:")
    assert adherence.snapshot(ctx=first_ctx)["consecutive_symbol_grep_miss"] == 1
    assert adherence.snapshot(ctx=second_ctx)["consecutive_symbol_grep_miss"] == 1


@pytest.mark.asyncio
async def test_bound_task_symbol_hint_avoids_workspace_query(
    grep, tmp_path, monkeypatch
):
    class Contract:
        search_exclude_patterns: tuple[str, ...] = ()

        def allows_search_result(self, path):
            return True

    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr(
        "vibe.core.tools.builtins.grep._lsp_available", lambda *_args, **_kwargs: True
    )
    ctx = InvokeContext(
        tool_call_id="bound",
        session_id="bound-session",
        task_contract=cast(Any, Contract()),
    )

    result = await collect_result(grep.run(GrepArgs(pattern="FooBar"), ctx))

    assert result._hint.startswith("NOTE:")
    assert "workspace_symbol" not in result._hint
    assert "find_references" in result._hint
