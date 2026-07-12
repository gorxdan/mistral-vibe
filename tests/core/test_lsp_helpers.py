from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from vibe.core.tools.base import ToolError
from vibe.core.tools.builtins.lsp import Lsp, LspArgs, LspConfig, LspOperation, LspState
from vibe.core.tools.ui import ToolCallDisplay


def _tool() -> Lsp:
    return Lsp(config_getter=lambda: LspConfig(), state=LspState())


# --------------------------------------------------------------------------- #
# Pure formatting / extraction helpers                                        #
# --------------------------------------------------------------------------- #


def test_extract_markup_string() -> None:
    assert Lsp._extract_markup("hello") == "hello"


def test_extract_markup_dict_with_value() -> None:
    assert Lsp._extract_markup({"value": "val"}) == "val"


def test_extract_markup_dict_with_kind_only() -> None:
    assert Lsp._extract_markup({"kind": "markdown"}) == ""


def test_extract_markup_list_of_strings_and_dicts() -> None:
    result = Lsp._extract_markup(["line1", {"value": "line2"}])
    assert "line1" in result and "line2" in result


def test_extract_markup_fallback_for_non_str() -> None:
    assert Lsp._extract_markup(42) == "42"


def test_as_location_list_none_returns_empty() -> None:
    assert Lsp._as_location_list(None) == []


def test_as_location_list_single_uri_dict() -> None:
    loc = {"uri": "file:///x", "range": {"start": {"line": 0}}}
    result = Lsp._as_location_list(loc)
    assert len(result) == 1 and result[0]["uri"] == "file:///x"


def test_as_location_list_target_uri_shape() -> None:
    raw = {
        "targetUri": "file:///def.py",
        "targetSelectionRange": {"start": {"line": 1, "character": 0}},
    }
    result = Lsp._as_location_list(raw)
    assert len(result) == 1
    assert result[0]["uri"] == "file:///def.py"


def test_as_location_list_list_of_mixed() -> None:
    raw = [{"uri": "file:///a"}, {"targetUri": "file:///b"}]
    result = Lsp._as_location_list(raw)
    assert len(result) == 2


def test_as_location_list_empty_dict_returns_empty() -> None:
    assert Lsp._as_location_list({}) == []


# --------------------------------------------------------------------------- #
# _format_* empty branches                                                    #
# --------------------------------------------------------------------------- #


def test_format_locations_empty() -> None:
    result = _tool()._format_locations("References", [])
    assert "none found" in result.summary


def test_format_locations_with_items() -> None:
    raw = [{"uri": "file:///x.py", "range": {"start": {"line": 4, "character": 2}}}]
    result = _tool()._format_locations("References", raw)
    assert "References (1)" in result.summary
    assert "x.py:5:3" in result.summary


def test_format_locations_reports_truncation() -> None:
    raw = [
        {
            "uri": f"file:///item-{index}.py",
            "range": {"start": {"line": index, "character": 0}},
        }
        for index in range(51)
    ]

    result = _tool()._format_locations("References", raw)

    assert len(result.locations) == 50
    assert result.total_count == 51
    assert result.returned_count == 50
    assert result.was_truncated
    assert "1 omitted" in result.summary


def test_format_hover_empty() -> None:
    result = _tool()._format_hover(None)
    assert "No hover information" in result.summary


def test_format_hover_with_content() -> None:
    result = _tool()._format_hover({"contents": "type info"})
    assert "type info" in result.summary


def test_format_symbols_empty() -> None:
    result = _tool()._format_symbols("Doc", [])
    assert "none found" in result.summary


def test_format_symbols_symbol_information() -> None:
    raw = [
        {
            "name": "foo",
            "containerName": "Bar",
            "location": {"uri": "file:///x.py", "range": {"start": {"line": 3}}},
        }
    ]
    result = _tool()._format_symbols("Doc", raw)
    assert "foo" in result.symbol_names
    assert "Bar" in result.summary


def test_format_symbols_document_symbol_selection_range() -> None:
    raw = [
        {
            "name": "fn",
            "selectionRange": {
                "start": {"line": 5, "character": 4},
                "end": {"line": 5, "character": 6},
            },
        }
    ]
    result = _tool()._format_symbols("Doc", raw)
    assert "fn" in result.symbol_names


def test_format_symbols_reports_truncation() -> None:
    raw = [
        {"name": f"symbol_{index}", "location": {"uri": f"file:///symbol-{index}.py"}}
        for index in range(101)
    ]

    result = _tool()._format_symbols("Symbols", raw)

    assert len(result.symbol_names) == 100
    assert result.total_count == 101
    assert result.returned_count == 100
    assert result.was_truncated
    assert "1 omitted" in result.summary


def test_format_call_items_empty() -> None:
    result = _tool()._format_call_items("Incoming", [])
    assert "none at position" in result.summary


def test_format_call_items_with_data_uri_fallback() -> None:
    raw = [
        {
            "name": "caller",
            "data": {"uri": "file:///y.py"},
            "range": {"start": {"line": 9}},
        }
    ]
    result = _tool()._format_call_items("Incoming", raw)
    assert "caller" in result.summary
    assert "y.py" in result.summary


def test_format_call_items_reports_truncation() -> None:
    raw = [
        {
            "name": f"caller_{index}",
            "uri": f"file:///caller-{index}.py",
            "range": {"start": {"line": index, "character": 0}},
        }
        for index in range(51)
    ]

    result = _tool()._format_call_items("Incoming", raw)

    assert len(result.locations) == 50
    assert result.total_count == 51
    assert result.returned_count == 50
    assert result.was_truncated
    assert "1 omitted" in result.summary


# --------------------------------------------------------------------------- #
# _symbol_rank                                                                #
# --------------------------------------------------------------------------- #


def test_symbol_rank_exact_prefix_substring_no_match() -> None:
    assert Lsp._symbol_rank({"name": "BaseModel"}, "basemodel")[0] == 0
    assert Lsp._symbol_rank({"name": "BaseModelHelper"}, "basemodel")[0] == 1
    assert Lsp._symbol_rank({"name": "myBaseModel"}, "basemodel")[0] == 2
    assert Lsp._symbol_rank({"name": "other"}, "basemodel")[0] == 3


def test_symbol_rank_test_names_penalized() -> None:
    tier, _ = Lsp._symbol_rank({"name": "test_base"}, "base")
    assert tier >= 10


# --------------------------------------------------------------------------- #
# _resolve_path validation                                                    #
# --------------------------------------------------------------------------- #


def test_resolve_path_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="cannot be empty"):
        _tool()._resolve_path("   ")


def test_resolve_path_nonexistent_raises(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="File not found"):
        _tool()._resolve_path(str(tmp_path / "nope.py"))


def test_resolve_path_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="directory"):
        _tool()._resolve_path(str(tmp_path))


def test_resolve_path_valid_file(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("x = 1")
    resolved = _tool()._resolve_path(str(f))
    assert resolved == str(f.resolve())


def test_resolve_path_relative_resolved_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = tmp_path / "rel.py"
    f.write_text("y = 2")
    monkeypatch.chdir(tmp_path)
    resolved = _tool()._resolve_path("rel.py")
    assert resolved == str(f.resolve())


# --------------------------------------------------------------------------- #
# _filter_gitignored + git helpers (async, process-mocked)                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_filter_gitignored_empty_returns_immediately() -> None:
    assert await _tool()._filter_gitignored([]) == []


@pytest.mark.asyncio
async def test_repo_toplevel_git_unavailable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError("no git")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    t = _tool()
    assert await t._repo_toplevel(Path.cwd()) is None


@pytest.mark.asyncio
async def test_check_ignore_nonexistent_path_not_ignored() -> None:
    t = _tool()
    verdicts = await t._check_ignore(Path.cwd(), None, ["/nonexistent/path"])
    assert verdicts == [False]


@pytest.mark.asyncio
async def test_check_ignore_git_not_found_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError("no git")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
    f = Path(__file__)
    t = _tool()
    verdicts = await t._check_ignore(Path.cwd(), Path.cwd(), [str(f)])
    assert verdicts == [False]


# --------------------------------------------------------------------------- #
# Class-level helpers: is_available, resolve_permission, UI data              #
# --------------------------------------------------------------------------- #


def test_is_available_no_config_returns_true() -> None:
    assert Lsp.is_available(None) is True


def test_is_available_with_installed_component() -> None:
    from tests.conftest import build_test_vibe_config

    cfg = build_test_vibe_config(installed_components=["lsp"])
    assert Lsp.is_available(cfg) is True
    cfg2 = build_test_vibe_config(installed_components=[])
    assert Lsp.is_available(cfg2) is False


def test_resolve_permission_returns_config_permission() -> None:
    ctx = _tool().resolve_permission(LspArgs(operation=LspOperation.HOVER))
    assert ctx is not None


def test_format_call_display_with_file() -> None:
    disp = Lsp.format_call_display(
        LspArgs(operation=LspOperation.HOVER, file_path="/x.py")
    )
    assert isinstance(disp, ToolCallDisplay)
    assert "hover" in disp.summary and "/x.py" in disp.summary


def test_format_call_display_workspace() -> None:
    disp = Lsp.format_call_display(
        LspArgs(operation=LspOperation.WORKSPACE_SYMBOL, file_path=None)
    )
    assert "(workspace)" in disp.summary


def test_get_status_text() -> None:
    assert "language server" in Lsp.get_status_text().lower()


def test_call_hierarchy_method_fallback_does_not_promise_equivalent_graph() -> None:
    hint = Lsp._method_not_found_hint(LspOperation.INCOMING_CALLS)

    assert "same caller/callee info" not in hint
    assert "usages" in hint


# --------------------------------------------------------------------------- #
# _path_within                                                                #
# --------------------------------------------------------------------------- #


def test_path_within_true() -> None:
    assert Lsp._path_within(Path("/root"), Path("/root/sub/file.py")) is True


def test_path_within_false() -> None:
    assert Lsp._path_within(Path("/root"), Path("/other/file.py")) is False
