from __future__ import annotations

import pytest

from tests.mock.utils import collect_result
from vibe.core.lsp import _adherence as adherence
from vibe.core.tools.base import BaseToolState
from vibe.core.tools.builtins.grep import Grep, GrepArgs, GrepToolConfig


@pytest.fixture(autouse=True)
def _reset_adherence():
    adherence.reset_for_test()
    yield
    adherence.reset_for_test()


def test_record_symbol_grep_miss_increments_counter():
    n1 = adherence.record_symbol_grep_miss()
    n2 = adherence.record_symbol_grep_miss()
    snap = adherence.snapshot()
    assert n1 == 1 and n2 == 2
    assert snap["symbol_grep_miss"] == 2
    assert snap["consecutive_symbol_grep_miss"] == 2
    assert snap["lsp_call"] == 0


def test_record_lsp_call_increments_counter_and_resets_consecutive():
    adherence.record_symbol_grep_miss()
    adherence.record_symbol_grep_miss()
    adherence.record_lsp_call("hover")
    adherence.record_lsp_call("find_references")
    snap = adherence.snapshot()
    assert snap["lsp_call"] == 2
    assert snap["symbol_grep_miss"] == 2
    assert snap["consecutive_symbol_grep_miss"] == 0


def test_snapshot_returns_copy():
    adherence.record_symbol_grep_miss()
    snap = adherence.snapshot()
    snap["symbol_grep_miss"] = 999
    # Mutating the snapshot does not affect the module counters.
    assert adherence.snapshot()["symbol_grep_miss"] == 1


def test_configure_disabled_silences_emit(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(adherence, "_build_handler", lambda: calls.append("built"))
    adherence.configure(enabled=False)
    adherence.record_symbol_grep_miss()
    adherence.record_lsp_call("hover")
    assert calls == []
    assert adherence.snapshot() == {
        "symbol_grep_miss": 1,
        "lsp_call": 1,
        "consecutive_symbol_grep_miss": 0,
    }


def test_configure_reenable_restores_emit(tmp_path, monkeypatch):
    monkeypatch.setattr(adherence.LOG_DIR, "_resolver", lambda: tmp_path)
    adherence.configure(enabled=False)
    adherence.record_symbol_grep_miss()
    assert not (tmp_path / "vibe-adherence.log").exists()
    adherence.configure(enabled=True)
    adherence.record_symbol_grep_miss()
    assert "symbol_grep" in (tmp_path / "vibe-adherence.log").read_text()


def test_default_unconfigured_is_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(adherence.LOG_DIR, "_resolver", lambda: tmp_path)
    adherence.record_lsp_call("hover")
    assert "op=hover" in (tmp_path / "vibe-adherence.log").read_text()


def test_reset_clears_counters():
    adherence.record_symbol_grep_miss()
    adherence.record_lsp_call("hover")
    adherence.reset_for_test()
    assert adherence.snapshot() == {
        "symbol_grep_miss": 0,
        "lsp_call": 0,
        "consecutive_symbol_grep_miss": 0,
    }


def test_symbol_grep_hint_bare_identifier_prefers_workspace_symbol():
    hint = adherence.symbol_grep_hint("FooBar", consecutive=1)
    assert hint.startswith("NOTE:")
    assert "workspace_symbol" in hint
    assert "FooBar" in hint
    assert "go_to_definition" not in hint


def test_symbol_grep_hint_escalates_after_threshold():
    soft = adherence.symbol_grep_hint("FooBar", consecutive=1)
    hard = adherence.symbol_grep_hint("FooBar", consecutive=adherence.ESCALATE_AFTER)
    assert soft.startswith("NOTE:")
    assert hard.startswith("ESCALATION:")
    assert "Stop using grep for symbols" in hard
    assert "workspace_symbol" in hard


def test_should_escalate_tracks_consecutive_misses():
    assert not adherence.should_escalate_symbol_grep()
    adherence.record_symbol_grep_miss()
    assert not adherence.should_escalate_symbol_grep()
    adherence.record_symbol_grep_miss()
    assert adherence.should_escalate_symbol_grep()
    adherence.record_lsp_call("workspace_symbol")
    assert not adherence.should_escalate_symbol_grep()


@pytest.mark.asyncio
async def test_symbol_grep_while_lsp_available_records_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr("vibe.core.tools.builtins.grep._lsp_available", lambda: True)
    grep = Grep(config_getter=lambda: GrepToolConfig(), state=BaseToolState())

    await collect_result(grep.run(GrepArgs(pattern="FooBar")))

    assert adherence.snapshot()["symbol_grep_miss"] == 1


@pytest.mark.asyncio
async def test_pipe_joined_symbol_grep_records_miss(tmp_path, monkeypatch):
    # Pipe-joined alts ("foo|bar|baz") = common alias-hunt form that evades
    # bare-identifier detection.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.py").write_text("def validate():\n    pass\n")
    monkeypatch.setattr("vibe.core.tools.builtins.grep._lsp_available", lambda: True)
    grep = Grep(config_getter=lambda: GrepToolConfig(), state=BaseToolState())

    await collect_result(
        grep.run(GrepArgs(pattern="validate_schema|SchemaValidator|validate"))
    )

    assert adherence.snapshot()["symbol_grep_miss"] == 1


@pytest.mark.asyncio
async def test_symbol_grep_hint_is_directive_note(tmp_path, monkeypatch):
    # Hint must be directive "NOTE:", not soft "looks like".
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr("vibe.core.tools.builtins.grep._lsp_available", lambda: True)
    grep = Grep(config_getter=lambda: GrepToolConfig(), state=BaseToolState())

    result = await collect_result(grep.run(GrepArgs(pattern="FooBar")))

    assert result._hint is not None
    assert result._hint.startswith("NOTE:")
    assert "lsp" in result._hint
    assert "workspace_symbol" in result._hint


@pytest.mark.asyncio
async def test_non_symbol_grep_does_not_record_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.py").write_text("error: boom\n")
    monkeypatch.setattr("vibe.core.tools.builtins.grep._lsp_available", lambda: True)
    grep = Grep(config_getter=lambda: GrepToolConfig(), state=BaseToolState())

    await collect_result(grep.run(GrepArgs(pattern="error: boom")))

    assert adherence.snapshot()["symbol_grep_miss"] == 0


@pytest.mark.asyncio
async def test_symbol_grep_when_lsp_unavailable_does_not_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr("vibe.core.tools.builtins.grep._lsp_available", lambda: False)
    grep = Grep(config_getter=lambda: GrepToolConfig(), state=BaseToolState())

    await collect_result(grep.run(GrepArgs(pattern="FooBar")))

    assert adherence.snapshot()["symbol_grep_miss"] == 0
