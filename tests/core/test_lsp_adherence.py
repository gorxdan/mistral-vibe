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
    adherence.record_symbol_grep_miss()
    adherence.record_symbol_grep_miss()
    snap = adherence.snapshot()
    assert snap["symbol_grep_miss"] == 2
    assert snap["lsp_call"] == 0


def test_record_lsp_call_increments_counter():
    adherence.record_lsp_call("hover")
    adherence.record_lsp_call("find_references")
    snap = adherence.snapshot()
    assert snap["lsp_call"] == 2
    assert snap["symbol_grep_miss"] == 0


def test_snapshot_returns_copy():
    adherence.record_symbol_grep_miss()
    snap = adherence.snapshot()
    snap["symbol_grep_miss"] = 999
    # Mutating the snapshot does not affect the module counters.
    assert adherence.snapshot()["symbol_grep_miss"] == 1


def test_reset_clears_counters():
    adherence.record_symbol_grep_miss()
    adherence.record_lsp_call("hover")
    adherence.reset_for_test()
    assert adherence.snapshot() == {"symbol_grep_miss": 0, "lsp_call": 0}


@pytest.mark.asyncio
async def test_symbol_grep_while_lsp_available_records_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test.py").write_text("def FooBar():\n    pass\n")
    monkeypatch.setattr("vibe.core.tools.builtins.grep._lsp_available", lambda: True)
    grep = Grep(config_getter=lambda: GrepToolConfig(), state=BaseToolState())

    await collect_result(grep.run(GrepArgs(pattern="FooBar")))

    assert adherence.snapshot()["symbol_grep_miss"] == 1


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
