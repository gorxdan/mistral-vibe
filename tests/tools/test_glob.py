from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil

import pytest

from tests.mock.utils import collect_result
from vibe.core.tools.base import BaseToolState, ToolError
from vibe.core.tools.builtins.glob import Glob, GlobArgs, GlobBackend, GlobToolConfig


@pytest.fixture
def glob_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GlobToolConfig()
    return Glob(config_getter=lambda: config, state=BaseToolState())


@pytest.fixture
def glob_walk_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_which = shutil.which

    def mock_which(cmd):
        if cmd == "rg":
            return None
        return original_which(cmd)

    monkeypatch.setattr("shutil.which", mock_which)
    config = GlobToolConfig()
    return Glob(config_getter=lambda: config, state=BaseToolState())


@pytest.mark.asyncio
async def test_finds_files_matching_pattern(glob_tool, tmp_path):
    (tmp_path / "a.py").write_text("x\n")
    (tmp_path / "b.txt").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern="**/*.py")))

    assert any("a.py" in p for p in result.paths)
    assert all("b.txt" not in p for p in result.paths)
    assert result.match_count == 1
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_matches_nested_pattern_scoped_to_dir(glob_tool, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.ts").write_text("x\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "util.ts").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern="src/**/*.ts")))

    assert any("app.ts" in p for p in result.paths)
    assert all("util.ts" not in p for p in result.paths)


@pytest.mark.asyncio
async def test_basename_pattern_matches_at_any_depth(glob_tool, tmp_path):
    (tmp_path / "top.py").write_text("x\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "deep.py").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern="*.py")))

    assert any("top.py" in p for p in result.paths)
    assert any("deep.py" in p for p in result.paths)


@pytest.mark.asyncio
async def test_results_sorted_by_mtime_desc(glob_tool, tmp_path):
    old = tmp_path / "old.py"
    mid = tmp_path / "mid.py"
    new = tmp_path / "new.py"
    for file in (old, mid, new):
        file.write_text("x\n")
    os.utime(old, (1000, 1000))
    os.utime(mid, (2000, 2000))
    os.utime(new, (3000, 3000))

    result = await collect_result(glob_tool.run(GlobArgs(pattern="*.py")))

    assert [Path(p).name for p in result.paths] == ["new.py", "mid.py", "old.py"]


@pytest.mark.asyncio
async def test_no_match_returns_empty(glob_tool, tmp_path):
    (tmp_path / "a.py").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern="**/*.rs")))

    assert result.paths == []
    assert result.match_count == 0
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_respects_default_excludes(glob_tool, tmp_path):
    (tmp_path / "included.py").write_text("x\n")
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "excluded.py").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern="**/*.py")))

    assert any("included.py" in p for p in result.paths)
    assert all("node_modules" not in p for p in result.paths)


@pytest.mark.asyncio
async def test_respects_vibeignore(glob_tool, tmp_path):
    (tmp_path / ".vibeignore").write_text("secret/\n")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "hidden.py").write_text("x\n")
    (tmp_path / "visible.py").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern="**/*.py")))

    assert any("visible.py" in p for p in result.paths)
    assert all("hidden.py" not in p for p in result.paths)


@pytest.mark.asyncio
async def test_max_results_truncation(glob_tool, tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x\n")

    result = await collect_result(
        glob_tool.run(GlobArgs(pattern="*.py", max_results=2))
    )

    assert len(result.paths) == 2
    assert result.match_count == 2
    assert result.was_truncated


@pytest.mark.asyncio
async def test_path_scopes_search_root(glob_tool, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inside.py").write_text("x\n")
    (tmp_path / "outside.py").write_text("x\n")

    result = await collect_result(
        glob_tool.run(GlobArgs(pattern="**/*.py", path="sub"))
    )

    assert any("inside.py" in p for p in result.paths)
    assert all("outside.py" not in p for p in result.paths)


@pytest.mark.asyncio
async def test_nonexistent_path_raises(glob_tool):
    with pytest.raises(ToolError) as err:
        await collect_result(glob_tool.run(GlobArgs(pattern="*.py", path="nope")))

    assert "Path does not exist" in str(err.value)


@pytest.mark.asyncio
async def test_empty_pattern_raises(glob_tool):
    with pytest.raises(ToolError) as err:
        await collect_result(glob_tool.run(GlobArgs(pattern="   ")))

    assert "Empty glob pattern" in str(err.value)


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep not available")
class TestRipgrepBackend:
    @pytest.mark.asyncio
    async def test_uses_ripgrep_backend(self, glob_tool):
        assert glob_tool._detect_backend() == GlobBackend.RIPGREP

    @pytest.mark.asyncio
    async def test_respects_dot_ignore(self, glob_tool, tmp_path):
        (tmp_path / ".ignore").write_text("hidden/\n")
        hidden = tmp_path / "hidden"
        hidden.mkdir()
        (hidden / "a.py").write_text("x\n")
        (tmp_path / "b.py").write_text("x\n")

        result = await collect_result(glob_tool.run(GlobArgs(pattern="**/*.py")))

        assert any("b.py" in p for p in result.paths)
        assert all("hidden" not in p for p in result.paths)

    @pytest.mark.asyncio
    async def test_use_default_ignore_false_includes_ignored(self, glob_tool, tmp_path):
        (tmp_path / ".ignore").write_text("hidden/\n")
        hidden = tmp_path / "hidden"
        hidden.mkdir()
        (hidden / "a.py").write_text("x\n")

        result = await collect_result(
            glob_tool.run(GlobArgs(pattern="**/*.py", use_default_ignore=False))
        )

        assert any("a.py" in p for p in result.paths)


class TestWalkBackend:
    @pytest.mark.asyncio
    async def test_uses_walk_backend(self, glob_walk_only):
        assert glob_walk_only._detect_backend() == GlobBackend.WALK

    @pytest.mark.asyncio
    async def test_finds_files_matching_pattern(self, glob_walk_only, tmp_path):
        (tmp_path / "a.py").write_text("x\n")
        (tmp_path / "b.txt").write_text("x\n")

        result = await collect_result(glob_walk_only.run(GlobArgs(pattern="**/*.py")))

        assert any("a.py" in p for p in result.paths)
        assert all("b.txt" not in p for p in result.paths)

    @pytest.mark.asyncio
    async def test_results_sorted_by_mtime_desc(self, glob_walk_only, tmp_path):
        old = tmp_path / "old.py"
        new = tmp_path / "new.py"
        old.write_text("x\n")
        new.write_text("x\n")
        os.utime(old, (1000, 1000))
        os.utime(new, (3000, 3000))

        result = await collect_result(glob_walk_only.run(GlobArgs(pattern="*.py")))

        assert [Path(p).name for p in result.paths] == ["new.py", "old.py"]

    @pytest.mark.asyncio
    async def test_respects_gitignore(self, glob_walk_only, tmp_path):
        (tmp_path / ".gitignore").write_text("ignored/\n")
        ignored = tmp_path / "ignored"
        ignored.mkdir()
        (ignored / "a.py").write_text("x\n")
        (tmp_path / "b.py").write_text("x\n")

        result = await collect_result(glob_walk_only.run(GlobArgs(pattern="**/*.py")))

        assert any("b.py" in p for p in result.paths)
        assert all("ignored" not in p for p in result.paths)

    @pytest.mark.asyncio
    async def test_respects_default_excludes(self, glob_walk_only, tmp_path):
        (tmp_path / "included.py").write_text("x\n")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "excluded.py").write_text("x\n")

        result = await collect_result(glob_walk_only.run(GlobArgs(pattern="**/*.py")))

        assert any("included.py" in p for p in result.paths)
        assert all("node_modules" not in p for p in result.paths)

    @pytest.mark.asyncio
    async def test_truncates_to_max_results(self, glob_walk_only, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("x\n")

        result = await collect_result(
            glob_walk_only.run(GlobArgs(pattern="*.py", max_results=2))
        )

        assert len(result.paths) == 2
        assert result.was_truncated

    @pytest.mark.asyncio
    async def test_symlink_loop_terminates(self, glob_walk_only, tmp_path):
        (tmp_path / "real.py").write_text("x\n")
        loop = tmp_path / "loop"
        loop.symlink_to(tmp_path, target_is_directory=True)

        result = await collect_result(glob_walk_only.run(GlobArgs(pattern="**/*.py")))

        assert any("real.py" in p for p in result.paths)


@pytest.mark.asyncio
async def test_absolute_pattern_scopes_to_literal_prefix(glob_tool, tmp_path):
    nested = tmp_path / "pkg" / "deep"
    nested.mkdir(parents=True)
    (nested / "in.py").write_text("x\n")
    (tmp_path / "pkg" / "out.py").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern=f"{nested}/**/*.py")))

    assert any("in.py" in p for p in result.paths)
    assert all("out.py" not in p for p in result.paths)


@pytest.mark.asyncio
async def test_absolute_pattern_walk_backend(glob_walk_only, tmp_path):
    nested = tmp_path / "pkg" / "deep"
    nested.mkdir(parents=True)
    (nested / "in.py").write_text("x\n")
    (tmp_path / "pkg" / "out.py").write_text("x\n")

    result = await collect_result(
        glob_walk_only.run(GlobArgs(pattern=f"{nested}/**/*.py"))
    )

    assert any("in.py" in p for p in result.paths)
    assert all("out.py" not in p for p in result.paths)


@pytest.mark.asyncio
async def test_offset_paginates_results(glob_tool, tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x\n")

    first = await collect_result(glob_tool.run(GlobArgs(pattern="*.py", max_results=2)))
    second = await collect_result(
        glob_tool.run(GlobArgs(pattern="*.py", max_results=2, offset=2))
    )

    first_names = {Path(p).name for p in first.paths}
    second_names = {Path(p).name for p in second.paths}
    assert len(first.paths) == 2
    assert len(second.paths) == 2
    assert first_names.isdisjoint(second_names)


@pytest.mark.asyncio
async def test_offset_beyond_end_returns_empty(glob_tool, tmp_path):
    (tmp_path / "a.py").write_text("x\n")

    result = await collect_result(glob_tool.run(GlobArgs(pattern="*.py", offset=10)))

    assert result.paths == []
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_offset_marks_truncated_when_more_remain(glob_tool, tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x\n")

    result = await collect_result(
        glob_tool.run(GlobArgs(pattern="*.py", max_results=2, offset=2))
    )

    assert len(result.paths) == 2
    assert result.was_truncated


@pytest.mark.asyncio
async def test_unc_path_rejected(glob_tool):
    with pytest.raises(ToolError) as err:
        await collect_result(
            glob_tool.run(GlobArgs(pattern="**/*.py", path="//host/share"))
        )

    assert "UNC" in str(err.value)


@pytest.mark.asyncio
async def test_unc_path_rejected_backslash(glob_tool):
    with pytest.raises(ToolError) as err:
        await collect_result(
            glob_tool.run(GlobArgs(pattern="**/*.py", path="\\\\host\\share"))
        )

    assert "UNC" in str(err.value)


@pytest.mark.asyncio
async def test_execute_kills_subprocess_on_cancel(glob_tool, monkeypatch):
    real_create = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    procs: list[asyncio.subprocess.Process] = []

    async def spy_create(*args, **kwargs):
        proc = await real_create(*args, **kwargs)
        procs.append(proc)
        spawned.set()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_create)

    task = asyncio.create_task(glob_tool._execute(["sleep", "30"]))
    await asyncio.wait_for(spawned.wait(), timeout=5.0)
    proc = procs[0]

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert proc.returncode is not None, "subprocess was not killed on cancellation"


@pytest.mark.asyncio
async def test_execute_kills_subprocess_on_timeout(glob_tool, monkeypatch):
    real_create = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    procs: list[asyncio.subprocess.Process] = []

    async def spy_create(*args, **kwargs):
        proc = await real_create(*args, **kwargs)
        procs.append(proc)
        spawned.set()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_create)
    monkeypatch.setattr(glob_tool.config, "default_timeout", 0.5)

    with pytest.raises(ToolError, match="timed out"):
        await glob_tool._execute(["sleep", "30"])

    assert procs, "subprocess never spawned"
    assert procs[0].returncode is not None, "subprocess was not killed on timeout"
