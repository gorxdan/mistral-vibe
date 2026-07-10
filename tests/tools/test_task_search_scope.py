from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from tests.mock.utils import collect_result
from vibe.core.config import (
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.tasking import TaskBrief, TaskManifestIdentity
from vibe.core.tasking._policy import BoundTaskContract, TaskContractAuthority
from vibe.core.tools.base import BaseToolState, InvokeContext
from vibe.core.tools.builtins.glob import Glob, GlobArgs, GlobToolConfig
from vibe.core.tools.builtins.grep import Grep, GrepArgs, GrepBackend, GrepToolConfig
from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspResult, LspState
from vibe.core.utils.io import write_safe
from vibe.core.verification_state import VerificationState


def _contract(
    root: Path, *, denied_paths: list[str] | None = None
) -> BoundTaskContract:
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="search-scope-v1",
        task_brief="Inspect scoped files",
        acceptance_contract="Focused checks must pass",
        allowed_paths=("**",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused", argv=("uv", "run", "pytest", "tests/focused.py")
            ),
        ),
    )
    brief = TaskBrief(
        objective="Inspect scoped files",
        allowed_paths=["**"],
        denied_paths=denied_paths or ["src/private/**"],
        acceptance_checks=["focused"],
        manifest=TaskManifestIdentity(name="investigate", version="1"),
    )
    return BoundTaskContract.bind(
        brief,
        authority=TaskContractAuthority.LEAD,
        workspace_root=root,
        verification_state=VerificationState.from_recipe(recipe),
    )


@pytest.mark.asyncio
async def test_task_search_filters_case_aliases_and_team_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    protected = (
        tmp_path / ".VIBE" / "secret.txt",
        tmp_path / "src" / "PRIVATE" / "secret.txt",
        tmp_path / "team" / "tasks.txt",
    )
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_safe(path, "needle\n")
    visible = tmp_path / "src" / "visible.txt"
    write_safe(visible, "needle\n")
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path / "team"))
    contract = _contract(tmp_path)
    ctx = InvokeContext(tool_call_id="scoped-search", task_contract=contract)
    glob = Glob(config_getter=GlobToolConfig, state=BaseToolState())
    grep = Grep(config_getter=GrepToolConfig, state=BaseToolState())

    glob_result = await collect_result(
        glob.run(GlobArgs(pattern="**/*.txt", use_default_ignore=False), ctx)
    )
    grep_result = await collect_result(
        grep.run(GrepArgs(pattern="needle", use_default_ignore=False), ctx)
    )

    assert glob_result.paths == [str(visible)]
    assert "visible.txt" in grep_result.matches
    assert ".VIBE" not in grep_result.matches
    assert "PRIVATE" not in grep_result.matches
    assert "team/tasks.txt" not in grep_result.matches


@pytest.mark.parametrize(
    ("backend", "binary"), [(GrepBackend.RIPGREP, "rg"), (GrepBackend.GNU_GREP, "grep")]
)
@pytest.mark.asyncio
async def test_task_grep_uses_unambiguous_filename_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: GrepBackend, binary: str
) -> None:
    if shutil.which(binary) is None:
        pytest.skip(f"{binary} is unavailable")
    monkeypatch.chdir(tmp_path)
    prefix = tmp_path / "src" / "public" / "foo"
    prefix.parent.mkdir(parents=True)
    write_safe(prefix, "ordinary\n")
    denied = tmp_path / "src" / "public" / "foo:1:bar.txt"
    write_safe(denied, "BOUNDARY_SECRET\n")
    contract = _contract(tmp_path, denied_paths=["src/public/foo:1:bar.txt"])
    ctx = InvokeContext(tool_call_id="colon-path", task_contract=contract)
    grep = Grep(config_getter=GrepToolConfig, state=BaseToolState())
    monkeypatch.setattr(grep, "_detect_backend", lambda: backend)

    result = await collect_result(
        grep.run(GrepArgs(pattern="BOUNDARY_SECRET", use_default_ignore=False), ctx)
    )

    assert result.match_count == 0
    assert "BOUNDARY_SECRET" not in result.matches


def test_task_lsp_filters_locations_and_rebuilds_summary(tmp_path: Path) -> None:
    visible = tmp_path / "src" / "visible.py"
    protected = tmp_path / ".VIBE" / "secret.py"
    contract = _contract(tmp_path)
    ctx = InvokeContext(tool_call_id="lsp-scope", task_contract=contract)
    tool = Lsp(config_getter=LspConfig, state=LspState())
    result = LspResult(
        operation="locations",
        summary=f"Definitions (2):\n  {visible}:1:1\n  {protected}:1:1",
        locations=[
            {"uri": visible.as_uri(), "range": {"start": {"line": 0}}},
            {"uri": protected.as_uri(), "range": {"start": {"line": 0}}},
        ],
    )

    scoped = tool._scope_result(result, ctx)

    assert len(scoped.locations) == 1
    assert scoped.locations[0]["uri"] == visible.as_uri()
    assert str(visible) in scoped.summary
    assert str(protected) not in scoped.summary
