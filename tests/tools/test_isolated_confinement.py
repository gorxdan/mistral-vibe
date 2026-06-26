from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest

from vibe.core.programmatic import _isolated_auto_approve
from vibe.core.tools.base import ToolError
from vibe.core.tools.utils import enforce_isolated_confine
from vibe.core.types import ApprovalResponse


def test_confinement_noop_when_root_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VIBE_ISOLATED_WORKTREE_ROOT", raising=False)
    enforce_isolated_confine(tmp_path / "anywhere.py")


def test_confinement_allows_path_inside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wt"
    root.mkdir()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(root))
    enforce_isolated_confine(root / "src" / "mod.py")


def test_confinement_rejects_path_outside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wt"
    root.mkdir()
    outside = tmp_path / "secret.env"
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(root))
    with pytest.raises(ToolError, match="confined to its worktree"):
        enforce_isolated_confine(outside)


def test_confinement_rejects_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "wt"
    (root / "src").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    link = root / "src" / "escape.txt"
    link.symlink_to(outside)
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(root))
    with pytest.raises(ToolError):
        enforce_isolated_confine(link)


@pytest.mark.asyncio
async def test_isolated_auto_approve_returns_yes() -> None:
    class _DummyArgs(BaseModel):
        pass

    response, feedback, _modified = await _isolated_auto_approve(
        "write_file", _DummyArgs(), "tc-1", None, None
    )
    assert response == ApprovalResponse.YES
    assert feedback is None
