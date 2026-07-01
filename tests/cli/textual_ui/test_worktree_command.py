from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_vibe_app
from vibe.cli.textual_ui.widgets.messages import UserCommandMessage
from vibe.core.worktree.manager import WorktreeHandle, worktree_manager


@pytest.mark.asyncio
async def test_worktree_merge_points_at_the_cli_merge_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handle = WorktreeHandle(
        original_repo_root=tmp_path / "repo",
        worktree_path=tmp_path / "wt",
        branch="vibe/test-1-2",
        create_head_sha="0" * 40,
    )
    monkeypatch.setattr(worktree_manager, "_active", handle)
    app = build_test_vibe_app()
    mounted: list[UserCommandMessage] = []

    async def capture(widget: UserCommandMessage) -> None:
        mounted.append(widget)

    monkeypatch.setattr(app, "_mount_and_scroll", capture)

    await app._worktree_command("merge")

    assert mounted
    content = mounted[0]._content
    # The locked rebase-then-ff lives in `vibe worktree merge`; raw `git merge`
    # instructions bypass it.
    assert f"vibe worktree merge {handle.branch}" in content
    assert "git merge" not in content
