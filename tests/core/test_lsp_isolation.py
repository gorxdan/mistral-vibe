from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from tests.conftest import build_test_vibe_config
from vibe.core.config import VibeConfig
from vibe.core.lsp._lifecycle import setup_lsp_for_config
from vibe.core.lsp._manager import clear_lsp_manager, get_lsp_manager
from vibe.core.tools.builtins.lsp import Lsp, LspConfig, LspState


def test_lsp_tool_is_unavailable_in_isolated_worktree(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(tmp_path))
    config = build_test_vibe_config(installed_components=["lsp"])

    assert not Lsp.is_available(config)


def test_lsp_setup_does_not_start_inside_isolated_worktree(
    tmp_path, monkeypatch
) -> None:
    clear_lsp_manager()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(tmp_path))
    config = build_test_vibe_config(installed_components=["lsp"])

    manager = setup_lsp_for_config(config, lambda: config, tmp_path, warmup=True)

    assert manager is None
    assert get_lsp_manager() is None


def test_lsp_lazy_setup_is_disabled_inside_isolated_worktree(
    tmp_path, monkeypatch
) -> None:
    clear_lsp_manager()
    monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(tmp_path))
    tool = Lsp(config_getter=LspConfig, state=LspState())

    assert tool._ensure_manager() is None


def test_lsp_setup_is_disabled_for_managed_topology(tmp_path) -> None:
    clear_lsp_manager()
    config = cast(
        VibeConfig,
        SimpleNamespace(
            installed_components=["lsp"],
            trusted_verification_recipe=SimpleNamespace(execution_topology=object()),
        ),
    )

    manager = setup_lsp_for_config(config, lambda: config, tmp_path, warmup=True)

    assert manager is None
    assert get_lsp_manager() is None
    assert not Lsp.is_available(config)
