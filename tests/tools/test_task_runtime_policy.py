from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.paths import DEFAULT_TOOL_DIR
from vibe.core.tasking import TaskManifestIdentity
from vibe.core.tools._task_manifest import resolve_task_manifest
from vibe.core.tools.base import BaseToolState, ToolError, ToolPermission
from vibe.core.tools.builtins.edit import Edit, EditArgs, EditConfig
from vibe.core.tools.builtins.read import Read, ReadConfig, ReadState
from vibe.core.tools.builtins.write_file import (
    WriteFile,
    WriteFileArgs,
    WriteFileConfig,
)
from vibe.core.tools.manager import NoSuchToolError, ToolManager, _compute_module_name
from vibe.core.tools.utils import resolve_file_tool_permission
from vibe.core.utils.io import write_safe


def test_runtime_allowlist_cannot_be_widened_by_search_or_pin() -> None:
    config = build_test_vibe_config(
        include_project_context=False, include_prompt_detail=False
    )
    manager = ToolManager(
        lambda: config,
        defer_mcp=True,
        runtime_allowlist=frozenset({"read", "tool_search"}),
    )

    assert set(manager.available_tools) <= {"read", "tool_search"}
    assert manager.pin_manifest_tools(["bash", "write_file"]) == []
    assert all(result.name != "bash" for result in manager.search_tools("bash"))
    with pytest.raises(NoSuchToolError):
        manager.get("bash")


def test_runtime_scoped_tools_are_hidden_without_a_trusted_manifest() -> None:
    config = build_test_vibe_config(
        include_project_context=False, include_prompt_detail=False
    )
    ordinary = ToolManager(lambda: config, defer_mcp=True)
    manifest = resolve_task_manifest(TaskManifestIdentity(name="verify", version="1"))
    scoped = ToolManager(
        lambda: config, defer_mcp=True, runtime_allowlist=frozenset(manifest.tools)
    )

    assert "task_checks" in ordinary.registered_tools
    assert "task_checks" not in ordinary.available_tools
    assert "task_checks" in scoped.available_tools


def test_runtime_manifest_is_authoritative_over_profile_enabled_tools() -> None:
    config = build_test_vibe_config(
        enabled_tools=["read"],
        installed_components=["lsp"],
        include_project_context=False,
        include_prompt_detail=False,
    )
    manifest = resolve_task_manifest(
        TaskManifestIdentity(name="investigate", version="1")
    )
    manager = ToolManager(lambda: config, runtime_allowlist=frozenset(manifest.tools))

    assert set(manager.available_tools) == set(manifest.tools)


def test_runtime_manifest_cannot_restore_a_disabled_tool() -> None:
    config = build_test_vibe_config(
        disabled_tools=["read"],
        include_project_context=False,
        include_prompt_detail=False,
    )
    manifest = resolve_task_manifest(
        TaskManifestIdentity(name="investigate", version="1")
    )
    manager = ToolManager(lambda: config, runtime_allowlist=frozenset(manifest.tools))

    assert "read" not in manager.available_tools


def test_runtime_manifest_does_not_import_project_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "VIBE_TEST_TASK_TOOL_IMPORTED"
    monkeypatch.delenv(marker, raising=False)
    write_safe(
        tmp_path / "shadow_read.py",
        "\n".join((
            "import os",
            f"os.environ[{marker!r}] = '1'",
            "from vibe.core.tools.builtins.read import Read",
            "class ShadowRead(Read):",
            "    pass",
        )),
    )
    config = build_test_vibe_config(
        tool_paths=[tmp_path],
        include_project_context=False,
        include_prompt_detail=False,
    )
    manifest = resolve_task_manifest(
        TaskManifestIdentity(name="investigate", version="1")
    )

    manager = ToolManager(lambda: config, runtime_allowlist=frozenset(manifest.tools))

    assert marker not in os.environ
    assert manager.registered_tools["read"] is Read


def test_runtime_manifest_does_not_scan_mutated_builtin_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "VIBE_TEST_PLANTED_BUILTIN_IMPORTED"
    monkeypatch.delenv(marker, raising=False)
    write_safe(
        tmp_path / "zzz.py",
        "\n".join((
            "import os",
            f"os.environ[{marker!r}] = '1'",
            "from vibe.core.tools.builtins.read import Read",
            "class ShadowRead(Read):",
            "    pass",
        )),
    )
    monkeypatch.setattr(
        "vibe.core.tools.manager.DEFAULT_TOOL_DIR", SimpleNamespace(path=tmp_path)
    )
    config = build_test_vibe_config(
        include_project_context=False, include_prompt_detail=False
    )
    manifest = resolve_task_manifest(
        TaskManifestIdentity(name="investigate", version="1")
    )

    manager = ToolManager(lambda: config, runtime_allowlist=frozenset(manifest.tools))

    assert marker not in os.environ
    assert manager.registered_tools["read"] is Read


def test_builtin_discovery_attaches_canonical_module_to_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = importlib.import_module("vibe.core.tools.builtins")
    old_module = importlib.import_module("vibe.core.tools.builtins.lsp")
    monkeypatch.delitem(sys.modules, "vibe.core.tools.builtins.lsp")
    monkeypatch.delattr(parent, "lsp")

    loaded = ToolManager._load_tools_from_file(DEFAULT_TOOL_DIR.path / "lsp.py")

    assert loaded
    assert parent.lsp is sys.modules["vibe.core.tools.builtins.lsp"]
    assert parent.lsp is not old_module


def test_failed_dynamic_tool_import_is_removed_from_module_cache(
    tmp_path: Path,
) -> None:
    tool_file = tmp_path / "broken.py"
    write_safe(
        tool_file,
        "\n".join((
            "from vibe.core.tools.builtins.read import Read",
            "class PartiallyDefined(Read):",
            "    pass",
            "raise RuntimeError('broken import')",
        )),
    )
    module_name = _compute_module_name(tool_file)

    assert ToolManager._load_tools_from_file(tool_file) is None
    assert module_name not in sys.modules


@pytest.mark.parametrize(
    "name", ["investigate", "implement-verify", "verify", "mechanical-edit"]
)
def test_trusted_manifests_reference_registered_tools(name: str) -> None:
    config = build_test_vibe_config(
        include_project_context=False, include_prompt_detail=False
    )
    manager = ToolManager(lambda: config)
    manifest = resolve_task_manifest(TaskManifestIdentity(name=name, version="1"))

    assert 6 <= len(manifest.tools) <= 10
    assert set(manifest.tools) <= set(manager.registered_tools)


def test_team_metadata_is_hard_denied_for_file_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    team_dir = tmp_path / "team"
    team_dir.mkdir()
    metadata = team_dir / "tasks.json"
    write_safe(metadata, "{}")
    monkeypatch.setenv("VIBE_TEAM_DIR", str(team_dir))

    permission = resolve_file_tool_permission(
        str(metadata),
        tool_name="read",
        allowlist=["*"],
        denylist=[],
        config_permission=ToolPermission.ALWAYS,
        sensitive_patterns=[],
    )
    assert permission is not None
    assert permission.permission is ToolPermission.NEVER

    read = Read(config_getter=lambda: ReadConfig(), state=ReadState())
    write = WriteFile(config_getter=lambda: WriteFileConfig(), state=BaseToolState())
    edit = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())

    with pytest.raises(ToolError, match="host-owned"):
        read._resolve_path(str(metadata))
    with pytest.raises(ToolError, match="host-owned"):
        write._prepare_and_validate_path(
            WriteFileArgs(path=str(team_dir / "new.json"), content="{}")
        )
    with pytest.raises(ToolError, match="host-owned"):
        edit._validate_args(
            EditArgs(file_path=str(metadata), old_string="{}", new_string="[]")
        )
