from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_vibe_config
from vibe.core.plugins.loader import apply_plugin_result, load_plugins_from_fs
from vibe.core.plugins.models import PluginManifest


def _make_plugin(root: Path, name: str, toml: str, dirs: list[str]) -> Path:
    pdir = root / name
    pdir.mkdir(parents=True)
    (pdir / "plugin.toml").write_text(toml)
    for d in dirs:
        (pdir / d).mkdir(parents=True, exist_ok=True)
    return pdir


def test_manifest_default_component_dirs() -> None:
    m = PluginManifest(name="p", tools="custom_tools")
    assert m.component_dirs("tools") == ["custom_tools"]
    assert m.component_dirs("agents") == ["agents"]  # default subdir


def test_valid_plugin_expands_paths_and_mcp(tmp_path) -> None:
    _make_plugin(
        tmp_path,
        "myplugin",
        """
name = "myplugin"
version = "1.0"
[[mcp_servers]]
name = "srv"
transport = "stdio"
command = "echo"
""",
        dirs=["agents", "tools", "skills"],
    )
    res = load_plugins_from_fs([], [tmp_path])
    assert res.plugins == ["myplugin"]
    assert any(p.name == "agents" for p in res.agent_paths)
    assert any(p.name == "tools" for p in res.tool_paths)
    assert any(p.name == "skills" for p in res.skill_paths)
    assert res.mcp_servers and res.mcp_servers[0]["name"] == "srv"
    assert res.issues == []


def test_malformed_manifest_is_an_issue_not_a_crash(tmp_path) -> None:
    p = tmp_path / "bad"
    p.mkdir()
    (p / "plugin.toml").write_text("this is not valid toml = = =")
    res = load_plugins_from_fs([], [tmp_path])
    assert res.plugins == []
    assert res.issues and "bad" in res.issues[0]


def test_path_escape_is_rejected(tmp_path) -> None:
    _make_plugin(tmp_path, "evil", 'name = "evil"\nagents = "../../etc"\n', dirs=[])
    res = load_plugins_from_fs([], [tmp_path])
    assert any("escapes the plugin root" in i for i in res.issues)
    assert res.agent_paths == []


def test_prompts_component_collected_into_prompt_paths(tmp_path) -> None:
    # The plugin manifest advertises a `prompts` component; it must land in
    # PluginLoadResult.prompt_paths and be folded onto config by
    # apply_plugin_result (previously declared but dropped on the floor).
    _make_plugin(tmp_path, "withprompts", 'name = "withprompts"\n', dirs=["prompts"])
    res = load_plugins_from_fs([], [tmp_path])
    assert res.plugins == ["withprompts"]
    assert any(p.name == "prompts" for p in res.prompt_paths)

    config = build_test_vibe_config()
    before = list(config.prompt_paths)
    apply_plugin_result(config, res)
    assert config.prompt_paths == [*before, *res.prompt_paths]


def test_name_collision_is_an_issue(tmp_path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "plugin.toml").write_text('name = "dup"\n')
    res = load_plugins_from_fs([], [tmp_path])
    assert res.plugins == ["dup"]  # only the first
    assert any("collision" in i for i in res.issues)


def test_enabled_allowlist(tmp_path) -> None:
    _make_plugin(tmp_path, "keep", 'name = "keep"\n', dirs=["tools"])
    _make_plugin(tmp_path, "drop", 'name = "drop"\n', dirs=["tools"])
    res = load_plugins_from_fs([], [tmp_path], enabled=["keep"])
    assert res.plugins == ["keep"]


def test_disabled_denylist(tmp_path) -> None:
    _make_plugin(tmp_path, "keep", 'name = "keep"\n', dirs=["tools"])
    _make_plugin(tmp_path, "drop", 'name = "drop"\n', dirs=["tools"])
    res = load_plugins_from_fs([], [tmp_path], disabled=["drop"])
    assert set(res.plugins) == {"keep"}


def test_explicit_plugin_path_to_root(tmp_path) -> None:
    pdir = _make_plugin(tmp_path, "direct", 'name = "direct"\n', dirs=["agents"])
    res = load_plugins_from_fs([pdir], [])
    assert res.plugins == ["direct"]


def test_apply_folds_paths_and_mcp_into_config(tmp_path) -> None:
    _make_plugin(
        tmp_path,
        "p",
        """
name = "p"
[[mcp_servers]]
name = "psrv"
transport = "stdio"
command = "echo"
""",
        dirs=["agents", "tools"],
    )
    config = build_test_vibe_config()
    n_agents = len(config.agent_paths)
    n_mcp = len(config.mcp_servers)

    res = load_plugins_from_fs([], [tmp_path])
    apply_plugin_result(config, res)

    assert len(config.agent_paths) == n_agents + 1
    assert any(p.name == "agents" for p in config.agent_paths)
    assert len(config.mcp_servers) == n_mcp + 1
    assert any(s.name == "psrv" for s in config.mcp_servers)


@pytest.mark.parametrize("dup_name", ["psrv"])
def test_apply_mcp_union_skips_existing_name(tmp_path, dup_name) -> None:
    _make_plugin(
        tmp_path,
        "p",
        f'name = "p"\n[[mcp_servers]]\nname = "{dup_name}"\ntransport = "stdio"\ncommand = "echo"\n',
        dirs=[],
    )
    config = build_test_vibe_config()
    res = load_plugins_from_fs([], [tmp_path])
    # First apply adds it; second apply must not duplicate.
    apply_plugin_result(config, res)
    apply_plugin_result(config, res)
    assert sum(1 for s in config.mcp_servers if s.name == dup_name) == 1
