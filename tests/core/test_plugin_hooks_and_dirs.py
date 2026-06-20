from __future__ import annotations

from pathlib import Path

from tests.conftest import build_test_vibe_config
from vibe.core.config.harness_files import HarnessFilesManager
from vibe.core.hooks.config import load_hooks_from_fs
from vibe.core.hooks.models import HookConfig, HookType
from vibe.core.plugins.loader import load_plugins_from_fs
from vibe.core.trusted_folders import trusted_folders_manager


def _plugin(root: Path, name: str, toml: str, extra: dict[str, str] | None = None):
    pdir = root / name
    pdir.mkdir(parents=True)
    (pdir / "plugin.toml").write_text(toml)
    for rel, body in (extra or {}).items():
        f = pdir / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    return pdir


# --------------------------------------------------------------------------- #
# Plugin hooks parsing                                                          #
# --------------------------------------------------------------------------- #

_INLINE_HOOK = """
name = "ph"
[[hooks]]
name = "ph-hook"
type = "post_agent_turn"
command = "echo hi"
"""


def test_inline_hooks_parsed(tmp_path) -> None:
    _plugin(tmp_path, "ph", _INLINE_HOOK)
    res = load_plugins_from_fs([], [tmp_path])
    assert [h.name for h in res.hooks] == ["ph-hook"]
    assert res.hooks[0].type == HookType.POST_AGENT_TURN


def test_hooks_path_parsed(tmp_path) -> None:
    _plugin(
        tmp_path,
        "ph",
        'name = "ph"\nhooks = "hooks.toml"\n',
        extra={
            "hooks.toml": '[[hooks]]\nname = "file-hook"\n'
            'type = "stop"\ncommand = "echo"\n'
        },
    )
    res = load_plugins_from_fs([], [tmp_path])
    assert [h.name for h in res.hooks] == ["file-hook"]


def test_hooks_path_escape_is_an_issue(tmp_path) -> None:
    _plugin(tmp_path, "ph", 'name = "ph"\nhooks = "../../evil.toml"\n')
    res = load_plugins_from_fs([], [tmp_path])
    assert any("escapes the plugin root" in i for i in res.issues)
    assert res.hooks == []


# --------------------------------------------------------------------------- #
# load_hooks_from_fs plugin-hook merge + gate                                  #
# --------------------------------------------------------------------------- #


def _phook(name: str) -> HookConfig:
    return HookConfig(name=name, type=HookType.POST_AGENT_TURN, command="echo")


def test_plugin_hooks_merged_when_enabled() -> None:
    config = build_test_vibe_config(enable_experimental_hooks=True)
    res = load_hooks_from_fs(config, plugin_hooks=[_phook("pa"), _phook("pb")])
    names = {h.name for h in res.hooks}
    assert {"pa", "pb"} <= names


def test_plugin_hooks_gated_off_when_disabled() -> None:
    config = build_test_vibe_config(enable_experimental_hooks=False)
    res = load_hooks_from_fs(config, plugin_hooks=[_phook("pa")])
    assert res.hooks == []  # gate blocks plugin hooks too


def test_plugin_hook_duplicate_name_is_an_issue() -> None:
    config = build_test_vibe_config(enable_experimental_hooks=True)
    res = load_hooks_from_fs(config, plugin_hooks=[_phook("dup"), _phook("dup")])
    assert sum(h.name == "dup" for h in res.hooks) == 1
    assert any("Duplicate hook name (plugin)" in i.message for i in res.issues)


# --------------------------------------------------------------------------- #
# Trust-gated plugin_dirs                                                       #
# --------------------------------------------------------------------------- #


def test_plugin_dirs_excludes_untrusted_project(tmp_path) -> None:
    (tmp_path / ".vibe" / "plugins").mkdir(parents=True)
    mgr = HarnessFilesManager(sources=("project",), cwd=tmp_path)
    # Not trusted → project plugins excluded; no user source → empty.
    assert mgr.plugin_dirs == []


def test_plugin_dirs_includes_trusted_project(tmp_path) -> None:
    plugins = tmp_path / ".vibe" / "plugins"
    plugins.mkdir(parents=True)
    trusted_folders_manager.trust_for_session(tmp_path)
    mgr = HarnessFilesManager(sources=("project",), cwd=tmp_path)
    assert plugins.resolve() in [p.resolve() for p in mgr.plugin_dirs]
