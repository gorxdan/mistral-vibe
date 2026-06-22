from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from vibe.core.config._settings import TomlFileSettingsSource


class _FakeMgr:
    """Minimal stand-in for HarnessFilesManager."""

    def __init__(self, config_file: Path | None, user_config_file: Path) -> None:
        self.config_file = config_file
        self.user_config_file = user_config_file

    trusted_workdir: Path | None = None


def test_deep_merge_override_wins_per_key() -> None:
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": [1, 2]}
    override = {"b": {"y": 99, "z": 3}, "d": 4}
    merged = TomlFileSettingsSource._deep_merge(base, override)
    # Override wins for b.y; base keeps b.x; new key d added; untouched keys preserved.
    assert merged == {"a": 1, "b": {"x": 1, "y": 99, "z": 3}, "c": [1, 2], "d": 4}
    # Base is not mutated.
    assert base["b"] == {"x": 1, "y": 2}


def test_deep_merge_override_replaces_non_dict_scalar() -> None:
    base = {"active_model": "glm"}
    override = {"active_model": "mistral-large"}
    assert TomlFileSettingsSource._deep_merge(base, override) == {
        "active_model": "mistral-large"
    }


def test_load_toml_merges_project_over_user(tmp_path: Path) -> None:
    """Regression: trusting a project dir must not drop the user config.

    A project .vibe/config.toml that sets only [tools.web_search] used to
    replace the user config wholesale, losing active_model/models/providers
    and falling back to defaults. Project keys must override, not replace.
    """
    user_file = tmp_path / "user.toml"
    user_file.write_text(
        'active_model = "glm"\n[[models]]\nalias = "glm"\nprovider = "zai"\n'
    )
    project_file = tmp_path / "project.toml"
    project_file.write_text(
        '[tools.web_search]\nsearxng_url = "http://localhost:8080"\n'
    )

    mgr = _FakeMgr(config_file=project_file, user_config_file=user_file)
    with patch(
        "vibe.core.config._settings.get_harness_files_manager", return_value=mgr
    ):
        src = TomlFileSettingsSource.__new__(TomlFileSettingsSource)
        data = src._load_toml()

    # User keys preserved (not dropped to defaults).
    assert data["active_model"] == "glm"
    assert any(m["alias"] == "glm" for m in data["models"])
    # Project override applied.
    assert data["tools"]["web_search"]["searxng_url"] == "http://localhost:8080"


def test_load_toml_no_merge_when_project_is_user_file(tmp_path: Path) -> None:
    """When config_file IS the user file, no self-merge happens."""
    user_file = tmp_path / "user.toml"
    user_file.write_text('active_model = "glm"\n')
    mgr = _FakeMgr(config_file=user_file, user_config_file=user_file)
    with patch(
        "vibe.core.config._settings.get_harness_files_manager", return_value=mgr
    ):
        src = TomlFileSettingsSource.__new__(TomlFileSettingsSource)
        data = src._load_toml()
    assert data == {"active_model": "glm"}
