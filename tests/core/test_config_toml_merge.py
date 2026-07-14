from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import tomli_w

from tests.trusted_verification import (
    HOST_ENVIRONMENT as _HOST_ENVIRONMENT,
    HOST_ENVIRONMENT_SHA256 as _HOST_ENVIRONMENT_SHA256,
    HOST_PYTHON as _HOST_PYTHON,
    HOST_PYTHON_SHA256 as _HOST_PYTHON_SHA256,
)
from vibe.core.config import TrustedVerificationRecipeConfig, VibeConfig
from vibe.core.config._settings import TomlFileSettingsSource
from vibe.core.utils.io import write_safe


class _FakeMgr:
    """Minimal stand-in for HarnessFilesManager."""

    def __init__(
        self,
        user_config_file: Path,
        sources: tuple[str, ...] = ("user",),
        project_config_files_with_roots: list[tuple[Path, Path]] | None = None,
    ) -> None:
        self.user_config_file = user_config_file
        self.sources = sources
        self.project_config_files_with_roots = project_config_files_with_roots or []

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

    mgr = _FakeMgr(
        user_config_file=user_file,
        sources=("user", "project"),
        project_config_files_with_roots=[(project_file, tmp_path)],
    )
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
    mgr = _FakeMgr(user_config_file=user_file)
    with patch(
        "vibe.core.config._settings.get_harness_files_manager", return_value=mgr
    ):
        src = TomlFileSettingsSource.__new__(TomlFileSettingsSource)
        data = src._load_toml()
    assert data == {"active_model": "glm"}


def _recipe(label: str) -> dict[str, object]:
    return {
        "recipe_version": label,
        "task_brief": f"Task {label}",
        "acceptance_contract": f"Contract {label}",
        "allowed_paths": ["vibe/core/target.py"],
        "checks": [
            {
                "name": "focused",
                "argv": [str(_HOST_PYTHON), "-c", "raise SystemExit(0)"],
                "executable_sha256": _HOST_PYTHON_SHA256,
                "environment_attestation_path": str(_HOST_ENVIRONMENT),
                "environment_attestation_sha256": _HOST_ENVIRONMENT_SHA256,
            }
        ],
    }


def test_project_cannot_supply_or_replace_trusted_recipe(tmp_path: Path) -> None:
    user_file = tmp_path / "user.toml"
    project_file = tmp_path / "project.toml"
    user_recipe = _recipe("user-v1")
    write_safe(
        user_file,
        tomli_w.dumps({
            "trusted_verification_recipe": user_recipe,
            "verification_subsystem": True,
        }),
    )
    write_safe(
        project_file,
        tomli_w.dumps({
            "Trusted_Verification_Recipe": _recipe("project-v1"),
            "Verification_Subsystem": False,
        }),
    )
    mgr = _FakeMgr(
        user_config_file=user_file,
        sources=("user", "project"),
        project_config_files_with_roots=[(project_file, tmp_path)],
    )

    with patch(
        "vibe.core.config._settings.get_harness_files_manager", return_value=mgr
    ):
        src = TomlFileSettingsSource.__new__(TomlFileSettingsSource)
        data = src._load_toml()

    assert data["trusted_verification_recipe"] == user_recipe
    assert data["verification_subsystem"] is True


def test_project_cannot_be_the_source_of_a_trusted_recipe(tmp_path: Path) -> None:
    user_file = tmp_path / "user.toml"
    project_file = tmp_path / "project.toml"
    write_safe(user_file, "")
    write_safe(
        project_file,
        tomli_w.dumps({"TRUSTED_VERIFICATION_RECIPE": _recipe("project-v1")}),
    )
    mgr = _FakeMgr(
        user_config_file=user_file,
        sources=("user", "project"),
        project_config_files_with_roots=[(project_file, tmp_path)],
    )

    with patch(
        "vibe.core.config._settings.get_harness_files_manager", return_value=mgr
    ):
        src = TomlFileSettingsSource.__new__(TomlFileSettingsSource)
        data = src._load_toml()

    assert "trusted_verification_recipe" not in data


def test_programmatic_trusted_recipe_cannot_disable_verification() -> None:
    recipe = TrustedVerificationRecipeConfig.model_validate(_recipe("programmatic-v1"))

    config = VibeConfig(
        trusted_verification_recipe=recipe, verification_subsystem=False
    )

    assert config.trusted_verification_recipe == recipe
    assert config.verification_subsystem is True
