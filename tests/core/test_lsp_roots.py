from __future__ import annotations

from pathlib import Path

from vibe.core.lsp._roots import (
    directory_matches_markers,
    nearest_manifest_root,
    resolve_workspace_root,
)


def test_nearest_manifest_root_selects_nearest_ancestor(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "api"
    source = package / "src" / "main.py"
    source.parent.mkdir(parents=True)
    (tmp_path / "pyproject.toml").touch()
    (package / "pyproject.toml").touch()

    root = nearest_manifest_root(source, tmp_path, ("pyproject.toml",))

    assert root == package


def test_nearest_manifest_root_supports_glob_markers(tmp_path: Path) -> None:
    package = tmp_path / "apps" / "api"
    source = package / "Program.cs"
    package.mkdir(parents=True)
    (package / "api.csproj").touch()

    root = nearest_manifest_root(source, tmp_path, ("*.csproj",))

    assert root == package


def test_nearest_manifest_root_handles_nonexistent_file(tmp_path: Path) -> None:
    package = tmp_path / "packages" / "new"
    package.mkdir(parents=True)
    (package / "Cargo.toml").touch()

    root = nearest_manifest_root(package / "src" / "lib.rs", tmp_path, ("Cargo.toml",))

    assert root == package


def test_nearest_manifest_root_does_not_escape_session(tmp_path: Path) -> None:
    session = tmp_path / "workspace"
    session.mkdir()
    (tmp_path / "pyproject.toml").touch()

    root = nearest_manifest_root(
        session / "src" / "main.py", session, ("pyproject.toml",)
    )

    assert root == session


def test_nearest_manifest_root_rejects_file_outside_session(tmp_path: Path) -> None:
    session = tmp_path / "workspace"
    external = tmp_path / "external"
    session.mkdir()
    external.mkdir()
    (external / "go.mod").touch()

    root = nearest_manifest_root(external / "main.go", session, ("go.mod",))

    assert root == session


def test_directory_matches_markers_ignores_parent_traversal(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (tmp_path / "secret.toml").touch()

    assert not directory_matches_markers(child, ("../secret.toml",))


def test_resolve_workspace_root_preserves_explicit_uri(tmp_path: Path) -> None:
    explicit = "untitled:custom-workspace"

    root = resolve_workspace_root(
        tmp_path / "src" / "main.py",
        tmp_path,
        ("pyproject.toml",),
        explicit_root_uri=explicit,
    )

    assert root.uri == explicit
    assert root.path is None
    assert root.explicit


def test_resolve_workspace_root_returns_discovered_path(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "go.mod").touch()

    root = resolve_workspace_root(package / "main.go", tmp_path, ("go.mod",))

    assert root.uri == package.as_uri()
    assert root.path == package
    assert not root.explicit
