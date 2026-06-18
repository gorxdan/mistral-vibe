from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from vibe.core.workflows.manager import WorkflowManager


def _make_config(workflow_paths: list[Path] | None = None) -> MagicMock:
    config = MagicMock()
    config.workflow_paths = workflow_paths or []
    return config


def test_discovers_bundled_workflows() -> None:
    mgr = WorkflowManager(lambda: _make_config())
    names = mgr.get_workflow_names()
    assert "deep-research" in names


def test_bundled_workflow_has_source() -> None:
    mgr = WorkflowManager(lambda: _make_config())
    info = mgr.get_workflow("deep-research")
    assert info is not None
    assert "async def main" in info.source
    assert info.is_bundled is True
    assert info.description


def test_get_nonexistent_workflow_returns_none() -> None:
    mgr = WorkflowManager(lambda: _make_config())
    assert mgr.get_workflow("nonexistent") is None


def test_discovers_custom_workflow(tmp_path: Path) -> None:
    wf_file = tmp_path / "my-audit.py"
    wf_file.write_text(
        "---\n"
        "name: my-audit\n"
        "description: Custom audit workflow\n"
        "---\n"
        "async def main():\n"
        "    return {}\n"
    )

    mgr = WorkflowManager(lambda: _make_config(workflow_paths=[tmp_path]))
    info = mgr.get_workflow("my-audit")
    assert info is not None
    assert info.description == "Custom audit workflow"
    assert info.is_bundled is False
    assert "async def main" in info.source


def test_custom_overrides_bundled(tmp_path: Path) -> None:
    wf_file = tmp_path / "deep_research.py"
    wf_file.write_text(
        "---\n"
        "name: deep-research\n"
        "description: My custom research\n"
        "---\n"
        "async def main():\n"
        "    return {}\n"
    )

    mgr = WorkflowManager(lambda: _make_config(workflow_paths=[tmp_path]))
    info = mgr.get_workflow("deep-research")
    assert info is not None
    assert info.description == "My custom research"
    assert info.is_bundled is False


def test_workflow_without_frontmatter(tmp_path: Path) -> None:
    wf_file = tmp_path / "simple.py"
    wf_file.write_text("async def main():\n    return {}\n")

    mgr = WorkflowManager(lambda: _make_config(workflow_paths=[tmp_path]))
    info = mgr.get_workflow("simple")
    assert info is not None
    assert info.name == "simple"
    assert info.description == ""
    assert "async def main" in info.source


def test_first_match_wins(tmp_path: Path) -> None:
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()

    (dir1 / "audit.py").write_text(
        "---\nname: audit\ndescription: First\n---\nasync def main():\n    return {}\n"
    )
    (dir2 / "audit.py").write_text(
        "---\nname: audit\ndescription: Second\n---\nasync def main():\n    return {}\n"
    )

    mgr = WorkflowManager(lambda: _make_config(workflow_paths=[dir1, dir2]))
    info = mgr.get_workflow("audit")
    assert info is not None
    assert info.description == "First"


def test_args_schema_parsed_from_frontmatter(tmp_path: Path) -> None:
    wf_file = tmp_path / "with-args.py"
    wf_file.write_text(
        "---\n"
        "name: with-args\n"
        "description: Takes args\n"
        "args_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    topic:\n"
        "      type: string\n"
        "---\n"
        "async def main():\n"
        "    return {}\n"
    )
    mgr = WorkflowManager(lambda: _make_config(workflow_paths=[tmp_path]))
    info = mgr.get_workflow("with-args")
    assert info is not None
    assert info.args_schema == {
        "type": "object",
        "properties": {"topic": {"type": "string"}},
    }


def test_description_with_colon_parsed_via_yaml(tmp_path: Path) -> None:
    # The old line-based parser kept the surrounding quotes; YAML strips them
    # and preserves the embedded colon.
    wf_file = tmp_path / "colon.py"
    wf_file.write_text(
        "---\n"
        "name: colon\n"
        'description: "Audit: find bugs"\n'
        "---\n"
        "async def main():\n    return {}\n"
    )
    mgr = WorkflowManager(lambda: _make_config(workflow_paths=[tmp_path]))
    info = mgr.get_workflow("colon")
    assert info is not None
    assert info.description == "Audit: find bugs"


def test_malformed_frontmatter_falls_back_to_stem(tmp_path: Path) -> None:
    wf_file = tmp_path / "broken.py"
    wf_file.write_text(
        "---\nname: [unclosed\n---\nasync def main():\n    return {}\n"
    )
    mgr = WorkflowManager(lambda: _make_config(workflow_paths=[tmp_path]))
    info = mgr.get_workflow("broken")
    assert info is not None  # name falls back to the filename stem
    assert "async def main" in info.source
