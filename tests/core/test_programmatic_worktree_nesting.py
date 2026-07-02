from __future__ import annotations

import json
import os
from pathlib import Path

from git import Repo
import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.mock_backend_factory import mock_backend_factory
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import SessionLoggingConfig
from vibe.core.programmatic import ProgrammaticOptions, run_programmatic
from vibe.core.session.session_loader import SessionLoader
from vibe.core.types import Backend, FunctionCall, ToolCall
from vibe.core.worktree.ephemeral import (
    create_ephemeral_worktree,
    remove_ephemeral_worktree,
)
from vibe.core.worktree.manager import worktree_manager


def _run_simple_programmatic(options: ProgrammaticOptions | None = None) -> None:
    with mock_backend_factory(
        Backend.MISTRAL,
        lambda provider, **kwargs: FakeBackend([[mock_llm_chunk(content="Done.")]]),
    ):
        run_programmatic(
            config=build_test_vibe_config(),
            prompt="hi",
            options=options
            or ProgrammaticOptions(agent_name=BuiltinAgentName.AUTO_APPROVE),
        )


@pytest.fixture
def enter_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.delenv("VIBE_ISOLATED_WORKTREE_ROOT", raising=False)
    calls: list[str] = []

    def spy(label: str, config: object) -> None:
        calls.append(label)
        return None

    monkeypatch.setattr(worktree_manager, "enter", spy)
    return calls


def test_programmatic_no_worktree_option_skips_entry(enter_spy: list[str]) -> None:
    _run_simple_programmatic(
        ProgrammaticOptions(agent_name=BuiltinAgentName.AUTO_APPROVE, no_worktree=True)
    )
    assert enter_spy == []


def test_programmatic_enters_worktree_by_default(enter_spy: list[str]) -> None:
    _run_simple_programmatic()
    assert enter_spy == ["programmatic"]


def test_isolated_child_does_not_nest_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = Repo.init(str(repo_dir))
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "t@t.com")
    (repo_dir / "f.txt").write_text("base\n")
    repo.index.add(["f.txt"])
    repo.index.commit("init")

    # Exactly what run_isolated_agent does before spawning the child process.
    wt = create_ephemeral_worktree(repo_dir, "worker", base_dir=tmp_path / "iso")
    try:
        monkeypatch.chdir(wt.path)
        monkeypatch.setenv("VIBE_ISOLATED_WORKTREE_ROOT", str(wt.path))
        monkeypatch.setenv("VIBE_ISOLATED_AUTO_APPROVE", "1")

        tool_call = ToolCall(
            id="call_1",
            index=0,
            function=FunctionCall(
                name="write_file",
                arguments=json.dumps({"path": "out.txt", "content": "done\n"}),
            ),
        )
        config = build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                enabled=True, save_dir=str(tmp_path / "sessions")
            )
        )
        nested_base = Path(config.worktree.base_dir)
        with mock_backend_factory(
            Backend.MISTRAL,
            lambda provider, **kwargs: FakeBackend([
                [mock_llm_chunk(content="Writing.", tool_calls=[tool_call])],
                [mock_llm_chunk(content="Done.")],
            ]),
        ):
            run_programmatic(
                config=config,
                prompt="hi",
                options=ProgrammaticOptions(agent_name=BuiltinAgentName.AUTO_APPROVE),
            )

        # The relative write landed inside the parent-created ephemeral worktree
        # (RED pre-fix: a nested worktree chdir made confinement reject it).
        assert (wt.path / "out.txt").exists()
        assert Repo(str(repo_dir)).git.branch("--list", "vibe/programmatic-*") == ""
        assert Path.cwd().resolve() == wt.path.resolve()
        assert not nested_base.exists() or not any(nested_base.iterdir())

        meta_files = list((tmp_path / "sessions").glob("*/meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        working_dir = meta["environment"]["working_directory"]
        assert working_dir == str(Path(wt.path).resolve())
        # The child session must not surface in the main repo's resume namespace.
        assert (
            SessionLoader.find_latest_session(
                config.session_logging, working_directory=repo_dir
            )
            is None
        )
    finally:
        os.chdir(repo_dir)
        remove_ephemeral_worktree(wt, keep_if_changed=False)
