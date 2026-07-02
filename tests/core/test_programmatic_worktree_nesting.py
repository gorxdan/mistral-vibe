from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.mock_backend_factory import mock_backend_factory
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.programmatic import ProgrammaticOptions, run_programmatic
from vibe.core.types import Backend
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
