from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import ModelConfig, SpendConfig
from vibe.core.llm.types import CompletionRequest
from vibe.core.teams.manager import TeamManager
from vibe.core.types import LLMMessage, Role
from vibe.core.usage import SpendBroker, SpendEnvelopeLimits, SpendScopeKind
from vibe.core.usage._process_context import (
    SPEND_PROCESS_CONTEXT_ENV,
    decode_spend_process_context,
)
from vibe.core.usage._session import SessionSpendAdapter, SpendBudgetExceededError


class _ImmediateProcess:
    pid = os.getpid()
    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""


@pytest.mark.asyncio
async def test_team_processes_share_one_group_with_distinct_agent_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, str]] = []

    async def fake_exec(*_args: object, **kwargs: object) -> _ImmediateProcess:
        env = kwargs["env"]
        assert isinstance(env, dict)
        captured.append(env)
        return _ImmediateProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        "vibe.core.teams.manager._team_dir_for", lambda name: tmp_path / name
    )
    root = SessionSpendAdapter.create(
        SpendConfig(max_calls=2, enforce_limits=True),
        "team-root",
        ledger_path=tmp_path / "ledger",
    )
    manager = TeamManager("lead", team_name="shared-spend", spend_adapter=root)

    await manager.spawn_teammate("alice", "one", max_turns=1)
    await manager.spawn_teammate("bob", "two", max_turns=1)
    await asyncio.gather(*manager._teammate_tasks.values())

    contexts = [
        decode_spend_process_context(env[SPEND_PROCESS_CONTEXT_ENV]) for env in captured
    ]
    assert len({context.agent_scope_id for context in contexts}) == 2
    assert {context.session_scope_id for context in contexts} == {root.session_scope_id}
    broker = SpendBroker(root.ledger_path)
    scopes = [broker.get_envelope(context.agent_scope_id) for context in contexts]
    assert all(scope is not None for scope in scopes)
    parents = {scope.parent_scope_id for scope in scopes if scope is not None}
    assert parents == {"team:shared-spend"}
    group = broker.get_envelope("team:shared-spend")
    assert group is not None
    assert group.kind is SpendScopeKind.TEAM


@pytest.mark.asyncio
async def test_team_spawn_removes_stale_inherited_spend_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    async def fake_exec(*_args: object, **kwargs: object) -> _ImmediateProcess:
        env = kwargs["env"]
        assert isinstance(env, dict)
        captured.update(env)
        return _ImmediateProcess()

    monkeypatch.setenv(SPEND_PROCESS_CONTEXT_ENV, "stale")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(
        "vibe.core.teams.manager._team_dir_for", lambda name: tmp_path / name
    )
    manager = TeamManager("lead", team_name="no-spend")

    await manager.spawn_teammate("alice", "one", max_turns=1)
    await asyncio.gather(*manager._teammate_tasks.values())

    assert SPEND_PROCESS_CONTEXT_ENV not in captured


@pytest.mark.asyncio
async def test_team_task_retry_reuses_one_cumulative_scope(tmp_path: Path) -> None:
    root = SessionSpendAdapter.create(
        SpendConfig(max_calls=10, enforce_limits=True),
        "team-retry",
        ledger_path=tmp_path / "ledger",
    )
    first_worker = root.child_agent(
        group_kind=SpendScopeKind.TEAM,
        group_id="team:retry-budget",
        agent_id="agent:worker-one",
    )
    second_worker = root.child_agent(
        group_kind=SpendScopeKind.TEAM,
        group_id="team:retry-budget",
        agent_id="agent:worker-two",
    )
    brief_hash = "a" * 64
    limits = SpendEnvelopeLimits(max_calls=1)
    first_attempt = first_worker.child_task(
        task_id="task-1", task_brief_hash=brief_hash, limits=limits
    )
    retry_attempt = second_worker.child_task(
        task_id="task-1", task_brief_hash=brief_hash, limits=limits
    )
    request = CompletionRequest(
        model=ModelConfig(name="test", alias="test", provider="test"),
        messages=[LLMMessage(role=Role.USER, content="work")],
        max_tokens=1,
    )

    await first_attempt.complete(FakeBackend(mock_llm_chunk()), request)
    denied_backend = FakeBackend(mock_llm_chunk())
    with pytest.raises(SpendBudgetExceededError):
        await retry_attempt.complete(denied_backend, request)

    assert first_attempt.agent_scope_id == retry_attempt.agent_scope_id
    assert denied_backend.requests_messages == []
    snapshot = SpendBroker(root.ledger_path).snapshot(first_attempt.agent_scope_id)
    assert snapshot.spent_calls == 1
    assert snapshot.rejected_calls == 1
