from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import (
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.output_formatters import create_formatter
from vibe.core.programmatic import (
    ProgrammaticOptions,
    _new_programmatic_loop,
    run_programmatic,
)
from vibe.core.tasking import TaskBrief, TaskBudget, TaskManifestIdentity
from vibe.core.tasking._process_context import (
    TASK_PROCESS_CONTEXT_ENV,
    TaskProcessContext,
)
from vibe.core.types import OutputFormat
from vibe.core.usage import SpendPurpose
from vibe.core.usage._process_context import (
    SPEND_PROCESS_CONTEXT_ENV,
    SpendProcessContext,
    SpendProcessContextError,
)
from vibe.core.usage._session import SessionSpendAdapter, SpendAdmissionBlockedError


def _task_brief() -> TaskBrief:
    return TaskBrief(
        objective="Update parser",
        allowed_paths=["src/parser.py"],
        acceptance_checks=["focused"],
        budget=TaskBudget(max_calls=2),
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )


def _task_config():
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="task-process-v1",
        task_brief="Update parser",
        acceptance_contract="Focused check passes",
        allowed_paths=("src/**",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused", argv=("true",), timeout_seconds=5
            ),
        ),
    )
    return build_test_vibe_config(trusted_verification_recipe=recipe)


def test_programmatic_startup_rejects_malformed_spend_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SPEND_PROCESS_CONTEXT_ENV, "not-json")

    with pytest.raises(SpendProcessContextError):
        run_programmatic(
            build_test_vibe_config(),
            "unused",
            options=ProgrammaticOptions(no_worktree=True),
        )


def test_programmatic_startup_rejects_tampered_spend_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = SpendProcessContext(
        ledger_path=str(tmp_path / "missing-ledger"),
        session_scope_id="session:missing",
        agent_scope_id="agent:missing",
        purpose=SpendPurpose.TEAM,
    )
    monkeypatch.setenv(SPEND_PROCESS_CONTEXT_ENV, context.model_dump_json())

    with pytest.raises(SpendAdmissionBlockedError):
        run_programmatic(
            build_test_vibe_config(),
            "unused",
            options=ProgrammaticOptions(no_worktree=True),
        )


def test_programmatic_task_context_requires_spend_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_context = TaskProcessContext.from_brief(_task_brief())
    monkeypatch.setenv(TASK_PROCESS_CONTEXT_ENV, task_context.model_dump_json())
    monkeypatch.delenv(SPEND_PROCESS_CONTEXT_ENV, raising=False)

    with pytest.raises(SpendAdmissionBlockedError, match="requires.*spend scope"):
        run_programmatic(
            _task_config(), "unused", options=ProgrammaticOptions(no_worktree=True)
        )


def test_programmatic_task_context_rejects_broad_agent_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _task_config()
    task_context = TaskProcessContext.from_brief(_task_brief())
    root = SessionSpendAdapter.create(
        config.spend, "broad-task", ledger_path=tmp_path / "ledger"
    )
    broad = root.child_agent(
        task_brief_hash=task_context.brief_hash
    ).export_process_context()
    monkeypatch.setenv(TASK_PROCESS_CONTEXT_ENV, task_context.model_dump_json())
    monkeypatch.setenv(SPEND_PROCESS_CONTEXT_ENV, broad.model_dump_json())

    with pytest.raises(SpendAdmissionBlockedError, match="limits exceed"):
        run_programmatic(
            config, "unused", options=ProgrammaticOptions(no_worktree=True)
        )


@pytest.mark.asyncio
async def test_worker_task_loops_do_not_share_transcript_or_tool_state(
    tmp_path: Path,
) -> None:
    config = build_test_vibe_config()
    root = SessionSpendAdapter.create(
        config.spend, "fresh-worker", ledger_path=tmp_path / "ledger"
    )
    opts = ProgrammaticOptions(agent_name=BuiltinAgentName.AUTO_APPROVE)
    first_backend = FakeBackend([mock_llm_chunk(content="first result")])
    second_backend = FakeBackend([mock_llm_chunk(content="second result")])
    first = _new_programmatic_loop(
        config.model_copy(deep=True),
        opts,
        create_formatter(OutputFormat.TEXT),
        backend=first_backend,
        spend_adapter=root.child_agent(agent_id="agent:first-task"),
    )
    second = _new_programmatic_loop(
        config.model_copy(deep=True),
        opts,
        create_formatter(OutputFormat.TEXT),
        backend=second_backend,
        spend_adapter=root.child_agent(agent_id="agent:second-task"),
    )
    first._files_read["first-only.py"] = "fingerprint"

    try:
        [event async for event in first.act("first task transcript marker")]
        [event async for event in second.act("second task")]
    finally:
        await first.aclose()
        await first.telemetry_client.aclose()
        await second.aclose()
        await second.telemetry_client.aclose()

    assert "first-only.py" not in second._files_read
    assert first.tool_manager is not second.tool_manager
    assert first._verification_state is not second._verification_state
    second_request = second_backend.requests_messages[0]
    assert all(
        "first task transcript marker" not in (message.content or "")
        for message in second_request
    )
