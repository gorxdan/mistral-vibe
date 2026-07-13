from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result
from vibe.core.agents.manager import AgentManager
from vibe.core.config import (
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.tasking import (
    TaskBrief,
    TaskBudget,
    TaskManifestIdentity,
    TaskOutcome,
    TaskOutcomeStatus,
)
from vibe.core.tasking._candidate import TaskCandidateValidation
from vibe.core.teams._task_checks import TaskCheckEvidence
from vibe.core.tools.background import BackgroundRegistry
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.task import Task, TaskArgs, TaskResult, TaskToolConfig
from vibe.core.types import (
    AssistantEvent,
    LLMMessage,
    Role,
    ToolCallEvent,
    ToolResultEvent,
)


async def _noop_completion_wake() -> None:
    pass


from vibe.core.verification_state import VerificationState


def _recipe() -> TrustedVerificationRecipeConfig:
    return TrustedVerificationRecipeConfig(
        recipe_version="task-contract-v1",
        task_brief="Implement the parser fix",
        acceptance_contract="The focused check must pass",
        allowed_paths=("vibe/core/parser.py", "tests/core/test_parser.py"),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused", argv=(sys.executable, "-c", "raise SystemExit(0)")
            ),
        ),
    )


def _brief() -> TaskBrief:
    return TaskBrief(
        objective="Implement the parser fix",
        inputs={"target": "vibe/core/parser.py:10"},
        allowed_paths=["vibe/core/parser.py", "tests/core/test_parser.py"],
        denied_paths=["vibe/core/agent_loop.py"],
        acceptance_checks=["focused"],
        budget=TaskBudget(max_tokens=5_000, max_calls=6),
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )


def _verify_brief() -> TaskBrief:
    return _brief().model_copy(
        update={"manifest": TaskManifestIdentity(name="verify", version="1")}
    )


def _verification_report(verdict: str = "PASS", result: str = "PASS") -> str:
    return (
        "### Check: focused\n"
        "**Command run:** focused\n"
        "**Output observed:** passed\n"
        f"**Result: {result}**\n"
        f"VERDICT: {verdict}"
    )


def _ctx() -> InvokeContext:
    config = build_test_vibe_config(
        include_project_context=False, include_prompt_detail=False
    )
    return InvokeContext(
        tool_call_id="task-contract",
        agent_manager=AgentManager(lambda: config),
        verification_state=VerificationState.from_recipe(_recipe()),
    )


def test_task_args_keep_legacy_string_api() -> None:
    args = TaskArgs(task="inspect the parser", agent="explore")

    assert args.task == "inspect the parser"
    assert args.prompt == "inspect the parser"
    assert args.summary == "inspect the parser"
    assert args.brief is None


def test_task_args_compile_structured_brief_without_verbose_display() -> None:
    brief = _brief()
    args = TaskArgs.model_validate({
        "task": brief.model_dump(mode="json"),
        "agent": "explore",
    })
    event = ToolCallEvent(
        tool_call_id="task-contract", tool_name="task", args=args, tool_class=Task
    )

    assert args.prompt.startswith("Execute this serialized task contract")
    assert args.summary == "Implement the parser fix"
    assert args.brief == brief
    assert Task.get_call_display(event).summary == (
        "Running explore agent: Implement the parser fix"
    )


def test_task_args_keep_serialized_task_brief_as_legacy_text() -> None:
    serialized = json.dumps(_brief().model_dump(mode="json"))

    args = TaskArgs(task=serialized, agent="worker")

    assert args.brief is None
    assert args.prompt == serialized


def test_task_args_keep_invalid_contract_shaped_json_as_legacy_text() -> None:
    payload = _brief().model_dump(mode="json")
    payload["allowed_paths"] = "vibe/core/parser.py"
    serialized = json.dumps(payload)

    args = TaskArgs(task=serialized, agent="worker")

    assert args.brief is None
    assert args.prompt == serialized


def test_task_args_keep_invalid_contract_shaped_json_syntax_as_legacy_text() -> None:
    task = '{"objective":"fix parser","manifest":{"name":"verify"'

    args = TaskArgs(task=task, agent="verifier")

    assert args.brief is None
    assert args.prompt == task


def test_task_args_leave_unrelated_json_as_legacy_text() -> None:
    task = json.dumps({"query": "inspect parser", "limit": 3})

    args = TaskArgs(task=task, agent="explore")

    assert args.task == task
    assert args.brief is None


def test_write_manifest_task_call_is_not_read_only() -> None:
    ctx = _ctx()
    assert ctx.agent_manager is not None
    args = TaskArgs(task=_brief(), agent="explore", async_run=False)

    assert Task.call_is_read_only(args, agent_manager=ctx.agent_manager) is False


@pytest.mark.asyncio
async def test_structured_brief_reaches_agent_and_returns_terminal_outcome() -> None:
    prompts: list[str] = []

    async def mock_act(prompt: str):
        prompts.append(prompt)
        yield AssistantEvent(
            content="Implemented and checked.\nTASK_OUTCOME: SUCCEEDED"
        )

    mock_loop = MagicMock()
    mock_loop.act = mock_act
    mock_loop.messages = [LLMMessage(role=Role.ASSISTANT, content="done")]
    ctx = _ctx()
    spend_adapter = MagicMock()
    ctx.spend_adapter = spend_adapter

    tool = Task(
        config_getter=lambda: TaskToolConfig(isolation="off"), state=BaseToolState()
    )
    with patch(
        "vibe.core.tools.builtins.task.AgentLoop", return_value=mock_loop
    ) as loop_cls:
        result = await collect_result(
            tool.run(TaskArgs(task=_brief(), agent="worker", async_run=False), ctx)
        )

    assert isinstance(result, TaskResult)
    assert result.completed is True
    assert result.outcome is not None
    assert result.outcome.status is TaskOutcomeStatus.SUCCEEDED
    assert result.outcome.manifest == _brief().manifest
    assert prompts == [TaskArgs(task=_brief(), agent="worker").prompt]
    contract = loop_cls.call_args.kwargs["params"].task_contract
    assert contract is not None
    assert contract.acceptance_check_ids == ("focused",)
    assert contract.allowed_tools == {
        "edit",
        "glob",
        "grep",
        "lsp",
        "read",
        "task_checks",
        "todo",
        "write_file",
    }
    limits = spend_adapter.child_task.call_args.kwargs["limits"]
    assert limits.max_total_tokens == 5_000
    assert limits.max_calls == 6


@pytest.mark.asyncio
async def test_serialized_task_brief_dispatches_as_legacy_without_recipe() -> None:
    serialized = json.dumps(_brief().model_dump(mode="json"))
    ctx = _ctx()
    ctx.verification_state = VerificationState()
    prompts: list[str] = []

    async def mock_act(prompt: str):
        prompts.append(prompt)
        yield AssistantEvent(content="Inspected the serialized request")

    mock_loop = MagicMock()
    mock_loop.act = mock_act
    mock_loop.messages = [LLMMessage(role=Role.ASSISTANT, content="done")]
    tool = Task(
        config_getter=lambda: TaskToolConfig(isolation="off"), state=BaseToolState()
    )

    with patch("vibe.core.tools.builtins.task.AgentLoop", return_value=mock_loop):
        result = await collect_result(
            tool.run(TaskArgs(task=serialized, agent="worker", async_run=False), ctx)
        )

    assert isinstance(result, TaskResult)
    assert result.completed
    assert prompts == [serialized]


@pytest.mark.asyncio
async def test_read_only_agent_rejects_write_capable_manifest() -> None:
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    with (
        patch("vibe.core.tools.builtins.task.AgentLoop") as loop,
        pytest.raises(ToolError, match="read-only agent.*write-capable"),
    ):
        await collect_result(
            tool.run(TaskArgs(task=_brief(), agent="explore", async_run=False), _ctx())
        )

    loop.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_rejects_non_verification_manifest() -> None:
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    with pytest.raises(ToolError, match="verify@1"):
        await collect_result(
            tool.run(TaskArgs(task=_brief(), agent="verifier", async_run=False), _ctx())
        )


@pytest.mark.asyncio
async def test_verifier_accepts_verify_manifest_and_terminal_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.workspace_fingerprint", lambda: "candidate"
    )
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )

    async def mock_act(prompt: str):
        prompts.append(prompt)
        yield AssistantEvent(content=_verification_report())

    mock_loop = MagicMock()
    mock_loop.act = mock_act
    mock_loop.messages = [LLMMessage(role=Role.ASSISTANT, content="done")]
    ctx = _ctx()
    ctx.scratchpad_dir = tmp_path
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    with patch("vibe.core.tools.builtins.task.AgentLoop", return_value=mock_loop):
        result = await collect_result(
            tool.run(
                TaskArgs(task=_verify_brief(), agent="verifier", async_run=False), ctx
            )
        )

    assert isinstance(result, TaskResult)
    assert result.outcome is not None
    assert result.outcome.succeeded
    assert prompts[0].endswith("VERDICT: PASS|FAIL|PARTIAL")
    assert "It is cleaned automatically" in prompts[0]
    assert "Do not create, copy, move, link, or remove files" in prompts[0]
    assert "read and write files here" not in prompts[0]
    assert ctx.verification_state is not None
    assert ctx.verification_state.has_pass()


@pytest.mark.asyncio
async def test_in_process_success_requires_trusted_check_evidence() -> None:
    recipe = _recipe().model_copy(
        update={
            "checks": (
                TrustedVerificationCheckConfig(
                    name="focused",
                    argv=(
                        sys.executable,
                        "-c",
                        "import sys; print('exact check failure'); sys.exit(9)",
                    ),
                ),
            )
        }
    )
    ctx = _ctx()
    ctx.verification_state = VerificationState.from_recipe(recipe)
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    outcome = await tool._finalize_in_process_outcome(
        TaskArgs(task=_brief(), agent="worker"),
        ctx,
        "Done\nTASK_OUTCOME: SUCCEEDED",
        completed=True,
        forced_status=None,
        diagnostic=None,
    )

    assert outcome.status is TaskOutcomeStatus.RETRYABLE
    assert "exit 9" in outcome.diagnostics[0]
    assert "exact check failure" in outcome.diagnostics[0]


@pytest.mark.asyncio
async def test_expired_structured_brief_is_blocked_before_agent_dispatch() -> None:
    brief = _brief().model_copy(
        update={"deadline": datetime.now(UTC) - timedelta(seconds=1)}
    )
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())

    with patch("vibe.core.tools.builtins.task.AgentLoop") as loop:
        result = await collect_result(
            tool.run(TaskArgs(task=brief, agent="worker", async_run=False), _ctx())
        )

    assert isinstance(result, TaskResult)
    assert result.completed is False
    assert result.outcome is not None
    assert result.outcome.status is TaskOutcomeStatus.BLOCKED
    assert "deadline" in result.outcome.diagnostics[0]
    loop.assert_not_called()


@pytest.mark.asyncio
async def test_async_isolated_task_preserves_structured_outcome() -> None:
    from vibe.core.workflows.runtime import IsolatedResult

    registry = BackgroundRegistry()
    registry.attach_completion_callback(_noop_completion_wake)
    ctx = _ctx()
    ctx.background_registry = registry
    tool = Task(
        config_getter=lambda: TaskToolConfig(isolation="always"), state=BaseToolState()
    )

    spawn_args: dict[str, object] = {}

    async def isolated(*args, **kwargs) -> IsolatedResult:
        spawn_args.update(kwargs)
        return IsolatedResult(output="Could not finish.\nTASK_OUTCOME: FAILED")

    with patch("vibe.core.tools.builtins.task.run_isolated_agent", isolated):
        result = await collect_result(
            tool.run(TaskArgs(task=_brief(), agent="worker"), ctx)
        )
        assert isinstance(result, TaskResult)
        assert result.task_id is not None
        await asyncio.sleep(0.05)

    [completion] = registry.pop_async_completions()
    assert completion.outcome is not None
    assert completion.outcome.status is TaskOutcomeStatus.FAILED
    assert spawn_args["deliver"] is False
    assert spawn_args["keep_worktree"] is True
    assert spawn_args["task_context"] is not None


@pytest.mark.asyncio
async def test_structured_isolated_scope_failure_never_delivers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe.core.workflows.runtime import IsolatedResult
    from vibe.core.worktree import ephemeral

    ctx = _ctx()
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    contract = tool._bind_contract(_brief(), ctx)
    candidate = TaskCandidateValidation(
        changed_paths=("outside.py",),
        scope_passed=False,
        diagnostics=("candidate changed paths outside the task contract: outside.py",),
    )
    delivered: list[object] = []
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.validate_task_candidate", lambda *args: candidate
    )
    monkeypatch.setattr(
        ephemeral, "deliver_ephemeral_worktree", lambda wt: delivered.append(wt) or True
    )
    monkeypatch.setattr(ephemeral, "remove_ephemeral_worktree", lambda *a, **k: False)
    wt = SimpleNamespace(path=tmp_path, base_sha="base", branch="candidate")

    result = await tool._finalize_isolated_result(
        TaskArgs(task=_brief(), agent="worker"),
        ctx,
        IsolatedResult(output="Done\nTASK_OUTCOME: SUCCEEDED", returncode=0, wt=wt),
        contract=contract,
        verification_attempt=None,
    )

    assert delivered == []
    assert result.outcome is not None
    assert result.outcome.status is TaskOutcomeStatus.BLOCKED
    assert result.outcome.changed_paths == ["outside.py"]
    assert result.branch == "candidate"


@pytest.mark.asyncio
async def test_structured_isolated_failed_check_never_delivers(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe.core.workflows.runtime import IsolatedResult
    from vibe.core.worktree import ephemeral

    ctx = _ctx()
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    contract = tool._bind_contract(_brief(), ctx)
    check = TaskCheckEvidence(
        name="focused",
        argv=("uv", "run", "pytest"),
        cwd=str(tmp_path),
        exit_code=7,
        timed_out=False,
        duration_ms=1,
        stdout="exact failure",
        stderr="",
    )
    candidate = TaskCandidateValidation(
        changed_paths=("vibe/core/parser.py",),
        scope_passed=True,
        checks=(check,),
        diagnostics=("check 'focused': exit 7\nstdout:\nexact failure",),
    )
    delivered: list[object] = []
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.validate_task_candidate", lambda *args: candidate
    )
    monkeypatch.setattr(
        ephemeral, "deliver_ephemeral_worktree", lambda wt: delivered.append(wt) or True
    )
    monkeypatch.setattr(ephemeral, "remove_ephemeral_worktree", lambda *a, **k: False)
    wt = SimpleNamespace(path=tmp_path, base_sha="base", branch="candidate")

    result = await tool._finalize_isolated_result(
        TaskArgs(task=_brief(), agent="worker"),
        ctx,
        IsolatedResult(output="Done\nTASK_OUTCOME: SUCCEEDED", returncode=0, wt=wt),
        contract=contract,
        verification_attempt=None,
    )

    assert delivered == []
    assert result.outcome is not None
    assert result.outcome.status is TaskOutcomeStatus.RETRYABLE
    assert "exact failure" in result.outcome.diagnostics[0]


@pytest.mark.asyncio
async def test_structured_isolated_verifier_pass_records_only_after_validation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vibe.core.workflows.runtime import IsolatedResult
    from vibe.core.worktree import ephemeral

    ctx = _ctx()
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )
    tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    contract = tool._bind_contract(_verify_brief(), ctx)
    check = TaskCheckEvidence(
        name="focused",
        argv=("uv", "run", "pytest"),
        cwd=str(tmp_path),
        exit_code=0,
        timed_out=False,
        duration_ms=1,
        stdout="passed",
        stderr="",
    )
    candidate = TaskCandidateValidation(
        changed_paths=(), scope_passed=True, checks=(check,)
    )
    monkeypatch.setattr(
        "vibe.core.tools.builtins.task.validate_task_candidate", lambda *args: candidate
    )
    monkeypatch.setattr(ephemeral, "deliver_ephemeral_worktree", lambda wt: True)
    monkeypatch.setattr(ephemeral, "remove_ephemeral_worktree", lambda *a, **k: True)
    wt = SimpleNamespace(path=tmp_path, base_sha="base", branch="candidate")

    result = await tool._finalize_isolated_result(
        TaskArgs(task=_verify_brief(), agent="verifier"),
        ctx,
        IsolatedResult(output=_verification_report(), returncode=0, wt=wt),
        contract=contract,
        verification_attempt=None,
    )

    assert result.outcome is not None
    assert result.outcome.succeeded
    assert ctx.verification_state is not None
    assert ctx.verification_state.has_pass()


def test_task_result_display_uses_explicit_failed_outcome() -> None:
    result = TaskResult(
        response="Could not proceed",
        completed=True,
        outcome=TaskOutcome(
            status=TaskOutcomeStatus.BLOCKED, summary="Subagent could not proceed"
        ),
    )
    event = ToolResultEvent(
        tool_call_id="task-contract", tool_name="task", result=result, tool_class=Task
    )

    display = Task.get_result_display(event)

    assert display.success is False
    assert display.message == "Agent outcome: blocked"
