from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest

from tests.trusted_verification import (
    HOST_ENVIRONMENT as _HOST_ENVIRONMENT,
    HOST_ENVIRONMENT_SHA256 as _HOST_ENVIRONMENT_SHA256,
    HOST_PYTHON as _HOST_PYTHON,
    HOST_PYTHON_SHA256 as _HOST_PYTHON_SHA256,
)
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
from vibe.core.tasking._policy import (
    BoundTaskContract,
    TaskContractAuthority,
    TaskContractError,
)
from vibe.core.teams._structured_attempt import evaluate_structured_attempt
from vibe.core.teams._task_checks import TaskCheckEvidence
from vibe.core.teams.manager import TeamManager
from vibe.core.teams.models import (
    LEGACY_TASK_PROTOCOL_VERSION,
    STRUCTURED_TASK_PROTOCOL_VERSION,
    Task,
    TaskStatus,
)
from vibe.core.teams.task_store import TaskStore
from vibe.core.teams.worker_loop import (
    WorkerTaskAttempt,
    run_team_worker_loop,
    worker_task_prompt,
)
from vibe.core.usage._session import SpendAdmissionBlockedError
from vibe.core.utils.io import write_safe
from vibe.core.verification_state import VerificationState


def _brief(*, objective: str = "Implement the parser fix") -> TaskBrief:
    return TaskBrief(
        objective=objective,
        inputs={"target": "vibe/core/parser.py:10"},
        allowed_paths=["vibe/core/parser.py", "tests/core/test_parser.py"],
        denied_paths=["vibe/core/agent_loop.py"],
        acceptance_checks=["focused"],
        budget=TaskBudget(max_tokens=5_000, max_calls=6),
        manifest=TaskManifestIdentity(name="implement-verify", version="1"),
    )


def _contract(
    brief: TaskBrief,
    workspace_root: Path,
    *,
    check_exit: int = 0,
    extra_checks: tuple[TrustedVerificationCheckConfig, ...] = (),
) -> BoundTaskContract:
    recipe = TrustedVerificationRecipeConfig(
        recipe_version="team-worker-v1",
        task_brief=brief.objective,
        acceptance_contract="Selected checks must pass",
        allowed_paths=("**",),
        checks=(
            TrustedVerificationCheckConfig(
                name="focused",
                argv=(
                    str(_HOST_PYTHON),
                    "-c",
                    f"import sys; print('focused'); sys.exit({check_exit})",
                ),
                executable_sha256=_HOST_PYTHON_SHA256,
                environment_attestation_path=str(_HOST_ENVIRONMENT),
                environment_attestation_sha256=_HOST_ENVIRONMENT_SHA256,
            ),
            *extra_checks,
        ),
    )
    return BoundTaskContract.bind(
        brief,
        authority=TaskContractAuthority.LEAD,
        workspace_root=workspace_root,
        verification_state=VerificationState.from_recipe(recipe),
    )


def test_legacy_completed_record_migrates_to_explicit_success(tmp_path: Path) -> None:
    payload = {
        "tasks": [
            {
                "id": "task-1",
                "description": "Legacy work",
                "status": "completed",
                "result": "done before the protocol upgrade",
            }
        ]
    }
    write_safe(tmp_path / "tasks.json", orjson.dumps(payload).decode())

    task = TaskStore(tmp_path).get_task("task-1")

    assert task is not None
    assert task.description == "Legacy work"
    assert task.protocol_version == LEGACY_TASK_PROTOCOL_VERSION
    assert task.outcome is not None
    assert task.outcome.status is TaskOutcomeStatus.SUCCEEDED
    assert task.outcome.summary == "done before the protocol upgrade"


def test_structured_task_persists_brief_and_legacy_description(tmp_path: Path) -> None:
    brief = _brief()
    created = TaskStore(tmp_path).add_task(brief)

    persisted = TaskStore(tmp_path).get_task(created.id)

    assert persisted is not None
    assert persisted.protocol_version == STRUCTURED_TASK_PROTOCOL_VERSION
    assert persisted.description == brief.objective
    assert persisted.brief == brief
    assert persisted.outcome is None
    assert "TASK_BRIEF_JSON:" in persisted.prompt


def test_expired_structured_task_is_blocked_before_team_claim(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    brief = _brief().model_copy(
        update={"deadline": datetime.now(UTC) - timedelta(seconds=1)}
    )
    task = store.add_task(brief)

    assert store.claim_task(task.id, "worker") is None

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.COMPLETED
    assert current.assignee is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.BLOCKED
    assert current.outcome.manifest == brief.manifest
    assert "deadline" in current.outcome.diagnostics[0]


def test_structured_plain_prose_cannot_complete_or_succeed(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    assert store.claim_task(task.id, "worker") is not None

    completion = store.complete_task(
        task.id,
        "Implemented and all tests look good",
        actor="worker",
        authoritative=True,
    )

    assert completion is not None
    assert completion.status is TaskStatus.PENDING
    assert completion.outcome is not None
    assert completion.outcome.status is TaskOutcomeStatus.RETRYABLE
    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.assignee is None
    assert current.claimed_at is None
    assert current.completed_at is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE
    assert current.outcome.succeeded is False


@pytest.mark.parametrize(
    "marker,status",
    [
        ("SUCCEEDED", TaskOutcomeStatus.SUCCEEDED),
        ("FAILED", TaskOutcomeStatus.FAILED),
        ("BLOCKED", TaskOutcomeStatus.BLOCKED),
    ],
)
def test_structured_explicit_terminal_marker_completes_lifecycle(
    tmp_path: Path, marker: str, status: TaskOutcomeStatus
) -> None:
    store = TaskStore(tmp_path)
    brief = _brief()
    task = store.add_task(brief)
    assert store.claim_task(task.id, "worker") is not None

    completed = store.complete_task(
        task.id,
        f"Attempt finished\nTASK_OUTCOME: {marker}",
        actor="worker",
        authoritative=True,
    )

    assert completed is not None
    assert completed.status is TaskStatus.COMPLETED
    assert completed.outcome is not None
    assert completed.outcome.status is status
    assert completed.outcome.manifest == brief.manifest


def test_structured_retryable_marker_requeues_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    assert store.claim_task(task.id, "worker") is not None

    completion = store.complete_task(
        task.id,
        "Provider timed out\nTASK_OUTCOME: RETRYABLE",
        actor="worker",
        authoritative=True,
    )

    assert completion is not None
    assert completion.status is TaskStatus.PENDING
    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.assignee is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE


def test_retry_prompt_is_bounded_and_omits_prior_result(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    brief = _brief()
    task = store.add_task(brief)
    assert store.claim_task(task.id, "worker1") is not None
    outcome = TaskOutcome(
        status=TaskOutcomeStatus.RETRYABLE,
        summary="FULL_PRIOR_TRANSCRIPT",
        diagnostics=["exact failed check", "x" * 20_000],
        evidence=["focused: exit 3"],
        manifest=brief.manifest,
    )
    completed = store.complete_task(
        task.id, outcome, actor="worker1", authoritative=True
    )
    assert completed is not None
    assert completed.result == "FULL_PRIOR_TRANSCRIPT"

    claimed = store.claim_task(task.id, "worker2")

    assert claimed is not None
    prompt = worker_task_prompt(claimed)
    assert claimed.result is None
    assert "exact failed check" in prompt
    assert "FULL_PRIOR_TRANSCRIPT" not in prompt
    assert len(prompt) <= len(task.prompt) + 5_000


def test_structured_outcome_cannot_change_manifest_identity(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    brief = _brief()
    task = store.add_task(brief)
    assert store.claim_task(task.id, "worker") is not None
    forged = TaskOutcome(
        status=TaskOutcomeStatus.SUCCEEDED,
        summary="done",
        manifest=TaskManifestIdentity(name="full-access", version="99"),
    )

    completion = store.complete_task(
        task.id, forged, actor="worker", authoritative=True
    )

    assert completion is not None
    assert completion.status is TaskStatus.PENDING
    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE
    assert current.outcome.manifest == brief.manifest


def test_failed_dependency_does_not_unblock_dependent_task(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    dependency = store.add_task(_brief(objective="Build dependency"))
    dependent = store.add_task(
        _brief(objective="Use dependency"), dependencies=[dependency.id]
    )
    assert store.claim_task(dependency.id, "worker") is not None
    assert (
        store.complete_task(
            dependency.id,
            "Could not build\nTASK_OUTCOME: FAILED",
            actor="worker",
            authoritative=True,
        )
        is not None
    )

    assert store.claim_task(dependent.id, "worker") is None
    assert dependent.id not in {task.id for task in store.get_available_tasks()}


@pytest.mark.asyncio
async def test_manager_fires_completion_hook_only_for_terminal_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    import vibe.core.teams.manager as manager_module

    monkeypatch.setattr(manager_module, "_team_dir_for", lambda _name: tmp_path)
    manager = TeamManager("lead", team_name="task-protocol")
    task = await manager.add_team_task(_brief())
    assert manager.task_store.claim_task(task.id, "worker") is not None
    dispatch = AsyncMock()
    monkeypatch.setattr(manager, "_dispatch_hook", dispatch)

    retry = await manager.complete_team_task(task.id, "ordinary prose")

    assert retry is not None
    assert retry.status is TaskStatus.PENDING
    dispatch.assert_not_awaited()

    assert manager.task_store.claim_task(task.id, "worker") is not None
    terminal = await manager.complete_team_task(
        task.id, "Finished\nTASK_OUTCOME: FAILED"
    )

    assert terminal is not None
    assert terminal.status is TaskStatus.COMPLETED
    dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_plain_prose_does_not_autocomplete_structured_task(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    prompts: list[str] = []

    async def run_task(claimed: Task) -> WorkerTaskAttempt:
        prompts.append(worker_task_prompt(claimed))
        assert claimed.brief is not None
        return WorkerTaskAttempt(
            "Implemented and tests pass", _contract(claimed.brief, tmp_path)
        )

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert len(prompts) == 1
    assert "TASK_BRIEF_JSON:" in prompts[0]
    assert current.status is TaskStatus.PENDING
    assert current.assignee is None
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE


@pytest.mark.asyncio
async def test_worker_completes_structured_task_with_terminal_marker(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())

    async def run_task(claimed: Task) -> WorkerTaskAttempt:
        assert "TASK_OUTCOME: SUCCEEDED" in worker_task_prompt(claimed)
        assert claimed.brief is not None
        return WorkerTaskAttempt(
            "Implemented and checked\nTASK_OUTCOME: SUCCEEDED",
            _contract(claimed.brief, tmp_path),
        )

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.COMPLETED
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_worker_failed_trusted_check_is_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())

    async def run_task(claimed: Task) -> WorkerTaskAttempt:
        assert claimed.brief is not None
        return WorkerTaskAttempt(
            "Implemented\nTASK_OUTCOME: SUCCEEDED",
            _contract(claimed.brief, tmp_path, check_exit=3),
        )

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.PENDING
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.RETRYABLE
    diagnostic = current.outcome.diagnostics[0]
    evidence = current.outcome.evidence[0]
    assert "exit 3" in diagnostic

    claimed = store.claim_task(task.id, "worker2")

    assert claimed is not None
    assert claimed.outcome is not None
    retry_prompt = worker_task_prompt(claimed)
    assert diagnostic in retry_prompt
    assert evidence in retry_prompt
    assert "Retry context:" in retry_prompt


@pytest.mark.asyncio
async def test_failed_check_repairs_in_same_attempt_before_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe.core.teams._structured_attempt as attempt_module

    brief = _brief()
    contract = _contract(brief, tmp_path)
    failed = TaskCheckEvidence(
        name="focused",
        argv=("focused",),
        cwd=str(tmp_path),
        exit_code=3,
        timed_out=False,
        duration_ms=1,
        stdout="exact assertion failure",
        stderr="",
    )
    passed = failed.model_copy(update={"exit_code": 0, "stdout": "passed"})
    check_results = [((failed,), None), ((passed,), None)]
    monkeypatch.setattr(
        attempt_module, "run_guarded_task_checks", lambda *_args: check_results.pop(0)
    )
    repair_prompts: list[str] = []

    async def repair(prompt: str) -> str:
        repair_prompts.append(prompt)
        return "Fixed the assertion\nTASK_OUTCOME: SUCCEEDED"

    outcome = await evaluate_structured_attempt(
        brief,
        contract,
        "Initial implementation\nTASK_OUTCOME: SUCCEEDED",
        repair=repair,
    )

    assert outcome.succeeded
    assert len(repair_prompts) == 1
    assert "exact assertion failure" in repair_prompts[0]
    assert check_results == []


@pytest.mark.asyncio
async def test_same_worker_spend_exhaustion_blocks_stable_task_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vibe.core.teams._structured_attempt as attempt_module

    brief = _brief()
    contract = _contract(brief, tmp_path)
    failed = TaskCheckEvidence(
        name="focused",
        argv=("focused",),
        cwd=str(tmp_path),
        exit_code=3,
        timed_out=False,
        duration_ms=1,
        stdout="exact assertion failure",
        stderr="",
    )
    monkeypatch.setattr(
        attempt_module, "run_guarded_task_checks", lambda *_args: ((failed,), None)
    )

    async def repair(_prompt: str) -> str:
        raise SpendAdmissionBlockedError("task spend is exhausted")

    outcome = await evaluate_structured_attempt(
        brief,
        contract,
        "Initial implementation\nTASK_OUTCOME: SUCCEEDED",
        repair=repair,
    )

    assert outcome.status is TaskOutcomeStatus.BLOCKED
    assert "exact assertion failure" in outcome.diagnostics[0]
    assert "task spend is exhausted" in outcome.diagnostics[1]


@pytest.mark.parametrize(
    "error",
    [
        TaskContractError("untrusted task manifest"),
        SpendAdmissionBlockedError("task spend is exhausted"),
    ],
)
@pytest.mark.asyncio
async def test_worker_blocks_stable_pre_attempt_failures_without_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    attempts = 0

    async def run_task(_claimed: Task) -> WorkerTaskAttempt:
        nonlocal attempts
        attempts += 1
        raise error

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert attempts == 1
    assert current.status is TaskStatus.COMPLETED
    assert current.outcome is not None
    assert current.outcome.status is TaskOutcomeStatus.BLOCKED
    assert type(error).__name__ in current.outcome.diagnostics[0]


@pytest.mark.asyncio
async def test_worker_runs_only_selected_trusted_checks(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBE_TEAM_DIR", str(tmp_path))
    monkeypatch.setenv("VIBE_TEAMMATE_NAME", "worker1")
    marker = tmp_path / "unselected-ran"
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())

    async def run_task(claimed: Task) -> WorkerTaskAttempt:
        assert claimed.brief is not None
        unselected = TrustedVerificationCheckConfig(
            name="unselected",
            argv=(
                str(_HOST_PYTHON),
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ),
            executable_sha256=_HOST_PYTHON_SHA256,
            environment_attestation_path=str(_HOST_ENVIRONMENT),
            environment_attestation_sha256=_HOST_ENVIRONMENT_SHA256,
        )
        return WorkerTaskAttempt(
            "Implemented\nTASK_OUTCOME: SUCCEEDED",
            _contract(claimed.brief, tmp_path, extra_checks=(unselected,)),
        )

    await run_team_worker_loop(
        run_task, idle_poll_s=0.01, max_idle_rounds=2, lease_s=900.0
    )

    store.reload()
    current = store.get_task(task.id)
    assert current is not None
    assert current.outcome is not None
    assert current.outcome.succeeded
    assert not marker.exists()


def test_structured_worker_cannot_self_complete(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    task = store.add_task(_brief())
    assert store.claim_task(task.id, "worker1") is not None

    refused = store.complete_task(
        task.id, "done\nTASK_OUTCOME: SUCCEEDED", actor="worker1"
    )

    assert refused is None
    current = store.get_task(task.id)
    assert current is not None
    assert current.status is TaskStatus.IN_PROGRESS
