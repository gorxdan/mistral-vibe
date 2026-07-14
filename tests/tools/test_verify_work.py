from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from git import Repo
from pydantic import ValidationError
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.trusted_verification import (
    HOST_ENVIRONMENT as _HOST_ENVIRONMENT,
    HOST_ENVIRONMENT_SHA256 as _HOST_ENVIRONMENT_SHA256,
    HOST_PYTHON as _HOST_PYTHON,
    HOST_PYTHON_SHA256 as _HOST_PYTHON_SHA256,
)
from vibe.core._verification_receipt import (
    VerificationReceipt,
    VerificationReceiptStore,
)
from vibe.core.agents.manager import AgentManager
from vibe.core.config import (
    TrustedVerificationCheckConfig,
    TrustedVerificationRecipeConfig,
)
from vibe.core.tools.base import BaseToolState, InvokeContext, ToolError
from vibe.core.tools.builtins.land_work import (
    LandWork,
    LandWorkArgs,
    LandWorkConfig,
    _require_verification_note,
)
from vibe.core.tools.builtins.verify_work import (
    VerifyWork,
    VerifyWorkArgs,
    VerifyWorkConfig,
    _await_trusted_recipe_completion,
)
from vibe.core.utils.io import write_safe
from vibe.core.verification_contract import parse_verification_report
from vibe.core.verification_state import VerificationState, VerifierAttemptDisposition
from vibe.core.worktree.manager import WorktreeHandle, worktree_manager


def _recipe(marker: str = "safe") -> TrustedVerificationRecipeConfig:
    return TrustedVerificationRecipeConfig(
        recipe_version=f"{marker}-v1",
        task_brief=f"Implement the {marker} candidate",
        acceptance_contract=f"The {marker} check must pass",
        allowed_paths=("candidate.txt",),
        checks=(
            TrustedVerificationCheckConfig(
                name=f"{marker}-check",
                argv=(str(_HOST_PYTHON), "-c", f"print('{marker} Total: 1')"),
                cwd=".",
                timeout_seconds=10,
                executable_sha256=_HOST_PYTHON_SHA256,
                environment_attestation_path=str(_HOST_ENVIRONMENT),
                environment_attestation_sha256=_HOST_ENVIRONMENT_SHA256,
                required_output_patterns=(marker,),
                test_count_pattern=r"Total:\s*(?P<count>\d+)",
                minimum_test_count=1,
                custom_runner=True,
            ),
        ),
    )


def _report() -> str:
    return (
        "### Check: independent review\n"
        "**Command run:**\n"
        "  uv run pytest tests/tools/test_verify_work.py\n"
        "**Output observed:**\n"
        "  focused verification passed\n"
        "**Result: PASS**\n\n"
        "VERDICT: PASS"
    )


def _record_current_verifier_pass(state: VerificationState) -> int:
    generation = state.begin_verifier_attempt()
    assert state.record_verifier_result(
        generation,
        VerifierAttemptDisposition.PASS,
        "Verifier PASS was recorded for the current candidate.",
    )
    state.record_verifier_pass(
        parse_verification_report(_report()), verifier_attempt_generation=generation
    )
    return generation


class _FakeConfig:
    verification_subsystem = True

    def __init__(self, recipe: TrustedVerificationRecipeConfig | None) -> None:
        self.trusted_verification_recipe = recipe


class _FakeAgentManager:
    def __init__(self, recipe: TrustedVerificationRecipeConfig | None) -> None:
        self.config = _FakeConfig(recipe)


@pytest.fixture(autouse=True)
def _clear_active_worktree():
    yield
    worktree_manager._active = None


def _linked_candidate(tmp_path: Path) -> tuple[Repo, Repo, WorktreeHandle]:
    main_path = tmp_path / "main"
    main_path.mkdir()
    main_repo = Repo.init(main_path)
    with main_repo.config_writer() as config:
        config.set_value("user", "name", "Test")
        config.set_value("user", "email", "test@example.com")
    write_safe(main_path / "README.md", "base\n")
    main_repo.index.add(["README.md"])
    base_sha = main_repo.index.commit("base").hexsha

    branch = "vibe/verified-candidate"
    worktree_path = tmp_path / "candidate"
    main_repo.git.worktree("add", "-b", branch, str(worktree_path), "HEAD")
    candidate_repo = Repo(worktree_path)
    write_safe(worktree_path / "candidate.txt", "candidate\n")
    candidate_repo.index.add(["candidate.txt"])
    candidate_repo.index.commit("candidate")
    handle = WorktreeHandle(
        original_repo_root=main_path,
        worktree_path=worktree_path,
        branch=branch,
        create_head_sha=base_sha,
    )
    worktree_manager._active = handle
    return main_repo, candidate_repo, handle


async def _collect(tool, args, ctx):
    return [result async for result in tool.run(args, ctx)]


@pytest.mark.asyncio
async def test_trusted_recipe_finishes_before_cancellation_propagates() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def operation():
        started.set()
        await release.wait()
        finished.set()
        return cast(VerificationReceipt, SimpleNamespace())

    task = asyncio.create_task(_await_trusted_recipe_completion(operation()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


def test_verify_work_schema_rejects_model_commands_and_paths() -> None:
    assert VerifyWork.get_parameters()["properties"] == {}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VerifyWorkArgs.model_validate({
            "argv": ["sh", "-c", "false"],
            "cwd": "/tmp",
            "allowed_paths": ["**"],
        })


def test_recipe_is_immutable_and_prebound_across_config_reload() -> None:
    safe_recipe = _recipe()
    config = build_test_vibe_config(trusted_verification_recipe=safe_recipe)
    loop = build_test_agent_loop(config=config)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        safe_recipe.task_brief = "replace the task"

    reloaded = config.model_copy(
        update={
            "trusted_verification_recipe": _recipe("evil"),
            "verification_subsystem": False,
        }
    )
    asyncio.run(loop.reload_with_initial_messages(base_config=reloaded))

    bound = loop._verification_state.trusted_recipe
    assert bound is not None
    assert bound.recipe_version == "safe-v1"
    assert bound.checks[0].argv[-1] == "print('safe Total: 1')"
    assert loop.base_config.trusted_verification_recipe == safe_recipe
    assert loop.base_config.verification_subsystem is True


def test_verify_work_uses_prebound_plan_and_receipt_reaches_land_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_repo, candidate_repo, _ = _linked_candidate(tmp_path)
    state = VerificationState.from_recipe(_recipe())
    state.receipt_store = VerificationReceiptStore(tmp_path / "receipts")
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )
    _record_current_verifier_pass(state)
    ctx = InvokeContext(
        tool_call_id="verify",
        agent_manager=cast(AgentManager, _FakeAgentManager(_recipe("evil"))),
        verification_state=state,
    )
    verify_tool = VerifyWork(
        config_getter=lambda: VerifyWorkConfig(), state=BaseToolState()
    )

    verified = asyncio.run(_collect(verify_tool, VerifyWorkArgs(), ctx))

    assert len(verified) == 1
    assert verified[0].passed
    assert state.receipt_reference is not None
    receipt = state.receipt_store.load_any(verified[0].receipt_id)
    assert receipt.evidence[0].argv[-1] == "print('safe Total: 1')"
    assert receipt.recipe_version == "safe-v1"

    land_tool = LandWork(config_getter=lambda: LandWorkConfig(), state=BaseToolState())
    landed = asyncio.run(_collect(land_tool, LandWorkArgs(), ctx))

    assert landed[0].merged
    assert landed[0].verification_receipt_id == verified[0].receipt_id
    assert landed[0].merge_commit_sha == main_repo.head.commit.hexsha
    assert candidate_repo.head.commit.hexsha in {
        parent.hexsha for parent in main_repo.head.commit.parents
    }


def test_verify_work_requires_verifier_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _linked_candidate(tmp_path)
    state = VerificationState.from_recipe(_recipe())
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )
    ctx = InvokeContext(
        tool_call_id="verify",
        agent_manager=cast(AgentManager, _FakeAgentManager(_recipe())),
        verification_state=state,
    )
    tool = VerifyWork(config_getter=lambda: VerifyWorkConfig(), state=BaseToolState())

    with pytest.raises(ToolError, match="verifier PASS"):
        asyncio.run(_collect(tool, VerifyWorkArgs(), ctx))


@pytest.mark.parametrize("mutation", ["head", "branch"])
def test_verify_work_race_publishes_no_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    _, candidate_repo, _ = _linked_candidate(tmp_path)
    state = VerificationState.from_recipe(_recipe())
    state.receipt_store = VerificationReceiptStore(tmp_path / "receipts")
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )
    _record_current_verifier_pass(state)
    original_run = state.run_bound_recipe

    def run_then_mutate(**kwargs):
        receipt = original_run(**kwargs)
        if mutation == "head":
            candidate_repo.git.commit("--allow-empty", "-m", "concurrent change")
        else:
            candidate_repo.git.checkout("-b", "same-head-concurrent-branch")
        return receipt

    monkeypatch.setattr(state, "run_bound_recipe", run_then_mutate)
    ctx = InvokeContext(
        tool_call_id="verify",
        agent_manager=cast(AgentManager, _FakeAgentManager(_recipe())),
        verification_state=state,
    )
    tool = VerifyWork(config_getter=lambda: VerifyWorkConfig(), state=BaseToolState())

    with pytest.raises(ToolError, match="topology changed"):
        asyncio.run(_collect(tool, VerifyWorkArgs(), ctx))

    assert state.receipt_reference is None


def test_configured_recipe_rejects_legacy_pass_and_trivial_waiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState.from_recipe(_recipe())
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )
    state.record_verifier_pass(parse_verification_report(_report()))
    ctx = InvokeContext(
        tool_call_id="land",
        agent_manager=cast(AgentManager, _FakeAgentManager(_recipe())),
        verification_state=state,
    )

    with pytest.raises(ToolError, match="requires the current trusted"):
        _require_verification_note(LandWorkArgs(), ctx)
    with pytest.raises(ToolError, match="trivial waivers cannot replace"):
        _require_verification_note(
            LandWorkArgs(verification_note="trivial: docs only"),
            ctx,
            changed_paths=["docs/guide.md"],
        )


def test_configured_recipe_without_bound_state_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = VerificationState()
    monkeypatch.setattr(
        "vibe.core.verification_state.workspace_fingerprint", lambda: "candidate"
    )
    state.record_verifier_pass(parse_verification_report(_report()))
    ctx = InvokeContext(
        tool_call_id="land",
        agent_manager=cast(AgentManager, _FakeAgentManager(_recipe())),
        verification_state=state,
    )

    with pytest.raises(ToolError, match="requires the current trusted"):
        _require_verification_note(LandWorkArgs(), ctx)


def test_verify_work_availability_requires_enabled_prebound_config(
    tmp_path: Path,
) -> None:
    _, _, handle = _linked_candidate(tmp_path)
    enabled = build_test_vibe_config(trusted_verification_recipe=_recipe())
    disabled = enabled.model_copy(update={"verification_subsystem": False})
    unconfigured = enabled.model_copy(update={"trusted_verification_recipe": None})

    assert worktree_manager.active == handle
    assert VerifyWork.is_available(enabled)
    assert not VerifyWork.is_available(disabled)
    assert not VerifyWork.is_available(unconfigured)
    worktree_manager._active = None
    assert not VerifyWork.is_available(enabled)
