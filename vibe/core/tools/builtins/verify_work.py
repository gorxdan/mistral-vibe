from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from git import Repo
from git.exc import GitError
from pydantic import BaseModel, ConfigDict

from vibe.core._verification_receipt import VerificationReceiptError
from vibe.core.execution_topology import (
    ExecutionTopologyError,
    ExecutionTopologySnapshot,
    validate_execution_topology,
)
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.worktree.manager import WorktreeHandle, worktree_manager

if TYPE_CHECKING:
    from vibe.core._verification_receipt import VerificationReceipt
    from vibe.core.config import TrustedExecutionTopologyConfig, VibeConfig
    from vibe.core.verification_state import VerificationState


@dataclass(frozen=True, slots=True)
class _VerificationTarget:
    repository_path: Path
    base_sha: str
    candidate_head: str
    expected_branch: str
    candidate_branch: str
    topology_snapshot: ExecutionTopologySnapshot | None = None
    worktree_handle: WorktreeHandle | None = None


class VerifyWorkArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerifyWorkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    receipt_id: str
    recipe_version: str
    base_sha: str
    candidate_head: str
    failed_checks: tuple[str, ...]
    message: str


class VerifyWorkConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class VerifyWork(
    BaseTool[VerifyWorkArgs, VerifyWorkResult, VerifyWorkConfig, BaseToolState],
    ToolUIData[VerifyWorkArgs, VerifyWorkResult],
):
    description: ClassVar[str] = (
        "Run the host-configured trusted verification recipe for the current "
        "worktree after an independent verifier PASS. This tool accepts no "
        "commands, paths, or other model-selected inputs and records a durable "
        "receipt when every prebound check passes."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        recipe = config.trusted_verification_recipe if config is not None else None
        topology = recipe.execution_topology if recipe is not None else None
        return bool(
            config is not None
            and config.verification_subsystem
            and recipe is not None
            and (
                worktree_manager.active is not None
                or (topology is not None and topology.state == "verification")
            )
        )

    @classmethod
    def format_call_display(cls, args: VerifyWorkArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary="Run trusted verification recipe")

    @classmethod
    def format_result_display(cls, result: VerifyWorkResult) -> ToolResultDisplay:
        return ToolResultDisplay(success=result.passed, message=result.message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Running host-configured verification checks"

    async def run(
        self, args: VerifyWorkArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[VerifyWorkResult, None]:
        state = _require_verification_state(ctx)
        recipe = state.trusted_recipe
        if recipe is None:
            raise ToolError(
                "verify_work has no trusted recipe prebound to this session; "
                "restart Vibe after configuring one"
            )
        topology = recipe.config.execution_topology
        target = _resolve_verification_target(topology)
        verifier_attempt_generation = state.current_verifier_pass_generation()
        if verifier_attempt_generation is None or not state.has_verifier_pass(
            expected_base_sha=target.base_sha
        ):
            raise ToolError(
                "verify_work requires a current recorded verifier PASS for this "
                "candidate workspace and landing base"
            )

        try:
            receipt = await _await_trusted_recipe_completion(
                asyncio.to_thread(
                    state.run_bound_recipe,
                    repository_path=target.repository_path,
                    base_sha=target.base_sha,
                    verifier_attempt_generation=verifier_attempt_generation,
                    publish=False,
                )
            )
        except (OSError, ValueError, VerificationReceiptError) as exc:
            raise ToolError(f"trusted verification could not run: {exc}") from exc

        if (
            not _verification_target_is_unchanged(target, topology)
            or receipt.repository.candidate_head != target.candidate_head
        ):
            raise ToolError(
                "candidate or control topology changed during trusted verification; "
                "run the verifier and verify_work again"
            )
        if receipt.passed:
            try:
                state.record_receipt(
                    receipt, verifier_attempt_generation=verifier_attempt_generation
                )
            except ValueError as exc:
                raise ToolError(
                    "verifier authorization changed during trusted verification; "
                    "run the verifier and verify_work again"
                ) from exc
        failed_checks = tuple(
            evidence.name for evidence in receipt.evidence if not evidence.passed
        )
        message = (
            f"Trusted verification passed; receipt {receipt.receipt_id} is bound "
            "to the current candidate."
            if receipt.passed
            else (
                f"Trusted verification failed; receipt {receipt.receipt_id} "
                "contains the check evidence."
            )
        )
        yield VerifyWorkResult(
            passed=receipt.passed,
            receipt_id=receipt.receipt_id,
            recipe_version=receipt.recipe_version,
            base_sha=receipt.repository.base_sha,
            candidate_head=receipt.repository.candidate_head,
            failed_checks=failed_checks,
            message=message,
        )


async def _await_trusted_recipe_completion(
    operation: Awaitable[VerificationReceipt],
) -> VerificationReceipt:
    task = asyncio.ensure_future(operation)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        except Exception:
            break
    try:
        receipt = task.result()
    except Exception:
        if cancellation is not None:
            raise cancellation from None
        raise
    if cancellation is not None:
        raise cancellation
    return receipt


def _require_verification_state(ctx: InvokeContext | None) -> VerificationState:
    if ctx is None or ctx.verification_state is None:
        raise ToolError("verify_work requires session verification state")
    if ctx.agent_manager is None or not ctx.agent_manager.config.verification_subsystem:
        raise ToolError("verify_work requires verification_subsystem = true")
    return ctx.verification_state


def _resolve_verification_target(
    topology: TrustedExecutionTopologyConfig | None,
) -> _VerificationTarget:
    target = (
        _managed_verification_target(topology)
        if topology is not None
        else _worktree_verification_target()
    )
    if target.candidate_branch != target.expected_branch:
        raise ToolError(
            "verify_work active branch does not match the bound candidate: "
            f"expected {target.expected_branch}, found {target.candidate_branch}"
        )
    return target


def _managed_verification_target(
    topology: TrustedExecutionTopologyConfig,
) -> _VerificationTarget:
    if topology.state != "verification":
        raise ToolError("verify_work requires execution_topology.state = verification")
    try:
        snapshot = validate_execution_topology(topology, current_directory=Path.cwd())
        candidate_branch = Repo(str(snapshot.candidate_worktree)).active_branch.name
    except (ExecutionTopologyError, GitError, TypeError, ValueError) as exc:
        raise ToolError(
            f"verify_work rejected the managed execution topology: {exc}"
        ) from exc
    return _VerificationTarget(
        repository_path=snapshot.candidate_worktree,
        base_sha=topology.baseline_sha,
        candidate_head=snapshot.candidate_head,
        expected_branch=topology.candidate_branch,
        candidate_branch=candidate_branch,
        topology_snapshot=snapshot,
    )


def _worktree_verification_target() -> _VerificationTarget:
    handle = worktree_manager.active
    if handle is None:
        raise ToolError("verify_work requires an active worktree session")
    try:
        base_sha = Repo(str(handle.original_repo_root)).head.commit.hexsha
        candidate_repo = Repo(str(handle.worktree_path))
        candidate_head = candidate_repo.head.commit.hexsha
        candidate_branch = candidate_repo.active_branch.name
    except (GitError, TypeError, ValueError) as exc:
        raise ToolError(
            f"verify_work could not inspect the repositories: {exc}"
        ) from exc
    return _VerificationTarget(
        repository_path=handle.worktree_path,
        base_sha=base_sha,
        candidate_head=candidate_head,
        expected_branch=handle.branch,
        candidate_branch=candidate_branch,
        worktree_handle=handle,
    )


def _verification_target_is_unchanged(
    target: _VerificationTarget, topology: TrustedExecutionTopologyConfig | None
) -> bool:
    try:
        if topology is not None and target.topology_snapshot is not None:
            current = validate_execution_topology(
                topology, current_directory=Path.cwd()
            )
            return current == target.topology_snapshot
        if target.worktree_handle is None:
            return False
        handle = target.worktree_handle
        current_base = Repo(str(handle.original_repo_root)).head.commit.hexsha
        candidate = Repo(str(handle.worktree_path))
        return (
            current_base == target.base_sha
            and candidate.head.commit.hexsha == target.candidate_head
            and candidate.active_branch.name == target.expected_branch
        )
    except (ExecutionTopologyError, GitError, OSError, TypeError, ValueError):
        return False


__all__ = ["VerifyWork", "VerifyWorkArgs", "VerifyWorkConfig", "VerifyWorkResult"]
