from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, ClassVar

from git import Repo
from git.exc import GitError
from pydantic import BaseModel, ConfigDict

from vibe.core._verification_receipt import VerificationReceiptError
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.worktree.manager import worktree_manager

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig


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
        return bool(
            worktree_manager.active is not None
            and config is not None
            and config.verification_subsystem
            and config.trusted_verification_recipe is not None
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
        handle = worktree_manager.active
        if handle is None:
            raise ToolError("verify_work requires an active worktree session")
        if ctx is None or ctx.verification_state is None:
            raise ToolError("verify_work requires session verification state")
        if (
            ctx.agent_manager is None
            or not ctx.agent_manager.config.verification_subsystem
        ):
            raise ToolError("verify_work requires verification_subsystem = true")

        state = ctx.verification_state
        recipe = state.trusted_recipe
        if recipe is None:
            raise ToolError(
                "verify_work has no trusted recipe prebound to this session; "
                "restart Vibe after configuring one"
            )
        if not state.has_verifier_pass():
            raise ToolError(
                "verify_work requires a current recorded verifier PASS for this "
                "candidate workspace"
            )

        try:
            main_repo = Repo(str(handle.original_repo_root))
            candidate_repo = Repo(str(handle.worktree_path))
            base_sha = main_repo.head.commit.hexsha
            candidate_head = candidate_repo.head.commit.hexsha
            candidate_branch = candidate_repo.active_branch.name
        except (GitError, TypeError, ValueError) as exc:
            raise ToolError(
                f"verify_work could not inspect the repositories: {exc}"
            ) from exc
        if candidate_branch != handle.branch:
            raise ToolError(
                "verify_work active branch does not match the worktree session: "
                f"expected {handle.branch}, found {candidate_branch}"
            )

        try:
            receipt = await asyncio.to_thread(
                state.run_bound_recipe,
                repository_path=handle.worktree_path,
                base_sha=base_sha,
            )
        except (OSError, ValueError, VerificationReceiptError) as exc:
            raise ToolError(f"trusted verification could not run: {exc}") from exc

        if receipt.repository.candidate_head != candidate_head:
            raise ToolError(
                "candidate HEAD changed while trusted verification was starting; "
                "run the verifier and verify_work again"
            )
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


__all__ = ["VerifyWork", "VerifyWorkArgs", "VerifyWorkConfig", "VerifyWorkResult"]
