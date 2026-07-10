from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from filelock import Timeout
from git import Repo
from git.exc import GitCommandError
from pydantic import BaseModel, ConfigDict, Field

from vibe.core.logger import logger
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from vibe.core.verification_contract import (
    is_trivial_change_set,
    is_trivial_verification_note,
)
from vibe.core.worktree.manager import merge_lock, worktree_manager

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig

# Merge commits land on the user's main branch and live in history forever;
# every landing is gated by the ASK permission (the user approves each call).
_DEFAULT_MESSAGE_TEMPLATE = "Merge worktree branch {branch}"


class LandWorkArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")
    commit_message: str | None = Field(
        default=None,
        description=(
            "Optional merge-commit message. Defaults to "
            "'Merge worktree branch <branch>'. Keep it neutral and descriptive."
        ),
    )
    verification_note: str | None = Field(
        default=None,
        description=(
            "A 'trivial: <reason>' waiver for committed documentation-only diffs. "
            "Model-authored verification reports are never accepted here."
        ),
    )
    verification_receipt_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Optional harness-created verification receipt ID. When omitted, the "
            "current session's trusted receipt is used."
        ),
    )


class LandWorkResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    merged: bool
    merge_commit_sha: str | None = None
    target_branch: str | None = None
    source_branch: str | None = None
    verification_receipt_id: str | None = None
    message: str


class LandWorkConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class LandWork(
    BaseTool[LandWorkArgs, LandWorkResult, LandWorkConfig, BaseToolState],
    ToolUIData[LandWorkArgs, LandWorkResult],
):
    description: ClassVar[str] = (
        "Land the current worktree branch into the main checkout via a --no-ff "
        "merge commit, executed by the unsandboxed host process (the bash tool's "
        "sandbox makes the main checkout read-only, so this is the only path that "
        "can write it). Call this ONCE when your work is complete, committed, and "
        "verified. A session with a trusted verification recipe requires its "
        "current receipt. Without a recipe, a current recorded verifier/workflow "
        "PASS or a docs-only 'trivial: <reason>' waiver satisfies the gate; pasted "
        "verification prose never does. Merge preserves original "
        "commit SHAs and is revertable via `git revert -m 1 <merge-sha>`. "
        "Requires user approval. Refuses if main is dirty or the branch is "
        "already merged. Only available inside an active worktree isolation session."
    )

    @classmethod
    def is_available(cls, config: VibeConfig | None = None) -> bool:
        return worktree_manager.active is not None

    @classmethod
    def format_call_display(cls, args: LandWorkArgs) -> ToolCallDisplay:
        branch = (
            worktree_manager.active.branch
            if worktree_manager.active
            else "<no worktree>"
        )
        return ToolCallDisplay(summary=f"Land {branch} into main (--no-ff)")

    @classmethod
    def format_result_display(cls, result: LandWorkResult) -> ToolResultDisplay:
        return ToolResultDisplay(success=result.merged, message=result.message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Waiting for user to approve the merge"

    async def run(
        self, args: LandWorkArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[LandWorkResult, None]:
        handle = worktree_manager.active
        if handle is None:
            raise ToolError(
                "land_work requires an active worktree isolation session. "
                "No worktree is active."
            )

        main_dir = handle.original_repo_root
        branch = handle.branch

        try:
            main_repo = Repo(str(main_dir))
        except GitCommandError as exc:
            raise ToolError(
                f"Could not open main checkout at {main_dir}: {exc}"
            ) from exc

        if main_repo.working_tree_dir is None:
            raise ToolError(f"{main_dir} is not a working tree (bare repo?).")

        try:
            target = main_repo.active_branch.name
        except (TypeError, GitCommandError):
            raise ToolError(
                f"Main checkout at {main_dir} is in a detached HEAD state; "
                "cannot merge into a detached HEAD."
            ) from None

        worktree_dirty_note = ""
        try:
            wt_repo = Repo(str(handle.worktree_path))
            if wt_repo.is_dirty(untracked_files=True):
                worktree_dirty_note = (
                    " NOTE: the worktree has uncommitted changes that were NOT "
                    "included in this merge; only committed work landed. Commit "
                    "and land again if you need those edits on main."
                )
        except GitCommandError as exc:  # pragma: no cover - defensive
            logger.debug("land_work: could not inspect worktree dirty state: %s", exc)

        if _is_merged(main_repo, branch):
            yield LandWorkResult(
                merged=False,
                target_branch=target,
                source_branch=branch,
                message=(f"{branch} is already merged into {target}; nothing to land."),
            )
            return

        message = args.commit_message or _DEFAULT_MESSAGE_TEMPLATE.format(branch=branch)
        verification_receipt_id: str | None = None

        try:
            with merge_lock(main_dir):
                if main_repo.is_dirty(untracked_files=False):
                    raise ToolError(
                        f"Main checkout at {main_dir} (on {target}) has a dirty working "
                        "tree. Landing a merge requires a clean main tree so the merge "
                        "can update it atomically. The user should commit or stash main's "
                        "changes first. Refusing to land."
                    )
                candidate_sha = main_repo.commit(branch).hexsha
                verification_receipt_id = _require_verification_note(
                    args,
                    ctx,
                    changed_paths=_changed_paths(main_repo, target, branch),
                    repository_path=handle.worktree_path,
                    expected_base_sha=main_repo.head.commit.hexsha,
                    expected_candidate_head=candidate_sha,
                )
                merge_sha = _merge_no_ff(main_repo, candidate_sha, message)
                if candidate_sha not in {
                    parent.hexsha for parent in main_repo.commit(merge_sha).parents
                }:
                    raise ToolError(
                        "Landing produced a merge that is not parented by the "
                        "verified candidate commit. Manual repository inspection is "
                        "required."
                    )
        except Timeout:
            raise ToolError(
                "Another merge is in progress on this repo (merge lock busy). "
                "Retry shortly."
            ) from None
        except GitCommandError as exc:
            _abort_if_merging(main_repo)
            raise ToolError(
                f"Merge of {branch} into {target} failed (likely conflicts with "
                f"recent main work): {exc}. The merge was aborted. Rebase your "
                f"branch onto current {target} and retry, or resolve manually "
                f"from the main checkout."
            ) from exc

        ahead = _ahead_count(main_repo, branch)
        yield LandWorkResult(
            merged=True,
            merge_commit_sha=merge_sha,
            target_branch=target,
            source_branch=branch,
            verification_receipt_id=verification_receipt_id,
            message=(
                f"Merged {branch} into {target} (--no-ff) as {merge_sha}."
                f"{worktree_dirty_note}"
                + (f" {ahead} commit(s) ahead remains on {branch}." if ahead else "")
            ),
        )


def _require_verification_note(
    args: LandWorkArgs,
    ctx: InvokeContext | None,
    *,
    changed_paths: list[str] | None = None,
    repository_path: Path | None = None,
    expected_base_sha: str | None = None,
    expected_candidate_head: str | None = None,
) -> str | None:
    if ctx is None or ctx.agent_manager is None:
        return None
    enabled = bool(getattr(ctx.agent_manager.config, "verification_subsystem", True))
    if not enabled:
        return None

    state = ctx.verification_state
    recipe_required = bool(
        state is not None and state.trusted_recipe is not None
    ) or bool(
        getattr(ctx.agent_manager.config, "trusted_verification_recipe", None)
        is not None
    )
    requested_receipt = args.verification_receipt_id
    selected_receipt = requested_receipt or (
        state.receipt_reference.receipt_id
        if state is not None and state.receipt_reference is not None
        else None
    )
    if selected_receipt is not None:
        if state is None or repository_path is None or expected_base_sha is None:
            raise ToolError(
                "land_work could not validate the verification receipt against the "
                "candidate repository. Run trusted verification again."
            )
        if state.has_valid_receipt(
            repository_path=repository_path,
            expected_base_sha=expected_base_sha,
            expected_candidate_head=expected_candidate_head,
            receipt_id=selected_receipt,
        ):
            return selected_receipt
        validation = state.last_receipt_validation
        detail = (
            validation.summary() if validation is not None else "receipt is invalid"
        )
        raise ToolError(
            f"land_work rejected stale or invalid verification receipt: {detail}. "
            "Run the trusted acceptance checks again for the current candidate."
        )

    note = (args.verification_note or "").strip()
    if recipe_required:
        raise ToolError(
            "land_work requires the current trusted verification receipt for this "
            "session recipe. Record a verifier PASS, then run verify_work. "
            "Pasted verification prose and trivial waivers cannot replace the receipt."
        )
    if is_trivial_verification_note(note):
        if changed_paths is not None and is_trivial_change_set(changed_paths):
            return None
        raise ToolError(
            "land_work rejected the trivial waiver: it is limited to committed "
            "documentation-only changes under docs/, openwiki/, or standard "
            "root documentation files."
        )
    if note:
        raise ToolError(
            "land_work rejected verification_note: model-supplied verification "
            "reports cannot authorize a merge. Run harness-trusted acceptance "
            "checks or use the session-recorded verification state."
        )
    if state is not None and state.has_pass():
        return None
    raise ToolError(
        "land_work requires a current session-recorded verifier or workflow PASS "
        "when verification_subsystem is on and no trusted recipe is configured. "
        "Run verification again, or pass 'trivial: <reason>' for a committed "
        "documentation-only diff. Pasted verification prose is not accepted. Set "
        "verification_subsystem = false to disable this gate."
    )


def _changed_paths(
    repo: Repo, target_branch: str, source_branch: str
) -> list[str] | None:
    try:
        output = repo.git.diff(
            "--no-renames", "--name-only", f"{target_branch}...{source_branch}", "--"
        )
    except GitCommandError:
        return None
    return [line for line in output.splitlines() if line]


def _is_merged(repo: Repo, branch: str) -> bool:
    try:
        repo.git.merge_base("--is-ancestor", branch, "HEAD")
        return True
    except GitCommandError:
        return False


def _merge_no_ff(repo: Repo, candidate_sha: str, message: str) -> str:
    repo.git.merge("--no-ff", "-m", message, candidate_sha)
    return repo.head.commit.hexsha


def _abort_if_merging(repo: Repo) -> None:
    try:
        if (Path(repo.git_dir) / "MERGE_HEAD").exists():
            repo.git.merge("--abort")
    except GitCommandError:  # pragma: no cover - defensive
        pass


def _ahead_count(repo: Repo, branch: str) -> int:
    try:
        return int(repo.git.rev_list("--count", f"HEAD..{branch}").strip() or "0")
    except (GitCommandError, ValueError):
        return 0
