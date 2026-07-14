from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from filelock import Timeout
from git import Repo
from pydantic import BaseModel, ConfigDict, Field

from vibe.core.execution_topology import (
    ExecutionTopologyError,
    validate_execution_topology,
)
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
from vibe.core.worktree._ref_transaction import (
    CheckedOutRefUpdateError,
    update_checked_out_branch,
)
from vibe.core.worktree._trusted_git import TrustedGitError, TrustedGitWorktree
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


@dataclass(frozen=True, slots=True)
class _LandingRequest:
    main_dir: Path
    target_branch: str
    source_branch: str
    candidate_worktree: Path
    message: str


@dataclass(frozen=True, slots=True)
class _LandingTransaction:
    merge_sha: str | None
    verification_receipt_id: str | None
    already_merged: bool


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
        "current receipt. Without a recipe, only a docs-only "
        "'trivial: <reason>' waiver can satisfy the gate; nontrivial work needs "
        "a host-pinned trusted recipe and receipt. Pasted verification prose "
        "never authorizes landing. Merge preserves original "
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
        _require_landing_authority(ctx)
        handle = worktree_manager.active
        if handle is None:
            raise ToolError(
                "land_work requires an active worktree isolation session. "
                "No worktree is active."
            )

        main_dir = handle.original_repo_root
        branch = handle.branch

        try:
            main_repo = TrustedGitWorktree.open(main_dir)
            target_ref = main_repo.head_ref()
            target = target_ref.removeprefix("refs/heads/")
        except TrustedGitError as exc:
            raise ToolError(f"Could not open exact main checkout: {exc}") from exc

        worktree_dirty_note = ""
        try:
            wt_repo = TrustedGitWorktree.open(handle.worktree_path)
            candidate_sha = wt_repo.head_sha()
            if not wt_repo.clean_against(candidate_sha, include_untracked=True):
                worktree_dirty_note = (
                    " NOTE: the worktree has uncommitted changes that were NOT "
                    "included in this merge; only committed work landed. Commit "
                    "and land again if you need those edits on main."
                )
        except TrustedGitError as exc:
            logger.debug("land_work: could not inspect worktree dirty state: %s", exc)

        message = args.commit_message or _DEFAULT_MESSAGE_TEMPLATE.format(branch=branch)
        request = _LandingRequest(
            main_dir=main_dir,
            target_branch=target,
            source_branch=branch,
            candidate_worktree=handle.worktree_path,
            message=message,
        )

        try:
            transaction = _perform_exact_landing(main_repo, request, args, ctx)
        except Timeout:
            raise ToolError(
                "Another merge is in progress on this repo (merge lock busy). "
                "Retry shortly."
            ) from None
        except (CheckedOutRefUpdateError, TrustedGitError) as exc:
            raise ToolError(
                f"Merge of {branch} into {target} failed (likely conflicts with "
                f"recent main work): {exc}. Landing was not reported as complete. "
                "Inspect the main checkout before retrying; the merge lock only "
                "coordinates Vibe landing operations, not external editors or Git "
                "processes."
            ) from exc

        if transaction.already_merged:
            yield LandWorkResult(
                merged=False,
                target_branch=target,
                source_branch=branch,
                message=(f"{branch} is already merged into {target}; nothing to land."),
            )
            return
        if transaction.merge_sha is None:
            raise ToolError("Landing did not produce an exact merge commit.")

        ahead = _ahead_count(main_repo, branch)
        yield LandWorkResult(
            merged=True,
            merge_commit_sha=transaction.merge_sha,
            target_branch=target,
            source_branch=branch,
            verification_receipt_id=transaction.verification_receipt_id,
            message=(
                f"Merged {branch} into {target} (--no-ff) as {transaction.merge_sha}."
                f"{worktree_dirty_note}"
                + (f" {ahead} commit(s) ahead remains on {branch}." if ahead else "")
            ),
        )


def _require_landing_authority(ctx: InvokeContext | None) -> None:
    if ctx is None or ctx.agent_manager is None:
        raise ToolError("land_work requires an authenticated invocation context")
    if ctx.verification_state is None:
        raise ToolError("land_work requires session verification state")
    if not bool(getattr(ctx.agent_manager.config, "verification_subsystem", True)):
        raise ToolError("land_work requires verification_subsystem = true")


def _revalidate_managed_topology(
    ctx: InvokeContext | None,
    request: _LandingRequest,
    *,
    expected_base_sha: str,
    expected_candidate_sha: str,
) -> None:
    _require_landing_authority(ctx)
    assert ctx is not None and ctx.verification_state is not None
    recipe = ctx.verification_state.trusted_recipe
    topology = recipe.config.execution_topology if recipe is not None else None
    if topology is None:
        return
    try:
        snapshot = validate_execution_topology(
            topology, current_directory=request.candidate_worktree
        )
    except (
        ExecutionTopologyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ToolError(
            f"land_work rejected the managed execution topology: {exc}"
        ) from exc
    expected_identity = (
        "verification",
        expected_base_sha,
        expected_candidate_sha,
        request.source_branch,
        expected_candidate_sha,
        request.candidate_worktree.resolve(),
    )
    observed_identity = (
        topology.state,
        topology.baseline_sha,
        topology.candidate_sha,
        topology.candidate_branch,
        snapshot.candidate_head,
        snapshot.candidate_worktree,
    )
    if observed_identity != expected_identity:
        raise ToolError(
            "land_work managed topology does not bind the exact landing base, "
            "candidate, branch, and physical worktree"
        )


def _perform_exact_landing(
    repo: TrustedGitWorktree,
    request: _LandingRequest,
    args: LandWorkArgs,
    ctx: InvokeContext | None,
) -> _LandingTransaction:
    with merge_lock(request.main_dir):
        if repo.head_ref() != f"refs/heads/{request.target_branch}":
            raise ToolError(
                "Main changed branches while landing was pending. Refusing to "
                "integrate into a different target branch."
            )
        base_sha = repo.head_sha()
        if not repo.clean_against(base_sha, include_untracked=True):
            raise ToolError(
                f"Main checkout at {request.main_dir} (on {request.target_branch}) "
                "has a dirty working tree. Landing a merge requires a clean main "
                "tree so the merge can materialize the exact candidate. The user "
                "should commit or stash main's changes first. Refusing to land."
            )
        candidate_sha = repo.branch_sha(request.source_branch)
        if _is_merged(repo, candidate_sha):
            return _LandingTransaction(None, None, True)

        receipt_id = _require_verification_note(
            args,
            ctx,
            changed_paths=_changed_paths(repo, base_sha, candidate_sha),
            repository_path=request.candidate_worktree,
            expected_base_sha=base_sha,
            expected_candidate_head=candidate_sha,
            landing_request=request,
        )
        reserved_state = None
        reserved_generation = None
        if receipt_id is not None:
            if ctx is None or ctx.verification_state is None:
                raise ToolError("landing receipt state became unavailable")
            reserved_state = ctx.verification_state
            reserved_generation = reserved_state.current_verifier_pass_generation()
            if reserved_generation is None or not (
                reserved_state.reserve_landing_authorization(
                    reserved_generation, receipt_id
                )
            ):
                raise ToolError(
                    "landing receipt authorization changed before the exact Git "
                    "transaction began"
                )
        try:
            _require_exact_landing_state(
                repo,
                target_branch=request.target_branch,
                source_branch=request.source_branch,
                expected_base_sha=base_sha,
                expected_candidate_sha=candidate_sha,
                candidate_worktree=request.candidate_worktree,
                require_clean_candidate=_uses_verifier_authority(args, ctx, receipt_id),
            )
            _revalidate_managed_topology(
                ctx,
                request,
                expected_base_sha=base_sha,
                expected_candidate_sha=candidate_sha,
            )
            merge_sha = _merge_no_ff(
                repo,
                expected_base_sha=base_sha,
                candidate_sha=candidate_sha,
                message=request.message,
            )
            parents = repo.commit_parents(merge_sha)
            if parents != [base_sha, candidate_sha]:
                raise ToolError(
                    "Landing produced a merge whose exact parents do not match the "
                    "authorized base and candidate commits. Manual repository "
                    "inspection is required."
                )
            return _LandingTransaction(merge_sha, receipt_id, False)
        finally:
            if reserved_state is not None and reserved_generation is not None:
                still_current = reserved_state.release_authorization(
                    reserved_generation, receipt_id=receipt_id
                )
                if not still_current:
                    logger.info(
                        "land_work: verification authorization changed after the "
                        "exact landing transaction"
                    )


def _require_verification_note(
    args: LandWorkArgs,
    ctx: InvokeContext | None,
    *,
    changed_paths: list[str] | None = None,
    repository_path: Path | None = None,
    expected_base_sha: str | None = None,
    expected_candidate_head: str | None = None,
    landing_request: _LandingRequest | None = None,
) -> str | None:
    if ctx is None or ctx.agent_manager is None:
        return None
    enabled = bool(getattr(ctx.agent_manager.config, "verification_subsystem", True))
    if not enabled:
        return None

    state = ctx.verification_state
    if (
        landing_request is not None
        and expected_base_sha is not None
        and expected_candidate_head is not None
    ):
        _revalidate_managed_topology(
            ctx,
            landing_request,
            expected_base_sha=expected_base_sha,
            expected_candidate_sha=expected_candidate_head,
        )
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
        if state.current_verifier_pass_generation() is None:
            raise ToolError(
                "land_work rejected the verification receipt because the latest "
                "verifier attempt is not a current PASS. Run the verifier and "
                "verify_work again."
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
    raise ToolError(
        "land_work requires a trusted verification receipt for nontrivial work. "
        "Configure a host-owned trusted recipe, record a current verifier PASS, "
        "and run verify_work; or pass 'trivial: <reason>' for a committed "
        "documentation-only diff. A legacy verifier PASS or pasted report cannot "
        "authorize landing. Set verification_subsystem = false to disable this gate."
    )


def _uses_verifier_authority(
    args: LandWorkArgs, ctx: InvokeContext | None, receipt_id: str | None
) -> bool:
    if receipt_id is not None:
        return True
    if (args.verification_note or "").strip():
        return False
    if ctx is None or ctx.agent_manager is None:
        return False
    return bool(getattr(ctx.agent_manager.config, "verification_subsystem", True))


def _require_exact_landing_state(
    repo: TrustedGitWorktree,
    *,
    target_branch: str,
    source_branch: str,
    expected_base_sha: str,
    expected_candidate_sha: str,
    candidate_worktree: Path,
    require_clean_candidate: bool,
) -> None:
    try:
        if repo.head_ref() != f"refs/heads/{target_branch}":
            raise ToolError("landing target branch changed after authorization")
        if repo.head_sha() != expected_base_sha:
            raise ToolError("landing base changed after authorization")
        if repo.branch_sha(source_branch) != expected_candidate_sha:
            raise ToolError("candidate branch changed after authorization")
        if not require_clean_candidate:
            return
        candidate = TrustedGitWorktree.open(candidate_worktree)
        if candidate.head_ref() != f"refs/heads/{source_branch}":
            raise ToolError("verified candidate worktree changed branches")
        if candidate.head_sha() != expected_candidate_sha:
            raise ToolError("verified candidate HEAD changed after authorization")
        if not candidate.clean_against(expected_candidate_sha, include_untracked=True):
            raise ToolError("verified candidate became dirty after authorization")
    except (TrustedGitError, OSError, TypeError, ValueError) as exc:
        raise ToolError(
            "Could not revalidate the exact landing base and candidate commits."
        ) from exc


def _changed_paths(
    repo: Repo | TrustedGitWorktree, base_sha: str, candidate_sha: str
) -> list[str] | None:
    try:
        if isinstance(repo, TrustedGitWorktree):
            return repo.changed_paths(base_sha, candidate_sha)
        if repo.working_tree_dir is None:
            return None
        trusted = TrustedGitWorktree.open(Path(repo.working_tree_dir))
        return trusted.changed_paths(base_sha, candidate_sha)
    except (TrustedGitError, TypeError, ValueError):
        return None


def _is_merged(repo: TrustedGitWorktree, candidate_sha: str) -> bool:
    return repo.is_ancestor(candidate_sha, repo.head_sha())


def _merge_no_ff(
    repo: TrustedGitWorktree,
    *,
    expected_base_sha: str,
    candidate_sha: str,
    message: str,
) -> str:
    tree_sha = repo.merge_tree(expected_base_sha, candidate_sha)
    merge_sha = repo.commit_tree(tree_sha, expected_base_sha, candidate_sha, message)
    update_checked_out_branch(
        repo, expected_head=expected_base_sha, new_commit=merge_sha
    )
    return merge_sha


def _ahead_count(repo: TrustedGitWorktree, branch: str) -> int:
    try:
        return repo.ahead_count(branch)
    except (TrustedGitError, ValueError):
        return 0
