from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from git.exc import GitError

from vibe.core.verification_contract import verification_observation_hashes
from vibe.core.verification_state import VerificationCompletionConstraint
from vibe.core.worktree._trusted_git import TrustedGitError, TrustedGitWorktree
from vibe.core.worktree.manager import worktree_manager

if TYPE_CHECKING:
    from vibe.core.agents.manager import AgentManager
    from vibe.core.config import VibeConfig
    from vibe.core.execution_topology import ExecutionTopologySnapshot
    from vibe.core.llm.models import ResolvedToolCall
    from vibe.core.tasking._policy import BoundTaskContract
    from vibe.core.types import AssistantEvent, LLMMessage, MessageList
    from vibe.core.verification_state import VerificationState


_NON_CANDIDATE_TOOLS = frozenset({
    "ask_user_question",
    "background",
    "enter_plan_mode",
    "exit_plan_mode",
    "land_work",
    "manage_memory",
    "schedule",
    "skill",
    "task_checks",
    "team",
    "team_message",
    "todo",
    "tool_search",
    "verify_work",
    "work_strategy",
    "workflow_results",
    "workflow_status",
    "workflow_stop",
})
_TERMINAL_BACKGROUND_STATUSES = frozenset({
    "blocked",
    "canceled",
    "cancelled",
    "completed",
    "completed_with_failures",
    "failed",
    "stopped",
})


class AgentLoopVerificationMixin:
    _base_config: VibeConfig
    _is_subagent: bool
    _task_contract: BoundTaskContract | None
    _verification_state: VerificationState
    agent_manager: AgentManager
    messages: MessageList
    _execution_topology_snapshot: ExecutionTopologySnapshot | None

    def _record_successful_verification_observation(
        self, tool_name: str, validated_args: object, result: dict[str, object]
    ) -> None:
        if tool_name != "bash":
            return
        command = getattr(validated_args, "command", None)
        if not isinstance(command, str) or not command.strip():
            return
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            return
        observations = verification_observation_hashes(command, stdout, stderr)
        if not observations:
            return
        evidence_hashes = getattr(
            self, "_successful_verification_evidence_hashes", None
        )
        if evidence_hashes is None:
            evidence_hashes = []
            self._successful_verification_evidence_hashes = evidence_hashes
        evidence_hashes.extend(observations)

    def _verification_tool_may_mutate_candidate(
        self, tool_call: ResolvedToolCall
    ) -> bool:
        if (
            self._is_subagent
            or not self._base_config.verification_subsystem
            or tool_call.tool_name in _NON_CANDIDATE_TOOLS
        ):
            return False
        if tool_call.tool_name == "bash":
            from vibe.core.agent_loop_orchestration import (
                is_observational_shell_command,
            )

            return not is_observational_shell_command(
                str(tool_call.args_dict.get("command", "")),
                background=bool(tool_call.args_dict.get("background", False)),
            )
        if tool_call.tool_name == "task":
            from vibe.core.agents.models import profile_requires_isolation

            try:
                profile = self.agent_manager.get_agent(
                    str(tool_call.args_dict.get("agent", ""))
                )
            except (KeyError, TypeError, ValueError):
                return True
            return profile_requires_isolation(profile)
        return not tool_call.tool_class.call_is_read_only(tool_call.validated_args)

    @staticmethod
    def _verification_workspace_fingerprint() -> str | None:
        from vibe.core.verification_state import workspace_fingerprint

        return workspace_fingerprint()

    @staticmethod
    def _verification_tool_runs_async_candidate(tool_call: ResolvedToolCall) -> bool:
        if tool_call.tool_name == "bash":
            return bool(tool_call.args_dict.get("background", False))
        if tool_call.tool_name == "task":
            return bool(tool_call.args_dict.get("async_run", False))
        return tool_call.tool_name in {"launch_workflow", "team_spawn"}

    def _preinvalidate_async_candidate_tool(
        self, tool_call: ResolvedToolCall, *, tracked: bool
    ) -> bool:
        if not tracked or not self._verification_tool_runs_async_candidate(tool_call):
            return False
        self._verification_state.record_candidate_mutation(
            invalidate_authorization=True
        )
        return True

    def _observe_candidate_tool(
        self,
        tool_call: ResolvedToolCall,
        *,
        tracked: bool,
        fingerprint_before: str | None,
        authorization_preinvalidated: bool,
    ) -> None:
        if not tracked or authorization_preinvalidated:
            return
        fingerprint_after = self._verification_workspace_fingerprint()
        fingerprint_unknown = fingerprint_before is None or fingerprint_after is None
        if fingerprint_unknown:
            self._verification_state.record_candidate_mutation(
                invalidate_authorization=True
            )
        elif fingerprint_before != fingerprint_after:
            self._verification_state.record_candidate_mutation(
                invalidate_authorization=False
            )

    def successful_verification_evidence_hashes(self) -> tuple[str, ...]:
        return tuple(getattr(self, "_successful_verification_evidence_hashes", ()))

    def _runtime_tool_policy(self) -> tuple[frozenset[str] | None, bool]:
        from vibe.core.tools._canonical_task_tools import managed_tool_allowlist

        task_allowlist = (
            self._task_contract.allowed_tools if self._task_contract else None
        )
        recipe = self._base_config.trusted_verification_recipe
        managed = bool(recipe is not None and recipe.execution_topology is not None)
        return (
            managed_tool_allowlist(
                self._base_config,
                is_subagent=self._is_subagent,
                task_allowlist=task_allowlist,
            ),
            managed,
        )

    def _validate_trusted_execution_topology(
        self, *, is_subagent: bool
    ) -> ExecutionTopologySnapshot | None:
        if is_subagent:
            return None
        recipe = self._verification_state.trusted_recipe
        if recipe is None or recipe.config.execution_topology is None:
            return None
        from vibe.core.execution_topology import validate_execution_topology

        return validate_execution_topology(
            recipe.config.execution_topology, current_directory=Path.cwd()
        )

    def _verification_completion_constraint(
        self, *, receipt_valid: bool | None = None
    ) -> VerificationCompletionConstraint | None:
        if self._is_subagent or not self._base_config.verification_subsystem:
            return None
        state = self._verification_state
        state.observe_workspace_change()
        if self._candidate_background_work_is_pending():
            state.record_candidate_mutation(invalidate_authorization=True)
        if receipt_valid is None:
            receipt_valid = self._current_trusted_receipt_is_valid()
        return state.completion_constraint(receipt_valid=receipt_valid)

    def _candidate_background_work_is_pending(self) -> bool:
        registry = getattr(self, "background_registry", None)
        if registry is None:
            return False
        try:
            entries = registry.list_tasks()
        except Exception:
            return True
        return any(
            str(entry.status).casefold() not in _TERMINAL_BACKGROUND_STATUSES
            for entry in entries
        )

    def _observe_verification_tool_result(
        self, tool_call: ResolvedToolCall, status: str, result: dict[str, object] | None
    ) -> None:
        if status != "success" or tool_call.tool_name != "todo" or result is None:
            return
        todos = result.get("todos")
        if not isinstance(todos, list):
            return
        open_ids: list[str] = []
        for item in todos:
            if not isinstance(item, dict):
                continue
            raw_status = item.get("status")
            item_status = getattr(raw_status, "value", raw_status)
            if item_status in {"completed", "cancelled"}:
                continue
            identifier = item.get("id")
            if isinstance(identifier, str) and identifier:
                open_ids.append(identifier)
        self._verification_state.record_open_todos(open_ids)

    def _current_trusted_receipt_is_valid(self) -> bool:
        state = self._verification_state
        if state.trusted_recipe is not None and state.receipt_reference is not None:
            topology = state.trusted_recipe.config.execution_topology
            if topology is not None:
                try:
                    from vibe.core.execution_topology import (
                        revalidate_execution_topology_snapshot,
                        validate_execution_topology,
                    )

                    snapshot = self._execution_topology_snapshot
                    if snapshot is None:
                        snapshot = validate_execution_topology(
                            topology, current_directory=Path.cwd()
                        )
                        self._execution_topology_snapshot = snapshot
                    else:
                        revalidate_execution_topology_snapshot(
                            topology, snapshot, current_directory=Path.cwd()
                        )
                    return state.has_valid_receipt(
                        repository_path=snapshot.candidate_worktree,
                        expected_base_sha=topology.baseline_sha,
                        expected_candidate_head=snapshot.candidate_head,
                    )
                except (GitError, OSError, RuntimeError, TypeError, ValueError):
                    return False
            handle = worktree_manager.active
            if handle is not None:
                try:
                    base_sha = TrustedGitWorktree.open(
                        handle.original_repo_root
                    ).head_sha()
                    candidate_head = TrustedGitWorktree.open(
                        handle.worktree_path
                    ).head_sha()
                    return state.has_valid_receipt(
                        repository_path=handle.worktree_path,
                        expected_base_sha=base_sha,
                        expected_candidate_head=candidate_head,
                    )
                except (OSError, TrustedGitError, TypeError, ValueError):
                    return False
        return False

    def _guard_managed_completion_claims(
        self, *, receipt_valid: bool | None = None
    ) -> bool:
        # Parent task collectors validate managed subagent VERDICT output.
        if self._is_subagent:
            return False
        state = self._verification_state
        recipe = state.trusted_recipe
        if recipe is None or recipe.config.execution_topology is None:
            return False
        if recipe.config.execution_topology.state == "active":
            return True
        if receipt_valid is None:
            receipt_valid = self._current_trusted_receipt_is_valid()
        return not state.completion_claim_is_authorized(receipt_valid=receipt_valid)

    def _is_topology_managed_root(self) -> bool:
        if self._is_subagent:
            return False
        recipe = self._verification_state.trusted_recipe
        return bool(recipe is not None and recipe.config.execution_topology is not None)

    def _prepare_verification_turn_output(
        self,
        message: LLMMessage,
        constraint: VerificationCompletionConstraint | None,
        guard_managed_claims: bool,
        *,
        buffer_for_verification: bool,
        managed_root: bool,
    ) -> tuple[LLMMessage, VerificationCompletionConstraint | None, bool, str, bool]:
        visible_content = (message.content or "").strip()
        visible_reasoning = (message.reasoning_content or "").strip()
        suppressed_tool_prose = False
        if message.tool_calls and (constraint is not None or guard_managed_claims):
            suppressed_tool_prose = bool(visible_content or visible_reasoning)
            if suppressed_tool_prose:
                message = message.model_copy(
                    update={"content": None, "reasoning_content": None}
                )
                self.messages.replace_at(len(self.messages) - 1, message)
                visible_content = ""
                visible_reasoning = ""
            constraint = None
            guard_managed_claims = False
        empty_tool_turn = bool(
            message.tool_calls and not visible_content and not visible_reasoning
        )
        publish_buffered_assistant = (
            buffer_for_verification
            and constraint is None
            and not guard_managed_claims
            and (not managed_root or empty_tool_turn)
            and not suppressed_tool_prose
        )
        return (
            message,
            constraint,
            guard_managed_claims,
            visible_content,
            publish_buffered_assistant,
        )

    def _replace_with_verification_constraint(
        self,
        message: LLMMessage,
        constraint: VerificationCompletionConstraint,
        *,
        preserve_tool_calls: bool = False,
    ) -> AssistantEvent:
        from vibe.core.types import AssistantEvent

        allowed_prefixes = ("BLOCKED:", "IN_PROGRESS:", "QUESTION:")
        if not self._is_topology_managed_root():
            allowed_prefixes += ("Status:", "STATUS:")
        content = self._host_handoff(
            constraint.render(), message.content, allowed_prefixes=allowed_prefixes
        )
        replacement = message.model_copy(
            update={
                "content": content,
                "tool_calls": message.tool_calls if preserve_tool_calls else None,
                "reasoning_content": None,
            }
        )
        self.messages.replace_at(len(self.messages) - 1, replacement)
        return AssistantEvent(content=content, message_id=replacement.message_id)

    def _replace_unverified_managed_completion(
        self, message: LLMMessage, *, preserve_tool_calls: bool = False
    ) -> AssistantEvent:
        from vibe.core.types import AssistantEvent

        recipe = self._verification_state.trusted_recipe
        topology = recipe.config.execution_topology if recipe is not None else None
        if topology is not None and topology.state == "active":
            authority = (
                "HOST ACTIVE-PHASE STATUS: HANDOFF\n\n"
                "The quoted model handoff may report READY_FOR_HOST_FREEZE, "
                "BLOCKED, or in-progress status. It is not a verification, "
                "completion, acceptance, or landing authorization."
            )
            prefixes = ("READY_FOR_HOST_FREEZE:", "BLOCKED:", "IN_PROGRESS:")
        else:
            authority = (
                "HOST VERIFICATION STATUS: BLOCKED\n\n"
                "The quoted model handoff is not authorized as verified, complete, "
                "ready for acceptance, or safe to land. A current verifier PASS "
                "and trusted receipt for the exact topology are still required."
            )
            prefixes = ("BLOCKED:", "IN_PROGRESS:", "QUESTION:")
        content = self._host_handoff(
            authority, message.content, allowed_prefixes=prefixes
        )
        replacement = message.model_copy(
            update={
                "content": content,
                "tool_calls": message.tool_calls if preserve_tool_calls else None,
                "reasoning_content": None,
            }
        )
        self.messages.replace_at(len(self.messages) - 1, replacement)
        return AssistantEvent(content=content, message_id=replacement.message_id)

    @staticmethod
    def _host_handoff(
        authority: str, model_content: object, *, allowed_prefixes: tuple[str, ...]
    ) -> str:
        raw = model_content if isinstance(model_content, str) else ""
        escaped = "".join(
            character
            if character in {"\n", "\t"} or character.isprintable()
            else ascii(character)[1:-1]
            for character in raw
        ).strip()
        if escaped.startswith(allowed_prefixes):
            quoted = "\n".join(f"> {line}" for line in escaped.splitlines())
        else:
            expected = ", ".join(allowed_prefixes)
            quoted = (
                "> (model handoff omitted because it did not start with an "
                f"allowed phase outcome: {expected})"
            )
        return (
            f"{authority}\n\n"
            "UNTRUSTED MODEL HANDOFF (quoted for operator context only):\n\n"
            f"{quoted}"
        )


__all__ = ["AgentLoopVerificationMixin"]
