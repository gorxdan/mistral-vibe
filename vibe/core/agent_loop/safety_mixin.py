"""Permission-gate + approval mixin for AgentLoop.

Provides the ASK-gated tool permission flow: ``_should_execute_tool`` (rule-store
decision, then delegate to the safety judge, then human approval) and
``_ask_approval`` (the approval-callback path). The fork-only safety-judge
subsystem it delegates to lives in the sibling ``agent_loop_safety_judge`` module
and is inherited via the class base.

Implicit dependencies on the host class (AgentLoop):

Attributes (set by AgentLoop.__init__):
    approval_callback         (ApprovalCallback | None)
    pending_judge_deferral    (str | None)
    _permission_store         (PermissionStore)

Properties (defined on AgentLoop):
    bypass_tool_permissions   (bool)
    messages                  (MessageList)

Methods (defined elsewhere on AgentLoop / sibling mixins):
    _judge_tool_safety(...) -> ToolDecision | None   [AgentLoopSafetyJudgeMixin]
    _fire_notification_hooks(kind, msg, tool_name) -> AsyncGenerator [AgentLoopHooksMixin]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from vibe.core.agent_loop._models import ToolDecision, ToolExecutionResponse
from vibe.core.agent_loop_safety_judge import AgentLoopSafetyJudgeMixin
from vibe.core.tools.base import ToolPermission
from vibe.core.tools.permissions import PermissionContext, RequiredPermission
from vibe.core.types import ApprovalResponse

if TYPE_CHECKING:
    from vibe.core.tools.base import BaseTool
    from vibe.core.tools.permissions import PermissionStore
    from vibe.core.types import ApprovalCallback


class AgentLoopSafetyMixin(AgentLoopSafetyJudgeMixin):
    """Mixin that adds the tool permission-gate + approval flow to AgentLoop.

    Inherits the safety-judge subsystem (→ Hooks) so ``_judge_tool_safety`` and
    the notification hooks resolve via the inheritance chain. See module
    docstring for the implicit contract with the host class.
    """

    # Declared for type-checking only; set by AgentLoop.__init__.
    approval_callback: ApprovalCallback | None
    pending_judge_deferral: str | None
    _permission_store: PermissionStore
    tool_manager: Any  # ToolManager — typed loosely to avoid a runtime import cycle

    @property
    def bypass_tool_permissions(self) -> bool: ...

    # ``messages``, ``_serialize_tool_input``, ``_fire_notification_hooks``, and
    # ``_patch_assistant_tool_call_args`` are inherited from AgentLoopHooksMixin
    # (via AgentLoopSafetyJudgeMixin) — redeclaring them here as stubs would
    # shadow the real implementations, so they are intentionally omitted.

    async def _should_execute_tool(
        self, tool: BaseTool, args: BaseModel, tool_call_id: str
    ) -> ToolDecision:
        if self.bypass_tool_permissions:
            return ToolDecision(
                verdict=ToolExecutionResponse.EXECUTE,
                approval_type=ToolPermission.ALWAYS,
            )

        async with self._permission_store.lock:
            tool_name = tool.get_name()
            ctx = tool.resolve_permission(args)

            if ctx is None:
                config_perm = self.tool_manager.get_tool_config(tool_name).permission
                ctx = PermissionContext(permission=config_perm)

            if ctx.permission == ToolPermission.ALWAYS:
                return ToolDecision(
                    verdict=ToolExecutionResponse.EXECUTE,
                    approval_type=ToolPermission.ALWAYS,
                )
            if ctx.permission == ToolPermission.NEVER:
                return ToolDecision(
                    verdict=ToolExecutionResponse.SKIP,
                    approval_type=ToolPermission.NEVER,
                    feedback=ctx.reason
                    or f"Tool '{tool_name}' is permanently disabled",
                )
            uncovered = [
                rp
                for rp in ctx.required_permissions
                if not self._permission_store.covers(tool_name, rp)
            ]
            if ctx.required_permissions and not uncovered:
                return ToolDecision(
                    verdict=ToolExecutionResponse.EXECUTE,
                    approval_type=ToolPermission.ALWAYS,
                )

        # Lock released: the safety-judge LLM call and human approval are slow;
        # holding the permission lock across them would serialize every parallel
        # ASK-gated tool. The rule-store reads above happened under the lock.
        judged = await self._judge_tool_safety(tool_name, args, uncovered)
        if judged is not None:
            return judged
        return await self._ask_approval(tool_name, args, tool_call_id, uncovered)

    async def _ask_approval(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission],
    ) -> ToolDecision:
        if not self.approval_callback:
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                approval_type=ToolPermission.ASK,
                feedback="Tool execution not permitted.",
            )
        await self._fire_notification_hooks(
            "permission_required", f"Approval needed for {tool_name}", tool_name
        )
        response, feedback, modified_args = await self.approval_callback(
            tool_name,
            args,
            tool_call_id,
            required_permissions,
            # Carry the judge's deferral reason (set in _judge_tool_safety) so
            # the host prompt can show WHY approval is needed even when the
            # call originated from a workflow/task subagent — the subagent's
            # loop-local pending_judge_deferral is invisible to the host, so the
            # note must travel with the callback itself.
            self.pending_judge_deferral,
        )

        match response:
            case ApprovalResponse.YES:
                verdict = ToolExecutionResponse.EXECUTE
            case ApprovalResponse.MODIFY:
                verdict = ToolExecutionResponse.EXECUTE
            case _:
                verdict = ToolExecutionResponse.SKIP

        return ToolDecision(
            verdict=verdict,
            approval_type=ToolPermission.ASK,
            feedback=feedback,
            modified_args=modified_args
            if response == ApprovalResponse.MODIFY
            else None,
        )
