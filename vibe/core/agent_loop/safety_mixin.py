"""Safety-judge + approval mixin for AgentLoop.

Provides the safety-judge LLM gate (verdict cache, args serialization, transcript
window), the ASK-gated approval flow, and the modification (re-validate +
re-dispatch) path. Extracted from the loop module.

Implicit dependencies on the host class (AgentLoop):

Attributes (set by AgentLoop.__init__):
    approval_callback         (ApprovalCallback | None)
    pending_judge_deferral    (str | None)
    _permission_store         (PermissionStore)
    _judge_verdict_cache      (OrderedDict[...], JudgeVerdict)
    _judge_verdict_cache_maxsize (int)
    _judge_model_alias_for_cache (str | None)
    _current_user_message_id  (str | None)

Properties (defined on AgentLoop):
    config                    (VibeConfig)
    bypass_tool_permissions   (bool)
    agent_profile             (AgentProfile)
    effective_model           (() -> ModelConfig)
    messages                  (MessageList)

Methods (defined elsewhere on AgentLoop / sibling mixins):
    _get_extra_headers(provider) -> dict[str, str]
    _serialize_tool_input(tool_call) -> dict[str, Any]   [AgentLoopHooksMixin]
    _fire_notification_hooks(kind, msg, tool_name) -> AsyncGenerator [AgentLoopHooksMixin]
    _patch_assistant_tool_call_args(call_id, tool_input) -> None  [AgentLoopHooksMixin]
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from vibe.core.agent_loop._limits import (
    JUDGE_ARGS_LIMIT,
    JUDGE_ARGS_TRUNCATED_SENTINEL,
    JUDGE_TRANSCRIPT_LIMIT,
    JUDGE_TRANSCRIPT_TURNS,
)
from vibe.core.agent_loop._models import ToolDecision, ToolExecutionResponse
from vibe.core.agent_loop_hooks import AgentLoopHooksMixin
from vibe.core.logger import logger
from vibe.core.tools.base import ToolPermission
from vibe.core.tools.permissions import PermissionContext, RequiredPermission
from vibe.core.types import ApprovalResponse, Role

if TYPE_CHECKING:
    from vibe.core.agents.models import AgentProfile
    from vibe.core.config import ModelConfig, VibeConfig
    from vibe.core.llm.models import ResolvedToolCall
    from vibe.core.tools.base import BaseTool
    from vibe.core.tools.permissions import PermissionStore
    from vibe.core.tools.safety_judge import JudgeVerdict, SafetyJudge
    from vibe.core.types import ApprovalCallback


class AgentLoopSafetyMixin(AgentLoopHooksMixin):
    """Mixin that adds the safety-judge + approval gate to AgentLoop.

    See module docstring for the implicit contract with the host class.
    """

    # Declared for type-checking only; set by AgentLoop.__init__.
    approval_callback: ApprovalCallback | None
    pending_judge_deferral: str | None
    _permission_store: PermissionStore
    _judge_verdict_cache: Any  # OrderedDict[tuple, JudgeVerdict] — exact type below
    _judge_verdict_cache_maxsize: int
    _judge_model_alias_for_cache: str | None
    _current_user_message_id: str | None
    tool_manager: Any  # ToolManager — typed loosely to avoid a runtime import cycle

    @property
    def config(self) -> VibeConfig: ...

    @property
    def bypass_tool_permissions(self) -> bool: ...

    @property
    def agent_profile(self) -> AgentProfile: ...

    # ``messages``, ``_serialize_tool_input``, ``_fire_notification_hooks``, and
    # ``_patch_assistant_tool_call_args`` are inherited from AgentLoopHooksMixin
    # (see class bases) — redeclaring them here as stubs would shadow the real
    # implementations, so they are intentionally omitted.

    def effective_model(self) -> ModelConfig: ...

    def _get_extra_headers(self, provider: Any | None = None) -> dict[str, str]: ...

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

    async def _judge_tool_safety(
        self, tool_name: str, args: BaseModel, uncovered: list[RequiredPermission]
    ) -> ToolDecision | None:
        # Cleared each decision; set to the judge's reason when it defers so the
        # approval UI can show why the user is being asked. Must not leak stale
        # values to the next prompt.
        self.pending_judge_deferral = None
        judge = self._resolve_safety_judge()
        if judge is None:
            return None
        # Drop cached verdicts when the judge model changes: a verdict produced
        # under one model must not be reused after swapping to another.
        judge_model = self.config.safety_judge.model
        if judge_model != self._judge_model_alias_for_cache:
            self._judge_verdict_cache.clear()
            self._judge_model_alias_for_cache = judge_model
        # args_key is a hash of the FULL serialized args so two calls differing
        # only past the judge-input window get distinct cache keys; args_repr is
        # what the judge actually sees (capped at JUDGE_ARGS_LIMIT, with a
        # sentinel appended when truncated).
        args_key, args_repr, truncated = self._serialize_args(args)
        flagged_reasons = [rp.label for rp in uncovered]
        # Recent transcript gives the judge intent context (a call the user
        # asked for vs one the agent decided unprompted). Hashed into the cache
        # key so different contexts don't share a verdict.
        transcript = self._judge_transcript_window()
        transcript_key = hashlib.sha256(
            transcript.encode("utf-8", errors="replace")
        ).hexdigest()
        # Truncation blind spot: when the args exceed the judge's input window,
        # a destructive tail can hide beyond what the model sees. This method is
        # only reached for ASK-gated calls, and `uncovered` non-empty means a
        # risk flag already surfaced — so a truncated payload here would let the
        # judge rule on a blind prefix while a real flag exists. Force-defer to
        # the user instead of trusting an auto-approve on a partial payload. The
        # sentinel in args_repr is a second line of defense for any truncated
        # payload that still reaches the judge (e.g. via a direct caller).
        if truncated and uncovered:
            self.pending_judge_deferral = (
                "arguments were truncated past the judge's input window; the "
                "hidden tail cannot be verified safe"
            )
            logger.info(
                "Safety judge force-deferred tool %r to user: args truncated "
                "past the %d-char input window",
                tool_name,
                JUDGE_ARGS_LIMIT,
            )
            return None
        cache_key = (tool_name, args_key, tuple(flagged_reasons), transcript_key)
        # Reuse a real verdict for an identical call instead of re-querying the
        # judge model. Fail-closed verdicts (verdict.failed) are never stored,
        # so a transient timeout/error is retried on the next identical call.
        verdict = self._judge_verdict_cache_get(cache_key)
        if verdict is None:
            verdict = await judge.judge(
                tool_name, args_repr, flagged_reasons, transcript=transcript
            )
            if not verdict.failed:
                self._judge_verdict_cache_put(cache_key, verdict)
        else:
            logger.debug(
                "Safety judge verdict cache hit for tool %r (safe=%s)",
                tool_name,
                verdict.safe,
            )
        if not verdict.safe:
            self.pending_judge_deferral = verdict.reason
            # Refusal is otherwise invisible (looks identical to judge-off):
            # log it so it's clear the judge ran and deferred to the user.
            logger.info(
                "Safety judge deferred tool %r to user: %s", tool_name, verdict.reason
            )
            return None
        logger.info("Safety judge auto-approved tool %r: %s", tool_name, verdict.reason)
        return ToolDecision(
            verdict=ToolExecutionResponse.EXECUTE,
            approval_type=ToolPermission.ALWAYS,
            feedback=f"Auto-approved by safety judge: {verdict.reason}",
            judge_approved=True,
        )

    @staticmethod
    def _serialize_args(args: BaseModel) -> tuple[str, str, bool]:
        try:
            blob = args.model_dump_json()
        except Exception:
            blob = str(args)
        digest = hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()
        truncated = len(blob) > JUDGE_ARGS_LIMIT
        if truncated:
            repr_ = blob[:JUDGE_ARGS_LIMIT] + JUDGE_ARGS_TRUNCATED_SENTINEL
        else:
            repr_ = blob
        return digest, repr_, truncated

    def _judge_transcript_window(self) -> str:
        turns: list[str] = []
        for msg in reversed(self.messages):
            if len(turns) >= JUDGE_TRANSCRIPT_TURNS:
                break
            content = (msg.content or "").strip()
            if not content:
                continue
            if msg.role == Role.USER and not msg.injected:
                turns.append(f"user: {content}")
            elif msg.role == Role.ASSISTANT:
                turns.append(f"assistant: {content}")
        if not turns:
            return ""
        turns.reverse()
        text = "\n".join(turns)
        if len(text) > JUDGE_TRANSCRIPT_LIMIT:
            text = text[:JUDGE_TRANSCRIPT_LIMIT] + "\n...[truncated]"
        return text

    def _judge_verdict_cache_get(
        self, key: tuple[str, str, tuple[str, ...], str]
    ) -> JudgeVerdict | None:
        if self._judge_verdict_cache_maxsize <= 0:
            return None
        cache = self._judge_verdict_cache
        verdict = cache.get(key)
        if verdict is not None:
            cache.move_to_end(key)
        return verdict

    def _judge_verdict_cache_put(
        self, key: tuple[str, str, tuple[str, ...], str], verdict: JudgeVerdict
    ) -> None:
        if self._judge_verdict_cache_maxsize <= 0:
            return
        cache = self._judge_verdict_cache
        cache[key] = verdict
        cache.move_to_end(key)
        while len(cache) > self._judge_verdict_cache_maxsize:
            cache.popitem(last=False)

    def _apply_modification(
        self, tool_call: ResolvedToolCall, modified_args: dict[str, Any]
    ) -> tuple[ResolvedToolCall, dict[str, Any]] | ToolDecision:
        tool_class = tool_call.tool_class
        args_model, _ = tool_class._get_tool_args_results()
        try:
            new_validated = args_model.model_validate(modified_args)
        except Exception as exc:
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                approval_type=ToolPermission.ASK,
                feedback=f"Modified arguments failed validation and were rejected: {exc}",
            )
        new_tool_call = tool_call.model_copy(update={"validated_args": new_validated})
        new_tool_input = self._serialize_tool_input(new_tool_call)
        self._patch_assistant_tool_call_args(tool_call.call_id, new_tool_input)
        return new_tool_call, new_tool_input

    def _resolve_modification(
        self,
        tool_call: ResolvedToolCall,
        tool_input: dict[str, Any],
        decision: ToolDecision,
    ) -> tuple[ResolvedToolCall, dict[str, Any], ToolDecision]:
        if decision.modified_args is None:
            return tool_call, tool_input, decision
        modified = self._apply_modification(tool_call, decision.modified_args)
        if isinstance(modified, ToolDecision):
            return tool_call, tool_input, modified
        return modified[0], modified[1], decision

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

    def _resolve_safety_judge(self) -> SafetyJudge | None:
        judge_cfg = self.config.safety_judge
        if not judge_cfg.enabled or not judge_cfg.model:
            return None
        judge_model = next(
            (m for m in self.config.models if m.alias == judge_cfg.model), None
        )
        if judge_model is None or not self.config.is_model_available(judge_model):
            return None
        try:
            provider = self.config.get_provider_for_model(judge_model)
        except ValueError:
            logger.warning(
                "Safety judge model %r has no provider; disabling judge",
                judge_model.alias,
            )
            return None
        if judge_model.alias == self.config.active_model:
            logger.warning(
                "Safety judge model %r is the same as the active model; "
                "an independent judge model is recommended.",
                judge_model.alias,
            )
        from vibe.core.tools.safety_judge import SafetyJudge

        return SafetyJudge(
            model=judge_model,
            provider=provider,
            config=self.config.safety_judge,
            extra_headers=self._get_extra_headers(provider),
            timeout=self.config.api_timeout,
        )
