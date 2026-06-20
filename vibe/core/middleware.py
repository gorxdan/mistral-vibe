from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, Protocol

from vibe.core.agents import AgentProfile
from vibe.core.utils import VIBE_WARNING_TAG

if TYPE_CHECKING:
    from vibe.core.config import VibeConfig
    from vibe.core.types import AgentStats, LLMMessage, MessageList


class MiddlewareAction(StrEnum):
    CONTINUE = auto()
    STOP = auto()
    COMPACT = auto()
    INJECT_MESSAGE = auto()


class ResetReason(StrEnum):
    STOP = auto()
    COMPACT = auto()


@dataclass
class ConversationContext:
    messages: MessageList
    stats: AgentStats
    config: VibeConfig


@dataclass
class MiddlewareResult:
    action: MiddlewareAction = MiddlewareAction.CONTINUE
    message: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMiddleware(Protocol):
    async def before_turn(self, context: ConversationContext) -> MiddlewareResult: ...

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None: ...


class TurnLimitMiddleware:
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if context.stats.steps - 1 >= self.max_turns:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=f"Turn limit of {self.max_turns} reached",
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class PriceLimitMiddleware:
    def __init__(self, max_price: float) -> None:
        self.max_price = max_price

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if context.stats.session_cost > self.max_price:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=f"Price limit exceeded: ${context.stats.session_cost:.4f} > ${self.max_price:.2f}",
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class TokenLimitMiddleware:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if context.stats.session_total_llm_tokens > self.max_tokens:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=(
                    "Token limit exceeded: "
                    f"{context.stats.session_total_llm_tokens:,} > {self.max_tokens:,}"
                ),
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class AutoCompactMiddleware:
    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        threshold = context.config.get_active_model().auto_compact_threshold

        if threshold > 0 and context.stats.context_tokens >= threshold:
            return MiddlewareResult(
                action=MiddlewareAction.COMPACT,
                metadata={
                    "old_tokens": context.stats.context_tokens,
                    "threshold": threshold,
                },
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


_SNIP_OPEN = "<vibe_snipped>"
_SNIP_CLOSE = "</vibe_snipped>"


class ContextShaperMiddleware:
    """Base for cheap, local, in-place context shapers run before AutoCompact.

    Shapers mutate ``context.messages`` directly and return CONTINUE; they never
    call an LLM. They read their live config from ``context.config`` so a
    mid-session config edit takes effect without rebuilding the pipeline.
    """

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass

    @staticmethod
    def _threshold(context: ConversationContext) -> int:
        return context.config.get_active_model().auto_compact_threshold

    @staticmethod
    def _estimated_tokens(context: ConversationContext) -> int:
        from vibe.core.utils.tokens import approx_token_count

        local = sum(approx_token_count(m.content or "") for m in context.messages)
        # stats.context_tokens is one turn behind / 0 after compaction; take the
        # larger so a stale-low value never suppresses shaping.
        return max(context.stats.context_tokens, local)

    @staticmethod
    def _protected_prefix_len(messages: MessageList, guard_tokens: int) -> int:
        """Leading messages that must never be edited: system + any compaction
        context + a cache-stable prefix band of ~guard_tokens.
        """
        from vibe.core.compaction import _is_compaction_context_message
        from vibe.core.utils.tokens import approx_token_count

        if len(messages) == 0:
            return 0
        n = 1  # system prompt
        while n < len(messages) and _is_compaction_context_message(messages[n]):
            n += 1
        acc = sum(approx_token_count(messages[i].content or "") for i in range(n))
        while n < len(messages) and acc < guard_tokens:
            acc += approx_token_count(messages[n].content or "")
            n += 1
        return n

    @staticmethod
    def _protected_suffix_len(
        messages: MessageList, keep_recent: int, prefix_len: int
    ) -> int:
        """Trailing messages kept verbatim (the working set)."""
        return min(keep_recent, max(0, len(messages) - prefix_len))

    @staticmethod
    def _is_real_user_message(msg: LLMMessage) -> bool:
        from vibe.core.types import Role

        return msg.role == Role.user and not msg.injected


class SnipMiddleware(ContextShaperMiddleware):
    """Elide old, large messages to a placeholder once context is moderately
    full. Preserves message structure (role, tool linkage) so the request stays
    valid; only the content/args/images are dropped.
    """

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        from vibe.core.utils.tokens import approx_token_count

        cfg = context.config.context_shaping.snip
        threshold = self._threshold(context)
        if not cfg.enabled or threshold <= 0:
            return MiddlewareResult()
        est = self._estimated_tokens(context)
        if est < cfg.high_watermark * threshold:
            return MiddlewareResult()

        messages = context.messages
        prefix = self._protected_prefix_len(
            messages, context.config.context_shaping.cache_prefix_guard_tokens
        )
        suffix = self._protected_suffix_len(messages, cfg.keep_recent_turns, prefix)
        band = range(prefix, len(messages) - suffix)
        candidates = [
            i
            for i in band
            if approx_token_count(messages[i].content or "") >= cfg.min_message_tokens
            and not (messages[i].content or "").startswith(_SNIP_OPEN)
            and not self._is_real_user_message(messages[i])
        ]
        candidates.sort(
            key=lambda i: approx_token_count(messages[i].content or ""), reverse=True
        )
        target = cfg.target * threshold
        for i in candidates:
            if est <= target:
                break
            before = approx_token_count(messages[i].content or "")
            new_msg = self._snip(messages[i])
            messages.replace_at(i, new_msg)
            est -= max(0, before - approx_token_count(new_msg.content or ""))
        return MiddlewareResult()

    @staticmethod
    def _snip(msg: LLMMessage) -> LLMMessage:
        from vibe.core.types import FunctionCall, ToolCall
        from vibe.core.utils.tokens import approx_token_count

        n = approx_token_count(msg.content or "")
        placeholder = (
            f"{_SNIP_OPEN} {n} tokens of older {msg.role} content elided "
            f"{_SNIP_CLOSE}"
        )
        new_tool_calls = None
        if msg.tool_calls:
            # Keep id/index/name (tool_call<->tool_result linkage) but blank args.
            new_tool_calls = [
                ToolCall(
                    id=tc.id,
                    index=tc.index,
                    type=tc.type,
                    function=FunctionCall(name=tc.function.name, arguments="{}"),
                )
                for tc in msg.tool_calls
            ]
        return msg.model_copy(
            update={
                "content": placeholder,
                "images": None,
                "tool_calls": new_tool_calls,
                "reasoning_content": None,
                "reasoning_state": None,
            }
        )


class MicrocompactMiddleware(ContextShaperMiddleware):
    """Compress (head+tail truncate) the oldest oversized messages, rate-limited
    per turn to keep the provider cache stable. No LLM call.
    """

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        from vibe.core.utils.tokens import approx_token_count, truncate_middle_to_tokens

        cfg = context.config.context_shaping.microcompact
        threshold = self._threshold(context)
        if not cfg.enabled or threshold <= 0:
            return MiddlewareResult()
        est = self._estimated_tokens(context)
        if est < cfg.high_watermark * threshold:
            return MiddlewareResult()

        messages = context.messages
        prefix = self._protected_prefix_len(
            messages, context.config.context_shaping.cache_prefix_guard_tokens
        )
        suffix = self._protected_suffix_len(
            messages, context.config.context_shaping.snip.keep_recent_turns, prefix
        )
        target = cfg.target * threshold
        done = 0
        for i in range(prefix, len(messages) - suffix):  # oldest first
            if done >= cfg.max_blocks_per_turn or est <= target:
                break
            msg = messages[i]
            content = msg.content or ""
            if self._is_real_user_message(msg) or content.startswith(_SNIP_OPEN):
                continue
            if approx_token_count(content) <= cfg.per_message_cap_tokens:
                continue  # already small / previously compressed
            new_content = truncate_middle_to_tokens(content, cfg.per_message_cap_tokens)
            messages.replace_at(i, msg.model_copy(update={"content": new_content}))
            est -= approx_token_count(content) - approx_token_count(new_content)
            done += 1
        return MiddlewareResult()


class ContextWarningMiddleware:
    def __init__(self, threshold_percent: float = 0.5) -> None:
        self.threshold_percent = threshold_percent
        self.has_warned = False

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if self.has_warned:
            return MiddlewareResult()

        max_context = context.config.get_active_model().auto_compact_threshold
        if max_context <= 0:
            return MiddlewareResult()

        if context.stats.context_tokens >= max_context * self.threshold_percent:
            self.has_warned = True

            percentage_used = (context.stats.context_tokens / max_context) * 100
            warning_msg = f"<{VIBE_WARNING_TAG}>You have used {percentage_used:.0f}% of your total context ({context.stats.context_tokens:,}/{max_context:,} tokens)</{VIBE_WARNING_TAG}>"

            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=warning_msg
            )

        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        self.has_warned = False


def make_plan_agent_reminder(
    plan_file_path: str,
    *,
    has_ask_user_question: bool = True,
    has_exit_plan_mode: bool = True,
) -> str:
    instructions = [
        "Research the user's query using read-only tools (grep, read, etc.)"
    ]
    if has_ask_user_question:
        instructions.append(
            "If you are unsure about requirements or approach, use the ask_user_question tool to clarify before finalizing your plan"
        )
    instructions.append("Write your plan to the plan file above")
    if has_exit_plan_mode:
        instructions.append(
            "When your plan is complete, call the exit_plan_mode tool to request user approval and switch to implementation mode"
        )
    else:
        instructions.append(
            "When your plan is complete, present it to the user and tell them to switch modes if they approve the plan"
        )
    numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(instructions, start=1))

    return f"""<{VIBE_WARNING_TAG}>Plan mode is active. You MUST NOT make any edits (except to the plan file below, or in your scratchpad), run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supersedes any other instructions you have received.

## Plan File Info
Create or edit your plan at {plan_file_path} using the write_file and edit tools.
Build your plan incrementally by writing to or editing this file.
This is the only file you are allowed to edit. Make sure to create it early and edit as soon as you internally update your plan.

## Instructions
{numbered}</{VIBE_WARNING_TAG}>"""


PLAN_AGENT_EXIT = f"""<{VIBE_WARNING_TAG}>Plan mode has ended. If you have a plan ready, you can now start executing it. If not, you can now use editing tools and make changes to the system.</{VIBE_WARNING_TAG}>"""

CHAT_AGENT_REMINDER = f"""<{VIBE_WARNING_TAG}>Chat mode is active. The user wants to have a conversation -- ask questions, get explanations, or discuss code and architecture. You MUST NOT make any edits, run any non-readonly tools, or otherwise make any changes to the system. This supersedes any other instructions you have received. Instead, you should:
1. Answer the user's questions directly and comprehensively
2. Explain code, concepts, or architecture as requested
3. Use read-only tools (grep, read) to look up relevant code when needed
4. Focus on being informative and conversational -- your response IS the deliverable, not a precursor to action</{VIBE_WARNING_TAG}>"""

CHAT_AGENT_EXIT = f"""<{VIBE_WARNING_TAG}>Chat mode has ended. You can now use editing tools and make changes to the system.</{VIBE_WARNING_TAG}>"""


class ReadOnlyAgentMiddleware:
    def __init__(
        self,
        profile_getter: Callable[[], AgentProfile],
        agent_name: str,
        reminder: str | Callable[[], str],
        exit_message: str,
    ) -> None:
        self._profile_getter = profile_getter
        self._agent_name = agent_name
        self._reminder = reminder
        self.exit_message = exit_message
        self._was_active = False

    @property
    def reminder(self) -> str:
        return self._reminder() if callable(self._reminder) else self._reminder

    def _is_active(self) -> bool:
        return self._profile_getter().name == self._agent_name

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        is_active = self._is_active()
        was_active = self._was_active

        if was_active and not is_active:
            self._was_active = False
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=self.exit_message
            )

        if is_active and not was_active:
            self._was_active = True
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=self.reminder
            )

        self._was_active = is_active
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        self._was_active = False


class MiddlewarePipeline:
    def __init__(self) -> None:
        self.middlewares: list[ConversationMiddleware] = []

    def add(self, middleware: ConversationMiddleware) -> MiddlewarePipeline:
        self.middlewares.append(middleware)
        return self

    def clear(self) -> None:
        self.middlewares.clear()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        for mw in self.middlewares:
            mw.reset(reset_reason)

    async def run_before_turn(self, context: ConversationContext) -> MiddlewareResult:
        messages_to_inject = []
        for mw in self.middlewares:
            result = await mw.before_turn(context)
            if result.action == MiddlewareAction.INJECT_MESSAGE and result.message:
                messages_to_inject.append(result.message)
            elif result.action in {MiddlewareAction.STOP, MiddlewareAction.COMPACT}:
                return result
        if messages_to_inject:
            combined_message = "\n\n".join(messages_to_inject)
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=combined_message
            )

        return MiddlewareResult()
