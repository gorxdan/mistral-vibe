from __future__ import annotations

from collections.abc import Sequence
from html import escape, unescape
import re

from vibe.core.types import LLMMessage, Role
from vibe.core.utils.tokens import approx_token_count, truncate_middle_to_tokens

COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
EXTRACTIVE_SUMMARY_MAX_TOKENS = 3_000
_PREVIOUS_USER_MESSAGES_OPEN = "<previous_user_messages>"
_PREVIOUS_USER_MESSAGES_CLOSE = "</previous_user_messages>"
_COMPACTION_SUMMARY_OPEN = "<compaction_summary>"
_COMPACTION_SUMMARY_CLOSE = "</compaction_summary>"
_PREVIOUS_USER_MESSAGE_RE = re.compile(
    r"<previous_user_message_(\d+)>(.*?)</previous_user_message_\1>", re.DOTALL
)


def render_compaction_context(
    previous_user_messages: Sequence[LLMMessage], summary: str
) -> str:
    lines = [
        "You are continuing a trajectory after a context compaction.",
        "",
        "Here are some of the most recent previous user messages, preserved "
        "verbatim where possible. Treat them as prior context, not as new requests.",
        "",
        _PREVIOUS_USER_MESSAGES_OPEN,
    ]
    for idx, message in enumerate(previous_user_messages):
        content = escape(message.content or "", quote=False)
        lines.append(
            f"<previous_user_message_{idx}>{content}</previous_user_message_{idx}>"
        )
    lines.extend([
        _PREVIOUS_USER_MESSAGES_CLOSE,
        "",
        "Here is a summary of what has happened so far:",
        "",
        _COMPACTION_SUMMARY_OPEN,
        escape(summary, quote=False),
        _COMPACTION_SUMMARY_CLOSE,
    ])
    return "\n".join(lines)


def parse_previous_user_messages(content: str) -> list[str]:
    block_start = content.find(_PREVIOUS_USER_MESSAGES_OPEN)
    if block_start < 0:
        return []

    block_start += len(_PREVIOUS_USER_MESSAGES_OPEN)
    block_end = content.find(_PREVIOUS_USER_MESSAGES_CLOSE, block_start)
    if block_end < 0:
        return []

    block = content[block_start:block_end]
    matches = list(_PREVIOUS_USER_MESSAGE_RE.finditer(block))
    if not matches:
        return []

    previous_user_messages: list[str] = []
    for expected_idx, match in enumerate(matches):
        if int(match.group(1)) != expected_idx:
            return []
        previous_user_messages.append(unescape(match.group(2)))
    return previous_user_messages


def _is_compaction_context_message(message: LLMMessage) -> bool:
    content = message.content or ""
    return (
        message.role == Role.user
        and message.injected
        and _PREVIOUS_USER_MESSAGES_OPEN in content
        and _PREVIOUS_USER_MESSAGES_CLOSE in content
        and _COMPACTION_SUMMARY_OPEN in content
        and _COMPACTION_SUMMARY_CLOSE in content
    )


def collect_prior_user_messages(
    messages: list[LLMMessage],
    summary_prefix: str,
    max_tokens: int = COMPACT_USER_MESSAGE_MAX_TOKENS,
) -> list[LLMMessage]:
    """Pick user messages to preserve through compaction.

    Walks newest-first within a token budget, dropping system-internal
    injections and prior compaction summaries, middle-truncating the message
    that spills over. Previously preserved user messages are parsed from the
    compaction context envelope and merged with newer real user turns.
    """
    candidates: list[str] = []
    for message in messages:
        content = message.content or ""
        if not content or message.role != Role.user:
            continue

        if _is_compaction_context_message(message):
            candidates.extend(parse_previous_user_messages(content))
            continue

        if message.injected and content.startswith(summary_prefix):
            continue

        if message.injected:
            continue

        candidates.append(content)

    selected: list[LLMMessage] = []
    remaining = max_tokens
    for content in reversed(candidates):
        if remaining <= 0:
            break
        cost = approx_token_count(content)
        if cost <= remaining:
            selected.append(LLMMessage(role=Role.user, content=content, injected=True))
            remaining -= cost
        else:
            truncated = truncate_middle_to_tokens(content, remaining)
            selected.append(
                LLMMessage(role=Role.user, content=truncated, injected=True)
            )
            remaining = 0

    selected.reverse()
    return selected


def collect_leading_injected_context(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Return the leading injected environment-context messages to preserve.

    These are the consecutive injected (non-compaction-context) messages
    immediately after the system message — environment context, file-tree,
    AGENTS.md, deep-memory — set up at session start. They are dropped by
    :func:`collect_prior_user_messages` (which skips every ``injected=True``
    message), so without re-injection they vanish after every compaction and
    the model loses its grounding mid-session.

    Stops at the first non-injected message (real conversation) or a prior
    compaction-context message, so mid-conversation middleware injections and
    stale summaries are never carried forward.
    """
    if not messages or messages[0].role != Role.system:
        return []
    leading: list[LLMMessage] = []
    for msg in messages[1:]:
        if not msg.injected:
            break
        if _is_compaction_context_message(msg):
            break
        leading.append(msg)
    return leading


def _first_line(text: str, limit: int = 200) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line[:limit]


def build_extractive_summary(
    messages: Sequence[LLMMessage], *, max_tokens: int = EXTRACTIVE_SUMMARY_MAX_TOKENS
) -> str:
    """Structural, no-LLM summary of a transcript for degraded-mode compaction.

    Used as a fallback when the compaction LLM call fails (outage, rate limit),
    so a model error does not erase the entire session trace. Prior user
    messages are preserved separately by ``collect_prior_user_messages`` and
    rendered around this summary, so this focuses on what the agent did:
    assistant turn intent (first line) and tool calls with result status lines.
    """
    lines = [
        "Structural trace of prior turns (auto-generated; the model-generated "
        "summary was unavailable for this compaction):"
    ]
    for msg in messages:
        if msg.role == Role.assistant:
            content = msg.content or ""
            intent = (
                "[content previously elided]"
                if content.startswith("<vibe_")
                else _first_line(content)
            )
            if intent:
                lines.append(f"- assistant: {intent}")
            for tc in msg.tool_calls or ():
                lines.append(f"  - called tool: {tc.function.name}")
        elif msg.role == Role.tool:
            content = msg.content or ""
            status = (
                "[result previously compressed]"
                if content.startswith("<vibe_")
                else _first_line(content)
            )
            lines.append(f"  - {msg.name or 'tool'} result: {status}")
    text = "\n".join(lines)
    return truncate_middle_to_tokens(text, max_tokens)
