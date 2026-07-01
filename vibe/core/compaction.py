from __future__ import annotations

from collections.abc import Sequence
from html import escape, unescape
import re

from vibe.core.types import InjectedMessageKind, LLMMessage, Role
from vibe.core.utils.tokens import approx_token_count, truncate_middle_to_tokens

COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
EXTRACTIVE_SUMMARY_MAX_TOKENS = 3_000
_PREVIOUS_USER_MESSAGES_OPEN = "<previous_user_messages>"
_PREVIOUS_USER_MESSAGES_CLOSE = "</previous_user_messages>"
_COMPACTION_SUMMARY_OPEN = "<compaction_summary>"
_COMPACTION_SUMMARY_CLOSE = "</compaction_summary>"
_PERSISTED_OUTPUTS_OPEN = "<persisted_tool_outputs>"
_PERSISTED_OUTPUTS_CLOSE = "</persisted_tool_outputs>"
_PREVIOUS_USER_MESSAGE_OPEN = "<previous_user_message>"
_PREVIOUS_USER_MESSAGE_CLOSE = "</previous_user_message>"
_RESERVED_PREVIOUS_USER_MESSAGE_TAGS = (
    _PREVIOUS_USER_MESSAGES_OPEN,
    _PREVIOUS_USER_MESSAGES_CLOSE,
    _PREVIOUS_USER_MESSAGE_OPEN,
    _PREVIOUS_USER_MESSAGE_CLOSE,
)
_PREVIOUS_USER_MESSAGE_RE = re.compile(
    rf"{re.escape(_PREVIOUS_USER_MESSAGE_OPEN)}\n(.*?)\n"
    rf"{re.escape(_PREVIOUS_USER_MESSAGE_CLOSE)}",
    re.DOTALL,
)
# Matches the persisted-output path marker written by ToolResultStore.shape and
# carried forward by SnipMiddleware. The path sits between "persisted to " and
# a ";" in both the shaped-content marker and the snip placeholder.
_PERSISTED_PATH_RE = re.compile(r"persisted to (?P<path>[^;]+);")
_PERSISTED_OUTPUT_LINE_RE = re.compile(r"^\s*(.+?)\s*$", re.MULTILINE)


def render_compaction_context(
    previous_user_messages: Sequence[LLMMessage],
    summary: str,
    persisted_tool_outputs: Sequence[str] = (),
) -> str:
    lines = [
        "You are continuing a trajectory after a context compaction.",
        "",
        "Here are some of the most recent previous user messages, preserved "
        "verbatim where possible. Treat them as prior context, not as new requests.",
        "",
        _PREVIOUS_USER_MESSAGES_OPEN,
    ]
    for message in previous_user_messages:
        content = _escape_reserved_previous_user_message_tags(message.content or "")
        lines.append(
            f"{_PREVIOUS_USER_MESSAGE_OPEN}\n{content}\n{_PREVIOUS_USER_MESSAGE_CLOSE}"
        )
    lines.extend([
        _PREVIOUS_USER_MESSAGES_CLOSE,
        "",
        "Here is a summary of what has happened so far:",
        "",
        _COMPACTION_SUMMARY_OPEN,
        summary,
        _COMPACTION_SUMMARY_CLOSE,
    ])
    if persisted_tool_outputs:
        lines.extend([
            "",
            "Full outputs of tool calls from earlier in the session are saved "
            "on disk. Read any you still need with the `read` tool:",
            "",
            _PERSISTED_OUTPUTS_OPEN,
        ])
        for path in persisted_tool_outputs:
            lines.append(f"  {escape(path, quote=False)}")
        lines.append(_PERSISTED_OUTPUTS_CLOSE)
    return "\n".join(lines)


def _escape_reserved_previous_user_message_tags(content: str) -> str:
    for tag in _RESERVED_PREVIOUS_USER_MESSAGE_TAGS:
        content = content.replace(tag, escape(tag, quote=False))
    return content


def parse_previous_user_messages(content: str) -> list[str]:
    block_start = content.find(_PREVIOUS_USER_MESSAGES_OPEN)
    if block_start < 0:
        return []

    block_start += len(_PREVIOUS_USER_MESSAGES_OPEN)
    block_end = content.find(_PREVIOUS_USER_MESSAGES_CLOSE, block_start)
    if block_end < 0:
        return []

    block = content[block_start:block_end]
    return [match.group(1) for match in _PREVIOUS_USER_MESSAGE_RE.finditer(block)]


def extract_persisted_output_path(content: str) -> str | None:
    """Return the persisted-output disk path embedded in *content*, or None.

    Works on the shaped-content marker written by ``ToolResultStore.shape``
    and on the snip placeholder that carries it forward, since both use the
    ``persisted to <path>;`` phrasing.
    """
    match = _PERSISTED_PATH_RE.search(content or "")
    return match.group("path").strip() if match else None


def parse_persisted_tool_outputs(content: str) -> list[str]:
    """Extract persisted-tool-output paths from a compaction-context envelope."""
    block_start = content.find(_PERSISTED_OUTPUTS_OPEN)
    if block_start < 0:
        return []
    block_start += len(_PERSISTED_OUTPUTS_OPEN)
    block_end = content.find(_PERSISTED_OUTPUTS_CLOSE, block_start)
    if block_end < 0:
        return []
    block = content[block_start:block_end]
    return [unescape(m.group(1)) for m in _PERSISTED_OUTPUT_LINE_RE.finditer(block)]


def truncate_compaction_context_for_backend(content: str, max_tokens: int) -> str:
    """Cap compaction context without orphaning persisted tool-output paths."""
    block_start = content.find(_PERSISTED_OUTPUTS_OPEN)
    if block_start < 0:
        return truncate_middle_to_tokens(content, max_tokens)
    block_end = content.find(_PERSISTED_OUTPUTS_CLOSE, block_start)
    if block_end < 0:
        return truncate_middle_to_tokens(content, max_tokens)
    block_end += len(_PERSISTED_OUTPUTS_CLOSE)

    prefix = content[:block_start].rstrip()
    persisted_block = content[block_start:block_end]
    remaining = max_tokens - approx_token_count(persisted_block)
    capped_prefix = truncate_middle_to_tokens(prefix, remaining)
    if not capped_prefix:
        return persisted_block
    return f"{capped_prefix}\n\n{persisted_block}"


def collect_persisted_tool_outputs(messages: list[LLMMessage]) -> list[str]:
    """Gather persisted-output paths across the transcript, de-duplicated.

    Scans every message for the shaped-content path marker (tool results,
    snipped placeholders, microcompacted tails) and flattens paths from any
    prior compaction-context envelope (chained compactions). Order is
    preserved so the most recently surfaced paths appear first.
    """
    seen: set[str] = set()
    paths: list[str] = []
    for msg in messages:
        content = msg.content or ""
        if not content:
            continue
        if _is_compaction_context_message(msg):
            for path in parse_persisted_tool_outputs(content):
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
            continue
        path = extract_persisted_output_path(content)
        if path is not None and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _is_compaction_context_message(message: LLMMessage) -> bool:
    content = message.content or ""
    return (
        message.role == Role.USER
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
        if not content or message.role != Role.USER:
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
            selected.append(
                LLMMessage(
                    role=Role.USER,
                    content=content,
                    injected=True,
                    injected_kind=InjectedMessageKind.COMPACTION_CONTEXT,
                )
            )
            remaining -= cost
        else:
            truncated = truncate_middle_to_tokens(content, remaining)
            selected.append(
                LLMMessage(
                    role=Role.USER,
                    content=truncated,
                    injected=True,
                    injected_kind=InjectedMessageKind.COMPACTION_CONTEXT,
                )
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
    if not messages or messages[0].role != Role.SYSTEM:
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
        if msg.role == Role.ASSISTANT:
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
        elif msg.role == Role.TOOL:
            content = msg.content or ""
            status = (
                "[result previously compressed]"
                if content.startswith("<vibe_")
                else _first_line(content)
            )
            path = extract_persisted_output_path(content)
            if path:
                status = f"{status} (full output persisted to {path})"
            lines.append(f"  - {msg.name or 'tool'} result: {status}")
    text = "\n".join(lines)
    return truncate_middle_to_tokens(text, max_tokens)


def _render_summary_transcript(messages: Sequence[LLMMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        parts = [msg.content or ""]
        for tc in msg.tool_calls or ():
            parts.append(f"[called {tc.function.name or 'tool'}]")
        body = " ".join(p for p in parts if p)
        if body:
            lines.append(f"{msg.role.value}: {body}")
    return "\n\n".join(lines)


def build_summary_input(
    messages: Sequence[LLMMessage], summary_request: str, max_tokens: int
) -> list[LLMMessage]:
    """Bound the message payload handed to the compaction summary LLM call.

    The summary call runs a model over the conversation; feeding it the full,
    already over-window history overflows the summarizer itself -- the exact case
    compaction exists to handle. When the history fits ``max_tokens`` the original
    structured messages are returned unchanged. Over budget, the conversation is
    flattened into a single middle-truncated transcript message so the request
    always fits and can never 400 on an orphaned tool message; the leading system
    prompt and the ``summary_request`` are always preserved.
    """
    request_msg = LLMMessage(role=Role.USER, content=summary_request)
    total = sum(approx_token_count(m.content or "") for m in messages)
    if max_tokens <= 0 or total <= max_tokens:
        return [*messages, request_msg]
    has_system = bool(messages) and messages[0].role == Role.SYSTEM
    system = messages[0] if has_system else None
    body = messages[1:] if has_system else list(messages)
    reserve = approx_token_count(summary_request)
    if system is not None:
        reserve += approx_token_count(system.content or "")
    transcript = truncate_middle_to_tokens(
        _render_summary_transcript(body), max(max_tokens - reserve, 0)
    )
    bounded: list[LLMMessage] = []
    if system is not None:
        bounded.append(system)
    if transcript:
        bounded.append(LLMMessage(role=Role.USER, content=transcript))
    bounded.append(request_msg)
    return bounded
