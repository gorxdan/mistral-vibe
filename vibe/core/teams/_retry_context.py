from __future__ import annotations

from vibe.core.tasking import TaskOutcome

__all__ = ["bounded_retry_context"]

_MAX_RETRY_CONTEXT_CHARS = 4_096
_TRUNCATION_MARKER = "\n...[retry context truncated]"


def bounded_retry_context(outcome: TaskOutcome | None) -> str:
    if outcome is None or not outcome.retryable:
        return ""
    lines = [
        "Previous attempt was queued for retry. Address these exact failures; "
        "do not repeat the same action."
    ]
    for label, items in (
        ("Diagnostic", outcome.diagnostics),
        ("Check evidence", outcome.evidence),
    ):
        for item in items:
            candidate = f"{label}:\n{item}"
            joined = "\n".join((*lines, candidate))
            if len(joined) <= _MAX_RETRY_CONTEXT_CHARS:
                lines.append(candidate)
                continue
            remaining = _MAX_RETRY_CONTEXT_CHARS - len("\n".join(lines)) - 1
            if remaining > len(_TRUNCATION_MARKER):
                lines.append(
                    candidate[: remaining - len(_TRUNCATION_MARKER)]
                    + _TRUNCATION_MARKER
                )
            return "\n".join(lines)
    return "\n".join(lines)
