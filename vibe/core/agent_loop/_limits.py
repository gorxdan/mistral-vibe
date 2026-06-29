"""Tuning constants for the agent loop.

Extracted from the loop module so the knobs that bound tool-result size, the
safety-judge input window, and the subagent fan-out cap live in one place and
stay free of any ``self``/class coupling.
"""

from __future__ import annotations

# Central cap on a single tool result's size before it enters the conversation.
# Tools may self-limit, but read/MCP/connector tools can return arbitrarily large
# blobs; this keeps one oversized result from blowing the context window (which
# would otherwise hard-fail the turn). ~100k chars ≈ 25k tokens.
MAX_TOOL_RESULT_CHARS = 100_000

# Inline preview size (head 75% + tail 25%) when a result exceeds the cap and is
# persisted to disk. Deliberately smaller than the cap so one oversized result
# no longer costs ~25k tokens of context; the full output is recoverable via the
# `read` tool using the path surfaced in the preview marker.
TOOL_RESULT_PREVIEW_CHARS = 12_000

# Aggregate cap on all tool results from a single parallel-tool-call turn.
# Prevents N medium results (each under the per-result cap) from collectively
# flooding context. Full content is persisted before any inline compression.
AGGREGATE_TOOL_RESULT_CHARS = 200_000

# A single result may occupy up to this fraction of the model's context budget
# before it is previewed-and-persisted. Scaling the fixed cap above to the
# window stops large-context models (e.g. glm, 880k) from truncating big reads —
# which forces ranged re-reads; small windows stay at MAX_TOOL_RESULT_CHARS via
# the floor, so behaviour is unchanged below a ~500k-token window.
TOOL_RESULT_WINDOW_FRACTION = 0.05
TOOL_RESULT_CHARS_PER_TOKEN = 4

# Safety-judge input window. _serialize_args hands the judge only this many
# chars of the serialized tool args. A destructive tail hidden past the cut is
# invisible to the judge, so (a) a sentinel is appended to the truncated repr
# warning the model it is judging a PARTIAL payload, and (b) _judge_tool_safety
# force-defers to the user when such a truncated call also carries a risk flag
# (uncovered permission) — never auto-approving on a blind prefix.
JUDGE_ARGS_LIMIT = 4000
JUDGE_ARGS_TRUNCATED_SENTINEL = (
    "\n\n...[TRUNCATED — the judge sees only the first "
    f"{JUDGE_ARGS_LIMIT} chars of these arguments. A destructive payload "
    "could be hidden beyond this point; do not auto-approve on the basis of "
    "the visible prefix.]"
)
# Capped recent-transcript window handed to the safety judge so it can tell a
# call the user explicitly requested from one the agent decided unprompted.
# Last user/assistant turns only (tool results and injections are noise), and
# the total is char-bounded so it never dominates the judge's input budget.
JUDGE_TRANSCRIPT_LIMIT = 2000
JUDGE_TRANSCRIPT_TURNS = 4

# Cap on how many subagent (task) fan-outs run concurrently in one turn. Bounds
# backend throughput contention / rate-limiting when the model emits several
# independent read-only task calls at once; ordinary concurrent tools (read,
# grep, glob) are not gated.
MAX_CONCURRENT_SUBAGENTS = 4
