Use the `todo` tool to track a task list.

- `action: "read"` — view the current list.
- `action: "write"` — replace the ENTIRE list (omitted items are dropped; resend everything you keep).

Use proactively for multi-step (3+ steps) or non-trivial work; capture requirements as you get them. Skip for trivial (<3 steps) or conversational tasks.

Rules: exactly ONE task `in_progress` at a time. Mark `in_progress` BEFORE starting, `completed` immediately after — only when FULLY done (no failing tests, no partial work). When blocked, keep it `in_progress` and add a blocker task. Keep items specific; drop irrelevant ones.

Item schema detail and a worked example: `tool-guides` skill.
