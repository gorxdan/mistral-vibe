Use `task` to delegate work to a subagent for independent execution.

## When to Use This Tool

Use for: context offload (work that would bloat main context) | specialized work (match the subagent to the task: exploration, research, etc.) | parallel independent tasks | autonomous work needing no user back-and-forth.

## Best Practices

1. **Write clear, detailed task descriptions** - the subagent works autonomously, so give it enough context to succeed independently.
2. **Choose the right subagent** - match it to the task type (see available subagents in system prompt).
3. **Establish a local baseline first** - for an unfamiliar repository, map packages with `glob`, identify central symbols and callers with `lsp`, and read the entry points before delegating.
4. **Prefer direct tools for coherent lookups** - if you know which file, symbol, or flow to inspect, use `read`/`lsp`/`grep` directly instead of spawning a subagent. File count alone does not make a task delegable.
5. **Trust the subagent's judgment** - don't micromanage the approach.

## Capabilities & limits

- **The investigation profiles are read-only.** `explore`, `research`, `planner`, `reviewer`, `debugger`, and `security` cannot write or edit files — they read, search, and report back. `reviewer`, `debugger`, and `security` may run `bash` for targeted checks; `bash` stays approval-gated, so it only runs if an approval path is available and is skipped in a headless/non-interactive run.
- **`worker` is the exception** — it has the full tool set (including writes), but it is meant for workflows with `isolation='worktree'`, where it runs as an auto-approved subprocess. In a plain `task` call a `worker`'s write/exec tools are approval-gated like any other, so don't rely on a `task`-spawned `worker` to actually mutate files — do edits yourself.
- Subagents **cannot ask the user questions** — give each a self-contained brief with everything it needs.
- Results are returned as text when the subagent completes.

## Non-blocking delegation (`async_run=true`)

For isolated (write-capable) subagents — `worker`, `editor`, `auto-approve`, or any profile with `bash`/`write_file`/`edit` — pass `async_run=true` to launch the subagent in its own git worktree subprocess and get a `task_id` back immediately instead of blocking until completion (same as the synchronous isolated path; only the parent's wait is removed).

- `async_run=true` is rejected for read-only in-process profiles (e.g. `explore`) — they share the parent's event loop, so "async" would not unblock it. Use `launch_workflow` with `parallel()` for in-process fan-out instead.
- The running task is visible via the `background` tool and cancellable with `background stop <task_id>`.
- Completion surfaces at the top of the next parent turn as a `BackgroundTaskCompletedEvent` carrying the subagent's response.

Use it for fan-out: 3+ independent write-capable delegations where the parent should keep working instead of waiting serially. For scripted fan-out with a budget cap and schema validation, prefer `launch_workflow`.
