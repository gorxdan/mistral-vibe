Use `task` to delegate work to a subagent for independent execution.

## When to Use This Tool

- **Context management**: Delegate tasks that would consume too much main conversation context
- **Specialized work**: Use the appropriate subagent for the type of task (exploration, research, etc.)
- **Parallel execution**: Launch multiple subagents for independent tasks
- **Autonomous work**: Tasks that don't require back-and-forth with the user

## Best Practices

1. **Write clear, detailed task descriptions** - The subagent works autonomously, so provide enough context for it to succeed independently

2. **Choose the right subagent** - Match the subagent to the task type (see available subagents in system prompt)

3. **Prefer direct tools for simple operations** - If you know exactly which file to read or pattern to search, use those tools directly instead of spawning a subagent

4. **Trust the subagent's judgment** - Let it explore and find information without micromanaging the approach

## Capabilities & limits

- **The investigation profiles are read-only.** `explore`, `research`, `planner`, `reviewer`, `debugger`, and `security` cannot write or edit files — they read, search, and report back. `reviewer`, `debugger`, and `security` may run `bash` for targeted checks; `bash` stays approval-gated, so it only runs if an approval path is available and is skipped in a headless/non-interactive run.
- **`worker` is the exception** — it has the full tool set (including writes), but it is meant for workflows with `isolation='worktree'`, where it runs as an auto-approved subprocess. In a plain `task` call a `worker`'s write/exec tools are approval-gated like any other, so don't rely on a `task`-spawned `worker` to actually mutate files — do edits yourself.
- Subagents **cannot ask the user questions** — give each a self-contained brief with everything it needs.
- Results are returned as text when the subagent completes.
