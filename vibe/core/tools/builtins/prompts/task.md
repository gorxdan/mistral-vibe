Use `task` to delegate work to a subagent for independent execution.

## When to use
Context offload (work that would bloat main context) | specialized work (match the subagent to the task) | parallel independent tasks | autonomous work needing no user back-and-forth.

## Best practices
- **Clear, detailed brief** — the subagent works autonomously and can't ask you questions; give it everything it needs to succeed.
- **Right subagent** — match the profile to the task type (see available subagents in the system prompt).
- **Baseline first** — for an unfamiliar repo, map packages with `glob`, find central symbols/callers with `lsp`, and read the entry points before delegating.
- **Prefer direct tools** — if you know the file/symbol/flow, use `read`/`lsp`/`grep` yourself; file count alone doesn't make a task delegable.
- **Don't micromanage** the subagent's approach.

## Capabilities & limits
- Investigation profiles are **read-only**: `explore`/`research`/`planner`/`reviewer`/`debugger`/`security` read, search, and report — no writes. `reviewer`/`debugger`/`security` may run approval-gated `bash` (skipped in a headless run).
- `worker`/`grunt` have the full tool set but are meant for `isolation='worktree'` workflows (auto-approved subprocess). In a plain `task` call their write/exec tools are approval-gated — don't rely on a `task`-spawned `worker`/`grunt` to mutate files; do edits yourself.
- Subagents can't ask the user — each brief must be self-contained.

## Background by default
A `task` call returns a `task_id` immediately and runs in the background; its result is delivered at the top of a later turn (you're auto-resumed). The run shows in the Tasks pane and the `background` tool, and is cancellable with `background stop <task_id>`. Pass `async_run=false` to block and get the result inline this turn. For scripted parallel fan-out with a budget cap and schema validation, prefer `launch_workflow`.
