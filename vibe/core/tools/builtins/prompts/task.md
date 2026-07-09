Use `task` to delegate work to a subagent for independent execution.

## When to use
Context offload | specialized profile match | parallel independent questions | no user back-and-forth needed.

## Rules
- **Brief must be self-contained** — subagents cannot ask the user; include file:line refs and the exact question.
- **Right profile** — see Available Subagents. For file-mutating `editor`/`grunt`/`worker` patterns, load the `agent-delegation` skill.
- **Local tools first** — known file/symbol/flow → `read`/`lsp`/`grep` yourself; file count alone is not a reason to delegate.
- **Don't micromanage** the subagent's approach.

## Capabilities
- Read-only investigators: `explore`/`research`/`planner`/`reviewer`/`debugger`/`security`/`verifier` (no writes; some may run jailed bash).
- Write-capable: `editor`/`worker`/`grunt` — default `isolation='auto'` → isolated worktree; `isolation='off'` is in-process and approval-gated.
- Background by default (`task_id` returned immediately; result auto-delivered next turn). `async_run=false` blocks. Scripted fan-out → `launch_workflow`.
