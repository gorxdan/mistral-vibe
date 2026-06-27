Use `workflow_stop` to stop one or all background workflow runs.

**When to use** (confirm via `workflow_status` first): runaway/over-budget run | agent stuck or looping | obsolete run the task no longer needs | `all=true` cleanup before relaunching a replacement.

**When NOT to use**: a run that is slow but progressing normally (check `workflow_status` first); cancelling discards in-flight work you may want results from.

**Args**: `run_id` (str, optional) — run id to stop, e.g. `wf-1`; ignored when `all`. `all` (bool, default false) — stop every active run. Exactly one of `run_id` or `all=true` is required.

**Result**: `stopped` (whether ≥1 run was cancelled) | `stopped_run_ids` (ids stopped) | `message` (human-readable outcome; an already-finished or unknown run is reported as not stopped, not an error).

**Notes**: Cancellation is immediate; in-flight agents are halted and their partial work is not recovered (resume replays only completed agents). Mirrors the `/workflows stop <id|all>` slash command but is callable from a model turn.
