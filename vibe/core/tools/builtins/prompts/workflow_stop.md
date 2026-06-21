Use `workflow_stop` to stop one or all background workflow runs.

## When to Use This Tool

- **Runaway workflow**: a run is spending far more tokens than expected (check with `workflow_status` first). Stop it instead of letting it run.
- **Misbehaving agent**: an agent is stuck in a loop or producing nonsense; cancel the whole run.
- **Obsolete run**: the task changed and a launched workflow is no longer relevant.
- **Cleanup**: stop everything with `all=true` before launching a replacement.

## When NOT to Use

- Don't stop a run that is simply slow but progressing normally — check `workflow_status` to confirm it is actually stuck before cancelling.
- Don't use this to interrupt a workflow you want results from; it discards in-flight work.

## Arguments

- `run_id` (str, optional): the run id to stop, e.g. `wf-1`. Ignored when `all` is true.
- `all` (bool, default false): stop every active run.

Exactly one of `run_id` or `all=true` is required.

## Result

- `stopped`: whether at least one run was cancelled.
- `stopped_run_ids`: the run ids that were stopped.
- `message`: human-readable outcome. An already-finished or unknown run is reported as not stopped (not an error).

## Notes

- Stopping cancels the run's asyncio task immediately; in-flight agents are halted and their partial work is not recovered (resume replays only completed agents).
- This mirrors the `/workflows stop <id|all>` slash command but is callable from a model turn.
