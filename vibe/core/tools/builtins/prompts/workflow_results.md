Retrieve a workflow run's actual outputs — the script's `return_value` plus
per-agent responses, errors, and schema-validation detail. This is the pull
path for a run's result; `launch_workflow` only returns a `run_id`.

## When to Use This Tool

- **After `launch_workflow`**: the run executes in the background, so its
  `return_value` and agent outputs are NOT in the launch result. They are
  auto-delivered as a message on completion — but call this when that message
  was missed, truncated, or you need the structured value.
- **Recovering failed agents**: a failed agent's raw response is included, plus
  `schema_errors` with field-level JSON-validation reasons (e.g.
  `$.findings[0].severity: 'medium' not in enum ['high','low']`) — so you can
  see *why* an agent's output was rejected, not just *that* it was.
- **Inspecting a running run**: results are also retrievable mid-run (finalized
  agents appear as they complete). `status` tells you if the run is still in
  flight; `return_value` is `None` until `main()` returns.

## When NOT to Use

- Live progress (agents in flight, phases, token totals, budget) →
  `workflow_status`.
- Stopping a run → `workflow_stop`.

## Arguments

- `run_id` (required) — e.g. `wf-1`. No "all runs" form.
- `phase` (optional) — filter to one named phase.
- `raw` (default false) — truncate each agent response to 4KB and the
  `return_value` to ~16KB. Pass `raw=true` for full, untruncated outputs.

## Result Shape

- `return_value` — what the script's `main()` returned (`None` while running).
  Structured values (dict/list) pass through when they fit the cap; larger
  values come back as a truncated string unless `raw=true`.
- `agent_results` — one dict per finalized agent: `{label, agent, phase,
  completed, response, error, schema_errors, tokens_in, tokens_out}`.
  `schema_errors` is the field-level list when JSON-schema validation failed,
  empty otherwise.
- `phases` — `{name, agents, completed, failed}` per phase.
- `status` — `running | paused | completed | completed_with_failures | failed |
  stopped`.

## Notes

- A run that the host turn never re-entered (the completion push landed in a
  finished turn) is fully recoverable here — the `return_value` is retained on
  the run entry and (for finished runs) persisted across sessions.
- Failed agents' raw responses are always included, even with `raw=false` (the
  cap only shortens very long ones).
