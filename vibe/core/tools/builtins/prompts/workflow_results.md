Retrieve a workflow run's actual outputs — the script's `return_value` plus
per-agent responses, errors, and schema-validation detail. This is the pull
path for a run's result; `launch_workflow` only returns a `run_id`.

Completion is auto-delivered to your context as a user message — don't poll; end
your turn and resume when it arrives. Call this tool ONLY on a missed/truncated
delivery or when you need the structured `return_value`.

## When to Use This Tool

- **Missed/truncated delivery**: a run that the host turn never re-entered (the
  completion push landed in a finished turn) is fully recoverable here — the
  `return_value` is retained on the run entry and, for finished runs, persisted
  across sessions.
- **Recovering failed agents**: a failed agent's raw response is included (always,
  even with `raw=false`; the cap only shortens very long ones), plus
  `schema_errors` with field-level JSON-validation reasons — so you can see *why*
  output was rejected, not just *that* it was. Example:
  `$.findings[0].severity: 'medium' not in enum ['high','low']`.
- **Inspecting a running run**: results are retrievable mid-run (finalized agents
  appear as they complete); `return_value` is `None` until `main()` returns.

## When NOT to Use

- Diagnosing a stuck or runaway run → ONE `workflow_status` check (not routine
  progress checks), then `workflow_stop`.
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
