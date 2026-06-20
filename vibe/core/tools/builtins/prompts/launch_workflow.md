Use `launch_workflow` to run a workflow script that orchestrates parallel agents in the background.

## When to Use This Tool

- **Multi-agent tasks**: Codebase audits, large migrations, cross-checked research that needs 3+ independent agents
- **Adversarial verification**: Findings that should be cross-checked by multiple skeptics
- **Dynamic loops**: Tasks that need to loop until a condition is met (dry rounds, budget exhaustion)
- **Parallel exploration**: Investigating multiple directories or angles simultaneously

## When NOT to Use

- Single-file edits or quick questions — work normally
- Tasks that need sequential, dependent steps — use subagents instead
- Tasks requiring user interaction — workflow agents cannot ask questions

## Script Format

The script must define `async def main()`. The runtime injects:

- `agent(prompt, *, agent="explore", label=None, phase=None, schema=None, isolation=None)` — spawn a subagent; `isolation="worktree"` runs it in a fresh git worktree (isolates file edits for parallel agents). Profiles: `explore` (grep/read), `research` (+web), `reviewer` (+bash), `worker` (full tools incl. MCP — pair with `isolation="worktree"`).
- `parallel(*thunks)` (or `parallel([thunks])`) — run thunks concurrently, results in order; a thunk that raises yields `None` (filter the results)
- `pipeline(items, *stages)` — run each item through all stages with no barrier between stages (item A can be in stage 3 while B is still in stage 1); each stage receives `(prev, item, index)`. One stage acts as a concurrent map.
- `phase(name)` — declare a phase for progress tracking
- `log(msg)` — log a progress message
- `budget` — token budget with `.total` and `.remaining()`
- `workflow(name, args=None)` — run another discovered workflow inline as a sub-step and return its result (shares this run's budget/agents; one level deep only)
- `args` — structured input from the invocation

## Best Practices

1. **Use schemas for structured output** — pass `schema=` to `agent()` for JSON-validated responses
2. **Use `parallel` for independent same-stage work; use `pipeline` for multi-stage per-item flows** where each stage consumes the prior stage's output (e.g. find→verify→synthesize), with no barrier between items' stages
3. **Declare phases with `phase()` for progress tracking**
4. **Guard loops with `budget.remaining()`** — stop when budget is exhausted
5. **Keep scripts focused** — one workflow per task, not a general-purpose program

## Limitations

- Scripts run in a restricted namespace (no `open`, `exec`, `os`, `subprocess`)
- Up to 16 concurrent agents, 1000 total per run
- The workflow runs in the background; the result appears when complete
- Use `/workflows` to check progress or stop a run
