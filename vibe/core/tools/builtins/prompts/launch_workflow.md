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

## Passing the script

Pass the script's **source text** in the `script` argument — the full Python
source, inline. Do **not** pass a file path; the tool does not read files. If you
wrote the script to a scratchpad file with `write_file`, paste its contents into
`script` (or `read` it and pass the body). The tool validates the source via AST
before it runs, so a clear validation error beats a confusing "no main()" later.

## Script Format

The script must define `async def main()`. The runtime injects:

- `agent(prompt, *, agent="explore", label=None, phase=None, schema=None, isolation=None)` — spawn a subagent; `isolation="worktree"` runs it in a fresh git worktree (isolates file edits for parallel agents). Profiles: `explore` (grep/read), `research` (+web), `reviewer` (+bash), `debugger` (+bash; systematic root-cause analysis of a failure or flaky test), `planner` (grep/read; returns a phased, code-grounded plan), `security` (+bash; defensive vuln audit with severity-ranked findings), `editor` (read/grep/write/edit, no bash/MCP; surgical edits — **requires** `isolation="worktree"`), `worker` (full tools incl. MCP — **requires** `isolation="worktree"`). `schema=` validates the agent's JSON output and **strips unknown keys by default** (`strip_unknown=True`), so an extra field in a reply degrades gracefully instead of discarding the agent's work.
- `parallel(*items, max_concurrency=None)` (or `parallel([items])`) — run items concurrently, results in order; an item that raises yields `None` (filter the results). Each item may be a **coroutine** — `parallel(agent("a"), agent("b"))` — or a zero-arg thunk — `parallel(lambda: agent("a"))`; both work (Python coroutines are lazy, so the bare form bounds concurrency identically). Pass `max_concurrency=N` (e.g. `3`) to cap in-flight items — use this instead of hand-rolling chunked waves when a provider limits concurrency.
- `pipeline(items, *stages, max_concurrency=None)` — run each item through all stages with no barrier between stages (item A can be in stage 3 while B is still in stage 1); each stage receives `(prev, item, index)`. One stage acts as a concurrent map. `max_concurrency=N` caps in-flight items.
- `phase(name)` — declare a phase for progress tracking. Works bare (`phase("x")`) or awaited (`await phase("x")`) — both are safe.
- `log(msg)` — log a progress message. Works bare or awaited, like `phase()`.
- `budget` — token budget with `.total` and `.remaining()`
- `workflow(name, args=None)` — run another discovered workflow inline as a sub-step and return its result (shares this run's budget/agents; one level deep only)
- `post_message(channel, message)` — post to a named channel on this run's shared board (visible to all agents/stages in the same run via `fetch_messages`)
- `fetch_messages(channel)` — return all messages posted to a channel so far (a copy)
- `flatten(items)` — flatten one level of nested lists (strings/dicts/bytes treated as atoms): `flatten([[1,2],[3]]) == [1,2,3]`
- `dedup_by(items, key)` — drop duplicates, keeping first occurrence; `key` maps each item to a hashable identity (e.g. `lambda f: f"{f['file']}:{f['line']}"`)
- `merge_by(items, key, merge)` — group by `key` and fold each group via `merge(acc, item)`; use to union findings, sum counts, or pick the best per group
- `args` — structured input from the invocation

You do **not** need to (and cannot) `import asyncio` — `agent`/`parallel`/`pipeline`
are already async and injected; call them and `await` the result.

## Starter template

Copy this and fill it in — it has the correct shape (no imports needed, schema,
phases, coroutine fan-out, filtered results, structured return):

```python
# find -> verify -> synthesize. `args` is the invocation input.
SCHEMA = {"type": "object",
          "properties": {"findings": {"type": "array"}}}

async def main():
    phase("find")
    found = await parallel(
        agent("TODO: finder prompt for lens A", schema=SCHEMA),
        agent("TODO: finder prompt for lens B", schema=SCHEMA),
    )
    items = [f for r in found if r for f in r.get("findings", [])]
    if not items:
        return {"summary": "nothing found", "items": []}

    phase("verify")
    # pipeline STAGE must be callable-of-item: use `lambda x:` (or a def), not a
    # bare agent(...). parallel above takes coroutines directly; pipeline doesn't.
    verified = await pipeline(items, lambda f: agent(
        f"TODO: adversarially verify this finding: {f}", schema=SCHEMA))

    phase("synthesize")
    report = await agent(f"TODO: synthesize {json.dumps(verified)}")
    return {"summary": report, "items": verified}
```

## Best Practices

1. **Use schemas for structured output** — pass `schema=` to `agent()` for JSON-validated responses; unknown keys are stripped, not fatal
2. **Use `parallel` for independent same-stage work; use `pipeline` for multi-stage per-item flows** where each stage consumes the prior stage's output (e.g. find→verify→synthesize), with no barrier between items' stages
3. **Cap concurrency with `max_concurrency=`** — pass it to `parallel`/`pipeline` instead of hand-rolling chunked waves, especially when a provider allows only 1-3 concurrent agents
4. **Declare phases with `phase()` for progress tracking**
5. **Guard loops with `budget.remaining()`** — stop when budget is exhausted
6. **Keep scripts focused** — one workflow per task, not a general-purpose program

## Safety boundary

`launch_workflow` is ASK-gated, so each launch is reviewed by the safety judge
(if configured) with a **workflow-aware** prompt: it judges the script's planned
surface — which agent profiles spawn (read-only vs full-tool `worker`), fan-out
across `parallel`/`pipeline`, and any destructive logic in the script itself —
not the Python syntax. If the judge defers, its reason reaches your launch
approval prompt so you know why.

In-process subagents (`explore`/`research`/`reviewer`/`editor`) consult the
judge per tool call like any agent, and any deferral is surfaced to the host
for approval. Isolated `worker`/`editor` agents get a **second judge pass at
spawn**: each worker's prompt is judged before its subprocess starts, and a
deferral is routed to your approval with the judge's reason — so even though
the worker runs auto-approved inside its worktree, its planned task is gated.
A worker you deny is recorded as failed; the run continues with the others.

## Sandbox restrictions

Scripts run in a restricted in-process namespace. The validator runs before
execution and rejects the script with a precise error if it breaks a rule. The
non-obvious traps (these are the ones that cost runs in practice):

- **A fixed set of modules is pre-bound — no `import` needed.** `json`, `re`,
  `math`, `statistics`, `collections`, `itertools`, `functools`, `datetime`,
  `decimal`, `copy`, `hashlib`, `base64`, `textwrap`, `unicodedata` are already in
  scope — just use `json.dumps(...)`. (You may still `import` them; it's a no-op.)
  Nothing else is importable — no `asyncio` (you don't need it — `agent`/
  `parallel`/`pipeline` are injected and awaitable), and no `os`, `sys`,
  `subprocess`, `pathlib`, `io`, `requests`, etc.
- **`str.format()` and `str.format_map()` are forbidden** (the format
  mini-language can traverse attributes/dunders from inside a string literal, an
  escape vector). Template with **f-strings** or **`%` formatting** instead.
  `"...".format(...)` is blocked both as a direct call and via aliasing.
- **No dunder access** (`obj.__class__`, `__globals__`, `__dict__`, `__mro__`,
  `__subclasses__`, …), no dunder dict keys, and no `getattr`/`setattr`/`delattr`/
  `globals`/`locals`/`vars`/`eval`/`exec`/`compile`/`open`/`input`/`__import__`.
- The builtins namespace is safelisted (no `open`, `exec`, `__import__`).
- **Correctness, not just safety, is checked pre-flight.** Beyond the security
  rules above, the validator rejects **undefined names** (a name used but never
  bound, injected, a pre-bound module, or a builtin) and a **coroutine used as a
  `pipeline` stage** (`pipeline(items, agent(...))` — a stage runs per item, so
  use `lambda x: agent(...)`). Note `parallel(agent(...))` is fine. These caught
  classes used to crash at exec time; now they fail at launch with the fix.

## Getting the result back

`launch_workflow` returns only `{run_id, launched, delivery}` — the run is
background and fire-and-forget from this tool. The script's `return_value` and
per-agent outputs are auto-delivered as a message on completion, but that
delivery is best-effort (capped at ~16KB, dropped if the host turn already
ended). **Re-read the result any time** with `workflow_results(run_id=...)`,
which returns the structured `return_value` plus per-agent responses/errors/
`schema_errors`. For finished runs the return value is also persisted across
sessions.

## Limitations

- Up to 16 concurrent agents, 1000 total per run (lower both with `max_concurrency=`)
- The workflow runs in the background; the result appears when complete
- Use `/workflows` to check progress or stop a run. From a model turn, query
  live progress (per-run agents, phases, in-flight agent token totals, budget)
  with the `workflow_status` tool instead of waiting for completion. Stop a
  runaway or misbehaving run with the `workflow_stop` tool
  (`run_id` for one run, or `all` for every active run).
