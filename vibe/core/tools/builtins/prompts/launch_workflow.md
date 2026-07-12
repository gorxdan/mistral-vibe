Use `launch_workflow` to run a workflow script that orchestrates parallel agents in the background.

## Local discovery comes first

Don't launch a workflow as the first step for an unfamiliar repo. First `glob` to
map packages, entry points, and tests; use `lsp` when available (otherwise
`grep`) to locate central symbols, references, and call paths; then read enough code to define independent
questions. Launch only if that reconnaissance shows real parallel work. A broad
request ("analyze this repo") or a high file count is not sufficient by itself.

## One question per agent — fan out for breadth

Each agent runs in its own context window, so breadth comes from launching more
agents, not from bigger prompts. A fat brief — "analyze the whole architecture" or
six investigation areas in one prompt — forces shallow coverage of every area and
is the single most common authoring failure. Split it: one mechanism, one area,
one repository per agent, then `parallel(*[agent(...) for x in areas])`. Name
exact files and symbols in each prompt. If a brief answers "and also", it is two
agents.

## When to Use This Tool

- **Multi-agent tasks**: reconnaissance reveals 3+ independent questions or separable implementation areas
- **Adversarial verification**: findings should be cross-checked by independent skeptics
- **Dynamic loops**: work loops until a condition is met (dry rounds, budget exhaustion)
- **Parallel exploration**: known directories/angles can be investigated independently

## When NOT to Use

- Initial repo discovery or one coherent architecture trace — use `glob`, targeted `read`, and `lsp` when available (otherwise `grep`)
- Single-file edits or quick questions — work normally
- Sequential, dependent steps — use subagents instead
- Tasks requiring user interaction — workflow agents cannot ask questions

## Passing the script

Pass the script's **source text** inline in `script` — full Python source, not a
file path (the tool does not read files). If you wrote it to a scratchpad with
`write_file`, paste the contents (or `read` it and pass the body). The tool
AST-validates the source before running, so a clear validation error beats a
confusing "no main()" later.

## Script Format

The script must define `async def main()`. The runtime injects:

- `agent(prompt, *, agent="explore", label=None, phase=None, schema=None, isolation=None)` — spawn a subagent; `isolation="worktree"` runs it in a fresh git worktree (isolates file edits for parallel agents). Profiles: `explore` (grep/read), `research` (+web), `reviewer` (+bash), `debugger` (+bash; systematic root-cause analysis of a failure or flaky test), `planner` (grep/read; returns a phased, code-grounded plan), `security` (+bash; defensive vuln audit with severity-ranked findings), `editor` (read/grep/write/edit, no bash/MCP; surgical edits — **requires** `isolation="worktree"`), `grunt` (full tools like `worker`; bulk/mechanical work on a cheap model via `grunt_model` — **requires** `isolation="worktree"`), `worker` (full tools incl. MCP — **requires** `isolation="worktree"`). `schema=` validates the agent's JSON output and **strips unknown keys by default** (`strip_unknown=True`), so an extra field degrades gracefully. If output can't be validated after retries the result is a **falsy** `SchemaValidationFailure` (a `dict` subclass): filter with the canonical `[r for r in results if r]` (NOT `isinstance(r, dict)` — that would wrongly include it), `r.get(k, default)` is safe, `json.dumps(results)` won't crash, and `isinstance(r, SchemaValidationFailure)` + `r.schema_errors` give the detail — one bad agent degrades the batch instead of crashing the run.
- `parallel(*items, max_concurrency=None)` (or `parallel([items])`) — run items concurrently, results in order; an item that raises yields `None` (filter the results). Each item may be a **coroutine** (`parallel(agent("a"), agent("b"))`) or a zero-arg thunk (`parallel(lambda: agent("a"))`); both work and bound concurrency identically. `max_concurrency=N` caps in-flight items when a provider limits concurrency.
- `pipeline(items, *stages, max_concurrency=None)` — run each item through all stages with no barrier between stages (item A can be in stage 3 while B is still in stage 1); each stage receives `(prev, item, index)` and acts as a concurrent map. `max_concurrency=N` caps in-flight items.
- `phase(name)` — declare a phase for progress tracking. Works bare (`phase("x")`) or awaited.
- `log(msg)` — log a progress message. Works bare or awaited, like `phase()`.
- `budget` — token budget with `.total` and `.remaining()`.
- `workflow(name, args=None)` — run another discovered workflow inline as a sub-step, returning its result (shares this run's budget/agents; one level deep only).
- `post_message(channel, message)` — post to a named channel on this run's shared board (visible to all agents/stages in the same run via `fetch_messages`).
- `fetch_messages(channel)` — return all messages posted to a channel so far (a copy).
- `flatten(items)` — flatten one level of nested lists (strings/dicts/bytes are atoms): `flatten([[1,2],[3]]) == [1,2,3]`.
- `dedup_by(items, key)` — drop duplicates, keeping first; `key` maps each item to a hashable identity (e.g. `lambda f: f"{f['file']}:{f['line']}"`).
- `merge_by(items, key, merge)` — group by `key` and fold each group via `merge(acc, item)`; use to union findings, sum counts, or pick the best per group.
- `args` — structured input from the invocation.

You do **not** need to (and cannot) `import asyncio` — `agent`/`parallel`/`pipeline`
are already async and injected; call them and `await` the result.

## Starter template

Copy this and fill it in — correct shape (no imports, schema, phases, coroutine
fan-out, filtered results, structured return):

```python
# find -> verify -> synthesize. `args` is the invocation input.
SCHEMA = {"type": "object",
          "properties": {"findings": {"type": "array"}}}
LENSES = ["correctness", "security", "concurrency"]  # fan out over a list

async def main():
    phase("find")
    # Fan out with a comprehension of COROUTINES — no lambda. The loop var binds
    # correctly per item. Do NOT write `lambda: agent(lens...)` over a loop: that
    # late-binds and every agent collapses to the LAST lens (silent, wrong).
    found = await parallel(*[
        agent(f"Review through the {lens} lens. TODO: details.",
              label=lens, schema=SCHEMA)
        for lens in LENSES
    ])
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

1. **Schemas for structured output** — pass `schema=` to `agent()`; unknown keys are stripped, not fatal
2. **`parallel` for independent same-stage work; `pipeline` for multi-stage per-item flows** (find→verify→synthesize), no barrier between items' stages. Fan out with `parallel(*[agent(...) for x in items])` (a comprehension of coroutines) — never `lambda: agent(x...)` over a loop var; the validator flags it
3. **Cap concurrency with `max_concurrency=`** on `parallel`/`pipeline` instead of hand-rolling chunked waves, especially when a provider allows only 1-3 concurrent agents
4. **Declare phases with `phase()`** for progress tracking
5. **Guard loops with `budget.remaining()`** — stop when budget is exhausted
6. **Keep scripts focused** — one workflow per task, not a general-purpose program

## Concurrency & rate limits

Up to __MAX_CONCURRENT_AGENTS__ agents run concurrently per workflow (the
runtime's global cap). Some providers throttle at 1-3 concurrent requests, and
retry is per-request and uncoordinated across agents, so a saturated provider
can fail several agents at once with `Retries exhausted`. Cap concurrency with
`max_concurrency=` on `parallel`/`pipeline` (Best Practice 3) instead of
hand-rolling chunked waves.

## Recovery (agent died of `Retries exhausted`)

Do not re-launch the same fan-out. Re-run that phase with `max_concurrency=1`,
or serialize via `pipeline`, or `schedule` a retry after the provider's
`Retry-After` (honored up to 60s).

## Safety boundary

`launch_workflow` is ASK-gated: each launch is reviewed by the safety judge (if
configured) with a **workflow-aware** prompt — it judges the script's planned
surface (which agent profiles spawn, read-only vs full-tool `worker`/`grunt`; fan-out
across `parallel`/`pipeline`; any destructive logic), not the Python syntax. A
deferral's reason reaches your launch approval prompt.

In-process subagents (`explore`/`research`/`reviewer`/`editor`) consult the judge
per tool call, and any deferral is surfaced to the host for approval. Isolated
`worker`/`editor`/`grunt` agents get a **second judge pass at spawn**: each one's
prompt is judged before its subprocess starts, and a deferral routes to your
approval with the judge's reason — so even though the isolated agent runs
auto-approved inside its worktree, its planned task is gated. An agent you deny
is recorded as **failed**; the run continues with the others.

## Sandbox restrictions

Scripts run in a restricted in-process namespace. The validator runs before
execution and rejects the script with a precise error if it breaks a rule. The
non-obvious traps (the ones that cost runs in practice):

- **A fixed set of modules is pre-bound — no `import` needed.** `json`, `re`,
  `math`, `statistics`, `collections`, `itertools`, `functools`, `datetime`,
  `decimal`, `copy`, `hashlib`, `base64`, `textwrap`, `unicodedata` are already in
  scope — just use `json.dumps(...)`. (You may still `import` them; it's a no-op.)
  Nothing else is importable — no `asyncio`, and no `os`, `sys`, `subprocess`,
  `pathlib`, `io`, `requests`, etc.
- **`str.format()` and `str.format_map()` are forbidden** (the format
  mini-language can traverse attributes/dunders from inside a string literal, an
  escape vector). Template with **f-strings** or **`%` formatting** instead;
  `"...".format(...)` is blocked both directly and via aliasing.
- **No dunder access** (`obj.__class__`, `__globals__`, `__dict__`, `__mro__`,
  `__subclasses__`, …), no dunder dict keys, and no `getattr`/`setattr`/`delattr`/
  `globals`/`locals`/`vars`/`eval`/`exec`/`compile`/`open`/`input`/`__import__`.
  The builtins namespace is safelisted.
- **Correctness, not just safety, is checked pre-flight.** The validator also
  rejects **undefined names** (a name used but never bound, injected, a pre-bound
  module, or a builtin) and a **coroutine used as a `pipeline` stage**
  (`pipeline(items, agent(...))` — a stage runs per item, so use
  `lambda x: agent(...)`). Note `parallel(agent(...))` is fine. These used to
  crash at exec time; now they fail at launch with the fix.

## Getting the result back

`launch_workflow` returns only `{run_id, launched, delivery}` — the run is
background and fire-and-forget. **Completion is auto-delivered**: when the run
finishes, its `return_value` and per-agent outputs are pushed into your context
as a user message — do not poll. End your turn (or continue other work); the
result arrives on its own.

Auto-delivery is best-effort (capped at ~16KB, dropped if the host turn already
ended). If missed or truncated, pull it on demand with
`workflow_results(run_id=...)`, which returns the structured `return_value` plus
per-agent responses/errors/`schema_errors`. For finished runs the return value is
also persisted across sessions.

## Don't poll — use a timer

**Never call `workflow_status` in a loop waiting for a run to finish.** That burns
turns for nothing; completion is delivered automatically. Two correct patterns:

- **Default — end your turn.** Launch, report the `run_id`, and stop. The result is injected when the run completes; resume from there.
- **Long run you want to revisit — arm a `schedule` timer.** `schedule create interval=2m prompt="check on workflow wf-1"` re-prompts you once (or recurring) without blocking. Don't call `workflow_status` repeatedly across turns to watch it.

`workflow_status` is a *diagnostic* tool — call it **once** when you suspect a run
is stuck or runaway (before `workflow_stop`), not as routine progress checks.

## Monitoring (TUI)

`/workflows` — x (stop), p (pause/resume), s (save script as `/<name>`
command), o (view script), Enter (drill into run/agent). In-flight agents show
live token totals. This is for the human watching the terminal, not a prompt
for you to poll.

## Limitations

- Up to __MAX_CONCURRENT_AGENTS__ concurrent agents, __MAX_TOTAL_AGENTS__ total per run (lower both with `max_concurrency=`)
- The workflow runs in the background; the result is auto-delivered on completion (see above) — do not poll.
- `/workflows` (TUI) lets the human watch progress or stop a run. From a model turn, stop a runaway or misbehaving run with the `workflow_stop` tool (`run_id` for one run, or `all` for every active run).
