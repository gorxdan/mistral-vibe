Use `launch_workflow` to run a workflow script that orchestrates parallel agents in the background.

## Local discovery comes first

Don't launch a workflow as the first step for an unfamiliar repo. First `glob` to
map packages, entry points, and tests; use `lsp` when available (otherwise
`grep`) to locate central symbols, references, and call paths; then read enough code to define independent
questions. Launch only if that reconnaissance shows real parallel work. A broad
request ("analyze this repo") or a high file count is not sufficient by itself.

## Start with two bounded evidence lanes

Give each agent one non-overlapping question and start with at most two evidence
lanes. A fat brief — "analyze the whole architecture" or six investigation areas
in one prompt — forces shallow coverage. Each prompt must state one exact
question, exact paths or symbols, and the required evidence (for example,
file:line citations, named tests, or a concrete counterexample). Add another
bounded strategy only after returned evidence identifies a concrete gap. Never
re-launch a duplicate broad audit; breadth comes from distinct evidence, not raw
agent count or bigger prompts.

## When to Use This Tool

- **Multi-agent tasks**: reconnaissance reveals two independent questions or separable implementation areas
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

- `agent(prompt, *, agent="explore", model=None, label=None, phase=None, schema=None, budget_estimate=None, isolation=None, strip_unknown=True, contract=None, citations=None, then=None)` — spawn a subagent; `isolation="worktree"` runs it in a fresh git worktree. Profiles: `explore`, `research`, `reviewer`, `debugger`, `planner`, `security`, `verifier`, `editor`, `grunt`, and `worker`; write-capable profiles require worktree isolation. `schema=` validates JSON output and strips unknown keys by default. Exhausted schema validation returns a falsy, JSON-serializable mapping; filter by truthiness and inspect `r.get("schema_errors", [])` for details. `contract=` validates isolated code artifacts; `citations=` verifies returned file/line/snippet evidence; `then="verifier"` freezes and verifies an isolated candidate before delivery.
- `parallel(*items, max_concurrency=None)` (or `parallel([items])`) — run coroutine or zero-argument callable items concurrently and return results in order. In an ordinary workflow, an item exception yields `None`; a strategy-bound expected-lane workflow re-raises it. Hard budget/agent/spend ceilings always propagate.
- `pipeline(items, *stages, max_concurrency=None)` — run each item through all stages with no cross-item barrier; each stage receives `(prev, item, index)`. Ordinary stage exceptions drop that item to `None`, while expected-lane strategy failures and hard ceilings propagate.
- `phase(name)` — declare a phase for progress tracking. Works bare (`phase("x")`) or awaited.
- `log(msg)` — log a progress message. Works bare or awaited, like `phase()`.
- `budget` — token budget with `.total` and `.remaining()`.
- `workflow(name, args=None)` — run another discovered workflow inline as a sub-step, returning its result (shares this run's budget/agents; one level deep only).
- `recipe(name, *, items=None, find_agent="explore", verify_agent="reviewer", synth=None, max_concurrency=None)` — run `find_verify` or `find_verify_synth`; each item expands into finder and verifier agent calls.
- `post_message(channel, message)` — post to a named channel on this run's shared board (visible to all agents/stages in the same run via `fetch_messages`).
- `fetch_messages(channel)` — return all messages posted to a channel so far (a copy).
- `flatten(items)` — flatten one level of nested lists (strings/dicts/bytes are atoms): `flatten([[1,2],[3]]) == [1,2,3]`.
- `dedup_by(items, key)` — drop duplicates, keeping first; `key` maps each item to a hashable identity (e.g. `lambda f: f"{f['file']}:{f['line']}"`).
- `merge_by(items, key, merge)` — group by `key` and fold each group via `merge(acc, item)`; use to union findings, sum counts, or pick the best per group.
- `team_task(description, dependencies=None)` — enqueue a task on the active process team and return its task ID, or `None` when no team is active.
- `args` — structured input from the invocation.

You do **not** need to (and cannot) `import asyncio` — `agent`/`parallel`/`pipeline`
are already async and injected; call them and `await` the result.

## Starter template

Copy this and replace every `REPLACE_...` token before launch. In Le Chaton mode,
copy the two lane IDs exactly from the accepted `work_strategy` receipt. In a
normal session, use two stable, unique literal labels for this bounded batch.
Each prompt must contain its exact question, paths or symbols, and evidence
contract. The bounded shape uses no imports, two literal lane labels, a schema,
a phase, and a structured return:

```python
SCHEMA = {"type": "object",
          "properties": {"findings": {"type": "array"}}}

async def main():
    phase("evidence")
    results = await parallel(
        agent(
            "QUESTION: REPLACE_WITH_ONE_EXACT_QUESTION_FOR_LANE_1\n"
            "SCOPE: REPLACE_WITH_EXACT_PATHS_AND_SYMBOLS_FOR_LANE_1\n"
            "EVIDENCE: REPLACE_WITH_FILE_LINE_TEST_OR_COUNTEREXAMPLE_REQUIREMENTS",
            label="REPLACE_WITH_EXACT_RECEIPT_LANE_ID_1",
            schema=SCHEMA,
        ),
        agent(
            "QUESTION: REPLACE_WITH_ONE_EXACT_QUESTION_FOR_LANE_2\n"
            "SCOPE: REPLACE_WITH_EXACT_PATHS_AND_SYMBOLS_FOR_LANE_2\n"
            "EVIDENCE: REPLACE_WITH_FILE_LINE_TEST_OR_COUNTEREXAMPLE_REQUIREMENTS",
            label="REPLACE_WITH_EXACT_RECEIPT_LANE_ID_2",
            schema=SCHEMA,
        ),
        max_concurrency=2,
    )
    return {"items": [result for result in results if result]}
```

## Best Practices

1. **Schemas for structured output** — pass `schema=` to `agent()`; unknown keys are stripped, not fatal
2. **`parallel` for independent same-stage work; `pipeline` for multi-stage per-item flows.** A strategy-bound workflow uses exactly one literal-labeled `agent()` call per declared lane; pipeline agent stages require a singleton seed
3. **Start with at most two evidence lanes** and cap concurrency with `max_concurrency=` instead of hand-rolling chunked waves; expand only for a concrete evidence gap
4. **Declare phases with `phase()`** for progress tracking
5. **Guard loops with `budget.remaining()`** — stop when budget is exhausted
6. **Keep scripts focused** — one workflow per task, not a general-purpose program

In a strategy-bound workflow, `workflow()` and `recipe()` do not create extra
capacity. Every downstream agent counts against the same two-lane cap and must
map to an exact lane in the active receipt. If that mapping cannot be proven
before launch, do not call the nested workflow or recipe; use the receipt-bound
literal `agent()` calls directly.

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

Non-isolated read-only subagents consult the judge per tool call, and any
deferral is surfaced to the host for approval. Isolated
`worker`/`editor`/`grunt` agents get a **second judge pass at spawn**: each one's
prompt is judged before its subprocess starts, and a deferral routes to your
approval with the judge's reason — so even though the isolated agent runs
auto-approved inside its worktree, its planned task is gated. An agent you deny
is recorded as **failed**. Ordinary workflows continue with the other items;
strategy-bound expected-lane failures propagate and stop that route.

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
