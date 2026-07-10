# Cost and Reliability Roadmap

- Status: implementation in progress
- North star: cost per verified pass
- Companion guardrail: false-done rate

## Objective

Make weaker agents dependable without paying for model calls that deterministic
code can replace. The model should propose actions and make genuinely semantic
decisions; the harness should own policy, spend, task state, retrieval, retries,
and proof of completion.

This roadmap combines reliability and cost work because the same changes improve
both. A typed task brief reduces planning errors and prompt size. Deterministic
verification is cheaper and harder to bluff than another review agent. Bounded
repair preserves an investigation instead of paying to repeat it. Local memory
retrieval removes a background model call from most turns.

Line numbers below describe the repository on 2026-07-09. Symbols are the
authoritative anchors when later upstream syncs move a line.

## Success metrics

Every benchmark run must report these metrics by task, model, model profile, and
harness revision:

- **Verified pass:** all hidden acceptance checks pass and the terminal
  verification receipt still matches the candidate repository state.
- **Cost per verified pass:** total paid model cost across all attempts divided by
  verified passes. Failed attempts, memory, judges, formatters, and verifiers all
  count.
- **Tokens per verified pass:** prompt, completion, reasoning, and cached input
  tokens, split into primary work and harness work.
- **False-done rate:** runs reported as successful whose hidden checks fail or
  whose receipt is missing, stale, or invalid.
- **Unsafe/out-of-scope mutation rate:** runs that violate an immutable policy or
  modify a path outside the task brief.
- **Repair recovery rate:** invalid calls/results recovered within the bounded
  repair budget without restarting the investigation.
- **Auxiliary utilization:** auxiliary model-call results subsequently consumed by
  a decision or delivered output, divided by all auxiliary calls.
- **Human intervention rate, wall time, call count, retry count, and peak paid
  concurrency.**

Initial release gates, to be tightened after the baseline is recorded:

- [ ] False-done rate is below 1% on the core suite and zero on policy/security
  fixtures.
- [ ] Cost per verified pass improves by at least 30% on the weakest supported
  model without reducing pass@1 by more than two percentage points.
- [ ] Harness model spend is at most 20% of total spend by default; optional
  maintenance is at most 5%.
- [ ] At least 99% of paid calls have a purpose, parent scope, token usage, cost,
  outcome, and budget decision recorded.
- [ ] Five-run medians and confidence intervals are reported; a single lucky run
  cannot pass a release gate.

## Current baseline and source anchors

| Area | Current behavior and gap |
|---|---|
| Usage | `AgentLoop._update_stats` records primary/compaction/subagent calls. `UsageRecord` carries backwards-compatible `call_kind` and `result_used`; the local `UsageMeter` handles auxiliary calls. The durable `SpendBroker` adds session/workflow/team/agent/call scopes, but the usage event stream does not yet expose the complete scope and outcome metadata. |
| Auxiliary calls | Memory collaborators and the safety judge reserve through both their smaller host-local meter and the shared session broker with distinct purposes. Optional memory fails open on exhaustion; safety judging falls back to human approval. Narration, MCP sampling, isolated subprocesses, and backend-internal retries remain boundaries. |
| Session limits | `SessionSpendAdapter` reserves before primary, compaction, and in-process task/workflow calls and shares a file-lock-backed parent envelope. Cumulative token caps are opt-in; adaptive prompt admission is calibrated from exact usage by provider/model/request shape, while $10, call, concurrency, retry, and per-call output defaults remain finite. Explicit config and runtime limits constrain admission; an unexpectedly token-dense call can reconcile above the remaining allowance once. Turn middleware remains as a compatibility guard. |
| Workflow budget | Workflow `Budget` still supplies its script-visible token allowance. In-process workflow agents receive child scopes under the session broker, and either local or shared spend exhaustion produces persisted `WorkflowStatus.BLOCKED`. Isolated workflow subprocesses do not yet inherit the parent ledger. |
| Provider concurrency | The process-global provider limiter defaults to four requests in `vibe/core/llm/provider_limiter.py:20-25,52-72`; it is a rate limiter, not a session cost policy. |
| Memory recall | Hybrid recall now scores locally first and asks the LLM only on an ambiguous cutoff. The weighted lexical/IDF selector and bounded query/store cache are at `vibe/core/memory/local_selector.py:57-218`; blocking and prefetch wiring are at `vibe/core/agent_loop_memory.py:152-228,286-377`. Defaults remain per-turn, with two bodies, 4,000 injected body characters, and a 4,000-character index at `vibe/core/config/_settings.py:306-371`. |
| Memory background work | Confident local recall launches no task, while an ambiguous/LLM-only prefetch is auto-consumed even when it finishes after the first poll at `vibe/core/agent_loop_memory.py:286-377`. Extraction, consolidation, and verification remain post-turn tasks at `vibe/core/agent_loop.py:1408-1418`, but now respect their explicit flags at `agent_loop_memory.py:398-403,621-626,839-844`; event-based write signals are still absent. |
| Compaction | Compaction swaps in a compact summary-only system prompt, suppresses tool schemas/tool choice, and reserves under the shared spend envelope with a distinct purpose. |
| Tool manifests | Remote catalogs and a few fork tools can be deferred in `vibe/core/tools/manager.py:229-269`; selected schemas are built at `vibe/core/llm/format.py:45-86`. The active task phase does not select a small builtin manifest, and `AgentLoop._available_tools` explicitly keeps the subset tier-invariant at `vibe/core/agent_loop.py:3437-3446`. |
| Capability scaling | `baseline_tier_for` still uses only context-window size at `vibe/core/baseline_scaling.py:46-58`, which is not a proxy for tool-use reliability. SMALL drops long orchestration prose but retains compact investigation and verification invariants from `vibe/core/_prompt_invariants.py`. |
| Task contracts | `TaskBrief` serializes objective, inputs, path scope, acceptance checks, optional budget/deadline, and manifest identity. The runtime rejects already-expired deadlines, but path scope, checks, per-task budgets, and manifest identity remain schema/prompt metadata rather than host-enforced constraints. Task and team entry points retain versioned legacy-string compatibility; recipes and phase manifests remain open. |
| Task outcomes | `TaskOutcome` has explicit succeeded/failed/blocked/retryable states and evidence fields. Team tasks persist outcomes, atomically requeue retryable work, and unlock dependencies only on success. The task tool preserves structured outcomes through asynchronous delivery; workflows preserve spend exhaustion as `BLOCKED`. |
| Verification | An optional immutable `trusted_verification_recipe` is prebound at AgentLoop creation. After a current verifier PASS, no-argument `verify_work` executes only its exact checks and creates a durable receipt bound to task/contract/config, repository state, check definitions, and full-output hashes. Configured sessions require that receipt; unconfigured sessions retain the current recorded verifier/workflow-pass gate. `land_work` revalidates the candidate and reports the merge commit SHA without persisting a separate landing record. |
| Tool repair | Tool argument parsing preserves bounded raw text and an exact structured diagnostic, then tries conservative fence/object/trailing-comma repair without inventing values. Schema strictness and formatter-call integration remain open. |
| Result repair | Workflow schema failure starts a fresh `AgentLoop` for every attempt at `vibe/core/workflows/runtime.py:1031-1083,1174-1244`, so a formatting error can rebill the whole investigation. |
| Loop detection | `LoopDetectionMiddleware` still detects identical trailing calls. The new repair controller adds canonical semantic progress snapshots, per-failure retry budgets, no-progress/oscillation detection, escalation decisions, and episode metrics; runtime call-site integration remains open. |
| Result cache | Workflow cache identity now includes repository state, effective tool manifest, harness version, model settings, agent profile, prompt, schema, contract, and isolation. Cached write-capable work and provenance/expiry policy still need hardening. |
| Background delivery | Large child results are persisted and clipped to a 4,000-character preview at `vibe/core/tools/_background_delivery.py:8-35`, but every completed child is still injected separately at `vibe/core/agent_loop.py:3308-3341`. |
| Long-lived workers | One `AgentLoop` handles every team task at `vibe/core/programmatic.py:131-151`, so task history and its token cost accumulate across unrelated queue items. |
| Hard policy | `AgentLoop._should_execute_tool` now resolves permission and honors `NEVER` before applying `bypass_tool_permissions` at `vibe/core/agent_loop.py:1912-1939`. Bash blockers at `vibe/core/tools/builtins/bash.py:643-670` and task deny rules at `vibe/core/tools/builtins/task.py:292-303` can no longer be bypassed by auto-approve mode. |
| Evaluation | The offline `evals` package validates versioned artifacts and receipt bindings, aggregates reliability/cost/repair/utilization metrics, reports deterministic confidence intervals, and compares aligned baseline/candidate trials. `scripts/evaluate_harness.py` exits nonzero for gate failure or invalid input. Fixture execution, trusted raw-event ingestion, and paid scheduled trials remain open. |

## Design constraints

- [ ] Deterministic local code is the default for policy, bookkeeping, retrieval,
  diff checks, tests, parsing, and routing.
- [ ] A cheap helper model is called only when deterministic confidence is below
  an explicit threshold; a strong model is an escalation after bounded failure.
- [ ] Budgets are reservations enforced before a call, not warnings calculated
  after the bill arrives.
- [ ] Acceptance criteria originate with the user, lead, recipe, or trusted task
  compiler. A worker cannot author the checks that declare its own work complete.
- [ ] A successful status is structured and evidence-bearing. Free-form prose is
  never interpreted as proof.
- [ ] Every retry receives the smallest useful diagnostic and preserves valid
  prior work.
- [ ] Paid concurrency is intentionally low by default. Parallelism is raised only
  when measurements show that latency value exceeds cache loss and extra spend.
- [ ] Full artifacts stay on disk; model context receives a bounded digest and an
  explicit retrieval pointer.

## Fork-friendly file strategy

Do not split, rename, or relocate upstream-owned files. `agent_loop.py`,
`config/_settings.py`, `llm/format.py`, `middleware.py`, `tools/manager.py`,
`tools/builtins/task.py`, and `tools/builtins/todo.py` exist upstream. Changes to
them must be small hooks into fork-owned siblings.

The memory, usage, workflows, teams, verification-state, launch-workflow, and
land-work modules are fork-added today. They can evolve without creating an
upstream modify/delete conflict, but should still remain stable rather than being
reorganized gratuitously.

| Concern | New fork-owned implementation | Thin existing-file hooks |
|---|---|---|
| Spend | Extend `vibe/core/usage/_meter.py` and `models.py`; add `_broker.py`, `_ledger.py`, `_context.py` | Preflight/reconcile calls around current completion seams; one `SpendConfig` field in `_settings.py` and `vibe_schema.py` |
| Compaction | Extend fork-owned `vibe/core/_compaction_request.py` | Keep the prompt/tool suppression hooks in upstream-owned `agent_loop.py` localized |
| Task protocol | `vibe/core/tasking/models.py`, `_compiler.py`, `_outcomes.py` | Accept `TaskBrief` in workflow/team/task entry points while retaining legacy prompt compatibility |
| Local memory | Extend `vibe/core/memory/local_selector.py`; add `_recall_cache.py`, `_signals.py` only when needed | Selector choice and scheduling hooks stay in `agent_loop_memory.py` |
| Recipes/manifests | `vibe/core/recipes/models.py`, `_runner.py`, `bundled/*.py`; `vibe/core/tools/_task_manifest.py` | Manifest selection hook in `ToolManager.manifest_tools`; keep `launch_workflow` as the advanced escape hatch |
| Verification | Extend `verification_contract.py`; add `_verification_receipt.py`, `_verification_runner.py` | `verification_state.py` stores receipts; `land_work.py` validates one |
| Repair | `vibe/core/repair/models.py`, `_controller.py`, `_progress.py`, `_json.py` | Narrow calls from format resolution, workflows, and middleware |
| Result cache | `vibe/core/result_cache/models.py`, `_store.py`, `_keys.py` | Workflow runtime consults the cache only for declared cache-safe work |
| Evaluation | `evals/runner.py`, `evals/models.py`, `evals/tasks/` | No production dependency; consume programmatic JSON/events and spend records |

Every new package must expose an explicit `__all__`. Task, receipt, repair, and
cache domain models use `ConfigDict(extra="forbid")`; config models follow their
neighboring `BaseSettings`/`SettingsConfigDict` behavior. Config merge fields get
the same per-leaf merge annotations as neighboring settings.

## Phase 0: Baseline, policy, and spend taxonomy

This phase makes later savings measurable and closes a policy bug before weaker
agents receive more autonomy.

### TODO

- [x] Add backwards-compatible call kinds for main, subagent, compaction, memory,
  safety-judge, and narrator work, plus initial result-utilization state.
- [x] Expand that initial taxonomy into `SpendPurpose` values at minimum for
  `primary`, `compaction`,
  `memory_recall`, `memory_extract`, `memory_consolidate`, `memory_verify`,
  `safety_judge`, `narration`, `workflow`, `team`, `repair`, and `verification`.
- [ ] Add stable `call_id`, `parent_call_id`, `session_id`, `run_id`, `agent_id`,
  `task_id`, `purpose`, `model`, and `provider` fields to the usage event stream.
  Preserve backwards loading for current `UsageRecord` JSONL lines.
- [ ] Record whether a call result was consumed, superseded, cancelled before
  dispatch, cancelled after dispatch, rejected by budget, or abandoned.
- [ ] Add a local request estimator that records prompt/schema bytes and estimated
  input/output tokens before dispatch. Keep provider-reported usage authoritative
  after dispatch.
- [ ] Measure the current system prompt, project context, tool schema, memory
  injection, and history contribution separately for each benchmark request.
- [x] Change permission flow so `resolve_permission` always runs and a `NEVER`
  decision is immutable. `bypass_tool_permissions` may bypass user consent for
  `ASK`; it must never bypass policy.
- [ ] Split deterministic policy denial from consent in a new policy result type.
  Run policy before the safety judge so an impossible call never spends judge
  tokens.
- [x] Give compaction a concise summary-only system prompt and suppress its tool
  manifest/tool choice, avoiding the normal coding-agent baseline on that call.
- [ ] Establish a versioned baseline with 20-30 fixture tasks, five trials per
  task, at least one intentionally weak paid model, and one stronger reference
  model.

### Acceptance criteria

- [ ] Auto-approve, chat, isolated worker, and workflow modes all refuse Bash
  blocking-sleep/control-character guardrails and task denylisted profiles.
- [x] Unit coverage proves a denylisted `NEVER` call short-circuits the safety
  judge with `bypass_tool_permissions` both disabled and enabled.
- [ ] A test enumerates every direct `backend.complete` and
  `backend.complete_streaming` production call and either observes a spend event
  or documents a non-paid exception.
- [ ] Existing usage history remains readable and `/status` totals do not change
  for equivalent input records.
- [ ] The baseline report includes all success metrics and stores raw run events,
  terminal diffs, check output, and pricing inputs for reproduction.

## Phase 1: Unified hierarchical spend broker

The broker is the only authority allowed to start a paid model call. Usage
recording remains an append-only observation surface; budget state is a separate,
transactional ledger.

### TODO

- [x] Implement the first process-local `UsageMeter`: atomic projected token/USD
  reservations, exact/estimated reconciliation, token/cost/call limits, call
  kind, result-used state, and append-only usage recording.
- [x] Route all four memory collaborators and the safety judge through that meter.
  They now also reserve under the shared broker with distinct purposes.
- [x] Implement `SpendEnvelope` hierarchy:
  `session -> workflow/team -> agent -> call`.
- [x] Support limits for prompt tokens, completion tokens, total tokens, USD,
  calls, concurrent paid calls, retries, and wall-clock deadline.
- [x] Reserve conservative prompt/completion spend before dispatch and reconcile
  provider usage/cost afterward. Treat missing usage as the reserved estimate and
  expose an `estimated=true` diagnostic.
- [x] Make cumulative token caps opt-in and calibrate adaptive prompt estimates
  from recent exact usage by provider, model, and request shape. Keep strict
  serialized token-bearing request estimation available, preserve
  explicit/runtime admission caps, and migrate only exact legacy generated
  defaults in configs and ledgers.
- [x] Send the admitted completion bound to routed backends when `max_tokens` is
  omitted. The `openai-chatgpt` Codex endpoint rejects that field, so this
  backend remains reservation-and-reconciliation enforced rather than
  provider-capped.
- [ ] Allocate default session capacity as 80% primary work, 15% repair and
  verification, and at most 5% optional maintenance. Permit unused child capacity
  to return to its parent, but never let a child borrow past the parent hard cap.
- [x] Default paid concurrency to one or two per provider/session. Keep the
  existing provider limiter as the outer infrastructure ceiling.
- [ ] Route primary AgentLoop preflight, compaction, narrator, workflow, team,
  isolated subprocess, repair, and verifier calls through the shared broker.
- [x] Route primary, compaction, in-process task/workflow, memory, and
  safety-judge calls through shared session admission.
- [ ] Pass a scoped ledger path and scope IDs to isolated subprocesses. Use
  file-lock-backed atomic reservations so parallel workers cannot overspend a
  shared envelope.
- [ ] Release stale reservations after a bounded lease when a worker dies; record
  the release as an auditable event.
- [ ] Add purpose-specific model policy: local first, configured cheap model for
  optional helpers, primary model for implementation, strong model only after an
  explicit escalation condition.
- [ ] Make optional work fail open when its allocation is exhausted: skip memory
  maintenance or narration without consuming primary capacity. Policy and
  required verification fail closed.
- [ ] Expose live and final snapshots with reserved, spent, remaining, cached,
  estimated, and rejected spend by purpose.

### Acceptance criteria

- [x] Process-local meter tests cover reservation rejection, successful usage,
  and missing-usage reconciliation without double accounting.
- [ ] No paid call can be dispatched without a successful broker reservation.
- [ ] Concurrent workflow/team/memory calls cannot exceed either their child cap
  or the session cap under a race test.
- [ ] Cancellation, provider errors, missing usage, retry, failover, and process
  death all reconcile exactly once.
- [ ] A session hard USD cap is not exceeded by more than one provider billing
  quantum; the documented quantum is derived from the reservation estimate.
- [ ] Memory/narration exhaustion leaves the primary task usable, while required
  verification exhaustion produces `BLOCKED`, never a false success.
- [ ] `UsageRecord` totals, broker totals, workflow totals, and isolated-process
  sentinel totals agree in integration tests.

## Phase 2: Local-first, event-driven memory

Memory must earn its context and API cost. The always-available file store stays;
paid selection becomes an ambiguity fallback rather than the normal path.

### TODO

- [x] Add a deterministic bounded local selector with weighted lexical/IDF scoring
  over id, title, tags, description, and index text, including a cutoff ambiguity
  signal and query/store-fingerprint LRU cache.
- [ ] Benchmark the bounded selector against SQLite FTS5/BM25 as stores grow;
  adopt an incremental SQLite index only when recall or scan cost justifies it.
- [ ] Retrieve top 1-2 entries locally and enforce one shared injection token/char
  budget across index plus bodies.
- [ ] Define confidence using absolute score, score gap, query coverage, and scope.
  The initial score-gap ambiguity fallback is complete; add calibrated coverage,
  scope, and spend-envelope thresholds.
- [x] In hybrid mode, skip the paid selector for a confident local result and call
  it only when the local cutoff is ambiguous.
- [x] Consume an ambiguous selector task through its completion callback even
  when it settles after the first poll; do not discard an already-completed task
  during turn cleanup.
- [ ] Cache recall by normalized query, project identity, store fingerprint,
  retrieval configuration, and reranker model. Persist enough metadata to explain
  why an entry was injected.
- [ ] Eliminate speculative paid prefetch. Do not start a reranker unless its
  result can be consumed in the current turn; if dispatch has started, retain its
  result for a future identical query rather than cancelling and discarding it.
- [ ] Add deterministic write signals for explicit preference, user correction,
  durable decision, stable environment fact, and explicit remember/forget intent.
  Extraction runs at session close or a quiet boundary only when a signal exists.
- [ ] Filter transient task state locally before extraction. Keep the existing
  per-session write cap as a final backstop.
- [ ] Store optional structured assertions and provenance with a memory. Verify
  filesystem/config assertions deterministically; use a model only to extract an
  assertion schema that local code can execute safely.
- [ ] Trigger consolidation by store size/duplicate score and verification by
  stale assertion, not merely by every completed `le-chaton` turn.
- [x] Make effort mode respect explicit memory flags. A quality preset may suggest
  settings, but must not silently force paid extraction, consolidation, or
  verification.
- [x] Lower default recall to two selected entries and 4,000 injected body
  characters; keep background maintenance off unless explicitly enabled.
- [ ] Evaluate per-session recall as the default against per-turn relevance and
  cache behavior before changing it.
- [ ] Emit local-hit, rerank, no-match, cache-hit, injected-bytes, and later-used
  events for evaluation.

### Acceptance criteria

- [x] Confident hybrid queries, local-only mode, and empty/disabled stores perform
  zero paid memory-selector calls in focused tests.
- [ ] Paid selector call rate falls by at least 80% against the Phase 0 workload,
  while relevant-memory recall@2 drops by no more than two percentage points.
- [ ] A late reranker never produces a billed-but-unrecorded result and never
  mutates the current turn after its first request is dispatched.
- [x] A selector result that completes after the first non-blocking poll is applied
  to the next model-facing memory state instead of being silently abandoned.
- [ ] Memory context remains byte-stable for identical query/store inputs, keeping
  the provider prefix cache eligible.
- [ ] Extraction fixtures reject one-shot task state and retain explicit user
  corrections/preferences without a per-turn extractor call.
- [ ] Memory's total cost and result-utilization rate are visible by purpose.

## Phase 3: Structured task briefs, small manifests, and finite orchestration

Weak agents should fill bounded roles, not invent their own protocol or Python
orchestration program.

### TODO

- [ ] Define `TaskBrief` with objective, inputs, allowed paths, denied paths,
  non-goals, acceptance checks, output schema, risk, deadline, spend allocation,
  tool phase, and retry policy.
- [x] Define `TaskOutcome` with `SUCCEEDED`, `FAILED`, `BLOCKED`, and `RETRYABLE`,
  plus evidence, diagnostics, changed paths, receipt ID, and remaining work.
- [ ] Keep lifecycle status (`PENDING`/`IN_PROGRESS`) separate from terminal
  outcome. Migrate teams and workflows with a versioned legacy-description
  adapter.
- [ ] Reject an empty acceptance contract unless a trusted caller creates an
  explicit no-check/trivial waiver with a reason and path scope.
- [ ] Add declarative recipes for `investigate`, `implement_verify`,
  `review_repair`, and `mechanical_edit`. A recipe owns phase transitions,
  manifests, budgets, checks, and escalation policy.
- [ ] Add a typed `launch_recipe` tool. Retain model-authored
  `launch_workflow(script=...)` as an advanced, ASK-gated escape hatch, not the
  default path offered to a weak model.
- [ ] Select 6-10 tools per phase from the task/recipe. Start with:
  `investigate = read/grep/glob/lsp`,
  `implement = read/edit/write_file/targeted bash`, and
  `verify = read/grep/lsp/jailed bash`.
- [ ] Send complete parameter schemas for selected tools. Hide the rest behind
  `tool_search` with concise stubs and deterministic nearest-name suggestions.
- [ ] Make manifest identity part of prompt/result cache keys and telemetry.
- [ ] Replace context-window-only behavior with a measured capability profile:
  tool accuracy, schema adherence, planning depth, correction rate, and context
  window.
- [x] Retain a compact investigation and verification invariant kernel at SMALL
  while omitting the longer orchestration prose.
- [ ] Route mechanical operations to the cheap/grunt model only when the recipe
  supplies all decisions. Escalate to a stronger model after repeated semantic
  failure, not after every formatting error.
- [x] Land the initial finite defaults: two concurrent host/task or workflow
  agents, 32 workflow agents, 500,000 workflow tokens, and 60 isolated turns.
  Shared USD/call/deadline and task-risk profiles remain open.
- [ ] Add shared USD/call/deadline envelopes, task-risk-specific caps, and bounded
  worker idle/task lifetime on top of the initial finite defaults.
- [ ] Reset or replace the long-lived worker `AgentLoop` between queue tasks;
  durable state belongs in `TaskBrief`, `TaskOutcome`, TaskStore, and memory, not
  an ever-growing transcript.
- [ ] Coalesce sibling completions into one host continuation per debounce window.

### Acceptance criteria

- [ ] A worker cannot widen allowed paths, acceptance checks, budget, or tool
  manifest in its response.
- [ ] Every terminal task has an explicit outcome; ordinary exceptions never
  become `None` or `COMPLETED`.
- [ ] The initial request's tool-schema tokens fall by at least 50% on the core
  suite without a statistically significant pass@1 regression.
- [x] SMALL/capability-limited profiles still receive policy, investigation, and
  verification invariants.
- [ ] Recipe runs are replayable from a serialized brief and recipe version.
- [x] No paid workflow starts with an unbounded total budget or unbounded child
  fan-out.

## Phase 4: Deterministic verification receipts

Completion proof should be generated by the harness and tied to the exact work
being delivered.

### TODO

- [x] Parse verifier output into structured verdict plus nonempty command/output
  evidence, require one final verdict, reject contradictory PASS/FAIL evidence,
  and record only a complete successful verifier run.
- [x] Reject model-authored `land_work` attestations and pasted reports. In
  unconfigured sessions, accept only a current workspace-bound recorded pass or
  a locally validated documentation-only `trivial: <reason>` waiver; configured
  sessions require their trusted receipt.
- [x] Bind the current in-memory pass to HEAD, index, working-tree diff, and
  untracked content. Invalidate it on workspace changes, mutating tools, session
  reset, or failed isolated-worktree delivery.
- [x] Define immutable `VerificationReceipt` data with receipt version, task-brief
  hash, recipe version, repository identity, base SHA, candidate HEAD/tree hash,
  dirty-tree state, diff hash, allowed-path result, check evidence, outcome,
  timestamps, and harness version.
- [x] Record each check as argv, cwd, timeout, exit code, bounded stdout/stderr
  excerpts, full-output artifact hash/path, and duration. Never trust a prose
  claim that a command ran.
- [x] Run the prebound trusted acceptance commands and path/diff invariants
  locally, recording required artifact hashes. Configured recipes reject an
  empty check set; their commands, working directories, timeouts, and path scope
  cannot come from model tool arguments.
- [x] Invalidate the receipt on HEAD, index, working-tree, task-brief, contract, or
  relevant configuration change. Either require a clean candidate tree or bind a
  deterministic dirty-tree hash.
- [x] Add receipt references and validation status to `VerificationState` while
  retaining current workspace-bound in-session flags as the unconfigured
  compatibility path. Workspace changes invalidate those flags.
- [ ] Convert the parsed report into a strict persisted receipt schema and bind it
  to trusted execution evidence. Regex recognition alone is no longer an
  authorization mechanism, but the current report still describes model-claimed
  commands rather than harness-executed checks.
- [ ] Invoke one model verifier only for high-risk or semantically ambiguous work.
  Give it the task brief, diff, deterministic results, and targeted artifacts,
  not the entire conversation.
- [x] Make `land_work` require a valid receipt for a configured session's
  candidate branch. Unconfigured sessions accept their current recorded pass or
  a harness-validated documentation-only trivial waiver; arbitrary nonempty
  `verification_note` remains rejected.
- [x] After merge, return the merge SHA and verify that the receipt candidate is
  a parent of the reported merge commit. No separate durable landing record is
  implemented.

### Acceptance criteria

- [x] A PASS becomes invalid after any relevant file edit, commit change, staged
  change, contract edit, or check-command change.
- [x] A bare `VERDICT: PASS` string and an arbitrary nonempty verification note
  cannot authorize landing.
- [x] Check commands come only from the session-prebound trusted recipe; a
  worker response cannot inject a shell command into verification.
- [ ] Deterministic low-risk tasks use zero verifier-model calls.
- [ ] High-risk verification receives bounded context and at most one normal
  verifier attempt plus one targeted retry.
- [x] False-done fixtures cover stale receipts, empty contracts, skipped checks,
  misleading model prose, dirty trees, and post-verification edits.

## Phase 5: Bounded repair and semantic progress

Use one controller shape everywhere:

`attempt -> deterministic check -> structured diagnostic -> targeted retry -> escalation`

### TODO

- [x] Define `FailureDiagnostic` with category, stable fingerprint, exact failing
  field/check, expected/actual, retryability, evidence pointer, and suggested
  minimal next action.
- [x] Preserve malformed JSON text and parser position in `ParsedToolCall`; never
  replace it silently with `{}`.
- [x] Apply conservative local JSON repair first: fence extraction, surrounding
  prose removal, and unambiguous syntax fixes only. Never invent a required value.
- [ ] If local repair fails, use a tiny formatter call containing only raw output,
  schema, and validation errors. It cannot use tools or repeat the task.
- [ ] Reuse the existing agent conversation for semantic/schema correction so the
  investigation and tool results remain in context. Do not construct a fresh
  loop for each output-format attempt.
- [ ] Return exact validation errors, nearest tool/field names, allowed enum
  values, and one minimal corrected-call example.
- [ ] Migrate action strings and new/changed argument models to `Literal`,
  `StrEnum`, discriminated unions, and `extra="forbid"`. For upstream-owned tools,
  stage strictness behind compatibility telemetry before changing defaults.
- [x] Track progress snapshots from repository diff hash, error fingerprint,
  acceptance-check state, files newly read, and tool-effect fingerprint.
- [x] Warn after no-progress repetition, stop after a second bounded strike, and
  escalate only when the failure class is eligible and budget remains.
- [x] Give each failure class its own retry budget. Parse/schema repair should not
  consume the same allowance as test failure or provider transport retry.
- [ ] Feed an exact failed check back to the same worker as a targeted repair
  brief. Preserve successful checks and forbid unrelated edits.
- [ ] Map terminal repair exhaustion to `FAILED` or `BLOCKED`; use `RETRYABLE` only
  with a concrete external condition or remaining deterministic action.

### Acceptance criteria

- [x] Malformed tool calls always surface the original parse/validation cause.
- [ ] A formatting-only schema failure never repeats repository exploration or
  tool execution.
- [ ] Repair call context is bounded independently of parent transcript size.
- [x] Repeated different commands that leave diff/check/error state unchanged are
  detected as no progress.
- [x] Retry count, recovered/not-recovered outcome, added cost, and escalation
  reason are recorded for every repair episode.
- [ ] Fixtures cover malformed JSON, unknown fields, wrong enum, failed test,
  unchanged failing command, oscillating edits, transport retry, and budget
  exhaustion.

## Phase 6: Safe caching, bounded delivery, and context reuse

Provider prompt caching and semantic result caching are different. Preserve the
stable-prefix work already present; add result reuse only where repository and
tool state make it sound.

### TODO

- [x] Persist large background-agent output and inject a bounded 4,000-character
  preview with a retrieval pointer instead of the full response.
- [ ] Keep system prompt, tool manifest, and all non-tail history byte-stable for
  equivalent turns. Extend the existing prompt-cache invariant tests to task
  manifests and memory recall.
- [x] Define a result-cache key from task/normalized query, effective model and
  settings, agent/recipe version, tool-manifest fingerprint, repository HEAD/tree
  or declared input-file hashes, schema/contract hash, and harness version.
- [ ] Cache only declared read-only, deterministic-enough operations by default.
  Never auto-replay a write-capable worker result or stale verification receipt.
- [ ] Persist cache provenance, expiry, dependency fingerprints, usage saved, and
  invalidation reason. A cache hit still emits a zero-cost call/result event.
- [ ] Add local caches for memory recall, deterministic tool discovery, safety
  verdicts with immutable-policy version, formatter repair, and read-only recipe
  stages where their dependencies are complete.
- [ ] Invalidate on changed dependency file, HEAD/tree, tool manifest, task brief,
  model settings, policy version, or schema. Test dirty working trees explicitly.
- [ ] Persist full child/workflow outputs once and inject a structured 1-2k-token
  digest with artifact pointer. Use deterministic extraction; do not add a summary
  model merely to save context.
- [ ] Batch sibling completions into one injection and one host continuation.
  Include per-child outcome/evidence pointers without concatenating full prose.
- [ ] Feed repair/verifier agents the brief, diff, failed checks, and artifact
  pointers rather than cloning the host history.
- [ ] Surface provider cache hit ratio separately from semantic result-cache hit
  rate and tokens avoided.

### Acceptance criteria

- [ ] Mutation fixtures produce zero stale cache hits across commit, dirty-tree,
  manifest, model, policy, and schema changes.
- [ ] Cache-disabled behavior remains equivalent and cache corruption fails to a
  normal execution, not a task failure.
- [ ] Background fan-out causes at most one automatic continuation per debounce
  window.
- [ ] Full results remain recoverable from their pointer after context shaping,
  session resume, and process restart.
- [ ] Cache metrics report lookup, hit, validated hit, invalidation, tokens saved,
  cost saved, and false-hit audit result.

## Phase 7: Observability, evaluation, and rollout

The eval harness is the product test for agent behavior. Unit tests prove
components; fixture repositories prove that the complete system helps weaker
models without hiding failures.

Compare two trusted offline datasets with:

```bash
uv run scripts/evaluate_harness.py \
  --baseline baseline.json \
  --candidate candidate.json \
  --output comparison.json \
  --release-gate
```

Exit status is `0` for a passing report, `1` for a gate failure, and `2` for
invalid input or I/O failure.

### TODO

- [ ] Extend usage/spend events and OTEL spans with purpose, scope hierarchy,
  budget reservation, outcome, cache status, repair episode, receipt ID, and
  result-utilization state. Current span seams are `vibe/core/tracing.py:346-441`
  for agent/chat usage and `:513-587` for tools.
- [ ] Replace the binary harness/user summary at
  `vibe/core/usage/_aggregator.py:203-243` with purpose and task breakdowns while
  retaining the old aggregate for compatibility.
- [ ] Add operator views for live budget, top auxiliary consumers, cancelled paid
  work, unused results, repair cost, cache savings, and cost per verified pass.
- [ ] Build hermetic fixture repositories covering narrow fixes, cross-file
  changes, unfamiliar code, malformed tool calls, failing tests, policy attacks,
  stale cache, memory relevance, and explicit blockers.
- [ ] Keep hidden acceptance tests outside the worker-visible repository. Evaluate
  terminal filesystem state and receipt validity, not response wording.
- [ ] Run each task five times per target model/profile. Record pass@1, pass@3,
  false done, unsafe mutation, repair recovery, interventions, tokens, dollars,
  duration, call graph, and auxiliary utilization.
- [ ] Add ablations for broker-only, local memory, task manifests, receipts,
  repair, cache, and the combined harness. Savings must be attributable.
- [ ] Add adversarial verifier tests where implementation-authored tests are
  circular or mock-only and hidden checks expose the defect.
- [ ] Add spend-accounting invariant tests: sum of leaf calls equals each parent
  envelope and the session total; no orphan call IDs; no negative reservations.
- [x] Version task briefs, recipes, capability profiles, pricing tables, policy,
  and eval datasets in every result.
- [ ] Roll out behind independent flags, then enable in order: observability,
  immutable policy, broker, local recall, finite manifests/recipes, receipts,
  repair, semantic cache.
- [ ] Keep a fast rollback for each behavior change. Observability and immutable
  policy are not rollback candidates once validated.

### Acceptance criteria

- [x] A single command produces a machine-readable comparison report and exits
  nonzero when a release gate regresses.
- [x] Runs are reproducible from model/provider identifiers, config snapshot,
  repository fixture hash, task/recipe version, random seed where supported, and
  raw event artifacts.
- [ ] CI runs deterministic unit/integration fixtures; a scheduled or explicitly
  approved job runs paid-model trials with a hard broker cap.
- [ ] The weakest supported model meets the false-done and policy gates before any
  cost optimization is declared successful.
- [ ] A stronger model used once is compared against a weak model plus repeated
  repair; routing chooses the lower measured cost per verified pass, not the
  cheaper per-token sticker price.

## Recommended implementation order

1. Finish Phase 0 call-graph attribution and route every remaining paid call
   through Phase 1's broker, including provider retries and isolated workers.
2. Complete Phase 3 recipes, phase manifests, workflow outcomes, and child
   fair-share allocations on top of the structured task protocol.
3. Finish Phase 2 event-driven memory writes and calibrated retrieval; local-first
   recall, explicit maintenance flags, and late-result consumption are complete.
4. Wire Phase 5's repair controller into workflow/schema/check failures without
   restarting successful investigation work.
5. Complete Phase 6 cache safety, provenance, and batched background delivery;
   repository/model/manifest/harness cache identity is already bound.
6. Add Phase 7 hermetic fixture execution and trusted raw-event ingestion, then
   run hard-capped paid-model A/B trials before default-on rollout.

The receipt core moves ahead of model verification because false success is the
largest trust risk. Remaining memory lifecycle work stays behind the broker so
reranker and maintenance spend is measured rather than inferred.

## Definition of done

- [ ] All production paid model calls pass through the broker and appear in the
  scope tree.
- [ ] Weak-model tasks use trusted briefs, finite recipes, small manifests,
  bounded retries, and explicit terminal outcomes.
- [ ] Memory is local-first and paid maintenance is event-driven and optional.
- [ ] A successful result carries a current, machine-validated receipt; stale or
  textual attestations cannot land work.
- [ ] Result reuse is dependency-keyed and has zero stale hits in the adversarial
  suite.
- [ ] Benchmark evidence demonstrates lower cost per verified pass without a
  reliability, safety, or human-intervention regression.
- [ ] New behavior is documented in README/operator config and the builtin Vibe
  skill when implementation lands.
- [ ] `uv run ruff check --fix .`, `uv run ruff format .`, `uv run pyright`, the
  full test suite, upstream-divergence guard, and paid eval release gates pass.
