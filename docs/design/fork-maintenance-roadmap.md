# Fork Maintenance and Quality Roadmap

Status: Draft execution plan

This roadmap turns the fork-quality audit into a sequence of small,
independently reversible iterations. Its purpose is to reduce upstream merge
cost, oversized fork-owned modules, duplicated configuration, and accumulated
test debt without losing product behavior, harness reliability, or the
performance improvements already delivered by the fork.

The governing rule is separation of concerns at the change level:

- Mechanical movement does not change behavior.
- Compatibility changes do not share a change set with structural movement.
- Performance optimization does not share a change set with either one.

If an iteration cannot prove its preservation contract, it does not land.

## Scope

### In scope

- Establishing a reproducible functional, message, performance, cost, and fork
  baseline from a clean commit.
- Correcting the upstream-divergence guard and restoring compatibility paths.
- Adding maintenance-specific non-regression gates and quality ratchets.
- Decomposing fork-owned monoliths behind stable facades.
- Moving fork-only responsibilities out of upstream-owned hot files through
  localized hooks and sibling modules.
- Bounding duplication between the two upstream-owned configuration models,
  sharing fork-only validation through sibling helpers, and enforcing parity
  automatically.
- Preserving spend admission, prompt caching, context shaping, verification,
  workflows, teams, background tasks, memory, LSP, and session behavior.
- Staging intentional workflow-failure and configuration compatibility changes
  behind explicit migration and rollback boundaries.
- Producing deterministic, reviewable evidence for every iteration.

### Out of scope

- Removing features merely to reduce line count.
- Renaming, splitting, relocating, or deleting upstream-owned files.
- Completing every open item in the cost and reliability roadmap.
- Changing model routing, prompts, tool visibility, or default workflow behavior
  during behavior-preserving iterations.
- Treating cosmetic generated-code cleanup as an architectural solution.
- Accepting a performance regression because the new architecture looks cleaner.
- Replacing current snapshot baselines without an approved user-facing change.
- Running uncontrolled paid-model evaluations without a hard spend envelope.

### Assumptions

- Python 3.12 and dependencies from the committed `uv.lock` are used.
- Commands run through `uv run`.
- The `upstream` remote and the pinned upstream baseline are available in a
  full-history checkout.
- The current LSP route-pool work is completed and green or isolated before the
  baseline is recorded. A dirty worktree is never used as the preservation
  baseline.
- Baselines are identified by commit SHA and artifact digest. They are not Git
  tags because this repository treats version tags as releases.
- Evidence is written to a dedicated workspace outside every Git candidate
  worktree. The evidence runner sets `VIBE_EVIDENCE_WORKSPACE` and
  `KILROY_RUN_ID`; the latter is an evidence-run identifier, not product
  configuration.
- Upstream syncs occur only between iterations. The pinned merge base and
  affected evidence are refreshed in a dedicated sync change.

## Audit baseline and motivation

The initial read-only audit compared committed `HEAD` to the local
`upstream/main` baseline at `ac8f1a09` (`v2.18.4`). It found:

- 993 fork-only commits across approximately 25 days.
- 968 changed files with 144,986 insertions and 10,775 deletions.
- 490 added files, 468 modified files, 8 deleted files, and 2 renamed files.
- 189 of 349 upstream production Python files no longer matching upstream.
- 212 changed hunks in `vibe/core/agent_loop.py`.
- 250 changed hunks in `vibe/cli/textual_ui/app.py`.
- A 2,960-line fork-owned `vibe/core/workflows/runtime.py` whose
  `_run_agent` method is approximately 565 lines and has cyclomatic complexity
  62.
- Strong mechanical controls: Ruff and Pyright passed, the iron-law and current
  divergence tests passed, and the fork has extensive test coverage.

The audit conclusion was not that the repository lacks tests or formatting
discipline. The risk is that feature throughput outpaced consolidation, leaving
large upstream overlays, fork-owned coordinators with too many responsibilities,
and preservation checks that do not yet fail on measured performance or
model-behavior regressions.

## Target state

The campaign is complete when:

1. Upstream paths remain present and divergence detection covers exact paths,
   renames, tests, and scripts.
2. Each fork-only subsystem connects to upstream-owned code through a small,
   localized seam.
3. Fork-owned modules pass normal complexity limits without blanket file-level
   suppression.
4. Workflow failures have explicit semantics without silently breaking existing
   user workflows.
5. Fork-added configuration fields and validators do not duplicate behavior;
   the two upstream-owned models have automated type/default/alias/merge-policy
   parity. A true single-source redesign lands upstream first or receives an
   explicit structural-divergence decision.
6. Functional, message, performance, cost, and safety behavior is backed by
   reproducible evidence.
7. Every iteration can be reverted without reverting unrelated cleanup.

## Definitions

### Behavior-preserving change

A change is behavior-preserving only when all applicable conditions are true:

- Public imports, entry points, configuration, defaults, return values, event
  order, error classes, messages, and side effects remain equivalent.
- Backend request bodies and prompt/cache inputs remain equivalent when the
  touched path contributes to them.
- Existing snapshots and protocol fixtures do not change.
- Relevant performance metrics remain within the baseline-derived noise
  envelope.
- No new upstream-owned path or hunk is introduced except an explicitly approved
  localized hook.

Moving an exception boundary, changing a default, converting `None` into a typed
object, altering tool exposure, or changing a model-visible string is not a
behavior-preserving refactor.

### Model-visible change

A model-visible change affects any of the following:

- System or agent prompts.
- Tool names, descriptions, schemas, or result text.
- Message ordering or compaction.
- Workflow results, retry behavior, or failure representation.
- Cache routing, request metadata, or backend request bodies.
- Model routing, reasoning settings, or provider selection.

Model-visible changes require the maintenance evaluation gate. Unit tests alone
cannot establish equivalence.

### Upstream-owned path

A path is upstream-owned when it exists at the pinned upstream baseline. Fork
edits may add localized hooks to it, but the path remains present, ordered, and
mergeable. A fork-added sibling may be freely refactored provided its public
contract remains stable.

### Iteration

An iteration is one independently revertible concern. An iteration may contain
several small PRs when each PR extracts one seam, but it cannot mix mechanical
movement, compatibility changes, or optimization.

## Delivery and rollback policy

- Start from a clean worktree and record the exact candidate SHA.
- Keep unrelated work, including active LSP work, outside the iteration.
- Use one PR or one normally revertible commit series per rollback boundary.
- Do not amend, force-push, or rewrite already-pushed history.
- Once a PR is open, merge `origin/main` into it when reconciliation is needed.
- Do not sync upstream midway through an extraction.
- Do not update performance baselines or snapshots in a candidate change merely
  because a gate failed.
- Baseline changes require their own evidence-backed change.
- Freeze all intended edits before final verification. Do not mutate the
  candidate while the verifier is running.
- A failed gate causes the iteration to stop. Revert or correct the iteration;
  do not widen a threshold without a separate review of measurement quality.

## Deliverables

| Artifact | Location | Description |
|---|---|---|
| Execution roadmap | `docs/design/fork-maintenance-roadmap.md` | This scope, sequence, DoD, evidence contract, and test matrix. |
| Design index entry | `docs/design/README.md` | Discoverable link to this roadmap. |
| Execution control layer | `docs/design/fork-maintenance/` | Authority matrix, machine-readable campaign status, packet template, and frozen iteration packets suitable for bounded agent execution. |
| Baseline manifest | `$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest/manifest.json` | Commit, environment, commands, scenarios, results, and artifact paths outside the candidate worktree. |
| Fork ownership report | Iteration evidence under `IT-12/` | Upstream/fork path ownership, changed paths, per-hotspot hunks, and absent paths. |
| Message baseline | Iteration evidence under `IT-01/`, `IT-02/`, `IT-03/`, and `IT-04/` | CLI/TUI/ACP/model-visible output fixtures with normalized volatile values. |
| Performance samples | Iteration evidence under `IT-14/` | Raw baseline/candidate samples, profiles, and comparison summary. |
| Maintenance eval mode | `evals/gates.py`, `evals/cli.py` | Non-regression comparison that does not require a 30% cost improvement. |
| Divergence guard | `scripts/check_upstream_divergence.py` | Exact baseline-path membership check covering configured upstream-owned paths. |
| Divergence fixtures | `tests/test_upstream_divergence.py` | Deletion, exact rename, copy-delete, accepted divergence, and shallow-clone cases. |
| Programmatic delivery test | `tests/e2e/test_cli_programmatic.py` | Real subprocess success and failure journeys for `vibe -p`. |
| Installed-wheel TUI lifecycle test | `tests/e2e/test_cli_tui_lifecycle.py` | One real installed-wheel journey covering onboarding, streaming, approval, result, resume, and exit evidence. |
| ACP subprocess lifecycle test | `tests/acp/test_acp_subprocess_lifecycle.py` | One real `vibe-acp` subprocess journey covering help, session, prompt, tool, usage, reload, and close. |
| Harness fixture runner | `scripts/run_harness_evals.py` | Hermetic baseline/candidate fixture execution with trusted event ingestion, fixed seeds, and mandatory spend cap. |
| Maintenance fixtures | `evals/fixtures/maintenance/` | Versioned core, policy, and security repository/task/recipe fixtures. |
| Performance evidence helper | Fork-added helper under `tests/perf/` | Common structured sample and environment output for current performance harnesses. |
| Performance comparison runner | `scripts/compare_performance.py` | Clean-ref worktree orchestration, calibration, randomized paired sampling, bootstrap non-inferiority report, and profile collection. |
| Evidence runner | `scripts/run_maintenance_evidence.py` | External-workspace scenario orchestration, normalization, artifact hashing, and manifest emission. |
| Fork baseline reporter | `scripts/report_fork_baseline.py` | Exact-path ownership, changed-path, lock identity, and hotspot-hunk reports against frozen refs. |
| Fork baseline reporter tests | `tests/maintenance/test_fork_baseline_report.py` | Temporary-repository exact-path, diff, hotspot, guard-characterization, and deterministic-output contracts. |
| Evidence contract tests | `tests/maintenance/test_evidence_contract.py` | Reproducibility, dirty-baseline rejection, controlled failure, best-effort artifacts, and manifest validation. |
| Characterization contracts | Existing mirrored test locations | Golden behavior for runtime, AgentLoop, VibeApp, config, spend, and orchestration seams. |
| Iteration verification report | Per-iteration evidence manifest | Applicable AC/IT status, exact commands, profiles, snapshots, and verifier verdict. |

## Gate tiers

### Tier A: every PR

- Targeted tests for the touched subsystem.
- Ruff check and formatting.
- Pyright.
- Iron-law and upstream-divergence tests.
- Changed-path, upstream-hunk, complexity, warning, and message-delta report.
- Applicable exact performance invariants.

### Tier B: iteration exit

- Full non-snapshot suite with coverage.
- Snapshot suite when any CLI/TUI/rendering path is reachable from the change.
- Integration scenarios mapped to the iteration.
- Paired performance comparison for touched hot paths.
- Disposable upstream merge rehearsal.
- Frozen-candidate verifier run.

### Tier C: model-visible or release boundary

- Maintenance evaluation on aligned hermetic fixtures.
- Paid model trials when the behavior depends on real provider/model behavior.
- Hard spend cap through the broker.
- Release notes and migration instructions for intentional compatibility changes.
- Full evidence crosscheck.

## Iteration plan

### Iteration 0: Freeze evidence, not code

Purpose: establish reliable preservation contracts before moving implementation.

Execution uses the packet controls under `docs/design/fork-maintenance/`.
Early Iteration 0 packet baselines identify their individual rollback boundaries;
they are not the final preservation baseline. After all Iteration 0 tooling and
characterization changes are consolidated and green, freeze that clean commit as
the campaign baseline, rerun baseline inventory, and capture it with manifest
`baseline_sha == candidate_sha`. Do not begin Iteration 1 before this bootstrap
is complete.

#### Entry conditions

- Active LSP work is green and landed or isolated from the baseline worktree.
- `uv run git status --short` is empty.
- The committed upstream baseline is available locally.
- No performance number is collected from a dirty tree.

#### Work

1. Record candidate SHA, upstream SHA, `uv.lock` digest, Python version, OS,
   machine/runner identity, CPU count, and relevant environment variables.
2. Record upstream-owned versus fork-added paths, absent upstream paths, modified
   upstream Python paths, and per-hotspot changed-hunk counts.
3. Run the full functional and snapshot suites and retain their reports.
4. Capture CLI help, programmatic output, TUI states, ACP events, and
   model-visible tool results using the message rules below.
5. Add the missing real programmatic subprocess scenario at
   `tests/e2e/test_cli_programmatic.py`.
6. Add characterization coverage before moving these seams:
   - `WorkflowRuntime`: results, event order, cancellation, spend, cache, repair,
     resume, isolation, and cleanup.
   - `AgentLoop`: backend requests, events, tool order, spend, cache inputs,
     traces, compaction, and resume.
   - `VibeApp`: message queue, history, pruning, startup, snapshots, streaming,
     and session exit.
   - Configuration: defaults, aliases, env loading, layered merge, validation,
     `model_dump`, reload, migration, and parity between both upstream models.
7. Make the current performance harnesses emit a common structured result while
   preserving their human-readable profiler output.
8. Run repeated baseline-versus-baseline samples to establish the natural-noise
   envelope for each measured metric.
9. Capture aligned harness-evaluation inputs where a fixture runner already
   exists and record the explicit model-behavior coverage gap where it does not.

#### Exit criteria

- AC-1.1 and AC-1.5 pass. AC-1.2 passes as an evidence-completeness assertion:
  IT-01 through IT-14 have baseline entries, and unavailable IT-15 is recorded
  as `status: fail` with a gap artifact and explicit missing-runner notes until
  Iteration 2 implements it. AC-1.3 and AC-1.4 have positive and negative
  evidence-runner tests.
- The full committed suite has no failures.
- IT-01 through IT-14 run in baseline/characterization mode and have evidence
  entries. Capabilities intentionally scheduled for later iterations, including
  repaired rename detection and final config parity, are recorded as explicit
  gaps rather than falsely reported as passing.
- Performance baselines contain raw samples, not only summaries.
- Message fixtures cover every currently reachable message group. Future
  Iteration 8 surfaces, including MSG-12 deprecation output and MSG-15 explicit
  failure policies, have `status: fail` gap fixtures until those messages exist.

#### Rollback boundary

Characterization and evidence tooling form their own PR. They do not change
production behavior. If the tooling is unreliable, revert it before structural
work begins.

### Iteration 1: Repair fork safety and compatibility paths

Purpose: make the fork guard enforce the repository's stated path-preservation
policy before further refactoring.

#### Work

1. Replace rename-sensitive `git diff --diff-filter=D` detection with exact
   baseline-tree path membership at `HEAD`, or use a demonstrably equivalent
   no-renames comparison.
2. Cover configured upstream-owned production, test, and script paths rather
   than only `vibe/**/*.py`.
3. Add fixtures for:
   - Direct deletion.
   - Exact rename.
   - Copy plus deletion of the original.
   - Accepted divergence with a nonempty reason.
   - Unexpected divergence.
   - Missing baseline in a shallow clone.
4. Restore original upstream paths through thin compatibility modules where
   possible, including the known `vibe/cli/turn_summary/port.py` case. For
   `scripts/bump_version.py`, either provide a deprecated forwarding entry point
   whose release semantics are documented accurately or accept the divergence;
   do not imply that tag-based `release.py` still performs the old literal
   version-editing workflow.
5. Review deleted upstream tests. Restore the original path or record why the
   absence is accepted; deleting a test is not automatically harmless to future
   merges.
6. Run a disposable merge-tree or equivalent upstream-sync rehearsal and retain
   the result.

#### Exit criteria

- The guard detects every injected deletion and rename.
- Known missing paths are restored or explicitly accepted with reviewed reasons.
- A clean full-history checkout exits 0.
- A shallow checkout skips with an accurate message.
- Runtime behavior and public compatibility are unchanged.

#### Rollback boundary

Guard changes and compatibility shims land together as one fork-safety PR. They
do not include extraction work.

### Iteration 2: Install non-regression and quality ratchets

Purpose: turn current observations into gates that can block regressions.

#### Work

1. Add a maintenance comparison mode to the offline evaluator. It must retain
   artifact/trial alignment and safety gates but must not require a 30% cost
   improvement from a neutral refactor.
2. Add structured performance comparison using the method in the Performance
   Evidence Contract.
3. Add fork metrics:
   - Missing upstream paths.
   - Modified upstream Python paths.
   - Changed hunks and lines for designated hotspots.
   - Size and complexity of fork-added production files.
   - Per-file Ruff suppressions.
   - Pytest warning inventory.
4. Establish current-value ceilings, then require monotonic improvement:
   - No new missing upstream path.
   - No new upstream-owned modified path without explicit approval.
   - No increase in hotspot hunks during extraction.
   - No new blanket per-file complexity suppression.
   - No new fork-added function above configured complexity limits.
5. Classify existing warnings. Fix fork-owned asyncio marker and resource cleanup
   warnings, then make new fork-owned warnings fatal.
6. Add negative tests proving that injected timing, memory, warning, complexity,
   fork-path, and message regressions fail their gate.
7. Include `evals/` in appropriate Ruff, Pyright, and coverage checks.

#### Maintenance evaluator policy

The maintenance mode must require:

- Identical dataset, task/category/model/profile coverage, artifacts, trial
  indices, and random seeds.
- Zero policy/security unsafe-mutation incidents.
- Core false-done rate below the existing threshold.
- No pass@1 drop beyond the existing two-percentage-point limit.
- Harness utilization at most 20%.
- Maintenance utilization at most 5%.
- Spend attribution completeness of at least 99%.
- No statistically credible regression beyond the baseline-derived envelope in
  tokens, cost, calls, retries, interventions, or wall time.

#### Exit criteria

- An unchanged candidate passes every new gate.
- Each injected regression fails the expected gate with an actionable report.
- The warning allowlist is exact, owned, and reasoned; broad filters are absent.
- The baseline cannot be silently replaced from a candidate run.

### Iteration 3: Decompose `WorkflowRuntime` mechanically

Purpose: reduce the largest fork-owned coordinator while preserving every
existing external and model-visible behavior.

`WorkflowRuntime` remains the stable facade throughout this iteration. Existing
imports and helper symbols keep compatibility forwards where needed.

#### Sub-iterations

1. Extract pure JSON, schema, and validation helpers.
2. Extract accounting, cost synchronization, reservation reconciliation, and
   finalization.
3. Extract isolated execution, cache identity, delivery, and cleanup.
4. Extract agent execution, response streaming, schema repair, and retry state.
5. Extract `parallel`/`pipeline` combinator implementation behind the existing
   methods.

Each sub-iteration is one PR. Later sub-iterations do not begin until the prior
one has complete evidence.

#### Preservation contract

- `WorkflowRuntime` constructor and public methods remain import-compatible.
- `parallel()` and `pipeline()` retain ordered results and bounded concurrency.
- Pipeline items retain no-barrier semantics.
- Ordinary child failures retain current legacy `None` behavior.
- Hard budget and spend failures continue to propagate and block the run.
- Cancellation, timeout, retry, live progress, resume, cache, repair, and
  worktree cleanup remain equivalent.
- Backend calls, token/cost accounting, and retained conversation behavior remain
  equivalent.
- No new per-file Ruff suppression is added.

#### Exit criteria per sub-iteration

- IT-05, IT-06, IT-10, IT-13, and IT-14 pass.
- Golden event and result transcripts are unchanged.
- `runtime.py` size and designated complexity debt decrease.
- New collaborators pass normal configured complexity limits.
- Workflow fanout remains concurrent and within the performance envelope.

### Iteration 4: Bound configuration duplication and enforce parity

Purpose: remove fork-added duplicated behavior without restructuring either of
the two upstream-owned configuration models.

#### Work

1. Preserve both upstream-owned models and their file structure:
   `vibe/core/config/_settings.py` and `vibe/core/config/vibe_schema.py`.
2. Inventory which duplicated fields, validators, and accessors are upstream
   structure versus fork-added behavior.
3. Move only fork-added shared validation/normalization logic into new sibling
   helpers, leaving small localized calls from both upstream files.
4. Retain merge policy in the upstream schema structure and add a generated
   parity inventory rather than locally replacing the schema with a new model.
5. Characterize and prove parity for:
   - Type and default/default-factory behavior.
   - Validation and serialization aliases.
   - Excluded fields.
   - Unknown fields and error messages.
   - User/project/harness/environment layer precedence.
   - Nested provider/model merge semantics.
   - Live reload and event propagation.
   - Session and provider migrations.
6. Replace the name-only sync instruction with generated parity tests for field
   type, default/default-factory, aliases, exclusions, and merge-policy coverage.
   Dual field declarations remain where upstream requires them.
7. Inventory no-consumer fork fields. Keep them accepted during this iteration.
   Wiring or deprecation is a later compatibility change.
8. Generate schema/config documentation inventories where doing so removes
   another manual source of truth. Keep curated rationale and examples
   hand-written.

#### Exit criteria

- All previously valid config fixtures produce equivalent runtime values.
- Invalid fixtures retain error location and meaning.
- Repeated migrations are idempotent.
- Explicit user values survive reload and migration.
- Import/startup performance remains within its envelope.
- The name-only “add every field twice” contract is replaced with semantic
  parity checks, and fork-added duplicated validator bodies are gone.
- Neither upstream-owned config file gains a broad structural overlay. A true
  single-source redesign is deferred to an upstream change or an explicitly
  approved divergence.

### Iteration 5: Shrink the `AgentLoop` upstream overlay

Purpose: move fork-only responsibilities out of an upstream-owned hotspot while
preserving upstream structure and hot-path performance.

#### Preparation

Create a hunk ownership map that associates every fork-only `agent_loop.py` hunk
with a subsystem and its current tests. Do not move upstream code merely to make
the file shorter.

#### Extraction order

1. Cold diagnostics, tracing, profiler, and observability wiring.
2. Verification state and receipt integration.
3. Spend/accounting and background-delivery integration.
4. Policy and middleware integration.
5. Tool dispatch and conversation/chat seams only after all earlier slices prove
   the extraction pattern and performance gate.

Each subsystem moves into a fork-owned sibling collaborator or mixin. The
upstream file retains a localized hook and compatibility forwarding methods when
callers or tests rely on the original name.

#### Preservation contract

- Backend completion requests and provider metadata are equivalent.
- System prompt, tool manifest, cache key, and stable prefix are byte-equivalent.
- Events, tool order, permission behavior, hooks, errors, retries, compaction,
  failover, and resume remain equivalent.
- Spend reservations and reconciliations remain exact.
- No blocking I/O moves onto the event-loop thread.
- Large response and reader/writer fanout remain within the performance envelope.

#### Exit criteria per seam

- Applicable IT-03, IT-04, IT-05, IT-06, IT-07, IT-08, IT-10, IT-13, and
  IT-14 scenarios pass.
- The subsystem leaves at most a localized construction/invocation hook in the
  upstream file.
- Total `agent_loop.py` fork hunk and line counts decrease.
- A sync rehearsal introduces no new conflict.

### Iteration 6: Shrink the `VibeApp` upstream overlay

Purpose: move fork-owned UI responsibilities out of the second-largest upstream
hotspot without losing startup, rendering, or long-session gains.

#### Extraction order

1. Cold dialogs and commands.
2. Workflow, team, background, spend, LSP, and status presenters/controllers.
3. Queue, history, image, and session lifecycle.
4. Streaming, layout, pruning, and event-loop-sensitive paths last.

Textual message handlers and private methods retain compatibility forwards where
tests or downstream callers depend on them.

#### Preservation contract

- Existing SVG snapshots remain unchanged for mechanical movement.
- Required PNG evidence shows the same key states.
- CLI/TUI messages and key bindings remain unchanged.
- Startup imports and cold start remain within their envelope.
- Streaming output is byte-equivalent and retains coalescing behavior.
- Memory/widget growth and pruning remain bounded.
- Session resume and windowing remain equivalent.

#### Exit criteria per seam

- IT-02, IT-13, and IT-14 pass.
- No unexpected snapshot or message delta exists.
- `app.py` fork hunk and line counts decrease.
- UI CPU, streaming scaling, and retained-memory slopes remain within envelope.

### Iteration 7: Remaining fork-owned hotspots

Purpose: apply the proven extraction pattern to sensitive fork-owned modules one
subsystem at a time.

#### 7A: Spend ledger

- Add a conservation test proving leaf calls sum to every parent scope and the
  session total.
- Prove no orphan call IDs and no negative reservations.
- Preserve reservation-before-dispatch, missing-usage charging, undispatched
  release, retry authorization, lease replay, cross-process attachment, reload,
  reset, and resume.
- Decompose only after these exact invariants pass.

#### 7B: Task, workflow, team, and background lifecycle

- Inventory overlapping lifecycle states and cancellation/delivery behavior.
- Consolidate internal machinery only behind current model-facing tool contracts.
- Do not hide or rename tools without usage evidence and a separate product
  decision.

#### 7C: LSP

- Begin only after route-pool behavior is stable and the full LSP suite is green.
- Preserve JSON-RPC, Unicode positions, pagination, diagnostics, routing,
  readiness, reload retirement, security, and user-visible summaries.
- Benchmark workspace routing and large-symbol-result paths before movement.

Each subsection is an independent iteration and rollback boundary.

### Iteration 8: Intentional compatibility changes

Purpose: improve contracts that cannot be corrected by behavior-preserving
movement.

This iteration is optional until its evaluation prerequisites exist. It never
shares a PR with structural extraction.

#### Workflow failure migration

1. Introduce a typed internal expected-failure representation and preserve the
   public adapter that converts ordinary failures to `None`.
2. Add an explicit failure policy such as strict, collect, or legacy best-effort.
3. Migrate bundled workflows to an explicit policy.
4. Add deprecation messaging for implicit legacy behavior.
5. Change the external default only in a declared compatibility-breaking release
   after the migration window.

Do not silently replace `None` with a truthy failure object.

#### Configuration retirement

1. Confirm a field has no production consumer and inventory real config usage.
2. Keep the field accepted while emitting an accurate deprecation message.
3. Provide migration behavior and documentation.
4. Remove only at a declared compatibility boundary.

#### Tool-surface consolidation

Do not consolidate or hide model-facing tools solely from static inspection.
First gather usage, selection-error, cancellation, and result-utilization data.
Any consolidation receives its own model-visible change contract.

#### Exit criteria

- Legacy external workflow fixtures continue to work during the migration.
- Strict, collect, and legacy behavior are each tested end to end.
- New messages appear in the message inventory and release notes.
- The maintenance evaluation and paid trials pass.
- The default switch or field removal can be reverted independently from the
  typed internals or migration machinery.

### Iteration 9: Convergence and final ratchets

Purpose: close the campaign without hiding remaining debt.

#### Work

- Remove the small set of narrating comments, needless temporaries, and
  promotional documentation language identified by the surface audit.
- Fix remaining owned warnings and make warning gates stricter.
- Parametrize redundant test matrices only after proving retained branch or
  mutation value; do not delete tests based only on similarity.
- Update README, OpenWiki, the builtin Vibe skill, configuration reference, and
  release notes for actual API/config changes.
- Lower hunk, complexity, warning, and suppression ceilings to the achieved
  values. Never raise them to land the final iteration.
- Run a final upstream sync rehearsal and all Tier A, B, and C gates.

#### Exit criteria

- All acceptance criteria pass.
- Every scenario has complete evidence.
- No documentation claims an unavailable config, tool, or behavior.
- No candidate mutation occurs after the final verifier starts.
- The final report states remaining accepted debt explicitly.

## Acceptance criteria

### Baseline and reproducibility

| ID | Criterion | Covered by |
|---|---|---|
| AC-1.1 | A baseline manifest exists and records clean baseline/candidate SHAs, upstream SHA, `uv.lock` digest, Python, OS, machine identity, commands, and artifact digests. | IT-13, IT-14 |
| AC-1.2 | Every scenario records pass/fail status and at least one readable evidence artifact. | IT-01 through IT-15 |
| AC-1.3 | Re-running deterministic CLI, ACP, AgentLoop, and divergence scenarios from the same commit produces equivalent normalized results. | IT-13 |
| AC-1.4 | A failed scenario still writes a manifest entry and best-effort evidence or an explicit missing-artifact note. | IT-13 |
| AC-1.5 | No baseline performance sample is accepted from a dirty worktree. | IT-14 |

### Fork safety and architecture

| ID | Criterion | Covered by |
|---|---|---|
| AC-2.1 | Deleting or renaming an upstream-owned production, test, or script path causes the divergence gate to fail. | IT-12 |
| AC-2.2 | Every accepted missing upstream path has a nonempty reviewed reason. | IT-12 |
| AC-2.3 | A shallow checkout emits an accurate skip result and a full-history checkout evaluates the pinned baseline. | IT-12 |
| AC-2.4 | A behavior-preserving extraction does not increase modified upstream paths or designated hotspot hunk counts. | IT-12, IT-13 |
| AC-2.5 | Each extracted fork subsystem leaves only localized hooks or compatibility forwards in upstream-owned files. | IT-13, semantic review SR-04 |
| AC-2.6 | No new blanket per-file complexity suppression is introduced. | IT-13 |

### Delivery surfaces and compatibility

| ID | Criterion | Covered by |
|---|---|---|
| AC-3.1 | `vibe --help`, `vibe --version`, `vibe-acp --help`, and the programmatic success/failure paths retain exit-code and normalized output contracts. | IT-01, IT-03 |
| AC-3.2 | A fresh installed wheel launches the TUI, streams output, handles approval, displays tool results, persists/resumes, and exits cleanly. | IT-02 |
| AC-3.3 | ACP initialize, session, prompt, tool, usage, command, reload, and close events retain required fields and meanings. | IT-03 |
| AC-3.4 | Tool approval/denial, failed-command recovery, parallel result order, compaction, and resumed history remain equivalent. | IT-04 |
| AC-3.5 | No user-facing or model-visible message changes during a behavior-preserving iteration without an approved message-delta record. | IT-01 through IT-11, IT-13 |
| AC-3.6 | Mechanical TUI refactors produce no unexpected snapshot delta. | IT-02, IT-13 |

### Orchestration, safety, and spend

| ID | Criterion | Covered by |
|---|---|---|
| AC-4.1 | Workflow `parallel()` and `pipeline()` preserve result ordering, bounded concurrency, and no-barrier pipeline behavior. | IT-05 |
| AC-4.2 | Workflow isolation, resume, cache identity, repair, cancellation, spend blocking, result delivery, and worktree cleanup remain equivalent. | IT-05 |
| AC-4.3 | Team claims, mailbox state, retry, dependency unlocking, per-task loop creation, and shared spend scopes remain correct across processes. | IT-06 |
| AC-4.4 | Background processes can launch, stream/tail, stop, and reap without blocking the foreground or leaking children. | IT-06 |
| AC-4.5 | Only a current verifier PASS plus a valid state-bound receipt can authorize `land_work`. | IT-07 |
| AC-4.6 | FAIL/PARTIAL, denied/skipped tools, dirty state, moved base, changed candidate, stale receipt, or pasted prose cannot authorize landing. | IT-07 |
| AC-4.7 | Spend reservations reconcile exactly across retries, missing usage, errors, children, reload, reset, and resume. | IT-10 |
| AC-4.8 | Rejected spend causes zero backend dispatches, and every paid production call is brokered or explicitly documented. | IT-10 |

### Memory, LSP, and configuration

| ID | Criterion | Covered by |
|---|---|---|
| AC-5.1 | Memory add, recall, update, trash/restore, scoping, and local-first selection remain correct. | IT-08 |
| AC-5.2 | Injected memory does not mutate persisted history or destabilize the cacheable prefix. | IT-08, IT-10 |
| AC-5.3 | LSP JSON-RPC, Unicode positions, routing, pagination, diagnostics, readiness, security, and reload retirement remain correct. | IT-09 |
| AC-5.4 | Layered TOML, environment, config reload, schema generation, and migrations preserve explicit user values and are idempotent. | IT-11 |
| AC-5.5 | Both upstream-owned configuration models have automated type/default/alias/exclusion/merge-policy parity, and fork-added shared validation has one sibling implementation. | IT-11, IT-13 |

### Performance and model behavior

| ID | Criterion | Covered by |
|---|---|---|
| AC-6.1 | Forbidden eager-import sets remain empty for guarded startup modules. | IT-14 |
| AC-6.2 | No new event-loop blocker crosses the configured 20 ms threshold. | IT-14 |
| AC-6.3 | Agent turn, fanout, TUI CPU, memory growth, streaming scaling, and index rebuild remain within their baseline-derived noise envelopes. | IT-14 |
| AC-6.4 | Fanout remains concurrent and ordered; a sequential implementation cannot pass only by matching wall time. | IT-04, IT-05, IT-14 |
| AC-6.5 | Prompt/cache inputs and deterministic request bodies remain byte-stable where required. | IT-04, IT-08, IT-10 |
| AC-6.6 | Cache-token normalization, pricing, and session cost accounting remain exact. | IT-10 |
| AC-6.7 | A model-visible change passes the maintenance evaluation on aligned fixtures before landing. | IT-15 |
| AC-6.8 | Paid trials run with a hard broker cap and meet safety, reliability, utilization, attribution, and non-regression gates before a default change. | IT-15 |

### Quality and rollout

| ID | Criterion | Covered by |
|---|---|---|
| AC-7.1 | Ruff, formatting, Pyright, pre-commit, iron laws, and divergence checks exit 0. | IT-13 |
| AC-7.2 | Full non-snapshot coverage remains at least 85%. | IT-13 |
| AC-7.3 | Fork-owned tests emit no unclassified warnings after Iteration 2. | IT-13 |
| AC-7.4 | Structural movement, compatibility changes, and optimization appear in separate rollback boundaries. | IT-13, semantic review SR-01 |
| AC-7.5 | A compatibility change includes tested migration behavior, message inventory, documentation, and a separately revertible default switch/removal. | IT-05, IT-11, IT-15, semantic review SR-03 |
| AC-7.6 | Documentation and the builtin Vibe skill describe only currently available behavior. | IT-13, semantic review SR-02 |

## User-facing message inventory

Behavior-preserving iterations freeze the message groups below. A later
intentional change must enumerate each new or changed literal/structured field
in its iteration-specific addendum.

| ID | Message surface | Trigger condition | Covered by |
|---|---|---|---|
| MSG-01 | CLI help, version, stdout, stderr, and exit status | Help/version invocation, programmatic success, missing key, backend failure | IT-01 |
| MSG-02 | Authentication, onboarding, config, and trust errors/prompts | Missing/invalid credentials, first run, broken config, untrusted workspace | IT-01, IT-02, IT-03, IT-11 |
| MSG-03 | TUI startup, streaming, resume, quit, and session status | Fresh start, streamed response, resume, normal and interrupted exit | IT-02 |
| MSG-04 | Permission, tool progress, result, denial, and failure displays | Approved/denied tool, nonzero command, timeout, truncated result | IT-02, IT-03, IT-04 |
| MSG-05 | Workflow launch, phase, completion, failure, blocked, stop, and resume output | Workflow lifecycle and spend exhaustion | IT-05 |
| MSG-06 | Team/task claim, status, retry, completion, dependency, and mailbox output | Team and task lifecycle | IT-06 |
| MSG-07 | Background handle, list/tail, completion, failure, and stop output | Background process/agent lifecycle | IT-06 |
| MSG-08 | Verification verdict, invalidation, trusted-check, receipt, and landing output | Verify/land success and every rejection path | IT-07 |
| MSG-09 | Memory add/list/update/trash/restore and recall failure output | Memory tool and automatic recall lifecycle | IT-08 |
| MSG-10 | LSP readiness, partial coverage, pagination, diagnostics, install/reload, and error output | LSP operations and lifecycle | IT-09 |
| MSG-11 | Spend status, projected-limit rejection, reset, reload, and usage output | Session/child spend lifecycle | IT-10 |
| MSG-12 | Config validation, migration, reload, and deprecation output | Config parsing, migration, live reload, retired field use | IT-11 |
| MSG-13 | ACP protocol messages and required fields | Initialize, auth, session, prompt, tools, usage, commands, close | IT-03 |
| MSG-14 | Model-visible tool results and recovery hints | Tool/workflow/team/background/memory/LSP/verification/spend results sent to the model | IT-04 through IT-10 |
| MSG-15 | Workflow strict/collect/legacy failure output | Iteration 8 failure-policy selection and failures | IT-05, IT-15 |

### Message comparison rules

The baseline comparator may normalize only:

- ANSI escape sequences.
- UUIDs and generated message/session/tool-call IDs.
- Temporary paths and platform-specific path separators.
- Timestamps and explicitly variable duration/counter values.
- Random local port numbers.

It must not normalize:

- Labels, severity, status, or verdict.
- Recommended action or recovery hint.
- Exit code.
- ACP field presence or semantic values.
- Tool name, result kind, or permission decision.
- Spend limit type or projected/actual relationship.
- Verification state or authorization result.

Every changed user-visible or model-visible output is classified in the PR
evidence as one of:

1. An unchanged contract whose characterization fixture was corrected.
2. An intentional product change with approval, updated tests, migration/release
   notes, and maintenance evaluation where model-visible.
3. An accidental change, which blocks the iteration.

## Test evidence contract

The evidence runner sets:

```bash
VIBE_EVIDENCE_WORKSPACE=/absolute/path/outside/all/git/worktrees
KILROY_RUN_ID=<unique-evidence-run-id>
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
```

`KILROY_RUN_ID` names the evidence run only. The evidence workspace is not the
candidate repository, is not nested inside any linked worktree, and is never
added to the candidate diff. This keeps evidence writes from changing the
workspace fingerprint or invalidating verification and landing authority.

Scenario status remains strictly `pass` or `fail`. A planned capability that is
not implemented is recorded as `fail` with a readable gap artifact and explicit
notes; the manifest does not invent a third `blocked` status.

| Item | Requirement |
|---|---|
| Canonical relative root within the evidence workspace | `.ai/runs/$KILROY_RUN_ID/test-evidence/latest/` |
| Absolute evidence root | `$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest/` |
| Scenario folder | `$EVIDENCE/IT-<id>/` |
| Manifest | `$EVIDENCE/manifest.json` |
| UI scenario | `surface=ui` or `surface=mixed`; include PTY transcript and PNG/JPG screenshots of key states |
| Non-UI scenario | `surface=non_ui`; include text or structured logs/reports |
| Performance scenario | Include raw samples, environment, comparison summary, and profiles |
| Failure behavior | Emit best-effort artifacts and a manifest entry; missing/unreadable artifacts are explicit findings |
| Candidate identity | Record baseline SHA, candidate SHA, upstream SHA, dirty-state check, and artifact digests |

### Manifest shape

```json
{
  "version": 1,
  "baseline_sha": "<40-hex commit>",
  "candidate_sha": "<40-hex commit>",
  "upstream_sha": "<40-hex commit>",
  "environment": {
    "python": "3.12.x",
    "platform": "linux-x86_64",
    "uv_lock_sha256": "<64-hex digest>",
    "runner": "<stable runner identity>"
  },
  "scenarios": [
    {
      "id": "IT-01",
      "surface": "non_ui",
      "status": "pass",
      "command": ["uv", "run", "pytest", "-n0", "..."],
      "recorded_environment": {},
      "exit_code": 0,
      "artifacts": [
        {
          "type": "log",
          "path": "IT-01/command.log",
          "sha256": "<64-hex digest>"
        }
      ],
      "metrics": {},
      "notes": []
    }
  ]
}
```

Manifest values use portable machine contracts: `command` is the exact argv
array executed without a shell, and artifact paths are POSIX strings relative
to the manifest directory. `$EVIDENCE/...` notation elsewhere in this document
describes filesystem locations for humans and is not stored literally.
`recorded_environment` is scenario-local and not part of immutable run identity;
top-level environment identity remains Python, platform, lock digest, and runner.

Each scenario is automatable, bounded, independent, proportional to the risk it
covers, and responsible for setting up its own deterministic starting state.

## Integration test scenarios

### IT-01: Programmatic CLI delivery

- Surface: `non_ui`
- Starting state: clean temporary Vibe home, deterministic local mock backend,
  fixed config, no external network.
- Actions:
  1. Invoke `vibe --help` and `vibe --version`.
  2. Invoke the real `vibe -p` entry point with a deterministic prompt.
  3. Trigger missing-key and backend-failure paths.
  4. Trigger broken-config and untrusted-workspace paths.
- Expected outcomes:
  - Help/version exit 0 with stable normalized output.
  - Success emits only the expected programmatic result and exits 0.
  - Failures exit nonzero with an actionable error and no traceback leakage.
- Verification:

```bash
uv run pytest -n0 \
  tests/cli/test_cli_wiring.py \
  tests/cli/test_programmatic_setup.py \
  tests/e2e/test_cli_programmatic.py
```

- Evidence: `$EVIDENCE/IT-01/command.log`,
  `$EVIDENCE/IT-01/stdout.json`, `$EVIDENCE/IT-01/stderr.json`,
  `$EVIDENCE/IT-01/exit-codes.json`, and `$EVIDENCE/IT-01/junit.xml`.
- Covers: AC-1.2, AC-1.3, AC-3.1, AC-3.5, AC-7.5; MSG-01, MSG-02.

### IT-02: Fresh-wheel TUI lifecycle

- Surface: `mixed`
- Starting state: freshly built wheel installed into an isolated environment,
  temporary Vibe home, deterministic mock server.
- Actions:
  1. Launch the installed wheel from a fresh Vibe home and complete first-run
     onboarding.
  2. Stream a model response.
  3. Trigger a tool approval, approve it, and display the result.
  4. Persist and resume the session through the installed entry point.
  5. Exercise interrupted exit and then exit normally.
- Expected outcomes:
  - Startup, streaming, approval, result, resume, and exit states are correct.
  - Mechanical iterations produce no snapshot delta.
  - Process exits cleanly without leaked resources.
- Verification:

```bash
uv run pytest -n0 \
  tests/e2e/test_cli_tui_fresh_install.py \
  tests/e2e/test_cli_tui_lifecycle.py \
  tests/e2e/test_cli_tui_streaming.py \
  tests/e2e/test_cli_tui_tool_approval.py \
  tests/e2e/agent_loop_characterization/test_resume.py \
  tests/snapshots
```

- Evidence: `$EVIDENCE/IT-02/command.log`,
  `$EVIDENCE/IT-02/pty-transcript.txt`, `$EVIDENCE/IT-02/junit.xml`,
  `$EVIDENCE/IT-02/snapshot-report.html`, and
  `$EVIDENCE/IT-02/screenshots/{onboarding,startup,approval,result,resume}.png`.
- Covers: AC-1.2, AC-3.2, AC-3.5, AC-3.6; MSG-02, MSG-03, MSG-04.

### IT-03: ACP lifecycle and protocol

- Surface: `non_ui`
- Starting state: real `vibe-acp` subprocess, deterministic client/mock backend,
  temporary config and session roots.
- Actions:
  1. Invoke `vibe-acp --help` and assert its exit/output contract.
  2. Launch the real `vibe-acp` subprocess, initialize, and advertise
     authentication capabilities.
  3. Exercise invalid credentials, then create a session and prompt it.
  4. Exercise tool permission/result and usage updates.
  5. Exercise commands and config reload.
  6. Close the session through the subprocess protocol.
- Expected outcomes:
  - Required protocol fields and semantic values remain stable.
  - Session resources close and no background work leaks.
- Verification:

```bash
uv run pytest -n0 \
  tests/acp/test_acp_entrypoint_smoke.py \
  tests/acp/test_acp_subprocess_lifecycle.py \
  tests/acp/test_acp.py \
  tests/acp/test_commands.py \
  tests/acp/test_usage_update.py \
  tests/acp/test_close_session.py
```

- Evidence: `$EVIDENCE/IT-03/command.log`,
  `$EVIDENCE/IT-03/events.jsonl`, `$EVIDENCE/IT-03/subprocess.log`,
  `$EVIDENCE/IT-03/help.txt`, `$EVIDENCE/IT-03/exit-codes.json`, and
  `$EVIDENCE/IT-03/junit.xml`.
- Covers: AC-1.2, AC-3.1, AC-3.3, AC-3.5; MSG-02, MSG-04, MSG-13.

### IT-04: Agent loop, tools, compaction, and resume

- Surface: `non_ui`
- Starting state: deterministic backend streams, temporary workspace/session,
  known tool fixtures.
- Actions:
  1. Approve and deny tool calls.
  2. Run failed, timed-out, and successful commands.
  3. Execute parallel read-only tools and serialized writes.
  4. Trigger result capping and compaction.
  5. Resume the complete history.
- Expected outcomes:
  - Permission, failure-recovery, result ordering, tool linkage, compaction, and
    resumed history remain equivalent.
  - Backend request and event fixtures match the baseline.
  - Model-facing tool results and recovery hints match the normalized baseline.
- Verification:

```bash
uv run pytest -n0 \
  tests/e2e/agent_loop_characterization \
  tests/agent_loop/e2e/test_e2e_tools.py \
  tests/agent_loop/e2e/test_e2e_bash.py \
  tests/agent_loop/e2e/test_e2e_compaction.py \
  tests/agent_loop/test_agent_tool_call.py \
  tests/core/test_tool_result_cap.py
```

- Evidence: `$EVIDENCE/IT-04/command.log`,
  `$EVIDENCE/IT-04/backend-requests.json`,
  `$EVIDENCE/IT-04/events.jsonl`, `$EVIDENCE/IT-04/model-results.json`,
  `$EVIDENCE/IT-04/session-manifest.json`, and `$EVIDENCE/IT-04/junit.xml`.
- Covers: AC-1.2, AC-2.5, AC-3.4, AC-3.5, AC-6.4, AC-6.5; MSG-04, MSG-14.

### IT-05: Workflow orchestration

- Surface: `non_ui`
- Starting state: deterministic in-process agent factory plus a disposable Git
  repository for isolated execution.
- Actions:
  1. Run `parallel()` above and below the concurrency limit.
  2. Run a staggered multi-stage pipeline and prove a fast item enters its next
     stage before a slow item completes its prior stage; record per-item order.
  3. Exercise ordinary failure, hard budget failure, cancellation, repair, and
     blocked/stop states.
  4. Launch the same lifecycle through the delivered `launch_workflow` and
     workflow status/stop/result tool surfaces, asserting phase and terminal
     messages sent to the user and model.
  5. Resume from a snapshot and test cache identity.
  6. Execute an isolated worker and verify cleanup/delivery.
  7. In Iteration 8, select strict, collect, and legacy policies explicitly and
     assert each end-to-end outcome and message.
- Expected outcomes:
  - Ordering, concurrency, current failure semantics, spend blocking, resume,
    cache, repair, worktree cleanup, and result delivery match the iteration's
    declared contract.
  - User- and model-facing phase, blocked, stop, completion, failure, and recovery
    payloads match the declared message contract.
- Verification:

```bash
uv run pytest -n0 \
  tests/core/workflows/test_runtime.py \
  tests/core/workflows/test_isolated_executor_integration.py \
  tests/core/workflows/test_resume.py \
  tests/core/workflows/test_result_delivery.py \
  tests/core/workflows/test_result_repair.py \
  tests/core/workflows/test_spend_blocking.py \
  tests/tools/test_launch_workflow.py \
  tests/tools/test_workflow_results.py \
  tests/tools/test_workflow_status.py \
  tests/tools/test_workflow_stop.py
```

- Evidence: `$EVIDENCE/IT-05/command.log`,
  `$EVIDENCE/IT-05/results.json`, `$EVIDENCE/IT-05/events.jsonl`,
  `$EVIDENCE/IT-05/model-results.json`, `$EVIDENCE/IT-05/spend.json`,
  `$EVIDENCE/IT-05/worktree-state.json`, and `$EVIDENCE/IT-05/junit.xml`.
- Covers: AC-1.2, AC-2.5, AC-4.1, AC-4.2, AC-6.4, AC-7.5; MSG-05, MSG-14,
  MSG-15.

### IT-06: Teams and background processes

- Surface: `non_ui`
- Starting state: temporary team store/mailbox, disposable subprocess fixtures,
  shared spend root.
- Actions:
  1. Spawn competing team workers, force concurrent claim contention, and prove
     exactly one worker claims each task.
  2. Exercise dependency blocking/unlocking and mailbox delivery.
  3. Prove each claimed task receives a fresh `AgentLoop` while cumulative retry
     and spend scope remain correct.
  4. Invoke team/task list and status surfaces through their delivered tools;
     exercise retry and completion and compare user/model-visible payloads.
  5. Launch successful and failing background processes/agents, observe their
     completion/failure delivery, list/tail them, stop a running task, and reap
     all children.
  6. Exercise shared spend scopes across children.
- Expected outcomes:
  - Atomic claim, structured outcome, retry, dependency, mailbox, lifecycle,
    cancellation, and spend behavior remain correct across processes.
  - Team/task/background status, retry, success, failure, completion, and recovery
    results match the normalized user/model message baseline.
- Verification:

```bash
uv run pytest -n0 \
  tests/core/teams \
  tests/tools/test_background_tool.py \
  tests/tools/test_background_registry.py \
  tests/tools/test_team_spawn.py \
  tests/tools/test_team_message.py \
  tests/tools/test_team_tool.py \
  tests/tools/test_team_task_protocol.py
```

- Evidence: `$EVIDENCE/IT-06/command.log`,
  `$EVIDENCE/IT-06/task-store.json`, `$EVIDENCE/IT-06/mailbox.jsonl`,
  `$EVIDENCE/IT-06/processes.log`, `$EVIDENCE/IT-06/outcomes.json`,
  `$EVIDENCE/IT-06/model-results.json`, `$EVIDENCE/IT-06/spend.json`,
  `$EVIDENCE/IT-06/cleanup.json`, and `$EVIDENCE/IT-06/junit.xml`.
- Covers: AC-1.2, AC-4.3, AC-4.4; MSG-06, MSG-07, MSG-14.

### IT-07: Verification receipt and landing authority

- Surface: `non_ui`
- Starting state: disposable Git repository, fixed task brief and trusted recipe,
  deterministic verifier outputs.
- Actions:
  1. Produce a verifier PASS, run trusted checks, create a receipt, and land.
  2. Exercise FAIL, PARTIAL, denied/skipped tools, dirty candidate, moved base,
     changed artifact, superseded generation, and pasted prose.
  3. Invoke `verify_work` and `land_work` through their delivered tool surfaces
     and compare every model-facing authorization/recovery payload.
- Expected outcomes:
  - Only the exact current PASS/receipt candidate lands.
  - Every invalidation path blocks landing with accurate output.
  - Model-facing verification and landing results match the message contract.
- Verification:

```bash
uv run pytest -n0 \
  tests/tools/test_verify_work.py \
  tests/tools/test_land_work.py \
  tests/core/test_verification_contract.py \
  tests/core/test_verification_receipt.py \
  tests/core/test_workspace_verification.py \
  tests/core/test_agent_loop_verification_state.py \
  tests/core/workflows/test_then_verifier.py
```

- Evidence: `$EVIDENCE/IT-07/command.log`,
  `$EVIDENCE/IT-07/receipt.json`, `$EVIDENCE/IT-07/check-output.log`,
  `$EVIDENCE/IT-07/verifier-report.txt`, `$EVIDENCE/IT-07/git-state.json`,
  `$EVIDENCE/IT-07/landing.json`, `$EVIDENCE/IT-07/model-results.json`, and
  `$EVIDENCE/IT-07/junit.xml`.
- Covers: AC-1.2, AC-4.5, AC-4.6; MSG-08, MSG-14.

### IT-08: Memory lifecycle and cache stability

- Surface: `non_ui`
- Starting state: temporary global/project memory roots, deterministic local and
  model selectors, fixed conversation.
- Actions:
  1. Add, list, update, trash, restore, and scope memories.
  2. Exercise local-first and hybrid selection.
  3. Trigger local and model recall failures and assert the index-only/fallback
     result and warning contract.
  4. Invoke `manage_memory` through the delivered tool surface and compare
     model-facing success/failure/recovery payloads.
  5. Inject late memory across consecutive turns.
  6. Resume and re-read persisted state.
- Expected outcomes:
  - CRUD/scoping/selection behavior remains correct.
  - Persisted history is unchanged by backend-only injection.
  - Stable prompt prefix remains byte-equivalent.
  - Model-facing memory results and recall recovery hints match the message
    contract.
- Verification:

```bash
uv run pytest -n0 \
  tests/core/test_memory.py \
  tests/core/test_memory_local_selector.py \
  tests/core/test_memory_inject_mode.py \
  tests/core/test_memory_signals.py \
  tests/core/test_prompt_caching.py
```

- Evidence: `$EVIDENCE/IT-08/command.log`,
  `$EVIDENCE/IT-08/memory-manifest.json`,
  `$EVIDENCE/IT-08/backend-requests.json`,
  `$EVIDENCE/IT-08/session-diff.patch`,
  `$EVIDENCE/IT-08/model-results.json`, and `$EVIDENCE/IT-08/junit.xml`.
- Covers: AC-1.2, AC-5.1, AC-5.2, AC-6.5; MSG-09, MSG-14.

### IT-09: LSP semantic operations and lifecycle

- Surface: `non_ui`
- Starting state: deterministic fake language servers and multiple temporary
  workspace roots.
- Actions:
  1. Exercise Unicode positions, symbols, definitions, references, pagination,
     diagnostics, and call hierarchy.
  2. Exercise workspace routing, partial coverage, readiness, reload retirement,
     and security boundaries.
  3. Trigger missing-server/install guidance, server startup failure, and an
     operation error; assert the user-visible recovery text.
  4. Invoke the LSP tool for success, partial coverage, pagination, and error
     cases; compare every model-facing result and recovery hint.
- Expected outcomes:
  - Semantic results and summaries remain correct.
  - Route/server lifecycle does not leak or race.
  - Model-facing LSP summaries, partial-coverage notices, and recovery hints match
    the normalized baseline.
- Verification:

```bash
uv run pytest -n0 \
  tests/core/test_lsp.py \
  tests/core/test_lsp_next_tranche.py \
  tests/core/test_lsp_pagination.py \
  tests/core/test_lsp_positions.py \
  tests/core/test_lsp_diagnostics.py \
  tests/core/test_lsp_route_pool.py \
  tests/core/test_lsp_security.py \
  tests/core/test_lsp_isolation.py \
  tests/cli/test_lsp_status.py
```

- Evidence: `$EVIDENCE/IT-09/command.log`,
  `$EVIDENCE/IT-09/jsonrpc.jsonl`, `$EVIDENCE/IT-09/readiness.json`,
  `$EVIDENCE/IT-09/route-pool.json`, `$EVIDENCE/IT-09/results.json`,
  `$EVIDENCE/IT-09/model-results.json`, and `$EVIDENCE/IT-09/junit.xml`.
- Covers: AC-1.2, AC-5.3; MSG-10, MSG-14.

### IT-10: Spend, usage, pricing, caching, and retry conservation

- Surface: `non_ui`
- Starting state: temporary durable ledger, deterministic providers/retries,
  fixed pricing and session limits.
- Actions:
  1. Reserve/reconcile successful, missing-usage, error, and retry calls.
  2. Exercise concurrent children, cross-process attachment, queueing, and hard
     rejection.
  3. Reload, reset, resume, and tighten limits.
  4. Exercise cache-read/write pricing and provider-authoritative cost.
  5. Run the production paid-call boundary inventory and assert every call site
     is brokered or appears in the reviewed documented-exception set.
  6. Invoke spend status, rejection, reset, and usage surfaces through the CLI,
     ACP, and delivered tool context; compare user/model-visible payloads.
- Expected outcomes:
  - Leaf/parent/session totals conserve.
  - Rejection dispatches no backend call.
  - No orphan/negative reservation exists.
  - CLI and ACP usage surfaces agree.
  - Spend status, rejection, reset, and recovery messages match the normalized
    user/model baseline.
- Verification:

```bash
uv run pytest -n0 \
  tests/test_spend_broker.py \
  tests/test_session_spend.py \
  tests/test_usage.py \
  tests/test_usage_meter.py \
  tests/test_cache_aware_pricing.py \
  tests/test_prompt_estimator.py \
  tests/core/test_provider_retry_spend.py \
  tests/core/test_auxiliary_spend.py \
  tests/core/teams/test_spend_integration.py \
  tests/cli/test_spend_command.py \
  tests/acp/test_usage_update.py
```

- Evidence: `$EVIDENCE/IT-10/command.log`,
  `$EVIDENCE/IT-10/ledger.json`, `$EVIDENCE/IT-10/conservation.json`,
  `$EVIDENCE/IT-10/backend-calls.jsonl`, `$EVIDENCE/IT-10/usage.json`,
  `$EVIDENCE/IT-10/model-results.json`, and `$EVIDENCE/IT-10/junit.xml`.
- Covers: AC-1.2, AC-4.7, AC-4.8, AC-5.2, AC-6.5, AC-6.6; MSG-11, MSG-14.

### IT-11: Configuration, reload, and migration

- Surface: `non_ui`
- Starting state: deterministic user/project/harness/env layers, legacy session
  and provider fixtures, live config consumer.
- Actions:
  1. Load and merge all layers.
  2. Validate aliases/defaults/exclusions and invalid inputs.
  3. Reload into live consumers.
  4. Run migrations twice.
  5. When Iteration 8 deprecations are enabled, load each deprecated field and
     assert its migration and deprecation output.
- Expected outcomes:
  - Runtime config values, merge behavior, error meaning, and explicit user values
    match the baseline.
  - Repeated migration is idempotent.
  - Semantic parity and merge-policy inventories for both upstream models are
    complete.
- Verification:

```bash
uv run pytest -n0 \
  tests/core/test_config_orchestrator.py \
  tests/core/test_config_bus.py \
  tests/core/test_config_toml_end_to_end.py \
  tests/core/test_config_toml_merge.py \
  tests/core/test_vibe_config_schema.py \
  tests/core/test_provider_contract_migration.py \
  tests/session/test_session_migration.py \
  tests/core/test_user_config_layer.py \
  tests/core/test_project_config_layer.py
```

- Evidence: `$EVIDENCE/IT-11/command.log`,
  `$EVIDENCE/IT-11/config-inputs.json`, `$EVIDENCE/IT-11/config-output.json`,
  `$EVIDENCE/IT-11/validation.json`, `$EVIDENCE/IT-11/reload-events.jsonl`,
  `$EVIDENCE/IT-11/migrations.json`, and `$EVIDENCE/IT-11/junit.xml`.
- Covers: AC-1.2, AC-5.4, AC-5.5, AC-7.5; MSG-02, MSG-12.

### IT-12: Upstream divergence and merge rehearsal

- Surface: `non_ui`
- Starting state: disposable full-history repositories generated from fixed
  baseline/current trees.
- Actions:
  1. Test clean, deletion, rename, copy-delete, accepted, unexpected, and shallow
     cases.
  2. Compare ownership/hunk metrics.
  3. Rehearse merging the current upstream tree.
- Expected outcomes:
  - All structural divergence is detected independent of rename similarity.
  - Clean/accepted cases exit 0; unexpected cases exit nonzero.
  - No new extraction conflict is introduced.
- Verification:

```bash
uv run scripts/check_upstream_divergence.py
uv run pytest -n0 tests/test_upstream_divergence.py tests/test_iron_laws.py
```

- Evidence: `$EVIDENCE/IT-12/command.log`,
  `$EVIDENCE/IT-12/divergence.json`, `$EVIDENCE/IT-12/repos.json`,
  `$EVIDENCE/IT-12/fork-metrics.json`,
  `$EVIDENCE/IT-12/merge-rehearsal.txt`, and `$EVIDENCE/IT-12/junit.xml`.
- Covers: AC-1.1 through AC-1.3, AC-2.1 through AC-2.5.

### IT-13: Full repository quality and compatibility gate

- Surface: `non_ui`
- Starting state: clean candidate with all iteration edits complete.
- Actions:
  1. Before freezing, run the repository's fixing/formatting and pre-commit
     commands. Review and retain any resulting edits, then repeat until they are
     a no-op.
  2. Freeze the clean candidate.
  3. Run check-only lint/format, type, coverage, snapshots, iron laws, and
     divergence checks. A check that changes the frozen candidate invalidates
     it and returns the iteration to step 1.
  4. Re-run deterministic IT-01, IT-03, IT-04, and IT-12 fixtures twice from the
     same commit and compare normalized artifacts.
  5. Run a controlled failing evidence scenario and assert that it still emits a
     failed manifest entry, best-effort artifacts, and explicit notes for any
     deliberately missing artifact.
- Expected outcomes: all commands exit 0; coverage is at least 85%; no
  unexpected snapshot/message/warning/ratchet delta exists.
- Verification:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run pre-commit run --all-files

# Freeze the clean candidate before the check-only verification below.
uv run ruff check .
uv run ruff format --check .
uv run pyright
COVERAGE_FILE="$EVIDENCE/IT-13/.coverage" \
uv run pytest --ignore tests/snapshots \
  --cov \
  --cov-report=term-missing \
  --cov-report="xml:$EVIDENCE/IT-13/coverage.xml" \
  --junitxml="$EVIDENCE/IT-13/junit.xml"
uv run pytest -n0 tests/snapshots \
  --snapshot-report="$EVIDENCE/IT-13/snapshot-report.html"
uv run scripts/check_upstream_divergence.py
uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
uv run pytest -n0 tests/maintenance/test_evidence_contract.py
```

- Evidence: `$EVIDENCE/IT-13/command.log`, `$EVIDENCE/IT-13/junit.xml`,
  `$EVIDENCE/IT-13/coverage.xml`, `$EVIDENCE/IT-13/snapshot-success.json`,
  failure-only `$EVIDENCE/IT-13/snapshot-report.html`,
  `$EVIDENCE/IT-13/warnings.json`, `$EVIDENCE/IT-13/ratchets.json`,
  `$EVIDENCE/IT-13/reproducibility.json`,
  `$EVIDENCE/IT-13/failure-evidence.json`, and
  `$EVIDENCE/IT-13/verifier-report.txt`.
- Artifact ownership: the evidence runner owns command/result/reproducibility and
  controlled-failure artifacts; coverage owns its XML; pytest-textual-snapshot
  owns `snapshot-report.html` only when differences exist. On success, the
  I00-P99 snapshot adapter writes `snapshot-success.json` with command, exit,
  snapshot count, and absence of diffs. The maintenance ratchet collector owns
  `warnings.json` and `ratchets.json`; the lead stores the host verifier report
  after binding it to the frozen candidate. I00-P99 must name the snapshot
  adapter, ratchet collector command, and verifier handoff before it may become
  `ready`.
- Covers: AC-1.1 through AC-1.4, AC-2.4 through AC-2.6, AC-3.5, AC-3.6,
  AC-5.5, AC-7.1 through AC-7.4, AC-7.6.

### IT-14: Paired performance and hot-path invariants

- Surface: `non_ui`
- Starting state: clean baseline and candidate worktrees on the same quiet
  runner, identical Python/lock/fixtures/env, warmups complete.
- Actions:
  1. Ask the evidence collector to use a deliberately dirty disposable baseline
     worktree and assert that collection is rejected before a benchmark runs.
  2. Run the calibrated paired sampling protocol defined below.
  3. Run startup/import, agent loop, TUI CPU, memory, streaming, and index
     workloads.
  4. Compare exact invariants and baseline-derived non-inferiority bounds.
- Expected outcomes:
  - Exact invariants pass.
  - No measured regression exceeds the natural-noise envelope.
  - Raw samples and profiles remain available for review.
- Verification workloads:

```bash
uv run scripts/compare_performance.py \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --output-dir "$EVIDENCE/IT-14" \
  --seed 20260712 \
  --calibration-samples 30 \
  --pairs 20 \
  --profile-pairs 5

# The runner invokes the existing workloads below with identical locked
# environments. They remain individually runnable for diagnosis.
VIBE_TRACE_LOOP=0.02 VIBE_PROFILE=1 \
  uv run pytest -n0 tests/agent_loop/test_perf_benchmark.py -s

VIBE_CPU_PROFILE=1 VIBE_CPU_TURNS=80 VIBE_CPU_TOOL=cprofile \
  uv run pytest -n0 tests/perf/test_cpu_session.py -s

VIBE_CPU_PROFILE=1 VIBE_CPU_TURNS=80 VIBE_CPU_TOOL=cprofile \
  VIBE_CPU_TOOL_CALLS=1 \
  uv run pytest -n0 tests/perf/test_cpu_session.py -s

VIBE_CPU_PROFILE=1 VIBE_CPU_TURNS=80 VIBE_CPU_TOOL=cprofile \
  VIBE_CPU_FANOUT=24 \
  uv run pytest -n0 tests/perf/test_cpu_session.py -s

VIBE_MEM_PROFILE=1 VIBE_MEM_TURNS=300 \
  uv run pytest -n0 tests/perf/test_memory_session.py -s

VIBE_MEM_PROFILE=1 VIBE_MEM_TURNS=300 VIBE_MEM_TOOL_CALLS=1 \
  uv run pytest -n0 tests/perf/test_memory_session.py -s

VIBE_STREAM_PROFILE=1 VIBE_STREAM_SIZES=12,24,48 \
  uv run pytest -n0 tests/perf/test_stream_render.py -s

VIBE_INDEX_PROFILE=1 VIBE_INDEX_RUNS=5 VIBE_INDEX_TOOL=cprofile \
  uv run pytest -n0 tests/perf/test_index_profile.py -s

uv run pytest -n0 \
  tests/test_import_cost.py \
  tests/cli/textual_ui/test_lazy_startup_imports.py \
  tests/session/test_session_logger.py \
  tests/agent_loop/test_agent_turn_sampling.py \
  tests/test_resource_monitor.py \
  tests/cli/textual_ui/test_streaming_message_buffer.py \
  tests/cli/textual_ui/test_bash_output_buffer.py \
  tests/cli/textual_ui/test_streaming_markdown.py \
  tests/cli/test_ui_session_incremental_renderer.py \
  tests/core/test_loop_detection_middleware.py
```

- Evidence: `$EVIDENCE/IT-14/command.log`,
  `$EVIDENCE/IT-14/environment.json`, `$EVIDENCE/IT-14/raw-samples.json`,
  `$EVIDENCE/IT-14/comparison.json`, `$EVIDENCE/IT-14/profiles/`,
  `$EVIDENCE/IT-14/blockers.json`, and
  `$EVIDENCE/IT-14/dirty-baseline-rejection.json`.
- Covers: AC-1.1, AC-1.2, AC-1.5, AC-6.1 through AC-6.6.

### IT-15: Maintenance model-behavior evaluation

- Surface: `non_ui`
- Starting state: aligned versioned repository fixtures, task briefs, recipes,
  pricing, policy, model/provider revisions, random seeds, baseline/candidate
  refs, and a positive approved `VIBE_EVAL_MAX_COST_USD` hard broker cap.
- Actions:
  1. Run the planned trusted fixture runner once with both refs so it owns one
     global spend cap and emits aligned baseline/candidate datasets plus raw
     events.
  2. Run deterministic fixtures for every cleanup comparison; enable paid trials
     when real provider/model behavior is relevant.
  3. Exercise a cap-exhaustion fixture and assert that remaining trials block
     without exceeding the approved total.
  4. Compare the emitted datasets with the maintenance gate.
- Expected outcomes:
  - Safety, reliability, pass, utilization, attribution, cost, token, call, and
    duration non-regression gates pass.
  - Artifacts are aligned and reproducible.
- Verification:

```bash
uv run scripts/run_harness_evals.py \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --fixtures evals/fixtures/maintenance \
  --output-dir "$EVIDENCE/IT-15" \
  --trials 5 \
  --seed 20260712 \
  --max-total-cost-usd "$VIBE_EVAL_MAX_COST_USD"

uv run scripts/evaluate_harness.py \
  --baseline "$EVIDENCE/IT-15/baseline.json" \
  --candidate "$EVIDENCE/IT-15/candidate.json" \
  --output "$EVIDENCE/IT-15/comparison.json" \
  --maintenance-gate
```

The current evaluator compares prebuilt datasets but does not execute fixtures.
Iteration 0 records this limitation; Iteration 2 must supply the maintenance
mode, trusted fixture execution/ingestion, and cap-exhaustion coverage before
IT-15 can authorize a model-visible change. Until those planned artifacts exist,
IT-15 is an explicitly blocked scenario rather than a paper PASS.

- Evidence: `$EVIDENCE/IT-15/command.log`,
  `$EVIDENCE/IT-15/baseline.json`, `$EVIDENCE/IT-15/candidate.json`,
  `$EVIDENCE/IT-15/raw-events/`, `$EVIDENCE/IT-15/comparison.json`,
  `$EVIDENCE/IT-15/spend.json`, and `$EVIDENCE/IT-15/artifacts.json`.
- Covers: AC-6.7, AC-6.8, AC-7.5; MSG-14, MSG-15.

## Performance evidence contract

### Exact invariants

The following are pass/fail without timing tolerance:

- Guarded startup imports do not load forbidden optional dependency sets.
- Rejected spend calls produce zero backend dispatches.
- Every production paid-call boundary is brokered or explicitly documented.
- Cache-routing keys, cache-token normalization, deterministic pricing, and
  stable request bodies remain exact.
- Stable system/tool/history prefixes remain byte-identical for equivalent
  turns.
- Stream and Bash chunks produce equivalent content and retain one scheduled
  flush per frame.
- Static session context writes only when its fingerprint changes.
- `fsync`, session archiving, and resource-tree sampling do not run on the event
  loop.
- Workflow status, result order, retry, cancellation, and concurrency maxima
  remain exact.
- Hot-path work-count tests stay bounded by their configured window rather than
  transcript/session length.

### Measured metrics

Use a short, deterministic gate workload for statistical comparison and retain
the long profiler workloads for diagnosis and confirmation.

#### Calibration

1. On the designated quiet runner, execute 30 clean-baseline samples in fresh
   processes after one unmeasured warmup per process.
2. Use fixed seed `20260712` to form 15 randomized A/A pairs and record order,
   temperature/frequency state when available, and all raw observations.
3. For each lower-is-better metric, compute the baseline median and relative
   median absolute deviation (`rMAD = MAD / median`).
4. Define the practical non-inferiority margin as
   `max(2%, 3 * rMAD)`. If that value exceeds 5%, the metric is too noisy to
   gate; stabilize the runner or fixture instead of capping or widening it.

#### Candidate comparison

1. Execute 20 fresh-process baseline/candidate pairs. Randomize baseline-first
   versus candidate-first order within each pair using seed `20260712`; do not
   run all baseline samples before all candidate samples.
2. Analyze the paired log ratio `log(candidate / baseline)` for each metric.
3. Bootstrap the paired median log ratio with 10,000 resamples and fixed seed
   `20260712`; compute a one-sided 95% upper confidence bound.
4. Pass when the upper bound is no greater than `log(1 + margin)`.
5. Do not remove outliers after seeing candidate identity. Timeouts, crashes, and
   incomplete samples fail the comparison and remain in raw evidence.
6. Record host load and sample order so autocorrelation or runner interference
   can be diagnosed. A visibly nonstationary run is invalidated and rerun in
   full, never selectively trimmed.

#### Long profiler confirmation

Run the full CPU, memory, stream, and index profiler workloads in five randomized
baseline/candidate pairs. These profiles are diagnostic rather than the sole
statistical gate. A new dominant hotspot, structural-invariant failure, or median
regression beyond the calibrated margin blocks the iteration pending review.

Apply this to:

- Cold help and initialization milliseconds.
- Agent turn and fanout milliseconds.
- CPU milliseconds per TUI turn.
- Retained KB per turn and widget-growth slope.
- Streaming total CPU, quarter ratio, and reply-size scaling.
- Index rebuild milliseconds per run.

Wall time alone is insufficient. Preserve structural performance indicators:

- Refreshes per chunk/frame.
- Writes and durable syncs per session round.
- Canonicalization/scanning calls per configured trailing window.
- Concurrent readers and serialized writers.
- Imported module sets.
- Event-loop blocker count and total blocked time.

### Historical context, not gates

The following historical improvements explain what the campaign must protect,
but they are not machine-independent absolute thresholds:

- `vibe --help`: approximately 340 ms to 70 ms.
- Headless initialization: approximately 0.62 s to 0.49 s.
- 48 KB streaming render: approximately 12.0 s to 6.3 s CPU.
- 1,000-token streamed reply: approximately 1,570 ms to 68 ms.
- 3.9 MB Bash output: approximately 4,226 ms to 540 ms and 966 refreshes to 2.
- Steady session persistence: approximately 77.2 KB to 4.0 KB written per round
  and 3.0 ms to 1.1 ms.
- Loop-detection scanning becoming independent of total history length.

Iteration 0 measures the current clean baseline; historical values are supporting
context only.

## Scenario-to-iteration matrix

| Iteration | Required scenarios |
|---|---|
| 0 | IT-01 through IT-14 in baseline/characterization mode; later-iteration gaps and blocked IT-15 recorded explicitly |
| 1 | IT-01, IT-03, IT-11, IT-12, IT-13 |
| 2 | IT-12, IT-13, IT-14; IT-15 maintenance-mode negative/positive fixtures |
| 3 | IT-05, IT-06, IT-10, IT-13, IT-14 |
| 4 | IT-01, IT-03, IT-10, IT-11, IT-13, IT-14 |
| 5 | IT-03, IT-04, IT-05, IT-06, IT-07, IT-08, IT-10, IT-13, IT-14 |
| 6 | IT-02, IT-13, IT-14 |
| 7A | IT-05, IT-06, IT-10, IT-13, IT-14 |
| 7B | IT-05, IT-06, IT-07, IT-10, IT-13, IT-14 |
| 7C | IT-03, IT-04, IT-09, IT-13, IT-14 |
| 8 workflow failure policy | IT-05, IT-13, IT-14, IT-15 |
| 8 config retirement | IT-01, IT-03, IT-11, IT-13, IT-14, IT-15 |
| 8 tool-surface change | Exact affected tool scenarios among IT-04 through IT-10, plus IT-13, IT-14, IT-15 |
| 9 | IT-01 through IT-15 |

## Semantic review scenarios

Some process criteria require reviewable judgment rather than runtime execution.

### SR-01: Change-boundary review

- Question: Does the iteration mix mechanical movement, compatibility changes,
  or optimization?
- Expected answer: No. The diff and commit series contain exactly one category.
- Evidence: PR description, diff, commits, acceptance declaration, rollback plan.
- Proves: AC-7.4.

### SR-02: Documentation truth review

- Question: Do README, OpenWiki, builtin skills, config reference, and release
  notes describe only implemented behavior and valid defaults?
- Expected answer: Yes. Every referenced command/config/tool is present and
  consumable through its real registry/schema.
- Evidence: changed docs, schema/registry inventory report, doc sync tests.
- Proves: AC-7.6.

### SR-03: Compatibility rollout boundary review

- Question: Does each default switch, field removal, or tool-surface change have
  tested migration behavior, complete message/documentation updates, and a
  commit/PR boundary that can be reverted without removing the preparatory typed
  internals or migration machinery?
- Expected answer: Yes. Preparation, deprecation/migration, and default
  switch/removal are separate revertible changes.
- Evidence: iteration diff and commits, migration tests, message delta,
  documentation/release notes, rollback command and affected-state inventory.
- Proves: AC-7.5.

### SR-04: Localized upstream seam review

- Question: For every subsystem extracted in this iteration, does the hunk
  ownership report show that upstream-owned files retain only the approved
  construction/invocation hooks and compatibility forwards, without moved or
  reordered upstream implementation?
- Expected answer: Yes. Every remaining hunk maps to a named seam in the
  iteration's structural allowlist, and the total hotspot hunk count does not
  increase.
- Evidence: `$EVIDENCE/SR-04/hunk-ownership.json`,
  `$EVIDENCE/SR-04/structural-allowlist.json`, candidate diff, upstream baseline
  diff, and merge-rehearsal report.
- Proves: AC-2.5.

## Crosscheck

Before an iteration is declared complete:

1. Every required scenario exercises a delivered entry point or real subsystem
   boundary, not only a private helper.
2. Every scenario is automatable, bounded, independent, and creates its own
   deterministic starting state.
3. Every required scenario has at least one readable evidence artifact.
4. UI scenarios include PNG/JPG evidence of key states in addition to snapshots.
5. Every acceptance criterion mapped to the iteration has a passing scenario or
   semantic review.
6. Every message group reachable from the iteration has a baseline comparison or
   approved change record.
7. At least one real delivery-form scenario runs for each affected surface:
   installed TUI, `vibe -p`, or `vibe-acp`.
8. The performance comparison uses the original baseline artifacts and clean
   baseline commit, not a regenerated candidate baseline.
9. Fork metrics show no new missing path and no unexplained increase in upstream
   overlap.
10. The candidate is frozen before verifier execution and remains unchanged
    until the verdict is recorded.
11. The evidence manifest contains every required scenario ID, status, command,
    artifacts, and explicit missing-artifact notes.
12. Any remaining debt or blocked scenario appears in the final report; it is not
    hidden by updating a threshold or snapshot.

## Prohibited sequencing

- Decomposition and workflow failure-semantics changes in the same PR.
- Configuration canonicalization and field removal in the same PR.
- Hot AgentLoop or streaming extraction before Iterations 0-2 are complete.
- Warning-as-error before existing warnings are classified and owned.
- Upstream sync during an extraction.
- Snapshot or performance-baseline regeneration to make a candidate green.
- Deleting upstream code or compatibility paths to improve a size metric.
- Tool-surface consolidation without usage evidence and model-visible evaluation.
- Paid trials without a hard broker cap.
- Final verification while the candidate is still changing.

## Completion report

The final campaign report must state:

- Baseline, final, and upstream SHAs.
- Restored and accepted upstream paths.
- Modified-upstream-path and hotspot-hunk changes.
- Fork-owned size/complexity and warning changes.
- Functional, snapshot, message, performance, spend, and evaluation results.
- Compatibility changes and their migration/rollback status.
- Documentation/schema/registry synchronization status.
- Remaining accepted debt and the reason it was not addressed.
- Final verifier verdict and exact evidence manifest path.
