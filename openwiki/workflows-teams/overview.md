# Workflows & Teams

Workflows and teams are two distinct multi-agent coordination mechanisms. Workflows orchestrate parallel agents as asyncio tasks within a single session. Teams coordinate multiple independent OS processes.

## Workflows

**Source**: `vibe/core/workflows/`

Workflows are Python scripts that orchestrate parallel agents for codebase audits, migrations, and cross-checked research. They run in the background as asyncio tasks, so the session stays responsive while agents work.

### Key Files

| File | Purpose |
|---|---|
| `runtime.py` (81 KB) | Main execution engine — async agent spawning, parallel/pipeline orchestration, JSON extraction, schema retry |
| `models.py` | Pydantic data models for workflow results |
| `schema.py` | Custom JSON-schema validator (no external dependency) |
| `budget.py` | Token budget tracking with reserve/reconcile pattern |
| `manager.py` | Discovers/saves workflow scripts from project + user dirs |
| `contract.py` | Post-execution contract verification |
| `security.py` | AST validation + sandbox namespace builder for workflow scripts |
| `bundled/` | Built-in workflow scripts |

### Workflow Scripts

A workflow script is a `.py` file with an `async def main()` function. Optional YAML frontmatter (`name:`, `description:`) precedes the Python source. The runtime injects these functions:

- `agent(prompt, *, agent="explore", model=None, label=None, phase=None, schema=None, budget_estimate=None, isolation=None, strip_unknown=True, contract=None, citations=None, then=None)` — spawn a subagent; `citations` verifies returned file/line/snippet evidence
  - Profiles: `explore`, `research`, `reviewer`, `debugger`, `planner`, `security`, `editor`, `grunt`, `verifier`, and `worker`; write-capable profiles require `isolation="worktree"`
  - `isolation="worktree"` runs it as a `vibe -p` subprocess in a fresh git worktree
- `parallel(*items, max_concurrency=None)` — run coroutines or zero-argument thunks concurrently and return results in argument order
- `pipeline(items, *stages, max_concurrency=None)` — run each item through its stages independently, with no cross-item barrier
- `recipe(name, *, items=None, find_agent="explore", verify_agent="reviewer", synth=None, max_concurrency=None)` — run the built-in `find_verify` or `find_verify_synth` pattern
- `phase(name)` — declare a phase for progress tracking
- `log(msg)` — log a progress message
- `post_message(channel, message)` / `fetch_messages(channel)` — exchange bounded in-run messages
- `flatten(items)` — flatten one list level without iterating string-like atoms
- `dedup_by(items, key)` / `merge_by(items, key, merge)` — deduplicate or fold grouped results
- `team_task(description, dependencies=None)` — enqueue work on the active process team, if one exists
- `budget` — token budget object with `.total` and `.remaining()`
- `workflow(name, args=None)` — run another discovered workflow inline (one level deep)
- `args` — structured input from the invocation command

Ordinary agent/parallel/pipeline failures degrade the affected result to `None`;
strategy-bound expected-lane failures and hard agent/budget/spend ceilings
propagate instead. One workflow reserves both host agent slots. Nested agents share that workflow's
budget, total-agent ceiling, and maximum concurrency; they do not create more
host lanes. Tests, builds, and package/dependency commands remain explicit-user
gates even for isolated auto-approved workers.

### Security Model

Scripts are validated via AST before execution (`security.py`):
- Unsafe imports blocked
- Dangerous calls blocked
- Dunder access blocked (`__self__`, `__closure__`, etc.)
- `str.format` blocked (prevents attribute traversal)
- Safelisted builtins only
- **Defense-in-depth, not a hard boundary** — scripts still `exec` in-process. The real boundary is the `launch_workflow` ASK gate + `disable_workflows` config. Treat model-authored scripts as untrusted.

### Budget System

**Source**: `vibe/core/workflows/budget.py`

`Budget` is a mutable token cap with reserve/reconcile:
- `reserve(estimate)` → `Reservation`
- `reconcile(reservation, actual_in, actual_out)` moves reserved→spent
- `restore_spent()` for resume-from-snapshot
- Raises `BudgetExhausted`
- `ReadOnlyBudget` — sandbox-safe proxy that stores only bound accessor callables, prevents workflow scripts from mutating spend via `__self__`/`__closure__` paths

In-process workflow agents also reserve under the parent session spend broker.
Exhausting either the workflow-local token budget or the shared session envelope
terminates the run as `WorkflowStatus.BLOCKED`; direct, schema, isolated,
parallel, and pipeline paths preserve that status instead of degrading it to a
`None` result. Saturating a positive shared concurrency limit queues the paid
call until capacity is available instead of blocking the run. Ordinary child
exceptions retain the existing recoverable `None`/`COMPLETED_WITH_FAILURES`
behavior.

### Schema Validation

**Source**: `vibe/core/workflows/schema.py`

Custom JSON-schema validator (no external dependency):
- `_validate_object/array/string/number/boolean/integer` — recursive validators
- `validate_against_schema()`, `build_response_format()`, `build_prompt_fallback()`
- `strip_unknown_properties()` — cleans LLM responses before validation
- `SchemaValidationFailure` — falsy dict subclass returned (not raised) when an agent exhausts schema retries in non-strict mode

Local conservative JSON repair runs before any paid retry. A schema correction
continues in the existing worker conversation with an exact diagnostic, so the
repository investigation is not repeated. The repair controller stops after
semantic no-progress/oscillation. An explicitly configured formatter alias may
receive one no-tools, 512-token syntax repair; an explicitly configured semantic
alias may receive one bounded no-tools escalation only after repeated semantic
failure.

### Bundled Workflows

**Source**: `vibe/core/workflows/bundled/`

| Command | Purpose |
|---|---|
| `/deep-research <question>` | Fans out web searches across angles, fetches and cross-checks sources, synthesizes a cited report |
| `/adversarial-review` | Adversarially review the current diff via diverse-lens finders, independent refute-verify, and gated synthesis |
| `/verify-contract` | Run a code task in an isolated worktree and gate delivery on a code-artifact contract (files must exist, match grep/size rules, pass tests) |
| `/security-fix-verify` | Pre-merge gate for a security FIX branch: refute-only per-finding panel, regression hunt, runtime gaps hard-block; emits a review packet, never pushes |

### Workflow Management

- Discovered from `workflow_paths` config, `.vibe/workflows/`, `~/.vibe/workflows/`, and bundled workflows
- Registered as `/<name>` slash commands
- `/workflows` — progress view showing all runs with status, agents, tokens, elapsed
- `/workflows stop <id|all>` — stop one or all runs
- `/workflows snapshot <id>` — show cached results for a run
- `/workflows resume <id>` — resume an unbound persisted snapshot
- Result reuse is disabled unless the trusted host supplies a SHA-256 fingerprint
  covering the complete dependency closure, including ignored/external reads and
  resolved instructions. The fingerprint is part of the cache identity alongside
  repository state, model/provider and routing settings, tool schemas/policy,
  prompt, schema/contract, and harness version. Only known in-process read-only
  profiles are eligible; resumed runs need the same fingerprint. Strategy-bound
  snapshots are recorded but cannot be resumed until exact lane authorization can
  be restored; start a new strategy-bound run instead.

### Effort Modes

- **normal** (default): work turn-by-turn
- **le-chaton**: max thinking + adaptive orchestration. Every primary host model request uses effective max thinking, including after a model switch or failover. The hands-on host keeps its normal tools and records an observed-scope `work_strategy` route:
  - `direct` for localized or sequentially coupled work
  - `task` for one or two productive independent lanes
  - `workflow` for exactly two staged or adversarial evidence lanes
  - `team` for long-running coordination
- Select via `/effort` or set `effort_mode = "le-chaton"` in config.toml
- A prompt containing `le chaton` or `lechaton` acquires a non-persistent Le Chaton lease. The lease survives matching asynchronous task, workflow, or team result delivery and restores the saved mode only after the host has acted on that result.
- The runtime gates substantive mutation and finalization until strategy and productive-delegation debt are satisfied. Each strategy starts at most two agent-owned lanes, and later expansion requires terminal evidence of a concrete gap. Risk cannot be downgraded during the active strategy lifecycle. Debt survives continuation turns and is correlated to immutable task, workflow, or team launch IDs until the matching terminal result arrives; superseded launches cannot satisfy or poison their replacement. Preflight reservations prevent concurrent tool calls from launching the same declared lane twice, and explicit task stops produce terminal failure receipts. A narrowly localized first edit/write may infer a bounded direct route; path or mutation expansion and delegation failure force scope reassessment. Verifier calls are completion checks, not productive delegation. Interactive teammate output is bounded, persisted when large, staged into the host context, and wakes an idle host.
- `disable_workflows = true` hides raw `launch_workflow` without disabling Le Chaton. Workflow routing falls back to `task` when available, or to an honest capability-constrained direct route when appropriate.
- Raw workflow scripts remain an advanced escape hatch; load the `workflow-authoring` skill before authoring one.

## Teams

**Source**: `vibe/core/teams/`

Agent teams coordinate multiple independent Vibe instances. Unlike subagents (in-memory, same session) or workflows (asyncio tasks, same event loop), teammates are **separate OS processes** — each is a full `vibe -p` invocation.

### Key Files

| File | Purpose |
|---|---|
| `__init__.py` | Public API: `Mailbox`, `Message`, `Task`, `TaskStatus`, `TaskStore`, `TeamConfig`, `TeamManager`, `TeamMember` |
| `manager.py` (13 KB) | Spawns/manages teammate subprocesses, persists team config |
| `mailbox.py` | File-based per-recipient inbox with locking |
| `task_store.py` | File-based task queue with dependency tracking + atomic claims |
| `models.py` | Pydantic models for all team entities |
| `_escalate.py` | Escalation logic |
| `errors.py` | `TeamStorageBusyError` |

### Shared State

Teammates coordinate via file-backed shared state with file locking:

- **TaskStore** (`task_store.py`): persists all tasks to single `tasks.json` under `FileLock`
  - `add_task()` — accepts a legacy description or structured `TaskBrief`, auto-generates `task-N` id, supports `dependencies[]`
  - `claim_task()` — **atomic read-modify-write** — re-reads under lock, checks `PENDING` + `_dependencies_met()`, sets `IN_PROGRESS` + assignee
  - `complete_task()` — persists an explicit `TaskOutcome`; `RETRYABLE` returns the task to `PENDING`, while other outcomes end the lifecycle
  - `_dependencies_met()` — requires every dependency outcome to be `SUCCEEDED`, not merely terminal

- **Mailbox** (`mailbox.py`): file-per-message, per-recipient inbox under `team_dir/mailbox/`
  - `send()` — writes `Message` JSON under lock (`FileLock`, 5s timeout → `TeamStorageBusyError`)
  - `_safe_name()` — validates member names, prevents path traversal
  - Messages sorted by timestamp + id (uuid filenames don't sort lexically)

- **TeamConfig** (`models.py`): team metadata — `team_name`, `members[]`, `team_dir`, `lead_session_id`

### Structured Task Protocol

Protocol v2 team tasks persist a `TaskBrief` with an objective plus structured
inputs, path scope, acceptance checks, optional budget/deadline, and tool
manifest identity. The host binds these values to a trusted recipe before
dispatch. Check IDs resolve only to prebound argv checks, per-task limits become
a child spend envelope, and the manifest becomes a runtime tool allowlist. The
worker cannot self-complete a structured task or read/write host-owned team
metadata or harness control-plane paths. Only the harness can execute the
selected checks and persist success. A failed check produces one bounded repair
turn with exact diagnostics in the same fresh worker conversation, followed by
a full selected-check rerun before that loop closes.

Terminal `TaskOutcome` values are `SUCCEEDED`, `FAILED`, `BLOCKED`, or
`RETRYABLE`, with evidence, diagnostics, changed paths, receipt ID, remaining
work, and manifest identity. Legacy description/result records remain loadable
through the protocol-v1 adapter.

Lifecycle (`PENDING`, `IN_PROGRESS`, terminal) is separate from outcome. A
retryable result is atomically requeued and does not fire a terminal completion
hook; downstream tasks unlock only after a succeeded outcome.

### Message Types

`MessageKind` (StrEnum): `TEXT`, `PERMISSION_REQUEST`, `PERMISSION_RESPONSE`, `PLAN_APPROVAL`, `SHUTDOWN` — structured typed messages for teammate↔lead communication.

### Team Manager

**Source**: `vibe/core/teams/manager.py`

`TeamManager`:
- Created with `lead_session_id`, auto-generates team name (`team-{hex}`)
- Creates `VIBE_HOME/teams/<name>/` directory
- Lazily initializes `TaskStore` and `Mailbox` via properties
- Tracks `_teammate_tasks` (asyncio tasks) and `_teammate_procs` (subprocess processes)
- Integrates with `HooksManager` for hook events
- Gives each teammate process a distinct Agent scope under one TEAM spend
  envelope
- Creates a fresh `AgentLoop`, transcript, tools, verification state, and child
  budget for every claimed worker task while retaining the shared process ledger

### Team Commands

- `/team spawn <name> <prompt>` — spawn a teammate as a separate process
- `/team list` — show teammates with name, status, PID
- `/team stop <name|all>` — stop one or all teammates
- `/team cleanup` — remove team directory

### Hook Integration

- `TeammateIdle` — teammate idle event
- `TaskCreated` / `TaskCompleted` — lead-initiated `/team task add|done` only (teammate writes don't fire lead-side hooks)
- Subagents inherit the parent's hook config so policies apply transitively

### Where to Start When Changing Teams

- **Team behavior**: `vibe/core/teams/manager.py` → `vibe/core/teams/models.py`
- **Task queue**: `vibe/core/teams/task_store.py`
- **Messaging**: `vibe/core/teams/mailbox.py`
- **Tests**: `tests/core/teams/test_teams.py`

## Comparison: Subagents vs Workflows vs Teams

| Feature | Subagents (task tool) | Workflows | Teams |
|---|---|---|---|
| **Process** | In-process (asyncio) | In-process (asyncio tasks) | Separate OS processes |
| **Session** | Same session | Same session | Independent sessions |
| **Coordination** | Tool call + result | Scripted orchestration | File-backed shared state |
| **Use case** | Delegate a focused task | Parallel multi-agent audits | Long-running multi-agent collaboration |
| **Isolation** | Worktree (write-capable) | Worktree (worker profile) | Full process isolation |
| **Context** | Fresh context per subagent | Fresh context per agent | Fresh loop per claimed worker task |

## Tests

- `tests/core/workflows/` — workflow runtime, schema, budget tests
- `tests/core/teams/test_teams.py` — team coordination tests
- `tests/tools/test_workflow_*` — workflow tool tests
- `tests/tools/test_team_spawn.py` — team spawn tests
