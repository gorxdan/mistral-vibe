# Architecture Overview

The repository has a single shared runtime core (`vibe/core/`) surfaced through three front-ends: the interactive TUI (`vibe`), programmatic mode (`vibe -p`), and the ACP server (`vibe-acp`). All three rely on the same `AgentLoop`, `ToolManager`, and `VibeConfig` core.

## System Shape

```
┌──────────────────────────────────────────────┐
│              ENTRY POINTS                     │
│  vibe (CLI)         vibe-acp (ACP)             │
│  entrypoint.py      entrypoint.py              │
│       │                  │                     │
│  ┌────▼────┐    ┌────────▼────────┐           │
│  │Textual   │    │ ACP Server      │           │
│  │TUI App   │    │ acp_agent_loop  │           │
│  │+ widgets │    │ + tools/        │           │
│  └────┬────┘    └────────┬────────┘           │
│       │    programmatic   │                     │
│       │     (programmatic │                     │
│       │      .py)         │                     │
│       ▼                  ▼                     │
│  ┌─────────────────────────────────┐           │
│  │  vibe.core.agent_loop.AgentLoop  │ ← shared │
│  │  + ToolManager + VibeConfig      │           │
│  └─────────────────────────────────┘           │
└──────────────────────────────────────────────┘
```

## AgentLoop: The Central Orchestrator

**Source**: `vibe/core/agent_loop.py`

`AgentLoop` (line 457) is the heart of the engine. It uses **multiple inheritance with mixin composition** to separate concerns:

```python
AgentLoop(
    AgentLoopMemoryMixin,
    AgentLoopOrchestrationMixin,
    AgentLoopVerificationMixin,
    AgentLoopFailoverMixin,
    AgentLoopSafetyJudgeMixin,
    AgentLoopHooksMixin,
)
```

**`AgentLoopParams`** (line 439) is a dataclass holding loop-level configuration: `max_turns`, `max_price`, `max_session_tokens`, `enable_streaming`, `is_subagent`, `headless`, `permission_store`, `mcp_registry`, `cache_store`, etc.

### Initialization Flow

1. `_init_base_state` — stores config, cache store, headless flag
2. `_init_registries` — sets up permission store, MCP registry, tool manager, agent manager, skill manager, turn/price/token limits
3. `_init_backend` — creates the LLM backend via `create_backend()` factory
4. `_init_session_identity` / `_init_messages` / `_init_session_state` — session ID, message list, telemetry, shared spend adapter
5. `_init_hooks` — hooks manager
6. `_init_rewind` — file snapshot/rewind system
7. Optionally defers heavy init (MCP integration, system prompt assembly) to a background thread

### Error Classification

`_raise_for_backend_error` (lines 284–305) maps raw backend exceptions into typed domain errors: `RateLimitError`, `ContextTooLongError`, `ResponseTooLongError`, `ContentFilterError`, `TransportError`, `ServerError`, `RefusalError`. These drive failover and compaction decisions.

## Mixin Subsystems

### AgentLoopHooksMixin
**Source**: `vibe/core/agent_loop_hooks.py`

Provides the hook lifecycle: `before_tool`, `after_tool`, `post_agent_turn`. Integrates with `HooksManager` and dispatches events like `BeforeToolInvocation`, `AfterToolInvocation`, `HookToolDenial`, `HookToolInputRewrite`, `UserPromptSubmitInvocation`, `SessionStartInvocation`, `PreCompactInvocation`. Hooks can deny tools, rewrite tool inputs, and inject text replacements.

### AgentLoopFailoverMixin
**Source**: `vibe/core/agent_loop_failover.py`

Handles model failover on rate-limit/overload/content-filter errors. Key methods:
- `_switch_to_fallback_model()` — iterates `config.fallback_models`, skipping already-tried aliases
- `_activate_model(model)` — creates a new backend via `create_backend()`, updates pricing and compaction threshold
- `_switch_to_chosen_model(alias)` — user-initiated model switch via `/model`

### AgentLoopSafetyJudgeMixin
**Source**: `vibe/core/agent_loop_safety_judge.py`

Fork-only LLM safety judge that pre-screens ASK-gated tool calls. Uses a verdict cache (`OrderedDict` keyed by tool name + args hash + transcript hash), truncation guards (args capped at `JUDGE_ARGS_LIMIT=4000` chars), and a transcript window (last 4 turns, 2000 chars). Force-defers to user when args are truncated AND a risk flag is present — never auto-approves on a blind prefix.
The session reuses the same judge instance while its model/provider/config,
timeout, and spend identity remain unchanged. If the provider rejects
`temperature`, the judge retries without it once and remembers the omission for
subsequent calls on that instance. A change to any part of that full identity
clears the verdict cache, including same-alias provider, model-config, or timeout
changes.

### AgentLoopMemoryMixin
**Source**: `vibe/core/agent_loop_memory.py`

Durable memory subsystem: recall (selection + prefetch), extraction, consolidation, verification. Uses async background tasks. Collaborates with `MemoryStore`, `MemorySelector`, `MemoryExtractor`, `MemoryConsolidator`, `MemoryVerifier` (in `vibe/core/memory/`).

### AgentLoopOrchestrationMixin
**Source**: `vibe/core/agent_loop_orchestration.py`

Owns root orchestration state, structured task context, and the host-side
coordination helpers used by Task and workflow execution.

### AgentLoopVerificationMixin
**Source**: `vibe/core/agent_loop_verification.py`

Installs the managed runtime capability ceiling, validates an optional frozen
execution topology before the first root turn, checks receipt freshness, and
replaces unsupported terminal completion claims with host verification status.
Intermediate tool turns suppress final-looking status banners, while typed or
ordinary tool-free status handoffs remain visibly quoted as untrusted context.

## AgentLoopLimits

**Source**: `vibe/core/agent_loop_limits.py`

Pure constants (no class coupling):
- `MAX_TOOL_RESULT_CHARS = 100_000` — per-result cap (~25k tokens)
- `TOOL_RESULT_PREVIEW_CHARS = 12_000` — inline preview size (head 75% + tail 25%)
- `AGGREGATE_TOOL_RESULT_CHARS = 200_000` — total cap for parallel tool calls
- `tool_result_hard_cap(threshold_tokens)` — scales cap to model's context budget
- `MAX_CONCURRENT_SUBAGENTS = 2`

## Tool Call Scheduling

**Source**: `vibe/core/agent_loop_tool_scheduler.py`

After the host-only `work_strategy` pre-batch barrier, tool calls are partitioned
in model-emitted order. Consecutive read-only calls form a concurrent wave. Each
non-read-only call is a singleton mutation barrier: prior reads finish before it
starts, and later calls wait for it to finish. Unknown and third-party tools use
the conservative non-read-only default. Classification is computed once when
the wave is built and passed unchanged to execution. All call events are
announced before execution. Events and results inside a read wave are then
forwarded as produced; the next wave waits until every invocation in the current
wave fully finalizes. An unexpected executor failure emits a terminal failure
for its call if needed, drains that wave, synthesizes failures for all announced
nonterminal calls, aborts later waves, and propagates.

## Data Flow: A Single Turn

```
User Input
    │
    ▼
AgentLoop (agent_loop.py)
    │
    ├── MiddlewarePipeline (middleware.py)
    │   └── context shaping, compaction, token/price limits, loop detection
    │
    ├── System Prompt Assembly (system_prompt.py)
    │   └── ProjectContextProvider → git status, worktree, tools, skills, agents
    │
    ├── LLM Backend (llm/backend/factory.py → MistralBackend | GenericBackend)
    │   └── Spend reservation → CompletionRequest → LLMChunk → reconciliation
    │
    ├── Tool Execution Flow:
    │   ├── Canonical managed catalog / normal discovery → ToolManager
    │   ├── Parse LLM response → ParsedToolCall → ResolvedToolCall
    │   ├── Ordered scheduler → bind classification; read waves + mutation barriers
    │   ├── Hooks — before_tool rewrite/deny; validate bound classification
    │   ├── Tool permission resolution — hard denials + explicit-user gates
    │   ├── Permission Store — ordinary ASK-pattern coverage
    │   ├── Safety Judge — pre-screen remaining eligible ASK calls
    │   ├── Human / ordinary auto-approve gate
    │   ├── InvokeContext → BaseTool.invoke() → ToolResultEvent
    │   └── Result size enforcement — preview + persist to disk
    │
    ├── Failover — on rate-limit/overload, switch to fallback model
    │
    ├── Memory — recall, extract, consolidate, verify (async background)
    │
    ├── Compaction (compaction.py) — auto-compact when context exceeds threshold
    │
    └── Loop back to LLM with tool results appended to MessageList
```

## Middleware Pipeline

**Source**: `vibe/core/middleware.py`

The middleware pipeline applies transformations to the conversation state before each LLM call. Key middlewares include:
- **Context shaping** — adjusts what goes into the prompt based on model tier
- **Compaction** — auto-compact when token count exceeds `auto_compact_threshold` (see `vibe/core/compaction.py`)
- **Token/price limits** — enforce session-level budgets
- **Loop detection** — detect and break repetitive agent loops
- **Harness capability breaker** — stop after three consecutive failures in one protected capability class

The design doc `docs/design/compaction.md` describes the multi-stage shaper pipeline (snip + microcompact) in detail.

## Spend Admission

**Source**: `vibe/core/usage/_broker.py`, `_ledger.py`, `_session.py`

Primary, compaction, task/workflow, team, memory-helper, safety-judge, narration,
repair, and verification calls share a durable hierarchical ledger
(`session -> workflow/team -> agent -> call`). Admission is reserved before
backend dispatch under a file lock, so sibling agents cannot race past the
parent cap. Provider usage reconciles the reservation; errors or missing usage
retain the estimate. Session resume rebinds the adapter to the resumed ledger,
and active calls renew their leases. Isolated children receive a versioned
`VIBE_SPEND_CONTEXT` that can attach only to an existing host-created Agent
scope. Lease replay releases provably undispatched calls and conservatively
charges dispatched calls until exact usage arrives.

Generic and Mistral provider retries authorize every redispatch against the
original call reservation. Each authorization conservatively charges another
reservation estimate across the token and USD scope hierarchy; the final
attempt reconciles exact usage when available. Retry count and policy/budget
rejection are durable;
streaming retries stop after the first yielded chunk. MCP sampling remains the
documented model-call boundary outside this broker; non-token-priced
text-to-speech and real-time transcription are also outside the token ledger.
Mistral's model-backed web search remains an explicit unrouted paid boundary.

## Bound Task Contracts

**Source**: `vibe/core/tasking/_policy.py`, `_process_context.py`,
`vibe/core/tools/_task_manifest.py`

A structured `TaskBrief` is frozen, then the host binds it to the session's
immutable trusted recipe. Acceptance values are check IDs, never commands; the
host resolves them to prebound argv checks. Task-bound `ToolManager` instances
import canonical builtins only before applying the allowlist to lookup, search,
and pinning. Edit/write paths are
checked after hook and user modification. Harness control-plane paths
(`.vibe/**`, `.agents/**`, `.git/**`, and every `AGENTS.md`) remain host-owned.
Callers pass the brief as an object. Serialized JSON strings remain legacy
free-form tasks and receive no structured contract authority. Structured
verifier work is pinned to `verify@1`, uses the strict terminal `VERDICT`
protocol, and cannot be paired with a write-capable manifest; other read-only
profiles likewise reject edit/write manifests.

An execution topology adds a stronger outer ceiling. Active roots have at most
`bash`, `edit`, `glob`, `grep`, `read`, `skill`, `task`, `todo`, and
`write_file`; verification roots have at most `glob`, `grep`, `read`,
`skill`, `task`, and `verify_work`. Project and plugin tools, MCP/connectors,
workflows, teams, web tools, `tool_search`, and `land_work` are outside this
catalog. Managed Task accepts only effective read-only built-in reviewer or
verifier profiles. They inherit the frozen recipe and receive at most `bash`,
`glob`, `grep`, `read`, and `skill`, intersected with a structured
manifest when present.

Write-capable isolated tasks return an undelivered worktree. The host inspects
committed, staged, working, deleted, renamed, and untracked paths, runs only the
selected trusted checks with `shell=False`, rechecks paths after those checks,
and fast-forwards the candidate only after all gates pass. `VIBE_TASK_CONTEXT`
binds the same frozen brief inside the subprocess; it cannot supply new checks
or a wider manifest.

## Verification Receipts

**Source**: `vibe/core/_verification_receipt.py`,
`vibe/core/_verification_runner.py`, `vibe/core/verification_state.py`

Trusted local checks create immutable receipts bound to the task brief,
acceptance contract, repository identity and state, configuration, check set,
and full-output artifact hashes. A configured `trusted_verification_recipe` may
come from host-controlled user, `VIBE_` environment, or programmatic config;
project TOML entries are removed case-insensitively. The recipe forces the
verification subsystem on and is frozen into `VerificationState` when
`AgentLoop` starts. Managed reviewer and verifier children inherit the frozen
value. After a current verifier PASS, no-argument `verify_work` executes only
that prebound plan against the active candidate and current main HEAD.
`land_work` revalidates its receipt, merges, and reports the merge commit SHA; it
does not persist a separate landing record.

Topology-bound verification is also a receipt-only path: `verify_work` may run
against its frozen baseline/candidate without an active worktree-manager handle,
while `land_work` remains limited to the normal host-managed landing session.

Trusted checks run direct argument arrays with `shell=False`; a shell or `env`
cannot be the executable, and either is rejected behind `uv run`. They use an
independent fail-closed Linux Bubblewrap sandbox with network disabled, scrubbed
host credentials and config, and disposable home/temp/caches. Each check receives
an exact-HEAD Git-exported snapshot with no Git metadata, a private digest-pinned
native executable, and only a per-check run directory writable. The runner
compares repository state before and after all checks.

A recipe may also carry a frozen execution topology. In that mode the host
validates physical control and candidate worktrees, exact lifecycle metadata and
SHAs, completed dependencies, and a durable external evidence directory before
the first model turn. Packet/status metadata must be regular tracked blobs at
the exact control commit, and Git probes discard ambient `GIT_*` variables and
user/system configuration. Evidence must be outside the system temporary tree
and may neither contain nor be contained by any worktree or Git common
directory. Control, evidence, Git administration, host logs, and receipts remain
read-only to model tools; sandbox startup fails closed for managed sessions.

The host finishes all intended candidate edits and any commit required by the
current workflow before verifier dispatch, then does not mutate it while
verification runs. Workspace, landing-base, and attempt
generation changes invalidate the result. Session scratch artifacts are cleaned
by the host, so the verifier leaves them in place; denied or skipped tool calls
invalidate the run.

Without a configured recipe, a current workspace- and base-bound verifier PASS
may authorize completion reporting, but it never authorizes non-trivial landing.
A workflow contract can gate
candidate delivery, but its model-authored PASS cannot authorize landing. A
workflow `then="verifier"` stage may authorize only exact-candidate delivery by
recording an actual current verifier PASS; it never authorizes landing. It
commits and fingerprints the isolated candidate before the
verifier runs, requires a clean and unchanged parent workspace, and records
authorization only when the delivered workspace exactly matches that candidate.
Pasted verification prose never authorizes a merge. The
documentation-only `trivial: <reason>` waiver is available only in this
unconfigured mode.

Verifier attempts retain a typed disposition independently of raw output. An
incomplete task, denied/skipped action, stale candidate, failed trusted check,
or missing receipt cannot be overwritten by a textual `VERDICT: PASS`; the host
replaces contradictory parent completion prose before it is streamed. Repeated
same-class filesystem, policy, or sandbox capability failures stop the turn
after three attempts.

## System Prompt Assembly

**Source**: `vibe/core/system_prompt.py`

`get_universal_system_prompt(tool_manager, config, skill_manager, agent_manager, ...)` assembles the full system prompt from:
- **Project context** — `ProjectContextProvider` (line 47) gathers git status (branch, remote, porcelain status, recent commits) with 30s TTL cache via `ThreadPoolExecutor`
- **Tool descriptions** — from `ToolManager`
- **Skill descriptions** — from `SkillManager`
- **Agent descriptions** — from `AgentManager`
- **Experiment flags** and **baseline scaling tier**
- **Worktree awareness** — when active, adds isolation instructions

Templates are loaded from `UtilityPrompt` / `SystemPrompt` enums. Prompt sections can be enabled/disabled and sized based on model tier (`BaselineTier` — see `vibe/core/baseline_scaling.py`).

## Session Persistence

**Source**: `vibe/core/session/`

- `SessionLogger` records each turn to `messages.jsonl` + `meta.json` in a session directory under `~/.vibe/sessions/`
- `context.json` stores static session context (tool schemas, config, system prompt) separately from per-round metadata
- `SessionLoader` reads sessions back for `--continue` / `--resume`
- Sessions are archived as `.tar.gz` with `fcntl` locking (POSX) to prevent concurrent archive races
- `session_migration.py` handles format upgrades for older sessions

## CLI / TUI Layer

**Source**: `vibe/cli/`

- `entrypoint.py` — outermost `vibe` command, fast `--help`/`--version` via pure argparse
- `cli.py` — real CLI bootstrap: loads config, trusted folders, hooks, plugins, telemetry, then constructs `AgentLoop`
- `textual_ui/app.py` — `VibeApp`, the main Textual app (219 KB)
- `textual_ui/widgets/` — 47 widget files: chat history, input, tool display, approval, model picker, session picker, etc.
- `textual_ui/handlers/event_handler.py` — central event/message dispatch
- `textual_ui/message_queue.py` — async bridge from agent loop output to Textual widgets

## ACP Layer

**Source**: `vibe/acp/`

The ACP server is an alternative frontend driven by external ACP clients (editors/IDEs). The same `AgentLoop` core runs underneath.

One ACP process binds the first canonical cwd and exact requested additional-
directory set for its lifetime. Multiple sessions may run concurrently only
inside that same workspace contract; another workspace requires a separate ACP
process. The server rejects mismatched new/load/fork operations before changing
cwd, trust-dependent configuration, or harness roots, and rejects prompt work if
the bound process cwd or roots drift. Loaded history must match the bound cwd.

Automated policy, Safety-Judge, and bypass-authorized Bash calls use the
core-managed sandbox and minimal environment. Human and stored-human calls are
revalidated against the same permission context and cwd immediately before the
editor client's terminal is created; that terminal receives the session's bound
cwd. Contextless direct tool calls are an internal test/embedding compatibility
path, not ACP authorization.

- `entrypoint.py` — bootstraps config, optionally runs `--setup`, starts `run_acp_server()`
- `acp_agent_loop.py` — implements the ACP protocol: initialize → new session → prompt → stream chunks → close
- `tools/base.py` — `BaseAcpTool` wraps core `BaseTool` with ACP `Client` + session state
- `tools/builtins/` — 10 ACP-specific tool implementations (bash, edit, grep, read, write_file, task, todo, skill, web_fetch, web_search)
- `commands/registry.py` — `AcpCommandRegistry` for editor commands like teleport

## Key Architectural Patterns

1. **Mixin composition** — `AgentLoop` is assembled from 6 focused mixins, each with a documented implicit contract
2. **Protocol-based backend** — `BackendLike` protocol enables dependency injection and test mocking
3. **Lazy initialization** — MCP SDK import, connector setup, and system prompt assembly are deferred to background threads
4. **Layered config** — TOML + env + harness files, with trust resolution and patch system
5. **Middleware pipeline** — context shaping, compaction, limits, loop detection are pluggable middlewares
6. **Safety-first tool execution** — LLM safety judge + permission store + hooks can all gate or modify tool calls
7. **Scale-aware limits** — tool result caps scale with model context window; system prompt sections scale with baseline tier

## Where to Start When Changing Architecture

- **Agent turn lifecycle**: `vibe/core/agent_loop.py` → `vibe/core/middleware.py` → `vibe/core/tools/manager.py` → tests in `tests/core/test_agent_loop_*.py` and `tests/agent_loop/`
- **Add/change a builtin tool**: `vibe/core/tools/base.py` → `vibe/core/tools/manager.py` → specific builtin in `vibe/core/tools/builtins/`
- **Add/change a workflow**: `vibe/core/workflows/manager.py` → `vibe/core/workflows/runtime.py` → `tests/core/workflows/`
- **Config behavior**: `vibe/core/config/_settings.py` → `vibe/core/config/builder.py` → `tests/core/test_config_*.py`
- **ACP behavior**: `vibe/acp/entrypoint.py` → `vibe/acp/acp_agent_loop.py` → `tests/acp/`

## Watch-Outs

- The repo is a fork with continuous upstream sync — avoid structural moves unless absolutely necessary
- Tool and workflow discovery are dynamic; file layout changes can change runtime behavior
- UI and session behavior are tested heavily with snapshots — small changes can require baseline regeneration
- The `agent_loop.py` file is 140 KB and the most churn-sensitive file in the repo — keep edits minimal and localized
