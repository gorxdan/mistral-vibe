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
AgentLoop(AgentLoopMemoryMixin, AgentLoopFailoverMixin, AgentLoopSafetyJudgeMixin, AgentLoopHooksMixin)
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

### AgentLoopMemoryMixin
**Source**: `vibe/core/agent_loop_memory.py`

Durable memory subsystem: recall (selection + prefetch), extraction, consolidation, verification. Uses async background tasks. Collaborates with `MemoryStore`, `MemorySelector`, `MemoryExtractor`, `MemoryConsolidator`, `MemoryVerifier` (in `vibe/core/memory/`).

## AgentLoopLimits

**Source**: `vibe/core/agent_loop_limits.py`

Pure constants (no class coupling):
- `MAX_TOOL_RESULT_CHARS = 100_000` — per-result cap (~25k tokens)
- `TOOL_RESULT_PREVIEW_CHARS = 12_000` — inline preview size (head 75% + tail 25%)
- `AGGREGATE_TOOL_RESULT_CHARS = 200_000` — total cap for parallel tool calls
- `tool_result_hard_cap(threshold_tokens)` — scales cap to model's context budget
- `MAX_CONCURRENT_SUBAGENTS = 2`

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
    │   ├── Parse LLM response → ParsedToolCall → ResolvedToolCall (via ToolManager)
    │   ├── Safety Judge — pre-screen ASK-gated tools
    │   ├── Hooks — before_tool / after_tool lifecycle
    │   ├── Permission Store — approval gate
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

The design doc `docs/design/compaction.md` describes the multi-stage shaper pipeline (snip + microcompact) in detail.

## Spend Admission

**Source**: `vibe/core/usage/_broker.py`, `_ledger.py`, `_session.py`

Primary, compaction, in-process task/workflow, memory-helper, and safety-judge
calls share a durable hierarchical ledger
(`session -> workflow/team -> agent -> call`). Admission is reserved before
backend dispatch under a file lock, so sibling agents cannot race past the
parent cap. Provider usage reconciles the reservation; errors or missing usage
retain the estimate. Session resume rebinds the adapter to the resumed ledger,
and active calls renew their leases.

The broker core is cross-process capable, but isolated subprocess, MCP sampling,
narration, and backend-internal retry attempts are not yet routed as distinct
calls.

## Verification Receipts

**Source**: `vibe/core/_verification_receipt.py`,
`vibe/core/_verification_runner.py`, `vibe/core/verification_state.py`

Trusted local checks create immutable receipts bound to the task brief,
acceptance contract, repository identity and state, configuration, check set,
and full-output artifact hashes. `land_work` revalidates the receipt against the
candidate before merging and records the landed commit. Model-authored verifier
prose cannot authorize a merge; the only non-receipt path is a locally validated
documentation-only trivial waiver.

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

- `entrypoint.py` — bootstraps config, optionally runs `--setup`, starts `run_acp_server()`
- `acp_agent_loop.py` — implements the ACP protocol: initialize → new session → prompt → stream chunks → close
- `tools/base.py` — `BaseAcpTool` wraps core `BaseTool` with ACP `Client` + session state
- `tools/builtins/` — 10 ACP-specific tool implementations (bash, edit, grep, read, write_file, task, todo, skill, web_fetch, web_search)
- `commands/registry.py` — `AcpCommandRegistry` for editor commands like teleport

## Key Architectural Patterns

1. **Mixin composition** — `AgentLoop` is assembled from 4 focused mixins, each with a documented implicit contract
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
