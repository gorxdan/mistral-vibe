# Tool System

The tool system is how the agent interacts with the world — reading files, running commands, searching code, delegating to subagents, and more. It's built around a plugin-like architecture with discovery, permission gating, and safety checks.

## BaseTool

**Source**: `vibe/core/tools/base.py`

`BaseTool` is the abstract base class for all tools. Every tool:
- Subclasses `BaseTool` with a Pydantic args model and `BaseToolConfig` generic
- Implements `async def run(args, ctx: InvokeContext)` — yields events progressively
- Declares a `ToolPermission` (`ALWAYS` / `ASK` / `NEVER`)
- Raises `ToolError` for failures, `ToolPermissionError` for authz failures

`read_only` supplies the default scheduling classification, while
`call_is_read_only(args)` may refine it per invocation. Maximal adjacent
read-only sequences run concurrently. A false classification, including the
conservative default for unknown tools, creates an ordered singleton barrier.
The scheduler computes this classification once, stores it on the wave, and
passes the bound value through execution rather than recomputing it after hooks.
If a hook or approval edit changes a scheduled read-only call into a mutation,
the call is rejected and must be retried as a separate mutation.
An unexpected executor exception produces a terminal failure for its call when
needed, drains the current wave, synthesizes failures for every announced call
that lacks a terminal result, aborts later waves, and then propagates.

### InvokeContext

`InvokeContext` (dataclass, line 49) is the rich context passed to every tool invocation:
- `tool_call_id`, `approval_callback`, `scheduler`
- `agent_manager`, `active_model`, `user_input_callback`
- `session_dir`, `permission_store`
- `files_read` — shared dict for read-before-edit enforcement
- `background_registry`, `safety_judge_factory`
- Workflow and team callbacks
- `tool_manager` — for meta-tools like `tool_search`

## ToolManager

**Source**: `vibe/core/tools/manager.py`

`ToolManager` (line 89) discovers tools from search paths, instantiates them, and manages MCP and connector registries.

- Takes a `config_getter` callable (lazy config access), `mcp_registry`, `connector_registry`, `permission_getter`
- `_all_tools: dict[str, type[BaseTool]]` — discovered tool classes keyed by name
- `_instances: dict[str, BaseTool]` — cached instances
- `_builtin_pins` / `_manifest_pins` — sticky pinning so remote activation can't evict in-use builtins
- `integrate_all()` — integrates MCP and connector tools (deferred to avoid ~100ms MCP SDK import on cold start)
- Fuzzy matching for tool name suggestions (`_TOOL_SEARCH_FUZZY_MATCH_THRESHOLD = 0.25`)

## Builtin Tools

**Source**: `vibe/core/tools/builtins/`

24 builtin tools, auto-discovered from the `builtins/` directory:

| Category | Tools |
|---|---|
| **File ops** | `read`, `write_file`, `edit`, `glob`, `grep` |
| **Execution** | `bash` (sandbox, background processes), `background` |
| **Agent coordination** | `task` (subagent spawning, 27 KB), `team`, `team_message`, `team_spawn`, `ask_user_question` |
| **Planning** | `enter_plan_mode`, `exit_plan_mode`, `todo` |
| **Memory** | `manage_memory` |
| **Web** | `webfetch`, `websearch` |
| **LSP** | `lsp` (40 KB — diagnostics, definitions, references, call hierarchy) |
| **Workflows** | `launch_workflow`, `workflow_results`, `workflow_status`, `workflow_stop` |
| **Skills** | `skill` |
| **Scheduling** | `schedule` |
| **Meta** | `tool_search` |

### Adding a New Builtin Tool

1. Create a new `.py` file in `vibe/core/tools/builtins/` (auto-discovered)
2. Subclass `BaseTool` with a Pydantic args model
3. Implement `async def run(args, ctx: InvokeContext)` — yield events progressively
4. Set `ToolPermission` (`ALWAYS` for safe read-only tools, `ASK` for destructive ones)
5. The tool is automatically available — no registration needed

## Permission System

**Source**: `vibe/core/tools/permissions.py`

`PermissionStore` manages the approval gate for tool execution. The flow:
1. LLM proposes a tool call
2. The scheduler binds the invocation's read-only classification
3. `before_tool` hooks fire and any rewrite is validated against that classification
4. Tool-specific resolution applies hard denials, then nondelegable explicit-authority gates
5. Permission Store coverage may satisfy ordinary ASK patterns
6. Safety Judge pre-screens only the remaining eligible ASK calls
7. The human or ordinary auto-approve gate decides the remaining call

Bash package acquisition, dependency-graph changes, and recognized
verification/build commands are explicit-user gates. Stored wildcard/session
rules, `permission = "always"`, auto-approve, and Safety Judge verdicts cannot
authorize them. Hard `NEVER` decisions remain authoritative.

Permissions can be configured per-tool in `config.toml`:
```toml
[tools.bash]
permission = "ask"   # always | ask | never
```

## Safety Judge

**Sources**: `vibe/core/tools/safety_judge.py`, `vibe/core/agent_loop_safety_judge.py`

An experimental LLM-based safety gate that pre-screens ASK-gated tool calls. Off by default.

- Uses a separate model (ideally different provider than active model)
- Pre-screens tool calls before the user prompt
- **Fails closed**: API error, timeout, refusal, or unparsable answer all fall back to human prompt
- Verdict cache keyed by tool name + args hash + transcript hash
- Reuses one judge instance while model/provider/config/timeout/spend identity is unchanged
- Clears cached verdicts whenever any part of that full identity changes, even when the model alias is unchanged
- After a provider rejects `temperature`, retries without it and remembers that omission for later calls in the session
- Truncation guards: args capped at 4000 chars, transcript at 2000 chars (last 4 turns)
- Force-defers to user when args are truncated AND a risk flag is present

Config:
```toml
[safety_judge]
enabled = true
model = "devstral-small"
max_tokens = 512
timeout = 15.0
```

**Security note**: An LLM judge is a probabilistic gate, not a guarantee. Keep your denylist authoritative. A compromised main model could craft calls designed to fool the judge.

## MCP Integration

**Source**: `vibe/core/tools/mcp/`

Model Context Protocol servers extend Vibe's capabilities with external tools. Key features:
- Three transports: `http`, `streamable-http`, `stdio`
- OAuth and static auth (API key) support
- MCP tools named `{server_name}_{tool_name}` (underscore-separated)
- Per-tool permission configuration
- `/mcp add` command for interactive OAuth server setup
- Configurable startup and tool execution timeouts

## LSP Integration

**Source**: `vibe/core/tools/builtins/lsp.py` (40 KB), `vibe/core/lsp/`

Language Server Protocol support provides semantic code intelligence:
- `go_to_definition`, `find_references`, `hover`, `incoming_calls`, `outgoing_calls`, `document_symbol`
- Diagnostics (errors, warnings) automatically surfaced to the model after `edit`/`write_file` calls
- Live `status` snapshots distinguish an enabled tool from a running server and route readiness by file extension
- Reference, symbol, and call-hierarchy collections use short-lived, session/task/workspace-bound opaque continuation tokens instead of discarding capped tails
- Human columns are Unicode code points converted to/from LSP UTF-16 positions; document-symbol trees retain child hierarchy
- Workspace roots are selected from the nearest bounded manifest marker, with separate server instances for monorepo roots
- Dynamically discovered roots use a configurable root-bucket LRU (`lsp_max_workspace_roots`, default 8); active operations, the session root, and explicit roots are protected while idle roots retire cleanly
- `workspace_symbol` reports resident/known root coverage and marks results partial when known roots have been retired
- **Opt-in**: install with `/lspstall`, remove with `/unlspstall`
- Builtin servers auto-discovered from project manifests; `[[lsp_servers]]` adds custom definitions
- Restricted child environment; additional server variables must be explicit in `env`
- Disabled in parent-spawned isolated-worktree subagents/workflows until language servers run inside the OS process sandbox (ordinary top-level programmatic worktrees are unaffected)
- The preferred tool for symbol questions — resolves imports, re-exports, and overloads that grep misses

## Bash Sandboxing

**Sources**: `vibe/core/tools/builtins/bash.py`, `vibe/core/tools/sandbox.py`, `vibe/core/tools/sandbox_seccomp.py`

The `bash` tool runs shell commands in a sandboxed environment:
- `sandbox.py` — bubblewrap (Linux) / sandbox-exec (macOS) wrapper
- `sandbox_seccomp.py` — seccomp filter for additional syscall restrictions
- `command_safety.py` — analyzes commands for dangerous patterns
- Background process support via `BackgroundRegistry` (`vibe/core/tools/background.py`)
- Each foreground call starts a fresh shell in the current process working directory; shell-local `cd` and environment changes do not persist across calls
- Core-managed Unix policy parsing and execution use one process-frozen absolute Bash executable, independent of the user's login shell
- Startup and injection variables such as `BASH_ENV`, exported Bash functions, `LD_PRELOAD`, `LD_AUDIT`, and `DYLD_INSERT_LIBRARIES` are always stripped; the default human compatibility scrub also removes other loader variables, which survive only with `sandbox.scrub_env = false`
- Nested shells, language interpreters, leading environment assignments, and executable-bearing package/Git forms require explicit user approval
- Sandbox backends and the probe helper are resolved once from the sanitized system path; the exact absolute backend executable that was selected is placed in the launch argv
- On Linux, automated execution accepts only root-owned regular executables without set-ID or group/world-write bits and without write access for the current non-root user; lexical and resolved ancestry must satisfy the same root-control and write restrictions
- Exact standalone `git log`, `show`, `blame`, and `grep` calls are hardened for automated execution; `diff`, `status`, wrapped/composed/redirected calls, and other Git forms require explicit user approval

Policy-, Safety-Judge-, and bypass-authorized core Bash calls use a trusted
system `PATH` and minimal noninteractive environment. Human and stored-human ACP
approvals are revalidated immediately before the editor client creates its
terminal. An ACP server binds its first canonical cwd and exact additional-
directory request for the process lifetime; same-workspace sessions may overlap,
while another workspace requires another ACP process.

Before sandbox launch, Bash rejects proven verification commands whose exit
status is hidden by pipes, lists, backgrounding, inversion, substitutions,
embedded newlines, nested shells, `find -exec`, or equivalent literal carriers.
Opaque or dynamic execution carriers require explicit user approval and are not
treated as verification evidence. Run checks directly; Bash already caps
displayed output and persists truncated output. Fish in a proven executable
position or recognized literal execution carrier is rejected because the Bash
parser cannot validate Fish syntax. Opaque or dynamic Fish-like positions and
the exact standalone `fish -v` and `fish --version` diagnostic exceptions require
explicit user approval.
Package/dependency changes and recognized test/build commands always require an
explicit user decision, including `dotnet test --no-restore`.

Background cleanup signals a process group only when the child PID is still
verified as both the process-group leader and session leader; otherwise it
signals the direct child. Default and xdist tests mock signal calls. Real
process-tree teardown probes are manual checks for disposable isolated hosts,
never graphical login sessions. Any live process, workflow, agent, team, or
scheduled loop invalidates verifier authorization until it reaches a terminal
state. The real probes are marked `process_e2e`, skipped by default, and require
`VIBE_PROCESS_E2E_DISPOSABLE=1 uv run pytest -n0 --run-process-e2e ...`.

## Result Size Limits

**Source**: `vibe/core/agent_loop_limits.py`

Tool results are size-limited to prevent context overflow:
- `MAX_TOOL_RESULT_CHARS = 100_000` — per-result cap (~25k tokens)
- `TOOL_RESULT_PREVIEW_CHARS = 12_000` — inline preview (head 75% + tail 25%)
- `AGGREGATE_TOOL_RESULT_CHARS = 200_000` — total cap for parallel tool calls
- `tool_result_hard_cap(threshold_tokens)` — scales cap to model's context budget (5% of window)
- Oversized results are persisted to disk via `tool_result_store.py`; the model sees a preview with a reference to the full output

## Subagent Isolation

**Source**: `vibe/core/tools/builtins/task.py` (27 KB)

Write-capable subagents (`worker`/`auto-approve`/`editor`) run in their own git worktree by default:
- Destructive commands and edits are scoped to a throwaway branch
- Branch is merged back only on success
- Read-only subagents (like `explore`) stay in-process
- Configurable with `task.isolation` (`off`/`auto`/`always`)
- Optional safety judge pre-flights the delegation prompt before the subprocess spawns

## Where to Start When Changing Tools

- **Add a new tool**: create a file in `vibe/core/tools/builtins/` — auto-discovered
- **Change tool permissions**: `vibe/core/tools/permissions.py` + `vibe/core/tools/base.py`
- **Change bash sandbox**: `vibe/core/tools/sandbox.py` + `vibe/core/tools/sandbox_seccomp.py`
- **Change MCP integration**: `vibe/core/tools/mcp/` + `vibe/core/tools/manager.py` (integrate_all)
- **Change LSP**: `vibe/core/lsp/` + `vibe/core/tools/builtins/lsp.py`
- **Change safety judge**: `vibe/core/tools/safety_judge.py` + `vibe/core/agent_loop_safety_judge.py`

## Tests

- `tests/tools/` — 38 test files covering bash, sandbox, safety judge, MCP, connectors, grep, glob, task, team spawn, websearch, workflows
- `tests/tools/test_bash.py` (25 KB), `test_sandbox.py` (37 KB), `test_safety_judge.py` (30 KB)
- `tests/core/test_tool_concurrency.py`, `test_tool_result_budget_middleware.py`, `test_tool_schema_trim.py`
