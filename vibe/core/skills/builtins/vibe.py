from __future__ import annotations

from vibe import __version__
from vibe.core.loop import MAX_LOOPS_PER_SESSION, MIN_INTERVAL_SECONDS
from vibe.core.skills.builtins.capsules import SkillDocCapsule
from vibe.core.workflows.runtime import DEFAULT_MAX_CONCURRENT

_PROMPT_TEMPLATE = """# Vibe CLI Self-Awareness

You are running inside **Mistral Vibe**, a CLI coding agent built by Mistral AI.
This skill gives you full knowledge of the application internals so you can help
the user understand, configure, and troubleshoot their Vibe installation.

## Going Deeper

For facts not covered here, fetch the README pinned to the running version:
https://github.com/mistralai/mistral-vibe/blob/v__VIBE_VERSION__/README.md
(do not use `main` — it may not match what is installed). Point the user at
https://docs.mistral.ai/vibe/code/overview for human-readable docs.

## VIBE_HOME

The user's Vibe home directory defaults to `~/.vibe` but can be overridden via
the `VIBE_HOME` environment variable. All user-level configuration, skills, tools,
agents, prompts, logs, and session data live here.

### Directory Structure

```
~/.vibe/
  config.toml          # Main configuration file (TOML format)
  hooks.toml           # User-level hook definitions (experimental)
  .env                 # API keys and credentials (dotenv format)
  vibehistory          # Command history
  trusted_folders.toml # Trust database for project folders
  agents/              # Custom agent profiles (*.toml)
  prompts/             # Custom prompts (*.md)
  skills/              # User-level skills (each skill is a subdirectory with SKILL.md)
  tools/               # Custom tool definitions
  workflows/           # User-level workflow scripts (*.py with YAML frontmatter)
  memory/              # Cross-session memory files (*.md with YAML frontmatter)
    projects/<hash>/   # Per-project memory namespaces (hash of trusted workdir)
  logs/
    vibe.log           # Main log file
    session/           # Session log files
  plans/               # Session plans
  teams/               # Team directories (created on demand, cleaned up on exit)

~/.agents/
  skills/              # Additional user-level skills directory
```

### Project-Local Configuration

When in a trusted folder, Vibe also looks for project-local configuration:
- `.vibe/config.toml` - Project-specific config (overrides user config)
- `.vibe/hooks.toml` - Project-specific hooks (requires trusted folder)
- `.vibe/skills/` - Project-specific skills
- `.vibe/tools/` - Project-specific tools
- `.vibe/agents/` - Project-specific agents
- `.vibe/workflows/` - Project-specific workflow scripts
- `.vibe/prompts/` - Project-specific prompts
- `.agents/skills/` - Standard agent skills directory

## Lifecycle: Exit, Update, Version, Resume

### Exit

Chat input (case-insensitive): `/exit`, `exit`, `quit`, `:q`, `:quit`.
Keyboard: `Ctrl+C` / `Ctrl+D` — press twice within ~1s to quit. For `Ctrl+C`,
the first press instead interrupts the running job or clears the input if either
is present. Set `ask_confirmation_on_exit = false` to make `Ctrl+D` quit on the
first press (also toggleable in `/config`); `Ctrl+C` always requires a second
press. `Ctrl+Z` suspends on POSIX (resume with `fg`).

### Update

In this fork, in-app update checks are off by default
(`enable_update_checks = false`). The `mistral-vibe` PyPI package is the upstream
release, not this fork, so never run `uv tool upgrade mistral-vibe`. Updates are
pulled from the `mistralai/mistral-vibe`
upstream via the `upstream` git remote and verified by the `upstream-sync` CI
workflow; apply them with `git pull upstream <tag> && uv sync --all-extras`.

### Version

`vibe --version` (or `-v`) prints it and exits. Not shown anywhere in-session.

### Resume

- `vibe -c` / `--continue`: most recent session in this terminal (TTY-scoped;
  falls back to latest in cwd).
- `vibe --resume [SESSION_ID]`: specific session; without an id, opens a picker.
- In-session: `/resume` (alias `/continue`).

#### Session storage & folder scoping

Local sessions are written under `~/.vibe/logs/session/` (override with
`session_logging.save_dir`). Each session records the `cwd` it ran in. The
`/resume` picker, `--continue`, and bare `--resume` (no id) are **scoped to the
current folder**: only sessions whose `cwd` matches where Vibe is launched are
listed, so the same directory shows its own history and nothing else. Switch
folders to see a different set. The explicit `--resume <SESSION_ID>` form is
**not** folder-scoped: it resolves the session by id regardless of which folder
it ran in. When Vibe Code is enabled, active **remote** sessions are listed
alongside local ones in the picker (tagged `remote`) and are not folder-scoped.

## Configuration (config.toml)

The configuration file uses TOML format. Settings can also be overridden via
environment variables with the `VIBE_` prefix (e.g., `VIBE_ACTIVE_MODEL=local`).

Custom prompt IDs are resolved in this order (first match wins): `prompt_paths`
from config.toml first, then project-local `.vibe/prompts/` in trusted project
roots, then `~/.vibe/prompts/` (user global), and finally the built-in bundled
prompts.

### Key Settings

```toml
# Model selection
active_model = "mistral-medium-3.5"  # Model alias to use (see [[models]])

# UI preferences
disable_welcome_banner_animation = false
autocopy_to_clipboard = true
file_watcher_for_autocomplete = true   # default: on
ask_confirmation_on_exit = true  # Require a second Ctrl+D to quit (Ctrl+C always confirms)

# Behavior
bypass_tool_permissions = false    # Skip tool approval prompts
system_prompt_id = "cli"          # System prompt: "cli", "lean", or custom .md filename
compaction_prompt_id = "compact"  # Compaction prompt: built-in "compact" or custom .md filename
enable_telemetry = true
enable_update_checks = false      # Off in this fork; upstream tracked via the upstream-sync CI workflow + git
enable_notifications = true
enable_system_trust_store = false  # Use OS trust store for outbound HTTPS
api_timeout = 720.0               # API request timeout in seconds
auto_compact_threshold = 200000   # Token count before auto-compaction

# Git commit behavior
include_commit_signature = true    # default: on — include commit guidance in system prompt

# Writing style
include_humanizer_guidance = true  # Prompt the model to avoid AI-writing patterns
caveman_thinking = true            # Compress reasoning/thinking blocks (terse; answer stays normal prose)

# System prompt composition
include_model_info = true         # Include model name in system prompt
include_project_context = true    # Include project context (git info, cwd) in system prompt
include_prompt_detail = true      # Include OS info, tool prompts, skills, and agents in system prompt
include_config_reference = true   # Always-on condensed config/MCP/providers reference (this CLI's self-knowledge)

# Voice features
voice_mode_enabled = false
narrator_enabled = false
active_transcribe_model = "voxtral-realtime"
active_tts_model = "voxtral-tts"

# Workflows and effort
effort_mode = "normal"            # "normal" or "le-chaton" (max thinking + auto-workflow)
disable_workflows = false         # Disable all workflow features
verification_subsystem = true     # Host verification layer (todo nudge + contract section → verifier subagent)
investigation_subsystem = true    # Host investigation layer (contract section: reproduce-before-fix guidance)
workflow_paths = []               # Additional dirs to search for workflow scripts

# Additional top-level keys (defaults shown)
theme = "ansi-dark"                # Default terminal theme (override via /theme or VIBE_THEME; empty coerces to this default)
displayed_workdir = ""            # Override the workdir label shown in the UI
compaction_model = ...            # Alias for compaction; default unset (uses active)
fallback_models = []              # Aliases tried if the active model errors
context_shaping = ...             # Sub-table controlling context window shaping
safety_judge = ...                # Sub-table for the safety-judge backend (gates tools/spawns)
prompt_paths = []                 # Additional dirs searched first for custom prompts
plugin_paths = []                 # Additional dirs searched for plugin manifests
installed_components = []         # Opt-in features, e.g. ["lsp"]
```

### Providers

```toml
[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
api_key_env_var = "MISTRAL_API_KEY"
backend = "mistral"

[[providers]]
name = "llamacpp"
api_base = "http://127.0.0.1:8080/v1"
api_key_env_var = ""
extra_headers = { "X-Custom-Header" = "value" }  # optional per-provider HTTP headers
```

`[[providers]]` also accepts `api_style`, `reasoning_field_name`,
`discover_models`, `project_id`, `region`, `cache` (sub-table: mode/style/extra_body),
`max_concurrent_requests`, and `requests_per_minute` (rate limiting). See
`vibe/core/config/_settings.py` (ProviderConfig, ProviderCacheConfig).

### Models

```toml
[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "mistral-medium-3.5"
temperature = 1.0                 # or "omit": never send it (for providers that reject it)
input_price = 1.5
output_price = 7.5
thinking = "high"                 # "off", "low", "medium", "high", "max"
auto_compact_threshold = 200000
# context_window = 262144         # optional: model's real window; derives (85%) or clamps (95%) auto_compact_threshold
supports_images = true            # vision-capable; allows @-mentioned images

[[models]]
name = "devstral-small-latest"
provider = "mistral"
alias = "devstral-small"
input_price = 0.1
output_price = 0.3

[[models]]
name = "devstral"
provider = "llamacpp"
alias = "local"
```

`[[models]]` also accepts `max_output_tokens` (int, default unset — seeds/caps
max-output escalation). See `vibe/core/config/_settings.py` (ModelConfig).

### Tool Configuration

```toml
# Additional tool search paths
tool_paths = ["/path/to/custom/tools"]

# Enable only specific tools (glob and regex supported)
enabled_tools = ["bash", "read", "grep"]

# Disable specific tools
disabled_tools = ["webfetch"]

# Dynamic remote-tool manifest. Enabled by default: when the catalog is large,
# remote MCP/connector tools are hidden from the per-turn manifest until
# discovered with tool_search; built-in tools remain visible.
[tool_manifest]
dynamic_subset_enabled = true
dynamic_subset_threshold = 80
dynamic_pinned_tool_limit = 8
dynamic_search_results = 8

# Per-tool configuration
[tools.bash]
allowlist = ["git", "npm", "python"]

# Web search backend. With no searxng_url, web_search uses Mistral web search.
# Set searxng_url (or the SEARXNG_URL env var) to use a local SearXNG instance.
[tools.web_search]
searxng_url = "http://localhost:8888"   # enables SearXNG; persisted via onboarding too
searxng_manage = true                    # let vibe run the container (docker/podman)
searxng_image = "searxng/searxng:latest"
searxng_container_name = "vibe-searxng"
searxng_port = 8888
searxng_autostart = true                 # start at session start if down
searxng_stop_on_exit = true              # stop on exit, only if vibe started it
# General-web engines vibe force-enables in every managed container so a single
# engine rate-limiting itself never zeroes results. Override to change; [] opts out.
searxng_enabled_engines = ["bing", "duckduckgo", "startpage", "google", "qwant", "mojeek"]
searxng_disabled_engines = []              # engines to force-disable in a managed container
searxng_timeout = 15.0                     # per-search timeout
searxng_health_timeout = 5.0               # startup health-check timeout
# search/request-level knobs: timeout (per request), model (alias for query
# rewriting), permission. See vibe/core/tools/builtins/websearch.py (WebSearchConfig).

# Task tool isolation: write-capable subagents run in their own git worktree
# so destructive commands/edits can't race the parent tree or sibling agents.
# "auto" (default) isolates only write-capable profiles (worker/auto-approve/
# editor/grunt, and any profile with write_file/edit or un-jailed bash); read-only
# and read-jailed profiles (explore/research/planner/reviewer/debugger/
# security/verifier) stay in-process for speed. "always" isolates every subagent;
# "off" forces in-process (the historical behavior).
# An isolated spawn is pre-flight judged: the safety judge (when configured)
# evaluates the delegation prompt before the subprocess starts — safe proceeds,
# a deferral routes through the host approval callback, and a denial blocks the
# spawn entirely (no subprocess, no worktree). Fail-open when no judge is set.
[tools.task]
isolation = "auto"
```

When `searxng_manage` is on and docker/podman is available, vibe starts a
configured-but-down SearXNG at session start and stops it on exit (only if vibe
started it); a mid-search down instance prompts to start it or fall back to
Mistral. See `docs/searxng-setup.md`.

**Special case — `find` command:** Even if `find` is in the bash allowlist,
Vibe detects `-exec`, `-execdir`, `-ok`, and `-okdir` predicates and will
prompt for user permission before running the command.

#### File Tool Permission Resolution

File-based tools (`read`, `grep`, `glob`, `write_file`, `edit`) resolve
permissions in this order (first match wins):

1. **Scratchpad** path → always allowed
2. **denylist** glob match → always denied
3. **allowlist** glob match → always allowed
4. **sensitive_patterns** match → requires approval
5. **Outside workdir** → requires approval (or denied if `permission = "never"`)
6. **Default** → uses the tool's `permission` setting

The **denylist** is checked before the allowlist — a path matching both lists
is denied. Both are checked before the outside-workdir boundary, so the
allowlist can still auto-approve access to directories outside the project.

### Skill Configuration

```toml
# Additional skill search paths
skill_paths = ["/path/to/custom/skills"]

# Enable only specific skills
enabled_skills = ["vibe", "custom-*"]

# Disable specific skills
disabled_skills = ["experimental-*"]
```

### Agent Configuration

```toml
# Additional agent search paths
agent_paths = ["/path/to/custom/agents"]

# Enable/disable agents
enabled_agents = ["default", "plan"]
disabled_agents = ["auto-approve"]

# Opt-in builtin agents (only affects agents with install_required=True, e.g. lean)
installed_agents = ["lean"]

# Agent profile to use when --agent is not passed
# (default: "default"). Valid values: "default", "plan", "accept-edits",
# "auto-approve", "lean" (only when listed in installed_agents), or any
# custom agent name from ~/.vibe/agents/ or .vibe/agents/. Subagents
# (e.g. "explore") are rejected. Applies in both interactive and programmatic
# (-p/--prompt) mode.
default_agent = "plan"
```

### MCP Servers

MCP (Model Context Protocol) servers supply tools to the agent. Add one via a
`[[mcp_servers]]` block in config or the token-free `/mcp add` form. Tools are
named `{name}_{raw-tool}` (e.g. `github_create-issue`). Management is token-free:
`/mcp` (server + connector browser with live status), `/mcp <name>` (list one
server's tools), `/mcp login|logout <name>` (OAuth), `/mcp refresh` (re-discover
tools after a config change).

`transport` is the discriminator that selects the model. Fields by transport:

| transport | required | extras |
|---|---|---|
| `stdio` | `command` (str or list), `args` | `env`, `cwd` |
| `http` | `url` | `auth` (legacy SSE) |
| `streamable-http` | `url` | `auth` (current standard — prefer for new servers) |

Auth (http / streamable-http only), written inline as `auth = { ... }`:

| `type` | fields |
|---|---|
| `static` | `api_key_env` (env var holding the token), `api_key_header` (default `Authorization`), `api_key_format` (default `Bearer {token}`), `headers` (extra header map) |
| `oauth` | `scopes` (list; `[]` = accept the server default), `client_id` (pre-registered, PKCE) OR `client_metadata_url` (RFC 9728 doc — the two are mutually exclusive; omit both for Dynamic Client Registration), `redirect_port` (default 47823) |

Shared per-server fields (all transports): `name` (tool prefix; normalized to
`[a-zA-Z0-9_-]`), `prompt` (usage hint appended to tool descriptions),
`disabled` (hide every tool; discovery still runs), `disabled_tools` (hide named
tools, without the prefix), `startup_timeout_sec` (default 10), `tool_timeout_sec`
(default 60), `sampling_enabled` (default true; lets the server request LLM
completions via createMessage).

Hosted OAuth MCP servers can be added from inside Vibe with the `/mcp add`
shortcut:

```text
/mcp add https://mcp.linear.app/mcp
/mcp add https://mcp.example.com/mcp --name docs --scope read --transport http --no-login
```

`/mcp add` is OAuth-only. It writes `auth.type = "oauth"` with optional
scopes and starts login by default. It uses `transport = "streamable-http"`
unless you pass `--transport http`. Pass `--no-login` to add the server without
starting OAuth login. The shortcut supports `streamable-http` and `http`
transports. For API-key/static auth, edit `config.toml` using the static auth
example below.

```toml
# Local subprocess (stdio).
[[mcp_servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { MCP_LOG_LEVEL = "debug" }      # see the env note below
disabled_tools = ["delete_repo"]       # hide named tools without removing the server

# Remote server, static token in a custom header.
[[mcp_servers]]
name = "internal"
transport = "streamable-http"
url = "https://mcp.internal.example.com"
auth = { type = "static", api_key_env = "MCP_API_KEY", api_key_header = "X-API-Key", api_key_format = "{token}" }

# OAuth 2.1 (streamable-http). DCR when client_id is omitted.
[[mcp_servers]]
name = "supabase"
transport = "streamable-http"
url = "https://mcp.supabase.com/mcp"
auth = { type = "oauth", scopes = [] }
```

`env` (stdio) is merged over a minimal inherited set — only `PATH`, `HOME`,
`USER`, `SHELL`, `LOGNAME`, `TERM` on Linux. Arbitrary variables (including keys
in `~/.vibe/.env`) are NOT inherited, and values are literal with no `${...}`
expansion, so pass anything the server needs in `env` explicitly.

OAuth: run `/mcp login <name>` to authorize in a browser (tokens are stored in
the OS keyring); `/mcp logout <name>` clears them. The `/mcp` browser shows
`logged in` or `needs login` and binds `L` to start login. For headless or CI,
prefer `auth.type = "static"` with `api_key_env` (a personal access token).

Backwards compatibility: the static-auth keys (`api_key_env`, `api_key_header`,
`api_key_format`, `headers`) may still be written at the server top level — they
are promoted into an `[auth] (type = "static")` block at load time. Mixing them
with an explicit `auth` block is an error.

### LSP (Language Server Protocol)

Opt-in code intelligence. Install with `/lspstall` (remove with `/unlspstall`);
configured servers warm up in the background when a CLI session starts in its
working directory. Declare one `[[lsp_servers]]` entry per language; the binary
must be on `PATH`. Surfaces
the `lsp` tool (definitions, references, hover, symbols, call hierarchy) and
auto-injects server diagnostics into the next turn after `edit`/`write_file`.

```toml
installed_components = ["lsp"]

[[lsp_servers]]
name = "pyright"
command = "pyright-langserver"
args = ["--stdio"]
languages = { ".py" = "python" }

[[lsp_servers]]
name = "typescript"
command = "typescript-language-server"
args = ["--stdio"]
languages = { ".ts" = "typescript", ".tsx" = "typescriptreact", ".js" = "javascript", ".jsx" = "javascriptreact" }

[[lsp_servers]]
name = "rust-analyzer"
command = "rust-analyzer"
languages = { ".rs" = "rust" }
```

`[[lsp_servers]]` also accepts `env` (map), `cwd`, `initialization_options` (map),
`root_uri`, `manifest_markers` (list), `startup_timeout_sec` (default 20.0), and
`request_timeout_sec` (default 10.0). See `vibe/core/config/_settings.py` (LSPServer).

Auto-discovery (default `lsp_auto_discover = true`): when LSP is installed,
Mistral Vibe probes the builtin preset list (pyright, typescript-language-server,
rust-analyzer, gopls, clangd), keeps those whose binary is on `PATH`, and
filters them by project manifest markers — so a Python-only repo spawns only
pyright, a Rust workspace spawns only rust-analyzer, etc. Each preset's
`manifest_markers` (e.g. `Cargo.toml`, `go.mod`, `pyproject.toml`,
`package.json`) gate inclusion; a marker present at the session root opts the
preset in. Set `lsp_auto_discover = false` to disable preset discovery entirely
and use only explicitly-declared `[[lsp_servers]]` entries (MCP-style explicit
config).

`/lsp` shows configured-server status (state + extensions + last error).

### Connectors

Mistral connectors are auto-discovered when the active provider is Mistral
and the API key env var is set. Toggle the master switch or hide individual
connectors / tools:

```toml
enable_connectors = true          # Master switch (default: true)

[[connectors]]
name = "github"
disabled = true                   # Hide all tools from this connector

[[connectors]]
name = "linear"
disabled_tools = ["delete_issue"] # Hide selected tools only
```

### Session Logging

```toml
[session_logging]
enabled = true
save_dir = ""                     # Defaults to ~/.vibe/logs/session
session_prefix = "session"
```

### Browser Sign-In

Browser sign-in lets users authenticate through the browser during onboarding.
Mistral providers use default browser sign-in URLs. Custom or renamed providers
must configure both URLs:

```toml
[[providers]]
browser_auth_base_url = "https://console.mistral.ai"
browser_auth_api_base_url = "https://console.mistral.ai/api"
```

Self-hosted deployments can point Vibe CLI upgrade and API-key links to their
Le Chat web deployment, where the Vibe API key is managed:

```toml
vibe_base_url = "https://chat.mistral.ai"
```

### Hooks (Experimental)

Hooks let users run shell commands automatically at lifecycle events.
**Experimental**, enabled with `enable_experimental_hooks = true` in
`config.toml` or `VIBE_ENABLE_EXPERIMENTAL_HOOKS=true`.

#### Config and hook types

Hooks live in `hooks.toml` files (separate from `config.toml`), discovered in
this order:

1. `<root>/.vibe/hooks.toml` for each trusted project root — loaded first (only
   when that root is trusted). With multiple `--add-dir`s, one project hook file
   per root is appended before the user file.
2. `~/.vibe/hooks.toml` — loaded second.

A duplicate `name` across the two files is reported as a config issue and the
project entry wins. Config-load errors (invalid TOML, missing required
fields) surface in the TUI as warnings and the offending hook is skipped.

```toml
[[hooks]]
name = "lint"                       # Required: unique within the file.
type = "post_agent_turn"            # Required. User-facing: post_agent_turn | before_tool | after_tool | user_prompt_submit. (Team/lifecycle hooks: teammate_idle | task_created | task_completed. Session hooks: session_start | session_end | stop | pre_compact.)
command = "eslint --quiet ."        # Required: shell command run in cwd.
timeout = 60.0                      # Default: 60s for all hooks.
description = "Run ESLint"          # Optional.

[[hooks]]
name = "deny-rm-rf"
type = "before_tool"
match = "bash"                      # Tool-name matcher (tool hooks only, default "*").
strict = true                       # Tool hooks only: escalate any failure to deny/clear.
command = "uv run python /path/to/guard-bash"
```

| Type | When it runs |
|---|---|
| `post_agent_turn` | Once per turn, after the agent finishes responding (no pending tool calls). |
| `before_tool` | Per tool call, before the user permission prompt. |
| `after_tool` | Per tool call, **iff the tool body actually ran**. `tool_status` is `success`, `failure`, or `cancelled`. Does not fire when the tool never executed (`before_tool` denial, user denial at the approval prompt, permission `NEVER`, or cancellation before the body started). |

**Matcher syntax** (same as `enabled_tools`): fnmatch glob by default
(`"bash"`, `"read_*"`, case-insensitive), or a regex full-match when the
pattern starts with `re:` (`"re:(read_file|grep)"`). `match` is forbidden on
`post_agent_turn`.

**Tool name conventions** for matchers:
- Built-in tools use their bare name (`bash`, `read_file`, …); see the Tools
  section above for the full list.
- MCP tools: `{server-name}_{raw-tool-name}` (e.g. `linear_create-issue`).
- Connector tools: `connector_{normalized-name}_{remote-tool-name}` (e.g.
  `connector_Google_Drive_search_files`).
- Subagents all route through `task`. Match with `match = "task"` and read
  `tool_input.agent` to discriminate by subagent.

Subagent invocations inherit the parent's hook config. Their hook events are
logged to the subagent's session log and don't propagate to the parent's UI.

#### MCP context injection (user_prompt_submit)

A `user_prompt_submit` hook can inject MCP-management context when the user's
prompt mentions MCP — so the agent answers MCP questions with focused, lower-
token guidance instead of loading the full skill. The hook runs before any LLM
turn (token-free for the model), sees `{"prompt": "..."}` on stdin, and returns
`hook_specific_output.additional_context` to append context for that turn.

`~/.vibe/hooks.toml`:

```toml
[[hooks]]
name = "mcp-context"
type = "user_prompt_submit"
command = "python ~/.vibe/hooks/mcp_context.py"
timeout = 5.0
```

`~/.vibe/hooks/mcp_context.py`:

```python
import json, sys

KEYWORDS = ("mcp server", "mcp ", "model context protocol", "connector", "oauth")
SNIPPET = (
    "MCP servers: manage with token-free slash commands — "
    "`/mcp` (browser + status), `/mcp login|logout <name>`, "
    "`/mcp refresh`, `/mcp add`. OAuth servers show `needs login` or "
    "`logged in` status. Configure via `[[mcp_servers]]` in config.toml."
)

def main() -> None:
    data = json.load(sys.stdin)
    prompt = (data.get("prompt") or "").lower()
    if any(kw in prompt for kw in KEYWORDS):
        json.dump(
            {"decision": "allow",
             "hook_specific_output": {"additional_context": SNIPPET}},
            sys.stdout,
        )
    else:
        json.dump({"decision": "allow"}, sys.stdout)

if __name__ == "__main__":
    main()
```

#### Wire protocol

Every hook is spawned in `cwd` and receives a JSON object on **stdin**
discriminated by `hook_event_name`:

```json
// post_agent_turn
{"hook_event_name": "post_agent_turn", "session_id": "...",
 "parent_session_id": null, "transcript_path": "...", "cwd": "..."}

// before_tool
{"hook_event_name": "before_tool", "session_id": "...", "parent_session_id": null,
 "transcript_path": "...", "cwd": "...",
 "tool_name": "bash", "tool_call_id": "call_42",
 "tool_input": {"command": "ls"}}

// after_tool
{"hook_event_name": "after_tool", "session_id": "...", "parent_session_id": null,
 "transcript_path": "...", "cwd": "...",
 "tool_name": "bash", "tool_call_id": "call_42",
 "tool_input": {"command": "ls"},
 "tool_status": "success",         // success | failure | cancelled
 "tool_output": {"stdout": "..."},  // structured result (success/cancelled); null otherwise
 "tool_output_text": "...",         // current text the LLM will see; mutable by prior hooks
 "tool_error": null,                // populated on failure/skipped
 "duration_ms": 42.5}
```

`parent_session_id` is set when running inside a subagent. Exceeding
`timeout` kills the whole process tree.

A hook signals back via its **exit code** and **stdout** (stderr is reserved
for diagnostics — Vibe never parses it for control):

| Exit | Stdout | Behavior |
|---|---|---|
| `0` | empty | Pass through (no action). |
| `0` | valid structured-response JSON object (schema below) | Act per the JSON fields. |
| `0` | anything else (free-form text, broken JSON, scalar/array, schema mismatch) | Failure path (see below). The parse error is in the message. |
| non-zero / timeout / spawn failure | — | Failure path. Reason taken from stderr, then stdout, then the exit code. |

Structured-response schema:

```json
{
  "decision": "allow" | "deny",          // optional; default "allow"
  "reason": "string",                     // required when decision == "deny"
  "system_message": "string",             // optional UI note
  "hook_specific_output": {
    "tool_input": { ... },                // before_tool only
    "additional_context": "string"        // after_tool only
  }
}
```

Unknown fields are tolerated at every level. Fields that aren't meaningful
for the current hook type are silently ignored.

**Don't self-name in `system_message` or `reason`** — the UI prefixes
hook-end-event content with `[hook-name]` automatically, and `before_tool`
denials are wrapped as ``Tool 'X' was denied by hook 'Y': {reason}`` before
the LLM sees them. A hook that writes ``"reason": "guard: refused..."``
will produce ``hook 'guard': guard: refused...`` downstream.

`decision: "deny"` per hook type:

| Hook | Effect of `decision: "deny"` |
|---|---|
| `before_tool` | Deny the tool call; `reason` is the tool error returned to the LLM. First deny short-circuits the remaining `before_tool` hooks for this call. |
| `after_tool` | Replace `tool_output_text` with `reason`. Pipeline continues; subsequent hooks see the replacement. |
| `post_agent_turn` | Inject `reason` as a retry user message. Capped at 3 retries per hook per user turn. |

Event-specific payloads:

- `hook_specific_output.tool_input` (`before_tool`): full replacement of the
  model's arguments. Vibe re-validates against the tool's schema **after each
  rewriting hook** — the first invalid rewrite aborts the chain and
  synthesizes a denial attributing the failure to that hook. Rewrites
  compose: hook N receives `tool_input` as rewritten by hooks 1..N-1.
- `hook_specific_output.additional_context` (`after_tool`): text appended
  (with `\n`) to the current `tool_output_text`. Composes with a same-hook
  `decision: "deny"`: deny replaces first, then `additional_context` is
  appended to the replacement.

**Failure path.** Any failure (non-zero exit, timeout, spawn failure,
non-conforming stdout) emits a UI warning and lets the gated action proceed
(fail open). With `strict = true` on a tool hook:

| Hook | Strict failure escalates to |
|---|---|
| `before_tool` | Deny the tool call with the failure reason. |
| `after_tool` | Clear `tool_output_text` (replace with empty). |

`strict` is forbidden on `post_agent_turn`.

#### Execution semantics

- Hooks of the same type fire sequentially in load order (project file first,
  then user file; declaration order within each file).
- Tool calls within a single LLM turn run **concurrently**; each call's hook
  chain runs serially but the chains run in parallel across calls. Hooks
  that touch shared state (filesystem, env) must coordinate themselves.
- `before_tool` rewrites take effect everywhere downstream: the user
  permission prompt sees the rewritten arguments, the tool runs with them,
  and the assistant message is patched so subsequent LLM turns reflect what
  actually ran.

### Memory

Cross-session memory stores durable notes as plain `*.md` files (YAML
frontmatter + body) under `~/.vibe/memory/`. Each turn, a selector (its own
standalone backend, like the safety judge) scans only the lightweight
frontmatter index and injects up to `max_selected` relevant bodies into the
system prompt. Selection fails open (no memories) on any error.

```toml
[memory]
enabled = true                   # Master switch
select_mode = "per-turn"         # "per-turn" | "per-session" | "always"
model = ...                      # Alias; default unset (None) — falls back to compaction, then active
max_selected = 5                 # Top-K injected
max_inject_chars = 8000          # Hard cap on total injected body text
max_entries_scanned = 200        # Cap on index lines sent to the selector
timeout = 20.0                   # Per-selection LLM timeout
prefetch = true                  # Warm the selector before the turn needs it
inject_mode = "append"           # How selected bodies are attached to the prompt
# Auto-extraction (write memories from conversation) and consolidation (merge
# similar memories) families are also configurable: auto_extract,
# auto_extract_model, auto_extract_max_writes, auto_extract_min_messages,
# auto_extract_timeout, consolidate, consolidate_model, consolidate_min_age_days,
# consolidate_min_candidates, consolidate_interval_days, consolidate_max_actions,
# consolidate_timeout. See vibe/core/config/_settings.py (MemoryConfig) for defaults.
```

**Scopes.** The `manage_memory` tool defaults new memories to the current
trusted project's private namespace (`~/.vibe/memory/projects/<hash>/`) when one
is active, so project-specific facts stay scoped to that project and don't leak
into others. Pass `scope = "user"` to write a global memory shared across every
project — reserve this for cross-project identity, preferences, and feedback.
Project memories live under `~/.vibe` (never in the repo), so they cannot be
committed; they shadow same-id global memories for that project only. Project
scope requires a trusted project directory; without one, new memories fall back
to global.

**Multi-session / multi-agent.** The project namespace is keyed by the repo's
git common dir, not the working-directory path, so every session and every git
worktree of one repository shares the same project memory. Different repos (and
non-git directories) stay isolated. To run several agents on one project without
colliding on git's shared index, give each its own worktree
(`git worktree add <path>`) — memory follows the repo, so all worktrees see the
same notes.

### Pattern Matching

Tool, skill, and agent names support three matching modes:
- **Exact**: `"bash"`, `"read"`
- **Glob**: `"bash*"`, `"mcp_*"`
- **Regex**: `"re:^serena_.*$"` (full match, case-insensitive)

## CLI Parameters

```
vibe [PROMPT]                       # Start interactive session with optional prompt
vibe -p TEXT / --prompt TEXT         # Programmatic mode using `default_agent`, one-shot, exit
vibe -p TEXT --auto-approve          # Programmatic mode with all tool calls approved
vibe -p TEXT --keep-alive SECONDS    # Keep firing scheduled-loop turns for SECONDS before exiting (-p only)
vibe --agent NAME                   # Select agent profile (falls back to `default_agent` config)
vibe --model ALIAS                  # Active model for this session (overrides `active_model`; also threaded into isolated subagents)
vibe --auto-approve / --yolo         # Shortcut for `--agent auto-approve`
vibe --workdir DIR                  # Change working directory
vibe --add-dir DIR                  # Extra working dir loaded for context (repeatable). Implicitly trusted.
vibe --trust                        # Trust cwd for this invocation only (not persisted)
vibe -c / --continue                # Continue most recent session in this terminal (TTY-scoped, falls back to latest in cwd)
vibe --resume [SESSION_ID]          # Resume a session (no ID opens an interactive picker)
vibe -v / --version                 # Show version
vibe --setup                        # Run setup (configure API key) and exit
vibe --check-upgrade                # Check for a Vibe update, prompt to install, and exit
vibe --max-turns N                  # Max assistant turns (programmatic mode)
vibe --max-price DOLLARS            # Max cost limit (programmatic mode)
vibe --max-tokens N                 # Max total session tokens (programmatic mode)
vibe --enabled-tools TOOL           # Enable specific tools (repeatable; under -p, disables all others)
vibe --output text|json|streaming   # Output format (programmatic mode)
vibe --worktree                     # Force worktree isolation ON (overrides mode="off")
vibe --no-worktree                  # Force worktree isolation OFF for this invocation
```

Worktree isolation is **on by default** for the interactive CLI and `vibe -p`:
writes land on a throwaway branch that is merged back into the original HEAD on
clean exit — rebased onto the latest HEAD first (so concurrent sessions don't
strand it), then fast-forwarded, including when the original tree was dirty at
start. The branch is kept for recovery only if it genuinely conflicts with
another session's changes; land it with `vibe worktree merge <branch>` (or
discard with `vibe worktree discard <branch>`). Set `worktree.mode = "off"` in
config to disable persistently, or `"auto-by-entrypoint"` for the legacy
programmatic-only split. ACP is not isolated (multi-session-per-process; tracked
as a follow-up).

The `vibe worktree` subcommand manages stranded branches outside the TUI
(dispatched before the main parser, so it works on a fresh checkout):

- `vibe worktree list` — show worktrees and any `vibe/*` branches holding
  unmerged work from prior sessions (also printed as a startup notice when
  stranded work exists).
- `vibe worktree merge <branch>` — land a branch into HEAD. Rebases onto HEAD
  first, then fast-forwards; aborts cleanly (keeping the branch) on a real
  conflict.
- `vibe worktree discard <branch>` — delete a branch that is no longer wanted
  (forces deletion of unmerged work; prompts unless `--force`).

## Built-in Agents

There are two kinds of agents:
- **Agents** are user-facing profiles selectable via `--agent` or `Shift+Tab`.
  They configure the model's behavior, tools, and system prompt.
- **Subagents** are model-facing: the model can spawn them autonomously to delegate
  subtasks (e.g. exploring the codebase). Users cannot select subagents directly.

### Agents

- **default**: Standard interactive agent; requires approval for tool executions
- **chat**: Read-only conversational mode (grep/read/`ask_user_question`/`task`); no file edits or shell
- **plan**: Read-only planning sandbox (writes only the plan file at `~/.vibe/plans/`)
- **accept-edits**: Auto-approves file edits but asks for other tools
- **auto-approve**: Auto-approves all tool calls
- **coordinator**: Orchestration-only lead (read/grep/glob + `task`/`launch_workflow`/`team`); cannot write files or run bash directly
- **lean**: Specialized Lean 4 proof assistant. Not available by default — must be
  installed with `/leanstall` (removed with `/unleanstall`)

### Plan Mode

Plan mode is a read-only sandbox (the `plan` profile) for researching a task and
drafting an implementation plan before edits. The plan file at
`~/.vibe/plans/<session>.md` (Ctrl+G to edit live) is the only writable target;
every turn injects a hard reminder forbidding other edits and non-readonly tools.

Entry — any of:
- User: `Shift+Tab`, `--agent plan`, or `default_agent = "plan"` in config.toml.
- Agent: calls the `enter_plan_mode` tool when it judges a task warrants planning
  (multi-file refactors, new features, architecturally significant work). Available
  in default / accept-edits / auto-approve; not available in plan or chat.

Exit — any of:
- User: `Shift+Tab` to cycle out.
- Agent: calls the `exit_plan_mode` tool once the plan file is ready (offered only
  in the plan profile).

Both agent-initiated transitions present a confirmation dialog, so the human
authorizes the plan↔execute boundary. Neither tool is available in programmatic
(`-p`) or ACP non-interactive sessions.

### Subagents

- **explore**: Read-only codebase exploration subagent (grep + read + lsp only).
  Spawned by the model, not selectable by the user.
- **research**: read-only web research (adds web search/fetch).
- **planner**: read-only (grep/read); returns a phased, code-grounded plan.
- **reviewer**, **debugger**, **security**: investigation/audit subagents that add a
  **jailed read-only bash** (allowlist auto-runs tests/lint/git-inspection; denies
  mutations, network, and package installs) — their key differentiator from explore/planner.
- **editor**: workflow-only subagent for surgical file edits (read/grep/lsp + write/edit,
  no bash/MCP). Requires `isolation="worktree"`.
- **worker**: full-capability workflow subagent (all builtin tools + MCP, no allowlist).
  Requires `isolation="worktree"`; bash is auto-confined to the worktree by the OS
  sandbox (bwrap) when one is available.
- **grunt**: write-capable subagent for bulk/mechanical work (renames, codemods,
  repetitive edits). Full tool surface like worker but routes onto a cheap model
  by default (`grunt_model`, falling back to `subagent_model` then the host) and
  ships a no-decisions prompt. Requires `isolation="worktree"`. Composes as the
  executor in a thinker-plan / grunt-execute / verifier-gate split.
- **verifier**: verdict-oriented gate that proves a *completed* implementation
  works by trying to break it, emitting a strict PASS/FAIL/PARTIAL verdict with
  command evidence. The host verification contract (on by default via
  `verification_subsystem`) requires spawning it before reporting non-trivial
  work done; the todo tool appends a nudge when a 3+ item list closes without a
  verify step.

Custom agents are TOML files in `~/.vibe/agents/NAME.toml`.

### Async subagents (background delegation)

The `task` tool accepts `async_run=true` (default) to delegate work to a
subagent in the background and return immediately with a `task_id` of the form
`asub-N`, instead of blocking the turn for the result. The result is delivered
to the host automatically when the subagent finishes. Use it for fan-out —
spawn what you need, then keep working or end the turn; the completion surfaces
at the top of a later turn. Isolated (write-capable) async subagents stream
their stdout to a log file under the scratchpad; in-process ones stream their
partial response. Either way the Tasks pane and the `background` tool show the
agent, model, elapsed time, turns used, worktree/branch, the prompt, and a live
log tail / streaming response while the subagent runs — so a long-running
background agent is observable, not a blind `asub-N` row.

## Built-in Slash Commands

- `/help` - Show help message
- `/config` - Edit config settings
- `/model` - Select active model
- `/thinking` - Select thinking level
- `/theme` - Select Textual UI theme (persisted in config)
- `/reload` - Reload configuration, agent instructions, and skills from disk
- `/clear` - Clear conversation history
- `/log` - Show path to current interaction log file
- `/debug` - Toggle debug console
- `/compact` - Compact conversation history by summarizing (optionally pass instructions to guide the summary)
- `/status` - Display agent statistics
- `/copy` - Copy the last agent message to the clipboard
- `/paste-image` - Paste an image from the OS clipboard into the prompt (supported platforms only)
- `/rename` - Rename the current session
- `/voice` - Configure voice settings
- `/mcp` (or `/connectors`) - Display available MCP servers and connectors. Pass a name
  to list its tools or open its auth panel when authentication is required.
  Subcommands: `/mcp login|logout <name>`, `/mcp refresh`, `/mcp add`.
- `/mcp add <url>` - Add a hosted OAuth MCP server. Supports `--name <alias>`,
  repeatable `--scope <scope>`, `--transport <http|streamable-http>`, and
  `--no-login`. Starts OAuth login by default. OAuth-only; use `config.toml`
  for API-key/static auth.
- `/mcp status` - Display MCP auth state (`ok`, `needs_auth`, `static`, `stdio`)
- `/mcp login <alias>` - Start OAuth login for an MCP server
- `/mcp logout <alias>` - Log out from an MCP server and delete stored OAuth
  secrets
- `/resume` (or `/continue`) - Browse and resume past sessions for the current
  folder (plus active remote sessions when Vibe Code is enabled). The picker
  header shows the folder being listed. Press `D` twice to delete a local saved
  session; remote sessions and the active session cannot be deleted here.
- `/rewind` - Rewind to a previous message
- `/loop <interval> <prompt>` - Schedule a recurring prompt (e.g. `/loop 30s ping`).
  Intervals: `Ns/Nm/Nh/Nd`, minimum __MIN_INTERVAL_S__s, max __MAX_LOOPS__ loops/session.
  - `/loop` (or `/loop list` / `/loop ls`) - List current scheduled loops.
  - `/loop cancel <id|all>` (aliases `rm`, `stop`, `delete`) - Cancel a loop.
  - Loops fire only when the agent is idle and the input bar is focused. At
    most one loop fires per poll. Overdue loops fire once on the next poll
    (no catch-up); `next_fire_at` advances to `now + interval`.
  - Loops are persisted in the session metadata (`loops` field of `meta.json`)
    and restored on `--resume`/`--continue`.
- `/proxy-setup` - Configure proxy and SSL certificate settings
- `/leanstall` - Install the Lean 4 agent (leanstral)
- `/unleanstall` - Uninstall the Lean 4 agent
- `/lspstall` - Install the LSP code-intelligence feature (enables the `lsp` tool and passive diagnostics)
- `/unlspstall` - Uninstall the LSP feature
- `/lsp` - Show LSP feature and configured-server status
- `/data-retention` - Show data retention information
- `/teleport` - Teleport session to Vibe Code Web (only available when Vibe Code is enabled)
- `/effort` - Select effort mode: `normal` (turn-by-turn) or `le-chaton` (max thinking + auto-workflow planning)
- `/tasks` (or `/workflows`, `Ctrl+W`) - Unified background-task manager (processes,
  workflows, teams, loops). Stays usable while the agent is busy or the queue is
  paused — that is when you need to watch or stop a background run.
  - `/tasks` (no args) - Open the task progress view
  - `/tasks list` (or `/workflows list`) - List all runs with status, agents, tokens, elapsed
  - `/tasks stop <id|all>` (or `/workflows stop <id|all>`) - Stop one or all runs
  - `/workflows snapshot <id>` - Show cached results for a run
  - `/workflows resume <run-id>` - Resume a stopped/finished run
- `/worktree` - Show worktree isolation status, diff, or trigger merge
  (`/worktree status`, `/worktree diff`, or `/worktree merge`)
- `/team` - Manage agent teams (multi-process coordination).
  - `/team list` - Show teammates with name, status, PID
  - `/team spawn <name> <prompt>` - Spawn a teammate as a separate vibe process
  - `/team stop <name|all>` - Stop one or all teammates
  - `/team cleanup` - Remove team directory and reset state
- `/<workflow-name> [args]` - Run a discovered workflow script (e.g. `/deep-research <question>`)
- `/exit` - Exit the application

## File Mentions (`@`)

Type `@` in the chat input to autocomplete files and folders from the
project tree. Pressing Tab/Enter inserts the chosen path. Behavior
depends on the file kind:

- **Text files** are read at submit and their contents are inlined into the
  prompt text (up to ~256KB).
- **Folders** are inserted as a resource link header (name + uri).
- **Image files** (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) become image
  attachments — sent alongside the prompt as native multimodal content for
  vision-capable models.

Image attachments:

- Require `supports_images = true` on the active model in `config.toml`.
  By default this is enabled only on `mistral-vibe-cli-latest`. Sending
  images to a non-vision model raises a clear error and the message is
  not added to the conversation.
- Snapshotted into `<session_dir>/attachments/<sha1>.<ext>` so that
  resumed sessions stay reproducible even if the source file is moved.
- Capped at 10 MB per image and 8 images per message.
- Out-of-project paths work via `@/abs/path/to.png` (the picker only
  suggests project files, but the `@`-parser accepts absolute paths).
  Drag-and-drop from Finder into Terminal, iTerm2, or Ghostty is
  intercepted at paste time: if the pasted content is a single bare
  path to an image file (raw, `\\ `-escaped, or quoted), the input
  automatically prepends `@` (and quotes paths containing spaces).
  Non-image paths are pasted verbatim so non-image use cases are not
  affected.
- Rendered in the chat bubble as a dim footer line linking each
  attachment to its snapshot. Clicking opens the file with the OS
  default image viewer.

## Input Queue

Messages submitted while the agent or a `!`-bash command is running are
queued instead of cancelling the in-flight work, and drain in FIFO order
once the job finishes. Prompts (plain, `/skill ...`, `@`-mentions) and
`!bash` commands can be queued; slash commands and `&teleport` are
rejected with a toast. **Ctrl+C** pops the last queued item (LIFO);
**Esc** interrupts the running job and pauses the queue; pressing Enter
(empty or not) on a paused queue resumes draining.

## Skills System

Skills are specialized instruction sets the model can load on demand.
Each skill is a directory containing a `SKILL.md` file with YAML frontmatter.

### Skill File Format

```markdown
---
name: my-skill
description: What this skill does and when to use it.
summary: One-line trigger shown in the skills index (optional; defaults to the first sentence of description).
user-invocable: true
allowed-tools: bash read
---

# Skill Instructions

Detailed instructions for the model...
```

Only `name` and `description` are required. The system prompt loads a short
trigger line per skill (the `summary`, or the first sentence of `description`);
the `skill` tool loads the full instructions on demand.

### Skill Search Order (first match wins)

Built-in skills are reserved: a custom skill whose name collides with a builtin
is silently skipped, so the order below only resolves custom-vs-custom
collisions. (Custom *agents* can override builtins on collision — skills cannot.)

1. `skill_paths` from config.toml
2. `.vibe/skills/` in trusted project roots (cwd trust root + each `--add-dir`)
3. `.agents/skills/` in trusted project roots
4. `~/.vibe/skills/` (user global)
5. `~/.agents/skills/` (user global, Agent Skills standard)

**Plugins.** A plugin manifest (`plugin.toml` under `<root>/.vibe/plugins/*/` or
`~/.vibe/plugins/*/`) additively extends every `*_paths` list (skills, tools,
agents, workflows, prompts) and union-merges MCP servers — so plugin-supplied
dirs sit at the same precedence as the matching `*_paths` entry above.

## Environment Variables

- `VIBE_HOME` - Override the Vibe home directory (default: `~/.vibe`)
- `MISTRAL_API_KEY` - API key for Mistral provider
- `OPENAI_API_KEY` - API key for the OpenAI provider (platform.openai.com,
  pay-per-token). Not required for the "Sign in with ChatGPT" subscription
  flow, which stores OAuth tokens in `$VIBE_HOME/auth/openai.json`.
- `VIBE_ACTIVE_MODEL` - Override active model
- `VIBE_*` - Any config field can be overridden with the `VIBE_` prefix
- `LOG_LEVEL` - Logging level for `$VIBE_HOME/logs/vibe.log`. One of `DEBUG`,
  `INFO`, `WARNING` (default), `ERROR`, `CRITICAL`. Invalid values fall back
  to `WARNING`.
- `LOG_MAX_BYTES` - Max size in bytes of `vibe.log` before rotation
  (default: `10485760`, i.e. 10 MiB).
- `DEBUG_MODE` - When `true`, forces `DEBUG`-level logging. Under `vibe-acp`
  it also attaches `debugpy` on `localhost:5678`.
- `VIBE_TYPING_GRACE_PERIOD_MS` - Milliseconds the agent waits for a typing
  pause before showing tool-approval / ask-user-question dialogs (default:
  `1000`). Set to `0` to disable. Negative or non-numeric values fall back
  to the default.
- `VIBE_EFFORT_MODE` - Override effort mode (`normal` or `le-chaton`)
- `VIBE_DISABLE_WORKFLOWS` - Set to `1`/`true` to disable all workflow features
- `VIBE_TEAM_NAME` - Team name (set by TeamManager when spawning teammates)
- `VIBE_TEAM_DIR` - Team directory path (set by TeamManager when spawning teammates)
- `VIBE_TEAMMATE_NAME` - Teammate name (set by TeamManager when spawning teammates)
- `VIBE_THEME` - Override the terminal theme name (same effect as `/theme`)
- `VIBE_PROFILE` - Set to `1` to activate the per-request profiler
- `VIBE_TRACE_LOOP` - Threshold in seconds; logs event-loop callbacks that block
  longer than it to a per-PID perf log in the log dir
- `VIBE_TRACE_STREAM` - Set to `1` to log one streaming-responsiveness summary
  per turn (TTFB, TTFR, chunk gaps) to the per-PID perf log
- `VIBE_ACP_LOGGING_ENABLED` - Set to `1`/`true`/`yes` to enable `vibe-acp` transport logging
- `VIBE_LSP_TRACE` - Set to `1`/`true` to trace LSP JSON-RPC traffic (debug)
- `OLLAMA_HOST`, `OLLAMA_CONTEXT_LENGTH` - Honored by the ollama provider to set the
  server URL and default context length

## API Keys (.env file)

The `.env` file in VIBE_HOME stores API keys in dotenv format:

```
MISTRAL_API_KEY=your-key-here
```

This file is loaded on startup and its values are injected into the environment.

## Trusted Folders

Vibe uses a trust system to prevent executing project-local config from untrusted
directories. The trust database is stored in `~/.vibe/trusted_folders.toml`.
Project-local config (`.vibe/` directory) is only loaded when the current
directory is explicitly trusted.

Interactive mode prompts to trust unknown folders. The prompt targets the
closest ancestor of the cwd (the cwd itself included) containing a `.git`
entry; the search excludes the user's home directory and the filesystem
root, and falls back to the cwd if no qualifying ancestor is found.
Programmatic mode (`-p`/`--prompt`) never prompts: the folder is untrusted.
Use `--trust` to trust cwd for the current invocation only (not persisted).

## Sensitive Files — DO NOT READ OR EDIT

NEVER read, display, or edit any of these files:
- `~/.vibe/.env` (or `$VIBE_HOME/.env`) — contains API keys and secrets
- Any `.env`, `.env.*` file in the project or VIBE_HOME

If the user asks to set or change an API key, instruct them to edit the `.env`
file themselves. Do not offer to read it, write it, or display its contents.
Do not use tools (read, write_file, bash cat/echo, etc.) to access these files.

## Workflows

Workflows are Python scripts that orchestrate parallel agents. They run in the
background as asyncio tasks on the same event loop as the TUI, so the session
stays responsive while agents work.

### Workflow Scripts

A workflow script is a `.py` file with an `async def main()` function. Optional
YAML frontmatter (`name:`, `description:`) precedes the Python source. The
runtime injects these functions into the script's namespace:

- `agent(prompt, *, agent="explore", model=None, label=None, phase=None, schema=None, budget_estimate=None, isolation=None, contract=None)` — spawn a subagent. Pass `isolation="worktree"` to run the agent as a `vibe -p` subprocess in a fresh git worktree (for parallel file-mutating agents that would otherwise conflict); its branch is kept for manual `git merge` if it changed files, else removed. Note: isolated agents run auto-approved/trusted (no interactive prompts reach a subprocess) — the worktree bounds file conflicts, not arbitrary command execution. Agent profiles: `explore` (grep/read), `research` (+web), `reviewer` (+bash), `debugger` (+bash; root-cause analysis), `planner` (grep/read; phased plan), `security` (+bash; vuln audit), `editor` (read/grep/write/edit, no bash/MCP — **requires** `isolation="worktree"`), `grunt` (full tool set like worker, but routes onto a cheap model via `grunt_model` and ships a no-decisions prompt for bulk/mechanical work — **requires** `isolation="worktree"`), or `worker` for the full tool set including any configured MCP tools (no allowlist — **requires** `isolation="worktree"`, where it runs auto-approved in its own worktree so its tools actually execute and writes can't race other agents). Pass `contract={...}` (requires `isolation="worktree"`) to validate the agent's FILES before delivery: `outputs` (path/must_contain/must_not_contain/must_match/must_not_match/min_size/max_size), `invariants` (grep pattern must or must not match across the tree), and `tests` (command + optional expected stdout). On pass the work is ff-merged into the parent repo; on fail a falsy `ContractFailure` (mirroring `SchemaValidationFailure`) carries the violations and the work is held back. Unlike `schema=`, which validates the agent's JSON return value, `contract=` validates the code it wrote. The bundled `/verify-contract` workflow demonstrates the pattern.
- `parallel(*items, max_concurrency=None)` (or `parallel([items])`) — run items concurrently, results in argument order; an item that raises yields `None` (filter the results), so one failure does not abort the batch. Each item may be a **coroutine** (`parallel(agent("a"), agent("b"))`) or a zero-arg thunk (`parallel(lambda: agent("a"))`) — both work, since Python coroutines are lazy and bound concurrency identically. Pass `max_concurrency=N` to cap in-flight items (e.g. `3` when a provider limits concurrency) instead of hand-rolling chunked waves.
- `pipeline(items, *stages, max_concurrency=None)` — run each item through all stages independently with no barrier between stages (item A can be in stage 3 while item B is still in stage 1); each stage receives `(prev, item, index)` and a stage that raises drops that item to `None`. A single stage behaves as a concurrent map. `max_concurrency=N` caps in-flight items.
- `phase(name)` — declare a phase for progress tracking
- `log(msg)` — log a progress message
- `budget` — token budget object with `.total` (int|None) and `.remaining()` (int|float)
- `workflow(name, args=None)` — run another discovered workflow inline as a sub-step and return its result; it shares this run's budget, agent counter, and result cache, and its phases merge into the live monitor. Nesting is one level deep — calling `workflow()` inside a nested run raises.
- `post_message(channel, message)` — post to a named channel on this run's shared in-process message board. Visible to every agent/stage in the same run via `fetch_messages`. Use for inter-agent handoffs that don't fit the barrier-return model (e.g. a finder posting partial results a verifier polls for).
- `fetch_messages(channel)` — return a copy of all messages posted to a channel so far.
- `flatten(items)` — flatten one level of nested lists (strings/dicts/bytes are atoms, not iterated): `flatten([[1,2],[3]]) == [1,2,3]`.
- `dedup_by(items, key)` — drop duplicates keeping the first occurrence; `key` maps each item to a hashable identity (e.g. `lambda f: f"{f['file']}:{f['line']}"`). Items whose key raises are kept as unique by `id()`.
- `merge_by(items, key, merge)` — group by `key` and fold each group via `merge(acc, item)` (acc starts at the first item); returns one merged value per key in first-seen order. Use to union findings, sum counts, or pick the highest-scored item per group.
- `args` — structured input from the invocation command (string or None)

Scripts are validated via AST before execution — for **safety and correctness**: it
rejects undefined names and a coroutine used as a `pipeline` stage
(`pipeline(items, agent(...))` — use `lambda x: agent(...)`; `parallel(agent(...))`
is fine). Scripts run in a restricted namespace. The non-obvious rules: a fixed set
of modules is **pre-bound — no import needed** (`json.dumps(...)` just works):
`json`, `re`, `math`,
`statistics`, `collections`, `itertools`, `functools`, `datetime`, `decimal`,
`copy`, `hashlib`, `base64`, `textwrap`, `unicodedata` — there is **no `asyncio`**
(you don't need it; `agent`/`parallel`/`pipeline` are injected and awaitable),
and no `os`/`sys`/`subprocess`/`pathlib`/`io`. `str.format()` and `str.format_map()`
are **forbidden** (the format mini-language traverses attributes/dunders from
inside a string literal) — template with f-strings or `%` formatting instead.
Also blocked: `exec`/`eval`/`compile`/`open`/`input`/`getattr`/`setattr`/`globals`/
`vars`/`__import__`, all dunder access, and dunder dict keys. The builtins
namespace is safelisted (no `open`, `exec`, `__import__`).

### Workflow Discovery

Workflow scripts are discovered from (first match wins):
1. `workflow_paths` in config.toml
2. `.vibe/workflows/` in project roots
3. `~/.vibe/workflows/` (user global)
4. Bundled workflows shipped with the CLI

Discovered workflows are registered as `/<name>` slash commands. Custom
workflows override bundled ones on name collision.

### Bundled Workflows

- `/deep-research <question>` — fans out web searches across 5 angles, extracts
  claims with structured output, verifies each claim via pipeline, synthesizes a
  cited report from verified claims only.
- `/security-fix-verify <args>` — pre-merge gate for a security FIX branch.
  Refute-only (default-to-broken) per-finding panel + regression hunt; anything
  not provable from the repo (DB columns, runtime permissions, event shapes)
  hard-blocks as a runtime check. Emits a human review packet — never pushes.
  `args = {base, branch, findings:[{id, original, must_be_true, file, commit?}]}`.

### Launching Workflows

Three ways to launch:
1. `/<workflow-name> [args]` — run a discovered workflow script
2. `launch_workflow` tool — the model writes a script inline and launches it
   (gated by ToolPermission.ASK, so the user approves)
3. Le chaton effort mode — the model is instructed to write and launch workflows
   for substantive tasks

A launch returns only `{run_id, launched, delivery}` — the run is background.
The script's `return_value` and per-agent outputs are auto-delivered as a
message on completion (best-effort; capped at ~16KB; dropped if the host turn
already ended). Re-read the result any time with the `workflow_results` tool
(`workflow_results(run_id=...)`), which returns the structured `return_value`
plus per-agent `response`/`error`/`schema_errors`. For finished runs the return
value is persisted across sessions. Use `workflow_status` for live progress
only.

Inside the script, an `agent(schema=...)` whose output can't be validated after
retries returns a **falsy** `SchemaValidationFailure` (a `dict` subclass): filter
with `[r for r in results if r]` (NOT `isinstance(r, dict)`, which would now
wrongly include it), `r.get(k, default)` is safe, `json.dumps(results)` will not
crash, and `isinstance(r, SchemaValidationFailure)` / `r.schema_errors` expose
the detail — so one failed agent degrades the batch instead of crashing the run.

### Task Manager (background processes, workflows, teams, loops)

`/tasks` (or `/workflows`, or `ctrl+w`) opens the Tasks pane — a unified monitor
for everything running in the background. It aggregates six categories into one
list with a category filter: bash processes spawned with `background=true`,
workflow runs, in-flight workflow agents, async subagents (`task(async_run=true)`,
ids `asub-N`), teammates, and scheduled loops. Keys:

| Key | Action |
|---|---|
| `1`-`5` | Filter: All / Processes / Workflows / Teams / Loops |
| `Enter` | Drill into the highlighted task's detail (process detail tails its log) |
| `x` | Stop the focused task (routes to the right owner by id) |
| `p` | Pause/resume the focused workflow run — in-flight agents finish, new agents block until resumed |
| `s` | Save the focused workflow run's script as a reusable `/<name>` command |
| `o` | View a workflow run's full script source |
| `r` | Refresh |
| `Esc` | Back one level |

The task-id grammar routes stop/pause to the right owner: `proc-N` (bash
process), `wf-N` (workflow run), `wf-N/live-AGENT` (in-flight agent),
`asub-N` (async subagent), `team:NAME` (teammate), `loop-LOOPID` (scheduled loop).

### Backgrounding processes

The bash tool takes `background: bool`. When true, it spawns the command,
registers it in the background registry, redirects stdout/stderr to a log under
the scratchpad, and returns immediately with a `background_task_id` (e.g.
`proc-3`) and the OS pid — the agent turn does NOT block on the command. Use
this for dev servers, watchers, and any long-lived process. Tail the output via
the `background` tool (`action='list', tail=50`) or the Tasks pane, and stop it
via `background action='stop', task_id='proc-3'` or the pane's `x` key.
Backgrounded processes are reaped on app exit so a forgotten server doesn't
orphan.

### Live status from a model turn

The `background` tool (`action='list'`) returns the same unified view the Tasks
pane shows: every running process, workflow run, in-flight agent, teammate, and
loop, with elapsed time and status. Use `action='stop', task_id=...` to cancel
one. This is how you stay aware of — and manage — what is running in the
background without the TUI.

### Stop workflows from a model turn

The `workflow_stop` tool cancels one run (`run_id`) or every active run
(`all=true`), mirroring the `/tasks stop <id|all>` slash command but
callable from a model turn. It cancels the run's asyncio task, halting
in-flight agents immediately. Use it to recover from a runaway workflow (spend
climbing without bound, an agent stuck in a read loop) rather than waiting for
budget exhaustion. Already-finished or unknown runs report `stopped=false`
with a message rather than erroring. Gated by `ToolPermission.ASK`.

### Resumability

Completed agent results are cached keyed on `sha256(agent:phase:prompt)[:16]`.
On resume, cached results are returned without re-running the agent. Snapshots
are persisted to session metadata (`workflow_snapshots` field) for cross-session
recovery via `/workflows resume <run-id>`. Only completed agents are cached;
failed agents re-run on resume.

### Budget

The runtime uses pessimistic reservation: `reserve()` deducts an estimate at
spawn time so runaway loops can't spawn past the floor. `reconcile()` releases
the reservation and records actual spend. Overspend is discovered at completion,
not prevented at spawn. `budget.total = None` means unlimited.

### Concurrency

Up to __MAX_CONCURRENT_AGENTS__ concurrent agents, 1000 total per run (constructor defaults on
`WorkflowRuntime`; not exposed as a config.toml key). `parallel` and
`pipeline` share the same semaphore as `spawn_agent`. Pass `max_concurrency=N`
to either to cap in-flight work below the global __MAX_CONCURRENT_AGENTS__ (e.g. `3` when a provider
allows only a few concurrent agents) — prefer this over hand-rolling chunked
waves.

## Effort Modes

Effort mode controls how the agent approaches substantive tasks.

- **normal** (default): work turn-by-turn as usual.
- **le-chaton**: max thinking + automatic workflow planning. The system prompt
  gains a section instructing the model to write workflow scripts for substantive
  tasks (audits, migrations, multi-file refactors) instead of working
  turn-by-turn.

Select via `/effort` command or set `effort_mode = "le-chaton"` in config.toml.
Typing "le chaton" or "lechaton" in a prompt triggers le chaton mode for that
turn (keyword is stripped from the prompt text).

The `launch_workflow` tool is available whenever workflows are not disabled
(it is not gated on le chaton). Le chaton mode additionally injects the
workflow API documentation into the system prompt and raises the active
model's thinking to max, so the model is more likely to discover and use
the tool.

`disable_workflows = true` disables all workflow features: `/workflows` is
unavailable, workflow commands are not registered, le chaton mode cannot be
activated, and the `launch_workflow` tool is hidden.

## Agent Teams

Agent teams coordinate multiple independent Vibe instances working together.
Unlike subagents (which run in-memory within a single session) or workflows
(which run as asyncio tasks on the same event loop), teammates are **separate
OS processes** — each is a full `vibe -p` invocation with its own context window.

### Architecture

```
Lead session (interactive vibe)
  └── TeamManager
       ├── TeamConfig (config.json, filelock-protected)
       ├── TaskStore (tasks.json, filelock-protected)
       └── Mailbox (mailbox/<recipient>/<msg>.json, per-inbox filelock)
            │
    ┌───────┼───────────┐
    ▼       ▼           ▼
 Teammate1  Teammate2  Teammate3
 (vibe -p)  (vibe -p)  (vibe -p)
```

### Shared State

- **TaskStore**: file-backed task list with file locking. Tasks have id,
  description, status (pending/in_progress/completed/blocked), assignee, and
  dependencies. Claim/complete operations enforce dependency ordering.
- **Mailbox**: per-recipient inbox directories with JSON message files.
  Teammates message each other directly without going through the lead.
- **TeamConfig**: team metadata (members, status, PIDs) in config.json.

Teammates interact with the shared TaskStore and Mailbox through the `team`
builtin tool (available only inside a teammate, gated on `VIBE_TEAM_DIR`):
`list_tasks`, `available_tasks`, `claim_task`, `complete_task`,
`send_message`, `read_messages`, `unread_messages`.

### Lead ↔ teammate messaging

The lead (host) does **not** get the teammate `team` tool. To steer teammates
and collect their replies, the lead uses the `team_message` tool against the
same shared Mailbox: `send_message` (to a teammate by name),
`read_messages` / `unread_messages` (the lead's own `lead` inbox). It is
available only while a team is active (errors with "No active team" otherwise).
Teammates reach the lead by addressing messages to `lead`. Task distribution
remains available to the user via `/team task add|done|list`.

### Team Management

- `/team spawn <name> <prompt>` — spawn a teammate as `vibe -p` subprocess
- `/team list` — show all teammates with name, status, PID
- `/team stop <name|all>` — stop one or all teammates
- `/team cleanup` — remove team directory

Teammates are spawned with `VIBE_TEAM_NAME`, `VIBE_TEAM_DIR`, and
`VIBE_TEAMMATE_NAME` env vars so they can access the shared state. The team
directory (`~/.vibe/teams/<name>/`) is cleaned up on exit.

### Hook Events

Three hook event types exist for team lifecycle:
- `TeammateIdle` — fires when a teammate finishes and goes idle.
- `TaskCreated` — fires when the lead creates a task (`/team task add`).
- `TaskCompleted` — fires when the lead completes a task (`/team task done`).

Note: `TaskCreated`/`TaskCompleted` fire only for lead-initiated task ops. A
teammate runs in a separate `vibe -p` process and writes the shared task store
directly, so tasks it claims/completes do **not** fire these hooks in the lead.

## Structured Output

The `response_format` parameter is threaded through all backend layers:
`AgentLoop.act()` → `_chat()` → `backend.complete()` → `adapter.prepare_request()`.
Supported by Mistral (native `json_schema` response format) and OpenAI-compatible
backends. Anthropic/Vertex accept but ignore it (prompt fallback handles it).

When a schema is set, the backend enforces JSON schema at the API level. The
workflow runtime adds a second validation layer: parse the response as JSON,
validate against the schema, retry on mismatch (up to 2 retries with the
validation error fed back to the model).

## How to Modify Configuration

To help the user modify their Vibe configuration:

1. **Read current config**: Read the file at `~/.vibe/config.toml` (or the path
   from `VIBE_HOME` env var if set)
2. **Create a backup**: Before any edit, copy the file to `config.toml.bak` in the
   same directory (e.g. `cp ~/.vibe/config.toml ~/.vibe/config.toml.bak`). This
   applies to any config file you are about to modify (`config.toml`,
   `trusted_folders.toml`, agent TOML files, etc.)
3. **Edit the TOML file**: Make changes using the edit tool
4. **Reload**: The user can run `/reload` to apply changes without restarting

For API keys, tell the user to edit `~/.vibe/.env` directly — never read or
write that file yourself.

For project-specific configuration, create/edit `.vibe/config.toml` in the
project root (the folder must be trusted first)."""


VIBE_DOC_CAPSULE = SkillDocCapsule(
    name="vibe",
    description=(
        "Authoritative reference for Mistral Vibe, the CLI agent you (the model) run "
        "inside. Loading this is the default, not a last resort: cheaper to "
        "load and be right than to answer from memory and be wrong.\n\n"
        "LOAD THIS SKILL — do not answer from memory — when the user:\n"
        '- asks anything about Vibe/Mistral Vibe, even indirectly ("this CLI", '
        '"this tool", "you", "the agent", "the harness");\n'
        "- asks why you did or did not act, how you decide, or any meta "
        "question about your behavior, instructions, tools, skills, or "
        "context;\n"
        "- asks how to make the CLI do X, or where a "
        "flag/command/env var/setting/file lives;\n"
        "- wants to change, inspect, debug, or reset their setup.\n\n"
        "If you are tempted to reconstruct how the CLI works from source or "
        "recall, STOP and load this skill — it matches the installed version; "
        "your memory does not.\n\n"
        "SCOPE: config, env vars (VIBE_*/LOG_*), providers/models, agents, "
        "skills, tools and permissions, commands and flags, hooks, "
        "MCP/connectors, LSP, trusted folders, workflows, effort modes, "
        "teams, structured output."
    ),
    summary=(
        "MUST LOAD for any question about Vibe/Mistral Vibe itself, the agent's own "
        "behavior, or how this CLI works. Covers config, MCP, providers, "
        "commands, flags, hooks, workflows, ~/.vibe — the source of truth."
    ),
    user_invocable=False,
    # Substitute the runtime's caps from the single source of truth so this doc
    # never holds a stale copy (the concurrent-agent cap lives in
    # vibe.core.workflows.runtime; the schedule limits in vibe.core.loop).
    # __VIBE_VERSION__ is left intact for render_agent_prompt() to fill later.
    prompt_template=_PROMPT_TEMPLATE
    .replace("__MAX_CONCURRENT_AGENTS__", str(DEFAULT_MAX_CONCURRENT))
    .replace("__MIN_INTERVAL_S__", str(MIN_INTERVAL_SECONDS))
    .replace("__MAX_LOOPS__", str(MAX_LOOPS_PER_SESSION)),
)
SKILL = VIBE_DOC_CAPSULE.to_skill_info(__version__)
