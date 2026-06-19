# Mistral Vibe

[![PyPI Version](https://img.shields.io/pypi/v/mistral-vibe)](https://pypi.org/project/mistral-vibe)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/release/python-3120/)
[![CI Status](https://github.com/mistralai/mistral-vibe/actions/workflows/ci.yml/badge.svg)](https://github.com/mistralai/mistral-vibe/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/mistralai/mistral-vibe)](https://github.com/mistralai/mistral-vibe/blob/main/LICENSE)

```
██████████████████░░
██████████████████░░
████  ██████  ████░░
████    ██    ████░░
████          ████░░
████  ██  ██  ████░░
██      ██      ██░░
██████████████████░░
██████████████████░░
```

**Mistral's open-source CLI coding assistant.**

Mistral Vibe is a command-line coding assistant powered by Mistral's models. It provides a conversational interface to your codebase, allowing you to use natural language to explore, modify, and interact with your projects through a powerful set of tools.

> [!WARNING]
> Mistral Vibe works on Windows, but we officially support and target UNIX environments.

### One-line install (recommended)

**Linux and macOS**

```bash
curl -LsSf https://mistral.ai/vibe/install.sh | bash
```

**Windows**

First, install uv

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, use uv command below.

### Using uv

```bash
uv tool install mistral-vibe
```

### Using pip

```bash
pip install mistral-vibe
```

## Table of Contents

- [Features](#features)
  - [Built-in Agents](#built-in-agents)
  - [Subagents and Task Delegation](#subagents-and-task-delegation)
  - [Interactive User Questions](#interactive-user-questions)
- [Terminal Requirements](#terminal-requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Interactive Mode](#interactive-mode)
  - [Trust Folder System](#trust-folder-system)
  - [Programmatic Mode](#programmatic-mode)
- [Voice Mode](#voice-mode)
- [Slash Commands](#slash-commands)
  - [Built-in Slash Commands](#built-in-slash-commands)
  - [Custom Slash Commands via Skills](#custom-slash-commands-via-skills)
- [Skills System](#skills-system)
  - [Creating Skills](#creating-skills)
  - [Skill Discovery](#skill-discovery)
  - [Managing Skills](#managing-skills)
- [Configuration](#configuration)
  - [Configuration File Location](#configuration-file-location)
  - [API Key Configuration](#api-key-configuration)
  - [Models and Providers](#models-and-providers)
    - [Adding OpenAI-compatible providers (Kimi, GLM/ZAI, etc.)](#adding-openai-compatible-providers-kimi-glmzai-etc)
  - [Custom System Prompts](#custom-system-prompts)
  - [Custom Agent Configurations](#custom-agent-configurations)
  - [Tool Management](#tool-management)
  - [MCP Server Configuration](#mcp-server-configuration)
  - [Session Management](#session-management)
  - [Update Settings](#update-settings)
  - [Custom Vibe Home Directory](#custom-vibe-home-directory)
- [Workflows](#workflows)
  - [Workflow Scripts](#workflow-scripts)
  - [Bundled Workflows](#bundled-workflows)
  - [Effort Modes](#effort-modes)
- [Agent Teams](#agent-teams)
- [Editors/IDEs](#editorsides)
- [Resources](#resources)
- [Data collection & usage](#data-collection--usage)
- [License](#license)

## Features

- **Interactive Chat**: A conversational AI agent that understands your requests and breaks down complex tasks.
- **Powerful Toolset**: A suite of tools for file manipulation, code searching, version control, and command execution, right from the chat prompt.
  - Read, write, and patch files (`read`, `write_file`, `edit`).
  - Execute shell commands in a stateful terminal (`bash`).
  - Recursively search code with `grep` (with `ripgrep` support).
  - Manage a `todo` list to track the agent's work.
  - Ask interactive questions to gather user input (`ask_user_question`).
  - Delegate tasks to subagents for parallel work (`task`).
- **Project-Aware Context**: Vibe automatically scans your project's file structure and Git status to provide relevant context to the agent, improving its understanding of your codebase.
- **Advanced CLI Experience**: Built with modern libraries for a smooth and efficient workflow.
  - Autocompletion for slash commands (`/`) and file paths (`@`).
  - Image attachments via `@` mentions — `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` files are sent to vision-capable models (e.g. Mistral Medium 3.5) as native multimodal content.
  - Persistent command history.
  - Beautiful Themes.
- **Highly Configurable**: Customize models, providers, tool permissions, and UI preferences through a simple `config.toml` file.
- **Safety First**: Features tool execution approval.
- **Multiple Built-in Agents**: Choose from different agent profiles tailored for specific workflows.
- **Workflow Orchestration**: Write Python scripts that orchestrate parallel agents for codebase audits, migrations, and cross-checked research. Run bundled workflows like `/deep-research` or create your own.
- **Effort Modes**: Switch between `normal` (turn-by-turn) and `le chaton` (max thinking + automatic workflow planning) via `/effort`.
- **Agent Teams**: Coordinate multiple independent Vibe instances working together as teammates, communicating via file-backed shared state.

### Built-in Agents

Vibe comes with several built-in agent profiles, each designed for different use cases:

- **`default`**: Standard agent that requires approval for tool executions. Best for general use.
- **`plan`**: Read-only agent for exploration and planning. Auto-approves safe tools like `grep` and `read`.
- **`accept-edits`**: Auto-approves file edits only (`write_file`, `edit`). Useful for code refactoring.
- **`auto-approve`**: Auto-approves all tool executions. Use with caution.

Use the `--agent` flag to select a different agent:

```bash
vibe --agent plan
```

To change the default agent used when `--agent` is not passed, set
`default_agent` in your `config.toml`:

```toml
default_agent = "plan"
```

Valid values are `default`, `plan`, `accept-edits`, `auto-approve`,
`lean` (only when listed in `installed_agents`), or the name of any
custom agent file in `~/.vibe/agents/` or the project's `.vibe/agents/`
directory. Subagents such as `explore` are not accepted.

> Note: `default_agent` applies in both interactive and programmatic
> (`-p` / `--prompt`) sessions. Pass `--auto-approve` when a run should
> approve all tool calls without prompting.

### Subagents and Task Delegation

Vibe supports subagents for delegating tasks. Subagents run independently and can perform specialized work without user interaction, preventing the context from being overloaded.

The `task` tool allows the agent to delegate work to subagents:

```
> Can you explore the codebase structure while I work on something else?

🤖 I'll use the task tool to delegate this to the explore subagent.

> task(task="Analyze the project structure and architecture", agent="explore")
```

Create custom subagents by adding `agent_type = "subagent"` to your agent configuration. Vibe comes with a built-in subagent called `explore`, a read-only subagent for codebase exploration used internally for delegation.

### Interactive User Questions

The `ask_user_question` tool allows the agent to ask you clarifying questions during its work. This enables more interactive and collaborative workflows.

```
> Can you help me refactor this function?

🤖 I need to understand your requirements better before proceeding.

> ask_user_question(questions=[{
    "question": "What's the main goal of this refactoring?",
    "options": [
        {"label": "Performance", "description": "Make it run faster"},
        {"label": "Readability", "description": "Make it easier to understand"},
        {"label": "Maintainability", "description": "Make it easier to modify"}
    ]
}])
```

The agent can ask multiple questions at once, displayed as tabs. Each question supports 2-4 options plus an automatic "Other" option for free text responses.

## Terminal Requirements

Vibe's interactive interface requires a modern terminal emulator. Recommended terminal emulators include:

- **WezTerm** (cross-platform)
- **Alacritty** (cross-platform)
- **Ghostty** (Linux and macOS)
- **Kitty** (Linux and macOS)

Most modern terminals should work, but older or minimal terminal emulators may have display issues.

## Quick Start

1. Navigate to your project's root directory:

   ```bash
   cd /path/to/your/project
   ```

2. Run Vibe:

   ```bash
   vibe
   ```

3. If this is your first time running Vibe, it will:
   - Create a default configuration file at `~/.vibe/config.toml`
   - Prompt you to enter your API key if it's not already configured
   - Save your API key to `~/.vibe/.env` for future use

   Alternatively, you can configure your API key separately using `vibe --setup`.

4. Start interacting with the agent!

   ```
   > Can you find all instances of the word "TODO" in the project?

   🤖 The user wants to find all instances of "TODO". The `grep` tool is perfect for this. I will use it to search the current directory.

   > grep(pattern="TODO", path=".")

   ... (grep tool output) ...

   🤖 I found the following "TODO" comments in your project.
   ```

## Usage

### Interactive Mode

Simply run `vibe` to enter the interactive chat loop.

- **Multi-line Input**: Press `Ctrl+J` or `Shift+Enter` for select terminals to insert a newline.
- **File Paths**: Reference files in your prompt using the `@` symbol for smart autocompletion (e.g., `> Read the file @src/agent.py`).
- **Shell Commands**: Prefix any command with `!` to execute it directly in your shell, bypassing the agent (e.g., `> !ls -l`).
- **External Editor**: Press `Ctrl+G` to edit your current input in an external editor.
- **Tool Output Toggle**: Press `Ctrl+O` to toggle the tool output view.
- **Todo View Toggle**: Press `Ctrl+T` to toggle the todo list view.
- **Debug Console**: Press `Ctrl+\` to toggle the debug console.
- **Agent Selection**: Press `Shift+Tab` to cycle through agents (default, plan, ...).
- **Exit**: Type `/exit`, `exit`, `quit`, `:q`, or `:quit` in the input box, or press `Ctrl+C` / `Ctrl+D` twice within ~1 second.

You can start Vibe with a prompt using the following command:

```bash
vibe "Refactor the main function in cli/main.py to be more modular."
```

### Trust Folder System

Vibe includes a trust folder system to ensure you only run the agent in directories you trust. When you first run Vibe in a new directory which contains a `.vibe` subfolder, it may ask you to confirm whether you trust the folder.

Trusted folders are remembered for future sessions. You can manage trusted folders through its configuration file `~/.vibe/trusted_folders.toml`.

This safety feature helps prevent accidental execution in sensitive directories.

### Programmatic Mode

You can run Vibe non-interactively by piping input or using the `--prompt` flag. This is useful for scripting.

```bash
vibe --prompt "Refactor the main function in cli/main.py to be more modular."
```

By default, it uses your configured `default_agent` (`default` unless changed).
To approve all tool calls without prompting, pass `--auto-approve` (also
available for interactive sessions):

```bash
vibe --prompt "Refactor the main function in cli/main.py to be more modular." --auto-approve
```

#### Programmatic Mode Options

When using `--prompt`, you can specify additional options:

- **`--max-turns N`**: Limit the maximum number of assistant turns. The session will stop after N turns.
- **`--max-price DOLLARS`**: Set a maximum cost limit in dollars. The session will be interrupted if the cost exceeds this limit.
- **`--max-tokens N`**: Set a maximum cumulative LLM token budget for the session, counting both prompt and completion tokens. The session will be interrupted if usage exceeds this limit.
- **`--agent NAME`**: Select the agent profile for this run.
- **`--auto-approve`**: Shortcut for `--agent auto-approve`. Approves all tool calls without prompting, including in interactive sessions.
- **`--enabled-tools TOOL`**: Enable specific tools. In programmatic mode, this disables all other tools. Can be specified multiple times. Supports exact names, glob patterns (e.g., `bash*`), or regex with `re:` prefix (e.g., `re:^serena_.*$`).
- **`--output FORMAT`**: Set the output format. Options:
  - `text` (default): Human-readable text output
  - `json`: All messages as JSON at the end
  - `streaming`: Newline-delimited JSON per message

Example:

```bash
vibe --prompt "Analyze the codebase" --max-turns 5 --max-price 1.0 --max-tokens 50000 --output json
```

## Voice Mode

> [!WARNING]
> Voice mode is experimental and may change in future releases.

Voice mode allows you to dictate input using your microphone instead of typing.

### Activating Voice Mode

Toggle voice mode on or off with the `/voice` slash command:

```
> /voice
```

### Recording Shortcuts

| Shortcut | Action           |
| -------- | ---------------- |
| `Ctrl+R` | Start recording  |
| Any key  | Stop recording   |
| `Escape` | Cancel recording |
| `Ctrl+C` | Cancel recording |

## Slash Commands

Use slash commands for meta-actions and configuration changes during a session.

### Built-in Slash Commands

Vibe provides several built-in slash commands. Use slash commands by typing them in the input box:

```
> /help
```

Key commands include `/model`, `/thinking`, `/effort`, `/config`, `/workflows`,
`/team`, `/deep-research`, `/loop`, `/rewind`, `/resume`, and `/exit`. Run
`/help` in-session for the full list.

### Custom Slash Commands via Skills

You can define your own slash commands through the skills system. Skills are reusable components that extend Vibe's functionality.

To create a custom slash command:

1. Create a skill directory with a `SKILL.md` file
2. Set `user-invocable = true` in the skill metadata
3. Define the command logic in your skill

Example skill metadata:

```markdown
---
name: my-skill
description: My custom skill with slash commands
user-invocable: true
---
```

Custom slash commands appear in the autocompletion menu alongside built-in commands.

## Skills System

Vibe's skills system allows you to extend functionality through reusable components. Skills can add new tools, slash commands, and specialized behaviors.

Vibe follows the [Agent Skills specification](https://agentskills.io/specification) for skill format and structure.

### Creating Skills

Skills are defined in directories with a `SKILL.md` file containing metadata in YAML frontmatter. For example, `~/.vibe/skills/code-review/SKILL.md`:

```markdown
---
name: code-review
description: Perform automated code reviews
license: MIT
compatibility: Python 3.12+
user-invocable: true
allowed-tools:
  - read
  - grep
  - ask_user_question
---

# Code Review Skill

This skill helps analyze code quality and suggest improvements.
```

### Skill Discovery

Vibe discovers skills from multiple locations:

1. **Custom paths**: Configured in `config.toml` via `skill_paths`
2. **Standard Agent Skills path** (project root, trusted folders only): `.agents/skills/` — [Agent Skills](https://agentskills.io) standard
3. **Local project skills** (project root, trusted folders only): `.vibe/skills/` in your project
4. **Global skills directories**: `~/.vibe/skills/` and `~/.agents/skills/`

```toml
skill_paths = ["/path/to/custom/skills"]
```

### Managing Skills

Enable or disable skills using patterns in your configuration:

```toml
# Enable specific skills
enabled_skills = ["code-review", "test-*"]

# Disable specific skills
disabled_skills = ["experimental-*"]
```

Skills support the same pattern matching as tools (exact names, glob patterns, and regex).

## Configuration

### Configuration File Location

Vibe is configured via a `config.toml` file. It looks for this file first in `./.vibe/config.toml` and then falls back to `~/.vibe/config.toml`.

### API Key Configuration

To use Vibe, you'll need a Mistral API key. You can obtain one by signing up at [https://console.mistral.ai](https://console.mistral.ai).

You can configure your API key using `vibe --setup`, or through one of the methods below.

Vibe supports multiple ways to configure your API keys:

1. **Interactive Setup (Recommended for first-time users)**: When you run Vibe for the first time or if your API key is missing, Vibe will prompt you to enter it. The key will be securely saved to `~/.vibe/.env` for future sessions.

2. **Environment Variables**: Set your API key as an environment variable:

   ```bash
   export MISTRAL_API_KEY="your_mistral_api_key"
   ```

3. **`.env` File**: Create a `.env` file in `~/.vibe/` and add your API keys:

   ```bash
   MISTRAL_API_KEY=your_mistral_api_key
   ```

   Vibe automatically loads API keys from `~/.vibe/.env` on startup. Environment variables take precedence over the `.env` file if both are set.

**Note**: The `.env` file is specifically for API keys and other provider credentials. General Vibe configuration should be done in `config.toml`.

### Models and Providers

Vibe talks to models through **providers**. Each provider points at an OpenAI-compatible (or Anthropic-compatible) endpoint, and each **model** references a provider by name. Configuration lives in `config.toml`; API keys live in `~/.vibe/.env`.

```toml
# A provider = an HTTP endpoint + auth.
#   - backend            : "generic" (default; OpenAI/Anthropic-compatible) or "mistral"
#   - api_style          : "openai" (default), "anthropic", "reasoning", "openai-responses", "vertex-anthropic"
#   - api_key_env_var    : name of the env var holding the key (loaded from ~/.vibe/.env)
#   - reasoning_field_name : field the API streams reasoning in (default "reasoning_content")
[[providers]]
name = "kimi"
api_base = "https://api.kimi.com/coding/v1"
api_key_env_var = "KIMI_API_KEY"
backend = "generic"
api_style = "openai"
reasoning_field_name = "reasoning_content"

# A model references a provider by name.
#   - name      : model id sent to the API (the "model" field)
#   - provider  : must match a [[providers]] name above
#   - alias     : short id used by `active_model`, `/model`, and `--model`
#   - thinking  : "off" | "low" | "medium" | "high" | "max"
#   - supports_images : enable image input
#   - auto_compact_threshold : token count that triggers auto-compaction (the effective per-model context budget; set ~80% of the model's real context window)
[[models]]
name = "kimi-k2.7-code"
provider = "kimi"
alias = "kimi"
thinking = "high"
input_price = 0.95
output_price = 4.0

# Select the default model by alias:
active_model = "kimi"
```

Custom providers/models are **merged** with the built-in Mistral defaults, so you keep Mistral available while adding others. Switch models at runtime with the `/model` slash command.

#### Adding OpenAI-compatible providers (Kimi K2.7, GLM-5.2/ZAI)

Most third-party coding models expose an OpenAI-compatible `/chat/completions` endpoint and stream reasoning in a `reasoning_content` field — the exact shape Vibe's generic backend expects, so no code changes are required. Use `api_style = "openai"` (the default): Vibe captures the streamed `reasoning_content` and displays it, and both Kimi and GLM default to thinking **enabled**, so reasoning works without Vibe needing to send any effort parameter.

> Do **not** use `api_style = "reasoning"` for these: Vibe's reasoning adapter parses content blocks and would drop the streamed `reasoning_content` field, hiding the model's thinking.

**Kimi K2.7 Code (Moonshot)** — `name` and prices from the Kimi platform; context 256k; supports text, image, and video input:

```toml
[[providers]]
name = "kimi"
api_base = "https://api.kimi.com/coding/v1"   # Kimi Code platform (coding-plan); Moonshot keys use https://api.moonshot.cn/v1
api_key_env_var = "KIMI_API_KEY"

# Standard model
[[models]]
name = "kimi-k2.7-code"          # ~180 tok/s; use "kimi-k2.7-code-highspeed" for the faster variant
provider = "kimi"
alias = "kimi"
thinking = "high"
input_price = 0.95               # cache miss; cache hit is $0.19/1M
output_price = 4.0
supports_images = true
auto_compact_threshold = 200000   # 256k context; compaction trigger (~76% of window)

[[models]]
name = "kimi-k2.7-code-highspeed"
provider = "kimi"
alias = "kimi-fast"
thinking = "high"
input_price = 1.90               # cache miss; cache hit is $0.38/1M
output_price = 8.0
supports_images = true
```

**GLM-5.2 (ZAI / Zhipu / Z.ai)** — 1M context, text input only; the ZAI coding plan is a flat subscription so per-token prices are set to `0.0` for usage tracking:

```toml
[[providers]]
name = "zai"
api_base = "https://api.z.ai/api/coding/paas/v4"   # Coding Plan endpoint. Pay-as-you-go keys use https://api.z.ai/api/paas/v4; China: https://open.bigmodel.cn/api/paas/v4
api_key_env_var = "ZAI_API_KEY"

[[models]]
name = "glm-5.2"
provider = "zai"
alias = "glm"
thinking = "high"
input_price = 0.0                # coding plan = flat subscription
output_price = 0.0
auto_compact_threshold = 880000   # 1M context, 128k max output; compaction trigger (~84% of window)
```

Then put the keys in `~/.vibe/.env`:

```sh
KIMI_API_KEY=sk-...
ZAI_API_KEY=...
```

Notes:

- **Reasoning display**: with `api_style = "openai"` (recommended), reasoning shows automatically as the model streams `reasoning_content`. Set `reasoning_field_name` only if a provider uses a different field name.
- **Thinking effort**: Vibe's `openai` style does not send `reasoning_effort`, so the in-app thinking slider won't change provider effort for these — each model uses its own default thinking level. GLM-5.2 additionally accepts a `thinking: { type }` parameter, which Vibe does not currently send; default thinking stays enabled.
- **Endpoint base**: `api_base` includes the version segment (`/v1` or `/api/paas/v4`) but **not** `/chat/completions`; Vibe appends that automatically.
- **Multi-turn reasoning**: if a provider rejects an assistant turn on long conversations because of how reasoning is replayed, set `thinking = "off"` for that model or report it — a dedicated adapter may be needed.
- **Wrong endpoint looks like "rate limit"**: a ZAI Coding Plan key sent to the pay-as-you-go `/api/paas/v4` endpoint returns HTTP 429 with `code 1113` "Insufficient balance or no resource package", which Vibe surfaces as a rate-limit error. If GLM reports rate limits you can't explain, check `api_base` matches your plan (`/api/coding/paas/v4` for the Coding Plan).
- **Kimi User-Agent gate**: the Kimi coding endpoint only serves approved clients, so its provider needs `extra_headers = { User-Agent = "KimiCLI/1.47.0" }`. A missing/odd User-Agent returns `403 access_terminated_error`.

### Safety Judge (experimental)

By default, any tool call that isn't auto-allowed by your allowlist/permission rules prompts you for approval. The **safety judge** lets a separate LLM auto-approve calls it deems safe, so you're only prompted for the genuinely risky ones. It is **off by default**.

```toml
[safety_judge]
enabled = true
model = "devstral-small"   # alias of a model from [[models]]; ideally a different model than your active one
max_tokens = 512
timeout = 15.0             # seconds; on timeout the judge fails closed (you are prompted)
# temperature is omitted -> uses the judge model's own temperature (some providers, e.g. Kimi, require a fixed value)
```

How it fits the existing controls:

- It only fills the **approval prompt** gap. Calls your denylist/guardrails mark as denied (`NEVER`) are still hard-blocked — the judge never sees them.
- `--auto-approve` is unchanged: it still bypasses everything (including the judge).
- It **fails closed**. No usable judge model, an API error, a timeout, a refusal, or an unparseable answer all fall back to the normal human prompt.
- Every judge auto-approval is logged.

> **Security note.** An LLM judge is a probabilistic gate, not a guarantee. The tool call it evaluates is authored by the (untrusted) main model, so a compromised or jailbroken main model could in principle craft a call designed to fool the judge. Keep your denylist authoritative, prefer a judge model from a different provider than your active model, and treat this as convenience, not a sandbox.

### TLS and Corporate Certificate Authorities

By default, Vibe uses the bundled `certifi` certificate roots for outbound HTTPS requests. If your organization installs private certificate authorities in the operating system trust store, you can opt in to the system trust store in `config.toml`:

```toml
enable_system_trust_store = true
```

`SSL_CERT_FILE` and `SSL_CERT_DIR` are still supported and are loaded as additional trust anchors.

### Custom System Prompts

You can create `AGENTS.md` files to add custom instructions. You can also replace the entire system prompt.

Place `AGENTS.md` files in:
- `~/.vibe/AGENTS.md` — user-level instructions for all projects
- Project directories — project-specific instructions, loaded from cwd up to the trust root

Priority: closer directories override more distant ones. Instructions in `AGENTS.md` override the default system prompt. Files are only loaded for trusted folders.

Custom system prompts entirely replace the default one (`prompts/cli.md`). Create a markdown file in the `~/.vibe/prompts/` directory with your custom prompt content.

To use a custom system prompt, set the `system_prompt_id` in your configuration to match the filename (without the `.md` extension):

```toml
# Use a custom system prompt
system_prompt_id = "my_custom_prompt"
```

This will load the prompt from `~/.vibe/prompts/my_custom_prompt.md`.

Project-local prompts in `.vibe/prompts/` are also supported and override user-level prompts with the same name. This applies to all custom prompts (system and compaction).

### Custom Compaction Prompts

Compaction uses the built-in prompt at `prompts/compact.md` by default. You can replace it with a custom prompt from `~/.vibe/prompts/` (or `.vibe/prompts/`) using the same resolution rules as system prompts.

To use a custom compaction prompt, set `compaction_prompt_id` in your configuration to match the filename (without the `.md` extension):

```toml
# Use a custom compaction prompt
compaction_prompt_id = "my_compaction_prompt"
```

Any extra instructions passed to `/compact ...` are appended after the configured compaction prompt.

### Custom Agent Configurations

You can create custom agent configurations for specific use cases (e.g., red-teaming, specialized tasks) by adding agent-specific TOML files in the `~/.vibe/agents/` directory.

To use a custom agent, run Vibe with the `--agent` flag:

```bash
vibe --agent my_custom_agent
```

Vibe will look for a file named `my_custom_agent.toml` in the agents directory and apply its configuration.

Example custom agent configuration (`~/.vibe/agents/redteam.toml`):

```toml
# Custom agent configuration for red-teaming
active_model = "mistral-medium-3.5"
system_prompt_id = "redteam"

# Disable some tools for this agent
disabled_tools = ["edit", "write_file"]

# Override tool permissions for this agent
[tools.bash]
permission = "always"

[tools.read]
permission = "always"
```

Note: This implies that you have set up a redteam prompt named `~/.vibe/prompts/redteam.md`.

### Tool Management

#### Enable/Disable Tools with Patterns

You can control which tools are active using `enabled_tools` and `disabled_tools`.
These fields support exact names, glob patterns, and regular expressions.

Examples:

```toml
# Only enable tools that start with "serena_" (glob)
enabled_tools = ["serena_*"]

# Regex (prefix with re:) — matches full tool name (case-insensitive)
enabled_tools = ["re:^serena_.*$"]

# Disable a group with glob; everything else stays enabled
disabled_tools = ["mcp_*", "grep"]
```

Notes:

- MCP tool names use underscores, e.g., `serena_list` not `serena.list`.
- Regex patterns are matched against the full tool name using fullmatch.

### MCP Server Configuration

You can configure MCP (Model Context Protocol) servers to extend Vibe's capabilities. Add MCP server configurations under the `mcp_servers` section:

```toml
# Example MCP server configurations
[[mcp_servers]]
name = "my_http_server"
transport = "http"
url = "http://localhost:8000"
headers = { "Authorization" = "Bearer my_token" }
api_key_env = "MY_API_KEY_ENV_VAR"
api_key_header = "Authorization"
api_key_format = "Bearer {token}"

[[mcp_servers]]
name = "my_streamable_server"
transport = "streamable-http"
url = "http://localhost:8001"
headers = { "X-API-Key" = "my_api_key" }

[[mcp_servers]]
name = "fetch_server"
transport = "stdio"
command = "uvx"
args = ["mcp-server-fetch"]
env = { "DEBUG" = "1", "LOG_LEVEL" = "info" }
```

Supported transports:

- `http`: Standard HTTP transport
- `streamable-http`: HTTP transport with streaming support
- `stdio`: Standard input/output transport (for local processes)

Key fields:

- `name`: A short alias for the server (used in tool names)
- `transport`: The transport type
- `url`: Base URL for HTTP transports
- `headers`: Additional HTTP headers
- `api_key_env`: Environment variable containing the API key
- `command`: Command to run for stdio transport
- `args`: Additional arguments for stdio transport
- `startup_timeout_sec`: Timeout in seconds for the server to start and initialize (default 10s)
- `tool_timeout_sec`: Timeout in seconds for tool execution (default 60s)
- `env`: Environment variables to set for the MCP server of transport type stdio

MCP tools are named using the pattern `{server_name}_{tool_name}` and can be configured with permissions like built-in tools:

```toml
# Configure permissions for specific MCP tools
[tools.fetch_server_get]
permission = "always"

[tools.my_http_server_query]
permission = "ask"
```

MCP server configurations support additional features:

- **Environment variables**: Set environment variables for MCP servers
- **Custom timeouts**: Configure startup and tool execution timeouts

Example with environment variables and timeouts:

```toml
[[mcp_servers]]
name = "my_server"
transport = "http"
url = "http://localhost:8000"
env = { "DEBUG" = "1", "LOG_LEVEL" = "info" }
startup_timeout_sec = 15
tool_timeout_sec = 120
```

### Hooks (Experimental)

Hooks wire arbitrary shell commands into Vibe's lifecycle to gate, audit, or rewrite agent behavior. **Experimental**, gated behind:

```toml
# config.toml
enable_experimental_hooks = true   # or env VIBE_ENABLE_EXPERIMENTAL_HOOKS=true
```

Declared in `<project>/.vibe/hooks.toml` (project, loaded first; trusted only) and `~/.vibe/hooks.toml` (user-global, loaded second; duplicates by `name` lose to the project entry):

```toml
[[hooks]]
name = "deny-rm-rf"
type = "before_tool"
match = "bash"                       # tool-name matcher (fnmatch glob + `re:` regex escape, case-insensitive)
command = "uv run python /path/to/guard-bash"
timeout = 60.0                       # seconds; default 60 for all hooks
strict = false                       # tool hooks only: turn failures into denials (before) / text-clears (after)
description = "Reject dangerous shell commands."
```

Subagents inherit the parent's hook config so policies apply transitively.

#### Common ground

Every hook is spawned with a JSON invocation on **stdin** (UTF-8) containing the session context: `session_id`, `parent_session_id`, `transcript_path`, `cwd`, plus `hook_event_name` discriminating the hook type. Tool hooks add tool-specific fields (below).

Every hook signals back via its **exit code** and **stdout**. The contract on stdout is strict: either empty (do nothing), or a JSON object matching the schema below. Use **stderr** for diagnostics / debug logs.

- **Exit `0`, empty stdout** — passthrough.
- **Exit `0`, valid JSON object on stdout** — structured response. Universal top-level fields:
  - `system_message` (string, optional) — shown to the user in the UI.
  - `decision` (`"allow"` | `"deny"`, optional, default `"allow"`) — the effect of `"deny"` depends on the hook type.
  - `reason` (string, optional) — accompanies `decision: "deny"`.
  - Event-specific payload under `hook_specific_output`.
- **Exit `0`, non-empty but non-conforming stdout** (free-form text, broken JSON, JSON scalar/array, schema mismatch) — treated as a hook failure with the parse error as the message. Warning by default; escalated to deny / clear under `strict = true` on a tool hook.
- **Any non-zero exit / timeout / spawn failure** — same failure path. Diagnostic taken from stderr (falling back to stdout, then the exit code).

Unknown JSON fields are tolerated at every level (forward-compatible). Fields that aren't meaningful for the current hook type are silently ignored.

#### `post_agent_turn`

Fires after every assistant turn that ends without pending tool calls.

- **Receives** (in addition to the session context): no extra fields.
- **Can return**:
  - `decision: "deny"` + `reason` — `reason` is injected as a new user message asking for a retry. Capped at **3 retries per hook per user turn**; further denies become terminal warnings.
  - `system_message` — UI-only.

#### `before_tool`

Fires per tool call, **before** the user permission prompt. First deny short-circuits remaining `before_tool` hooks for that call.

- **Receives** (in addition to the session context): `tool_name`, `tool_call_id`, `tool_input` (the model's raw arguments).
- **Can return**:
  - `decision: "deny"` + `reason` — denies the tool call; `reason` becomes the tool error the LLM sees.
  - `hook_specific_output.tool_input` (object) — **full replacement** of the model's arguments. Re-validated against the tool's schema (validation failure → synthesized denial). Rewrites compose left-to-right across hooks. The rewritten arguments are also what the permission prompt displays, what the tool runs with, and what subsequent LLM turns see on the assistant message.
  - `system_message` — UI-only.

#### `after_tool`

Fires per tool call **if and only if the tool body actually ran**. `tool_status` is `success`, `failure`, or `cancelled` (cancellation during the tool body — cancellation is shielded so audit hooks still run). Does not fire when the tool never executed: `before_tool` denial, user denial at the approval prompt, permission `NEVER`, or cancellation before the body started.

- **Receives** (in addition to the session context): `tool_name`, `tool_call_id`, `tool_input` (post-rewrite), `tool_status`, `tool_output` (structured result dict; null on failure), `tool_output_text` (the running text the LLM will see, mutable by prior hooks), `tool_error`, `duration_ms`.
- **Can return**:
  - `decision: "deny"` + `reason` — replaces `tool_output_text` with `reason`. Pipeline continues; subsequent hooks see the replacement.
  - `hook_specific_output.additional_context` (string) — **appended** (with a `\n` separator) to `tool_output_text`. Composes with a same-hook deny: deny replaces first, then `additional_context` is appended to the replacement.
  - `system_message` — UI-only.

### Session Management

#### Session Continuation and Resumption

Vibe supports continuing from previous sessions:

- **`--continue`** or **`-c`**: Continue from the most recent saved session
- **`--resume`**: Open an interactive session picker
- **`--resume SESSION_ID`**: Resume a specific session by ID (supports partial matching)
- **`/resume`** or **`/continue`**: Open the session picker from inside Vibe; press `D` twice to delete a local saved session. The active session cannot be deleted from this picker.

```bash
# Continue from last session
vibe --continue

# Open session picker
vibe --resume

# Resume specific session
vibe --resume abc123
```

Session logging must be enabled in your configuration for these features to work.

#### Working Directory Control

Use the `--workdir` option to specify a working directory:

```bash
vibe --workdir /path/to/project
```

This is useful when you want to run Vibe from a different location than your current directory.

Use `--add-dir` (repeatable) to make additional directories available to the agent for the duration of the session:

```bash
vibe --add-dir /path/to/other-project --add-dir /path/to/library
```

Each path is implicitly trusted (no trust prompt) and contributes its `AGENTS.md` and `.vibe/` configuration (tools, skills, agents, prompts, hooks) to the session. File-tool permissions treat each `--add-dir` path the same way as your primary working directory — reads and writes inside them don't require the "outside workdir" prompt. Nested paths collapse: passing `/repo` and `/repo/sub` is equivalent to passing just `/repo`.

### Update Settings

Vibe checks PyPI at most once per day during a session. When a newer version is found, the next launch shows an update prompt before opening the chat, offering to either update immediately (via `uv tool upgrade mistral-vibe` or `brew upgrade mistral-vibe`) or continue with the current version.

To disable the daily check entirely, add this to your `config.toml`:

```toml
enable_update_checks = false
```

### Notification Settings

Vibe can notify you when the agent needs your attention (awaiting approval, asking a question, or task complete). This is useful when you switch to another window while the agent works.

To disable notifications:

```toml
enable_notifications = false
```

### Custom Vibe Home Directory

By default, Vibe stores its configuration in `~/.vibe/`. You can override this by setting the `VIBE_HOME` environment variable:

```bash
export VIBE_HOME="/path/to/custom/vibe/home"
```

This affects where Vibe looks for:

- `config.toml` - Main configuration
- `.env` - API keys
- `agents/` - Custom agent configurations
- `prompts/` - Custom system and compaction prompts
- `tools/` - Custom tools
- `workflows/` - Custom workflow scripts
- `logs/` - Session logs

## Workflows

Workflows are Python scripts that orchestrate parallel agents. They run in the
background as asyncio tasks, so the session stays responsive while agents work.

### Workflow Scripts

A workflow script is a `.py` file with an `async def main()` function. Optional
YAML frontmatter (`name:`, `description:`) precedes the Python source. The
runtime injects these functions:

- `agent(prompt, *, agent="explore", label=None, phase=None, schema=None)` — spawn a subagent
- `parallel(*thunks)` — run thunks concurrently, results in order (a thunk that raises yields `None`)
- `pipeline(items, *stages)` — run each item through all stages independently, no barrier between stages; each stage receives `(prev, item, index)`. One stage acts as a concurrent map.
- `phase(name)` — declare a phase for progress tracking
- `log(msg)` — log a progress message
- `budget` — token budget object with `.total` and `.remaining()`
- `workflow(name, args=None)` — run another discovered workflow inline as a sub-step, returning its result (shares this run's budget/agent count; one level deep)
- `args` — structured input from the invocation command

Scripts are validated via AST before execution (unsafe imports, dangerous calls,
dunder access blocked). Discovered from `workflow_paths` config, `.vibe/workflows/`,
`~/.vibe/workflows/`, and bundled workflows. Registered as `/<name>` slash commands.

### Bundled Workflows

- `/deep-research <question>` — fans out web searches across 5 angles, extracts
  claims with structured output, verifies each claim, synthesizes a cited report.

### Managing Workflow Runs

- `/workflows` — open a progress view showing all runs with status, agents, tokens, elapsed
- `/workflows list` — list runs as text
- `/workflows stop <id|all>` — stop one or all runs
- `/workflows snapshot <id>` — show cached results for a run

Completed agent results are cached for resumability. Snapshots persist to session
metadata for cross-session recovery.

### Effort Modes

- **normal** (default): work turn-by-turn.
- **le-chaton**: max thinking + automatic workflow planning. The system prompt
  instructs the model to write workflow scripts for substantive tasks.

Select via `/effort` or set `effort_mode = "le-chaton"` in config.toml. Typing
"le chaton" in a prompt triggers it for that turn. Disable all workflow features
with `disable_workflows = true`.

## Agent Teams

Agent teams coordinate multiple independent Vibe instances. Unlike subagents
(in-memory, same session) or workflows (asyncio tasks, same event loop),
teammates are **separate OS processes** — each is a full `vibe -p` invocation.

### Team Commands

- `/team spawn <name> <prompt>` — spawn a teammate as a separate process
- `/team list` — show teammates with name, status, PID
- `/team stop <name|all>` — stop one or all teammates
- `/team cleanup` — remove team directory

### Shared State

Teammates coordinate via file-backed shared state with file locking:
- **TaskStore**: shared task list with dependencies and claim/complete operations
- **Mailbox**: per-recipient inbox for inter-agent messaging
- **TeamConfig**: team metadata (members, status, PIDs)

Team directories live under `~/.vibe/teams/<name>/` and are cleaned up on exit.

## Editors/IDEs

Mistral Vibe can be used in text editors and IDEs that support [Agent Client Protocol](https://agentclientprotocol.com/overview/clients). See the [ACP Setup documentation](docs/acp-setup.md) for setup instructions for various editors and IDEs.

## Resources

- [CHANGELOG](CHANGELOG.md) - See what's new in each version
- [CONTRIBUTING](CONTRIBUTING.md) - Guidelines for feature requests, feedback and bug reports

## Data collection & usage

Use of Vibe is subject to our [Privacy Policy](https://legal.mistral.ai/terms/privacy-policy) and may include the collection and processing of data related to your use of the service, such as usage data, to operate, maintain, and improve Vibe. You can disable telemetry in your `config.toml` by setting `enable_telemetry = false`.


## License

Copyright 2025 Mistral AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the [LICENSE](LICENSE) file for the full license text.
