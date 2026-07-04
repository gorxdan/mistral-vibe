# Mistral Vibe — OpenWiki Quickstart

Mistral Vibe is an open-source CLI coding assistant powered by Mistral's models (and other LLM providers). It provides a conversational interface to your codebase — you use natural language to explore, modify, and interact with your projects through a powerful set of tools.

This is a **fork of `mistralai/mistral-vibe`** (remote `upstream`), synced continuously. The fork survives on cheap git merges; the guiding principle is *add, don't restructure* — new features go in new sibling files with thin hooks into upstream-owned files, never by splitting or renaming upstream code.

## Key Facts

- **Language**: Python 3.12+, managed with `uv` (never bare `python`/`pip`)
- **Package**: `mistral-vibe` on PyPI (upstream); this fork is installed from git
- **License**: Apache 2.0
- **Entry points**: `vibe` (CLI/TUI), `vibe-acp` (ACP server for editor/IDE integration)
- **Config**: `~/.vibe/config.toml` (TOML), API keys in `~/.vibe/.env`
- **Override home**: `VIBE_HOME` environment variable

## Three Front-Ends, One Engine

All three front-ends share the same core engine in `vibe/core/`:

1. **Interactive TUI** — Textual-based rich terminal UI. The primary user experience. Entry: `vibe` → `vibe/cli/entrypoint.py` → `vibe/cli/cli.py` → `VibeApp` (`vibe/cli/textual_ui/app.py`).
2. **Programmatic mode** — Non-interactive, for scripting. `vibe --prompt "..."` or `vibe -p "..."`. Uses `vibe/core/programmatic.py`.
3. **ACP server** — Agent Client Protocol for editor/IDE integration. `vibe-acp` → `vibe/acp/entrypoint.py` → `vibe/acp/acp_agent_loop.py`.

The shared heart is **`AgentLoop`** (`vibe/core/agent_loop.py`) — a multi-mixin class that orchestrates LLM calls, tool execution, memory, safety, and failover.

## Core Concepts

| Concept | What it does | Source |
|---|---|---|
| **Agent Loop** | Orchestrates the conversation: LLM call → parse tool calls → execute tools → feed results back | `vibe/core/agent_loop.py` |
| **Tools** | 24+ builtins (read, write, edit, bash, grep, glob, lsp, task, etc.) + MCP + connectors | `vibe/core/tools/` |
| **Workflows** | Python scripts that orchestrate parallel agents for audits, migrations, research | `vibe/core/workflows/` |
| **Teams** | Multiple independent `vibe -p` subprocesses coordinating via file-backed shared state | `vibe/core/teams/` |
| **Memory** | Durable `*.md` notes under `~/.vibe/memory/`, LLM-selected per turn | `vibe/core/memory/` |
| **Skills** | Markdown instruction files (with YAML frontmatter) injected into agent context | `vibe/core/skills/` |
| **Config** | Layered TOML config (user, project, harness) with Pydantic models | `vibe/core/config/` |
| **LLM Backends** | Mistral SDK, generic OpenAI/Anthropic-compatible, with model failover | `vibe/core/llm/` |

## Getting Started (Development)

```bash
# Clone and install
git clone <repo-url> && cd mistral-vibe
uv sync --all-extras

# Run the CLI
uv run vibe

# Run the ACP server
uv run vibe-acp

# Run tests
uv run pytest

# Type check
uv run pyright

# Lint
uv run ruff check --fix . && uv run ruff format .
```

## Repository Layout

```
vibe/
├── core/           # Engine: agent loop, tools, LLM, config, workflows, teams, memory
├── cli/            # Textual TUI + CLI bootstrap
├── acp/            # Agent Client Protocol server (editor/IDE)
└── setup/          # First-run wizards
tests/              # pytest + pytest-asyncio, Fake* stubs, iron-laws tests
docs/               # ACP setup, proxy, SearXNG, design specs
scripts/            # Release, upstream-divergence check, install
.github/workflows/  # CI, upstream-sync, release, snapshot tests
```

## Documentation Sections

- [Architecture Overview](architecture/overview.md) — AgentLoop composition, data flow, middleware, system prompt assembly
- [Tool System](tools/overview.md) — BaseTool, ToolManager, builtins, permissions, safety judge, MCP, LSP
- [Workflows & Teams](workflows-teams/overview.md) — Workflow scripts, bundled workflows, agent teams, effort modes
- [Configuration & LLM Backends](config-models/overview.md) — VibeConfig, layered config, providers, models, failover
- [Operations: Fork, Testing & CI](operations/overview.md) — Fork strategy, iron laws, testing, CI, release

## Important Conventions

Read [AGENTS.md](/AGENTS.md) for the full conventions guide. Key highlights:

- **Always `uv run`** — never bare `python`/`pip`; git through `uv run` for pre-commit hooks
- **Read before edit** — runtime-enforced; the `files_read` dict in `InvokeContext` tracks this
- **File I/O** — use `read_safe`/`write_safe`/`atomic_replace`/`write_durable`, never raw `Path.read_text()`
- **No `# type: ignore`**, no `# noqa`, no `# pyright: ignore` — fix at source (enforced by iron-laws tests)
- **Fork rule** — new features go in new sibling files, never restructure upstream files
- **Logging** — `logger.error("msg %s", val)` not f-strings; `raise ... from e`
