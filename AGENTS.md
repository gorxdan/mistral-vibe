# AGENTS.md

Conventions for **Mistral Vibe** — Python 3.12+ CLI coding assistant managed with `uv`.

Layout: `vibe/core` (engine: agent loop, tools, LLM backends, config, workflows, teams) | `vibe/cli` (Textual TUI) | `vibe/acp` (Agent Client Protocol) | `vibe/setup` (first-run wizards) | `tests/` (autouse fixtures in `conftest.py`, doubles in `tests/stubs/` named `Fake*`)

## Rules

Retrieval over recall | Read before edit (runtime-enforced) | Always `uv run` (never bare `python`/`pip`; git through `uv run` for pre-commit) | Strict pyright (no `# type: ignore`, no `# noqa`, no relative imports; fix at source) | Modern Python (built-in generics + `|` unions, `match`/`case`, early returns, `pathlib.Path`/`anyio.Path`, f-strings, never `Optional`/`Union`/`Dict`/`List`) | Pydantic (`model_validate`/validators, `ConfigDict(extra=...)` always set, no `from_sdk`) | Tests (`pytest`+`pytest-asyncio`+`respx`, no docstrings, autouse fixtures) | Lint (`ruff check --fix . && ruff format .` after changes, `pyright` gates CI) | File I/O (`read_safe`/`read_safe_async` over raw `Path.read_text()`) | Logging (`logger.error("msg %s", val)` not f-strings, `raise ... from e`)

## Commands

`uv run vibe` | `uv run vibe-acp` — entry points
`uv run pytest` — full suite (parallel via xdist)
`uv run pyright` — strict type check
`uv run ruff check --fix . && uv run ruff format .` — after every change
`uv run pre-commit run --all-files` — full lint pass

## Conventions

`__init__.py` exposes `__all__` | private modules prefixed `_` | models in `models.py` | config in `_settings.py`/`_config.py` | abstract interfaces use `_port.py` suffix | tests mirror source layout
Enums: `StrEnum`/`IntEnum` with `auto()` UPPERCASE; mix-in type before `Enum`; methods/`@property` over lookup tables
Walrus `:=` only when it shortens | never-nester (early returns) | no comments/docstrings except hard-to-spot corners | never call private methods outside class in prod (tests OK)

## Pydantic detail

Discriminated unions: sibling final classes + shared base, `Annotated[Union[...], Field(discriminator=...)]`. Never narrow discriminator in subclass (LSP violation, pyright rejects). `validation_alias` for kebab-case TOML keys. `Raises:` only for actually-raised exceptions.

## Async

`asyncio.create_task` + queues over blanket `gather` | `anyio.Path` for async file I/O | `AsyncGenerator[Event, None]` for streams | `httpx.AsyncClient` (mock with `respx`)

## Tools

Subclass `BaseTool` (`tools/base.py`) with Pydantic args model + `BaseToolConfig` generic. Implement `async def run(args, ctx: InvokeContext)`, yield events progressively. `ToolError` for failures, `ToolPermissionError` for authz. Declare `ToolPermission` (ALWAYS/ASK/NEVER).
Search: `lsp` for symbol questions (`go_to_definition`/`find_references`/`hover`/`incoming_calls`/`outgoing_calls`/`document_symbol` — resolves imports/re-exports/overloads grep misses; the semantic tool, prefer it for symbols when available) | `grep` for literal text (error messages, log lines, config values, regex) | `glob` to find files by name | Bash for system+git only

## Logging & errors

`from vibe.core.logger import logger` (stdlib `logging` + `StructuredLogFormatter`). Env: `LOG_LEVEL` (default `WARNING`), `LOG_MAX_BYTES`. Logs in `~/.vibe/logs/vibe.log`. `%s` positional args (deferred formatting, grep-friendly). Module-local exception hierarchies with `_fmt()` helper.

## TCSS

`$text-muted` → pair with `&:ansi { text-style: dim; }` | never `ansi_*` colors — use `$primary`/`$foreground`/`$surface`/`$error` (ANSI derived automatically)

## File I/O detail

`read_safe`/`read_safe_async`/`decode_safe` return `ReadSafeResult(text, encoding)`: UTF-8 → BOM → locale → `charset_normalizer` lazily. `raise_on_error=True` only when distinguishing corrupt files. Default replaces undecodable with U+FFFD.

## Tests

`@pytest.mark.asyncio` | mock HTTP with `respx` | autouse fixtures `config_dir`, `tmp_working_directory` | exempt from `ANN`/`PLR` ruff rules (`per-file-ignores`)

## Git

No `--amend`, no `--force`, no `--force-with-lease`. New commits + plain `git push`. Rejected push → rebase (never merge, never force-push).

## Versioning

hatch-vcs derives the version from `vX.Y.Z` git tags (`dynamic = ["version"]` in `pyproject.toml`). Never hand-edit a version literal: no `version =` in pyproject, no `__version__ =` string in `vibe/__init__.py` (it reads `importlib.metadata.version("chaton")`). A tag = a release; commits past a tag auto-produce a PEP 440 dev version. Cut releases with `uv run scripts/release.py <major|minor|patch>` (creates the tag); dev runs reflect the last `uv sync`.

## Workflows & Teams

`vibe/core/workflows/`: runtime (models, budget, schema validator, AST security, runtime with spawn_agent/parallel/pipeline, manager/discovery). `vibe/core/teams/`: TaskStore + Mailbox (file-backed), TeamManager (subprocess spawning). `bundled/` scripts have YAML frontmatter (excluded from ruff/pyright).
Workflow scripts: restricted namespace, safelisted builtins, AST validator blocks unsafe imports/dunders/`str.format`. **Defense-in-depth not hard boundary** — still `exec`s in-process; real boundary is `launch_workflow` ASK gate + `disable_workflows`. Treat model-authored scripts as untrusted.
`launch_workflow` hidden when `disable_workflows = true` (`is_available(config)`). Teammates spawned as `vibe -p` subprocesses; shared state via `filelock` (no in-process locks). Hooks: `TeammateIdle` (teammate idle), `TaskCreated`/`TaskCompleted` (lead-initiated `/team task add|done` only — teammate writes don't fire lead-side hooks).

## Verification

Host-agent completeness layer (on by default; `verification_subsystem = false` to disable). `verifier` subagent (`agents/models.py`) is a verdict gate distinct from `reviewer`: proves a completed implementation works by trying to break it, emits a strict `VERDICT: PASS|FAIL|PARTIAL` with command evidence. `_get_verification_contract_section` (`system_prompt.py`) tells the host to spawn it before reporting non-trivial work done; the todo tool (`tools/builtins/todo.py`) appends a structural nudge when a 3+ item list closes without a verify step. Read-only (read/grep/lsp + jailed bash, reuses `_review_bash_overrides`).

## Autoimprovement

Suggest new AGENTS.md rules from user input/PR comments when generalizable | suggest README.md updates for features | keep builtin Vibe Skill (`vibe/core/skills/builtins/vibe.py`) current (args, flags, config, commands, agents, file discovery).
