# Operations: Fork Strategy, Testing & CI

## Fork Strategy

This repository is a fork of `mistralai/mistral-vibe` (remote `upstream`), synced continuously. The fork survives on cheap git merges; the enemy is **structural divergence** — git merges by path, so deleting/renaming/splitting a file upstream creates permanent `modify/delete` conflicts that can never 3-way merge.

### Core Rule: Add, Don't Restructure

From `AGENTS.md`:

- **New feature → new sibling file, thin hook in the upstream file.** Never split, rename, or relocate an upstream-owned file.
- **When you must edit an upstream file, keep edits minimal and localized** — small hunks 3-way merge; scattered edits and reordering conflict.
- **Do not reorder upstream's methods/functions.**

Example pattern: `vibe/core/agent_loop.py` (upstream-owned, 140 KB) + sibling `agent_loop_*.py` mixins (fork-added: `agent_loop_memory.py`, `agent_loop_failover.py`, `agent_loop_safety_judge.py`). `AgentLoop` composes fork-only subsystems as sibling-mixin bases.

### Upstream Divergence Guard

**Sources**: `scripts/check_upstream_divergence.py`, `tests/test_upstream_divergence.py`

- Compares files at `_MERGE_BASE` (a pinned merge-base SHA) against HEAD
- Flags any file that was deleted/renamed/moved and is not on an explicit allowlist (`_ACCEPTED_DIVERGENCE`)
- Rationale: structural divergence causes unmergeable conflicts on every future upstream sync
- Baseline can be overridden via `VIBE_UPSTREAM_BASE` env var (used by the upstream-sync workflow)
- **Bump `_MERGE_BASE` after each upstream sync**

### Upstream Sync Workflow

**Source**: `.github/workflows/upstream-sync.yml`

Automated upstream merge + divergence check. Updates are pulled from the `mistralai/mistral-vibe` upstream via the `upstream` git remote. To apply manually:

```bash
git pull upstream <tag> && uv sync --all-extras
```

### Update Checks

In-app update checks are **off by default** in this fork (`enable_update_checks = false`). The `mistral-vibe` PyPI package is the upstream release, not this fork's git install — never run `uv tool upgrade mistral-vibe`.

## Iron Laws

**Source**: `tests/test_iron_laws.py`

Enforces structural invariants on the production codebase via AST inspection. These are quality ratchets that can only improve, never regress:

1. **`test_config_models_declare_extra`** — Every Pydantic `BaseModel`/`BaseSettings` in `vibe/core/config/` must explicitly set `model_config = ConfigDict(extra=...)`. No accidental extra-field acceptance.

2. **`test_prod_type_ignore_ratchet`** — `# type: ignore` count must stay ≤ `TYPE_IGNORE_BUDGET` (0). Ratchet-only: never increase.

3. **`test_prod_noqa_ratchet`** — `# noqa` count must stay ≤ `NOQA_BUDGET` (3). Same ratchet approach.

4. **`test_prod_pyright_ignore_ratchet`** — `# pyright: ignore` is **banned entirely** (budget = 0). Fix at source.

## Testing

### Test Stack

- **Runner**: `pytest` with `pytest-asyncio` (async tests via `@pytest.mark.asyncio`)
- **Parallel**: `pytest-xdist` for parallel execution
- **HTTP mocking**: `respx` for mocking `httpx.AsyncClient`
- **Coverage**: `pytest-cov` with `--cov-report=xml`
- **Snapshots**: separate test suite for terminal UI regression coverage

### Test Structure

| Directory | Coverage |
|---|---|
| `tests/core/` (120+ files) | Unit-level core behavior: agent loop, config, tools, LLM, LSP, memory, hooks, MCP, telemetry, sandbox |
| `tests/tools/` (38 files) | Tool runtime: bash, sandbox, safety judge, MCP, connectors, grep, glob, task, websearch, workflows |
| `tests/cli/` | CLI and TUI behavior |
| `tests/acp/` | ACP mode |
| `tests/backend/` | Provider and adapter behavior |
| `tests/e2e/` | End-to-end CLI flows |
| `tests/snapshots/` | Terminal UI regression (SVG snapshots) |
| `tests/setup/` | Onboarding and sign-in flows |

### Test Conventions

From `AGENTS.md`:
- `@pytest.mark.asyncio` for async tests
- Mock HTTP with `respx`
- Autouse fixtures: `config_dir`, `tmp_working_directory`
- Tests exempt from `ANN`/`PLR` ruff rules (`per-file-ignores`)
- No docstrings in tests
- Test doubles in `tests/stubs/` named `Fake*` (e.g., `FakeBackend`, `FakeMCPRegistry`)
- Tests mirror source layout

### Conftest

**Source**: `tests/conftest.py`

Shared fixtures:
- `FakeBackend` — test-injectable LLM backend
- `FakeMCPRegistry`, `FakeVoiceManager`, `FakeWhoAmIGateway`, `FakeUpdateCacheRepository`, `FakeUpdateGateway`
- `sandbox_e2e_available()` — probes for bubblewrap/sandbox-exec
- `HarnessFilesManager` for test isolation
- Keyring mocked via `keyring_utils`

## CI

**Source**: `.github/workflows/ci.yml`

### CI Pipeline (3 jobs)

| Job | What | Key Steps |
|---|---|---|
| **pre-commit** | Linting | `uv sync --all-extras` → `pre-commit run --all-files --show-diff-on-failure` (cached) |
| **tests** | Full test suite | `uv sync` → verify `vibe --help` and `vibe-acp --help` → install `ripgrep` → `pytest --ignore tests/snapshots --cov --cov-report=xml` → upload coverage |
| **snapshot-tests** | Snapshot regression | `pytest tests/snapshots` → `continue-on-error: true`, uploads `snapshot_report.html` on failure, then fails |

- **Python 3.12**, `uv` for all dependency management
- Concurrency cancellation per ref
- Full git history (`fetch-depth: 0`) so the upstream-divergence guard can reach its baseline

### Other Workflows

| File | Purpose |
|---|---|
| `build-and-upload.yml` | PyInstaller binary build + upload (release tags) |
| `release.yml` | PyPI wheel publish from release branch |
| `upstream-sync.yml` | Automated upstream merge + divergence check |
| `security-audit.yml` | Dependency security scan |
| `update-snapshots.yml` | Regenerate snapshot test baselines |
| `issue-labeler.yml` | Auto-label GitHub issues |

### CI Pinning Rules

From `AGENTS.md`:
- Pin every `uses:` to a full **commit SHA** with an exact version comment
- Resolve to the commit, not the annotated-tag object
- Never pin a moving major tag (`v9`)

## Versioning & Release

### Versioning

- `hatch-vcs` derives the version from `vX.Y.Z` git tags (`dynamic = ["version"]` in `pyproject.toml`)
- Never hand-edit a version literal: no `version =` in pyproject, no `__version__ =` string in `vibe/__init__.py` (reads `importlib.metadata.version("mistral-vibe")`)
- A tag = a release; commits past a tag auto-produce a PEP 440 dev version
- Dev runs reflect the last `uv sync`

### Release Scripts

| Script | Purpose |
|---|---|
| `scripts/release.py` | Computes next semver, patches `extension.toml`, scaffolds changelog, creates `vX.Y.Z` tag. Supports `--dry-run` and `--init-baseline`. |
| `scripts/prepare_release.py` | Builds release branch from previous public tag, cherry-picks `-private` commits, squashes. **Freezes full transitive dependency graph** into `pyproject.toml` from `uv.lock` so PyPI wheel has exact pinned `Requires-Dist:` entries. `main` keeps `>=` ranges; each release re-snapshots. |

```bash
uv run scripts/release.py <major|minor|patch>
```

## Git Discipline

From `AGENTS.md`:

- No `--amend`, no `--force`, no `--force-with-lease`
- New commits + plain `git push`
- Push rejected because the upstream of the current branch advanced → rebase the current branch onto its upstream (never merge it in, never force-push)
- Once a PR is open, reconcile the base branch (`origin/main`) by merging it into the current branch (not rebase — rebasing rewrites already-pushed history)
- Run git through `uv run` (`uv run git commit`, `uv run git push`) so pre-commit hooks resolve the project venv

## Lint & Type Check

```bash
# After every change:
uv run ruff check --fix . && uv run ruff format .

# Type check (standard mode, pinned in pyproject):
uv run pyright

# Full lint pass:
uv run pre-commit run --all-files
```

## What to Verify When Changing Code

| Change Area | Tests to Run |
|---|---|
| Agent loop | `tests/core/test_agent_loop_*.py`, `tests/agent_loop/` |
| Tools | `tests/tools/`, `tests/core/test_tool_*.py` |
| UI/widgets | `tests/snapshots/` (expect baseline regeneration) |
| Config | `tests/core/test_config_*.py` |
| Workflows | `tests/core/workflows/`, `tests/tools/test_workflow_*` |
| ACP | `tests/acp/` |
| Fork-sensitive file | Confirm no upstream sync trap created; run `tests/test_upstream_divergence.py` |

## High-Signal Source Files

- `AGENTS.md` — operational rules and fork discipline (read this first)
- `pyproject.toml` — package metadata, entry points, tool configuration
- `scripts/check_upstream_divergence.py` — divergence guard
- `scripts/release.py` and `scripts/prepare_release.py` — release flow
- `tests/test_iron_laws.py` — structural quality ratchets
- `.github/workflows/ci.yml` — CI pipeline definition
- `CONTRIBUTING.md` — contribution guidelines
