from __future__ import annotations

from vibe.core.skills.models import SkillInfo, SkillScope, SkillSource

_PROMPT = """# /init — set up AGENTS.md for this repository

You are bootstrapping a project's AGENTS.md: the file Vibe reads at the start \
of every session in this repo (and that every other AGENTS.md-aware tool reads \
too). A good AGENTS.md lets a fresh session be productive immediately instead \
of re-discovering the project's conventions by trial and error.

The goal is the smallest file that makes the model reliably correct. Bigger is \
worse: every line competes for the context budget on every turn, and generic \
advice the model already follows just trains it to skim the file.

## Phase 1 — confirm scope

Use `ask_user_question` (one question, two options) to confirm what to write:

- "Where should AGENTS.md live?"
  - "Project root (AGENTS.md)" — team-shared, committed. Recommended for most repos.
  - "User-level (~/.vibe/AGENTS.md)" — personal, applies to all your projects. Use this only if the user is clearly asking about their global setup, not this repo.

If an AGENTS.md already exists at the chosen path, you will be *updating* it, \
not creating it — read it first in Phase 2 and propose changes as a diff in \
Phase 3, never a silent overwrite.

If the user passes an argument to `/init` (e.g. `/init --user`), treat it as \
the answer to this question and skip asking.

## Phase 2 — survey the project

Do the survey yourself with `glob`, `grep`, `read`, `bash`, and `lsp`. Use \
`lsp` (`document_symbol`, `workspace_symbol`, `find_references`) for anything \
structural — call graphs, public API surface, where a symbol is defined or \
consumed — so you understand the architecture without reading every file. You \
are looking only for things the model could not guess from reading the code on \
demand. Read, in rough order of signal:

1. Manifest and build files — `pyproject.toml`, `package.json`, `Cargo.toml`, \
   `go.mod`, `pom.xml`, `mix.exs`, `deno.json`, `Makefile`, `justfile`. Note the \
   build/test/lint/format commands, especially non-standard ones.
2. README and any `CONTRIBUTING.md` / `DEVELOPMENT.md`.
3. CI config — `.github/workflows/`, `.gitlab-ci.yml`, `.circleci/`. The jobs \
   there reveal the canonical build and test sequence.
4. Existing AGENTS.md at the target path (if any) and any nested ones.
5. Tool-rule files from other agents, if present — `.cursor/rules/`, \
   `.cursorrules`, `.github/copilot-instructions.md`, `.windsurfrules`, \
   `.clinerules`, `CLAUDE.md`. If they encode real conventions, those should \
   flow into AGENTS.md rather than live in a competing file.
6. Formatter and linter configs (`ruff.toml`, `.prettierrc`, `biome.json`, \
   `.golangci.yml`, eslint configs, `.editorconfig`).
7. The repo's top-level shape — use `lsp workspace_symbol` and \
   `document_symbol` to map packages, entry points, and central abstractions \
   without reading every file. Is it a monorepo with workspaces? A single \
   package? Multiple deployable apps?

Run `git remote -v` and note the hosting platform (GitHub, GitLab, etc.) and \
the default branch name.

What you write down now is the *raw material*. You will cut most of it in \
Phase 3. Do not write the file yet.

## Phase 3 — propose before writing

Synthesize what you found into a proposed AGENTS.md and show it to the user \
*before* writing anything. Use `ask_user_question` with the proposal as the \
`question` text (or as a fenced markdown block in the question). Offer:

- "Looks good — write it"
- "Trim it more" (you cut harder — see the exclusion rules below)
- "I'll edit" (the user takes over; you stop)

The dialog overlay hides any text you emit before it, so put the full proposal \
*inside* the question, not in a message above it.

### What belongs in AGENTS.md

Keep a line only if it passes this test: **would a fresh session get this wrong \
without the file?** If yes, keep it. If no, cut it. Concretely, keep:

- Build, test, lint, and format commands the model can't infer from the \
  manifest — non-standard scripts, the exact flag for running a single test, \
  required sequences ("type-check before commit", "run migrations after pull").
- Repo etiquette the model would otherwise guess at — branch naming, commit \
  message style, PR conventions, where review happens.
- Architectural decisions that span multiple files and are not obvious from any \
  single one — the layering, the boundaries between modules, where new code \
  goes for a given kind of change.
- Required environment or setup steps that block the first run (and aren't in \
  the README).
- Non-obvious gotchas, invariants, or "don't do X" warnings proven by past bugs.
- Genuine conventions that differ from the language defaults (tabs vs spaces, \
  import ordering, naming).
- The important parts of any existing tool-rule files you read — subsume them \
  rather than duplicating them.

### What does NOT belong

- File-by-file directory walks or component lists. The model discovers these by \
  reading the code when it needs them.
- Standard language or framework conventions the model already knows.
- Generic engineering advice ("write tests", "handle errors gracefully", \
  "don't commit secrets").
- Anything that changes frequently — reference the source by path instead.
- Long reference material — point to it (`see docs/architecture.md`) rather \
  than copying it in.
- Commands obvious from the manifest (`npm test`, `cargo build`, `pytest` with \
  no flags).

Be specific. "Run a single test with `uv run pytest -k 'test_name'`" is worth \
keeping; "run the tests" is not.

### If updating an existing AGENTS.md

Propose a diff, not a replacement. For each change, say why: which section was \
vague, which command was wrong, which new convention the repo now follows. Do \
not delete existing content the user wrote unless you can justify it.

## Phase 4 — write the file

Once the user approves (or edits), write the AGENTS.md with `write_safe` (for a \
new file) or `edit` (for updates). No header preamble is required — start with \
a `# <project name>` H1 and a one-line description of what the repo is, then \
the sections you kept.

If you're writing the project-root file, also check whether it should be \
gitignored. Project-root `AGENTS.md` is normally committed (it's team-shared). \
User-level `~/.vibe/AGENTS.md` is inherently personal and needs no gitignore.

## Phase 5 — summary

Tell the user in two or three lines what you wrote and where, and remind them \
the file is a starting point — they should read it and adjust. Mention that \
they can run `/init` again later to re-survey after the project changes. Do not \
suggest installing other tools or skills; that's outside this skill's scope.
"""

SKILL = SkillInfo(
    name="init",
    description=(
        "Survey the current repository and write (or update) an AGENTS.md that "
        "captures the build/test/lint commands, conventions, and non-obvious "
        "gotchas a fresh session needs to be productive. Proposes before "
        "writing; never overwrites an existing file without a reviewed diff."
    ),
    summary="Generate or update a repo's AGENTS.md from a project survey.",
    user_invocable=True,
    prompt=_PROMPT,
    source=SkillSource.BUILTIN,
    scope=SkillScope.BUILTIN,
)
