---
packet_schema: 1
id: I00-P01
title: External evidence runner and contract
iteration: 0
state: draft
change_class: tooling
risk: medium
owner: null
reviewer: null
verifier: null
depends_on: []
baseline_sha: null
candidate_sha: null
upstream_sha: null
worktree: null
branch: null
execution_profile: null
evidence:
  workspace: null
  run_id: null
  scenarios:
    - IT-13
acceptance_criteria:
  - AC-1.1
  - AC-1.2
  - AC-1.3
  - AC-1.4
  - AC-1.5
  - AC-7.1
messages:
  - I00-P01-MSG-01
paths:
  allowed:
    - scripts/run_maintenance_evidence.py
    - tests/maintenance/test_evidence_contract.py
  forbidden:
    - vibe/**
    - tests/snapshots/**
    - pyproject.toml
    - uv.lock
    - scripts/check_upstream_divergence.py
    - docs/**
---

# I00-P01: External Evidence Runner and Contract

Status: `draft`; implementation is not authorized until the lead fills the
frontmatter and records `ready` in `../status.yaml`.

## Outcome

Add a deterministic, non-shell evidence runner that executes one maintenance
scenario command from a clean committed candidate, writes best-effort artifacts
outside every Git worktree, hashes those artifacts, and atomically adds one
`pass` or `fail` scenario entry to the campaign manifest.

The runner is proven by contract tests covering success, controlled command
failure, dirty-candidate rejection, invalid evidence locations, normalization,
identity consistency, artifact loss, and concurrent manifest updates.

## Why this packet exists

Every later packet depends on evidence that does not dirty or mutate the
candidate it is meant to prove. The roadmap currently names
`scripts/run_maintenance_evidence.py` and
`tests/maintenance/test_evidence_contract.py` as planned deliverables; neither
exists at the planning snapshot. Implementing them first removes the need for
later workers to invent manifest, hashing, failure, or location semantics.

This packet is tooling only. It changes no `vibe` production path, CLI product
message, snapshot, provider request, prompt, harness workload, or performance
threshold.

## Definition of Ready

The campaign lead checks every item before setting `ready`:

- [ ] The execution-layer documentation is reviewed and committed.
- [ ] `baseline_sha` is the full clean commit immediately before this packet.
- [ ] `upstream_sha` is the full pinned upstream commit used by Iteration 0.
- [ ] Owner, reviewer, verifier, isolated worktree, branch, and execution
      profile are assigned in this packet and `../status.yaml`.
- [ ] The worktree is clean and contains neither unrelated changes nor another
      packet's edits.
- [ ] The absolute evidence workspace and unique run ID are assigned and do not
      overlap another active run.
- [ ] The lead confirms the CLI, manifest, normalization, and exit-code contracts
      below; the worker is not being asked to redesign them.
- [ ] No concurrent packet may modify either allowed path.

## Frozen lead decisions

- Architecture: one fork-added script owns CLI parsing, subprocess execution,
  normalization, hashing, locking, and manifest updates. Do not add a package,
  framework, plugin, service, or production hook.
- Invocation: execute an argument vector after `--` with `subprocess.run` and
  `shell=False`. Never accept or evaluate a shell command string.
- Candidate: the current repository must be clean, `HEAD` must equal
  `--candidate-ref`, and all three refs must resolve to full commits before the
  scenario command starts.
- Evidence: `VIBE_EVIDENCE_WORKSPACE`, `KILROY_RUN_ID`, and `--runner-id` are
  required. Evidence stays outside the repository, Git common directory, and
  all paths returned by `git worktree list --porcelain`.
- Manifest: schema version 1; scenario status is only `pass` or `fail`. A runner
  configuration/preflight error exits 2 and never launches the scenario.
- Scenario result: exit 0 only when the child exits 0 and every required artifact
  exists and is readable. A child failure or missing artifact exits 1 after
  writing best-effort evidence and a failed manifest entry.
- Concurrency: distinct scenario IDs may be recorded concurrently. Manifest
  read-modify-write is serialized by `filelock.FileLock`; scenario IDs are
  sorted in the manifest. Duplicate IDs are rejected before child execution.
- File I/O: repository helpers from `vibe.core.utils.io` perform text and durable
  writes. Do not use raw `Path.write_text()`, `open(..., "w")`, or ad hoc temp
  rename logic.
- Normalization: raw stdout/stderr are always retained. Optional normalized
  artifacts use only the exact transformations below; semantic text is frozen.
- Network/spend: no live network, provider credential, or paid model.
- Landing: the worker may not land, push, merge, or mark the packet complete.

## Worker discretion

The worker may choose private function and dataclass names inside the new script
and private fixture/helper names inside the new test file. The worker may split
the script into small functions but may not create additional repository files.

## Scope

### In scope

- Create `scripts/run_maintenance_evidence.py` with the exact interface and
  behavior below.
- Create `tests/maintenance/test_evidence_contract.py` with hermetic temporary
  Git repositories and no external network.
- Validate the roadmap manifest schema, external evidence boundary, failure
  behavior, and candidate identity.

### Out of scope

- Running the full Iteration 0 campaign.
- Implementing fork inventory, performance comparison, harness evaluations, UI
  screenshots, ACP capture, or programmatic CLI characterization.
- Changing the roadmap, packet documents, Git hooks, CI, dependencies,
  `pyproject.toml`, or `uv.lock`.
- Normalizing JSON/protocol semantics or comparing baseline to candidate.
- Supporting arbitrary shell syntax, pipelines, redirection, environment-file
  loading, remote artifact upload, or cleanup of prior evidence runs.

## Allowed paths

- `scripts/run_maintenance_evidence.py` — new evidence CLI and helpers.
- `tests/maintenance/test_evidence_contract.py` — new hermetic contract tests.

No `tests/maintenance/__init__.py` is needed. If test discovery proves otherwise,
stop; do not add it without revising the allowlist.

## Forbidden paths and actions

- `vibe/**` — no production behavior change.
- `tests/snapshots/**` — no snapshot acceptance.
- `docs/**` — the implementation packet does not revise its own contract.
- `pyproject.toml` and `uv.lock` — use existing stdlib, `filelock`, and project
  I/O helpers; add no dependency.
- `scripts/check_upstream_divergence.py` — divergence repair belongs to
  Iteration 1.
- No upstream-owned deletion, rename, split, relocation, broad formatting, or
  accepted-divergence edit.
- No environment value whose name contains `KEY`, `TOKEN`, `SECRET`,
  `PASSWORD`, or `CREDENTIAL` may be persisted.

## Required reading and inputs

Read before editing:

- `AGENTS.md`
- `openwiki/quickstart.md`
- Roadmap: “Delivery and rollback policy,” “Iteration 0,” “Message comparison
  rules,” “Test evidence contract,” “Manifest shape,” and IT-13/IT-14.
- `vibe/core/utils/io.py`
- Existing `filelock` usage under `vibe/core/teams/` and
  `vibe/core/workflows/`.
- `.pre-commit-config.yaml` and the test settings in `pyproject.toml`.

Required lead-filled inputs:

```bash
BASELINE_SHA=<frontmatter baseline_sha>
UPSTREAM_SHA=<frontmatter upstream_sha>
VIBE_EVIDENCE_WORKSPACE=<frontmatter evidence.workspace>
KILROY_RUN_ID=<frontmatter evidence.run_id>
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
RUNNER_ID=<stable non-secret machine/runner label assigned by the lead>
```

## Preflight

Run before any edit:

```bash
test -n "$BASELINE_SHA"
test -n "$UPSTREAM_SHA"
test -n "$VIBE_EVIDENCE_WORKSPACE"
test -n "$KILROY_RUN_ID"
test -n "$RUNNER_ID"
uv run git status --short
uv run git rev-parse HEAD
uv run git rev-parse "$BASELINE_SHA^{commit}"
uv run git rev-parse "$UPSTREAM_SHA^{commit}"
uv run git worktree list --porcelain
test ! -e scripts/run_maintenance_evidence.py
test ! -e tests/maintenance/test_evidence_contract.py
```

Required results:

- `git status --short` is empty.
- `git rev-parse HEAD` equals `BASELINE_SHA` at packet start.
- Both refs resolve to the frontmatter values.
- The two deliverable paths do not already contain another change.
- The evidence workspace is absolute and outside every listed worktree and Git
  common directory.

Any mismatch requests `blocked` with no edits.

## Command-line contract

The script supports this exact form:

```bash
VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
KILROY_RUN_ID="$KILROY_RUN_ID" \
uv run scripts/run_maintenance_evidence.py \
  --scenario IT-13 \
  --surface non_ui \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --upstream-ref "$UPSTREAM_SHA" \
  --runner-id "$RUNNER_ID" \
  --required-artifact junit=junit.xml \
  --record-env PYTHONHASHSEED \
  --normalize-output \
  -- \
  uv run pytest -n0 tests/maintenance/test_evidence_contract.py \
    --junitxml "$EVIDENCE/IT-13/junit.xml"
```

Required arguments and validation:

- `--scenario`: `IT-` followed by two decimal digits. Reject any other value.
- `--surface`: one of `ui`, `mixed`, or `non_ui`.
- `--baseline-ref`, `--candidate-ref`, `--upstream-ref`: required refs that are
  resolved and stored as full 40-character commit SHAs.
- `--runner-id`: required nonempty label; it must not contain a path separator
  or newline.
- `--required-artifact TYPE=RELATIVE_PATH`: optional and repeatable. The path is
  relative to the scenario directory, cannot escape it, and cannot collide with
  runner-owned filenames.
- `--record-env NAME`: optional and repeatable. Only an explicitly named,
  present, non-secret environment variable is recorded.
- `--normalize-output`: optional. It adds normalized stdout/stderr artifacts
  while always preserving raw text.
- `--`: required separator followed by at least one argument. Execute the vector
  directly with no shell.

Exit status:

- `0`: child exited 0, all required artifacts are readable, hashes were written,
  and the manifest entry is `pass`.
- `1`: child ran but exited nonzero, or a required artifact is missing/unreadable;
  the manifest entry is `fail` with notes.
- `2`: invalid arguments, refs, workspace, candidate cleanliness/identity,
  duplicate scenario, manifest identity/schema, secret env name, or evidence
  infrastructure failure. The scenario command did not start.

## Artifact and manifest contract

For scenario `IT-13`, the runner owns:

```text
$EVIDENCE/
├── manifest.json
├── .manifest.lock
└── IT-13/
    ├── command.json
    ├── command.log
    ├── stdout.txt
    ├── stderr.txt
    ├── result.json
    ├── stdout.normalized.txt   # only with --normalize-output
    ├── stderr.normalized.txt   # only with --normalize-output
    └── <required artifacts created by the child>
```

`command.json` stores the argument vector, working directory, selected recorded
environment, and start/end timestamps. It never stores the full environment.
`command.log` is a readable rendering that keeps stdout and stderr in separate
labelled sections; ordering between the two streams is not reconstructed.
`result.json` stores child exit code, duration, required-artifact findings,
scenario status, and notes. JSON is UTF-8, sorted by key, indented, and ends in a
newline.

`manifest.json` follows roadmap schema version 1 and additionally records each
scenario's `started_at`, `finished_at`, `result_path`, and artifact entries.
Artifact paths are POSIX-style paths relative to `$EVIDENCE`, never absolute.
Every readable regular-file artifact except `.manifest.lock` receives a SHA-256
digest. Symlinks are rejected as required artifacts.

On first write, create manifest identity from the resolved refs and environment.
On later writes, reject a version or identity mismatch. Never merge scenarios
from different baseline/candidate/upstream SHAs or lock digests into one run.

## Normalization contract

Raw evidence is never normalized or overwritten. With `--normalize-output`,
write separate normalized text using only these ordered transformations:

1. Remove ANSI CSI/OSC escape sequences; retain the visible text.
2. Replace the resolved candidate worktree path with `<WORKTREE>`.
3. Replace the resolved evidence root with `<EVIDENCE>`.
4. Replace UUIDs in canonical 8-4-4-4-12 hexadecimal form with `<UUID>`.
5. Replace ISO-8601 UTC timestamps containing a date and time with
   `<TIMESTAMP>`.
6. Replace decimal ports only when immediately preceded by `127.0.0.1:` or
   `localhost:` with `<PORT>`.

Do not normalize labels, severity, status, verdict, recovery guidance, exit
codes, arbitrary numbers/counters, tool names, protocol keys/values, spend
semantics, or verification state. The tests pin both changed and unchanged
examples.

## Implementation procedure

1. Create the script with stdlib argument parsing, typed immutable/data
   containers where useful, and a `main() -> int` entry point.
2. Resolve the repository root from the script location and Git refs using
   argument-vector subprocess calls. Validate clean status and candidate
   identity before creating the scenario directory or launching the child.
3. Resolve the evidence root from the two required environment variables.
   Reject empty, relative, repository-contained, Git-common-contained, and
   linked-worktree-contained paths using resolved path ancestry checks.
4. Validate scenario, surface, runner ID, recorded env names, command vector,
   and required artifact paths. Reject duplicates and path traversal.
5. Under the manifest lock, initialize or validate manifest identity and reserve
   the scenario ID before launching. A reservation is represented as a private
   runner state, not a third public scenario status. If implementation cannot
   recover a crashed reservation without inventing public status, stop and ask
   the lead rather than weakening the status contract.
6. Run the child with `shell=False`, repository root as `cwd`, inherited
   environment, captured text stdout/stderr, and no retry.
7. Persist raw command/output/result artifacts with project safe/durable I/O.
   If normalization was requested, persist separate normalized outputs.
8. Inspect required artifacts without following symlinks, record explicit notes
   for missing/unreadable items, and determine `pass` or `fail`.
9. Hash all readable artifacts, atomically write the sorted manifest entry under
   the lock, and return the contracted exit code.
10. Add contract tests using temporary Git repositories and temporary external
    evidence roots. Commands in tests write sentinel files only inside temporary
    directories and never contact a network.

## Required contract tests

At minimum, individual tests prove:

1. A passing child creates the fixed files, a `pass` entry, full SHAs,
   environment identity, relative paths, and correct SHA-256 digests.
2. A nonzero child returns 1 and creates raw outputs, result data, and a `fail`
   manifest entry.
3. A missing required artifact returns 1 and names the missing relative path in
   manifest notes while retaining other evidence.
4. A dirty candidate returns 2 before a sentinel child command can run.
5. A candidate ref different from `HEAD` returns 2 before child execution.
6. Evidence roots that are relative, inside the repository, inside its Git
   common directory, or inside any linked worktree return 2.
7. `..`, absolute, symlink, duplicate, or runner-owned required-artifact paths
   are rejected.
8. Missing/secret `--record-env` names are rejected or omitted according to the
   contract without ever serializing their values. Secret-like names return 2.
9. A duplicate scenario or manifest identity mismatch returns 2 and preserves
   the existing manifest byte-for-byte.
10. Normalized output changes every permitted volatile example and preserves
    labels, recovery guidance, exit codes, tool names, protocol fields, and
    arbitrary counters.
11. Two concurrent processes recording distinct scenario IDs do not lose either
    entry and leave valid sorted JSON.
12. Repeating the same deterministic command in two run IDs yields equivalent
    normalized scenario data after excluding the explicitly variable timestamps
    and duration.
13. Evidence writes leave the candidate's `git status --short` empty.

## Behavioral and structural invariants

- No import, startup, CLI, TUI, ACP, AgentLoop, workflow, team, config, prompt,
  spend, or provider production path changes.
- Child execution never uses a shell and never retries.
- Candidate cleanliness and identity are checked before the child runs.
- Failed child execution still produces best-effort evidence and public status
  `fail`.
- Infrastructure/preflight errors never masquerade as scenario failures and
  never launch the child.
- Evidence paths cannot escape the assigned external root.
- Artifact hashing and manifest updates are deterministic and atomic.
- No new dependency, suppression, accepted divergence, snapshot, performance
  baseline, or user/model-facing product message is introduced.

## User-facing and model-visible messages

The script is developer tooling, not a product entry point. Its diagnostics are
still a local CLI contract for later automation.

| Message ID | Trigger | Expected contract | Allowed normalization | Evidence |
|---|---|---|---|---|
| I00-P01-MSG-01 | Invalid input/preflight, scenario fail, or scenario pass | Concise stderr on exit 2; concise summary with scenario ID/status/evidence path on exit 0/1; no traceback for expected errors | Worktree/evidence paths only in normalized artifact; never status or recovery guidance | `$EVIDENCE/IT-13/stdout.txt`, `stderr.txt` |

Diagnostic text must name the failed check and recovery action without printing
secret values. Exact literals may be chosen in implementation, then pinned by
tests in this packet; later packets treat them as frozen.

## Acceptance criteria

| ID | Criterion | Proof |
|---|---|---|
| I00-P01-AC1 | A clean passing scenario emits the complete schema-v1 manifest and all fixed artifacts outside every worktree. | Contract tests; IT-13 manifest |
| I00-P01-AC2 | A child exit failure or missing required artifact emits `status: fail`, returns 1, and retains best-effort evidence. | Controlled-failure tests; IT-13 failure evidence |
| I00-P01-AC3 | Dirty/mismatched candidates and invalid evidence roots are rejected before child execution with exit 2. | Sentinel preflight tests |
| I00-P01-AC4 | Artifact digests, relative paths, resolved identities, lock digest, and recorded non-secret environment data are accurate. | Manifest/digest tests |
| I00-P01-AC5 | Concurrent distinct scenarios produce one valid sorted manifest without lost updates; duplicates cannot overwrite evidence. | Concurrency/duplicate tests |
| I00-P01-AC6 | Normalization changes only the explicitly permitted volatile fields and raw evidence remains untouched. | Normalization table tests |
| I00-P01-AC7 | Repeated deterministic runs normalize equivalently and do not dirty the candidate. | Reproducibility/status tests |
| I00-P01-AC8 | Targeted Ruff, format, Pyright, evidence tests, iron laws, and divergence checks pass without changing forbidden paths. | Exact verification commands |

Roadmap mapping: AC-1.1 and AC-1.2 are established by AC1/AC4; AC-1.3 by AC7;
AC-1.4 by AC2; AC-1.5 by AC3; AC-7.1 by AC8.

## Integration scenario

### IT-13: Evidence-contract slice of the repository quality gate

- Starting state: clean frozen candidate containing only the two allowed files;
  full baseline/upstream history; absolute external evidence workspace; no
  network; deterministic temporary repositories used by tests.
- Actions:
  1. Run the contract test file through the newly committed runner.
  2. Within pytest, exercise pass, controlled failure, missing artifact, dirty
     candidate, invalid root, normalization, identity, digest, duplicate, and
     concurrent-writer cases.
  3. Run the same deterministic fixture in a second run ID and compare normalized
     results.
- Expected outcome: the outer scenario passes; the intentional inner failure is
  asserted and retained as test evidence rather than failing the outer suite;
  candidate status remains clean.
- Failure evidence: outer runner writes raw stdout/stderr, result, manifest
  notes, and any produced JUnit report before returning 1 when possible.
- Artifacts: `$EVIDENCE/IT-13/{command.json,command.log,stdout.txt,stderr.txt,result.json,junit.xml}`
  plus the manifest entry and contract-test temporary-artifact summary written
  into JUnit output.
- Covers: AC-1.1 through AC-1.5, AC-7.1, I00-P01-MSG-01.

## Acceptance-to-scenario map

| Requirement | Scenario/review |
|---|---|
| I00-P01-AC1 through I00-P01-AC7 | IT-13 contract slice |
| I00-P01-AC8 | IT-13 plus quality/fork commands |
| AC-1.1 through AC-1.5 | IT-13 contract slice |
| AC-7.1 | Targeted quality/fork commands |
| I00-P01-MSG-01 | IT-13 raw/normalized output artifacts |

## Exact verification commands

Before candidate freeze, run fixing/formatting only on the allowed files:

```bash
uv run ruff check --fix scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run ruff format scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run pytest -n0 tests/maintenance/test_evidence_contract.py
uv run pyright scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
uv run scripts/check_upstream_divergence.py
uv run git diff --check
uv run git status --short
```

Review all formatter edits, freeze/commit through the lead-approved Git workflow,
fill `candidate_sha`, and require a clean worktree. Then run check-only commands:

```bash
uv run ruff check scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run ruff format --check scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run pyright scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py

VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
KILROY_RUN_ID="$KILROY_RUN_ID" \
uv run scripts/run_maintenance_evidence.py \
  --scenario IT-13 \
  --surface non_ui \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --upstream-ref "$UPSTREAM_SHA" \
  --runner-id "$RUNNER_ID" \
  --required-artifact junit=junit.xml \
  --normalize-output \
  -- \
  uv run pytest -n0 tests/maintenance/test_evidence_contract.py \
    --junitxml "$EVIDENCE/IT-13/junit.xml"

uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
uv run scripts/check_upstream_divergence.py
uv run git diff "$BASELINE_SHA".."$CANDIDATE_SHA" -- \
  scripts/run_maintenance_evidence.py \
  tests/maintenance/test_evidence_contract.py
uv run git diff --name-only "$BASELINE_SHA".."$CANDIDATE_SHA"
uv run git status --short
```

The final name-only output contains exactly the two allowed paths. The final
status is empty. Any check that modifies the frozen candidate invalidates freeze.

## Stop conditions

Stop and request `blocked` when:

- Safe implementation requires a third repository path or dependency.
- The project I/O helpers cannot be used without importing a production stack
  that materially changes script startup or causes a forbidden edit.
- Reliable manifest locking cannot be achieved with the existing `filelock`
  dependency.
- The evidence workspace cannot be proven outside every Git path.
- A test requires weakening candidate cleanliness, following symlinks, using a
  shell, recording secrets, or adding a third public scenario status.
- An existing test/quality/fork gate fails outside the two allowed files.
- The diff contains production, docs, snapshot, config, lockfile, suppression,
  or unrelated changes.
- A required command is denied, skipped, or needs unauthorized network/payment.
- The candidate changes after verification starts.

The smallest lead decision must name the exact proposed path or contract change;
the worker must not implement it speculatively.

## Rollback

Remove only:

- `scripts/run_maintenance_evidence.py`
- `tests/maintenance/test_evidence_contract.py`

External evidence may be retained for diagnosis but is not part of rollback.
Do not change the roadmap, status, snapshots, thresholds, or another packet.

## Completion report

Report:

- Packet ID, baseline/candidate/upstream SHAs, assigned run ID, and evidence
  manifest path.
- Exact two changed paths and confirmation that no third path changed.
- Each verification command and exit code.
- I00-P01-AC1 through AC8 and roadmap AC-1.1 through AC-1.5/AC-7.1 with direct
  evidence paths.
- Controlled-failure, dirty-rejection, normalization, digest, duplicate, and
  concurrency results.
- Product message, snapshot, production performance, fork metric, dependency,
  and suppression deltas, each explicitly `none` unless a blocker was raised.
- Any skip, denial, flake, missing artifact, or unresolved finding.
- Clean frozen status and confirmation that no push, merge, landing, or status
  completion action was taken.
