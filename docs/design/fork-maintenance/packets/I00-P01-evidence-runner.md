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
evidence_operator: null
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
  runner_id: null
  scenarios:
    - IT-13
  scenario_contracts: []
packet_acceptance_criteria:
  - I00-P01-AC01
  - I00-P01-AC02
  - I00-P01-AC03
  - I00-P01-AC04
  - I00-P01-AC05
  - I00-P01-AC06
  - I00-P01-AC07
  - I00-P01-AC08
  - I00-P01-AC09
  - I00-P01-AC10
  - I00-P01-AC11
  - I00-P01-AC12
  - I00-P01-AC13
  - I00-P01-AC14
  - I00-P01-AC15
  - I00-P01-AC16
  - I00-P01-AC17
  - I00-P01-AC18
  - I00-P01-AC19
roadmap_contributions:
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

Execution state is read only from this packet's frontmatter and the matching
`../status.yaml` entry at the assigned clean `CONTROL_SHA`; this prose does not
duplicate state. Implementation is authorized only when both record `ready` and
the lead has filled every required field.

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
- [ ] `baseline_sha` is the full clean packet-start commit immediately before
      this packet; it is not yet the final campaign preservation baseline.
- [ ] `upstream_sha` is the full pinned upstream commit used by Iteration 0.
- [ ] Owner, reviewer, verifier, evidence operator, isolated worktree, branch,
      and execution profile are assigned in this packet and `../status.yaml`.
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
- Invocation: execute an argument vector after `--` with `subprocess.Popen`,
  `shell=False`, and an isolated process group/session so timeout cleanup reaches
  descendants. Never accept or evaluate a shell command string.
- Candidate: the current repository must be clean, `HEAD` must equal
  `--candidate-ref`, and all three refs must resolve to full commits before the
  scenario command starts.
- Git state: use a sanitized Git environment with `GIT_CONFIG_NOSYSTEM=1`,
  `GIT_CONFIG_GLOBAL=/dev/null`, and `GIT_OPTIONAL_LOCKS=0`. Cleanliness is exact
  NUL-safe `git status --porcelain=v1 -z --untracked-files=all
  --ignore-submodules=none`; do not rely on user `status.*` configuration.
- Bootstrap: this packet's candidate may later contribute to the consolidated
  Iteration 0 baseline, but only the lead freezes `campaign_baseline_sha` after
  every Iteration 0 deliverable passes. This packet does not self-promote it.
- Evidence: `VIBE_EVIDENCE_WORKSPACE`, `KILROY_RUN_ID`, and `--runner-id` are
  required. Evidence stays outside the repository, Git common directory, and
  all paths returned by `git worktree list --porcelain`.
- Manifest: schema version 1; scenario status is only `pass` or `fail`. A runner
  configuration/preflight error exits 2 and never launches the scenario.
- Scenario result: exit 0 only when the child exits 0 and every required artifact
  exists and is readable. A child failure or missing artifact exits 1 after
  writing best-effort evidence and a failed manifest entry.
- Concurrency: distinct scenario IDs may be recorded concurrently. Manifest
  read-modify-write is serialized by `filelock.FileLock`; durable reservation
  files prevent duplicate execution while the lock is released; scenario IDs
  are sorted in the manifest. Duplicate IDs are rejected before child execution.
- File I/O: repository helpers from `vibe.core.utils.io` perform text and durable
  writes. Do not use raw `Path.write_text()`, `open(..., "w")`, or ad hoc temp
  rename logic.
- Normalization: after the clean-environment/argv security preflight, child
  stdout/stderr are retained verbatim. Optional normalized artifacts use only
  the exact transformations below; semantic text is frozen.
- Network/spend: no live network, provider credential, or paid model.
- Platform: runner schema v1 is Linux/POSIX-only. Preflight rejects other
  platforms with exit 2. Timeout cleanup uses `start_new_session=True`; it sends
  SIGTERM and, after a fixed five-second grace period, SIGKILL to the group only
  while the child PID remains both its session and process-group leader.
  Otherwise it signals only the direct child.
- Child environment: copy the parent environment, remove names containing the
  secret terms case-insensitively, remove proxy variables, and set
  `UV_OFFLINE=1` and `PIP_NO_INDEX=1`. Preflight rejects when any credential-like
  parent variable has a nonempty value; the evidence operator must launch from a
  clean environment rather than relying on lossy output redaction. Packet
  commands may add only explicitly approved non-secret variables. External/paid
  execution requires a later separately reviewed runner extension.
- Operator launch environment: every frozen packet command invokes the runner
  through `env -i` with only `PATH`, `HOME`, `TMPDIR`, fixed locale/timezone,
  deterministic Git controls, offline package controls, packet-approved `VIBE_`
  controls, `KILROY_RUN_ID`, and explicitly recorded non-secret variables. This
  allowlist is part of the command contract; never replace it with the inherited
  operator environment.
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
CONTROL_WORKTREE=<absolute clean control worktree>
CONTROL_SHA=<immutable control commit supplied by the lead>
BASELINE_SHA=<frontmatter baseline_sha>
UPSTREAM_SHA=<frontmatter upstream_sha>
VIBE_EVIDENCE_WORKSPACE=<frontmatter evidence.workspace>
KILROY_RUN_ID=<frontmatter evidence.run_id>
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
RUNNER_ID=<stable non-secret machine/runner label assigned by the lead>
REPO_ROOT=<absolute candidate worktree root>
```

## Preflight

Run before any edit:

```bash
test -n "$CONTROL_WORKTREE"
test -n "$CONTROL_SHA"
test -n "$PATH"
test -n "$HOME"
test "$(uv run git -C "$CONTROL_WORKTREE" status --short)" = ""
test "$(uv run git -C "$CONTROL_WORKTREE" rev-parse HEAD)" = "$CONTROL_SHA"
test -n "$BASELINE_SHA"
test -n "$UPSTREAM_SHA"
test -n "$VIBE_EVIDENCE_WORKSPACE"
test -n "$KILROY_RUN_ID"
test -n "$RUNNER_ID"
test -n "$REPO_ROOT"
test "$(uname -s)" = "Linux"
test "$(GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_OPTIONAL_LOCKS=0 uv run git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)" = ""
test "$(uv run git rev-parse --show-toplevel)" = "$REPO_ROOT"
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

Run the canonical control-metadata validator in
`../task-packet-template.md` with `PACKET_ID=I00-P01` and
`PACKET_RELATIVE_PATH=docs/design/fork-maintenance/packets/I00-P01-evidence-runner.md`.
An assertion failure requests `ready -> blocked` with no edits.

## Command-line contract

The script supports this exact form:

```bash
PYTHONHASHSEED=0 \
VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
KILROY_RUN_ID="$KILROY_RUN_ID" \
uv run scripts/run_maintenance_evidence.py \
  --repo-root "$REPO_ROOT" \
  --scenario IT-13 \
  --surface non_ui \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --upstream-ref "$UPSTREAM_SHA" \
  --runner-id "$RUNNER_ID" \
  --timeout-seconds 300 \
  --lock-timeout-seconds 10 \
  --required-artifact junit=junit.xml \
  --record-env PYTHONHASHSEED \
  --normalize-output \
  -- \
  uv run pytest -n0 tests/maintenance/test_evidence_contract.py \
    --junitxml "$EVIDENCE/IT-13/junit.xml"
```

Required arguments and validation:

- `--repo-root`: required absolute Git worktree used for ref, worktree,
  cleanliness, command cwd, and post-command fingerprint checks. Temporary-repo
  contract tests pass their own roots; never infer it from the script location.
- `--scenario`: `IT-` followed by two decimal digits. Reject any other value.
- `--surface`: one of `ui`, `mixed`, or `non_ui`.
- `--baseline-ref`, `--candidate-ref`, `--upstream-ref`: required refs that are
  resolved and stored as full 40-character commit SHAs.
- `--runner-id`: required nonempty label; it must not contain a path separator
  or newline.
- `--timeout-seconds`: required integer from 1 through 86400. On timeout,
  terminate then kill the child process group within a bounded cleanup window
  only while signal-time ownership checks still pass; otherwise signal the
  direct child. Retain outputs, record a failed scenario, and return 1.
- `--lock-timeout-seconds`: required integer from 1 through 60. Every
  `.manifest.lock` acquisition uses this bound. Timeout returns 2 in category
  `collision` before dispatch, or 3 with emergency evidence after dispatch;
  recovery names the lock path and instructs the lead to inspect its owner.
- `--required-artifact TYPE=RELATIVE_PATH`: optional and repeatable. The path is
  relative to the scenario directory, cannot escape it, and cannot collide with
  runner-owned filenames.
- `--gap-note TEXT`: optional and repeatable. Each nonempty note documents a
  planned or unavailable scenario capability, causes the final public scenario
  status to be `fail`, writes `gap.json`, and returns 1 even when the child exits
  0. It can never force a pass or hide a child failure.
- `--record-env NAME`: optional and repeatable. Names must match
  `[A-Z_][A-Z0-9_]*`, be present, and not contain a secret-like term. An invalid,
  missing, or secret-like requested name returns 2 before child execution.
- `--normalize-output`: optional. It adds normalized stdout/stderr artifacts
  while preserving the preflight-approved child text verbatim.
- `--`: required separator followed by at least one argument. Execute the vector
  directly with no shell.

Exit status:

- `0`: child exited 0, all required artifacts are readable, hashes were written,
  and the manifest entry is `pass`.
- `1`: child ran but exited nonzero, a required artifact is missing/unreadable,
  or at least one `--gap-note` was supplied; the manifest entry is `fail` with
  notes.
- `2`: invalid arguments, refs, workspace/run ID, candidate
  cleanliness/identity, duplicate scenario, manifest identity/schema, secret env
  name, or pre-dispatch evidence infrastructure failure. The scenario command
  did not start.
- `3`: post-dispatch evidence persistence, hashing, or manifest-finalization
  failure. The child may have run. Emit a best-effort
  `$EVIDENCE/.runner-failure-<scenario>.json` with `command_started`, child exit
  when known, error category, candidate SHA, and `manifest_committed`. When false,
  no scenario status is claimed. When true, the durable manifest entry remains
  authoritative and exit 3 reports cleanup debt, such as reservation unlink
  failure. This is a campaign blocker requiring lead recovery.

`KILROY_RUN_ID` must match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`. Resolve the full
root and reject symlink ancestry escapes. An existing run root is reusable only
when its manifest identity matches. A new run root without a manifest may
contain only the current scenario directory and files explicitly named by
`--required-artifact`; any undeclared or runner-owned preexisting file is a
collision and returns 2. This permits a packet to stage declared report artifacts
before the runner executes the scenario command without permitting silent
overwrite.

Artifact type labels match `[a-z][a-z0-9_-]{0,31}`. Types and relative paths are
each unique, paths are at most 240 characters, and manifest artifact entries
sort by `(type, path)`. Re-resolve containment after child exit and reject a
symlink in any ancestor component, not only at the leaf. Hard-linked required
artifacts whose inode is also reachable inside the repository or a Git worktree
are rejected.

## Artifact and manifest contract

For scenario `IT-13`, the runner owns:

```text
$EVIDENCE/
├── manifest.json
├── .manifest.lock
├── .reservations/
│   └── IT-13.json          # present only while the scenario owns its reservation
└── IT-13/
    ├── command.json
    ├── command.log
    ├── stdout.txt
    ├── stderr.txt
    ├── result.json
    ├── gap.json                # only with one or more --gap-note values
    ├── stdout.normalized.txt   # only with --normalize-output
    ├── stderr.normalized.txt   # only with --normalize-output
    └── <required artifacts created by the child>
```

`command.json` stores the argument vector as a JSON string array, working
directory, selected scenario-recorded environment, and start/end timestamps. It
never stores the full environment. Scenario-recorded variables are not part of
top-level manifest identity, so scenarios in one frozen run may record different
non-secret variables.
`command.log` is a readable rendering that keeps stdout and stderr in separate
labelled sections; ordering between the two streams is not reconstructed.
`result.json` stores `command_started`, `child_exit_code`, `child_signal`,
`timed_out`, `duration_seconds`, `required_artifacts`, `gap_notes`, scenario
`status`, and `notes`. `gap.json` is exactly
`{"version": 1, "notes": [<ordered gap-note strings>]}` when present. JSON is
UTF-8, sorted by key, indented, and ends in a newline.
Each `required_artifacts` entry is
`{"type": <label>, "path": <relative path>, "status": "readable"|"missing"|"unreadable"}`
and sorts by `(type, path)`.

`manifest.json` follows roadmap schema version 1 and additionally records each
scenario's `started_at`, `finished_at`, `result_path`, and artifact entries.
Artifact paths are POSIX-style paths relative to `$EVIDENCE`, never absolute.
Every readable regular-file artifact inside the scenario directory receives a
SHA-256 digest. `manifest.json` and `.manifest.lock` are run metadata rather than
scenario artifacts and are not self-hashed. Symlinks are rejected as required
artifacts.

For schema v1, the roadmap's example `$EVIDENCE/IT-01/command.log` is descriptive
notation, not a literal stored value. Stored artifact paths are relative POSIX
strings such as `IT-01/command.log`, and the scenario `command` field is the
exact argv string array rather than a shell-formatted string. Environment fields
are exactly `python`, `platform`, `uv_lock_sha256`, and `runner`. The sorted
explicit `recorded_environment` map lives in the scenario entry and
`command.json`, not immutable run identity.

On first write, create manifest identity from the resolved refs and environment.
On later writes, reject a version or identity mismatch. Never merge scenarios
from different baseline/candidate/upstream SHAs or lock digests into one run.

Reservation records are private schema version 1 JSON containing scenario ID,
resolved candidate SHA, process ID, runner ID, and creation timestamp. Under the
manifest lock, reject an existing manifest scenario or reservation, then create
the reservation with `write_durable` before releasing the lock.
Release the lock while the child runs. Reacquire it to write the manifest, then
remove the reservation only after the manifest is durable.

If reservation removal fails after the manifest is durable, do not roll back or
duplicate the manifest entry. Emit exit 3 emergency evidence with
`manifest_committed: true`; the manifest status is authoritative and prevents a
duplicate run. The lead resolves the stale reservation as cleanup debt.

There is no automatic stale-owner recovery. A crash leaves its reservation in
place; later attempts return 2 without running and name the reservation path.
Only the campaign lead may remove it after proving no owner process remains and
retaining the record as failure evidence. This conservative policy avoids PID
reuse and cross-host clock assumptions. Distinct scenario reservations do not
block one another except for the short manifest-lock sections.

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

Cross-run equivalence uses a separate canonical comparison projection; it does
not compare volatile raw JSON bytes. The projection:

1. Replaces each run's exact evidence root with `<EVIDENCE>` in argv,
   scenario-recorded environment values, and text artifacts.
2. Omits start/finish timestamps, process IDs, reservation metadata, and duration.
3. Retains scenario ID/surface/status, child exit/signal/timeout, gaps, notes,
   required-artifact findings, semantic command arguments, and every
   nonvolatile artifact digest.
4. Compares canonical projections of `command.json` and `result.json` rather
   than their raw digests; normalized stdout/stderr and required deterministic
   artifacts retain digest comparison.

The test writes the compared projections and field-level result into its test
artifact/JUnit output. Evidence-root replacement is allowed only in this
comparison projection, not in stored raw command data.

## Implementation procedure

1. Create the script with stdlib argument parsing, typed immutable/data
   containers where useful, and a `main() -> int` entry point.
2. Resolve and validate the explicit repository root and Git refs using
   argument-vector subprocess calls under the frozen Git environment. Validate
   exact clean status and candidate identity before creating the scenario
   directory or launching the child.
3. Resolve the evidence root from the two required environment variables.
   Reject empty, relative, repository-contained, Git-common-contained, and
   linked-worktree-contained paths using resolved path ancestry checks.
4. Validate scenario, surface, runner ID, recorded env names, command vector,
   and required artifact paths. Secret-term matching is case-insensitive. Reject
   command arguments that contain a credential-like flag/name whether the value
   is joined (`name=value`/`name:value`) or supplied as the following split
   argument. Reject before persisting any argv, plus duplicates and path
   traversal.
5. Under the manifest lock, initialize or validate manifest identity, validate
   any explicitly declared preexisting artifacts, and durably reserve the
   scenario ID using the exact private reservation schema/policy above. The
   reservation is not a third public scenario status.
6. Run the child with `shell=False`, repository root as `cwd`, the frozen
   sanitized child environment, captured text stdout/stderr, an isolated process
   group/session, the required timeout, and no retry. Timeout or signal
   termination is a scenario `fail` with best-effort artifacts.
7. After every direct-child exit, probe the verified-owned Linux process group
   for surviving descendants. If any remain and the child PID still identifies
   both the session and group, SIGTERM, wait five seconds, SIGKILL, reap, force
   scenario `fail`, and record descendant cleanup. Otherwise signal only the
   direct child and record that group ownership could not be established.
8. Persist security-screened command/output/result artifacts with project
   safe/durable I/O. Because credential-bearing parent environments and argv are
   rejected pre-dispatch, stored stdout/stderr retain child text without
   credential-value substitution. If normalization was requested, persist
   separate normalized outputs.
9. Recheck `HEAD` and the same exact NUL-safe Git status after child cleanup. A
   moved or dirty candidate is a scenario `fail`; retain evidence and never
   revert the mutation.
10. Inspect required artifacts without following symlinks, record explicit notes
   for missing/unreadable items, apply any explicit gap notes, and determine
   `pass` or `fail`.
11. Hash all readable artifacts, atomically write the sorted manifest entry under
   the lock, and return the contracted exit code.
12. Add contract tests using temporary Git repositories and temporary external
    evidence roots. Configure repository-local test author identity/timestamps;
    never use global Git identity. Commands in tests write sentinel files only
    inside temporary directories and never contact a network.

## Required contract tests

At minimum, individual tests prove:

1. A passing child creates the fixed files, a `pass` entry, full SHAs,
   environment identity, relative paths, and correct SHA-256 digests from a
   fixture commit containing `uv.lock`.
2. A nonzero child returns 1 and creates raw outputs, result data, and a `fail`
   manifest entry.
3. A missing required artifact returns 1 and names the missing relative path in
   manifest notes while retaining other evidence.
4. One or more explicit gap notes return 1, write deterministic `gap.json`, and
   cannot turn a child failure into a pass.
5. Default tests mock every OS signal while proving the ownership guard. The
   opt-in `process_e2e` timeout probe runs only with `-n0` on a disposable
   non-graphical host, terminates the verified-owned group, returns 1, and
   records the timeout plus best-effort stdout/stderr.
6. A dirty candidate returns 2 before a sentinel child command can run.
7. A candidate ref different from `HEAD` returns 2 before child execution.
8. Evidence roots that are relative, inside the repository, inside its Git
   common directory, or inside any linked worktree return 2.
9. Unsafe run IDs and `..`, absolute, symlink, duplicate, or runner-owned
   required-artifact paths are rejected.
10. Missing, malformed, or secret-like `--record-env` names return 2 before child
    execution and never serialize their values.
11. A duplicate scenario or manifest identity mismatch returns 2 and preserves
   the existing manifest byte-for-byte.
12. A preexisting scenario directory is accepted only when every existing file
    is a declared required artifact; undeclared/runner-owned collisions return 2
    without overwrite.
13. Normalized output changes every permitted volatile example and preserves
    labels, recovery guidance, exit codes, tool names, protocol fields, and
    arbitrary counters.
14. Two concurrent processes recording distinct scenario IDs do not lose either
    entry and leave valid sorted JSON.
15. A simulated crash leaves a durable reservation; a duplicate cannot run, no
    public third status appears, and only documented lead cleanup permits retry.
16. Repeating the same deterministic command in two run IDs yields equivalent
    normalized scenario data after excluding the explicitly variable timestamps
    and duration.
17. Evidence writes leave the candidate's `git status --short` empty; a child
    that dirties or moves the candidate produces `fail` and the runner does not
    hide or revert it.
18. A forced post-dispatch write/hash/finalization error returns 3 and records
    `command_started`, `manifest_committed`, and best-effort emergency evidence
    without inventing or erasing a public verdict.
19. A nonempty mixed-case credential environment name or joined/split
    credential-like argv form is rejected before dispatch or persistence.
20. A held manifest lock times out at the configured bound and returns the
    correct pre- or post-dispatch infrastructure exit without hanging.
21. A direct child that exits 0 after spawning a detached-in-group descendant
    is cleaned up within the fixed grace period and yields `status: fail`.
22. A forced reservation-unlink failure after manifest commit returns 3 while
    preserving the authoritative manifest entry and reporting cleanup debt.

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
| I00-P01-MSG-01 | Scenario pass/fail or runner error | Exit 0 stdout: `PASS <IT-ID> <absolute-scenario-path>\n`; exit 1 stdout: `FAIL <IT-ID> <absolute-scenario-path>\n`; exit 2 stderr: `ERROR <category>: <detail>\nRecovery: <action>\n`; exit 3 stderr: `FATAL evidence error: <detail>\nRecovery: inspect <absolute-emergency-artifact>\n`; the other stream is empty and expected errors have no traceback | Worktree/evidence paths only in normalized artifact; never category, status, detail, or recovery action | Contract-test subprocess captures in `$EVIDENCE/IT-13/junit.xml` and result/emergency artifacts |

Exit-2 categories are exactly `arguments`, `platform`, `repository`,
`workspace`, `collision`, `manifest`, or `environment`. Detail text contains
only these frozen templates:

- `arguments`: `invalid <field>: <safe-value>` / `correct <field> and retry`.
- `platform`: `unsupported platform: <platform>` / `run this packet on Linux`.
- `repository`: `<check> failed for <repo-root>: <safe-value>` /
  `restore the frozen clean candidate and retry with a new run ID`.
- `workspace`: `<check> failed: <path>` /
  `choose an absolute evidence workspace outside every Git worktree`.
- `collision`: `<kind> already exists: <path-or-id>` /
  `inspect the existing run or reservation; only the lead may authorize cleanup`.
- `manifest`: `<field> mismatch: expected <safe-value>, got <safe-value>` /
  `use a new run ID for the frozen identities`.
- `environment`: `<name> is missing or unsafe` /
  `set an approved non-secret variable or remove --record-env`.

Tests pin one exact representative literal for every category plus exit 0/1/3.
The worker may substitute only the bracketed validated values and may not invent
a new category, prefix, stream, detail grammar, recovery action, or layout.

## Acceptance criteria

| ID | Criterion | Proof |
|---|---|---|
| I00-P01-AC01 | A clean passing fixture creates one schema-v1 manifest entry with `status: pass`. | Passing contract test |
| I00-P01-AC02 | Every runner-owned and declared artifact is outside all Git worktrees. | External-boundary contract test |
| I00-P01-AC03 | The nonzero-child result projection equals `{status: fail, child_exit_code: <fixture>, manifest_entries: 1}`. | Controlled child-failure test |
| I00-P01-AC04 | The missing-artifact projection equals `{status: fail, finding: {path: <fixture>, status: missing}}`. | Missing-artifact test |
| I00-P01-AC05 | The explicit-gap projection equals `{status: fail, gap_json_notes: <ordered input notes>}`. | Gap contract test |
| I00-P01-AC06 | A dirty or moved candidate is rejected before dispatch. | Sentinel repository-state test |
| I00-P01-AC07 | An invalid/escaping evidence root is rejected before dispatch. | Workspace containment matrix |
| I00-P01-AC08 | The manifest identity projection exactly equals the fixture's SHAs, lock digest, runner/environment fields, relative paths, and computed SHA-256 values. | Identity/digest test |
| I00-P01-AC09 | Concurrent distinct reservations produce one sorted manifest without lost entries. | Concurrent-process test |
| I00-P01-AC10 | Each duplicate/stale reservation attempt projection equals `{command_started: false, existing_bytes_unchanged: true}`. | Duplicate/crash-reservation test |
| I00-P01-AC11 | Normalized artifacts change only the six permitted volatile classes. | Positive/negative normalization table |
| I00-P01-AC12 | Two deterministic run IDs produce equivalent normalized scenario data. | Reproducibility comparison |
| I00-P01-AC13 | The child-mutation projection equals `{status: fail, post_status_dirty: true, reverted: false}`. | Post-child fingerprint test |
| I00-P01-AC14 | The timeout projection equals `{status: fail, timed_out: true, group_alive_after_grace: false}`. | Descendant-timeout test |
| I00-P01-AC15 | The injected finalization-fault projection equals `{runner_exit: 3, command_started: true, manifest_committed: <fixture expectation>}`. | Injected finalization-fault test |
| I00-P01-AC16 | The credential preflight result equals rejection for every mixed-case environment and joined/split argv fixture. | Secret rejection matrix |
| I00-P01-AC17 | A held manifest lock returns the contracted bounded infrastructure result instead of hanging. | Pre/post-dispatch lock-timeout tests |
| I00-P01-AC18 | The normal-exit descendant projection equals `{direct_exit: 0, status: fail, group_alive_after_grace: false}`. | Background-descendant test |
| I00-P01-AC19 | The packet quality projection equals the frozen command/exit map plus changed-path set `{scripts/run_maintenance_evidence.py, tests/maintenance/test_evidence_contract.py}`. | Command log and name-only diff |

These criteria establish evidence mechanisms and contribute to roadmap AC-1.1
through AC-1.5 and AC-7.1. They do not declare those campaign-wide criteria
complete; the outer IT-13 entry remains `fail` until I00-P99 runs the full gate.

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
- Expected outcome: the child contract suite passes and candidate status remains
  clean. The outer IT-13 entry is deliberately `fail` because this packet does
  not run the full repository coverage, snapshot, warning/ratchet,
  cross-scenario reproducibility, and verifier gate.
- Failure evidence: outer runner writes raw stdout/stderr, result, manifest
  notes, and any produced JUnit report before returning 1 when possible.
- Artifacts: `$EVIDENCE/IT-13/{command.json,command.log,stdout.txt,stderr.txt,result.json,junit.xml,gap.json}`
  plus the manifest entry and contract-test temporary-artifact summary written
  into JUnit output.
- Covers: I00-P01-AC01 through I00-P01-AC18; I00-P01-MSG-01.
- Packet quality/diff gates cover: I00-P01-AC19.
- Contributes to: AC-1.1 through AC-1.5, AC-7.1.

## Acceptance-to-scenario map

| Requirement | Scenario/review |
|---|---|
| I00-P01-AC01 through I00-P01-AC18 | IT-13 contract slice |
| I00-P01-AC19 | Targeted quality/fork commands and diff review |
| AC-1.1 through AC-1.5 | Contribution only; final verdict belongs to I00-P99 |
| AC-7.1 | Contribution only; final verdict belongs to I00-P99 |
| I00-P01-MSG-01 | IT-13 contract-test subprocess captures and result artifacts |

## Exact verification commands

Before candidate freeze, run fixing/formatting only on the allowed files:

```bash
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_OPTIONAL_LOCKS=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run ruff check --fix scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run ruff format scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run pytest -n0 tests/maintenance/test_evidence_contract.py
uv run pyright scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
  uv run scripts/check_upstream_divergence.py
uv run pre-commit run --all-files
uv run git diff --check
uv run git status --short
```

Review all formatter edits, then stop at the freeze handoff. The lead reviews and
creates the candidate commit, writes `candidate_sha` to a new clean control
commit, and assigns the evidence operator. The worker does not commit or edit
control files. Require a clean candidate worktree. The evidence operator reruns
the canonical validator with `CONTROL_SHA` set to the newly assigned clean
verification-state control commit, `EXPECTED_PACKET_STATE=verification`, and
`EXPECTED_CANDIDATE_SHA="$CANDIDATE_SHA"`, then runs check-only commands:

```bash
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_OPTIONAL_LOCKS=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run ruff check scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run ruff format --check scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py
uv run pyright scripts/run_maintenance_evidence.py tests/maintenance/test_evidence_contract.py

# The runner rejects every nonempty credential-like parent variable. Remove the
# Git command-injection triple before evidence dispatch; do not exempt `KEY`.
unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
set +e
env -i \
  PATH="$PATH" \
  HOME="$HOME" \
  TMPDIR="${TMPDIR:-/tmp}" \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  TZ=UTC \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_OPTIONAL_LOCKS=0 \
  UV_OFFLINE=1 \
  PIP_NO_INDEX=1 \
  PYTHONHASHSEED=0 \
  VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
  KILROY_RUN_ID="$KILROY_RUN_ID" \
  uv run scripts/run_maintenance_evidence.py \
  --repo-root "$REPO_ROOT" \
  --scenario IT-13 \
  --surface non_ui \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --upstream-ref "$UPSTREAM_SHA" \
  --runner-id "$RUNNER_ID" \
  --timeout-seconds 300 \
  --lock-timeout-seconds 10 \
  --required-artifact junit=junit.xml \
  --record-env PYTHONHASHSEED \
  --gap-note "Full IT-13 coverage, snapshot, warning, ratchet, and cross-scenario reproducibility gates are not run by I00-P01." \
  --gap-note "The frozen-candidate verifier report is not attached by I00-P01." \
  --normalize-output \
  -- \
  uv run pytest -n0 tests/maintenance/test_evidence_contract.py \
    --junitxml "$EVIDENCE/IT-13/junit.xml"
IT13_EXIT=$?
set -e
test "$IT13_EXIT" -eq 1
RESULT_PATH="$EVIDENCE/IT-13/result.json" \
GAP_PATH="$EVIDENCE/IT-13/gap.json" \
uv run python -c 'import json, os; from pathlib import Path; from vibe.core.utils.io import read_safe; result = json.loads(read_safe(Path(os.environ["RESULT_PATH"])).text); gap = json.loads(read_safe(Path(os.environ["GAP_PATH"])).text); assert result["command_started"] is True; assert result["child_exit_code"] == 0; assert result["status"] == "fail"; assert len(result["gap_notes"]) == 2; assert gap["notes"] == result["gap_notes"]'

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
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
- I00-P01-AC01 through AC19 with direct evidence paths, plus the explicitly
  non-final roadmap contribution status.
- Controlled-failure, dirty-rejection, normalization, digest, duplicate, and
  concurrency results.
- Product message, snapshot, production performance, fork metric, dependency,
  and suppression deltas, each explicitly `none` unless a blocker was raised.
- Any skip, denial, flake, missing artifact, or unresolved finding.
- Clean frozen status and confirmation that no push, merge, landing, or status
  completion action was taken.
