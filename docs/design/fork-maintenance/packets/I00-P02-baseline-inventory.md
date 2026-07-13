---
packet_schema: 1
id: I00-P02
title: Baseline identity and fork inventory
iteration: 0
state: draft
change_class: tooling
risk: medium
owner: null
reviewer: null
verifier: null
evidence_operator: null
depends_on:
  - I00-P01
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
    - IT-12
packet_acceptance_criteria:
  - I00-P02-AC01
  - I00-P02-AC02
  - I00-P02-AC03
  - I00-P02-AC04
  - I00-P02-AC05
  - I00-P02-AC06
  - I00-P02-AC07
  - I00-P02-AC08
  - I00-P02-AC09
  - I00-P02-AC10
  - I00-P02-AC11
  - I00-P02-AC12
roadmap_contributions:
  - AC-1.1
  - AC-1.2
  - AC-2.4
  - AC-7.1
messages:
  - I00-P02-MSG-01
paths:
  allowed:
    - scripts/report_fork_baseline.py
    - tests/maintenance/test_fork_baseline_report.py
  forbidden:
    - vibe/**
    - scripts/check_upstream_divergence.py
    - tests/test_upstream_divergence.py
    - tests/test_iron_laws.py
    - tests/snapshots/**
    - pyproject.toml
    - uv.lock
    - docs/**
---

# I00-P02: Baseline Identity and Fork Inventory

Execution state is read only from this packet's frontmatter and the matching
`../status.yaml` entry at the assigned clean `CONTROL_SHA`; this prose does not
duplicate state. Implementation is authorized only when both record `ready`,
I00-P01 is `complete`, and the lead has filled every required field.

## Outcome

Add a deterministic report command that compares the frozen packet candidate to
the frozen upstream tree and writes machine-readable repository identity, path
ownership, changed-path, and per-hotspot hunk metrics. Capture those reports
under IT-12 while honestly recording the known Iteration 1 divergence-guard and
merge-rehearsal gaps as `status: fail`.

The result is a provisional, reproducible before-state. The lead reruns the same
report after all Iteration 0 packets are consolidated and freezes that final
result as the campaign before-state for structural work. This packet does not
repair the divergence guard, change an accepted divergence, or claim that full
IT-12 behavior already passes.

## Why this packet exists

Later iterations cannot prove that they reduced merge cost unless the campaign
first records exact upstream-owned paths, fork-added paths, missing upstream
paths, modified upstream paths, and hotspot hunk counts from a clean commit.
The roadmap has aggregate audit numbers, but those numbers are prose from an
earlier snapshot rather than a reproducible artifact tied to the execution
baseline.

I00-P01 is required because the inventory and its known gaps must be written to
the canonical external evidence manifest without dirtying the candidate.

## Definition of Ready

The lead checks every item before setting `ready`:

- [ ] I00-P01 is `complete`, its evidence runner exists, and its contract tests
      pass at the assigned baseline.
- [ ] `baseline_sha` is the exact clean packet-start commit and `upstream_sha`
      is the exact upstream comparison tree. The campaign preservation baseline
      is intentionally still pending Iteration 0 consolidation.
- [ ] Both commits are locally available with full history.
- [ ] The `upstream` remote is not fetched, advanced, or synced during this
      packet; refs are immutable SHAs.
- [ ] Owner, reviewer, verifier, evidence operator, isolated worktree, branch,
      execution profile, evidence workspace, run ID, and stable runner ID are
      assigned.
- [ ] No other active packet modifies either allowed path or writes IT-12.
- [ ] The lead confirms the two initial hotspots:
      `vibe/core/agent_loop.py` and `vibe/cli/textual_ui/app.py`.
- [ ] The lead confirms that IT-12 must be recorded `fail` for the three explicit
      gaps below; the worker must not reinterpret a current guard success as a
      full scenario pass.

## Frozen lead decisions

- Architecture: one fork-added reporting script plus one hermetic test file. Do
  not alter or wrap the existing divergence guard.
- Comparison direction: `upstream_sha` is the reference tree and the packet's
  frozen `candidate_sha` is the provisional fork tree. The same command is
  rerun later with the consolidated campaign baseline as the fork tree.
- Rename handling: inventory uses exact tree path membership and `--no-renames`.
  A same-content rename is an upstream path absent plus a fork-added path.
- Scope: inventory every tracked path, with explicit production (`vibe/`), test
  (`tests/`), and script (`scripts/`) subsets. Do not limit to Python.
- Hotspots: report `vibe/core/agent_loop.py` and
  `vibe/cli/textual_ui/app.py` in the initial invocation. Later packets may add
  hotspots through the repeatable CLI flag without changing schema.
- Output: deterministic schema-v1 JSON with no timestamps, host paths, branch
  names, abbreviated SHAs, or unordered sets.
- Known IT-12 gaps: exact rename/copy-delete enforcement, guard coverage of all
  configured upstream-owned production/test/script paths, and automated
  disposable upstream merge rehearsal are not implemented here. They are
  written through I00-P01 `--gap-note`, so IT-12 is `fail` by design.
- Behavior: no product, guard, snapshot, config, prompt, provider, cost, or
  performance behavior changes.
- Network/spend: no fetch, live network, provider credential, or paid model.
- Landing: the worker may not land, push, merge, or mark complete.
- Baseline promotion: the packet candidate is not the final campaign baseline.
  Only the lead may consolidate Iteration 0 and rerun this report with manifest
  `baseline_sha == candidate_sha`.

## Worker discretion

The worker may choose private function/dataclass names and test helper names.
The worker may factor Git output parsing into private functions inside the one
script. The JSON field names and comparison semantics below are frozen.

## Scope

### In scope

- Create `scripts/report_fork_baseline.py`.
- Create `tests/maintenance/test_fork_baseline_report.py`.
- Generate `repos.json`, `fork-metrics.json`, and `divergence.json` from clean,
  frozen commits.
- Capture existing guard output and limitations without changing the guard.
- Record the three known IT-12 gaps through the evidence runner.
- Add an opt-in test-local artifact hook in the allowed test file: when all
  `VIBE_MAINTENANCE_FORK_REPORT_*` variables below are present, the test invokes
  the reporter against the frozen candidate and writes the three IT-12 reports.
  With none present, the normal temporary-repository matrix runs without
  campaign evidence writes; a partial variable set fails.

### Out of scope

- Fixing rename/copy-delete detection or expanding the enforcement guard.
- Adding/removing accepted divergence paths or reasons.
- Performing or automating an upstream merge rehearsal.
- Changing upstream-owned files, Git history, remotes, refs, worktrees, tags, or
  the roadmap's audit prose.
- Measuring runtime performance or code complexity.
- Inferring that two differently named paths are equivalent.

## Allowed paths

- `scripts/report_fork_baseline.py` — new deterministic inventory CLI.
- `tests/maintenance/test_fork_baseline_report.py` — new temporary-repository
  contract tests.

## Forbidden paths and actions

- `scripts/check_upstream_divergence.py`, `tests/test_upstream_divergence.py`,
  and `tests/test_iron_laws.py` — characterize, do not repair or weaken.
- `vibe/**`, `tests/snapshots/**`, `docs/**`, `pyproject.toml`, and `uv.lock`.
- No fetch, pull, merge, rebase, reset, checkout restoration, tag, commit amend,
  force operation, branch deletion, or worktree deletion.
- No raw working-tree file reads to compare committed refs. Read tree identity
  and blobs through Git so the report is commit-addressed.
- No rename similarity inference. Always use exact path identity and
  `--no-renames` where diff output is involved.

## Required reading and inputs

Read before editing:

- `AGENTS.md`
- `openwiki/quickstart.md`
- Roadmap definitions: “Upstream-owned path,” “Iteration 0,” “Test evidence
  contract,” “IT-12,” “Crosscheck,” and “Prohibited sequencing.”
- `scripts/check_upstream_divergence.py`
- `tests/test_upstream_divergence.py`
- `tests/test_iron_laws.py`
- I00-P01 and the implemented evidence-runner help output.

Lead-filled inputs:

```bash
CONTROL_WORKTREE=<absolute clean control worktree>
CONTROL_SHA=<immutable control commit supplied by the lead>
BASELINE_SHA=<frontmatter baseline_sha>
UPSTREAM_SHA=<frontmatter upstream_sha>
VIBE_EVIDENCE_WORKSPACE=<frontmatter evidence.workspace>
KILROY_RUN_ID=<frontmatter evidence.run_id>
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
RUNNER_ID=<assigned stable runner label>
REPO_ROOT=<absolute candidate worktree root>
```

## Preflight

Run without editing:

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
test "$(GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_OPTIONAL_LOCKS=0 uv run git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)" = ""
test "$(uv run git rev-parse --show-toplevel)" = "$REPO_ROOT"
uv run git rev-parse HEAD
uv run git rev-parse "$BASELINE_SHA^{commit}"
uv run git rev-parse "$UPSTREAM_SHA^{commit}"
uv run git merge-base "$BASELINE_SHA" "$UPSTREAM_SHA"
uv run git cat-file -e "$BASELINE_SHA:uv.lock"
uv run git cat-file -e "$UPSTREAM_SHA^{tree}"
uv run git worktree list --porcelain
test -f scripts/run_maintenance_evidence.py
test -f tests/maintenance/test_evidence_contract.py
test ! -e scripts/report_fork_baseline.py
test ! -e tests/maintenance/test_fork_baseline_report.py
```

Required results:

- Status is empty and `HEAD` equals `BASELINE_SHA` at packet start.
- Both full refs, their merge base, the baseline `uv.lock`, and the upstream tree
  are available without fetching.
- I00-P01 deliverables exist and pass their targeted tests.
- Both new deliverable paths are unused.
- Evidence root is external and IT-12 is unclaimed by another writer.

Any mismatch requests `blocked` with no edits.

Run the canonical control-metadata validator in
`../task-packet-template.md` with `PACKET_ID=I00-P02` and
`PACKET_RELATIVE_PATH=docs/design/fork-maintenance/packets/I00-P02-baseline-inventory.md`.
An assertion failure requests `ready -> blocked` with no edits.

## Reporting command contract

The script supports exactly:

```bash
VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
KILROY_RUN_ID="$KILROY_RUN_ID" \
VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" \
VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
uv run scripts/report_fork_baseline.py \
  --repo-root "$REPO_ROOT" \
  --fork-ref "$CANDIDATE_SHA" \
  --upstream-ref "$UPSTREAM_SHA" \
  --output-dir "$EVIDENCE/IT-12" \
  --hotspot vibe/core/agent_loop.py \
  --hotspot vibe/cli/textual_ui/app.py
```

Arguments:

- `--repo-root`: required absolute Git worktree. This explicit seam is used by
  temporary-repository tests; it is never inferred from the script location.
- `--fork-ref`: required commit; must resolve to current clean `HEAD`.
- `--upstream-ref`: required commit; need not be an ancestor, but a merge base
  must exist.
- `--output-dir`: required absolute external directory. It must equal the
  resolved `$EVIDENCE/IT-12` derived from the two environment variables; reject
  any other directory.
- `--hotspot`: repeatable repository-relative exact path. Reject absolute paths,
  traversal, duplicates, or a path absent from both trees.

Success writes all three files and exits 0. Invalid refs, dirty/moved fork HEAD,
bad output path, Git failure, malformed Git output, or write failure exits 2
with a concise diagnostic and no traceback. The reporter has no “partial pass.”
It never invokes a shell and never fetches.

## JSON contracts

All JSON is UTF-8, sorted by key, indented, newline-terminated, and written with
repository safe/durable I/O. Every path list is lexicographically sorted. Counts
are derived from the emitted lists rather than separately guessed.

### `repos.json`

```json
{
  "version": 1,
  "fork_sha": "<40-hex>",
  "upstream_sha": "<40-hex>",
  "merge_base_sha": "<40-hex>",
  "fork_is_clean_head": true,
  "uv_lock_sha256": "<digest of FORK_REF:uv.lock bytes>",
  "git_version": "<git --version output>",
  "comparison": "exact-path-membership-no-renames"
}
```

### `fork-metrics.json`

```json
{
  "version": 1,
  "fork_sha": "<40-hex>",
  "upstream_sha": "<40-hex>",
  "paths": {
    "upstream_all": ["<every path in upstream tree>"],
    "fork_all": ["<every path in fork tree>"],
    "upstream_owned_present": ["<intersection>"],
    "fork_added": ["<fork minus upstream>"],
    "absent_upstream": ["<upstream minus fork>"],
    "modified_upstream": ["<same path, different blob/mode/type>"],
    "modified_upstream_python": ["<modified upstream paths ending .py>"],
    "modified_upstream_production": ["<modified upstream paths under vibe/>"],
    "modified_upstream_tests": ["<modified upstream paths under tests/>"],
    "modified_upstream_scripts": ["<modified upstream paths under scripts/>"]
  },
  "counts": {
    "upstream_all": 0,
    "fork_all": 0,
    "upstream_owned_present": 0,
    "fork_added": 0,
    "absent_upstream": 0,
    "modified_upstream": 0,
    "modified_upstream_python": 0,
    "modified_upstream_production": 0,
    "modified_upstream_tests": 0,
    "modified_upstream_scripts": 0
  },
  "diff": {
    "added_paths": 0,
    "deleted_paths": 0,
    "modified_paths": 0,
    "total_additions": 0,
    "total_deletions": 0
  },
  "hotspots": [
    {
      "path": "vibe/core/agent_loop.py",
      "upstream_present": true,
      "fork_present": true,
      "changed": true,
      "hunks": 0,
      "additions": 0,
      "deletions": 0
    }
  ]
}
```

Tree membership and modification compare exact Git tree entries, including
mode/type/blob identity. Added/deleted/modified path counts come from those exact
tree collections. Textual line totals come from NUL-safe
`git diff --no-renames --numstat -z`; binary entries are counted as changed paths
but contribute zero textual lines and are named in a `binary_paths` list added
under `diff`. Hotspot hunk count is the number of `@@` headers from
`git diff --no-renames --unified=0` for that exact path.

### `divergence.json`

This is characterization of the current guard, not a replacement result:

```json
{
  "version": 1,
  "guard_path": "scripts/check_upstream_divergence.py",
  "guard_baseline_sha": "<resolved baseline used by guard>",
  "guard_upstream_ref_sha": "<same frozen upstream SHA>",
  "baseline_available": true,
  "upstream_unsynced_count": 0,
  "guard_exit_code": 0,
  "accepted_paths": [],
  "unexpected_paths": [],
  "limitations": [
    "deletion diff is rename-similarity-sensitive",
    "scope is limited to vibe/**/*.py",
    "disposable upstream merge rehearsal is not implemented"
  ]
}
```

Use the existing guard's public root-parameterized report functions with
`VIBE_UPSTREAM_BASE=$UPSTREAM_SHA` and `VIBE_UPSTREAM_REF=$UPSTREAM_SHA`. Do not
import its private constants or call its fixed script-root `main()` from a
temporary repository. A nonzero derived guard result is recorded accurately and
causes the reporter to exit 1 after writing all three files; it is never coerced
to success.

## Implementation procedure

1. Create the reporter with stdlib argument parsing and a `main() -> int` entry
   point. Use argument-vector Git subprocesses with `shell=False` and the
   required explicit repository root.
2. Resolve full fork/upstream/merge-base SHAs, require clean current HEAD equal
   to fork SHA, and validate the exact external scenario output directory.
3. Read both complete tree inventories using a NUL-safe Git format that retains
   path and tree-entry identity. Use `--literal-pathspecs`, `--no-ext-diff`,
   `--no-textconv`, `--no-renames`, `--diff-algorithm=myers`, no color, and a
   subprocess environment with `GIT_CONFIG_NOSYSTEM=1`,
   `GIT_CONFIG_GLOBAL=/dev/null`, and `GIT_LITERAL_PATHSPECS=1`. Parse byte
   output by NUL boundaries and round-trip paths with `os.fsdecode`/`os.fsencode`;
   do not parse human-formatted status output.
4. Build exact sorted set/intersection/difference path collections and compare
   tree entries to identify modifications.
5. Parse NUL-safe `--no-renames --numstat -z` for overall textual/binary line
   metrics. Derive path-status counts from exact tree sets. Count `@@` headers
   and numstat values for each named hotspot.
6. Hash committed `FORK_REF:uv.lock` bytes, not the working-tree file.
7. Run the unchanged divergence guard public functions with a separate sanitized
   Git environment: `GIT_CONFIG_NOSYSTEM=1`,
   `GIT_CONFIG_GLOBAL=/dev/null`, `GIT_OPTIONAL_LOCKS=0`, no
   `GIT_LITERAL_PATHSPECS`, and Git config env entries pinning
   `diff.renames=true`. The missing literal flag preserves the guard's
   `vibe/**/*.py` wildcard; pinned rename detection makes its known
   similarity-sensitive limitation deterministic. Capture its result/limitations.
8. Write the three deterministic JSON files with project safe/durable I/O.
9. Add temporary-repository tests that create upstream/fork commits covering
   unchanged, modified, added, deleted, exact rename, copy-delete, binary,
   executable-mode, Unicode/space-containing path, and hotspot cases.
   Configure deterministic repository-local test author name/email and commit
   timestamps; never depend on user/global Git identity.
10. Prove repeatability by running the reporter twice into separate external
    directories and comparing output bytes.
11. In the test file, implement the exact opt-in artifact hook. It reads only
  `VIBE_MAINTENANCE_FORK_REPORT_ROOT`,
  `VIBE_MAINTENANCE_FORK_REPORT_DIR`,
    `VIBE_MAINTENANCE_FORK_REPORT_REF`, and
    `VIBE_MAINTENANCE_FORK_REPORT_UPSTREAM`; all-or-none is mandatory. It invokes
    the real reporter CLI as a subprocess against `REPO_ROOT`, checks its result,
    and lets pytest create JUnit in the same recorded child command. The evidence
    pytest invocation uses `-s`, so the reporter's frozen stdout/stderr reaches
    the outer runner artifacts. The subprocess uses the three frozen ref/output
    values and exactly two `--hotspot` arguments:
    `vibe/core/agent_loop.py` and `vibe/cli/textual_ui/app.py`.

## Required contract tests

Individual tests prove:

1. Exact path sets and all derived counts match a known temporary repository.
2. An exact rename is represented as one absent upstream path and one fork-added
   path regardless of content similarity.
3. A copy-delete retains the copy as fork-added and the original as absent.
4. Same-path content, mode, and Git object-type changes are modified upstream;
   unchanged entries are not.
5. Binary paths are named and do not fabricate line counts.
6. Space, Unicode, tab, and newline-capable Git paths are parsed without
   splitting or loss by the chosen NUL-safe format.
7. Hotspot hunk/addition/deletion counts match a fixed multi-hunk diff; absent
   and unchanged hotspots report explicit booleans and zeros.
8. `uv.lock` digest comes from the frozen fork commit even if the working-tree
   file would differ; a dirty worktree is rejected before reporting.
9. Relative, in-repository, wrong-scenario, symlink-escape, or unassigned output
   paths are rejected.
10. Existing guard baseline availability, frozen upstream count, derived exit
    code, public accepted/unexpected path lists, and fixed limitations are
    recorded without modifying the guard.
11. Two identical invocations produce byte-identical JSON.
12. Git failures and malformed output return 2 with no traceback and never
    produce a misleading success report.

If the existing guard exposes insufficient public data for accepted/unexpected
lists, record the exit/stdout and leave those two arrays absent rather than
importing private constants or functions. Stop if the lead requires new guard
API; changing it is outside this packet.

## Behavioral and structural invariants

- The report is commit-addressed and independent of branch movement after refs
  resolve.
- Exact upstream path identity, not rename similarity, determines ownership.
- All tracked paths are inventoried; production/test/script subsets are explicit.
- The current divergence guard remains byte-for-byte unchanged and its known
  limitations remain visible.
- No Git worktree, ref, index, commit, object reachability, remote, or config is
  mutated by the reporter.
- No product code, snapshot, user/model message, provider request, prompt, spend,
  harness workload, or performance threshold changes.
- Running the report and evidence capture leaves candidate status empty.

## User-facing and model-visible messages

The reporter is developer tooling. It has one local CLI diagnostic contract and
no reachable product/model surface.

| Message ID | Trigger | Expected contract | Allowed normalization | Evidence |
|---|---|---|---|---|
| I00-P02-MSG-01 | Report success, guard finding, or reporter error | Exit 0 stdout: `PASS fork-report <fork-sha> <upstream-sha> <absolute-output-dir>\n`; exit 1 stdout: `FAIL fork-report <fork-sha> <upstream-sha> <absolute-output-dir>\n`; exit 2 stderr: `ERROR fork-report <category>: <detail>\nRecovery: <action>\n`; the other stream is empty and expected errors have no traceback | Full worktree/evidence paths only; never status, category, detail, or recovery | `$EVIDENCE/IT-12/{stdout.txt,stderr.txt,divergence.json}` |

Exit-2 categories are exactly `arguments`, `platform`, `repository`, `workspace`,
`git`, or `write`, with these frozen detail/recovery templates:

- `arguments`: `invalid <field>: <safe-value>` / `correct <field> and retry`.
- `platform`: `unsupported platform: <platform>` / `run this packet on Linux`.
- `repository`: `<check> failed for <repo-root>: <safe-value>` /
  `restore the frozen clean candidate and retry`.
- `workspace`: `<check> failed: <path>` /
  `use the assigned external IT-12 directory`.
- `git`: `<operation> failed with exit <code>` /
  `repair local full-history repository state and retry`.
- `write`: `<relative-file> failed with <error-class>` /
  `restore the assigned external evidence directory and retry`.

Tests pin pass, guard-fail, and one exact representative per category. The
worker may substitute only bracketed validated values and may not invent a new
prefix, stream, category, detail grammar, recovery action, or layout.

## Acceptance criteria

| ID | Criterion | Proof |
|---|---|---|
| I00-P02-AC01 | The `repos.json` identity projection equals the fixture's full fork, upstream, and merge-base SHAs. | Identity fixture and IT-12 artifact |
| I00-P02-AC02 | `repos.json` lock digest equals committed `FORK_REF:uv.lock` bytes. | Lock-digest fixture |
| I00-P02-AC03 | Every exact upstream/fork path appears in the correct sorted ownership set. | Temporary-repository path matrix |
| I00-P02-AC04 | Every emitted ownership count equals the length of its corresponding list. | Schema consistency test |
| I00-P02-AC05 | NUL-safe no-renames line/binary metrics match the fixed diff fixture. | Diff-metric test |
| I00-P02-AC06 | The complete sorted `hotspots` array equals the two frozen per-path presence/hunk/addition/deletion objects. | Multi-hunk fixture |
| I00-P02-AC07 | The classification projection for rename, copy-delete, mode, binary, and unusual-path fixtures equals the frozen exact-path matrix. | Classification matrix |
| I00-P02-AC08 | `divergence.json` records the unchanged guard result against the frozen refs. | Guard characterization test |
| I00-P02-AC09 | The IT-12 gap projection equals `{status: fail, child_exit_code: 0, gap_notes: <three approved strings>}`. | Manifest, result, and gap assertions |
| I00-P02-AC10 | Two identical invocations emit byte-identical report JSON. | Repeatability test |
| I00-P02-AC11 | The candidate fingerprint tuple `(HEAD, porcelain-v1)` after evidence equals its pre-evidence tuple. | Pre/post fingerprint |
| I00-P02-AC12 | The packet quality projection equals the frozen command/exit map plus changed-path set `{scripts/report_fork_baseline.py, tests/maintenance/test_fork_baseline_report.py}`. | Command log and name-only diff |

These criteria contribute baseline identity/metrics to roadmap AC-1.1, AC-1.2,
AC-2.4, and AC-7.1. They do not complete those campaign criteria. IT-12 remains
`fail`; AC-2.1 through AC-2.3 remain Iteration 1 work, and I00-P99 owns the final
campaign verdict.

## Integration scenario

### IT-12: Baseline inventory and explicit gap slice

- Starting state: clean frozen packet candidate and upstream commits with full
  history; I00-P01 complete; deterministic reporter tests; no fetch/network.
- Actions:
  1. Run the reporter against frozen candidate/upstream SHAs with both hotspots.
  2. Run current divergence guard/tests unchanged.
  3. Execute the temporary-repository report matrix.
  4. Record three gap notes for enforcement rename/copy-delete behavior, full
     configured production/test/script guard scope, and disposable merge
     rehearsal.
- Expected outcome: inventory and current guard characterization are readable
  and accurate; the scenario manifest is deliberately `fail` until Iteration 1
  closes the named gaps; evidence runner returns 1 for the gap, not 2.
- Failure evidence: runner raw outputs/result, reporter files when available,
  `gap.json`, manifest notes, and JUnit.
- Artifacts:
  `$EVIDENCE/IT-12/{repos.json,fork-metrics.json,divergence.json,gap.json,command.json,command.log,stdout.txt,stderr.txt,result.json,junit.xml}`.
- Covers: I00-P02-AC01 through I00-P02-AC11; I00-P02-MSG-01.
- Packet quality/diff gates cover: I00-P02-AC12.
- Contributes to: AC-1.1, AC-1.2, AC-2.4 baseline, AC-7.1.

## Acceptance-to-scenario map

| Requirement | Scenario/review |
|---|---|
| I00-P02-AC01 through I00-P02-AC11 | IT-12 baseline/gap slice |
| I00-P02-AC12 | Targeted quality/fork commands and diff review |
| AC-1.1, AC-1.2 | Contribution only; final verdict belongs to I00-P99 |
| AC-2.4 baseline | Metric contribution only; no non-increase verdict yet |
| AC-7.1 | Contribution only; final verdict belongs to I00-P99 |
| I00-P02-MSG-01 | IT-12 raw output and divergence artifact |

## Exact verification commands

Before freeze, run:

```bash
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_OPTIONAL_LOCKS=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run ruff check --fix scripts/report_fork_baseline.py tests/maintenance/test_fork_baseline_report.py
uv run ruff format scripts/report_fork_baseline.py tests/maintenance/test_fork_baseline_report.py
uv run pytest -n0 \
  tests/maintenance/test_fork_baseline_report.py \
  tests/maintenance/test_evidence_contract.py
uv run pyright \
  scripts/report_fork_baseline.py \
  tests/maintenance/test_fork_baseline_report.py
uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
  uv run scripts/check_upstream_divergence.py
uv run pre-commit run --all-files
uv run git diff --check
uv run git status --short
```

Stop at the freeze handoff. The lead reviews and creates the candidate commit,
writes `candidate_sha` to a new clean control commit, and assigns the evidence
operator. The worker does not commit or edit control files. Require clean
candidate status before post-freeze evidence. The evidence operator reruns the
canonical validator with `CONTROL_SHA` set to the newly assigned clean
verification-state control commit, `EXPECTED_PACKET_STATE=verification`, and
`EXPECTED_CANDIDATE_SHA="$CANDIDATE_SHA"`.

```bash
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_OPTIONAL_LOCKS=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run ruff check scripts/report_fork_baseline.py tests/maintenance/test_fork_baseline_report.py
uv run ruff format --check scripts/report_fork_baseline.py tests/maintenance/test_fork_baseline_report.py
uv run pyright scripts/report_fork_baseline.py tests/maintenance/test_fork_baseline_report.py

# The evidence runner treats `KEY` as credential-like. Remove the Git
# command-injection triple before dispatch instead of weakening that preflight.
unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
```

Run the evidence scenario. Its recorded pytest child invokes the real reporter
through the opt-in hook, so the manifest owns the command that creates all four
required artifacts. The runner is expected to exit 1 because explicit gaps
must produce `status: fail`; exit 0 or 2 is a packet failure. The `env -i`
allowlist is mandatory and must not be replaced with the operator's inherited
environment:

```bash
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
  VIBE_MAINTENANCE_FORK_REPORT_ROOT="$REPO_ROOT" \
  VIBE_MAINTENANCE_FORK_REPORT_DIR="$EVIDENCE/IT-12" \
  VIBE_MAINTENANCE_FORK_REPORT_REF="$CANDIDATE_SHA" \
  VIBE_MAINTENANCE_FORK_REPORT_UPSTREAM="$UPSTREAM_SHA" \
  VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" \
  VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
  VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
  KILROY_RUN_ID="$KILROY_RUN_ID" \
  uv run scripts/run_maintenance_evidence.py \
  --repo-root "$REPO_ROOT" \
  --scenario IT-12 \
  --surface non_ui \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --upstream-ref "$UPSTREAM_SHA" \
  --runner-id "$RUNNER_ID" \
  --timeout-seconds 300 \
  --lock-timeout-seconds 10 \
  --required-artifact repository=repos.json \
  --required-artifact metrics=fork-metrics.json \
  --required-artifact divergence=divergence.json \
  --required-artifact junit=junit.xml \
  --record-env PYTHONHASHSEED \
  --record-env VIBE_MAINTENANCE_FORK_REPORT_REF \
  --record-env VIBE_MAINTENANCE_FORK_REPORT_UPSTREAM \
  --gap-note "Iteration 1 must enforce exact rename and copy-delete detection." \
  --gap-note "Iteration 1 must cover configured upstream-owned production, test, and script paths." \
  --gap-note "Iteration 1 must add the disposable upstream merge rehearsal." \
  --normalize-output \
  -- \
  uv run pytest -n0 -s \
    tests/maintenance/test_fork_baseline_report.py \
    tests/test_upstream_divergence.py \
    tests/test_iron_laws.py \
    --junitxml "$EVIDENCE/IT-12/junit.xml"
IT12_EXIT=$?
set -e
test "$IT12_EXIT" -eq 1
RESULT_PATH="$EVIDENCE/IT-12/result.json" \
GAP_PATH="$EVIDENCE/IT-12/gap.json" \
uv run python -c 'import json, os; from pathlib import Path; from vibe.core.utils.io import read_safe; result = json.loads(read_safe(Path(os.environ["RESULT_PATH"])).text); gap = json.loads(read_safe(Path(os.environ["GAP_PATH"])).text); assert result["command_started"] is True; assert result["child_exit_code"] == 0; assert result["status"] == "fail"; assert len(result["gap_notes"]) == 3; assert gap["notes"] == result["gap_notes"]; assert all(item["status"] == "readable" for item in result["required_artifacts"])'
```

The `set +e` block is only for asserting the documented gap exit and does not
hide any other status. Inspect `result.json` and require child exit 0, all
required artifacts readable, exactly three gap notes, and manifest `fail`.

Final checks:

```bash
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
  uv run scripts/check_upstream_divergence.py
uv run git diff "$BASELINE_SHA".."$CANDIDATE_SHA" -- \
  scripts/report_fork_baseline.py \
  tests/maintenance/test_fork_baseline_report.py
uv run git diff --name-only "$BASELINE_SHA".."$CANDIDATE_SHA"
uv run git status --short
```

The final name-only output contains exactly the two allowed paths and status is
empty. Any candidate mutation after freeze invalidates verification.

## Stop conditions

Stop and request `blocked` when:

- I00-P01 is incomplete, its manifest contract differs, or IT-12 is already
  claimed in the run.
- The frozen refs are missing, dirty, moved, shallow in a way that prevents the
  inventory, or require a fetch.
- Correct NUL-safe tree/diff parsing or safe output requires a third file or new
  dependency.
- The existing guard must be changed or a private guard API imported to satisfy
  the report.
- A report cannot classify rename/copy-delete through exact membership without
  rename similarity.
- The current guard or iron laws fail for a reason not caused by the two allowed
  files.
- IT-12 appears `pass`, uses `blocked`, omits a gap/artifact, or returns runner
  infrastructure exit 2/3.
- Generated output includes a secret, absolute worktree path, timestamp,
  nondeterministic ordering, abbreviated SHA, or branch name.
- Any forbidden/unrelated path changes or the candidate changes after freeze.

## Rollback

Remove only:

- `scripts/report_fork_baseline.py`
- `tests/maintenance/test_fork_baseline_report.py`

Retain external baseline artifacts for audit if useful, but mark them superseded
if the reporter is reverted. Do not edit the existing guard, accepted divergence,
roadmap, snapshots, thresholds, refs, or another packet.

## Completion report

Report:

- Packet and all three full SHAs; evidence root and manifest path.
- Exact two changed paths and name-only diff proof.
- Counts for upstream/fork paths, fork-added, absent upstream, modified upstream,
  modified upstream Python/production/tests/scripts, and both hotspot hunks.
- Current guard exit/accepted/unexpected summary without claiming its gaps are
  fixed.
- IT-12 `fail` status, runner exit 1, child exit 0, all three exact gap notes,
  and every artifact digest.
- Each command/exit and I00-P02 acceptance result.
- Product message, snapshot, production performance, cost, provider, prompt,
  dependency, suppression, accepted-divergence, and upstream-path deltas,
  explicitly `none`.
- Clean frozen candidate, no denied/skipped command, and confirmation that no
  push, merge, landing, or completion-state edit was performed.
