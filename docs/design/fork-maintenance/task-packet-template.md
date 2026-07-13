# Fork Maintenance Task-Packet Template

Copy this file to `packets/I<iteration>-P<sequence>-<slug>.md`, replace every
angle-bracket placeholder, remove instructional text, and keep the section
order. A packet remains `draft` while any required field is `null`, empty,
ambiguous, or unresolved.

```yaml
---
packet_schema: 1
id: I<NN>-P<NN>
title: <bounded outcome>
iteration: <number or named sub-iteration>
state: draft
change_class: <characterization|tooling|mechanical|compatibility|optimization|documentation>
risk: <low|medium|high>
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
  scenarios: []
acceptance_criteria: []
messages: []
paths:
  allowed: []
  forbidden: []
---
```

# <Packet ID>: <Title>

Status: `draft`

## Outcome

State one externally verifiable result. Do not describe a broad aspiration or
combine multiple rollback boundaries.

## Why this packet exists

Tie the result to a roadmap risk, dependency, or preservation contract. Explain
why it is ordered here and what later work it unblocks.

## Definition of Ready

The campaign lead checks every item before changing the packet to `ready`:

- [ ] `baseline_sha` and `upstream_sha` are full, resolvable 40-character commits.
- [ ] `owner`, `reviewer`, `verifier`, `worktree`, `branch`, and
      `execution_profile` are assigned.
- [ ] The assigned worktree is clean and contains no unrelated work.
- [ ] Every dependency is `complete` in `status.yaml`.
- [ ] `VIBE_EVIDENCE_WORKSPACE` is absolute and outside the repository, Git
      common directory, and every linked worktree.
- [ ] `KILROY_RUN_ID` is unique and the scenario directories are not shared by
      another active writer.
- [ ] Every allowed path is sufficient, every forbidden path is explicit, and
      no path overlaps another active packet.
- [ ] Every command and referenced fixture exists, except files this packet is
      explicitly responsible for creating.
- [ ] Lead-only decisions below are resolved; no compatibility, message,
      snapshot, performance, architecture, or baseline decision is delegated.
- [ ] Rollback removes only this packet's changes.

If any item is false, the packet stays `draft` and no implementation begins.

## Frozen lead decisions

List exact decisions the worker must apply without reinterpretation:

- Architecture and dependency direction: <decision>.
- Observable behavior: <unchanged, or exact intentional delta>.
- Message/snapshot policy: <frozen groups or approved delta record>.
- Performance policy: <unaffected, exact invariants, or exact paired workload>.
- Upstream seam policy: <no upstream edits, or named paths and localized hooks>.
- Network/spend policy: <none, or provider and hard cap>.
- Landing policy: <normally worker cannot land>.

## Worker discretion

List the small local decisions the worker may make. Anything not listed is not
automatically permitted; consult the authority matrix and stop when in doubt.

## Scope

### In scope

- <one observable deliverable or named file/symbol>

### Out of scope

- <adjacent behavior or cleanup that must not be folded in>

## Allowed paths

The worker may modify only:

- `<path>` — <why and expected kind of edit>.

Generated evidence is written only below the assigned external evidence root.
It is not an allowed repository path.

## Forbidden paths and actions

- `<path or glob>` — <reason>.
- No upstream-owned deletion, rename, split, relocation, reordering, or broad
  formatting.
- No snapshot, baseline, threshold, suppression, lockfile, or unrelated doc
  update unless explicitly listed in `allowed_paths`.
- No commit, push, PR, merge, or landing action unless explicitly authorized.

## Required reading and inputs

Read these before editing:

- `AGENTS.md`
- `openwiki/quickstart.md`
- `<roadmap section>`
- `<source/test files that establish the real contract>`

Required runtime inputs:

```bash
BASELINE_SHA=<40-hex>
UPSTREAM_SHA=<40-hex>
VIBE_EVIDENCE_WORKSPACE=<absolute external directory>
KILROY_RUN_ID=<unique run identifier>
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
```

## Preflight

Run exactly, without editing first:

```bash
uv run git status --short
uv run git rev-parse HEAD
uv run git rev-parse "$BASELINE_SHA^{commit}"
uv run git rev-parse "$UPSTREAM_SHA^{commit}"
uv run git worktree list --porcelain
```

Define the required empty status and SHA relationships. Add packet-specific
checks for fixtures, tools, network isolation, evidence-root exclusion, or
dependencies. State the exact blocker behavior for any mismatch.

## Implementation procedure

Number every action. Each step names:

1. The file or command.
2. The intended change or observation.
3. The immediate assertion that proves the step succeeded.
4. The evidence artifact written for the step.

Do not use phrases such as “refactor as needed,” “add suitable tests,” “clean
up,” or “ensure quality.” Resolve those choices here.

## Behavioral and structural invariants

- <public/entry-point behavior that remains exact>.
- <message, event, request, ordering, persistence, or side-effect invariant>.
- <fork path/hunk invariant>.
- <performance invariant or explicit statement that production hot paths are
  untouched>.

## User-facing and model-visible messages

| Message ID | Trigger | Expected contract | Allowed normalization | Evidence |
|---|---|---|---|---|
| `<MSG-ID or packet-local ID>` | <condition> | <literal/semantic/exit contract> | <roadmap-approved volatile fields only> | `<path>` |

Write “None; no reachable message surface” only when the packet cannot execute
or change a user/model-facing path. Unexpected changes block the packet; the
worker does not update fixtures to accept them.

## Acceptance criteria

Each criterion is singular, observable, and binary.

| ID | Criterion | Proof |
|---|---|---|
| `<packet>-AC1` | <one behavior or artifact> | <test/scenario/artifact> |

Include every applicable roadmap AC. Packet-local criteria may strengthen but
must not replace them.

## Integration scenarios

For each scenario include all of:

### <IT/SR ID>: <name>

- Starting state: <clean deterministic state>.
- Actions: <ordered real entry-point actions>.
- Expected outcome: <observable result and status>.
- Failure evidence: <what remains when the action fails>.
- Artifacts: `<exact external paths>`.
- Covers: `<AC IDs and MSG IDs>`.

UI or mixed scenarios require screenshots of key states in addition to
transcripts/snapshots. Non-UI scenarios require readable logs or structured
reports.

## Acceptance-to-scenario map

| Requirement | Scenario or review |
|---|---|
| `<AC/MSG ID>` | `<IT/SR ID>` |

No acceptance criterion or reachable message group may be unmapped.

## Evidence contract

Specify:

- Exact scenario directory: `$EVIDENCE/<IT-ID>/`.
- Required filenames and media/types.
- Manifest fields, including command, exit code, artifact digest, notes, and
  `pass`/`fail` status.
- Best-effort failure artifacts and explicit missing-artifact notes.
- Candidate identity and cleanliness proof.
- Assertion that evidence writes leave candidate `git status --short` unchanged.

## Exact verification commands

Separate commands into:

1. Targeted implementation checks.
2. Required scenario command.
3. Repository quality/fork gates proportional to the packet.
4. Final diff and cleanliness checks.

Use `uv run` and preserve required options such as `-n0`. Do not use `--fix`
after candidate freeze.

## Stop conditions

Include all general authority-matrix stops plus packet-specific examples. State
which AC, IT, MSG, path, or invariant is at risk and the smallest lead decision
that could unblock it.

## Rollback

Name every repository path created or modified and the evidence paths that may
be retained. Rollback must not revert other packets, regenerate baselines, or
alter unrelated work.

## Completion report

The worker reports:

- Packet ID and candidate SHA.
- Exact changed paths and why each was allowed.
- Commands and exit codes.
- Acceptance-criterion table with pass/fail and evidence links.
- Message/snapshot/performance/fork-metric deltas, including explicit “none.”
- Evidence manifest and scenario paths.
- Remaining gaps, skips, flakes, denied tools, or blockers.
- Confirmation that candidate state is clean and frozen.
- Confirmation that no landing action was taken.

The lead, not the worker, records final packet state.
