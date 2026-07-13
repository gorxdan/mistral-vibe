# Fork Maintenance Execution Guide

Status: Planning control document

This directory turns the campaign-level
[fork maintenance roadmap](../fork-maintenance-roadmap.md) into bounded work
that can be assigned to an implementation agent without giving that agent
campaign-level decision authority.

The roadmap remains the source of truth for scope, ordering, preservation
contracts, acceptance criteria, integration scenarios, evidence, and rollback.
The files here add the operational detail needed to execute it safely:

- [Campaign status](status.yaml) records the current baseline, dependencies,
  packet states, assignments, and readiness blockers.
- [Authority matrix](authority-matrix.md) states which decisions belong to the
  campaign lead, worker, evidence operator, reviewer, and verifier.
- [Task-packet template](task-packet-template.md) is the required schema for all
  implementation packets.
- [`packets/`](packets/) contains the frozen, iteration-sized assignments.

These documents do not authorize implementation by themselves. A packet is
authorized only when the campaign lead changes its state to `ready`, fills every
required identity and assignment field, and records the same state in
`status.yaml`.

## Packet baseline versus campaign baseline

Two related identities must not be conflated during Iteration 0:

- A packet's frontmatter `baseline_sha` is the clean commit from which that
  packet starts and against which its bounded diff is reviewed.
- The campaign preservation baseline is the final clean consolidated Iteration 0
  commit after evidence/characterization tooling is complete. Later structural,
  compatibility, and optimization packets compare against this commit.

Iteration 0 necessarily builds the tools used to record its own final baseline.
Therefore its early packets may have different packet baselines and candidates.
Because manifest identity includes candidate SHA, each packet also uses its own
unique evidence run ID. Packet artifacts are diagnostic/provisional and cannot
be appended to a manifest created for a different candidate. The consolidated
Iteration 0 baseline capture receives a separate final campaign run ID.
After all Iteration 0 deliverables are consolidated and green, the lead records
that commit in `status.yaml` as `campaign_baseline_sha` and performs a
baseline-only capture with manifest `baseline_sha == candidate_sha`. Provisional
inventory from I00-P02 is rerun at that final commit. No Iteration 1 structural
work begins before this bootstrap is complete.

## Control worktree and candidate worktrees

Packet/status metadata and implementation candidates live on separate branches:

- The **control worktree** is a dedicated `maintenance/fork-maintenance-control`
  branch containing `docs/design/fork-maintenance/`. The lead alone updates and
  commits packet state, assignments, baseline/candidate SHAs, and evidence IDs.
- A **candidate worktree** branches from the packet `baseline_sha`. It contains
  implementation edits only and never edits packet/status documents.

This separation avoids a self-referential SHA: a commit cannot contain its own
hash. A ready control commit may safely name the candidate's starting SHA, and a
later control commit may record the frozen candidate SHA without changing that
candidate.

Every assignment supplies `CONTROL_WORKTREE` and its immutable `CONTROL_SHA`
outside the packet file. The worker reads the packet/status from that exact
control commit and verifies the control worktree is clean. The packet copy in
the candidate branch may still show an older `draft` state and is not the
execution authority. If the control branch moves, the assignment remains bound
to `CONTROL_SHA`; a revised packet requires a new assignment.
After candidate freeze, the lead issues a second assignment whose
`CONTROL_SHA` is the new clean verification-state control commit; the evidence
operator must not reuse the ready-state control SHA.

## Precedence

When instructions disagree, apply them in this order:

1. Current user instructions and the repository `AGENTS.md`.
2. A lead-frozen task packet whose state is `ready`, `active`, or
   `verification`.
3. The campaign roadmap.
4. This execution guide and the packet template.

Stop and escalate when two applicable instructions at the same level conflict,
or when obeying a higher-level instruction would invalidate a packet contract.
The worker must not silently choose one interpretation.

## Packet lifecycle

Packets use this state machine:

```text
draft -> ready -> active -> verification -> complete
                    |             |
                    +-> blocked <-+
```

- `draft`: incomplete or not yet authorized. No implementation may begin.
- `ready`: the lead has frozen scope, SHAs, dependencies, ownership, commands,
  and evidence locations. The assigned worker may begin.
- `active`: the worker has passed preflight and is making the allowed changes.
- `verification`: implementation is frozen. Candidate and repository access is
  read-only. The assigned evidence operator may run the one frozen scenario and
  write only its assigned external evidence artifacts; the reviewer and
  verifier remain read-only.
- `complete`: the lead accepted the evidence and verifier verdict.
- `blocked`: work stopped without widening scope or changing contracts.

Only the campaign lead may move a packet from `draft` to `ready`, from `ready`
to `blocked`, from `blocked` back to an executable state, or from
`verification` to `complete`.
The assigned worker may report `active`, `verification`, or `blocked`, but does
not edit `status.yaml`; the lead records all state changes.

## Iteration 0 dependency graph

The first three packets deliberately establish infrastructure before product
characterization:

```text
I00-P01 Evidence runner and contract
        |
        +----> I00-P02 Baseline identity and fork inventory
        |
        +----> I00-P04 Backend error boundary (required, not yet authored)
                       |
                       +--> I00-P03 Programmatic CLI characterization
```

`I00-P02` may run after I00-P01. `I00-P03` additionally waits for I00-P04 because
static inspection found an uncaught typed backend error at the real programmatic
entry point. Independent ready packets may run in parallel only when their
baselines are frozen and each writer has a separate isolated worktree. They do
not share implementation paths. Only the lead updates control status.

## Lead preparation checklist

Before changing a packet to `ready`, the campaign lead must:

1. Start from a clean, committed baseline after unrelated work has landed or
   moved to another worktree.
2. Record full 40-character baseline and upstream SHAs in both the packet and
   `status.yaml`. During Iteration 0, the packet baseline is recorded separately
   from the still-unset final campaign baseline.
3. Assign one worker, one reviewer, one verifier, one evidence operator, an
   isolated branch/worktree, and an explicit model or execution profile where
   the orchestration surface supports it.
4. Assign an absolute `VIBE_EVIDENCE_WORKSPACE` outside the repository and every
   linked Git worktree, a packet-unique `KILROY_RUN_ID`, and a stable non-secret
   runner identity. Never reuse a run ID across different candidate SHAs.
5. Confirm every dependency is `complete`, every required planned artifact now
   exists or is created by the packet, and every exact command resolves.
6. Resolve all lead-only decisions. A packet with an unresolved architecture,
   compatibility, message, snapshot, performance, or baseline decision stays
   `draft`.
7. Confirm the allowed paths are sufficient and contain no unrelated user work.
8. Freeze the packet. Scope changes after this point return it to `draft` or
   `blocked` for lead review.

## Worker operating procedure

An assigned worker follows this sequence exactly:

1. Read `AGENTS.md`, `openwiki/quickstart.md`, the roadmap sections referenced
   by the packet, the authority matrix, and the entire packet.
2. Verify that packet and `status.yaml` both say `ready` in the assigned clean
   control commit, all SHAs are full and resolvable, dependencies are complete,
   the candidate worktree is clean, and the evidence root is external.
3. Record the preflight result without changing repository files. If any check
   fails, report `blocked` and stop.
4. Change only allowed paths. Existing unrelated changes are never reformatted,
   moved, reverted, staged, or committed.
5. Run the packet's tests and evidence steps. Use `uv run`; do not substitute a
   narrower command because a required command is slow or fails.
6. Treat unexpected output, message, snapshot, API, performance, or fork-metric
   differences as findings, not baselines to update.
7. After pre-freeze checks pass, stop and hand the candidate to the lead. The
   worker is not authorized to create the candidate commit.
8. The lead creates a normal commit in the candidate worktree, records its SHA
   in a new control commit, assigns one evidence operator to the frozen external
   artifact write, and assigns read-only review and verification.
9. Run the packet's final diff-boundary check. If the diff contains a forbidden
   path or the candidate changed during verification, stop.
10. Submit the completion report defined by the packet. Do not declare the
   packet complete and do not land it.

## Isolation and concurrency

- Read-only survey and review work may share a clean candidate worktree.
- Every write-capable packet gets its own Git worktree and branch.
- Concurrent writers never share a worktree, evidence scenario directory, or
  manifest-writing process.
- An orchestration surface that cannot guarantee isolated writers is limited to
  read-only work for this campaign.
- The lead selects the model/profile. A worker may not substitute a cheaper,
  stronger, or differently configured model when that changes the packet's
  cost, behavior, or verification contract.
- No paid backend or external network is used unless the packet explicitly
  permits it and states a hard spend cap. Deterministic loopback fixtures are
  allowed only when named by the packet.

## Evidence boundary

Every packet uses the canonical external root:

```bash
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
```

The evidence workspace must not be the repository, a child of the repository,
a Git common directory, or any linked worktree. Evidence writes must leave the
candidate's `git status --short` unchanged. Scenario statuses in the evidence
manifest are only `pass` or `fail`; a missing future capability is a failed
scenario with a readable gap artifact and notes.

The packet lifecycle state `blocked` is campaign coordination state and must
not be written as a third scenario status.

A verifier may PASS a characterization/tooling packet whose scenario is an
explicit expected `fail` only when the packet requires that exact gap, the child
checks pass, and no unexpected failure is present. This PASS verifies honest
gap recording; it does not convert the roadmap scenario or campaign criterion
to pass.

## Verification and landing

The worker freezes all intended edits before verification. The verifier is
read-only and tests the exact candidate SHA against the packet's acceptance
criteria and scenarios. A denied or skipped verifier action invalidates the
verifier run. Any candidate mutation after verifier start invalidates its
verdict.

The candidate evidence scenario is recorded once under the packet run ID. The
verifier reads that manifest and reruns check-only commands without appending a
duplicate scenario entry. Its host verdict/report is stored separately by the
lead and bound to the same candidate SHA.

Only the campaign lead may accept a verifier result, mark a packet `complete`,
or authorize landing through the repository's normal verification mechanism.
Packet prose, pasted reports, reviewer approval, or a worker's local success do
not grant landing authority.

## Current packets

| Packet | Purpose | Dependency | Initial state |
|---|---|---|---|
| [I00-P01](packets/I00-P01-evidence-runner.md) | Build the external evidence runner and its failure/reproducibility contract. | None | `draft` |
| [I00-P02](packets/I00-P02-baseline-inventory.md) | Record exact baseline identity, fork ownership, hotspots, and divergence state. | I00-P01 | `draft` |
| [I00-P03](packets/I00-P03-programmatic-cli-characterization.md) | Add deterministic real-subprocess characterization for `vibe -p`. | I00-P01, planned I00-P04 | `draft` |

They remain `draft` until the documentation change is reviewed and committed,
the applicable packet starting SHA is frozen, and the evidence workspace and
assignees are recorded.

The first three packets are not the whole of Iteration 0. The control status
reserves I00-P99 for final consolidation, complete IT-01 through IT-15 status
crosscheck, authoritative baseline-only capture, verifier receipt, and
`campaign_baseline_sha` promotion. The lead must author and freeze that packet
before any Iteration 1 work; an early worker may not improvise it.
