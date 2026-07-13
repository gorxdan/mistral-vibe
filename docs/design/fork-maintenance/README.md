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
  campaign lead, worker, reviewer, and verifier.
- [Task-packet template](task-packet-template.md) is the required schema for all
  implementation packets.
- [`packets/`](packets/) contains the frozen, iteration-sized assignments.

These documents do not authorize implementation by themselves. A packet is
authorized only when the campaign lead changes its state to `ready`, fills every
required identity and assignment field, and records the same state in
`status.yaml`.

## Precedence

When instructions disagree, apply them in this order:

1. Current user instructions and the repository `AGENTS.md`.
2. A lead-frozen task packet whose state is `ready` or `active`.
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
- `verification`: implementation is frozen. Only read-only checks and the
  assigned verifier may run.
- `complete`: the lead accepted the evidence and verifier verdict.
- `blocked`: work stopped without widening scope or changing contracts.

Only the campaign lead may move a packet from `draft` to `ready`, from
`blocked` back to an executable state, or from `verification` to `complete`.
The assigned worker may report `active`, `verification`, or `blocked`, but does
not edit `status.yaml`; the lead records all state changes.

## Iteration 0 dependency graph

The first three packets deliberately establish infrastructure before product
characterization:

```text
I00-P01 Evidence runner and contract
        |\
        | +--> I00-P03 Programmatic CLI characterization
        |
        +----> I00-P02 Baseline identity and fork inventory
```

`I00-P02` and `I00-P03` may run in parallel only after `I00-P01` is complete,
the campaign baseline is frozen, and each writer has a separate isolated
worktree. They do not share implementation paths. Only the lead updates the
shared status file and evidence manifest coordination data.

## Lead preparation checklist

Before changing a packet to `ready`, the campaign lead must:

1. Start from a clean, committed baseline after unrelated work has landed or
   moved to another worktree.
2. Record full 40-character baseline and upstream SHAs in both the packet and
   `status.yaml`.
3. Assign one worker, one reviewer, one verifier, an isolated branch/worktree,
   and an explicit model or execution profile where the orchestration surface
   supports it.
4. Assign an absolute `VIBE_EVIDENCE_WORKSPACE` outside the repository and every
   linked Git worktree, plus a unique `KILROY_RUN_ID`.
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
2. Verify that packet and `status.yaml` both say `ready`, all SHAs are full and
   resolvable, dependencies are complete, the assigned worktree is clean, and
   the evidence root is external.
3. Record the preflight result without changing repository files. If any check
   fails, report `blocked` and stop.
4. Change only allowed paths. Existing unrelated changes are never reformatted,
   moved, reverted, staged, or committed.
5. Run the packet's tests and evidence steps. Use `uv run`; do not substitute a
   narrower command because a required command is slow or fails.
6. Treat unexpected output, message, snapshot, API, performance, or fork-metric
   differences as findings, not baselines to update.
7. Run the packet's final diff-boundary check. If the diff contains a forbidden
   path or the candidate changed during verification, stop.
8. Submit the completion report defined by the packet. Do not declare the
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
- No paid backend or live network is used unless the packet explicitly permits
  it and states a hard spend cap. The first three packets permit neither.

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

## Verification and landing

The worker freezes all intended edits before verification. The verifier is
read-only and tests the exact candidate SHA against the packet's acceptance
criteria and scenarios. A denied or skipped verifier action invalidates the
verifier run. Any candidate mutation after verifier start invalidates its
verdict.

Only the campaign lead may accept a verifier result, mark a packet `complete`,
or authorize landing through the repository's normal verification mechanism.
Packet prose, pasted reports, reviewer approval, or a worker's local success do
not grant landing authority.

## Current packets

| Packet | Purpose | Dependency | Initial state |
|---|---|---|---|
| [I00-P01](packets/I00-P01-evidence-runner.md) | Build the external evidence runner and its failure/reproducibility contract. | None | `draft` |
| [I00-P02](packets/I00-P02-baseline-inventory.md) | Record exact baseline identity, fork ownership, hotspots, and divergence state. | I00-P01 | `draft` |
| [I00-P03](packets/I00-P03-programmatic-cli-characterization.md) | Add deterministic real-subprocess characterization for `vibe -p`. | I00-P01 | `draft` |

They remain `draft` until the documentation change is reviewed and committed,
the resulting baseline SHA is frozen, and the evidence workspace and assignees
are recorded.
