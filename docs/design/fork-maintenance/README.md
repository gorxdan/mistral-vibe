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
- [Harness integrity contract](../harness-integrity.md) defines the
  host-provisioned topology, protected state, trusted-command, circuit-breaker,
  and completion-report rules used to execute those assignments.

These documents do not authorize implementation by themselves. A packet is
eligible for host provisioning only when the campaign lead freezes every
required identity and assignment field and authorizes `ready`, then the trusted
host records and commits that state in both control documents. Model work begins
only after the host provisions and validates the topology, the lead authorizes
`active`, and a root AgentLoop starts successfully with that frozen active
topology.

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
After all Iteration 0 deliverables are consolidated and green, the lead
authorizes the campaign baseline. The host records that commit in `status.yaml`
as `campaign_baseline_sha` and performs a
baseline-only capture with manifest `baseline_sha == candidate_sha`. Provisional
inventory from I00-P02 is rerun at that final commit. No Iteration 1 structural
work begins before this bootstrap is complete.

## Host-provisioned control and candidate worktrees

Packet/status metadata and implementation candidates live on separate branches:

- The **control worktree** is a dedicated `maintenance/fork-maintenance-control`
  branch containing `docs/design/fork-maintenance/`. The lead alone authorizes
  changes to packet state, assignments, baseline/candidate SHAs, and evidence
  IDs; the trusted host updates and commits them.
- A **candidate worktree** branches from the packet `baseline_sha`. It contains
  implementation edits only and never edits packet/status documents.

This separation avoids a self-referential SHA: a commit cannot contain its own
hash. A control commit may name the candidate's starting SHA, and a later
control commit may record the frozen candidate SHA without changing that
candidate.

The host installs the recipe through user configuration, a `VIBE_` environment
setting, or programmatic initialization. Project `.vibe/config.toml` cannot
supply or replace it, and a bound recipe forces verification on. Within that
recipe, execution authority is the frozen
`trusted_verification_recipe.execution_topology`, not campaign shell variables,
packet prose, or a model-run preflight. The topology binds the packet path and
ID, control path and SHA, candidate path and branch, lifecycle state, baseline,
upstream, optional frozen candidate SHA, durable evidence identity, runtime
caps, and, for verification, the canonical evidence-manifest SHA-256.

At root AgentLoop startup, the host verifies that control and candidate are
distinct registered physical worktrees, both are clean at their expected SHAs,
dependencies are complete, packet/status scenarios are identical and sorted,
and evidence is a durably writable directory outside
`/tmp` that neither contains nor is contained by any worktree or the Git common
directory. Packet and status metadata must be regular tracked blobs read from
the exact control commit. Git probes discard ambient `GIT_*` variables and
user/system Git configuration. Any mismatch aborts startup before a model turn.
A copied directory, plumbing-built ref, working-tree metadata substitution,
temporary path, or agent assertion is not an allowed substitute.

The active and verification assignments are separate frozen sessions:

These are the only executable managed states. `ready`, `blocked`, and
`complete` cannot start a managed AgentLoop.

| Topology field | Active worker session | Verification session |
|---|---|---|
| `state` | `active` | `verification` |
| `candidate_sha` | Absent/`null` | Full frozen 40-character SHA |
| `evidence_manifest_sha256` | Absent/`null` | Canonical 64-character manifest digest |
| Candidate `HEAD` at startup | `baseline_sha` | `candidate_sha` |
| Packet and `status.yaml` | Both `active` | Both `verification` |

After candidate freeze, the host ends the active session, creates the candidate
commit, and records verification state, candidate SHA, and the exact sorted
scenario list in an initial control commit. The host runner finalizes evidence;
the host hashes the canonical manifest and records that digest in packet and
status in a second, final verification control commit. Only then may it start a
new AgentLoop with the verification topology. Configuration reload cannot
change a topology frozen into an existing session. `max_turns` and
`max_session_tokens` cap the managed root session at runtime.

The active candidate is writable only at packet-allowed paths through bounded
file tools. Managed Bash sees it read-only and is for check-only commands. That
write access does not authorize an unreceipted completion claim; AgentLoop
replaces such a claim with current host verification status.

Managed sessions use an authoritative canonical capability ceiling. Active
roots have at most `bash`, `edit`, `glob`, `grep`, `read`, `skill`,
`task`, `todo`, and `write_file`; verification roots have at most `glob`,
`grep`, `read`, `skill`, `task`, and `verify_work`. Managed Task accepts
only effective read-only built-in reviewer and verifier profiles. Their maximum
catalog is `bash`, `glob`, `grep`, `read`, and `skill`, intersected with
any structured manifest. Project/plugin tools, MCP/connectors, workflows,
teams, web tools, `tool_search`, and `land_work` cannot enter this catalog.
Managed read tools are confined to the candidate and control trees, evidence
root, session scratchpad, host skill roots, and active prompt files. Host logs,
receipts, runtime state, and unrelated host paths are not readable. Strict Bash
also exposes pinned toolchain/runtime and minimal system roots while masking
unrelated host top-level trees, and rejects `background=true`.

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
- `ready`: the lead has frozen scope, SHAs, dependencies, ownership, trusted
  check IDs, and evidence identity. The host may provision the topology; no
  worker AgentLoop starts in this state.
- `active`: the host has validated the provisioned topology, recorded the
  lead-authorized transition, and started the worker in the exact candidate
  worktree. `candidate_sha` is absent and candidate `HEAD` initially equals
  `baseline_sha`.
- `verification`: implementation is frozen. Candidate and repository access is
  read-only. `candidate_sha` is required and equals candidate `HEAD`. The
  initial verification control commit authorizes only the approved host runner
  to finalize its assigned durable evidence artifacts. A managed verification
  AgentLoop may start only after the final control commit also records the
  matching `evidence.manifest_sha256`; reviewer and verifier model tools remain
  read-only.
- `complete`: the lead accepted the evidence and verifier verdict.
- `blocked`: work stopped without widening scope or changing contracts.

Only the campaign lead may authorize a lifecycle transition. The trusted host,
not a model role, records and commits the transition. The assigned worker may
request `verification` or report `blocked`, but cannot edit `status.yaml`,
commit either worktree, or start the next lifecycle session.

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
not share implementation paths. Only the lead authorizes control status
changes; the host records them.

## Lead and host preparation checklist

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
   runner identity. Record the same nonempty sorted unique scenario list in the
   packet and status. Never reuse a run ID across different candidate SHAs.
5. Confirm every dependency is `complete`, every required planned artifact now
   exists or is created by the packet, and every exact command resolves.
6. Resolve all lead-only decisions. A packet with an unresolved architecture,
   compatibility, message, snapshot, performance, or baseline decision stays
   `draft`.
7. Confirm the allowed paths are sufficient and contain no unrelated user work.
   Literal paths authorize exact files; recursive directory scope uses
   `<directory>/**`.
8. Freeze the packet. Scope changes after this point return it to `draft` or
   `blocked` for lead review.

Before starting the worker, the host must then:

1. Create distinct registered physical control and candidate worktrees and a
   durable evidence workspace outside `/tmp`, `/run`, `/dev/shm`, and any
   `tmpfs`/`ramfs` mount that does not overlap any linked worktree or Git
   administration in either direction. Active startup runs the durable
   write/`fsync`/read/unlink/parent-`fsync` probe; verification startup is
   read-only.
2. Configure `trusted_verification_recipe.execution_topology` with every
   required active field. `state` is `active`; `candidate_sha` is absent;
   `evidence_manifest_sha256` is absent; candidate `HEAD` equals `baseline_sha`.
   Set `max_turns` and `max_session_tokens` to the packet's host-approved caps.
3. Configure every receipt-authorizing check as direct `argv`, `cwd`, and
   timeout values plus the pre-provisioned executable's SHA-256 and a separate
   host-owned environment-attestation path and SHA-256. A shell or `env`
   executable is invalid, including behind `uv run`; trusted recipes reject
   `uv` and pre-commit entrypoints. Pre-provision every dependency and do not
   substitute another package-manager installation command because the runtime
   does not classify every package-manager CLI. The executable must be native,
   not a shebang wrapper; use a pinned
   interpreter plus `-m <module>` or a script argument. The runner executes a
   private copy, while the attestation remains a host assertion rather than a
   transitive dependency hash.
4. Validate control/candidate cleanliness and identity, packet/status agreement,
   roles, execution profile, evidence identity, exact frozen scenario
   contracts, and completed dependencies.
5. Confirm Bubblewrap or Seatbelt is available for strict managed model Bash,
   and Linux Bubblewrap specifically is available for trusted checks. Trusted
   checks do not support Seatbelt; neither mode accepts `unshare` or an
   unsandboxed fallback.
6. Ask the lead to authorize `ready -> active`, record and commit that
   transition, then start the root AgentLoop in the candidate worktree. Startup
   repeats topology validation before the model receives a turn.

If provisioning or validation fails, the host reports the exact mismatch and
the lead decides whether to block; the host records the resulting state. It
does not start a model to find a workaround.

The repository currently validates provisioned topology and contains trusted
check, receipt, delivery, and landing primitives. It does not expose one
campaign-host command that performs worktree provisioning, packet/status
transitions, attested-environment installation, or evidence execution and
finalization. Each ready packet must name the trusted external operator
workflow that performs those steps; without one, it remains blocked.

## Worker operating procedure

An assigned worker follows this sequence exactly:

1. Read `AGENTS.md`, `openwiki/quickstart.md`, the roadmap sections referenced
   by the packet, the authority matrix, and the entire packet.
2. Confirm the host startup report says the topology is `active` and identifies
   this packet, candidate root, baseline SHA, branch, and durable evidence run.
   Do not rerun or replace host topology validation with copied shell commands.
3. If runtime observations contradict the host report, report `blocked` and
   stop. Do not create worktrees, rewrite refs, relocate evidence, or modify
   control files to make the assignment appear valid.
4. Change only allowed paths. Existing unrelated changes are never reformatted,
   moved, reverted, staged, or committed.
5. Run packet-authorized development checks in check-only mode against the
   read-only Bash mount, with bytecode/test caches disabled or redirected to the
   scratchpad. Apply fixes through bounded file tools. Receipt-authorizing
   checks run only through the host-frozen recipe; the worker cannot substitute
   commands, shells, pipelines, or narrower checks because a required check is
   slow or fails.
6. Treat unexpected output, message, snapshot, API, performance, or fork-metric
   differences as findings, not baselines to update.
7. After pre-freeze checks pass, stop and hand the candidate to the lead using
   `READY_FOR_HOST_FREEZE:`, `BLOCKED:`, or `IN_PROGRESS:`. The worker is not
   authorized to create the candidate commit.
8. After lead approval, the host creates a normal candidate commit and an
   initial control commit recording `state: verification`, its full SHA, and the
   exact sorted scenario list. No verification AgentLoop starts yet. The one
   approved host runner finalizes the frozen external evidence under the
   manifest lock.
9. The host validates the strict canonical manifest, computes its SHA-256, and
   records the same `manifest_sha256` in packet and status in a second, final
   verification control commit. It starts a fresh read-only verification
   session only with a topology that binds that final control SHA and digest.
10. In the verification session, the host runs the packet's frozen diff-boundary
   check and the read-only verifier inspects its result. A forbidden path or any
   candidate change invalidates verification.
11. The host assembles the completion record defined by the packet from worker,
    reviewer, verifier, outcome, and receipt data. No model declares the packet
    complete or lands it.

## Isolation and concurrency

- The host provisions and removes every campaign worktree. Model tools may list
  worktrees but may not add, move, repair, prune, or remove them or write Git
  refs, reflogs, packed refs, or worktree administration.
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
- Auto-approve does not override a `NEVER` rule, protected host path, or a
  configured safety-judge deferral. A judge error, timeout, refusal, spend
  denial, or invalid response remains a deferral and fails closed.
- Auto-approve requires Bubblewrap or Seatbelt, disables network, scrubs the
  model environment, and uses disposable caches. It does not widen the managed
  tool ceiling or make candidate/control/evidence/Git state writable.

Three consecutive failures in the same capability class end the turn as
`HOST CAPABILITY STATUS: BLOCKED`. Relevant classes include filesystem
confinement, policy denial, and sandbox startup. The worker does not attempt a
fourth workaround, switch to Git plumbing, delete host state, or ask the user to
repair harness administration.

## Evidence boundary

Every packet uses the canonical external root:

```bash
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
```

The evidence workspace must be neither an ancestor nor a descendant of the
repository, a Git common directory, or any linked worktree. It must also be
outside `/tmp`, `/run`, `/dev/shm`, every other system-temporary location, and
Linux `tmpfs`/`ramfs`. Active startup verifies a durable write, file `fsync`,
read-back, unlink, and parent-directory `fsync`; verification startup is
non-mutating. Evidence writes must leave the candidate's `git status --short`
unchanged. Scenario statuses in the evidence manifest are only `pass` or
`fail`; a missing future capability is a failed scenario with a readable gap
artifact and exact preauthorized notes.

Manifest finalization and startup validation both hold the manifest lock and
require an empty `.reservations` directory. Packet, status, and topology bind
the same digest, exact sorted scenarios, and parsed-value-identical scenario
contracts. Those contracts freeze direct command argv, exact recorded
environment, surface, required artifact types, the result JSON schema, expected
status, and ordered note/gap-note allowlists. Strict canonical JSON, identities,
exact root/tree inventories, every artifact hash, and the committed candidate
`uv.lock` digest must match. Symlinks, hardlinks, duplicate keys, non-finite
numbers, extra fields or files, and missing artifacts fail the gate.

The evidence workspace is read-only to model tools. The evidence operator
selects the approved scenario; the host runner performs its write. A verifier
reads the durable evidence path supplied by the host. A session scratchpad is
temporary, may have a different mount view, and cannot satisfy campaign
evidence requirements.

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

The candidate evidence scenario is recorded once under the packet run ID by an
approved host runner. The verifier reads that manifest and requests only the
frozen direct-argv checks; it cannot append a duplicate scenario entry. The host
stores the verdict separately and binds it to the same candidate SHA.

Raw subagent prose is diagnostic only. `completed`, the structured task
`outcome`, current candidate/base identity, and the configured receipt are
authoritative. A literal `VERDICT: PASS` does not count when execution was
interrupted, a tool was denied or skipped, the outcome failed, the attempt was
superseded, or the receipt is missing or stale. Until authority is current,
AgentLoop suppresses contradictory tool-free completion text and emits a host
`IN_PROGRESS`, `PARTIAL`, or `BLOCKED` status instead.

Trusted checks are frozen `argv` arrays executed with `shell=False`. A shell or
`env` cannot be the executable, and either is rejected behind `uv run`; a
pipeline, `set +e`, or trailing successful command can never create receipt
authority. The trusted runner requires Linux Bubblewrap; Seatbelt is not
supported for trusted checks. It uses an exact-HEAD Git-exported snapshot with
no Git metadata,
executes a pre-provisioned absolute or sanitized-`PATH` executable whose
SHA-256 matches the recipe, disables network, scrubs host credentials/config,
uses disposable writable state, and caps combined stdout/stderr at 1 MiB. It
never bootstraps a trusted-check environment.

In the managed verification session, no-argument `verify_work` uses the frozen
topology and remains available without a legacy `worktree_manager.active`
record. Model-provided arguments cannot replace the frozen recipe.

Only the campaign lead may accept a verifier result or authorize completion and
landing. The host records the transition or performs the landing operation.
Packet prose, pasted reports, reviewer approval, or a worker's local success do
not grant authority.

Delivery and landing act on the exact authorized object ID and update the
destination ref with compare-and-swap. If the expected ref moves, the operation
fails instead of resolving or delivering the ref's new target.

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
