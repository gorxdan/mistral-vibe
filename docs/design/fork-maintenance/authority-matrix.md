# Fork Maintenance Authority Matrix

Status: Governing execution policy

This matrix prevents a bounded implementation assignment from turning into an
implicit architecture, compatibility, or baseline decision. It applies to every
packet under [`packets/`](packets/) and supplements the repository `AGENTS.md`
and the [campaign roadmap](../fork-maintenance-roadmap.md).

## Roles

- **Campaign lead** owns campaign scope, baselines, packet readiness,
  assignments, sequencing, accepted risk, evidence acceptance, and landing.
- **Packet worker** implements one ready packet within its frozen path and
  behavior boundaries, runs the prescribed checks, and reports evidence or a
  blocker.
- **Reviewer** performs a read-only diff and contract review. Review approval is
  advisory and cannot change packet state or authorize landing.
- **Verifier** attacks the frozen candidate using the packet's acceptance
  criteria and produces a verdict with command evidence. The verifier cannot
  modify the candidate or repair failures.
- **Evidence operator** may run approved deterministic commands against a
  frozen SHA and write only to the assigned external evidence root. Unless a
  packet says otherwise, the worker fills this role before freeze and the
  verifier fills it after freeze.

One person or agent may hold more than one role only when the packet records it.
The verifier must still run in a separate read-only turn after candidate freeze.

## Decision matrix

| Decision or action | Campaign lead | Worker | Reviewer | Verifier |
|---|---|---|---|---|
| Define campaign scope or iteration order | Decides | No | Advises | No |
| Create or materially change a packet | Decides | Proposes by blocker report | Advises | No |
| Change `draft` to `ready` | Decides | No | No | No |
| Begin a `ready` packet | Assigns | Executes after preflight | No | No |
| Change allowed or forbidden paths | Decides; returns packet to `draft` | No | Advises | No |
| Select baseline, candidate, or upstream SHA | Decides and records | Validates only | Validates only | Validates only |
| Select evidence workspace or run ID | Decides and records | Validates and uses | Reads | Validates and uses |
| Choose branch, worktree, model, or execution profile | Decides | Uses assigned values | No | Uses assigned values |
| Edit an allowed implementation path | No direct requirement | Yes | No | No |
| Edit a path not listed as allowed | Decides only through packet revision | No | No | No |
| Touch an upstream-owned file | Explicitly approves named localized hunks | Only when packet names the path and seam | Reviews | Verifies |
| Add, delete, rename, split, or relocate an upstream-owned path | Explicit divergence decision required | No | No | No |
| Add a fork-owned sibling file named by the packet | Approves through packet | Yes | Reviews | Verifies |
| Reorder or broadly reformat upstream implementation | No for this campaign unless separately scoped | No | Flags | Fails candidate |
| Change architecture or dependency direction | Decides through a new/revised packet | No | Advises | No |
| Change public API, config, default, event order, error type, or side effect | Decides as compatibility work with migration boundary | No in behavior-preserving packet | Flags | Fails candidate |
| Change user-facing or model-visible messages | Approves message-delta record | No unless packet is an intentional message change | Flags | Compares |
| Update snapshots or golden message fixtures | Approves only for intentional behavior change | No in mechanical/characterization work | Flags | Fails unexplained delta |
| Update performance thresholds or baseline samples | Approves in a separate measurement decision | No | Flags | Rejects candidate-selected baseline |
| Relax lint, type, coverage, warning, complexity, spend, or divergence gates | Decides only as separately justified policy work | No | Flags | Fails silent relaxation |
| Add an accepted divergence or suppression | Explicit reviewed decision | No | Advises | Validates reason and scope |
| Use live network or a paid model | Approves packet and hard cap | Only as explicitly specified | No | Validates cap and attribution |
| Classify an unexpected test/message/performance difference | Decides after evidence | Reports and stops | Advises | Reports and fails/partials |
| Move `active` to `blocked` or `verification` | Records | Reports transition | No | No |
| Move `blocked` back to execution | Decides | No | No | No |
| Produce verifier verdict | Receives | No | No | Decides `PASS`, `FAIL`, or `PARTIAL` |
| Accept evidence and mark `complete` | Decides | No | No | No |
| Commit, push, open a PR, merge, or land | Explicitly authorizes as applicable | Only if packet explicitly includes the action | No | No |

## Worker discretion

Within the frozen packet, a worker may make only local implementation choices
that do not alter observable behavior, scope, or repository structure. Examples:

- Names of private test helpers inside an allowed new test file.
- Early-return versus small local helper when both obey repository conventions.
- Test parametrization that preserves every named case and assertion.
- Ordering of independent local verification commands before the mandatory
  final sequence.

The worker must stop rather than decide when a choice affects public behavior,
an upstream-owned hunk, a message, normalization, evidence schema, performance
measurement, dependency direction, or any path outside the allowlist.

## State-transition authority

| From | To | Who may request | Who records/authorizes | Required evidence |
|---|---|---|---|---|
| `draft` | `ready` | Lead, reviewer recommendation | Lead | Definition of Ready complete |
| `ready` | `active` | Assigned worker | Lead records; worker may start after preflight | Clean SHA/worktree and resolved dependencies |
| `active` | `blocked` | Worker, reviewer, or verifier | Lead records | Structured blocker report |
| `active` | `verification` | Worker | Lead records candidate SHA | Completion report and clean frozen candidate |
| `verification` | `blocked` | Reviewer or verifier | Lead records | Failed/partial evidence or invalidated candidate |
| `verification` | `complete` | Verifier returns PASS | Lead alone | Accepted manifest, review, and current verifier PASS |
| `blocked` | `ready` or `active` | Lead | Lead | Revised/resolved packet, with scope refrozen if changed |

No role self-promotes a packet to `complete`. A worker must not edit
`status.yaml` to make repository state appear authorized.

## Baseline and evidence authority

The lead freezes three distinct identities:

- `baseline_sha`: the clean preservation baseline measured before the packet.
- `candidate_sha`: the frozen implementation commit presented to verification.
- `upstream_sha`: the exact upstream tree used for ownership and mergeability.

All are full 40-character commits. Branch names, abbreviated hashes, tags, and
working-tree contents are not identities. A moved branch does not change a
frozen SHA.

The evidence operator may create scenario artifacts only under the assigned
external root. The operator may not:

- Regenerate baseline samples after seeing candidate results.
- Replace a failed scenario with a different command.
- omit or normalize an inconvenient semantic difference.
- Record `pass` when a required artifact is missing or unreadable.
- Record `blocked`; evidence status is strictly `pass` or `fail`.
- Mutate the candidate to make verification succeed.

## Message and snapshot authority

For behavior-preserving packets, roadmap message groups are frozen. The worker
captures actual output and compares it after only the roadmap-approved volatile
normalization. Labels, severity, status, verdict, recovery guidance, exit code,
protocol fields, tool identity, spend semantics, and verification authorization
are never normalized away.

An unexpected message or snapshot change blocks the packet. The worker does not
approve the change by updating a fixture. Intentional changes require a revised
packet with the exact old/new message inventory, migration impact, documentation
work, evaluation requirements, and rollback boundary.

## Performance authority

The worker may run only the frozen workloads, sample counts, seeds, environment,
and comparison method. The lead owns any change to those inputs. A regression
cannot be accepted by widening a threshold, discarding samples, changing the
machine, rebuilding the baseline from the candidate, or selecting a favorable
subset.

Performance collection stops before execution when either worktree is dirty.
Natural-noise calibration and paired comparison are evidence; they are not
worker-adjustable tolerances.

## Isolation and multi-agent authority

The lead may use multiple agents for independent read-only review. Write work is
parallel only when:

- each packet is ready and dependency-compatible;
- each writer has a distinct worktree and branch;
- allowed paths do not overlap;
- scenario evidence directories do not overlap; and
- manifest updates are serialized by the lead or the approved runner.

Shared-worktree team modes are read-only for this campaign. If an orchestration
surface does not expose per-worker isolation or the requested model/profile, the
lead must select another surface or run the work serially. A worker never spawns
additional writers unless its packet explicitly delegates that authority.

## Stop and escalation protocol

The worker stops immediately when:

- preflight identity, cleanliness, dependency, or evidence checks fail;
- a required command, fixture, ref, or planned artifact is absent;
- an edit outside `allowed_paths` appears necessary;
- an upstream-owned file needs a hunk not named in the packet;
- behavior, message, snapshot, schema, protocol, performance, or fork metrics
  differ unexpectedly;
- a test exposes a product defect that would require production repair;
- unrelated user changes overlap the packet;
- a required check is denied, skipped, flaky beyond the specified retry policy,
  or cannot run without network/payment not authorized by the packet; or
- candidate state changes after verification starts.

The blocker report must contain:

```text
Packet: <id>
State requested: blocked
Frozen baseline/candidate/upstream: <SHAs or candidate not yet frozen>
Failed step or command: <exact value>
Observed result: <exit code and concise evidence path>
Contract at risk: <AC/IT/MSG/path/invariant>
Why the packet cannot decide: <authority boundary>
Smallest lead decision needed: <one concrete question>
Candidate mutations already made: <paths or none>
Safe rollback: <exact revert/removal boundary>
```

The worker may preserve diagnostic evidence but must not continue speculative
implementation while waiting for the decision.
