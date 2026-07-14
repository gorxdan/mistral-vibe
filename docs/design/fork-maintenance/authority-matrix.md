# Fork Maintenance Authority Matrix

Status: Governing execution policy

This matrix prevents a bounded implementation assignment from turning into an
implicit architecture, compatibility, or baseline decision. It applies to every
packet under [`packets/`](packets/) and supplements the repository `AGENTS.md`
and the [campaign roadmap](../fork-maintenance-roadmap.md).

## Roles

- **Campaign lead** decides campaign scope, baselines, packet readiness,
  assignments, sequencing, accepted risk, evidence acceptance, and landing.
  The lead authorizes control-plane changes but does not delegate their
  execution to a model.
- **Trusted host** is the non-model control plane. It provisions and validates
  physical worktrees and durable evidence, freezes the verification recipe,
  performs candidate/control commits and lifecycle writes after lead approval,
  finalizes and digest-binds the manifest, runs trusted checks, stores receipts,
  and performs exact-object compare-and-swap delivery and landing actions.
- **Packet worker** implements one ready packet within its frozen path and
  behavior boundaries after host-validated active startup, runs development
  checks, and reports results or a blocker. It does not commit or change
  campaign state.
- **Reviewer** performs a read-only diff and contract review. Review approval is
  advisory and cannot change packet state or authorize landing.
- **Verifier** attacks the frozen candidate using the packet's acceptance
  criteria and produces a verdict with command evidence. The verifier cannot
  modify the candidate or repair failures.
- **Evidence operator** selects and observes an approved deterministic scenario
  once against a frozen SHA. The trusted host runner performs the durable
  evidence write. Unless a packet says otherwise, the lead assigns this role
  after candidate freeze. The verifier reads that evidence and reruns approved
  check-only commands without writing a duplicate scenario ID into the same run.

One person or agent may hold more than one model role only when the packet
records it. The trusted host is not an assignable model role. The verifier must
still run in a separate read-only turn after candidate freeze.

For every row below, "decides" or "authorizes" belongs to the campaign lead;
control-plane execution belongs to the trusted host. Auto-approve does not
transfer host authority to the model.

## Decision matrix

| Decision or action | Campaign lead | Worker | Evidence operator | Reviewer | Verifier |
|---|---|---|---|---|---|
| Define campaign scope or iteration order | Decides | No | No | Advises | No |
| Create or materially change a packet | Decides | Proposes by blocker report | No | Advises | No |
| Change `draft` to `ready` | Decides | No | No | No | No |
| Provision and validate execution topology | Authorizes | No | No | Reads result | Validates assigned identity only |
| Begin a `ready` packet | Authorizes `active` after host provisioning | Executes only after successful active AgentLoop startup | No | No | No |
| Change allowed or forbidden paths | Decides; returns packet to `draft` | No | No | Advises | No |
| Select baseline, candidate, or upstream SHA | Decides; host records | Validates only | Validates only | Validates only | Validates only |
| Select evidence workspace, run ID, sorted scenarios, or final manifest digest | Decides; host provisions, finalizes, and records | Reads assigned identity | Selects approved scenario | Reads | Validates only |
| Choose branch, worktree, model, or execution profile | Decides | Uses assigned values | Uses assigned values | No | Uses assigned values |
| Edit an allowed implementation path | No direct requirement | Yes | No | No | No |
| Edit a path not listed as allowed | Decides only through packet revision | No | No | No | No |
| Touch an upstream-owned file | Explicitly approves named localized hunks | Only when packet names the path and seam | No | Reviews | Verifies |
| Add, delete, rename, split, or relocate an upstream-owned path | Explicit divergence decision required | No | No | No | No |
| Add a fork-owned sibling file named by the packet | Approves through packet | Yes | No | Reviews | Verifies |
| Reorder or broadly reformat upstream implementation | No for this campaign unless separately scoped | No | No | Flags | Fails candidate |
| Change architecture or dependency direction | Decides through a new/revised packet | No | No | Advises | No |
| Change public API, config, default, event order, error type, or side effect | Decides as compatibility work with migration boundary | No in behavior-preserving packet | No | Flags | Fails candidate |
| Change user-facing or model-visible messages | Approves message-delta record | No unless packet is an intentional message change | Captures only | Flags | Compares |
| Update snapshots or golden message fixtures | Approves only for intentional behavior change | No in mechanical/characterization work | No | Flags | Fails unexplained delta |
| Update performance thresholds or baseline samples | Approves in a separate measurement decision | No | Selects/observes frozen measurement; host runner executes | Flags | Rejects candidate-selected baseline |
| Relax lint, type, coverage, warning, complexity, spend, or divergence gates | Decides only as separately justified policy work | No | No | Flags | Fails silent relaxation |
| Add an accepted divergence or suppression | Explicit reviewed decision | No | No | Advises | Validates reason and scope |
| Use live network or a paid model | Approves packet and hard cap | Only as explicitly specified for development checks | Selects/observes approved capped scenario; host runner executes evidence run | No | Validates cap and attribution |
| Run a frozen evidence scenario after candidate freeze | Assigns exactly one operator | No | Selects/observes once; host runner writes assigned artifacts | Observes | May request approved check-only commands, not the same scenario ID |
| Run receipt-authorizing checks | Authorizes frozen recipe | No | No | Observes | Requests no-argument host verification after PASS; host runs direct argv |
| Mutate control, evidence, Git administration, host logs, or receipts | Authorizes applicable host operation | No | No | No | No |
| Classify an unexpected test/message/performance difference | Decides after evidence | Reports and stops | Reports observed failure without changing inputs; host records | Advises | Reports and fails/partials |
| Move `active` to `blocked` or `verification` | Authorizes; host records | Reports requested state | No | No | No |
| Move `blocked` back to execution | Authorizes; host records | No | No | No | No |
| Produce verifier verdict | Receives | No | No | No | Decides `PASS`, `FAIL`, or `PARTIAL` |
| Accept evidence and mark `complete` | Decides; host records | No | No | No | No |
| Commit the implementation candidate | Authorizes; host performs | No | No | No | No |
| Commit control-plane packet/status changes | Authorizes; host performs | No | No | No | No |
| Push, open a PR, merge, or land | Explicitly authorizes; host performs | No | No | No | No |

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
| `draft` | `ready` | Lead, with reviewer recommendation if assigned | Lead authorizes; trusted host records | Definition of Ready complete |
| `ready` | `blocked` | Trusted host after topology failure | Lead authorizes; trusted host records | Structured host blocker; no model session or candidate edit |
| `ready` | `active` | Lead | Lead authorizes; trusted host records | Physical topology, clean identities, metadata, dependencies, and durable evidence path validated before AgentLoop startup |
| `active` | `blocked` | Worker or trusted host | Lead authorizes; trusted host records | Structured blocker report or host capability failure |
| `active` | `verification` | Worker reports implementation ready for freeze | Lead authorizes; trusted host creates the candidate, commits initial verification state with its SHA and exact sorted scenarios, finalizes evidence, then commits the manifest digest in a second final control commit | Completion report, clean candidate, strict finalized manifest, and successful verification-topology startup from only the final control commit |
| `verification` | `blocked` | Evidence operator, reviewer, verifier, or trusted host | Lead authorizes; trusted host records | Failed/partial evidence, invalid receipt, or invalidated candidate |
| `verification` | `complete` | Verifier produces a current PASS | Lead alone authorizes; trusted host records | Accepted manifest, review, structured outcome, and valid current receipt |
| `blocked` | `ready` or `active` | Lead | Lead authorizes; trusted host records | Revised or resolved packet, with scope refrozen if changed and topology revalidated |

No role self-promotes a packet to `complete`. A worker must not edit
`status.yaml` to make repository state appear authorized.

## Control-plane authority

The host supplies the recipe through user, `VIBE_` environment, or programmatic
configuration. Project TOML cannot supply or replace it. Its frozen
`trusted_verification_recipe.execution_topology` is execution authority. A
copied command, campaign shell variable, packet example, or model-written
preflight cannot substitute for it. Before the first model turn, AgentLoop
validates two distinct, existing, physical, registered worktrees; exact clean
heads and branch; roles, execution profile, dependency states, exact sorted
scenarios, run identity, and verification manifest digest;
and a durable evidence workspace outside `/tmp` that does not overlap any
worktree or the Git common directory in either direction. Packet/status data
must be regular tracked blobs from the exact control commit. Git probes discard
ambient `GIT_*` variables and user/system configuration. Startup fails closed
on any mismatch.

An active assignment contains `state = "active"`, the committed control SHA,
baseline and upstream SHAs, and neither candidate nor manifest SHA. A
verification assignment is a fresh session with `state = "verification"`, the
second and final verification control SHA, the full frozen candidate SHA, and
the canonical `evidence_manifest_sha256`. Packet and `status.yaml` state,
scenarios, and digest must exactly match the configured values. Topology
`max_turns` and `max_session_tokens` are hard runtime caps.

The worker never commits in either worktree. It also cannot mutate the control
worktree, evidence workspace, Git administration, host logs, or verification
receipts through model tools. After lead authorization, the trusted host ends
the active session, creates the candidate commit, and records candidate SHA,
verification state, and sorted scenarios in an initial control commit. The host
runner finalizes evidence under the manifest lock. The host then records its
canonical digest in both packet and status in a second final control commit and
only then starts the verification session. Control commits never enter the
candidate diff.

Topology installs an authoritative canonical tool ceiling. Active roots have at
most `bash`, `edit`, `glob`, `grep`, `read`, `skill`, `task`, `todo`,
and `write_file`; verification roots have at most `glob`, `grep`, `read`,
`skill`, `task`, and `verify_work`. Managed Task accepts only effective
read-only built-in reviewer and verifier profiles, whose ceiling is `bash`,
`glob`, `grep`, `read`, and `skill`, intersected with any structured
manifest. Project/plugin tools, MCP/connectors, workflows, teams, web tools,
`tool_search`, and `land_work` cannot be added by config or delegation.
Managed reads are confined to the candidate/control/evidence roots, session
scratchpad, host skills, and active prompt files. Host logs, receipts, runtime
state, and unrelated host paths are denied. Strict managed Bash additionally
sees pinned toolchain/runtime and minimal system roots, masks unrelated host
top-level trees, mounts the candidate read-only, and rejects `background=true`.
Candidate changes must use the bounded file tools; literal allowed paths are
exact and recursive scope requires `<directory>/**`.

The repository validates provisioned topology and implements trusted-check,
receipt, delivery, and landing primitives. Worktree provisioning, lifecycle
writes, attested-environment installation, and evidence execution/finalization
still require a trusted external host workflow. If that workflow is unavailable,
the lead blocks the packet before model startup.

## Baseline and evidence authority

The lead freezes four distinct identities:

- Packet `baseline_sha`: the clean starting commit used for that packet's diff
  and rollback boundary.
- `candidate_sha`: the frozen implementation commit presented to verification.
- `upstream_sha`: the exact upstream tree used for ownership and mergeability.
- Campaign `campaign_baseline_sha`: the consolidated clean Iteration 0 commit
  used as the preservation baseline for subsequent iterations.

All are full 40-character commits. Branch names, abbreviated hashes, tags, and
working-tree contents are not identities. A moved branch does not change a
frozen SHA.

During Iteration 0, packet baselines may advance as behavior-neutral tooling and
characterization land. The campaign baseline remains unset until the complete
Iteration 0 candidate is frozen, then receives a baseline-only evidence capture
where manifest baseline and candidate identities are equal. A worker cannot
promote its own packet candidate to the campaign baseline.

Every packet candidate uses a unique run ID. A manifest cannot mix evidence from
different candidate SHAs. The final consolidated baseline capture uses a new
campaign run ID; it references provisional packet evidence by digest/path when
useful but does not merge incompatible manifest entries.

The evidence operator selects and observes a scenario; only the trusted host
runner creates its artifacts. In active state, the assigned external root must
pass the host's write, file-fsync, read-back, unlink, and parent-fsync probe; the
verification gate is read-only. The root must not be under `/tmp`, `/run`,
`/dev/shm`, or a `tmpfs`/`ramfs` mount, and may
neither contain nor be contained by any linked worktree or the Git common
directory. Model scratchpads and copied reports are not durable evidence. The
host finalizes and validates the manifest while holding its lock and with an
empty `.reservations` directory. Packet, status, and topology must bind its
canonical digest, exact sorted scenario IDs, and parsed-value-identical
scenario contracts. Each contract freezes direct argv, environment, surface,
artifacts, result schema, status, and ordered note allowlists. Strict schema and
canonical JSON, exact tree inventories, every artifact hash, all identities, and the
committed candidate `uv.lock` digest must match; symlinks, hardlinks, duplicate
keys, non-finite values, extra entries, and missing artifacts are rejected. The
operator may not:

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
- manifest updates are serialized by the trusted host runner.

Shared-worktree team modes are read-only for this campaign. If an orchestration
surface does not expose per-worker isolation or the requested model/profile, the
lead must select another surface or run the work serially. A worker never spawns
additional writers unless its packet explicitly delegates that authority.

Trusted checks are frozen direct `argv` arrays. The host rejects a shell or
`env` executable, including either selected behind `uv run`. Trusted recipes
reject `uv` and pre-commit entrypoints. The host must pre-provision dependencies
and must not substitute another package-manager installation command; the
runtime does not classify every package-manager CLI. They
execute only on Linux in fail-closed Bubblewrap; Seatbelt is not supported.
Each check uses an exact-HEAD Git-exported snapshot with no Git metadata and a
private copy of a pre-provisioned native executable with the configured SHA-256;
shebang wrappers fail closed. Network is off, a separate host-owned environment
attestation is hashed before and after execution, credentials/config are
scrubbed, writable state is disposable, and
combined output is capped at 1 MiB. Auto-approve cannot override a hard policy
denial, a `NEVER`
permission, or a configured safety-judge deferral. Auto-approve also requires a
confining backend, disables network, and cannot widen the managed tool ceiling.

Approved delivery and landing use the exact authorized object ID and update the
destination ref with compare-and-swap. A moved expected ref fails the operation;
the host never re-resolves it and substitutes the new target. Checked-out
worktree materialization is a cooperative multi-file operation: the merge lock
serializes Vibe landing operations, while the operator must keep external
editors and Git processes idle during the approved landing window.

## Stop and escalation protocol

The worker stops immediately when:

- host startup or runtime identity, cleanliness, dependency, or evidence checks
  fail;
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

Filesystem-confinement, policy-denial, and sandbox-startup failures are host
capability failures, not invitations to retry with a weaker command. Three
consecutive failures in the same class end the turn with `BLOCKED`. The worker
reports the smallest host or lead decision needed and does not invent a waiver.

Raw reviewer or verifier prose is advisory. Structured `completed`, `outcome`,
and receipt state are authoritative. If a final response claims completion
while those fields do not support it, the host replaces the claim with the
recorded partial, failed, or blocked result before emission.

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
