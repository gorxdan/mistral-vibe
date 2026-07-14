# Fork Maintenance Task-Packet Template

Copy this file to `packets/I<iteration>-P<sequence>-<slug>.md`, replace every
angle-bracket placeholder, remove instructional text, and keep the section
order. A packet remains `draft` while any field required for `ready` is `null`,
empty, ambiguous, or unresolved. `candidate_sha` and
`evidence.manifest_sha256` are intentional lifecycle exceptions: both stay
`null` through `ready`/`active`; the candidate is recorded in the initial
verification control commit and the manifest digest in the second, final one.

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
  manifest_sha256: null
  scenarios: []
  scenario_contracts: []
packet_acceptance_criteria: []
roadmap_contributions: []
messages: []
paths:
  allowed: []
  forbidden: []
---
```

Field completeness by state:

| Field | `draft` | `ready`/`active` | `verification`/`complete` |
|---|---|---|---|
| Owner/reviewer/verifier/evidence operator, dependencies, packet baseline/upstream SHA, worktree, branch, execution profile, evidence workspace/run/runner IDs, scenarios, paths, packet ACs, roadmap contributions | May be incomplete | Required and frozen | Required and frozen |
| `candidate_sha` | `null` | `null` until freeze | Full frozen 40-character commit |
| `evidence.manifest_sha256` | `null` | `null` | Full lowercase 64-character digest before a verification AgentLoop starts |
| Campaign preservation baseline | May be unset during Iteration 0 bootstrap | May be unset only for authorized Iteration 0 packets | Required before Iteration 0 exits or later iterations begin |

The trusted host writes `baseline_sha`, `candidate_sha`, and
`evidence.manifest_sha256` across the ready/active, initial verification, and
final verification control commits, never in the implementation candidate. The
frozen execution topology supplies the immutable final control commit SHA
because a file cannot contain the hash of its own commit.

# <Packet ID>: <Title>

Execution state comes from the host-configured execution topology and must match
this packet's frontmatter and the `status.yaml` entry at its exact
`control_sha`. Prose cannot supply or override state.

## Outcome

State one externally verifiable result. Do not describe a broad aspiration or
combine multiple rollback boundaries.

## Why this packet exists

Tie the result to a roadmap risk, dependency, or preservation contract. Explain
why it is ordered here and what later work it unblocks.

## Definition of Ready

The campaign lead checks every item before authorizing `ready`; the host then
records and commits that state:

- [ ] `baseline_sha` and `upstream_sha` are full, resolvable 40-character commits.
- [ ] `baseline_sha` is the packet's clean starting commit. If Iteration 0 is
      still bootstrapping, it is not mislabeled as the final campaign baseline.
- [ ] `owner`, `reviewer`, `verifier`, `evidence_operator`, `worktree`, `branch`,
      and `execution_profile` are assigned.
- [ ] The ready packet/status are committed in a clean dedicated control
      worktree; the host will bind that commit as `control_sha`.
- [ ] The assigned worktree is clean and contains no unrelated work.
- [ ] Every dependency is `complete` in `status.yaml`.
- [ ] `VIBE_EVIDENCE_WORKSPACE` is absolute, outside the system temporary
      directory, and neither contains nor is contained by the repository, Git
      common directory, or any linked worktree. `/tmp` is prohibited.
- [ ] `KILROY_RUN_ID` is unique and the scenario directories are not shared by
      another active writer.
- [ ] `runner_id` is a stable non-secret machine/runner label.
- [ ] `evidence.scenarios` is nonempty, sorted, unique, and exactly matches the
      packet's `status.yaml` `required_scenarios` entry.
- [ ] `evidence.scenario_contracts` is sorted by unique `id`, has exactly the
      same IDs as `evidence.scenarios`, and is YAML-value identical to the
      packet status entry's `evidence.scenario_contracts`.
- [ ] Every allowed path is sufficient, every forbidden path is explicit, and
      no path overlaps another active packet.
- [ ] Literal allowed paths name exact files. Any intended recursive directory
      scope is written explicitly as `<directory>/**`.
- [ ] Every command and referenced fixture exists, except files this packet is
      explicitly responsible for creating.
- [ ] Every receipt-authorizing command is represented by a named direct `argv`
      check in the host recipe. No trusted check invokes a shell.
- [ ] The trusted recipe is installed through host-controlled user, `VIBE_`
      environment, or programmatic configuration. It is not in project TOML.
- [ ] The host can provision distinct registered physical control/candidate
      worktrees and a durable evidence directory before any model starts.
- [ ] Bubblewrap or Seatbelt is available for strict managed model Bash. Linux
      Bubblewrap is separately available for trusted checks; trusted checks do
      not support Seatbelt. `unshare` and unsandboxed fallback are unacceptable.
- [ ] Lead-only decisions below are resolved; no compatibility, message,
      snapshot, performance, architecture, or baseline decision is delegated.
- [ ] Rollback removes only this packet's changes.

If any item is false, the packet stays `draft` and no implementation begins.
Changing it to `ready` permits host provisioning, not worker execution. The
worker starts only after the host records `active` and AgentLoop accepts the
frozen topology.

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
- No model-run commit, control transition, push, PR, merge, or landing action.
  The campaign lead authorizes these actions and the host performs them.
- No mutation of the control worktree, evidence workspace, Git administration,
  host logs, or verification receipt storage through any model tool.

## Required reading and inputs

Read these before editing:

- `AGENTS.md`
- `openwiki/quickstart.md`
- `<roadmap section>`
- `<source/test files that establish the real contract>`

## Host execution topology

This section is a declaration for the campaign lead and host. It is not a
model-run preflight. Copying values from this packet into shell variables does
not create execution authority.

The host configures the exact values below under the frozen
`trusted_verification_recipe.execution_topology`:

| Field | Packet value or source |
|---|---|
| `packet_id` | `<packet id>` |
| `packet_path` | `<repository-relative packet path>` |
| `status_path` | `docs/design/fork-maintenance/status.yaml` |
| `control_worktree` | `<absolute host-provisioned physical worktree>` |
| `control_sha` | `<full SHA of committed lifecycle state>` |
| `candidate_worktree` | Frontmatter `worktree` |
| `candidate_branch` | Frontmatter `branch` |
| `baseline_sha` | Frontmatter `baseline_sha` |
| `upstream_sha` | Frontmatter `upstream_sha` |
| `evidence_workspace` | Frontmatter `evidence.workspace` |
| `run_id` | Frontmatter `evidence.run_id` |
| `runner_id` | Frontmatter `evidence.runner_id` |
| `max_turns` | `<host-approved positive cap>` |
| `max_session_tokens` | `<host-approved positive cap>` |

State-specific fields are exact:

`active` and `verification` are the only executable managed states. `ready`,
`blocked`, and `complete` are control-plane states.

| Field or observation | Active implementation session | Verification session |
|---|---|---|
| `state` | `active` | `verification` |
| `candidate_sha` | Absent/`null` | Required full 40-character SHA |
| `evidence_manifest_sha256` | Absent/`null` | Frontmatter `evidence.manifest_sha256` |
| Candidate `HEAD` | `baseline_sha` | `candidate_sha` |
| Packet and status state | Both `active` | Both `verification` |
| Model write authority | Allowed candidate paths only | Read-only |

The active session ends before the host creates the candidate commit. The host
records verification state, candidate SHA, and exact sorted scenarios in an
initial control commit. The host runner then finalizes evidence, and the host
records its canonical digest in packet and status in a second, final
verification control commit. Only that final commit can start the verification
AgentLoop. Do not reuse the active recipe or reload configuration to adopt the
transition. `max_turns` and `max_session_tokens` are enforced runtime ceilings.

Active candidate write access remains limited to `allowed_paths` and does not
authorize an unreceipted completion claim. In verification, no-argument
`verify_work` uses the frozen topology even when no legacy
`worktree_manager.active` record exists.

## Host startup gate

Before the first model turn, root AgentLoop must accept the topology. The host
gate verifies:

- Control and candidate are distinct registered physical Git worktrees.
- The current session root is the configured candidate worktree.
- Both worktrees are clean; control and candidate `HEAD` values, candidate
  branch, baseline, upstream, and optional candidate SHA are exact.
- Packet frontmatter and `status.yaml` are regular tracked blobs read from the
  exact control SHA and agree with topology, role assignments, execution
  profile, evidence identity, dependencies, and exact sorted scenario list.
- Every dependency is `complete`.
- Packet/dependency IDs are unique. Git probes ignore ambient `GIT_*` variables
  and user/system Git configuration.
- The evidence workspace exists outside `/tmp`, neither contains nor is
  contained by any linked worktree or Git administration, and passes a durable
  write/`fsync`/read/remove probe.
- Verification startup holds the manifest lock, requires empty `.reservations`,
  and matches the topology/packet/status manifest digest. The strict canonical
  schema, identities, exact root/tree inventories, every artifact SHA-256, and
  committed candidate `uv.lock` digest all validate; symlinks, hardlinks,
  duplicate keys, non-finite values, extras, and missing artifacts are rejected.

Any mismatch aborts AgentLoop construction. The host reports it to the lead;
after the lead's decision, the host records `blocked`. No worker starts. A model
must not create or repair worktrees, use Git plumbing as a substitute, move
evidence, alter packet/status data, or retry the same missing capability through
another command.

The current repository validates this pre-existing topology but does not expose
one command that provisions campaign worktrees, records lifecycle transitions,
or runs/finalizes evidence. Name the trusted external operator workflow used by
this packet. If none exists, the packet is not ready.

The [harness integrity contract](../harness-integrity.md) is authoritative for
protected paths, completion replacement, the three-failure circuit breaker,
and auto-approve behavior.

Managed runtime tools are ceilings, not packet choices. Active roots have at
most `bash`, `edit`, `glob`, `grep`, `read`, `skill`, `task`, `todo`,
and `write_file`; verification roots have at most `glob`, `grep`, `read`,
`skill`, `task`, and `verify_work`. Managed Task accepts only effective
read-only built-in reviewer/verifier profiles, capped at `bash`, `glob`, `grep`,
`read`, and `skill`, then intersected with any structured manifest.
Project/plugin tools, MCP/connectors, workflows, teams, web tools,
`tool_search`, and `land_work` cannot enter this catalog.
Managed reads are confined to the candidate/control/evidence roots, session
scratchpad, host skill roots, and active prompt files. Host logs, receipts,
runtime state, and unrelated host paths are denied. Strict managed Bash rejects
`background=true`.

## Trusted command plan

List each receipt-authorizing check by its host recipe name and direct argument
array. Do not provide a shell command as execution authority.

| Check ID | `argv` | Executable SHA-256 | Environment attestation path/SHA-256 | `cwd` | Timeout | Purpose |
|---|---|---|---|---|---|---|
| `<check-id>` | `["/opt/vibe-checks/bin/python3.12", "-m", "pytest", "-n0", "<path>"]` | `<64 lowercase hex>` | `</absolute/host-owned/attestation.json>` / `<64 lowercase hex>` | `.` | `<seconds>` | `<criterion>` |

The host executes these arrays with `shell=False`. A shell or `env` cannot be
the executable, and either is rejected behind `uv run`. Pipelines, `set +e`,
command substitution, and trailing-success status masking are not valid trusted
checks. Trusted recipes reject `uv` and pre-commit entrypoints. The host must
pre-provision dependencies and must not substitute another package-manager
installation command; the runtime does not classify every package-manager CLI.
The runner is Linux Bubblewrap-only; Seatbelt is not a
trusted-check backend. It descriptor-validates a pre-provisioned native
executable, verifies its digest, executes a private read-only copy, and rejects
shebang wrappers. It creates an exact-HEAD Git-exported snapshot with no Git
metadata, verifies the separate host-owned environment attestation before and
after execution, disables network, scrubs host credentials/config, uses
disposable writable state, and limits combined stdout/stderr to 1 MiB.
The attestation is a host assertion, not a transitive dependency-tree hash.

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

A criterion may compare one named structured projection to one fully enumerated
expected value; that is one binary equality. Do not join unrelated behavior,
evidence completeness, and quality-gate outcomes in one row. If the projection
is not explicitly named and frozen, split its fields into separate criteria.

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
- The exact sorted scenario list and canonical manifest SHA-256 recorded in both
  packet and status.
- Manifest-lock validation with empty `.reservations`, strict canonical schema,
  exact root/tree inventories, artifact hashes, and committed `uv.lock` digest.

Freeze each scenario's semantics in packet frontmatter and copy the exact parsed
YAML value to the matching status entry:

```yaml
evidence:
  scenarios:
    - IT-00
  scenario_contracts:
    - id: IT-00
      surface: non_ui
      command: ["/opt/maintenance/bin/run-it-00", "--frozen"]
      recorded_environment:
        policy: exact
        values:
          PROFILE: linux-x86_64-python3.12
      required_artifact_types: [result]
      result_schema:
        $schema: https://json-schema.org/draft/2020-12/schema
        type: object
        required: [status]
        properties:
          status: {const: pass}
        additionalProperties: false
      expected_status: pass
      allowed_notes: []
      allowed_gap_notes: []
```

`status.yaml` stores the same list under the packet entry's
`evidence.scenario_contracts`; its IDs also match `required_scenarios`. Commands
are nonempty direct argument arrays, never shell
strings. `recorded_environment.policy` is `exact`, so the manifest must contain
exactly the named values. Artifact types are sorted and unique and must include
`result`; that artifact is inline JSON validated by a self-contained Draft
2020-12 schema no larger than 64 KiB, with no references. Notes and gap notes
are exact ordered allowlists. A passing contract permits neither. A failing
contract must authorize at least one exact failure or gap note.

The evidence directory is durable host state and read-only to model tools. The
evidence operator selects the approved scenario; a host runner writes it. A
scratchpad or `/tmp` artifact cannot satisfy this contract.

## Exact verification commands

Separate commands into:

1. Targeted implementation checks.
2. Required scenario command.
3. Repository quality/fork gates proportional to the packet.
4. Final diff and cleanliness checks.

Use `uv run` for ordinary development/repository commands and preserve required
options such as `-n0`. In a topology-bound worker session, Bash sees the
candidate read-only: use check-only modes such as `ruff check --no-fix` and
`ruff format --check`, disable or redirect test caches, and apply changes only
with bounded file tools. Do not use `--fix` after candidate freeze. For every
receipt-authorizing command, repeat the exact pinned pre-provisioned direct
`argv` entry from the Trusted command plan; it must not use `uv`. Human-readable
shell examples are explanatory only and never override the host recipe.

## Stop conditions

Include all general authority-matrix stops plus packet-specific examples. State
which AC, IT, MSG, path, or invariant is at risk and the smallest lead decision
that could unblock it.

Three consecutive filesystem-confinement, policy-denial, or sandbox-startup
failures in the same class end the turn as host `BLOCKED`. Do not specify a
fourth workaround.

## Rollback

Name every repository path created or modified and the evidence paths that may
be retained. Rollback must not revert other packets, regenerate baselines, or
alter unrelated work.

## Completion report

Before freeze, the worker reports the implementation facts available in the
active session. After lead approval, the host creates the commits, starts the
verification session, and augments the record with host-owned identities and
structured verification state. The combined report contains:

- Packet ID, then the host-recorded candidate SHA after freeze.
- Exact changed paths and why each was allowed.
- Commands and exit codes.
- Acceptance-criterion table with pass/fail and evidence links.
- Message/snapshot/performance/fork-metric deltas, including explicit “none.”
- Evidence manifest and scenario paths, exact sorted scenario IDs, and the
  host-recorded canonical `manifest_sha256`.
- Remaining gaps, skips, flakes, denied tools, or blockers.
- Confirmation that candidate state is clean and frozen.
- Host-recorded active and verification topology identities, including the
  active, initial-verification, and final-verification control SHAs, and
  confirmation that model tools never modified control,
  evidence, Git administration, host logs, or receipts.
- Host-recorded task `completed`, structured `outcome`, and receipt state. Raw
  verifier prose is diagnostic material and cannot override those fields.
- Confirmation that no landing action was taken.
- If delivery or landing was separately authorized, proof that it used the
  exact authorized object ID and compare-and-swap ref update.

The lead alone authorizes final packet state; the trusted host records it.
