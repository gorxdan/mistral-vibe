# Harness integrity contract

Status: Governing execution policy for managed maintenance work

This contract applies when `trusted_verification_recipe.execution_topology` is
configured. It separates campaign decisions from execution authority. The
campaign lead decides what should happen. The host provisions the resources,
validates their identity, performs control-plane writes, and records receipts.
Model prose and model-run preflight commands are never execution authority.

## Authority boundary

The recipe may come from host-controlled user configuration, a `VIBE_`
environment setting, or a programmatic `VibeConfig` initialization. Project
`.vibe/config.toml` files are never recipe authority: the loader removes
`trusted_verification_recipe` case-insensitively from every project layer. A
bound recipe also forces `verification_subsystem = true`, so project config
cannot disable its host checks.

The host-configured recipe is frozen when the root `AgentLoop` is created. Its
`execution_topology` must name the exact packet, control commit, candidate
worktree, candidate identity, and evidence location for that session. Reloading
configuration cannot change those values in a running session or replace the
recipe inherited by a managed reviewer or verifier.

The packet and `status.yaml` describe the assignment. They do not prove that
the assignment exists. A copied command block, environment variable, Git ref,
directory containing a `.git` file, or agent assertion cannot substitute for a
host-validated topology.

The host owns these actions:

- Create and remove physical control and candidate worktrees.
- Create candidate and control commits.
- Record packet lifecycle transitions and their control commits.
- Run trusted checks and persist verification receipts.
- Write the durable evidence workspace through approved runners.
- Merge, land, push, or open a pull request after the required human decision.

Model tools may edit only assigned candidate paths. They cannot mutate the
control worktree, durable evidence workspace, Git administration, host logs, or
verification receipt storage. This remains true in auto-approve mode.

### Implemented boundary and external host dependency

The repository currently implements the fail-closed model boundary: topology
validation at `AgentLoop` startup, managed tool ceilings and path policy,
trusted-check execution, receipt validation, and exact-object delivery/landing
primitives. It does **not** currently expose one campaign-host command or
service that creates the physical campaign worktrees, writes packet/status
lifecycle transitions, installs an attested check environment, or runs and
finalizes an evidence scenario.

Those operations therefore require a trusted human-operated or external host
workflow. The operator must create the state before starting Vibe; the runtime
validates it but does not synthesize missing state. If that workflow is absent,
the packet is blocked before a model starts. Commands or reports copied into a
packet are documentation, not a substitute implementation and not authority to
let a worker perform control-plane operations.

## Required topology

The host supplies the topology under the frozen verification recipe:

```toml
[trusted_verification_recipe.execution_topology]
packet_id = "I00-P01"
packet_path = "docs/design/fork-maintenance/packets/I00-P01-evidence-runner.md"
status_path = "docs/design/fork-maintenance/status.yaml"
state = "active"
control_worktree = "/absolute/durable/worktrees/fork-maintenance-control"
control_sha = "1111111111111111111111111111111111111111"
candidate_worktree = "/absolute/durable/worktrees/i00-p01-candidate"
candidate_branch = "maintenance/i00-p01"
baseline_sha = "2222222222222222222222222222222222222222"
upstream_sha = "3333333333333333333333333333333333333333"
evidence_workspace = "/absolute/durable/maintenance-evidence"
run_id = "i00-p01.20260713.1"
runner_id = "linux-x86_64-python3.12"
max_turns = 80
max_session_tokens = 2000000
```

`status_path` may be omitted only when the campaign uses the default
`docs/design/fork-maintenance/status.yaml`.

### Active and verification identities

Managed topology has exactly two executable session states. `ready`, `blocked`,
and `complete` are control-plane states and cannot start a managed AgentLoop.

| Field | Active implementation session | Verification session |
|---|---|---|
| `state` | Exactly `active` | Exactly `verification` |
| `candidate_sha` | Absent or `null`; predeclaring it is invalid | Required full 40-character SHA |
| `evidence_manifest_sha256` | Absent or `null`; predeclaring it is invalid | Required full lowercase SHA-256 |
| Candidate `HEAD` | Exactly `baseline_sha` at startup | Exactly `candidate_sha` at startup |
| Packet and status state | Both `active` | Both `verification` |
| Candidate cleanliness | Clean before the first model turn | Clean and frozen before the first verifier turn |
| Permitted model writes | Packet-allowed candidate paths | None; verification is read-only |

All other topology fields are required in both states: `packet_id`,
`packet_path`, `status_path`, `control_worktree`, `control_sha`,
`candidate_worktree`, `candidate_branch`, `baseline_sha`, `upstream_sha`,
`evidence_workspace`, `run_id`, and `runner_id`.

`max_turns` and `max_session_tokens` are host ceilings in both states. They
default to 80 turns and 2,000,000 session tokens. A caller may tighten but
cannot raise either limit for a topology-bound session.

The active and verification configurations belong to different sessions. After
implementation, the host ends the active session, creates the candidate commit,
and records an initial `state: verification` control commit with the exact
candidate, run, runner, and sorted scenario assignment. No verification
AgentLoop starts from that intermediate commit. The approved host runner
finalizes the evidence run, releases its manifest lock, and removes every
reservation. The host hashes the canonical manifest and records that digest as
`evidence.manifest_sha256` in packet and status metadata in a second, final
verification control commit. Only a recipe naming that second `control_sha`,
the same `evidence_manifest_sha256`, and the frozen candidate may start the
verification AgentLoop. A model must not edit configuration or reuse the active
session to simulate either transition. In the active state the candidate
remains writable only within the packet allowlist, but a tool-free completion
claim without current receipt authority is still replaced by host status.

## Startup validation

Root `AgentLoop` construction fails before a model turn unless the host verifies
all of the following:

1. The session starts inside the assigned candidate worktree.
2. Control and candidate are distinct directories registered as physical Git
   worktrees of the same repository.
3. Both worktrees are clean, including untracked files and submodule state.
4. Control `HEAD` equals `control_sha`; the baseline and upstream SHAs resolve
   to those exact full commits; candidate branch and `HEAD` match the configured
   state. Git probes discard ambient `GIT_*` variables and disable user/system
   Git configuration and optional locks.
5. Packet frontmatter and `status.yaml` are regular tracked blobs read from the
   exact `control_sha`, not the working tree. They agree with the topology for state,
   baseline, candidate, upstream, branch, candidate path, evidence identity,
   roles, execution profile, and dependencies. Packet `evidence.scenarios` and
   status `required_scenarios` are the same nonempty, sorted, unique list.
6. Packet IDs and dependency IDs are unique, and every dependency is
   `complete`.
7. The evidence workspace already exists and neither contains nor is contained
   by any linked worktree or the Git common directory.
8. The evidence workspace is not beneath `/tmp`, `/run`, `/dev/shm`, another
   configured system-temporary root, or a Linux `tmpfs`/`ramfs` mount.
9. Active startup performs a host write, file `fsync`, read-back, unlink, and
   parent-directory `fsync` probe. Verification startup is non-mutating and
   rechecks the path, mount, and frozen evidence identity without repeating the
   probe.
10. Verification topology supplies `evidence_manifest_sha256`, and both control
    documents record it as `evidence.manifest_sha256`. Under the run's manifest
    lock, the host requires an empty `.reservations` directory, canonical strict
    schema-v1 JSON, the exact sorted control scenario contracts, exact
    baseline/candidate/upstream/runner identities, and the `uv.lock` digest read
    from the committed candidate tree. The run root and each scenario directory
    contain exactly the declared entries; every artifact is a descriptor-safe,
    single-link regular file whose streamed SHA-256 matches the manifest.

Failure aborts session construction. The harness reports the exact mismatch;
the lead decides whether to block and the host records the resulting state. No
model is started to repair, emulate, or work around missing topology.

Packet `evidence.scenario_contracts` and the status entry's
`evidence.scenario_contracts` are parsed-value identical, sorted by unique scenario ID,
and have exactly the IDs assigned by packet `evidence.scenarios` and status
`required_scenarios`. Each contract freezes the surface, direct argument array,
exact recorded environment, sorted required artifact types (including
`result`), self-contained Draft 2020-12 result schema, expected status, and
ordered note/gap-note allowlists. Passing scenarios authorize no failure or gap
notes. Failing scenarios must authorize at least one exact note. The manifest is
validated against these semantics, not merely against the scenario names.

## Trusted checks

Receipt-authorizing checks are configured by the host as direct argument
arrays:

```toml
[[trusted_verification_recipe.checks]]
name = "focused-tests"
argv = ["/opt/vibe-checks/bin/python3.12", "-m", "pytest", "-n0", "tests/tools"]
cwd = "."
timeout_seconds = 600
executable_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
environment_attestation_path = "/opt/vibe-checks/environment.json"
environment_attestation_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
required_output_patterns = ["passed"]
forbidden_output_patterns = ["FAILED"]
```

The runner executes `argv` with `shell=False`. Configuration rejects a shell or
`env` as the executable and rejects shells or `env` selected behind `uv run`.
Pipelines, `set +e`, command substitution, and a trailing successful command
therefore cannot mask a failed check. Packet code blocks may explain a check,
but only the frozen host recipe can authorize a receipt.

Help, version, collection-only, list, dry-run, no-run, failure-masking, and
structurally empty-selection modes are not check evidence. `dotnet test` must
report a positive, non-conflicting executed-test count even when the recipe
omits custom output assertions. Other runners may bind required or forbidden
output patterns and a named `(?P<count>...)` test-count pattern with
`minimum_test_count`; every observed count must agree and meet the minimum.
Output regexes run in a killably time-bounded isolated worker. Unknown runners
fail closed unless `custom_runner = true` is paired with required output and a
positive count contract. Every trusted check pins the resolved,
pre-provisioned host executable and a separate host-owned
environment-attestation file. The runner descriptor-validates and hashes the
source executable, copies it to a private read-only path, and executes only the
copy. The sandbox preserves the configured executable path as `argv[0]` so a
copied native interpreter retains its expected virtual-environment prefix
discovery. Shebang wrappers are rejected; recipes name a native interpreter and
pass `-m <module>` or a script path as arguments. The source, copy, and
attestation are rechecked after execution. Candidate-owned executables and
`uv` or `pre-commit` bootstrap entrypoints are not authority; all check
dependencies must already be provisioned in host-owned runtime roots. Combined
stdout and stderr are capped at 1 MiB; exceeding the cap terminates the process
tree and produces failed evidence.

The runtime does not attempt to classify every package-manager CLI. The host
must not configure another installation command as a trusted check; offline
execution and read-only runtime roots do not prevent writes into the disposable
check directories.

The executable copy does not freeze its dynamic loader, shared libraries,
language packages, or other runtime roots. Those remain host-owned read-only
inputs. The environment-attestation file is a host assertion about their
provisioned state, not a transitive dependency-tree hash; the host must make the
roots immutable or exclude concurrent writers for an authority-bearing run.
Native programs that derive resources from their physical executable location,
rather than `argv[0]`, require a separately attested deployment layout.

Trusted checks do not inherit model Bash settings. Each check requires Linux
Bubblewrap; Seatbelt, `unshare`, and unsandboxed fallback cannot authorize a
receipt. The host first creates a detached, read-only source snapshot at the
exact candidate `HEAD` and tree, then gives each check a scrubbed offline
environment with disposable home, temp, and caches. Only its per-check run
directory is writable. The snapshot contains no `.git`, refs, history, or live
candidate/Git-common mount, and its content hash is checked again after
execution. The runner also compares the original repository state before
and after all checks; a dirty, out-of-scope, moved, or changed candidate fails
the receipt. See [Shell sandbox](sandbox.md) for the distinct model-Bash and
trusted-check backend contracts.

## Managed capability ceiling

A topology-bound session builds its tool manager from a canonical in-repository
catalog. The configured `enabled_tools`, `disabled_tools`, project tools, MCP
servers, connectors, workflow tools, and future discovered tools cannot widen
this catalog. Tool-specific permission and availability checks may narrow it.
LSP is intentionally absent because project-selected language-server
executables are not host-pinned.

The maximum root catalogs are:

- Active: `bash`, `edit`, `glob`, `grep`, `read`, `skill`, `task`,
  `todo`, and `write_file`.
- Verification: `glob`, `grep`, `read`, `skill`, `task`, and
  `verify_work`.

Managed `task` calls may select only the built-in `reviewer` and `verifier`
profiles. A named profile is rejected if its effective profile is write-capable
or unjailed. These subagents run in process, inherit the frozen recipe and
topology, and have at most `bash`, `glob`, `grep`, `read`, and `skill`.
A structured task manifest intersects that ceiling and may add only the
host-scoped `task_checks` tool needed by its contract; it cannot introduce a
model-selected capability. Managed subagents do not receive `task` and cannot
delegate recursively.

The `skill` entry exposes the canonical Skill tool and applicable skill text. It
does not restore plugin-provided tools, MCP/connectors, web tools, workflow/team
launchers, `tool_search`, or `land_work`.

Managed read tools resolve paths only within the assigned candidate, control
worktree, evidence workspace, session scratchpad, active host prompt files, and
host skill roots. Host logs, receipt storage, and runtime state remain denied
even when nested under an otherwise readable root. Strict model Bash also sees
the exact assigned roots, configured executable/runtime directories, and the
minimal operating-system roots needed to start those programs; unrelated host
top-level trees are masked. It is always foreground: `background=true` is
rejected before process launch so a child cannot outlive the bound sandbox or
verification attempt.

In an active managed session, candidate mutations go through `edit` or
`write_file`; Bash receives a read-only candidate bind. Shell checks must use
check-only modes and disposable caches (for example `ruff check --no-fix`,
`ruff format --check`, `PYTHONDONTWRITEBYTECODE=1`, and a disabled or redirected
pytest cache). A formatter or fixer that must rewrite files is a lead-approved
host step, or its changes are applied explicitly through the bounded file tools
before handoff.

Managed `edit` and `write_file` capture the assigned root, ancestor directory,
and target identities before permission is requested. After approval they walk
the captured path with no-follow descriptors and publish through the captured
parent directory. `edit` reads from the pinned target descriptor and preserves
its encoding, newline convention, and mode; `write_file` uses a no-replace
publish. Identity changes fail closed. This protects against ordinary symlink
and rename swaps, but it cannot make an external writer cooperative: a process
with host access can still mutate the same inode or race the final basename.
Managed campaigns must exclude other candidate writers while a file tool call
is pending.

## Verifier and completion authority

A verifier report has two layers:

1. Raw subagent prose, including a literal `VERDICT: PASS` line.
2. Host-observed execution fields: `completed`, structured `outcome`, current
   candidate/base identity, and the required receipt.

The second layer always wins. A raw PASS is non-authoritative when execution did
not complete, the outcome did not succeed, a tool was denied or skipped, the
candidate or base moved, the attempt was superseded, or a configured receipt is
missing, stale, or invalid.

The host records each attempt as `PENDING`, `PASS`, `FAIL`, `PARTIAL`, or
`INVALID`. A terminal disposition is accepted once, from `PENDING`; late
callbacks cannot replace it. A conflicting late non-PASS result after PASS
supersedes the authorization generation instead of rewriting that terminal
result. Receipt authority records the attempt generation, and a new attempt or
a recorded non-PASS result clears the prior reference. Receipt persistence and
validation recheck that generation and the exact receipt-store/reference
identity after storage I/O. Candidate delivery and landing reserve current
authorization while they perform their exact mutation; revocation is applied
when the reservation releases and cannot authorize a second operation.
Until current authorization exists, AgentLoop buffers a tool-free completion
before it reaches the user. Contradictory model text is discarded and replaced
with `HOST VERIFICATION STATUS: IN_PROGRESS`, `PARTIAL`, or `BLOCKED` plus the
host diagnostic. Remediation tool calls may continue, but the model cannot
describe the work as verified, complete, ready for acceptance, or safe to land.

With a trusted recipe, a current verifier PASS is still incomplete until
`verify_work` creates a valid receipt. Non-trivial landing always requires that
receipt; a legacy state-recorded PASS is diagnostic only. Without a recipe,
only the locally validated documentation-only trivial waiver remains. Pasted
reports never authorize landing.

In a managed verification session, no-argument `verify_work` uses the frozen
topology and recipe. It remains available when no legacy
`worktree_manager.active` record exists; the managed topology is the candidate
identity. Supplying model-authored arguments cannot override the frozen recipe.
The recipe configuration hash includes the full execution topology, including
`evidence_manifest_sha256`, so a receipt for a different manifest cannot be
reused. If `verify_work` is cancelled after its worker starts, the coroutine
waits for the bounded trusted-check worker to finish before propagating
cancellation. Its unpublished receipt is discarded, so no check process or
receipt write continues after tool/session cleanup believes the call ended.

Verified workflow delivery and host landing never merge a mutable branch name.
They bind exact parent/base and candidate commit IDs, recheck cleanliness and
verified fingerprints immediately before mutation, create the exact
fast-forward or merge result from those IDs, and update the checked-out target
ref with a compare-and-swap transaction. A moved target, moved candidate, or
post-verifier candidate change preserves or rejects the candidate instead of
landing a different tree. The documentation-only trivial path also freezes the
exact candidate SHA before authorization.

The ref compare-and-swap is exact, but checked-out worktree materialization is
not an atomic multi-file filesystem transaction. The merge lock serializes Vibe
landing operations only. External editors and Git processes must remain idle
during the approved landing window; any reported failure requires inspection
before retrying.

## Capability circuit breaker

The harness counts consecutive failures in one capability class. The protected
classes include filesystem confinement, hard policy denial, and sandbox
startup. After three consecutive failures in the same class, middleware ends
the turn before another model or tool call and emits:

```text
HOST CAPABILITY STATUS: BLOCKED
```

The agent must report the blocker. It may not try a fourth path, substitute Git
plumbing for a physical worktree, delete host state, or ask the user to clean up
administrative files. A successful capability result resets the consecutive
window; failures from different classes do not combine.

## Auto-approve and the safety judge

Auto-approve changes how permitted `ASK` calls are handled. It does not weaken
hard policy:

- `NEVER` remains denied.
- Control/evidence/Git/log/receipt protections remain denied.
- A configured safety judge still evaluates `ASK` calls.
- Judge approval may execute the call.
- Judge deferral remains a denial in auto-approve mode; auto-approve cannot
  convert it into approval.
- Judge timeout, refusal, spend rejection, API failure, or invalid output fails
  closed as a deferral.

Auto-approve also forces strict model-process control for Bash: Bubblewrap or
Seatbelt is required, network is disabled, the environment is scrubbed, and
tool caches are disposable. In an ordinary session, the current workspace,
scratchpad, configured `write_dirs`, and normal Git commit path remain writable.
That is deliberately not the managed maintenance boundary. When execution
topology is present, its smaller tool catalog and protected-path policy apply,
and model Bash has no writable candidate bind.

## Operator sequence

For a managed packet, use this order:

1. The lead freezes the packet in `ready` and asks the host to provision the
   topology. No worker session exists yet.
2. The host creates the physical worktrees and durable evidence directory,
   validates the ready assignment, and reports any mismatch to the lead.
3. The lead authorizes `ready -> active`; the host records and commits the
   transition and starts an active session with `candidate_sha` absent.
4. AgentLoop revalidates the full active topology before the worker receives a
   turn. The worker edits only allowed candidate paths.
5. The worker reports implementation results without committing or changing
   campaign state. Its final line starts with `READY_FOR_HOST_FREEZE:`,
   `BLOCKED:`, or `IN_PROGRESS:`; the host labels the quoted text as untrusted
   operator context.
6. The host ends the active session. After lead approval, it creates the
   candidate commit and an initial verification-state control commit containing
   the exact `candidate_sha` and sorted scenario assignment.
7. No model starts. The approved host runner writes and finalizes the assigned
   evidence under the manifest lock, with no remaining reservations.
8. The host hashes the canonical manifest, records `manifest_sha256` in both
   control documents, and creates the second, final verification control
   commit.
9. The host starts a fresh verification AgentLoop whose topology names the
   final control SHA, exact candidate SHA, and `evidence_manifest_sha256`.
   Reviewer and verifier remain read-only and read the durable evidence path,
   never `/tmp` or a session scratchpad.
10. A current verifier PASS allows `verify_work` to run the frozen checks. Only a
   valid receipt can satisfy managed completion and landing gates.
11. The lead decides whether to accept or land. The host performs the selected
    commit, transition, or landing operation.

Any missing host capability ends this sequence as blocked. The model does not
redesign the sequence while executing it.
