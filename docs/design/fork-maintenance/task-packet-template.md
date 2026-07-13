# Fork Maintenance Task-Packet Template

Copy this file to `packets/I<iteration>-P<sequence>-<slug>.md`, replace every
angle-bracket placeholder, remove instructional text, and keep the section
order. A packet remains `draft` while any field required for `ready` is `null`,
empty, ambiguous, or unresolved. `candidate_sha` is the intentional exception:
it stays `null` through `ready`/`active` and becomes required when the packet
enters `verification`.

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
  scenarios: []
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
| Campaign preservation baseline | May be unset during Iteration 0 bootstrap | May be unset only for authorized Iteration 0 packets | Required before Iteration 0 exits or later iterations begin |

`baseline_sha` and `candidate_sha` are written in a separate committed control
worktree, not in the implementation candidate. The assignment supplies the
immutable control commit SHA externally because a file cannot contain the hash
of its own commit.

# <Packet ID>: <Title>

Execution state is read only from this packet's frontmatter and the matching
`status.yaml` entry at the assigned clean `CONTROL_SHA`; do not duplicate it in
prose.

## Outcome

State one externally verifiable result. Do not describe a broad aspiration or
combine multiple rollback boundaries.

## Why this packet exists

Tie the result to a roadmap risk, dependency, or preservation contract. Explain
why it is ordered here and what later work it unblocks.

## Definition of Ready

The campaign lead checks every item before changing the packet to `ready`:

- [ ] `baseline_sha` and `upstream_sha` are full, resolvable 40-character commits.
- [ ] `baseline_sha` is the packet's clean starting commit. If Iteration 0 is
      still bootstrapping, it is not mislabeled as the final campaign baseline.
- [ ] `owner`, `reviewer`, `verifier`, `evidence_operator`, `worktree`, `branch`,
      and `execution_profile` are assigned.
- [ ] The ready packet/status are committed in a clean dedicated control
      worktree; the assignment will bind that commit as `CONTROL_SHA`.
- [ ] The assigned worktree is clean and contains no unrelated work.
- [ ] Every dependency is `complete` in `status.yaml`.
- [ ] `VIBE_EVIDENCE_WORKSPACE` is absolute and outside the repository, Git
      common directory, and every linked worktree.
- [ ] `KILROY_RUN_ID` is unique and the scenario directories are not shared by
      another active writer.
- [ ] `runner_id` is a stable non-secret machine/runner label.
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
CONTROL_WORKTREE=<absolute clean control worktree>
CONTROL_SHA=<immutable control commit supplied by the lead>
REPO_ROOT=<absolute candidate worktree root>
BASELINE_SHA=<40-hex>
UPSTREAM_SHA=<40-hex>
VIBE_EVIDENCE_WORKSPACE=<absolute external directory>
KILROY_RUN_ID=<unique run identifier>
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
```

## Preflight

Run exactly, without editing first:

```bash
test -n "$CONTROL_WORKTREE"
test -n "$CONTROL_SHA"
test "$(uv run git -C "$CONTROL_WORKTREE" status --short)" = ""
test "$(uv run git -C "$CONTROL_WORKTREE" rev-parse HEAD)" = "$CONTROL_SHA"
test "$(uv run git rev-parse --show-toplevel)" = "$REPO_ROOT"
test "$(GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_OPTIONAL_LOCKS=0 uv run git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)" = ""
uv run git rev-parse HEAD
uv run git rev-parse "$BASELINE_SHA^{commit}"
uv run git rev-parse "$UPSTREAM_SHA^{commit}"
uv run git worktree list --porcelain
```

Define the required empty status and SHA relationships. Add packet-specific
checks for fixtures, tools, network isolation, evidence-root exclusion, or
dependencies. State the exact blocker behavior for any mismatch.

### Canonical control-metadata validator

Every packet preflight runs this read-only validator after the basic Git checks.
Set `PACKET_ID` and `PACKET_RELATIVE_PATH` to the assigned control packet. This
binds shell inputs, roles, dependencies, and candidate worktree identity to the
exact clean `CONTROL_SHA` instead of trusting copied values.

```bash
PACKET_ID=<packet id> \
PACKET_RELATIVE_PATH=<docs/design/fork-maintenance/packets/file.md> \
EXPECTED_PACKET_STATE=ready \
CONTROL_WORKTREE="$CONTROL_WORKTREE" \
CONTROL_SHA="$CONTROL_SHA" \
BASELINE_SHA="$BASELINE_SHA" \
EXPECTED_CANDIDATE_SHA="" \
OBSERVED_CANDIDATE_SHA="$(uv run git rev-parse HEAD)" \
UPSTREAM_SHA="$UPSTREAM_SHA" \
REPO_ROOT="$REPO_ROOT" \
CANDIDATE_BRANCH="$(uv run git branch --show-current)" \
VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
KILROY_RUN_ID="$KILROY_RUN_ID" \
RUNNER_ID="$RUNNER_ID" \
uv run python - <<'PY'
from pathlib import Path
import os
import subprocess

import yaml

from vibe.core.utils.io import read_safe


control = Path(os.environ["CONTROL_WORKTREE"])
git_environment = os.environ.copy()
git_environment.update(
    GIT_CONFIG_NOSYSTEM="1",
    GIT_CONFIG_GLOBAL="/dev/null",
    GIT_OPTIONAL_LOCKS="0",
)


def control_git(*arguments: str) -> str:
    result = subprocess.run(
        ["uv", "run", "git", "-C", str(control), *arguments],
        check=True,
        capture_output=True,
        env=git_environment,
        text=True,
    )
    return result.stdout


assert control_git(
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
    "--ignore-submodules=none",
) == "", "control worktree is dirty"
control_sha = os.environ["CONTROL_SHA"]
assert control_git("rev-parse", "HEAD").strip() == control_sha, "control HEAD moved"
assert control_git("rev-parse", f"{control_sha}^{{commit}}").strip() == control_sha
packet_text = read_safe(control / os.environ["PACKET_RELATIVE_PATH"]).text
parts = packet_text.split("---", 2)
assert len(parts) == 3 and not parts[0].strip(), "invalid packet frontmatter"
packet = yaml.safe_load(parts[1])
status = yaml.safe_load(
    read_safe(control / "docs/design/fork-maintenance/status.yaml").text
)
entry = next(item for item in status["packets"] if item["id"] == os.environ["PACKET_ID"])
assert packet["id"] == entry["id"] == os.environ["PACKET_ID"]
assert packet["state"] == entry["state"] == os.environ["EXPECTED_PACKET_STATE"]
for field in (
    "depends_on",
    "owner",
    "reviewer",
    "verifier",
    "evidence_operator",
    "baseline_sha",
    "candidate_sha",
    "upstream_sha",
    "worktree",
    "branch",
    "execution_profile",
):
    assert packet[field] == entry[field], field
for field in ("workspace", "run_id", "runner_id"):
    assert packet["evidence"][field] == entry["evidence"][field], field
if os.environ["EXPECTED_PACKET_STATE"] == "ready":
    assert packet["candidate_sha"] is None
    assert os.environ["OBSERVED_CANDIDATE_SHA"] == packet["baseline_sha"]
else:
    assert os.environ["EXPECTED_PACKET_STATE"] == "verification"
    assert packet["candidate_sha"] == os.environ["EXPECTED_CANDIDATE_SHA"]
    assert os.environ["OBSERVED_CANDIDATE_SHA"] == packet["candidate_sha"]
assert packet["baseline_sha"] == os.environ["BASELINE_SHA"]
assert packet["upstream_sha"] == os.environ["UPSTREAM_SHA"]
assert packet["worktree"] == os.environ["REPO_ROOT"]
assert packet["branch"] == os.environ["CANDIDATE_BRANCH"]
assert packet["evidence"]["workspace"] == os.environ["VIBE_EVIDENCE_WORKSPACE"]
assert packet["evidence"]["run_id"] == os.environ["KILROY_RUN_ID"]
assert packet["evidence"]["runner_id"] == os.environ["RUNNER_ID"]
for role in ("owner", "reviewer", "verifier", "evidence_operator"):
    assert isinstance(packet[role], str) and packet[role].strip(), role
assert isinstance(packet["execution_profile"], str) and packet["execution_profile"].strip()
states = {
    item["id"]: item["state"]
    for section in ("packets", "required_future_packets")
    for item in status.get(section, [])
}
for dependency in packet["depends_on"]:
    assert states.get(dependency) == "complete", dependency
PY
```

Any assertion failure requests `<EXPECTED_PACKET_STATE> -> blocked` without
candidate edits, where the expected state is `ready` or `verification`. Do not
edit the packet, status, or shell values to make the validator pass; the lead
must issue a new clean control commit and assignment.

After the lead commits the candidate and a new control commit records
`state: verification`, the evidence operator reruns the same block with
`CONTROL_SHA` set to that newly assigned immutable control commit,
`EXPECTED_PACKET_STATE=verification` and
`EXPECTED_CANDIDATE_SHA="$CANDIDATE_SHA"`; candidate `HEAD` must equal that SHA.

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
- Control SHA used and confirmation that the candidate never modified the
  control files.
- Confirmation that no landing action was taken.

The lead, not the worker, records final packet state.
