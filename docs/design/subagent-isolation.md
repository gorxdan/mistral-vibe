# Design Spec — default worktree isolation for write-capable agents

**Effort:** M  |  **Verdict:** sound_with_fixes (after adversarial review)  |  **Feasible:** True  |  **Depends on:** none

## Goal

No write-capable agent shares a working tree with another agent or the live
checkout. Destructive commands (`rm -rf`, `git reset --hard`, an `edit` to the
wrong file) are scoped to a throwaway worktree whose branch is merged back only
on success. Read-only agents (incl. the read-jailed `reviewer`/`debugger`/
`security`) stay cheap and in-process. Today only some surfaces do this; the
gap is `task()` for write profiles.

## Current state

Two worktree mechanisms exist and serve different concurrency models.

**Mechanism A — `WorktreeManager`** (`vibe/core/worktree/manager.py`): one
worktree per *process*, via `os.chdir`. Nested-enter guard raises
(`manager.py:121`). Subagents inherit the process cwd and never call `enter()`
(`task.py:133`, `runtime.py:1054`). Activated at entrypoints:
- interactive CLI via `--worktree` or `[worktree] mode="on"` (`cli.py:343`)
- programmatic `-p` via `mode="auto-by-entrypoint"` default-on
  (`programmatic.py:87`)
- team teammates automatically, because they ARE `vibe -p` subprocesses
  (`teams/manager.py:211`)

**Mechanism B — `EphemeralWorktree`** (`vibe/core/worktree/ephemeral.py`): one
worktree per *agent*, no chdir, the agent runs as a `vibe -p` subprocess with
`cwd=wt.path`. Concurrent-safe by construction. Reference implementation is the
workflow isolated executor `_default_isolated_executor`
(`runtime.py:1379-1472`): create → spawn `vibe -p` in worktree with
`VIBE_WORKFLOW_EMIT_STATS=1` → capture stdout → `--ff-only` deliver
(`deliver_ephemeral_worktree`) → remove unless changed (`keep_if_changed`).
**Two caveats discovered in review:** (i) the executor hardcodes `--agent
auto-approve` (`runtime.py:1412-1413`) — the requested profile is not threaded
into the subprocess today; (ii) it is otherwise self-contained (zero `self.`
references in the method body), so it is extractable once (i) is fixed.

**Per-surface isolation today:**

| Surface | Runs as | Worktree | Controlled by |
|---|---|---|---|
| Interactive main | process | off by default | `--worktree` / `mode="on"` (Mech A) |
| Programmatic main (`-p`) | process | on | `auto-by-entrypoint` (Mech A) |
| Team teammate | `vibe -p` subprocess | on | automatic via `-p` entrypoint (Mech A) |
| `task()` subagent | **in-process** | inherits parent | nothing (`task.py:133`) |
| Workflow agent (default) | **in-process** | inherits parent | nothing (`runtime.py:1054`) |
| Workflow agent (`isolation="worktree"`) | `vibe -p` subprocess | on (own) | opt-in per `agent()` (Mech B) |

**Existing rule and its gap.** `_validate_workflow_profile` (`runtime.py:575`)
requires `isolation="worktree"` when a profile has *no* `enabled_tools`
allowlist. This catches `worker`/`auto-approve` but **misses `editor`**
(`models.py:337` has `enabled_tools=["read","grep","write_file","edit"]` — an
allowlist, yet it writes files). So even the workflow rule under-isolates today.
The generalized predicate below closes this.

**Read-jailed profiles are safe in-process by enforcement, not convention.**
`reviewer`/`debugger`/`security` carry bash in their allowlist, but `_REVIEW_BASH_DENYLIST` (`models.py:149-171`) hard-denies `rm`, `mv`, `git reset`,
`git checkout`, `git worktree`, etc. via `ToolPermission.NEVER` enforced per
command node. Their bash is read-only. They do not need isolation.

## Target design

Lift Mechanism B (the workflow isolated executor) into `task()` and make it the
default for write-capable profiles. Keep read-only and read-jailed profiles
in-process. Three pieces:

1. A predicate `profile_requires_isolation(profile) -> bool` (in
   `agents/models.py`) returns True when the profile can write destructively:
   **no `enabled_tools` allowlist** (full tools) **OR** the allowlist contains a
   write-capable tool (`write_file`, `edit`, or un-jailed `bash`). This isolates
   `worker`/`auto-approve`/`editor`; keeps `explore`/`research`/`planner` and the
   read-jailed `reviewer`/`debugger`/`security` in-process. (A constant
   `_WRITE_TOOLS = {"write_file", "edit"}` plus the bash-jailed flag drive it.)
2. `task()` calls it. If True, run the subagent via the isolated path
   (Mechanism B); if False, today's in-process path (unchanged).
3. Extract `_default_isolated_executor` into a module-level
   `run_isolated_agent(prompt, agent, *, label, max_turns, deliver) -> IsolatedResult`
   so both `task()` and workflows call it. **Thread the real `agent` into the
   `vibe -p` cmd** — the current `--agent auto-approve` hardcode
   (`runtime.py:1412-1413`) is a latent bug that this extraction fixes. The
   `VIBE_WORKFLOW_EMIT_STATS`/`_parse_stats` mechanism (`runtime.py:388`,
   `programmatic.py:161`) is reused unchanged.

### Why not "always isolate everything"

Cost. In-process subagents (`explore`, `research`, `reviewer`, `debugger`) are
fanned out broadly — a workflow may spawn 16 of them. Each subprocess+worktree
is a full `vibe -p` cold start plus a worktree create/remove (git I/O). (No
startup benchmark exists in the repo to ground a number — see Risks — but the
direction is clear: Python import + config load + provider init per spawn.) A
read fan-out paying that tax for isolation that buys nothing (read-only agents
cannot overwrite each other) is a poor trade. The profile-gated split keeps
reads fast and isolates only what can destroy work.

### Sequential vs concurrent (where the safety value lives)

The user's stated problem — agents "overwriting each other's work" — is a
*concurrency* problem. Two cases:
- **Concurrent writers** (workflow `parallel`/`pipeline` fanning out `worker`s,
  or a team). Each needs its own tree. **Workflows already enforce this**
  (`runtime.py:575` + Mech B); teams already isolate (each is a `-p` subprocess).
  This design's predicate fix (isolate `editor` too) closes the remaining hole.
- **Sequential `task()` delegation.** A single lead delegating one `task()` at a
  time has no concurrent writer to race. Isolating it buys *less* safety than
  the concurrent case, but still pays the latency and the behavior changes below.
  The honest framing: `task()` isolation is defense-in-depth against a
  destructive subagent (one bad `rm -rf` no longer takes the live tree), not a
  concurrency fix. Worth doing for that property alone, with eyes open on cost.

## Integration points

- `vibe/core/agents/models.py` — add `profile_requires_isolation(profile)` plus
  `_WRITE_TOOLS = {"write_file", "edit"}`. No model change.
- `vibe/core/workflows/runtime.py:1379-1472` — extract into module-level
  `run_isolated_agent(...)`; **replace the `--agent auto-approve` hardcode with
  the passed `agent`**. `WorkflowRuntime` keeps a thin wrapper that threads its
  `isolated_executor` test seam and its `agent` through.
- `vibe/core/tools/builtins/task.py` — branch in `run()` on
  `profile_requires_isolation(agent_profile)`. Isolated branch builds a
  `TaskResult` from subprocess stdout **plus a worktree/branch handle** (see
  Delivery below). In-process branch unchanged. Update the description
  ("runs in-memory" → conditional).
- `vibe/core/worktree/ephemeral.py` — no change; helpers already
  concurrency-safe and self-contained.

## Config

- `task.isolation: "off" | "auto" | "always"` (default `"auto"`). `auto` =
  profile-gated. `always` = isolate even read-only profiles. `off` = today's
  behavior. New `TaskToolConfig` field.
- Reuse `[worktree]` config (`base_dir`, `branch_prefix`, `merge`, `cleanup`,
  `carry_dirty`, `carry_ignored`) — no new paths.

## Delivery (the behavior-change surface — read carefully)

Today's in-process `task()` does three things a subprocess path breaks, and the
design must address each:

1. **Edits land live.** An in-process write-capable subagent's `edit`s appear in
   the shared tree; the lead reads them next turn. `deliver=False` would make
   them silently invisible — a regression. `deliver=True` preserves visibility
   but auto-ff-merges work the lead never reviewed.
   → **Default `deliver=True` for `task()`** (match current behavior: delegated
     edits are visible). The lead already chose to delegate to a write profile,
     so "edits should land" is the expected semantics. Document it. (`deliver`
     stays contract-gated `False`-by-default for workflows, where the contract
     gates landing.)
2. **Approval gating.** In-process write tools route through the parent's
   approval callback (`task.py:147`). The subprocess runs `--trust` → fully
   auto-approved. Write-capable `task()` delegation goes from approval-gated to
   unattended.
   → **This is the intended trade** for isolation (an isolated subagent in its
     own tree cannot usefully prompt the parent per-edit). Document it as a
     deliberate behavior change, not an oversight. Users who need per-edit
     approval keep `task.isolation="off"`.
3. **Hooks / session-log locality.** In-process subagents share the parent's
   hook manager (`hook_config_result=ctx.hook_config_result`, `task.py:142`) and
   log under `ctx.session_dir/"agents"` (`task.py:127`). A subprocess loads its
   own config and session; the parent's hooks no longer observe its tool calls
   and its transcript lands under `VIBE_HOME`, not the parent's session dir.
   → **v1 accepts both**, documented as known behavior changes. A future
     `--parent-session`/`--parent-hooks` passthrough to `vibe -p` could restore
     both; out of scope here.
4. **`TaskResult` must carry a recovery handle.** If `deliver=False` (or ff
   refuses), the kept worktree/branch is otherwise an orphan the lead can't
   reference. Add `TaskResult.worktree_path` / `.branch` (None for the
   in-process path and on clean delivery) so the lead can `git merge <branch>`.

## Algorithm (the isolated `task()` branch)

1. Resolve `agent_profile`; bail if not a subagent (today's check, unchanged).
2. If `profile_requires_isolation(profile)` is False, or `task.isolation == "off"`:
   today's in-process path.
3. Else: `wt = create_ephemeral_worktree(Path.cwd(), label=args.agent)`.
4. Build the `vibe -p` cmd (shape of `runtime.py:1408-1419`) **with `--agent
   args.agent`** (not the auto-approve hardcode), `cwd=wt.path`, `env` with
   `VIBE_WORKFLOW_EMIT_STATS=1`.
5. `await proc.communicate()`. On `CancelledError`, kill the process group and
   wait before the `finally` removes the worktree (copy `runtime.py:1438-1451` —
   this ordering is load-bearing against the EBUSY race).
6. Parse stdout → `TaskResult.response`; parse stderr stats (logged only —
   `task()` has no budget to charge).
7. `deliver = task.isolation != "off"` (default True for `task()`). On success +
   deliver, `deliver_ephemeral_worktree(wt)`. Set `TaskResult.worktree_path`/
   `.branch` only when delivery was skipped or refused (recovery path).

## Edge cases

- **Cancellation / Ctrl-C.** Must reap the subprocess + group before worktree
  removal or git refuses EBUSY and the worktree leaks. Workflow executor solved
  this (`runtime.py:1438-1451`); reuse verbatim.
- **Non-git cwd.** `create_ephemeral_worktree` raises via GitPython. Fall back to
  in-process with a warning.
- **Repo mid-operation / dirty submodules.** `WorktreeManager` checks these
  (`manager.py:441-463`); the ephemeral path does not. Add the same guards or
  accept that `git worktree add` fails and fall back.
- **Read-only profile + `always`.** Isolates anyway (explicit override).
- **MCP.** In-process subagents are MCP-free by design (`runtime.py:1069`); an
  isolated subprocess discovers MCP itself. Carries over unchanged.
- **Result streaming.** In-process `task()` streams `AssistantEvent`s to the UI
  incrementally (`task.py:163`). The subprocess path returns final stdout only.
  Acceptable for v1; revisit if long-running isolated tasks need progress.

## Test plan

- Unit `profile_requires_isolation`: True for `worker`, `auto-approve`, `editor`
  (editor is the key assertion — the original predicate got this wrong); False
  for `explore`, `research`, `planner`, `reviewer`, `debugger`, `security`.
  Also: a hypothetical profile with `enabled_tools=["bash"]` and no jail → True.
- Unit: the `--agent` passed to the subprocess equals the requested profile (not
  `auto-approve`) — regression test for the hardcode fix.
- Integration `task()` isolated branch (write profile): an `edit` to a file does
  NOT appear in the parent tree mid-run; after completion with `deliver=True` it
  DOES appear (ff-merged); the worktree is removed on clean delivery; on ff
  refusal `TaskResult.branch` is set and the worktree is kept.
- Integration cancellation: cancelling `task()` mid-run reaps the subprocess and
  removes the worktree (no EBUSY leak).
- Regression: read-only profiles (`explore`) still run in-process (assert no
  `create_ephemeral_worktree` call).
- `task.isolation == "off"` forces in-process even for write profiles;
  `"always"` forces isolation for `explore`.

## Risks

- **Cold-start latency on every write-capable subagent.** Per subprocess + git
  worktree I/O. No benchmark exists in the repo to quantify it; the direction is
  clear. Mitigated by the profile gate (only write profiles pay) and the
  `task.isolation="off"` escape hatch. If unacceptable in practice, a persistent
  warm-`vibe -p` pool is the next step — out of scope.
- **Approval bypass (intentional).** Write-capable `task()` delegation runs
  unattended in its worktree. Documented behavior change; `off` opt-out.
- **Hook / session-log locality lost.** Parent hooks no longer observe isolated
  subagent tool calls; transcript lands outside the parent session. v1 accepts;
  future passthrough could restore.
- **No new index-lock hazard.** Each isolated subagent's worktree has its own
  git index, so the `git reset`/`add` race that bites co-located agents cannot
  occur — this is the property that addresses the user's stated concern in the
  concurrent case.

## Adversarial verification (completed)

A reviewer pass against source found three blocking errors in the first draft,
all confirmed against code and fixed here:

1. **Editor was under-isolated.** Original predicate ("no allowlist = isolate")
   missed `editor` (`models.py:337` has an allowlist yet writes). Fixed: the
   predicate now also triggers when the allowlist contains a write tool. The
   workflow rule (`_validate_workflow_profile`) now calls
   `profile_requires_isolation` too, so `editor` is forced to
   `isolation='worktree'` on both the `task()` and workflow paths.
2. **Reviewer/debugger/security premise was false.** The draft assumed their
   bash was destructive; `_REVIEW_BASH_DENYLIST` (`models.py:149-171`)
   hard-denies destructive commands. They are read-jailed and stay in-process.
   The original Open Question #1 (isolate them via a stricter rule) is dropped.
3. **`--agent auto-approve` hardcode** (`runtime.py:1412-1413`) would have made
   every isolated `task()` run as auto-approve, ignoring the requested profile.
   Fixed: thread the real `agent` through (regression test added).

The review also surfaced four omitted behavior changes, now in *Delivery* above:
approval bypass, edit-visibility (→ `deliver=True` default for `task()`),
hook/session-log locality, and the missing recovery handle on `TaskResult`.

## Implementation status: landed

Implemented in `profile_requires_isolation` (`agents/models.py`), the shared
`run_isolated_agent` + `IsolatedResult` (`workflows/runtime.py`), the
`task.isolation` config + `_run_isolated` branch (`tools/builtins/task.py`),
and the `--agent` hardcode fix in `_default_isolated_executor`. Covered by
`tests/core/test_agents_models.py` (6 predicate tests incl. the editor case).
pyright clean, ruff clean, 310 task+workflow tests pass. Known follow-ups
deferred per Out-of-scope: warm subprocess pools, result streaming, hook/log
passthrough.

---

## Out of scope (explicit)

- Persistent warm-`vibe -p` pools (latency mitigation, future work).
- Streaming results from isolated subagents (v1 returns final stdout only).
- `--parent-hooks` / `--parent-session` passthrough to restore hook/log locality.
- Changing `task()` to accept a `contract=` (workflow-only concept).

The workflow-path predicate wiring (formerly listed here as a one-line
follow-up) landed in `d30c2e7`: `_validate_workflow_profile` now calls
`profile_requires_isolation`, so `editor` is forced to `isolation='worktree'`
on both the `task()` and workflow paths.

---

## Host default isolation (follow-up, landed)

The subagent isolation above covers `task()` and workflow agents. The *host*
agent (the interactive CLI and `vibe -p`) ran in the user's live checkout by
default, behind `--worktree` / `mode="auto-by-entrypoint"`. That left the host
racing any other writer sharing the checkout — the lead's index could pick up
teammate or sibling-session work, and commits needed pathspec gymnastics to
avoid shipping someone else's changes.

Closed by flipping two `WorktreeConfig` defaults in `vibe/core/config/_settings.py`:

- `mode`: `"auto-by-entrypoint"` → `"on"`. The interactive CLI and `vibe -p`
  now default to worktree isolation. `worktree_enabled()` is unchanged; its
  existing callers (`cli.py`, `programmatic.py`) go ON under the new default.
- `merge`: `"manual"` → `"auto-ff"`. Clean sessions fast-forward back into the
  original HEAD on exit. `_try_auto_ff` still refuses safely if the original
  tree is dirty or HEAD moved (`manager.py:529-540`), so under contention the
  branch is kept for a manual merge — isolation prevents races, and when the
  tree is contended the human decides the merge.

Opt-out: `--no-worktree` (interactive CLI) forces isolation off for one
invocation; `mode="off"` disables it persistently; `mode="auto-by-entrypoint"`
restores the legacy programmatic-only split. `--worktree` is kept as an
explicit override.

### ACP: deferred (architecture blocker)

ACP was intended to inherit the host default but cannot. `WorktreeManager` is
a process-wide singleton with a nested-enter guard that raises on re-entry,
and it works via `os.chdir` (one process cwd). ACP supports **multiple
concurrent sessions** in one process (`acp_agent_loop.py:361`:
`self.sessions: dict[str, AcpSessionLoop]`), each with its own `cwd`. Wiring
`worktree_manager.enter()` into ACP's `new_session` means session B crashes on
the nested guard (or clobbers session A's cwd if the guard is bypassed).

The programmatic path is one-shot (one session per process), and the CLI is
one interactive session per process — both fit the `WorktreeManager` model.
ACP does not.

ACP isolation needs a per-session worktree mechanism without process chdir —
closer to `EphemeralWorktree` (`vibe/core/worktree/ephemeral.py`) than to
`WorktreeManager` — wired into the ACP session lifecycle. That is substantial
new code and is tracked as a separate change. ACP stays unwired (unchanged
from today) until then.

---

## Dirty-start merge-back + stranded-work recovery (follow-up, landed)

The host default above flipped `merge` to `auto-ff`, but `_try_auto_ff` refused
on *any* dirty original tree. When a session started dirty (the common case —
the user's uncommitted edits stay in the live tree all session), the refusal
fired every exit and the work stranded on the `vibe/*` branch. The ephemeral
path compounded it: a refused delivery kept the whole worktree *directory* on
disk, accumulating unboundedly under `VIBE_HOME/worktrees`.

Fixed in three layers (`manager.py`, `ephemeral.py`, `runtime.py`,
`worktree_cmd.py`):

- **Auto-land dirty-start work.** `_try_auto_ff` now fingerprints the original
  tree's carried-relevant dirt (sha256 of the carry patch) at enter
  (`WorktreeHandle.entry_dirty_fingerprint`) and again at exit. If the tree is
  clean → plain `--ff-only`. If the live dirt still matches the fingerprint →
  `_stash_ff_drop`: stash the live dirt (excluding the same untracked
  `carry_ignored` paths the carry excluded, so an untracked `.env` is never
  swept), `--ff-only` (which re-materialises dirt + agent work, since the WIP
  commit already contains the carried dirt), then **drop** the now-redundant
  stash. If the fingerprint changed — a concurrent writer added uncommitted work
  the branch does not contain — the merge is **held** (branch kept) so that work
  is never dropped. Stash entries are located by a unique message, not a raw SHA
  (`git stash drop/pop` reject SHAs). This supersedes the "refuses if the
  original tree is dirty" behavior noted in the Host-default section above.
- **Ephemeral leak fix.** `remove_ephemeral_worktree(keep_if_changed=True)`
  commits the agent's work onto the branch, then force-removes the *directory*
  while keeping the branch ref — the recoverable unit is a cheap branch, not a
  whole checkout. The stash-drop logic is **not** shared with this path (it has
  no carry-dirty, so a drop there would lose the parent's dirt).
- **Visibility + recovery.** `print_startup_report` surfaces unmerged `vibe/*`
  branches at interactive startup (the prior orphan log was file-only, invisible
  at the WARNING default). `_gc_abandoned_worktrees` deletes only merged-and-old
  orphan branches (never unmerged work, never a live worktree). A
  `vibe worktree {list,merge,discard}` subcommand (pre-dispatched before the
  positional-prompt parser) lists/recovers/clears them. New `WorktreeConfig`
  fields: `report_on_startup`, `gc_age_days`.
- **Steering.** The worktree system-prompt sections and the editor/coordinator
  profiles nudge shell-capable agents to commit finished work with a real
  message (the WIP auto-save remains the backstop for tool-restricted profiles
  like `editor`, which has no shell).
