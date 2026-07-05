from __future__ import annotations

from vibe.core.skills.models import SkillInfo, SkillScope, SkillSource

# Pulled on demand via the `skill` tool; ~5k tokens the host pays only when delegating.
_PROMPT = """\
# Everyday delegation to write-capable subagents

The `task` tool's picker routes you to read-only investigators (explore/research/\
planner/reviewer/debugger/security/verifier). This skill covers the three \
write-capable subagents — `editor`, `grunt`, `worker` — for everyday (non-workflow) \
use. All three auto-isolate in their own git worktree under the task tool's default \
(`isolation='auto'`), run async by default, and deliver their result at the top of a \
later turn. They are NOT synchronous helpers — each lands work on a separate \
reviewable branch, not your live tree.

## Pick by the task, not the tool surface

| Agent | Unique edge | Use when |
|---|---|---|
| `editor` | read/grep/lsp/write/edit, **no bash, no MCP** — lowest blast radius | the change is fully specified and you've found the sites (e.g. via `lsp find_references`). It cannot run tests or escape via shell. |
| `grunt` | full set but **routes to a cheap model** (`grunt_model`) + a no-decisions prompt | bulk mechanical work — N-file rename, same field across M files. Cheaper than worker; literal, not reasoning. |
| `worker` | full set **+ MCP** | the task needs bash (build/test/install) AND writes in one branch, or needs an MCP tool you don't want in your own context. |

grunt vs worker for everyday work: grunt is cheaper but literal — if the brief \
has any ambiguity it stops and reports `AMBIGUOUS` rather than guessing; worker \
reasons. Pick grunt when the work is dumb, worker when it isn't. editor is the \
safest when you've already pinned the exact edits and want zero shell exposure.

## Everyday patterns

**1. Plan-execute-verify (no workflow script).** You do recon + design. Hand the \
fully-specified change to `grunt` (`async_run=true`). When it lands, spawn \
`verifier` on its branch. You get the thinker-executor-gate composition manually: \
mechanical execution offloaded to a cheap model, gated independently. Highest-value \
everyday pattern.

**2. Branch-per-risk for risky changes.** Instead of editing your live tree on a \
sketchy refactor, `task` a single `worker` with the brief. It lands on its own \
branch; you `git diff`, merge or discard. Reversible and reviewable.

**3. Parallel independent edits.** Spawn 2-3 `editor`/`grunt` calls in ONE turn \
with disjoint file sets (e.g. "rename X across `vibe/core/`", "rename X across \
`vibe/cli/`"). Each isolates separately; you merge both. Breadth comes from the \
count of parallel briefs — workflow fan-out shape without authoring a script.

**4. Free your context during long edits.** While a `grunt` mechanically applies a \
30-site rename, you keep reading the next module or answering the user. The rename \
delivers later; your context stays on synthesis.

**5. Precision over self-editing.** When you've found 12 call sites via \
`lsp find_references`, hand `editor` the exact `file:line` list with "rename \
exactly X to Y in exactly these." It read-then-edits each, keeps count, reports \
`AMBIGUOUS` rather than guessing. You spend zero context tokens on the mechanical \
pass.

## Brief hygiene (matters more for write-capable agents)

- Hand the agent the concrete context you already have — exact `file:line` refs, \
the diff, the specific change. It verifies rather than re-discovers.
- For `grunt`/`editor`, resolve every ambiguity in the brief yourself first. A \
`grunt`/`editor` that hits an unstated case stops and reports it back; a `worker` \
will reason and guess. Match the agent to the brief's completeness.
- Name the files explicitly. Worktree isolation is git isolation from the live \
checkout, NOT a security sandbox — symlinked deps and absolute paths can still \
reach outside the worktree.
- All three leave files in final state and do not commit; their branch is merged \
back on exit.

## What does not work

- **`async_run=false` for long tasks** blocks your whole turn and defeats the \
point. Use it only for short tasks where you need the result inline this turn.
- **Don't use them for exploration.** That's `explore`'s niche — in-process, no \
worktree-creation cost on every spawn.
- **Don't hand `grunt` a design decision.** Its prompt says stop-and-report; \
you'll get a blocked result and a re-spin. Resolve the ambiguity yourself first, \
or use `worker`.
- **No mid-run messaging.** These agents cannot message you mid-task (only \
teammates can, via the mailbox). If you need back-and-forth, that's `team_spawn`, \
not `task`.

## When to reach for a workflow instead

Three or more INTERDEPENDENT agents (phases, schema validation, a shared budget \
cap) → author a `launch_workflow` script (load the `workflow-authoring` skill \
first). For 1-3 independent edits in everyday flow, plain `task` calls are lighter \
and you stay in the loop. The boundary is dependency, not count.
"""


SKILL = SkillInfo(
    name="agent-delegation",
    description=(
        "Everyday (non-workflow) delegation to the write-capable subagents "
        "(`editor`, `grunt`, `worker`): when to pick each, the plan-execute-verify "
        "composition, branch-per-risk, parallel independent edits, and brief "
        "hygiene. The task tool's picker covers read-only investigators; load this "
        "when delegating actual file-mutating work to editor/grunt/worker."
    ),
    summary=(
        "Everyday delegation to write-capable subagents (editor/grunt/worker) — "
        "pick, patterns, brief hygiene."
    ),
    user_invocable=False,
    prompt=_PROMPT,
    source=SkillSource.BUILTIN,
    scope=SkillScope.BUILTIN,
)
