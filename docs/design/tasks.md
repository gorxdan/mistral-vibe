# Design Spec — tasks-pane (unified background task manager)

**Effort:** M–L  |  **Verdict:** _pending review_  |  **Feasible:** True  |  **Depends on:** none

## Current state

Mistral Vibe has five distinct "background thing" subsystems, each with bespoke state ownership, bespoke cancellation, and **no unified visibility or control**:

| Subsystem | State owner | Cancel path | TUI visibility |
|---|---|---|---|
| Agent bash processes | _none_ — `Bash.run()` (`vibe/core/tools/builtins/bash.py:622-722`) spawns via `create_subprocess_{shell,exec}` and awaits `communicate()` inline; PID discarded after the await | n/a (blocks until timeout) | none |
| TUI `!cmd` | `VibeApp._bash_task` single slot (`vibe/cli/textual_ui/app.py:450`) | `_kill_running_process` (`app.py:1492`) | none (inline only) |
| Workflow runs + live agents | `WorkflowRunner._runs` (`workflow_runner.py:141`) + `WorkflowRuntime._live_agents` (`runtime.py:232`) | `WorkflowRunner.stop()` / `cancel_agent()` | **only** live pane (`WorkflowsApp`, `widgets/workflows_app.py`), `ctrl+w` |
| Teammate subprocesses | `TeamManager._teammate_procs` + `_teammate_tasks` (`teams/manager.py:47-48`) | `_terminate_proc()` → `stop_teammate()` (`manager.py:282-326`) | none — inline markdown from `/team list` (`app.py:2858`) |
| Schedule loops | `LoopManager` via `ScheduledLoopRunner.manager` (`scheduled_loop_runner.py:64-69`) | `LoopManager` cancel | none — inline markdown from `/loop list` |

Grep for `ProcessManager` / `process_table` / `pid_registry` / `TaskManager` → **0 hits**. There is no process table anywhere. The agent's bash tool cannot background a long-lived process at all — a dev server, `vite watch`, or `pytest --loop` either blocks the turn until timeout or has to be handed to the user to run manually. This is the core UX gap.

`WorkflowsApp` (`widgets/workflows_app.py`) is the **only** live-monitoring precedent: 1s `set_timer` poll (`_POLL_INTERVAL`, line 31), drill-down list→detail→agent/script views, message-based actions (`StopRequested`, `PauseToggleRequested`, `AgentCancelRequested`, `SaveRequested`) handled on `VibeApp` at `app.py:2988-3064`. It is a clean template to generalize.

Navigation: panes are a closed `BottomApp` enum (`app.py:244-266`), one mounted at a time in `#bottom-app-container`, switched via `_switch_from_input` / `_switch_to_input_app` (`app.py:3560`, `3686`). `ctrl+w` toggles Input↔Workflows (`action_toggle_workflows`, `app.py:4176`). `_BUSY_ALLOWED_COMMANDS = frozenset({"workflows"})` (`app.py:376`) is the allowlist for commands that must stay reachable while the agent is busy — a tasks pane must be in this set or it is rejected mid-turn.

Built-in tools follow a fixed shape (`vibe/core/tools/builtins/task.py` is the cleanest template): `XArgs`/`XResult` BaseModels, `XToolConfig(BaseToolConfig)`, `X(BaseTool[…], ToolUIData[…])` with `description` ClassVar, `resolve_permission`, async `run(args, ctx)`. Action-enum tools already exist (`schedule`, `team_message`) — the established pattern for "operate on a subsystem" tools.

## Target design

Two layers.

**Layer 1 — `BackgroundRegistry`** (`vibe/core/tools/background.py`, new): a session-scoped singleton that **owns** background processes (the new primitive) and **aggregates** the other four categories read-only. It exposes one `list_tasks()` for reading and one `stop(task_id)` that routes cancellation to whichever owner backs the id. Nothing else duplicates state; the registry delegates.

**Layer 2 — `TasksApp` pane** (`vibe/cli/textual_ui/widgets/tasks_app.py`, new): replaces `WorkflowsApp` as the `ctrl+w` surface. Category filter row (`All · Processes · Workflows · Teams · Loops`) over a unified `TaskEntry` list; drill-down per category; same keybindings as today (`x` stop, `p` pause [workflow], `s` save [workflow], `r` refresh, `o` script [workflow]) plus `c` to cancel any row. 1s poll, unchanged.

**Agent awareness** — one new builtin tool `background` with `action: "list" | "stop"` (matches the `schedule` / `team_message` action-enum pattern), backed by the same registry. This is what makes the host agent "aware": it can enumerate and cancel what is running.

**Bash backgrounding** — `background: bool = False` on `BashArgs`. When true, `Bash.run()` spawns, redirects stdout+stderr to a log file under the session dir, registers the PID in the registry, and returns immediately with a handle. The agent stays unblocked; the process survives across turns and is killable from the pane or the `background` tool.

```
┌─ TasksApp (ctrl+w) ─────────────────────────────────────────────┐
│ [All] Processes  Workflows  Teams  Loops            r refresh   │
│                                                                  │
│  proc-1  process   running   12.4s   vite --port 5173            │
│  wf-2    workflow  running   45.1s   3 agents  12.4k tok  audit  │
│  wf-2/live-a7  agent  running  12.0s  explore  3.1k tok          │
│  team:bob  team    running   2m10s   vibe -p "refactor auth"     │
│  loop-l9k2 loop    waiting   fires in 4m   "recheck CI"          │
│                                                                  │
│ ↑↓ Navigate  Enter Detail  x Stop  c Cancel  r Refresh  Esc Back │
└──────────────────────────────────────────────────────────────────┘
```

## Data structures

```python
# vibe/core/tools/background.py (new)

class TaskCategory(StrEnum):
    PROCESS = "process"      # agent bash background spawns (owned by registry)
    WORKFLOW = "workflow"    # workflow runs (read from WorkflowRunner)
    AGENT = "agent"          # in-flight workflow agents (read from runtime._live_agents)
    TEAM = "team"            # teammate subprocesses (read from TeamManager)
    LOOP = "loop"            # schedule timers (read from LoopManager)

@dataclass
class TaskEntry:
    task_id: str            # "proc-1" | "wf-2" | "wf-2/live-a7" | "team:bob" | "loop-l9k2"
    category: TaskCategory
    label: str              # command / run+phase / teammate prompt / loop prompt
    status: str             # "running" | "completed" | "failed" | "paused" | "stopped" | "waiting"
    started_at: float       # monotonic; Loops use next_fire_at for "waiting"
    elapsed: float
    detail: dict            # category-specific (pid, returncode, agent_count, tokens, interval…)
    parent_id: str | None   # agent → workflow run; else None
    can_pause: bool         # True only for workflow runs
    can_save: bool          # True only for workflow runs

@dataclass
class _BgProc:
    task_id: str
    proc: asyncio.subprocess.Process
    command: str
    cwd: Path
    log_path: Path
    started_at: float
    finalized: asyncio.Task | None  # background awaiter that flips status when proc exits

class BackgroundRegistry:
    def __init__(self) -> None: ...
    # ownership (processes only)
    async def register_process(self, proc, *, command, cwd, log_path) -> str: ...
    async def read_log_tail(self, task_id, lines: int = 50) -> str: ...
    # aggregation (all five categories) — adapters injected once at TUI startup
    def attach_workflow_runner(self, r: WorkflowRunner) -> None: ...
    def attach_team_manager(self, m: TeamManager) -> None: ...
    def attach_loop_manager(self, m: LoopManager) -> None: ...
    def attach_tui_bash(self, ref: Callable[[], asyncio.Task | None]) -> None: ...
    def list_tasks(self, *, category: TaskCategory | None = None) -> list[TaskEntry]: ...
    async def stop(self, task_id: str) -> bool: ...   # routes to the right owner
    async def pause(self, task_id: str) -> bool: ...  # workflow runs only
```

ID scheme is prefix-stable so the registry can route `stop("wf-2/live-a7")` to `WorkflowRunner.cancel_agent("wf-2", "a7")` by splitting on `/`, and `stop("team:bob")` to `TeamManager.stop_teammate("bob")` by stripping the prefix. Processes are owned outright.

## Integration points

**New files**

- `vibe/core/tools/background.py` — `BackgroundRegistry`, `TaskEntry`, `TaskCategory`, `_BgProc`. Pure asyncio + dataclasses; reads other owners via the injected adapters. Fully unit-testable with fakes.
- `vibe/core/tools/builtins/background.py` — the `background` agent tool (`BackgroundArgs` action-enum, `BackgroundResult`, `BackgroundToolConfig`, `Background(BaseTool[…], ToolUIData[…])`). Mirrors `task.py` structure. Reads `ctx.background_registry`.
- `vibe/cli/textual_ui/widgets/tasks_app.py` — `TasksApp(Container)`. Lifts the list/detail/script renderers out of `workflows_app.py`; adds the category filter and per-category detail renderers (process = log tail; team/loop = status card). Emits `Closed`, `TaskStopRequested(task_id)`, `TaskPauseRequested(task_id)`, `SaveRequested(run_id, …)` (workflow only, reuses existing).

**Edits**

- `vibe/core/tools/base.py` — `InvokeContext` (L44-74): add `background_registry: BackgroundRegistry | None = None`. Tolerates None (tool must raise clearly if invoked without one, mirroring `task.py:110-111`).
- `vibe/core/tools/builtins/bash.py`:
  - `BashArgs` (L334-338): add `background: bool = Field(default=False, description="…")`.
  - `BashResult` (L341-345): add `background_task_id: str | None = None`, `pid: int | None = None`. `returncode=-1` sentinel while running.
  - `Bash.run()` (L622-722): at the top, if `args.background` and `ctx.background_registry` is set, take the **non-blocking** branch — spawn with `stdout/stderr` redirected to a log file under `ctx.session_dir / "bg" / f"{task_id}.log"`, call `register_process`, yield a `BashResult` with the handle, `return`. Do NOT `await communicate()`; do NOT `kill_async_subprocess` in `finally`. If `background=True` but no registry on ctx, raise `ToolError("background execution is not available in this context")`.
- `vibe/core/tools/builtins/bash.py` — `Bash.run()` sandbox interaction (L636-681): the background branch still calls `_resolve_sandbox` so a backgrounded server is sandboxed identically to a foreground call. Reuse the same spawn argv construction; only the stdio redirection and the await/return differ.
- `vibe/cli/textual_ui/app.py`:
  - Construct `self._background_registry = BackgroundRegistry()` at startup; attach the four adapters once `_workflow_runner`, `_team_manager`, `_loop_runner` exist.
  - Thread it into every `InvokeContext` the agent loop builds (find the single construction site and add the field).
  - `BottomApp` enum (L244-266): rename `Workflows` → `Tasks` (or add `Tasks` alongside and route both). Keep `WorkflowSaveApp` separate (it's a dialog, not the list pane).
  - `BINDINGS` (L390-412): repurpose `ctrl+w` → `action_toggle_tasks` (keep the key for muscle memory; rename the action). **Open question Q1** on the final key.
  - `_BUSY_ALLOWED_COMMANDS` (L376): `"workflows"` → `"tasks"`.
  - `_workflows_command` (L2799-2813) → `_tasks_command`; keep `/workflows` as a thin alias that calls `_tasks_command` for back-compat. New `/tasks` command.
  - `_switch_to_workflows_app` (L2983-2986) → `_switch_to_tasks_app`, mounting `TasksApp(registry=self._background_registry, workflow_runner=self._workflow_runner, workflow_manager=self._workflow_manager)`.
  - `on_workflows_app_*` handlers (L2988-3064) → `on_tasks_app_*`. Stop/pause now key off `task_id` and route via `registry.stop(task_id)` / `registry.pause(task_id)`; `SaveRequested` keeps its current workflow-only path. Keep `on_workflow_save_app_*` (L3045-3064) unchanged.
- `vibe/cli/commands.py` — register `"tasks"` and keep `"workflows"` as alias (~L188, mirroring the workflows entry).
- `vibe/cli/textual_ui/app.tcss` — add `#tasks-app` / `#tasks-content` rules; the existing `#workflows-app` rules can be copied and renamed, then the old ones removed once `workflows_app.py` is deleted.
- `vibe/core/tools/manager.py` — register the new `background` builtin alongside `task`/`bash`/`schedule` (find the builtin discovery/registration list).

**Deletion**

- `vibe/cli/textual_ui/widgets/workflows_app.py` — once `TasksApp` reproduces its workflow/agent/script views, delete it. Its `WorkflowSaveApp`-related imports move to `tasks_app.py` or stay in a retained `workflow_save_app.py`.

## Config

No new required config. One optional field for tunability:

- `tools.bash.background_max_lifetime` (int seconds, default `86400`) — hard ceiling after which a backgrounded process is auto-reaped with a pane notification, so forgotten servers don't leak across long sessions. Default 24h; 0 = no ceiling.

`background: bool` on `BashArgs` is a per-call arg, not config. No `[tools.bash.background]` block is required for v1.

## Algorithm

1. Create `vibe/core/tools/background.py`: `TaskCategory`, `TaskEntry`, `_BgProc`, `BackgroundRegistry` with `register_process` (spawns a background `asyncio.create_task` that `await proc.wait()`s and flips `_BgProc` status to completed/failed on exit, recording `returncode`), `read_log_tail` (seeks the log file, returns last N lines), `list_tasks` (builds `TaskEntry` from each attached source + owned processes), and `stop` (prefix-routes: `proc-*` → `proc.terminate()` then SIGKILL after 3s, mirroring `TeamManager._terminate_proc` at `manager.py:282-297`; `wf-*/*` → `cancel_agent`; `wf-*` → `WorkflowRunner.stop`; `team:*` → `stop_teammate`; `loop-*` → `LoopManager` cancel).
2. Add `background_registry` to `InvokeContext`. Wire it from `VibeApp` into the `AgentLoop`'s context construction.
3. Bash: add `background` to `BashArgs`, the two optional fields to `BashResult`, and the non-blocking branch at the top of `run()`. Log path under `ctx.session_dir / "bg"`; create the dir lazily.
4. Create `vibe/core/tools/builtins/background.py`: `list` returns a compact table of `TaskEntry` (id, category, status, elapsed, label); `stop` calls `registry.stop(task_id)` and returns success/failure. Permission: `ALWAYS` (it only touches things the user/agent already launched).
5. Register the `background` builtin in the tool manager.
6. Create `vibe/cli/textual_ui/widgets/tasks_app.py`: category filter row (OptionList or tab bar), unified `TaskEntry` list, per-category detail views. Poll loop and message structure lifted from `workflows_app.py`. Process detail tails the log via `registry.read_log_tail`.
7. Rewire `app.py`: enum, keybinding, busy-allowlist, command handlers, message handlers, `_switch_to_*`. Keep `/workflows` + `ctrl+w` as aliases.
8. Delete `workflows_app.py`; move `WorkflowSaveApp` if it lives there.
9. Tests + docs.

## Edge cases

- `background=True` with no registry on ctx (e.g. ACP/headless run without the TUI wiring) — raise a clear `ToolError`, do not silently run in foreground.
- A backgrounded process exits between polls — the `_BgProc.finalized` awaiter flips status; the pane shows it as completed/failed with the returncode; it stays in the list until reaped (GC policy: keep finalized entries for the session, cap at e.g. 50, drop oldest).
- Sandbox + background: the non-blocking branch must still construct the sandbox argv; only stdio redirection differs. A backgrounded server under `--unshare-net` is useless — document, and when `sandbox.allow_network=false` and `background=True`, warn.
- `stop` on an already-finalized task — return False, pane shows "already finished" (mirrors `on_workflows_app_agent_cancel_requested` at `app.py:3005-3010`).
- Workflow live-agent `task_id` collision: an agent id could in principle contain `/`; agent ids are token-hex so they won't, but assert the split yields exactly two parts and fall back to "not found" otherwise.
- `disable_workflows` config (`app.py:2800`): the Tasks pane stays available (processes/teams/loops are not workflows); only the Workflows category is hidden/empty.
- App exit / ctrl+c with background processes running — `BackgroundRegistry.shutdown()` in the app's exit path terminates all owned processes (reuse `_terminate_proc` logic). Without this, backgrounded servers orphan to init. Mirror `TeamManager`'s `_signal_proc_group` (`manager.py:262-280`) for process-group reaping.
- TUI `!cmd` integration: optional v2 — surface `self._bash_task` as a `process`-category entry via the `attach_tui_bash` adapter. v1 can skip this since `!cmd` is foreground-blocking anyway.
- Standalone `task`-tool subagents (not in a workflow) are currently untracked everywhere. **Open question Q3** — instrument the `task` tool to register them, or defer to v2.
- Log file growth: a chatty server can write GBs. Cap log writes via a rotating handler or a size ceiling (`tools.bash.background_max_log_bytes`, default 16 MiB) — truncate-from-front when exceeded.

## Test plan

- Unit (`background.py`, no TUI): `list_tasks` aggregates from fakes for each category; `stop` routes correctly per id prefix; `register_process` records the proc and returns a stable id; the finalized awaiter flips status on `proc.wait()` completion; `read_log_tail` returns the last N lines of a temp log.
- Unit: id routing table — `proc-1`→terminate, `wf-2`→`WorkflowRunner.stop`, `wf-2/live-a7`→`cancel_agent("wf-2","a7")`, `team:bob`→`stop_teammate("bob")`, `loop-l9k2`→loop cancel, unknown→False.
- Unit (`bash.py`): `background=True` with a registry spawns and returns immediately with `background_task_id`/`pid` set, does NOT call `communicate()`, does NOT kill the proc in finally. `background=True` without registry raises `ToolError`. `background=False` is byte-identical to today (regression: assert `communicate` IS awaited).
- Unit (`background` tool): `list` returns the registry's entries; `stop` returns True/False from the registry; permission resolves to `ALWAYS`.
- Integration: launch a backgrounded `sleep 30`, confirm it appears in the registry as running, `stop` it, confirm status flips and the proc is reaped.
- Integration: backgrounded server (`python -m http.server`) — confirm the port is live after the tool returns, the pane tails the log, and `stop` frees the port.
- TUI snapshot test (if the repo has them): `TasksApp` renders the category filter, lists entries from a fake registry, `x` emits `TaskStopRequested` with the right id, drill-down tails a fake log.
- Regression: `ctrl+w` still opens the pane; `/workflows` still works as an alias; `_BUSY_ALLOWED_COMMANDS` admits `/tasks` while the agent is busy.
- Exit hygiene: app shutdown terminates all owned background processes (no orphans).

## Risks

- **Orphaned processes** — the highest-stakes risk. A backgrounded server whose parent vibe exits must be reaped; otherwise the feature leaks processes every session. Mitigation: `start_new_session=True` + `BackgroundRegistry.shutdown()` in the app exit path using process-group signaling; also consider `PR_SET_PDEATHSIG` (Linux) so the child dies if vibe dies. Must be tested.
- **Refactor blast radius** — replacing `WorkflowsApp` touches the enum, keybindings, command registry, and ~6 message handlers. Keeping `/workflows` + `ctrl+w` as aliases and keeping `WorkflowSaveApp` separate contains it, but it's the riskiest part of the change.
- **Log file resource use** — unbounded logs from chatty servers. Mitigation: size cap with front-truncation; document.
- **Two ways to cancel** — the `background` tool and the Tasks pane both call `registry.stop`; they must stay consistent. Mitigation: single code path, both are thin callers.
- **`disable_workflows` interaction** — the pane must not break when workflows are disabled. Mitigation: only the Workflows category hides.
- **Process-group reach** — a backgrounded `npm run dev` spawns children; terminating only the shell PID orphans the children. Mitigation: reuse `TeamManager._signal_proc_group` (start_new_session + `os.killpg(os.getpgid(pid))`).
- **Scope creep** — standalone `task`-subagent tracking and TUI `!cmd` integration are tempting to fold in; both should be v2 to keep this shippable.

## Open questions (need your call before implementation)

- **Q1 — Keybinding.** Your "down arrow" is taken (`alt+down`=rewind, `shift+down`=scroll, bare `down`=input history). Options: (a) keep `ctrl+w`, repurpose the action name to "tasks" — zero muscle-memory breakage; (b) move to `ctrl+t` and free `ctrl+w`; (c) something else.
- **Q2 — Process output capture.** v1 proposal: redirect to a log file, pane tails it (simple, survives process death). Alternative: in-memory ring buffer via async queue (richer, lost on exit). Log file is the pragmatic v1.
- **Q3 — Standalone `task` subagents.** Today they're untracked. Show them in the Tasks pane (requires instrumenting the `task` tool to register/deregister), or defer to v2? Recommend defer — keeps this change focused on the bash gap, which is the actual blocker.
- **Q4 — Doc home.** I wrote this to scratchpad per your choice. The repo convention is `docs/design/` (cf. `sandbox.md`). Promote to `docs/design/tasks.md` once you're happy, or keep in scratchpad?

## Phasing (if greenlit)

1. `BackgroundRegistry` + `InvokeContext` wiring + unit tests (no UI). Shippable on its own — the `background` tool works headless.
2. `background` flag on `Bash.run()` + the `background` agent tool. The bash gap is now closed; manageable via `/tasks` slash command rendering the registry inline (like `/team list` today).
3. `TasksApp` pane replacing `WorkflowsApp` + app.py rewire + alias retention.
4. Exit-hygiene (process reaping) + log caps + integration tests.

Each phase is independently mergeable. Phase 1+2 alone already fix the "launch a web server" UX you raised; phase 3 adds the unified visibility.
