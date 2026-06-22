# Design Spec — shell-sandbox

**Effort:** L → revised **L**  |  **Verdict:** `sound_with_fixes`  |  **Feasible:** True  |  **Depends on:** none

## Current state
vibe's bash tool runs commands with NO OS-level sandbox. In `vibe/core/tools/builtins/bash.py:516-524`, `Bash.run()` calls `asyncio.create_subprocess_shell(args.command, ..., env=_get_base_env(), executable=_get_shell_executable(), start_new_session=True)` directly against the user's real shell with the full inherited environment (`_get_base_env` at L73-87 does `{**os.environ, ...}`, so it forwards every env var including secrets). Confinement today is purely textual/advisory, all evaluated in `resolve_permission` (L451-480) BEFORE the process is spawned:

- denylist / standalone-denylist → `ToolPermission.NEVER` (hard deny) via `_resolve_guardrail_permission` (L362-394).
- allowlist prefixes → `ALWAYS` (no prompt) via `_is_allowlisted` (L351-354).
- `_collect_outside_dirs` (L193-236) statically parses arguments of a fixed set of `_PATH_COMMANDS` (cat/cp/rm/mv/mkdir/... L173-188) and, for paths resolving outside `Path.cwd()` (`is_path_within_workdir`, utils.py:42-53) and outside the scratchpad (`is_scratchpad_path`), emits an `OUTSIDE_DIRECTORY` RequiredPermission → `ASK`. This is trivially bypassed: `python -c`, `sh -c "rm /etc/x"`, `eval`, `$(...)`, env-var paths, `tee`, `dd`, redirections, symlinks, or any binary not in `_PATH_COMMANDS` are never inspected, and on Windows `resolve_permission` returns None unconditionally (L452-453).
- The optional LLM safety judge (`vibe/core/tools/safety_judge.py`, gated by `SafetyJudgeConfig` in `_settings.py:211-238`, wired at `agent_loop.py:1014` `_resolve_safety_judge` and consulted on the `ASK` path) only decides whether to *skip the human prompt* — it never constrains the spawned process.

So nothing prevents an approved-or-allowlisted command from writing outside the workspace, reading `~/.ssh`/`~/.aws`/env secrets, or making arbitrary network calls. Config reaches the tool via `self.config` (a `BashToolConfig`, base.py:163-174); `BaseToolConfig` has `model_config = ConfigDict(extra="allow")` so new fields are additive. The session scratchpad lives under a tempfile dir (`vibe/core/scratchpad.py:12-31`) and is surfaced to tools as `InvokeContext.scratchpad_dir` (base.py:58). Cross-platform helper `is_windows()` exists in `vibe/core/utils/platform.py`.

## Target design
Add an opt-in, config-gated OS-level sandbox that wraps the bash subprocess as layer 5 (after, never replacing, the permission gate + safety judge). Default OFF → byte-for-byte current behavior. When enabled, every bash invocation is launched under a platform-appropriate sandbox wrapper that (a) confines filesystem WRITES to an allowlisted set of roots (workspace = `Path.cwd()`, scratchpad dir, plus user-configured `write_dirs`, plus per-call `OUTSIDE_DIRECTORY` grants the user already approved), keeping most of the filesystem readable but read-only; and (b) optionally blocks network egress.

Architecture — a new module `vibe/core/tools/sandbox.py` exposing:
  - `SandboxConfig` (pydantic, lives in `_settings.py` next to `SafetyJudgeConfig`; attached to `BashToolConfig`).
  - `SandboxSpec` dataclass: the resolved, absolute write-roots, read-roots, allow_network bool, env-allowlist.
  - `build_sandbox_command(shell_cmd: str, *, executable: str|None, spec: SandboxSpec) -> tuple[list[str]|None, str]` — returns `(argv_prefix, backend_name)` where `argv_prefix` is the wrapper argv to which the shell invocation is appended, or `None` when no usable backend exists (graceful fallback).
  - Backend resolver `detect_backend()` that probes (via `shutil.which`) for, in order: Linux → `bwrap` (bubblewrap), else `unshare` (user+mount+net namespaces); macOS → `sandbox-exec` (seatbelt); Windows → none.

The bash tool keeps using `create_subprocess_shell` semantics but, when sandboxing is active, switches to `create_subprocess_exec(*argv_prefix, shell_exe, "-c", args.command, ...)` so the wrapper is PID 1 of the new namespace and the user command runs as the shell `-c` payload inside it. The textual guardrails in `resolve_permission` are unchanged and still run first; the sandbox is purely defense-in-depth at spawn time.

Backend wrappers:
- bubblewrap (preferred Linux): `bwrap --die-with-parent --unshare-pid --unshare-uts --unshare-ipc [--unshare-net] --dev /dev --proc /proc --tmpfs /tmp --ro-bind / / --bind <root1> <root1> --bind <root2> <root2> ... --chdir <cwd> -- <shell> -c <command>`. `--ro-bind / /` makes the whole tree readable but read-only; each write-root is re-bound rw on top. `--unshare-net` added only when `allow_network=false`.
- unshare (fallback, no bwrap): `unshare --user --map-root-user --mount [--net] -- <shell> -c '<remount-ro-then-command>'`. Weaker (remount-ro of `/` then bind-mount write roots inside the namespace via a tiny prelude); used only if bwrap absent.
- seatbelt (macOS): generate an SBPL profile string to a temp file under the scratchpad: `(version 1)(deny default)(allow process*)(allow file-read*)` + per-root `(allow file-write* (subpath "<root>"))` + network either `(allow network*)` or `(deny network*)`, then `sandbox-exec -f <profile> <shell> -c <command>`.
- Windows / no backend: `build_sandbox_command` returns `(None, "none")`. The tool logs once and runs unsandboxed unless `require_backend=true`, in which case the call fails closed with a `ToolError`.

Env scrubbing: when sandbox enabled and `scrub_env=true`, `_get_base_env` output is filtered to an allowlist (`PATH, HOME, USER, LANG, LC_*, TERM, SHELL, TMPDIR` + the noninteractive vars vibe already sets + user-listed `env_passthrough`), dropping `*_API_KEY`, `*_TOKEN`, `AWS_*`, `GH_TOKEN`, etc. This closes the secret-exfil hole even when network is allowed.

Composition with existing layers (order, outermost-first): denylist NEVER (resolve_permission) → permission prompt / safety judge (ASK) → OS sandbox (spawn). The sandbox NEVER relaxes a denial; it only adds containment to commands the upper layers already permitted. The `OUTSIDE_DIRECTORY` grants the user approves at the prompt are passed into the spec as extra write-roots so an approved out-of-tree write still succeeds inside the sandbox (otherwise approved commands would mysteriously fail).

## Integration points

- `vibe/core/config/_settings.py` — **SandboxConfig (new, after SafetyJudgeConfig L238)**: Add a BaseSettings model with model_config extra='ignore' and the keys listed in the config section. Export it from vibe/core/config/__init__.py alongside SafetyJudgeConfig so bash.py can import it.
- `vibe/core/tools/builtins/bash.py` — **BashToolConfig (L243-266)**: Add field `sandbox: SandboxConfig = Field(default_factory=SandboxConfig)`. Import SandboxConfig from vibe.core.config. extra='allow' on the base means this is backward-compatible with existing TOML.
- `vibe/core/tools/sandbox.py` — **new module: SandboxSpec, detect_backend(), build_sandbox_command(), _build_seatbelt_profile()**: New file. Pure functions + dataclass; no I/O except shutil.which probing and writing the seatbelt profile to scratchpad. Fully unit-testable in isolation.
- `vibe/core/tools/builtins/bash.py` — **Bash.run() subprocess creation (L510-524)**: Before spawning, call a new helper `self._resolve_sandbox(ctx)` -> (argv_prefix, backend) | (None, 'disabled'/'none'). If argv_prefix is not None, call asyncio.create_subprocess_exec(*argv_prefix, shell_exe, '-c', args.command, stdout=..., stderr=..., stdin=DEVNULL, env=sandbox_env, start_new_session=True) where shell_exe = _get_shell_executable() or '/bin/sh'. Else keep the existing create_subprocess_shell path. kill_async_subprocess/timeout/output handling stay identical.
- `vibe/core/tools/builtins/bash.py` — **_get_base_env (L73-87) + new _build_sandbox_env()**: Add an env-scrubbing wrapper used only when sandbox.scrub_env is true; leave _get_base_env untouched for the unsandboxed path.
- `vibe/core/tools/builtins/bash.py` — **Bash._resolve_sandbox(ctx) (new method)**: Reads self.config.sandbox; if not enabled returns (None,'disabled'). Builds write_roots = [Path.cwd()] + scratchpad (ctx.scratchpad_dir) + config.write_dirs (expanded) + approved OUTSIDE_DIRECTORY roots (see edge cases). Calls detect_backend()+build_sandbox_command(). On (None,'none'): if require_backend raise ToolError, else log-once and return (None,'none').
- `vibe/core/tools/base.py` — **InvokeContext (L45-74)**: No new field strictly required — scratchpad_dir already present. ctx may be None (run signature allows it); _resolve_sandbox must tolerate ctx is None / scratchpad_dir is None.
- `vibe/docs or config reference` — **config docs**: Document the [tools.bash.sandbox] block. Out of scope for code but list in README/config docs.

## Config

- `tools.bash.sandbox.enabled` (bool, default `false`) — Master switch. False => identical to today (create_subprocess_shell, full env). True => wrap every bash spawn in the OS sandbox.
- `tools.bash.sandbox.write_dirs` (list[str], default `[]`) — Extra absolute/~-expanded directories (beyond cwd workspace + scratchpad) where writes are allowed inside the sandbox.
- `tools.bash.sandbox.allow_network` (bool, default `true`) — When false, drop network namespace (bwrap --unshare-net / unshare --net / seatbelt deny network*). Default true to avoid breaking package installs/git.
- `tools.bash.sandbox.scrub_env` (bool, default `true`) — When true (and sandbox enabled), pass only an env allowlist into the sandboxed process, dropping API keys/tokens/cloud creds.
- `tools.bash.sandbox.env_passthrough` (list[str], default `[]`) — Extra env var names to allow through when scrub_env is true (e.g. a CI token a build genuinely needs).
- `tools.bash.sandbox.require_backend` (bool, default `false`) — Fail-closed toggle. If true and no sandbox binary is available (e.g. Windows, missing bwrap), bash calls error instead of silently running unsandboxed.
- `tools.bash.sandbox.backend` (str (auto|bwrap|unshare|sandbox-exec|none), default `auto`) — Override backend autodetection for testing / forcing a specific wrapper.
- `tools.bash.sandbox.extra_args` (list[str], default `[]`) — Escape hatch: extra raw flags appended to the wrapper argv (e.g. additional --ro-bind mounts) for advanced setups.

## Algorithm
 1. 1. Add SandboxConfig to _settings.py and export it; add `sandbox` field to BashToolConfig in bash.py.
 2. 2. Create vibe/core/tools/sandbox.py. Define SandboxSpec(write_roots: list[Path], allow_network: bool, env: dict[str,str], extra_args: list[str]).
 3. 3. detect_backend(override): if override != 'auto' return it. On Windows return 'none'. On Linux: shutil.which('bwrap') -> 'bwrap'; elif which('unshare') -> 'unshare'; else 'none'. On macOS: which('sandbox-exec') -> 'sandbox-exec'; else 'none'.
 4. 4. build_sandbox_command(shell_cmd_unused, executable, spec, backend): dispatch to per-backend builder. Each returns argv_prefix (list) that ENDS right before the shell exe (the caller appends `shell_exe -c command`). For seatbelt, write the generated SBPL profile to a temp file under the scratchpad and reference it; return its path so the caller can unlink in finally.
 5. 5. bwrap builder: ['bwrap','--die-with-parent','--unshare-pid','--unshare-uts','--unshare-ipc','--dev','/dev','--proc','/proc','--tmpfs','/tmp','--ro-bind','/','/'] + (['--unshare-net'] if not allow_network) + flatten(['--bind',str(r),str(r)] for r in canonicalized write_roots) + ['--chdir',str(Path.cwd())] + spec.extra_args + ['--'].
 6. 6. unshare builder (fallback): ['unshare','--user','--map-root-user','--mount'] + (['--net'] if not allow_network) + ['--']; the mount-ro-of-/ + rebind-rw is done by a small shell prelude prepended to the command (documented limitation: weaker than bwrap).
 7. 7. seatbelt builder: profile = '(version 1)(deny default)(allow process-exec)(allow process-fork)(allow sysctl-read)(allow file-read*)' + per write-root '(allow file-write* (subpath "<abs>"))' + ('(allow network*)' if allow_network else '(deny network*)'); write to scratchpad tmp; return ['sandbox-exec','-f',profile_path].
 8. 8. In Bash.run(): compute sandbox via _resolve_sandbox(ctx) BEFORE spawning. If enabled and argv_prefix not None: shell_exe = _get_shell_executable() or '/bin/sh'; argv = [*argv_prefix, shell_exe, '-c', args.command]; spawn via create_subprocess_exec(*argv, env=spec.env, start_new_session=(not is_windows())). Else: existing create_subprocess_shell path.
 9. 9. _resolve_sandbox(ctx): if not config.sandbox.enabled -> (None,'disabled'). Collect write_roots = {Path.cwd().resolve()} | {ctx.scratchpad_dir} | {expanduser/resolve each config.write_dirs} | approved-outside-dirs (see edge cases). Build env (scrubbed or _get_base_env). backend=detect_backend(config.sandbox.backend). If backend=='none': if require_backend raise ToolError else _warn_once+return (None,'none'). Else build_sandbox_command -> (argv_prefix, backend).
 10. 10. _build_sandbox_env(): start from _get_base_env(); if not scrub_env return it; else keep only allowlist names (PATH,HOME,USER,LOGNAME,LANG,TERM,SHELL,TMPDIR,LC_*,SSL_CERT_FILE,SSL_CERT_DIR,CI,NONINTERACTIVE,NO_TTY,DEBIAN_FRONTEND,GIT_PAGER,PAGER,LESS) + config.env_passthrough names that are present.
 11. 11. finally: existing kill_async_subprocess(proc) covers both spawn paths; additionally unlink the seatbelt profile temp file if one was created.
 12. 12. Tests + docs.

## Edge cases
- ctx is None or ctx.scratchpad_dir is None (tool can be invoked without a scratchpad) — skip scratchpad write-root, never crash.
- Approved OUTSIDE_DIRECTORY writes: a command the user explicitly approved to write outside cwd must still work. resolve_permission already computed those dirs via _collect_outside_dirs; pass the same set into write_roots so the sandbox doesn't silently break an approved action. (If not threaded through, document that out-of-tree writes require listing in write_dirs.)
- bwrap requires the kernel to allow unprivileged user namespaces; some hardened distros disable them (sysctl kernel.unprivileged_userns_clone=0) — bwrap then fails at spawn. Detect the spawn failure and, if require_backend is false, fall back to unsandboxed with a one-time warning rather than hard-failing every command.
- Network-dependent allowlisted commands (git fetch, pip/npm install) break if allow_network=false — default allow_network=true mitigates; document the trade-off.
- /tmp tmpfs vs commands expecting persistent /tmp across calls: each bash call is a fresh sandbox, so /tmp is per-call ephemeral; note this changes behavior for scripts that stash state in /tmp between bash tool calls.
- Symlinks inside write-roots that point outside: bwrap binds resolve realpaths; ensure write_roots are realpath-resolved so a symlinked workspace still maps correctly.
- Path.cwd() changes if a prior command did `cd` — but each bash call is a new process; cwd of the vibe process is the stable workspace root, so --chdir uses that, matching current behavior.
- macOS SBPL profile injection: write-root paths are inserted into the profile string; canonicalize and reject roots containing quotes/newlines to prevent profile breakage.
- Windows: detect_backend -> 'none'; with default require_backend=false this is a no-op (same as today). resolve_permission already returns None on Windows, so layered guarantees are unchanged.
- Nested sandbox (vibe already running inside a container/sandbox): bwrap-in-bwrap or denied userns; treat spawn failure as graceful fallback.
- executable=None on some POSIX (SHELL unset): fall back to /bin/sh for the -c payload.

## Test plan
- Unit (sandbox.py, no OS dep): detect_backend honors override and platform branches via monkeypatched shutil.which + is_windows.
- Unit: bwrap builder emits --unshare-net iff allow_network is false; emits one --bind pair per write-root; includes --chdir cwd; appends extra_args before --.
- Unit: seatbelt profile contains deny-default, a file-write subpath line per root, and allow/deny network matching config; rejects roots with quotes.
- Unit: _build_sandbox_env drops API_KEY/TOKEN/AWS_* but keeps PATH/HOME and env_passthrough entries; passthrough only when present.
- Unit: _resolve_sandbox returns (None,'disabled') when enabled=false; raises ToolError when backend=='none' and require_backend=true; warns+returns (None,'none') otherwise.
- Integration (Linux CI w/ bwrap, skipif not which('bwrap')): enabled sandbox + allow_network=false: `curl`/`wget` to a host fails; a write to cwd succeeds; a write to /etc/<tmp> fails with permission/read-only error captured in stderr; secret env var (FAKE_API_KEY) is absent under scrub_env.
- Integration: sandbox enabled, write to a path listed in write_dirs succeeds; same path outside write_dirs fails.
- Regression: enabled=false reproduces exact current behavior (create_subprocess_shell still used) — assert via spy/patch that create_subprocess_exec is NOT called.
- Composition: a denylisted command (e.g. `vim`) still returns NEVER from resolve_permission regardless of sandbox config (sandbox never loosens denials).
- macOS (skipif not Darwin): sandbox-exec profile blocks write to /etc and allows write to cwd.
- Cleanup: seatbelt temp profile file is unlinked after run, including on timeout/exception.

## Risks
- False sense of security: textual guardrails run pre-spawn and are bypassable; the OS sandbox is the real boundary but is OFF by default, so users may believe they are protected when they are not. Mitigation: document clearly; consider a startup notice when bash runs unsandboxed in shared/CI contexts.
- Graceful-fallback silently runs unsandboxed when the binary/kernel feature is missing — an attacker-influenced model could rely on this. require_backend=true is the fail-closed answer but breaks portability; choose default carefully.
- Breaking legitimate workflows: read-only / network-off confinement can break git, package managers, build tools, and out-of-tree writes, generating confusing failures. Defaults (allow_network=true, write_dirs threading approved dirs) reduce but don't eliminate this.
- unshare fallback is materially weaker than bubblewrap (the ro-remount prelude is fragile); presenting it as 'sandboxed' may overstate protection. Consider treating unshare as best-effort only.
- Per-call ephemeral /tmp and fresh namespace change semantics for scripts that pass state through /tmp or background processes between bash calls.
- Performance: spawning bwrap per command adds setup latency; negligible for most but measurable in tight loops.
- Maintenance: three OS-specific code paths to keep working; CI coverage realistically only on Linux.

---
## Adversarial verification

**Verdict:** `sound_with_fixes`  |  **Feasible:** True

**Integration points exist:** Verified against actual code. vibe/core/tools/builtins/bash.py: _get_base_env (L73-87), _get_shell_executable (L67-70), _collect_outside_dirs (L193-236), _PATH_COMMANDS (L173-188), BashToolConfig (L243-266), resolve_permission (L451-480 incl. Windows early-return L452-453), and Bash.run subprocess block (L516-524 create_subprocess_shell with env=_get_base_env(), executable=_get_shell_executable(), start_new_session) all exist exactly as the spec describes. vibe/core/config/_settings.py: SafetyJudgeConfig is a BaseSettings with SettingsConfigDict(extra='ignore') ending at L238 — confirmed. vibe/core/config/__init__.py exports SafetyJudgeConfig (L30, L123) so a sibling export pattern is valid. vibe/core/tools/base.py: BaseToolConfig has model_config=ConfigDict(extra='allow') (L131); InvokeContext is a @dataclass (L44-74) with scratchpad_dir: Path|None=None (L58); run signature ctx: InvokeContext|None=None (L178) — all confirmed. vibe/core/scratchpad.py uses tempfile.mkdtemp and is_scratchpad_path (L12-31, L39). vibe/core/utils/platform.py is_windows (L25). agent_loop.py _resolve_safety_judge at L1014 and ASK-path wiring confirmed. kill_async_subprocess and is_windows are re-exported from vibe.core.utils (used by bash.py). No existing sandbox module. I additionally verified the config load path (manager.py get_tool_config L410-431) which the spec did NOT mention but which the design depends on.

**Wrong assumptions:**
- integration_points entry for InvokeContext says 'No new field strictly required' and that the approved OUTSIDE_DIRECTORY grants can be 'passed into the spec as extra write-roots' / 'resolve_permission already computed those dirs ... pass the same set into write_roots'. This is false as written: resolve_permission runs in agent_loop._should_execute_tool, entirely separate from invoke/run. The approved grants are persisted in self._permission_store keyed by glob session patterns ('<dir>/*'); they are NOT passed to the tool. InvokeContext (verified L44-74) carries no approved-permissions or outside-dirs field. So the grants do NOT flow to _resolve_sandbox as the spec implies. (Fixable: _resolve_sandbox has args.command and can re-derive via _collect_outside_dirs(_extract_commands(args.command)) itself — but that re-derivation is unspecified.)
- SandboxConfig is specified as a BaseSettings (like SafetyJudgeConfig). BaseSettings reads OS env vars at instantiation; with no env_prefix, fields like 'enabled'/'backend'/'write_dirs' could be silently populated from stray env vars named ENABLED/BACKEND/etc. SafetyJudgeConfig shares this latent issue, but for a security-sandbox config an env-driven 'enabled=false' or injected write_dir is a footgun. Spec doesn't set an env_prefix or env_ignore.
- Spec frames the config sub-key merge purely via BaseToolConfig extra='allow' being 'additive'. The real merge happens in two layers it never names: (1) vibe_schema.py tools field is dict[str,dict[str,Any]] with WithShallowMerge — so a project [tools.bash.sandbox] REPLACES (not deep-merges) a user [tools.bash.sandbox]; (2) manager.get_tool_config (L428) does {**default.model_dump(), **user_overrides} then config_class.model_validate. Both work for nested SandboxConfig (I tested round-trip with BaseSettings nested in a BaseModel field — passes), but the spec's stated reason is incomplete.

**Missing pieces:**
- Mechanism to thread approved OUTSIDE_DIRECTORY grants into write_roots. Either (a) _resolve_sandbox re-derives them from args.command via the existing _collect_outside_dirs(_extract_commands(args.command)) helpers (cleanest, no new wiring), or (b) add an InvokeContext field plus capture logic in agent_loop._should_execute_tool. The spec assumes the data is already available; it is not. Without one of these, every approved out-of-tree write silently fails under sandbox unless also listed in write_dirs.
- Spec does not specify that the unsandboxed/disabled path must remain byte-identical regarding executable=. Today create_subprocess_shell passes executable=_get_shell_executable() which may be None. Fine, but the regression test ('create_subprocess_exec is NOT called when disabled') must also assert env and executable unchanged.
- No handling for the SandboxConfig BaseSettings env-var leakage (recommend env_prefix or switching to plain BaseModel, since this config is never meant to be env-driven).
- Spec lists detect spawn-failure-and-fallback for bwrap (unprivileged userns disabled) as an edge case, but the algorithm (step 8) builds argv and spawns once with no try/except that distinguishes 'sandbox wrapper failed to init' from 'user command failed'. Needs an explicit spawn-time probe or a sentinel so fallback can trigger; run()'s current except-Exception wraps everything into a generic ToolError, which would mask it.

**Corrections to fold in:**
- Change integration_points InvokeContext entry: drop 'no new field required' framing. Specify that _resolve_sandbox re-derives approved outside-dirs locally via outside = _collect_outside_dirs(_extract_commands(args.command)) and adds each (and its parent) to write_roots — reusing helpers already in bash.py — rather than relying on grants flowing from resolve_permission. (resolve_permission only ASKs about those dirs; it does not store them on ctx.) Note: this widens write-roots to any outside dir the command references, slightly broader than 'only approved ones', so gate it or document it.
- Make SandboxConfig a plain pydantic BaseModel (not BaseSettings), OR give it model_config with an env_prefix that cannot collide (e.g. VIBE_SANDBOX_) and document it, to prevent stray env vars silently enabling/altering the sandbox. SafetyJudgeConfig's BaseSettings choice should not be copied uncritically for a security control.
- In the algorithm/edge-cases, add an explicit bwrap/unshare spawn-failure detection: attempt the sandboxed spawn, and if the wrapper itself errors (nonzero from bwrap before exec, or OSError/FileNotFoundError on the wrapper binary), and require_backend is false, warn-once and retry unsandboxed via the existing create_subprocess_shell path. Distinguish wrapper-init failure from user-command failure (e.g. bwrap returns 127/126 with a recognizable stderr marker, or probe bwrap --version once and cache).
- State the config-merge reality explicitly: nested [tools.bash.sandbox] merges through vibe_schema WithShallowMerge (project replaces user at the sandbox sub-key) and manager.get_tool_config shallow-dump-merge + model_validate. Confirmed working for nested config, but call it out so a reviewer doesn't expect deep per-field merge of the sandbox block across config layers.
- Add a regression assertion that on Windows and on the disabled path env is still _get_base_env() unscrubbed and create_subprocess_shell is used (scrub_env must be a no-op when sandbox.enabled is false, even if scrub_env defaults true).
