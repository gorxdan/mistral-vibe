# Shell sandbox

Status: Current runtime contract

The Bash tool uses an operating-system sandbox when a supported backend is
available. The sandbox is one layer in the tool authorization path:

1. Tool availability and the active runtime capability ceiling.
2. Immutable command and protected-path policy.
3. Tool permission and, for `ASK` calls, the safety judge or user decision.
4. Sandbox construction and process launch.

A later layer cannot reverse an earlier denial. In particular, auto-approve
does not bypass a `NEVER` permission, protected host state, the managed tool
catalog, a configured safety-judge deferral, or a Bash explicit-user gate.

This document covers model-invoked Bash and the separate trusted-check runner.
The [harness integrity contract](harness-integrity.md) defines the full managed
maintenance boundary.

## Runtime modes

| Mode | Backend requirement | Writable filesystem | Network | Environment and Git |
|---|---|---|---|---|
| Ordinary session, sandbox enabled | Best available backend; may fall back | Current workspace, session scratchpad, configured `write_dirs`, permission-derived outside directories, and a private Vibe tool cache | Controlled by `allow_network` only on a capable backend | Scrubbed by default; host Git, SSH, and GitHub variables are retained for the user's session |
| Ordinary session, sandbox disabled | None | Host process permissions | Host process access | Existing unsandboxed shell environment |
| Auto-approve | Bubblewrap on Linux or Seatbelt on macOS; otherwise fail closed | Current workspace, session scratchpad, and configured `write_dirs` | Disabled | Strict scrubbed environment and disposable caches; an interactive root user may explicitly approve Git commits |
| Isolated or task-contracted writer | Bubblewrap or Seatbelt; otherwise fail closed | Assigned isolated root and scratchpad | Disabled | Strict scrubbed environment and disposable caches; Git administration is read-only and host delivery creates the commit |
| Topology-bound model Bash | Bubblewrap or Seatbelt; otherwise fail closed | Session scratchpad only | Disabled | Strict scrubbed environment and disposable caches; candidate, control, evidence, Git administration, logs, and receipts are read-only |
| Receipt-authorizing trusted check | Linux Bubblewrap only; every other platform/backend fails | A new per-check run directory only | Disabled | Direct `argv` against an exact-HEAD Git-exported snapshot with no Git metadata, private copy of a pinned native executable, host environment attestation, scrubbed offline environment, disposable home and caches |

The current workspace remains writable in ordinary auto-approve mode. That mode
hardens process execution but does not turn an ordinary session into a managed
maintenance session. A topology-bound worker changes candidate files through
`edit` and `write_file`, whose path checks enforce the frozen recipe. Bash has
no writable candidate bind in that mode. A topology-bound verification root has
neither file-writing tool.

Authorization origin controls the child environment independently of runtime
mode. Policy-, Safety-Judge-, and bypass-authorized core Bash calls use a trusted
system `PATH` and minimal noninteractive environment without ambient
credentials, loader settings, Python paths, or configured passthrough. Human and
stored-human approvals use the configured compatibility environment, scrubbed by
default and retaining the remaining host environment only when
`sandbox.scrub_env = false`.
Minimal environment selection alone does not make a non-strict call require a
sandbox; the strict modes in the table do.

ACP preserves this boundary in two paths. Automated policy, Safety-Judge, and
bypass calls use core-managed Bash. Human and stored-human approvals are
revalidated immediately before launch and then use the editor client's terminal
at the process-bound workspace. One ACP process accepts multiple sessions only
for the same canonical cwd and exact additional-directory request; a different
workspace is rejected before process cwd or harness roots change. Full
concurrent multi-workspace ACP requires separate server processes.

## Backends and fallback behavior

`backend = "auto"` selects Bubblewrap (`bwrap`) on Linux when usable, then the
Linux `unshare` fallback, or Seatbelt (`sandbox-exec`) on macOS. Unsupported
platforms resolve to no backend.

Backend binaries and the Bubblewrap probe helper are resolved from the sanitized
system path and cached as absolute paths. The builder receives that resolved
identity and emits it directly, so ambient or project `PATH` changes cannot swap
the wrapper between capability detection and launch.

On Linux, automated execution also requires a root-owned regular executable with
no set-ID or group/world-write bits and no write access for the current non-root
user. Both its lexical and resolved path must have root-controlled ancestry that
is non-group/world-writable and non-writable by that user. Project, user,
symlink-ambiguous, or otherwise unprovable executable graphs require explicit
user approval. Automated Git inspection is limited to exact standalone `git
log`, `show`, `blame`, and `grep` forms, which are rewritten with fixed config and
renderer-disabling flags. Worktree `diff`/`status` and unhardenable Git forms need
explicit user approval.

- Bubblewrap mounts the host filesystem read-only, adds only the selected write
  roots, creates a fresh `/tmp`, and can create a network namespace. Its seccomp
  filter blocks selected high-risk syscall families when `seccomp = true`.
- Seatbelt permits filesystem reads, grants writes only to selected roots, and
  applies the configured network rule.
- `unshare` supplies namespace isolation but does not enforce filesystem write
  confinement or the requested network rule. Vibe warns when it is used. It is
  never accepted for auto-approve, isolated/task-contracted, topology-bound, or
  trusted-check execution.
- In an ordinary non-strict session, no backend or a wrapper launch failure may
  fall back to an unsandboxed process unless `require_backend = true`. The
  already-scrubbed environment is preserved during wrapper fallback.
- Strict model Bash requires Bubblewrap or Seatbelt even when
  `sandbox.enabled = false`. Missing or failed confinement stops the Bash call.
  Receipt-authorizing trusted checks are narrower and require Linux Bubblewrap;
  Seatbelt is not a supported trusted-check backend.

Bubblewrap `extra_args` is retained for narrow namespace/runtime compatibility;
it is not a path grant or filesystem-policy extension. Vibe rejects mount,
overlay, tmpfs, device, proc, chdir, environment, argv, argument-file, and
command-terminator flags before constructing the wrapper. Host policy mounts are
appended after accepted arguments. The weaker `unshare` backend rejects all
`extra_args`. Do not use `extra_args` to alter the command graph or add access.

## Configuration

The sandbox defaults on:

```toml
[tools.bash.sandbox]
enabled = true
write_dirs = []
allow_network = true
scrub_env = true
env_passthrough = []
require_backend = false
backend = "auto"       # auto | bwrap | unshare | sandbox-exec | none
seccomp = true          # Bubblewrap only
```

These settings describe ordinary sessions. Strict runtime modes override them:
they disable network, ignore ordinary permission-derived write roots and
persistent caches, use a scrubbed environment, and require a confining backend.
Topology-bound Bash also ignores configured `write_dirs`.
Strict model control rejects `background=true`; a managed Bash process must
finish within the foreground sandbox and cannot enter the background registry.

Ordinary background-process teardown signals a group only when the tracked
child is verified as both the session leader and process-group leader. A process
that shares its caller's session is terminated directly. Signal-escalation
tests mock the operating-system signal calls; destructive process-tree probes
are manual isolated checks and are not part of the parallel test suite.

`env_passthrough` should contain variable names, never values. It applies to the
ordinary scrubbed environment. Strict modes do not pass configured environment
exceptions through to model processes.

Startup and injection variables such as `BASH_ENV`, exported Bash functions,
`LD_PRELOAD`, `LD_AUDIT`, and `DYLD_INSERT_LIBRARIES` are always stripped.
Automated calls inherit no ambient loader configuration. The default human
compatibility scrub also removes loader variables; they survive only when
`sandbox.scrub_env = false`.

## Protected state

Writable ordinary roots are layered with read-only protection for Vibe config
and environment files. Git hooks and Git configuration remain read-only even
when ordinary Git metadata is writable. The Bash hard-policy layer also blocks
worktree administration, ref updates, destructive reset/clean operations, and
host-path deletion independently of wrapper construction.

Managed topology adds the configured control worktree, durable evidence root,
Git common directory, host logs, and receipt storage to protected paths. The
same roots are enforced by model file tools, not only Bash. A broad packet path
glob or auto-approve setting cannot reopen them.

## Trusted checks

Trusted checks do not reuse the Bash tool configuration. The host runner:

- accepts a frozen nonempty argument array and invokes it with `shell=False`;
- rejects a shell or `env` as the executable and rejects shells or `env` behind
  `uv run`;
- requires Linux Bubblewrap with no Seatbelt, `unshare`, or unsandboxed fallback;
- resolves a pre-provisioned absolute executable or bare name on the sanitized
  system path, rejects candidate-owned/bootstrap executables, descriptor-validates
  its configured `executable_sha256`, copies the verified native executable to a
  private runner-owned path, executes only that read-only copy, and preserves
  the configured executable path as `argv[0]` for interpreter prefix discovery;
- rejects shebang wrappers; recipes invoke a pinned native interpreter directly
  and pass `-m <module>` or a script path as arguments;
- requires a host-owned environment-attestation file and verifies its configured
  digest before and after the check;
- rejects `uv`/`pre-commit` bootstrap entrypoints and runs with offline package
  and cache settings; all dependencies must already exist in host-owned runtime
  roots;
- exports the exact candidate `HEAD` from Git objects into a detached frozen
  source snapshot, verifies its tree before and after execution, and exposes no
  original candidate, Git common directory, refs, history, or `.git` metadata;
- gives the process only a disposable writable run directory, home, temp, and
  cache tree;
- disables network and strips host credentials and user Git configuration;
- caps combined stdout and stderr at 1 MiB, terminating the process tree when
  the cap is exceeded; and
- captures repository state before and after all checks, failing the receipt if
  the candidate is dirty, outside the allowed path set, or changed by a check.

The executable copy closes the source-path swap window for the object executed
by the check. It does not freeze the executable's dynamic loader, shared
libraries, language packages, or other runtime roots. Those roots stay visible
read-only inside the check sandbox but remain host-owned objects. The environment
attestation is a pre/post-checked host assertion about that provisioned state;
it is not a transitive hash of the dependency tree. Receipt-authorizing hosts
must provision those roots immutably or exclude concurrent writers while a
check runs. A native binary that derives resources from its physical copied
location rather than `argv[0]` needs a separately attested deployment layout.

The runner sends a process-group signal only after verifying at signal time that
the child PID is both the process-group leader and session leader; otherwise it
signals only the direct child. Default and xdist tests mock these OS calls.
Real process-tree escalation probes are manual checks for disposable isolated
hosts, never graphical login sessions. They are marked `process_e2e`, skipped
by default, and require
`VIBE_PROCESS_E2E_DISPOSABLE=1 uv run pytest -n0 --run-process-e2e ...`.
Cleanup of a deliberately detached descendant that exits the tracked tree
before observation is still an open hardening item. Trusted recipes must
therefore use bounded test, lint, and analysis commands, not daemon launchers.

## Operational guidance

- Use Bubblewrap on Linux hosts that run auto-approved, isolated, or managed
  work. Treat `unshare` as a warning-only compatibility fallback for ordinary
  interactive sessions.
- Set `require_backend = true` when an ordinary session must not fall back.
- Keep `write_dirs` narrow. It grants write access to the entire named tree in
  ordinary and ordinary auto-approved sessions.
- Do not treat ordinary sandboxing as a confidentiality boundary. Most host
  files remain readable, although read-only, and network is allowed by default.
- Use a host-controlled trusted verification recipe and execution topology when
  candidate identity, evidence durability, and landing authority matter.
