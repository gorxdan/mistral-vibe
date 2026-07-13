---
packet_schema: 1
id: I00-P03
title: Programmatic CLI subprocess characterization
iteration: 0
state: draft
change_class: characterization
risk: medium
owner: null
reviewer: null
verifier: null
evidence_operator: null
depends_on:
  - I00-P01
  - I00-P04
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
  scenarios:
    - IT-01
packet_acceptance_criteria:
  - I00-P03-AC01
  - I00-P03-AC02
  - I00-P03-AC03
  - I00-P03-AC04
  - I00-P03-AC05
  - I00-P03-AC06
  - I00-P03-AC07
  - I00-P03-AC08
  - I00-P03-AC09
  - I00-P03-AC10
  - I00-P03-AC11
  - I00-P03-AC12
  - I00-P03-AC13
  - I00-P03-AC14
  - I00-P03-AC15
  - I00-P03-AC16
  - I00-P03-AC17
  - I00-P03-AC18
  - I00-P03-AC19
  - I00-P03-AC20
roadmap_contributions:
  - AC-1.2
  - AC-1.3
  - AC-3.1
  - AC-3.5
  - AC-7.1
messages:
  - MSG-01
  - MSG-02
  - MSG-03
  - MSG-04
paths:
  allowed:
    - tests/e2e/test_cli_programmatic.py
  forbidden:
    - vibe/**
    - tests/e2e/conftest.py
    - tests/e2e/common.py
    - tests/e2e/mock_server.py
    - tests/snapshots/**
    - pyproject.toml
    - uv.lock
    - docs/**
---

# I00-P03: Programmatic CLI Subprocess Characterization

Execution state is read only from this packet's frontmatter and the matching
`../status.yaml` entry at the assigned clean `CONTROL_SHA`; this prose does not
duplicate state. Implementation is authorized only when both record `ready`,
I00-P01 and I00-P04 are `complete`, and the lead has filled every required field.

## Outcome

Add one E2E test module that invokes the real `vibe` console entry point in
programmatic mode and freezes six current delivery contracts: parser-only help
and version, deterministic success, missing API key, backend authentication
failure, invalid configuration, and untrusted-workspace warning behavior.

Only the new test file may change. If current production behavior fails a named
contract, the packet stops and reports the product gap; it does not repair
production code, weaken the assertion, xfail/skip the case, or record traceback
leakage as an acceptable baseline.

## Why this packet exists

Existing tests cover programmatic functions and CLI wiring mostly in-process,
and the bundled-binary smoke test covers missing-key behavior. They do not
provide one deterministic source-tree subprocess journey that proves stdout,
stderr, exit status, runtime isolation, request count, config failure, and trust
warning behavior together. The roadmap explicitly names
`tests/e2e/test_cli_programmatic.py` as a missing Iteration 0 deliverable.

I00-P01 is required so MSG-01/MSG-02 output and exit-code evidence can be stored
outside the candidate and compared in later iterations.

## Definition of Ready

The lead checks every item before setting `ready`:

- [ ] I00-P01 and I00-P04 are complete; the evidence runner passes and the real
      backend-auth subprocess meets case 4 without a traceback.
- [ ] The packet starts from a clean full-history commit with frozen baseline
      and upstream SHAs.
- [ ] Owner, reviewer, verifier, evidence operator, isolated worktree, branch,
      execution profile, evidence workspace, run ID, and runner ID are assigned.
- [ ] No other active packet modifies the one allowed file or writes IT-01.
- [ ] The current entry point, existing unit tests, config schema, mock server,
      and safe I/O helper were reread at the frozen baseline.
- [ ] I00-P04's candidate is included in this packet baseline and its verifier
      PASS is current.
- [ ] All six expected stream/exit contracts below are frozen as characterization
      targets; no ambiguous “actionable error” judgment remains with the worker.

## Frozen lead decisions

- Delivery form: every case invokes `uv run --no-sync vibe` as a subprocess.
  Never call or monkeypatch `entrypoint.main`, `run_cli`, or `run_programmatic`.
- Scope: create exactly one test file. Shared fixtures/helpers and production
  files remain unchanged even if duplication is locally less elegant.
- Provider: deterministic generic provider/model on a loopback server only.
- Test key: `MISTRAL_VIBE_E2E_API_KEY=fake-test-key`; never use normal provider
  credentials or inherit them into the child.
- Worktree behavior: child config sets `[worktree] mode = "off"` and every
  programmatic invocation includes `--no-worktree`.
- Trust: all runtime cases pass `--trust` except the one case that characterizes
  the nonfatal untrusted-workspace warning.
- Normalization: expected literals remain exact. Only the resolved temporary
  workdir may become `<WORKDIR>` and path separators may normalize to `/` in the
  trust warning.
- Network: no external network and no paid model. Local servers bind
  `127.0.0.1` on an ephemeral port.
- Snapshots: expected strings live in the test module; add no golden/snapshot
  file.
- Product behavior: this is characterization. Any required production fix is a
  new packet with its own rollback and message review.
- Landing: the worker may not land, push, merge, or mark complete.

## Worker discretion

The worker may choose private test-helper names and whether closely related
cases use a small local dataclass, context manager, or fixture. The six cases,
child environment, config values, subprocess invocation, expected streams, and
allowed path are not discretionary.

## Scope

### In scope

- Create `tests/e2e/test_cli_programmatic.py`.
- Write test-local config and invalid-config files with
  `vibe.core.utils.io.write_safe`.
- Reuse `tests.e2e.mock_server.StreamingMockServer` for success/untrusted cases.
- Implement a test-local loopback HTTP server that returns one deterministic
  HTTP 401 response for the backend-auth case.
- Capture exact stdout, stderr, return code, backend request count/payload, trust
  persistence, and absence of traceback/onboarding/splash/ANSI leakage.
- Isolate each test-local CLI child in a Linux process group and prove bounded
  SIGTERM/SIGKILL cleanup for timeout and leaked-descendant fixtures.
- When `VIBE_MAINTENANCE_TEST_EVIDENCE` is set to the assigned IT-01 directory,
  write the four structured per-case artifacts defined below with project safe
  I/O. This test-only hook is the frozen artifact producer for this packet.

### Out of scope

- JSON/streaming/stdin output matrices, tool calls/approval, sessions/resume,
  product signal-delivery behavior, worktree creation, malformed TOML syntax,
  TUI, ACP, performance, or installed-wheel behavior. Test-local process-group
  cleanup signals required by AC20 remain in scope.
- Production error-boundary fixes, message rewrites, retries/failover changes,
  shared E2E fixture cleanup, or mock-server enhancement.
- Live provider calls, DNS, non-loopback sockets, paid execution, proxy use,
  update/model discovery, telemetry, experiments, memory, session logging, or
  sandbox behavior.

## Allowed paths

- `tests/e2e/test_cli_programmatic.py` — new subprocess characterization only.

Runtime writes are limited to the pytest `tmp_path` roots, local loopback socket
state, and the assigned external `$EVIDENCE/IT-01` directory when the explicit
test evidence environment variable is present. The test must not write the
repository, real user home, or default `~/.vibe`.

## Forbidden paths and actions

- `vibe/**` — no production fix or instrumentation.
- `tests/e2e/conftest.py`, `tests/e2e/common.py`, and
  `tests/e2e/mock_server.py` — use but do not modify.
- `tests/snapshots/**`, `docs/**`, `pyproject.toml`, and `uv.lock`.
- No monkeypatch of child production symbols, in-process entry-point call,
  network mocking that bypasses the real HTTP client, or broad output filtering.
- No xfail, skip, retry decorator, loose substring-only replacement of exact
  contracts, or assertion deletion to make current behavior green.

## Required reading and inputs

Read before editing:

- `AGENTS.md`
- `openwiki/quickstart.md`
- Roadmap: “Behavior-preserving change,” “Iteration 0,” “User-facing message
  inventory,” “Message comparison rules,” and IT-01.
- `vibe/cli/entrypoint.py`
- `vibe/cli/cli.py`, especially `load_config_or_exit` and
  `_run_programmatic_mode`.
- `vibe/core/programmatic.py`
- `vibe/core/config/models.py` and relevant fields in
  `vibe/core/config/_settings.py`.
- `tests/e2e/conftest.py`, `tests/e2e/common.py`, and
  `tests/e2e/mock_server.py`.
- `tests/cli/test_cli_wiring.py`, `tests/cli/test_programmatic_setup.py`, and the
  programmatic missing-key case in `tests/cli/smoke_binary.py`.

Lead-filled inputs:

```bash
CONTROL_WORKTREE=<absolute clean control worktree>
CONTROL_SHA=<immutable control commit supplied by the lead>
BASELINE_SHA=<frontmatter baseline_sha>
UPSTREAM_SHA=<frontmatter upstream_sha>
VIBE_EVIDENCE_WORKSPACE=<frontmatter evidence.workspace>
KILROY_RUN_ID=<frontmatter evidence.run_id>
EVIDENCE="$VIBE_EVIDENCE_WORKSPACE/.ai/runs/$KILROY_RUN_ID/test-evidence/latest"
RUNNER_ID=<assigned stable runner label>
REPO_ROOT=<absolute candidate worktree root>
```

## Preflight

Run without editing:

```bash
test -n "$CONTROL_WORKTREE"
test -n "$CONTROL_SHA"
test -n "$PATH"
test -n "$HOME"
test "$(uv run git -C "$CONTROL_WORKTREE" status --short)" = ""
test "$(uv run git -C "$CONTROL_WORKTREE" rev-parse HEAD)" = "$CONTROL_SHA"
test -n "$BASELINE_SHA"
test -n "$UPSTREAM_SHA"
test -n "$VIBE_EVIDENCE_WORKSPACE"
test -n "$KILROY_RUN_ID"
test -n "$RUNNER_ID"
test -n "$REPO_ROOT"
test "$(uname -s)" = "Linux"
test "$(GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_OPTIONAL_LOCKS=0 uv run git status --porcelain=v1 --untracked-files=all --ignore-submodules=none)" = ""
test "$(uv run git rev-parse --show-toplevel)" = "$REPO_ROOT"
uv run git rev-parse HEAD
uv run git rev-parse "$BASELINE_SHA^{commit}"
uv run git rev-parse "$UPSTREAM_SHA^{commit}"
uv run git worktree list --porcelain
test -f scripts/run_maintenance_evidence.py
test -f tests/maintenance/test_evidence_contract.py
test ! -e tests/e2e/test_cli_programmatic.py
uv run --no-sync vibe --version
```

Required results:

- Status is empty and `HEAD` equals `BASELINE_SHA` at packet start.
- Both commits resolve and I00-P01 deliverables exist.
- The allowed test path does not already contain other work.
- The entry point runs without synchronizing the environment.
- The assigned execution host/profile is Linux; another platform requests
  `ready -> blocked` before edits.
- The evidence root is external and IT-01 is unclaimed by another writer.

Any mismatch requests `blocked` with no edits.

Run the canonical control-metadata validator in
`../task-packet-template.md` with `PACKET_ID=I00-P03` and
`PACKET_RELATIVE_PATH=docs/design/fork-maintenance/packets/I00-P03-programmatic-cli-characterization.md`.
An assertion failure requests `ready -> blocked` with no edits.

## Deterministic test setup

### Test-local config writer

Use `write_safe` to create `$VIBE_HOME/config.toml` with these exact semantics:

```toml
active_model = "mock-model"
enable_update_checks = false
enable_telemetry = false
enable_otel = false
api_retry_max_elapsed_time = 0

[[providers]]
name = "mock-provider"
api_base = "<loopback server API base>"
api_key_env_var = "MISTRAL_VIBE_E2E_API_KEY"
backend = "generic"
discover_models = false

[[models]]
name = "mock-model"
provider = "mock-provider"
alias = "mock-model"

[experiments]
enable = false

[memory]
enabled = false

[session_logging]
enabled = false

[worktree]
mode = "off"

[tools.bash.sandbox]
enabled = false
```

Do not call the shared `write_e2e_config`: its smaller config does not freeze the
side-effect and retry controls required by this packet.

### Filesystem isolation

Each case creates fresh sibling paths under `tmp_path`:

```text
tmp_path/
├── vibe-home/
├── workdir/
└── user-home/
```

The test creates directories explicitly. It never reads or writes the real user
home. The untrusted case creates `workdir/AGENTS.md` with `write_safe`; all other
workdirs contain no project configuration file.

### Child environment

Start from `os.environ.copy()`, then:

1. Remove every inherited name beginning `VIBE_`.
2. Remove `PYTHONPATH`.
3. Remove known provider credentials, including at minimum `MISTRAL_API_KEY`,
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`,
   and every inherited name ending `_API_KEY`.
4. Remove upper/lowercase `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY`.
5. Set only these test controls:

```text
VIBE_HOME=<tmp_path/vibe-home>
VIBE_TEST_DISABLE_KEYRING=1
HOME=<tmp_path/user-home>
USERPROFILE=<tmp_path/user-home>
XDG_CONFIG_HOME=<tmp_path/user-home/.config>
XDG_CACHE_HOME=<tmp_path/user-home/.cache>
XDG_DATA_HOME=<tmp_path/user-home/.local/share>
NO_COLOR=1
TERM=dumb
COLUMNS=4096
LINES=50
PYTHONHASHSEED=0
NO_PROXY=127.0.0.1,localhost
no_proxy=127.0.0.1,localhost
```

Add `MISTRAL_VIBE_E2E_API_KEY=fake-test-key` only to cases that require a key.
The fixed wide terminal prevents Rich from wrapping single-line warning/error
contracts in captured non-TTY output. Before invocation, assert that the
resolved workdir string is at most 512 characters; a longer fixture path is a
setup failure, not a skip or normalization opportunity. Do not print the child
environment in test failure messages.

`VIBE_MAINTENANCE_TEST_EVIDENCE`, when present in the outer pytest process, is
never forwarded to a `vibe` child because the child scrub removes every
inherited `VIBE_` name before adding the fixed child allowlist.

### Subprocess helper

Invoke from `tests.TESTS_ROOT.parent` with:

```text
uv run --no-sync vibe
  --workdir <workdir>
  --no-worktree
  [--trust]
  <case arguments>
```

Use `subprocess.Popen` with an argument list, `shell=False`,
`stdin=subprocess.DEVNULL`, text mode, captured stdout/stderr, and
`start_new_session=True`; inspect `returncode` explicitly rather than using
automatic exit checking. Freeze 30 seconds as the production timeout and
five seconds as the termination grace period in private constants; a private
timeout argument is allowed only for the hermetic cleanup test. On Linux the
new session makes the child PID the process-group ID, isolating `uv`, `vibe`,
and any descendant from pytest and the outer evidence runner.

Call `communicate(timeout=...)`. Expected cases must not time out. On
`subprocess.TimeoutExpired`, send SIGTERM to the whole group, wait at most five
seconds, send SIGKILL if any group member remains, reap the direct child, assert
that the group no longer exists, and raise
`AssertionError("CLI subprocess timed out after <SECONDS> seconds; process group terminated.")`
from the timeout. Substitute the timeout using compact decimal formatting, so
the production value is `30` and the test value is `0.1`. Do not retry.

After every normal `communicate` return, probe the group before returning the
captured result. If a descendant remains after the direct child exits, apply
the same bounded SIGTERM/SIGKILL cleanup, assert the group is gone, and raise
`AssertionError("CLI subprocess left descendants after exit; process group terminated.")`.
This normal-exit probe is required because the new nested session is outside
I00-P01's outer pytest process group.

Add one hermetic parametrized cleanup test. Launch a synthetic
`uv run --no-sync python -c <script>` through the same helper. In one parameter,
the direct child and descendant outlive a 0.1-second injected timeout. In the
other, the direct child spawns a long-lived descendant that closes inherited
stdio and then exits 0. Record the descendant PID in a temporary file with
`write_safe`, read it with `read_safe`, and require the matching exact assertion,
no live process group, and no live descendant after the five-second grace. The
test must never signal pytest's process group.

### Loopback servers

- Success and untrusted cases reuse `StreamingMockServer`, always stopping it in
  `finally` or a context manager. The default response is
  `Hello from mock server`.
- The backend-auth case defines a test-local `ThreadingHTTPServer` bound to
  `127.0.0.1:0`. It accepts only the chat-completions path, increments a
  thread-safe request count, suppresses server logs, and responds HTTP 401 with
  UTF-8 JSON `{"error":{"message":"invalid test credentials"}}`.
- Server shutdown closes the socket and joins the thread even when an assertion
  fails.
- Success/auth-failure cases assert exactly one backend request per attempt. Any
  request to another host, an unexpected path, or a second model request within
  an attempt fails the packet.

### Structured evidence and reproducibility

Each logical journey runs twice inside its named test using fresh temporary
homes and, when applicable, a fresh server. “Help” and “version” are recorded as
separate journeys even though they share one test function. Before asserting
the expected contract, retain both attempts in a local result object so an
unexpected result can still be written as best-effort evidence.

When `VIBE_MAINTENANCE_TEST_EVIDENCE` is unset, no structured artifact is
written. When set, it must resolve to the assigned absolute `$EVIDENCE/IT-01`
directory outside the repository. After each logical journey, update these
files with `read_safe`/`write_safe` and
`FileLock(lock_path, timeout=PROGRAMMATIC_EVIDENCE_LOCK_TIMEOUT_SECONDS)`, where
the private constant is exactly `10` and `lock_path` is exactly
`$EVIDENCE/IT-01/.programmatic-evidence.lock`:

- `stdout.json`
- `stderr.json`
- `exit-codes.json`
- `reproducibility.json`
- `.programmatic-evidence.lock`

The first three are JSON objects keyed by
`help`, `version`, `success`, `missing_key`, `backend_auth`, `invalid_config`,
and `untrusted`. Stream entries contain `raw` and `normalized` two-element
arrays, one per attempt; only the trust-warning workdir/path separator differs
between raw and normalized data. Exit entries are two-element integer arrays.
Reproducibility entries record whether normalized stdout, normalized stderr,
and exit code match between attempts, plus an overall boolean. Never write the
child environment, key, full request payload, or server endpoint.

Files are UTF-8, sorted by key, indented, newline-terminated, and rewritten
atomically. The lock allows a diagnostic direct invocation with xdist to avoid
lost updates, although the required evidence command remains `-n0`. Allow a
private writer-helper timeout argument only as a test seam. Catch
`filelock.Timeout` and raise
`AssertionError("Timed out after 10 seconds acquiring the programmatic evidence lock.")`
from it. Do not write a partial JSON file after timeout. The outer runner then
records the nonzero child result and missing required artifacts as `fail`; a
direct diagnostic terminates instead of waiting indefinitely.

Add a hermetic contention test in this module. Hold a temporary lock, call the
writer helper with a 0.01-second injected timeout, require the exact assertion
within one second, and separately assert that the production default equals 10
seconds. The temporary contention test never uses the assigned IT-01 artifact
directory.

## Exact test cases

### 1. `test_help_and_version_exit_without_runtime_setup`

Use an empty Vibe home containing `config.toml` with exact bytes
`providers = "not-a-list"\n`; omit every provider key.

- `--version`: exit 0, stdout exactly `vibe {__version__}\n`, stderr empty.
- `--help`: exit 0, stderr empty; stdout starts `usage: vibe`, contains
  `-p [TEXT], --prompt [TEXT]`, `--output {text,json,streaming}`, and `VIBE_HOME`;
  it contains neither `Traceback` nor onboarding text.
- These calls omit programmatic `--workdir/--no-worktree` because argparse must
  exit before runtime setup; use only `uv run --no-sync vibe --version` and
  `uv run --no-sync vibe --help` with the isolated child environment.

### 2. `test_programmatic_prompt_prints_only_final_response`

For each of two attempts, run with valid config/key/server, `--trust`, and
`-p "Return the fixture greeting"`.

- Exit 0.
- Stdout exactly `Hello from mock server\n`.
- Stderr exactly empty.
- No ANSI, splash, warning, onboarding, or traceback in either stream.
- Exactly one backend request per attempt; `model == "mock-model"`,
  `stream is False`, and the final user message content equals the prompt
  exactly.

### 3. `test_programmatic_missing_api_key_fails_without_onboarding`

Write valid config pointing to a started loopback request-counting sentinel, but
omit the test key. The failure must precede dispatch.

- Exit 1.
- Stdout empty.
- Stderr exactly:

```text
Error: Missing MISTRAL_VIBE_E2E_API_KEY environment variable for mock-provider provider. Set the environment variable (e.g. in ~/.vibe/.env or your shell), or run `vibe --setup` once interactively.
```

The actual string ends with one newline. No traceback, `Welcome to Mistral Vibe`,
setup prompt, or key value appears.

### 4. `test_programmatic_backend_auth_failure_is_actionable`

For each of two attempts, run with valid config/key and a fresh deterministic
401 server.

- Exactly one request per attempt.
- Exit 1 and stdout empty.
- Normalized stderr exactly:

```text
Error: API error from mock-provider (model: mock-model): Invalid API key. Please check your API key and try again.
```

The actual string ends with one newline. Neither stream contains traceback,
ANSI, raw request JSON, `fake-test-key`, endpoint/port, or temporary path.

Static inspection found that this case exposes an uncaught
`UnclassifiedBackendError`: `_run_programmatic_mode` catches
`RuntimeError | ValueError`, while the typed backend error derives directly from
`Exception` in the planning snapshot. I00-P04 exists to close that boundary
before this packet starts. Any traceback or different boundary here is a
regression against the completed dependency: stop, report the regression, and
do not change the expected string or touch `vibe/**`.

### 5. `test_programmatic_invalid_config_is_bounded`

Use semantic invalid TOML `providers = "not-a-list"`; do not use malformed TOML
syntax in this packet.

- Exit 1.
- Stderr empty.
- With `NO_COLOR=1`, stdout exactly:

```text
Invalid configuration (1 error(s)):
  - providers: Input should be a valid list
```

The actual string ends with one newline. No traceback or onboarding appears.

### 6. `test_programmatic_untrusted_workspace_warns_and_completes`

For each of two attempts, create `workdir/AGENTS.md`, omit `--trust`, and
otherwise use a fresh success config/key/server.

- Exit 0 and stdout exactly `Hello from mock server\n`.
- After replacing the resolved workdir with `<WORKDIR>` and `\\` with `/`,
  stderr exactly:

```text
Warning: <WORKDIR> is not trusted; project configuration (AGENTS.md) will be ignored. Re-run with --trust to trust this folder temporarily.
```

The actual string ends with one newline. Exactly one backend request occurs per
attempt; no trust dialog, onboarding, splash, ANSI, or traceback appears. The
isolated Vibe home contains no persisted trusted-folder entry for the workdir.

## Behavioral and structural invariants

- The real console entry point, config loader, HTTP backend, and programmatic
  output path execute in a child process.
- Success prints only the final response and makes one model request.
- Expected failures are bounded, noninteractive, secret-free, and traceback-free.
- Untrusted programmatic operation warns and succeeds; it does not prompt or
  persist trust.
- Help/version short-circuit before config/key/runtime setup.
- No production/shared fixture/snapshot/config/lockfile change.
- No external network, update/model discovery, telemetry, experiment, memory,
  session-log, worktree, sandbox, or paid-provider side effect.
- The new characterization adds no production hot-path work and changes no
  performance threshold or harness.

## User-facing and model-visible messages

| Message ID | Trigger | Expected contract | Allowed normalization | Evidence |
|---|---|---|---|---|
| MSG-01 | `--help`, `--version`, programmatic success, backend failure | Exact streams and exit codes stated in cases 1, 2, and 4 | None | `$EVIDENCE/IT-01/{stdout.json,stderr.json,exit-codes.json}` |
| MSG-02 | Missing key, invalid config, untrusted project config | Exact streams and exit codes stated in cases 3, 5, and 6 | Workdir/path separator only for trust warning | Same plus command log/JUnit |
| MSG-03 | Test-local evidence lock cannot be acquired within 10 seconds | `Timed out after 10 seconds acquiring the programmatic evidence lock.` | None | JUnit and outer runner result/missing-artifact evidence |
| MSG-04 | Test-local CLI child times out or leaves a descendant | `CLI subprocess timed out after <SECONDS> seconds; process group terminated.` or `CLI subprocess left descendants after exit; process group terminated.` according to the trigger | Numeric timeout substitution only | Cleanup-test JUnit and outer runner result |

MSG-03 and MSG-04 are packet-local test-harness diagnostics, not product-facing
or model-visible messages. No model-visible tool result changes. The success
response is deterministic fixture content, not a changed prompt/tool contract.

## Acceptance criteria

| ID | Criterion | Proof |
|---|---|---|
| I00-P03-AC01 | The version subprocess result equals `{returncode: 0, stdout: "vibe <version>\n", stderr: ""}`. | Case 1 version journey |
| I00-P03-AC02 | The help projection equals the frozen return code, stderr, prefix, required-marker set, and forbidden-marker set. | Case 1 help journey |
| I00-P03-AC03 | The success subprocess result equals `{returncode: 0, stdout: "Hello from mock server\n", stderr: ""}`. | Case 2 streams/exit |
| I00-P03-AC04 | The success request projection equals one object with the frozen model, stream flag, and final user content. | Case 2 server record |
| I00-P03-AC05 | The missing-key subprocess result equals the frozen `{returncode, stdout, stderr}` object. | Case 3 streams/exit |
| I00-P03-AC06 | Missing-key invocation performs zero backend dispatches. | Case 3 sentinel/no server request |
| I00-P03-AC07 | The backend-auth subprocess result equals the frozen `{returncode, stdout, stderr}` object. | Case 4 streams/exit |
| I00-P03-AC08 | Backend-auth invocation sends exactly one request. | Case 4 server count |
| I00-P03-AC09 | The backend-auth forbidden-leak intersection with `{traceback, key, request body, endpoint, temporary path}` is empty. | Case 4 negative-leak assertions |
| I00-P03-AC10 | The invalid-config subprocess result equals the frozen `{returncode, stdout, stderr}` object. | Case 5 streams/exit |
| I00-P03-AC11 | The invalid-config forbidden-marker intersection with `{traceback, onboarding}` is empty. | Case 5 negative assertions |
| I00-P03-AC12 | The untrusted subprocess result equals the frozen `{returncode, stdout, normalized_stderr}` object. | Case 6 streams/exit |
| I00-P03-AC13 | Untrusted invocation sends exactly one backend request. | Case 6 server count |
| I00-P03-AC14 | Untrusted invocation creates no persisted trust entry. | Case 6 isolated home inspection |
| I00-P03-AC15 | Each logical journey's attempt-one projection equals its attempt-two normalized `(stdout, stderr, returncode)` projection. | `reproducibility.json` |
| I00-P03-AC16 | The IT-01 required-artifact projection equals the complete declared path/type/digest set. | Required-artifact manifest assertions |
| I00-P03-AC17 | The candidate diff contains only `tests/e2e/test_cli_programmatic.py`. | Name-only diff |
| I00-P03-AC18 | The observed command/exit map equals the frozen quality/fork command/exit map. | Command log |
| I00-P03-AC19 | The lock-contention projection equals `{default_timeout_seconds: 10, injected_timeout_seconds: 0.01, elapsed_lt_seconds: 1, exception: "Timed out after 10 seconds acquiring the programmatic evidence lock."}`. | Hermetic held-lock test |
| I00-P03-AC20 | The child-cleanup projection equals `{start_new_session: true, timeout_seconds: 0.1, grace_seconds: 5, timeout_group_alive: false, timeout_descendant_alive: false, normal_exit_group_alive: false, normal_exit_descendant_alive: false, diagnostics: <MSG-04 pair>}`. | Hermetic timeout/normal-exit descendant test |

These criteria contribute to roadmap AC-1.2, AC-1.3, AC-3.1, AC-3.5, and
AC-7.1. They do not complete those campaign-wide criteria; ACP and final
repository gates remain for later Iteration 0 packets and I00-P99.

## Integration scenario

### IT-01: Programmatic CLI delivery

- Starting state: clean frozen candidate; fresh temporary Vibe/user homes and
  workdir per case; deterministic loopback server/config; scrubbed environment;
  no external network or paid provider.
- Actions: invoke real help/version; invoke real programmatic success; trigger
  missing-key, backend-auth, invalid-config, and untrusted-workspace paths; run
  the hermetic held-lock timeout and subprocess-group cleanup checks.
- Expected outcome: all six exact cases pass twice; success/untrusted cases make
  one request; expected failures are noninteractive/actionable/traceback-free;
  trust is not persisted; lock contention and synthetic descendant leakage
  terminate with the frozen diagnostics and no surviving process.
- Failure evidence: pytest/JUnit failure, raw runner streams/result, manifest
  `fail`, and explicit missing-artifact notes. Do not update expected output.
- Artifacts:
  `$EVIDENCE/IT-01/{command.json,command.log,stdout.txt,stderr.txt,result.json,stdout.normalized.txt,stderr.normalized.txt,junit.xml,stdout.json,stderr.json,exit-codes.json,reproducibility.json,.programmatic-evidence.lock}`.
- Covers: I00-P03-AC01 through I00-P03-AC16, I00-P03-AC19,
  I00-P03-AC20; MSG-01 through MSG-04.
- Contributes to: AC-1.2, AC-1.3, AC-3.1, AC-3.5, AC-7.1.

I00-P01 captures outer command streams, result, JUnit, and manifest digests. The
test-local opt-in evidence hook above owns the four structured per-case files;
no shared fixture, second reporting architecture, or unresolved artifact owner
remains.

## Acceptance-to-scenario map

| Requirement | Scenario/review |
|---|---|
| I00-P03-AC01 through I00-P03-AC15, I00-P03-AC19, I00-P03-AC20 | IT-01 journeys, held-lock contract, and process-group cleanup |
| I00-P03-AC16 | IT-01 external evidence manifest |
| I00-P03-AC17 | Candidate diff review |
| I00-P03-AC18 | Targeted/full quality and fork commands |
| AC-1.2, AC-1.3, AC-3.1, AC-3.5, AC-7.1 | Contribution only; final verdict belongs to I00-P99 |
| MSG-01 through MSG-04 | IT-01 raw/normalized streams, JUnit, and runner result artifacts |

## Exact verification commands

Before freeze, run the mutating format/fix commands, review their output, and
repeat targeted tests without changing the command:

```bash
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_OPTIONAL_LOCKS=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run ruff check --fix tests/e2e/test_cli_programmatic.py
uv run ruff format tests/e2e/test_cli_programmatic.py
uv run pytest -n0 tests/e2e/test_cli_programmatic.py
uv run pytest -n0 \
  tests/cli/test_cli_wiring.py \
  tests/cli/test_programmatic_setup.py \
  tests/e2e/test_cli_programmatic.py
uv run ruff check tests/e2e/test_cli_programmatic.py
uv run ruff format --check tests/e2e/test_cli_programmatic.py
uv run pyright
uv run pytest --ignore tests/snapshots
uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
  uv run scripts/check_upstream_divergence.py
uv run pre-commit run --all-files
uv run git diff --check
uv run git status --short
```

`pre-commit` contains mutating fix hooks, so it runs before freeze. Review any
mutation and ensure it affects only the allowed file; otherwise stop. Then stop
at the freeze handoff. The lead reviews and creates the candidate commit, writes
`candidate_sha` to a new clean control commit, and assigns the evidence operator.
The worker does not commit or edit control files. Require empty candidate status.
The evidence operator reruns the canonical validator with `CONTROL_SHA` set to
the newly assigned clean verification-state control commit,
`EXPECTED_PACKET_STATE=verification`, and
`EXPECTED_CANDIDATE_SHA="$CANDIDATE_SHA"`.

After freeze, use check-only commands and capture IT-01. The `env -i` allowlist
is mandatory and must not be replaced with the operator's inherited environment:

```bash
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_OPTIONAL_LOCKS=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run ruff check tests/e2e/test_cli_programmatic.py
uv run ruff format --check tests/e2e/test_cli_programmatic.py
uv run pyright

# Keep the Git settings for check commands, then remove the credential-like
# `GIT_CONFIG_KEY_0` name before the evidence runner's security preflight.
unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0
env -i \
  PATH="$PATH" \
  HOME="$HOME" \
  TMPDIR="${TMPDIR:-/tmp}" \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  TZ=UTC \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_OPTIONAL_LOCKS=0 \
  UV_OFFLINE=1 \
  PIP_NO_INDEX=1 \
  PYTHONHASHSEED=0 \
  VIBE_MAINTENANCE_TEST_EVIDENCE="$EVIDENCE/IT-01" \
  VIBE_EVIDENCE_WORKSPACE="$VIBE_EVIDENCE_WORKSPACE" \
  KILROY_RUN_ID="$KILROY_RUN_ID" \
  uv run scripts/run_maintenance_evidence.py \
  --repo-root "$REPO_ROOT" \
  --scenario IT-01 \
  --surface non_ui \
  --baseline-ref "$BASELINE_SHA" \
  --candidate-ref "$CANDIDATE_SHA" \
  --upstream-ref "$UPSTREAM_SHA" \
  --runner-id "$RUNNER_ID" \
  --timeout-seconds 900 \
  --lock-timeout-seconds 10 \
  --required-artifact junit=junit.xml \
  --required-artifact stdout=stdout.json \
  --required-artifact stderr=stderr.json \
  --required-artifact exits=exit-codes.json \
  --required-artifact reproducibility=reproducibility.json \
  --required-artifact evidence_lock=.programmatic-evidence.lock \
  --record-env PYTHONHASHSEED \
  --normalize-output \
  -- \
  uv run pytest -n0 tests/e2e/test_cli_programmatic.py \
    --junitxml "$EVIDENCE/IT-01/junit.xml"

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=diff.renames
export GIT_CONFIG_VALUE_0=true
uv run pytest -n0 \
  tests/cli/test_cli_wiring.py \
  tests/cli/test_programmatic_setup.py \
  tests/e2e/test_cli_programmatic.py
uv run pytest -n0 tests/test_iron_laws.py tests/test_upstream_divergence.py
VIBE_UPSTREAM_BASE="$UPSTREAM_SHA" VIBE_UPSTREAM_REF="$UPSTREAM_SHA" \
  uv run scripts/check_upstream_divergence.py
uv run git diff "$BASELINE_SHA".."$CANDIDATE_SHA" -- \
  tests/e2e/test_cli_programmatic.py
uv run git diff --name-only "$BASELINE_SHA".."$CANDIDATE_SHA"
uv run git status --short
```

The name-only output contains exactly
`tests/e2e/test_cli_programmatic.py`; status is empty. A post-freeze mutation
invalidates evidence and verifier state.

## Stop conditions

Stop and request `blocked` when:

- I00-P01 is incomplete or its required-artifact contract cannot accept the
  five explicitly owned test artifacts.
- The backend-auth contract has not been proven at the assigned baseline and no
  separately authorized production-fix dependency is complete. This packet
  stays `draft`; the characterization worker does not discover the known
  boundary only after beginning implementation.
- Initial status/ref/evidence isolation is invalid or the allowed file overlaps
  another change.
- Any named case fails against committed production behavior. Do not xfail,
  skip, loosen exact strings, swallow a stream, filter a traceback, add retries,
  or patch production.
- Backend auth leaks an uncaught `UnclassifiedBackendError`, traceback, secret,
  raw request, endpoint, or temporary path.
- More than one model request occurs, an unexpected HTTP path is used, or any
  non-loopback network/proxy/provider request is attempted.
- A child times out/leaks, server cleanup fails, trust persists, or real user
  home/repository state changes.
- Correct implementation needs any second repository path.
- A logical journey's two in-test attempts differ after the sole allowed
  normalization.
- Full/quality/fork checks fail outside the allowed file or pre-commit mutates a
  forbidden path.
- Candidate changes after verification starts.

## Rollback

Remove only `tests/e2e/test_cli_programmatic.py`. External IT-01 evidence may be
retained for diagnosis but must be marked superseded if the packet is reverted.
Do not touch production code, shared E2E helpers, snapshots, config, lockfile,
roadmap, thresholds, or another packet.

## Completion report

Report:

- Packet and full baseline/candidate/upstream SHAs; exact allowed diff path;
  evidence root and manifest.
- Each case's exit code, exact normalized streams, request count, and trust
  persistence result without printing any key.
- Every logical journey's two-attempt normalized equivalence result.
- Every command/exit and I00-P03-AC01 through I00-P03-AC20 result.
- MSG-01 and MSG-02 change classification: `unchanged` or
  `accidental-blocking`; characterization must report `unchanged`. Report
  harness-only MSG-03 and MSG-04 as intentional-approved additions and confirm
  their exact held-lock and child-cleanup diagnostics.
- Structured IT-01 artifacts present/missing and scenario `pass`/`fail`; never
  hide missing `stdout.json`, `stderr.json`, `exit-codes.json`, or
  `reproducibility.json`.
- Product, snapshot, prompt, provider request schema, cost/spend, performance,
  dependency, suppression, accepted-divergence, and upstream-path deltas,
  explicitly `none`.
- Any denial, skip, flake, timeout, cleanup issue, or blocker.
- Clean frozen candidate and confirmation that no push, merge, landing, or
  completion-state edit occurred.
