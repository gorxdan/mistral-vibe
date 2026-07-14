# Configuration & LLM Backends

## Configuration System

**Source**: `vibe/core/config/`

Vibe uses a layered TOML configuration system built on Pydantic settings. Config is loaded from multiple sources and merged with well-defined precedence.

### VibeConfig

**Source**: `vibe/core/config/_settings.py` (line 706)

`VibeConfig(BaseSettings)` is the central config object. Key fields:

| Category | Fields |
|---|---|
| **Models/Providers** | `providers: list[ProviderConfig]`, `models: list[ModelConfig]`, `active_model: str`, `fallback_models: list[str]`, `compaction_model`, `subagent_model`, `grunt_model`, `model_routing` |
| **Limits** | `api_timeout`, `api_retry_max_elapsed_time`, `auto_compact_threshold`, `max_output_escalation`, `context_shaping`, `spend`, `auxiliary_budget` |
| **Subsystems** | `memory: MemoryConfig`, `safety_judge: SafetyJudgeConfig`, `project_context: ProjectContextConfig`, `experiments: ExperimentsConfig`, `worktree: WorktreeConfig` |
| **Tools** | `tools` (dict), `tool_paths`, `enabled_tools`, `disabled_tools` |
| **UI/Behavior** | `theme`, `vim_keybindings`, `bypass_tool_permissions`, `enable_telemetry`, `caveman_thinking`, `include_project_context` |
| **Prompts** | `system_prompt_id`, `compaction_prompt_id` |
| **Workflows** | `disable_workflows`, `workflow_paths` |
| **Verification** | `verification_subsystem`, `trusted_verification_recipe` |
| **VibeCode (cloud)** | `vibe_code_enabled`, `vibe_code_base_url`, `vibe_code_workflow_id`, `vibe_code_api_key_env_var` |

### Config Architecture

- **Layer system** (`config/layer.py`): `ConfigLayer`, `RawConfig`, trust resolution — supports layered config (user, project, harness)
- **Schema** (`config/schema.py`, `config/vibe_schema.py`): `ConfigSchema`, `VibeConfigSchema` — merge strategies (`WithReplaceMerge`, `WithConcatMerge`, `WithUnionMerge`, `WithConflictMerge`)
- **Patch system** (`config/patch.py`): `ConfigPatch`, `AddOperationPatch`, `RemoveOperationPatch`, `ReplaceOperationPatch`
- **Orchestrator** (`config/orchestrator.py`): config loading and layer composition
- **Models** (`config/models.py`): `ModelConfig`, `ProviderConfig`, `MCPServer`, `ConnectorConfig`
- **Builder** (`config/builder.py`): config construction
- **Event bus** (`config/event_bus.py`): `ConfigChangeEvent` for reactive config updates
- **Fingerprint** (`config/fingerprint.py`): `file_fingerprint` for trust verification

### Config File Locations

1. `~/.vibe/config.toml` — user-level TOML base
2. `./.vibe/config.toml` and trusted ancestor project files — project TOML
   overlays
3. `VIBE_` environment variables — override TOML fields
4. Programmatic `VibeConfig` initialization — highest precedence
5. `~/.vibe/.env` — API keys loaded separately by `load_dotenv_values()`
6. `VIBE_HOME` — override the Vibe home directory

The runtime settings order is programmatic initialization, environment, merged
TOML, then file secrets. Project TOML normally overlays user TOML. One field is
deliberately different: every project `trusted_verification_recipe` entry is
removed case-insensitively before merge. A trusted recipe may come only from the
user layer, host environment, or programmatic initialization. When present, it
forces `verification_subsystem = true`; a project cannot disable the bound
verification contract.

Custom providers/models are **merged** with built-in Mistral defaults (concat merge for lists, replace merge for scalars), so you keep Mistral available while adding others.

### Key Config Types

From `vibe/core/config/__init__.py`:
- `ModelConfig` — model id, provider, alias, thinking level, pricing, context threshold, image support
- `ProviderConfig` — endpoint, API key env var, backend type, API style
- `SafetyJudgeConfig` — enabled, model, max_tokens, timeout
- `SpendConfig` — shared session prompt/completion/total token, USD, call,
  concurrency, retry, and deadline limits for routed paid calls
- `AuxiliaryBudgetConfig` — shared per-session token, call, and USD caps for standalone helper models
- `MemoryConfig` — enabled, select_mode, model, max_selected, max_inject_chars, timeout
- `SandboxConfig` — bash sandbox settings
- `WorktreeConfig` — subagent isolation settings
- `EffortLevel` — normal / le-chaton
- `LSPServer` — LSP server configuration; `lsp_max_workspace_roots` (default 8)
  bounds dynamically discovered monorepo roots without counting session or
  explicit roots
- `ToolManifestConfig` — tool manifest pins
- `PurposeModelRoutingConfig` — explicit aliases for formatting, retrieval
  ambiguity, mechanical work, and semantic escalation

### Custom Agents

Custom agent configurations are TOML files in `~/.vibe/agents/` (or `.vibe/agents/`). Each can override `active_model`, `system_prompt_id`, `enabled_tools`, `disabled_tools`, and per-tool permissions. Select with `--agent <name>`. A topology-bound Task call still accepts only an effective read-only built-in `reviewer` or `verifier`; a custom profile cannot widen that managed delegation boundary.

### Custom Prompts

- System prompts: `~/.vibe/prompts/<id>.md` (replaces `prompts/cli.md` default)
- Compaction prompts: `~/.vibe/prompts/<id>.md` (replaces `prompts/compact.md` default)
- Project-local prompts in `.vibe/prompts/` override user-level with the same name
- Set `system_prompt_id` / `compaction_prompt_id` in config to select

### AGENTS.md

AGENTS.md files provide custom instructions layered over the system prompt:
- `~/.vibe/AGENTS.md` — user-level, all projects
- Project directories — loaded from cwd up to trust root; closer overrides more distant
- Only loaded for trusted folders

## LLM Backend System

**Source**: `vibe/core/llm/`

### BackendLike Protocol

**Source**: `vibe/core/llm/types.py`

`BackendLike` is a `Protocol` defining the backend contract:
- `async __aenter__` / `async __aexit__` — async context manager
- `async complete(request) -> LLMChunk` — non-streaming completion
- `complete_streaming(request) -> AsyncGenerator[LLMChunk]` — streaming completion

`CompletionRequest` (frozen dataclass): `model`, `messages`, `temperature`, `tools`, `max_tokens`, `tool_choice`, `extra_headers`, `metadata`, `response_format`, `extra_body`.

### Backend Factory

**Source**: `vibe/core/llm/backend/factory.py`

`create_backend(provider, timeout, retry_max_elapsed_time, enable_otel)` dispatches on `provider.backend`:
- `Backend.MISTRAL` → `MistralBackend` (uses `mistralai` SDK)
- `Backend.GENERIC` → `GenericBackend` (httpx-based, supports OpenAI/Anthropic adapters)
- Supports test-injectable `BACKEND_FACTORY` override

### Backends

| Backend | Source | Description |
|---|---|---|
| **MistralBackend** | `llm/backend/mistral.py` | Uses `mistralai` SDK, Mistral-specific message/tool formats, `RetryConfig`/`BackoffStrategy` |
| **GenericBackend** | `llm/backend/generic.py` | httpx-based, composes adapters: `OpenAIAdapter`, `AnthropicAdapter`, `ChatGPTResponsesAdapter`, `OpenAIResponsesAdapter`, `ReasoningAdapter` |
| **Anthropic** | `llm/backend/anthropic.py` | Direct Anthropic API adapter |
| **OpenAI Responses** | `llm/backend/openai_responses.py` | OpenAI Responses API |
| **Bedrock** | `llm/backend/` | AWS Bedrock wrapper |
| **Vertex** | `llm/backend/` | Google Vertex AI wrapper |

### API Styles

Configured via `api_style` in provider config:
- `openai` (default) — standard OpenAI-compatible `/chat/completions`, streams `reasoning_content`
- `anthropic` — Anthropic Messages API
- `reasoning` — reasoning model adapter (parses content blocks)
- `openai-responses` — OpenAI Responses API
- `vertex-anthropic` — Vertex AI with Anthropic API

**Important**: Do NOT use `api_style = "reasoning"` for providers that stream `reasoning_content` (like Kimi, GLM) — the reasoning adapter parses content blocks and would drop the streamed `reasoning_content` field.

### LLM Models

**Source**: `vibe/core/llm/models.py`

- `ParsedToolCall` — raw tool call from LLM (name, raw_args, call_id)
- `ResolvedToolCall` — validated tool call (name, `tool_class`, `validated_args: BaseModel`, call_id)
- `FailedToolCall` — failed parsing/validation
- `ParsedMessage` / `ResolvedMessage` — collections of the above

### Model Failover

**Source**: `vibe/core/agent_loop_failover.py`

When the active model hits rate-limit/overload/content-filter errors:
1. `_switch_to_fallback_model()` iterates `config.fallback_models`
2. Skips already-tried aliases
3. `_activate_model(model)` creates a new backend, updates pricing and compaction threshold
4. User can manually switch with `/model` → `_switch_to_chosen_model(alias)`

### Adding Custom Providers

Most third-party coding models expose an OpenAI-compatible endpoint:

```toml
[[providers]]
name = "kimi"
api_base = "https://api.kimi.com/coding/v1"
api_key_env_var = "KIMI_API_KEY"
backend = "generic"
api_style = "openai"

[providers.cache]
session_keyed = true
session_key_field = "prompt_cache_key"

[[models]]
name = "kimi-for-coding"
provider = "kimi"
alias = "kimi"
thinking = "high"
pricing_mode = "subscription"
auto_compact_threshold = 200000
```

Key notes:
- `api_base` includes the version segment but NOT `/chat/completions` — Vibe appends that automatically
- Kimi Code uses the real client identity and `prompt_cache_key`; do not spoof another client's User-Agent
- ZAI Coding Plan caching is automatic and gets no undocumented cache key
- A wrong endpoint (e.g., coding plan key sent to pay-as-you-go) returns HTTP 429 that looks like a rate limit

## Prompt cache and pricing contracts

`ProviderCacheConfig` controls wire behavior: `mode`, `style`, `extra_body`,
`cache_key`, `session_keyed`, `session_key_field`, and Anthropic's optional
`ttl` (`5m` or `1h`). Canonical OpenAI and
ChatGPT requests use `prompt_cache_key`; Mistral and Kimi presets explicitly opt
in to that field; OpenRouter uses `session_id`; Anthropic-compatible providers
can use message `cache_control` breakpoints. Other compatible endpoints remain
keyless unless configured, because unknown top-level fields can be rejected.

`ModelConfig` separately controls billing with `pricing_mode` (`auto`, `api`,
`subscription`, `free`, or `unknown`), `input_price`, `cached_input_price`,
`cache_write_input_price`, and `output_price`. Cache reads and writes are
preserved in provider telemetry, the spend ledger, workflow child costs, usage
history, tracing, ACP, and the status card. Estimated fallback costs are marked
with `~`; explicit subscription/free usage is an exact incremental `$0.0000`.
When OpenRouter supplies its authoritative `usage.cost`, that charged amount
takes precedence over model-table estimates.

There is no speculative session-start model call. It would consume money or
plan quota and only moves first-turn work earlier; the first real request warms
the cache for subsequent turns.

## Skills System

**Source**: `vibe/core/skills/`

Skills are markdown instruction files (with YAML frontmatter) injected into agent context.

### Discovery

`SkillManager` (`skills/manager.py`) discovers skills from:
1. Built-in skills (always loaded first)
2. Custom paths (`skill_paths` in config)
3. `.agents/skills/` (Agent Skills standard, trusted folders only)
4. `.vibe/skills/` (project-local, trusted folders only)
5. `~/.vibe/skills/` and `~/.agents/skills/` (global)

User/project skills shadow builtins by name (first-wins dedup). Filtered by `enabled_skills` (allowlist) or `disabled_skills` (blocklist) using glob/regex matching.

### Built-in Skills

**Source**: `vibe/core/skills/builtins/`

| Skill | Source |
|---|---|
| **vibe** (75 KB) | `builtins/vibe.py` — core assistant instructions |
| **workflow** | `builtins/workflow.py` — workflow usage guide |
| **tool_guides** | `builtins/tool_guides.py` — tool usage guide |
| **capsules** | `builtins/capsules.py` — capsule skill support |

### Skill Metadata

- `SkillSource` (StrEnum): `BUILTIN`, `LOCAL`, `REGISTRY`
- `SkillScope` (StrEnum): `BUILTIN`, `GLOBAL`, `PROJECT`
- `SkillMetadata`: name (regex-validated slug), description, summary, frontmatter fields
- Registry skills can be versioned and pinned via `RegistryRef`

## Memory System

**Source**: `vibe/core/memory/`

Durable cross-session notes as plain `*.md` files under `~/.vibe/memory/`.

### How It Works

1. **Turn start**: `LocalMemorySelector` ranks the `MemoryStore` index by id, title, description, tags, and index metadata. In the default `hybrid` mode, the standalone LLM `MemorySelector` runs only when local candidates tie near the selection cutoff. A confident match (including a confident empty result) uses no selector API call.
2. **After turn**: `MemoryExtractor` runs only when local detection sees explicit
   remember intent, a preference/correction, or a durable decision; routine task
   prose causes no extractor call.
3. **Periodically**: `MemoryConsolidator` merges duplicates → `MemoryStore.trash()` for reversibility.
4. **Verification**: `MemoryVerifier` re-checks factual claims against current codebase → tags `STALE`/`BROKEN`.

All LLM components use **standalone backends** — failures are best-effort no-ops that never trigger model failover.

Standalone memory and safety-judge calls share `auxiliary_budget` (defaults:
50,000 tokens, 24 calls, and $1 per running agent). A zero limit disables
auxiliary dispatch. This local helper meter is an inner envelope in addition to
the shared session ledger. Restarting Vibe creates a fresh auxiliary envelope.

## Session Spend Broker

**Source**: `vibe/core/usage/`, `vibe/core/config/_spend_config.py`

`SpendConfig` leaves cumulative prompt, completion, and total token caps unset by
default. The finite defaults remain $10, 512 calls, two concurrent calls, 16
retries, a 32,768-token per-call output bound, and a 256-token minimum admitted
output. Explicit `max_prompt_tokens`, `max_completion_tokens`, and
`max_total_tokens` values are preflight admission caps; runtime `max_price` and
`max_session_tokens` can tighten the USD and token envelopes. Adaptive estimates
can reconcile above the remaining allowance by one unexpectedly token-dense
call. Strict mode is the most conservative option when minimizing that overshoot
risk matters.

`SessionSpendAdapter` reserves against a file-lock-backed hierarchy before
primary, compaction, task/workflow/team, memory, safety-judge, narration, repair,
and verification dispatch,
then reconciles provider usage. The default `prompt_estimator_mode = "adaptive"`
starts conservatively and learns from exact reconciliations persisted in the
ledger. Observations are isolated by provider, model, and request shape, and only
comparable request sizes calibrate one another. `"strict"` mode disables learning
and reserves the serialized token-bearing request byte ceiling. Missing usage is
charged at the reservation estimate.

Positive paid-call concurrency limits apply backpressure rather than exhausting
the budget: in-process agents and attached subprocesses wait and retry the atomic
reservation after active calls settle. An explicit zero concurrency limit remains
a hard admission denial. Token, USD, cumulative-call, retry, and deadline limits
remain terminal when reached.

An exact, untouched legacy generated `[spend]` table with the released 128-call
signature is migrated once by removing its old 400,000 prompt, 100,000
completion, and 500,000 total token defaults. Customized or partial tables
remain explicit hard limits. Existing ledgers can relax matching legacy defaults
only for fields omitted after migration. Config reload can raise or lower the
configured token, cost, call, concurrency, and retry ceilings; an existing
absolute deadline remains fixed or tightens. `/spend` reports the active
envelope, while `/spend reset` durably starts a fresh ledger without clearing the
conversation and rebinds subsequent team launches.

Routed requests that omit `max_tokens` receive the affordable completion bound,
which is reduced atomically across active scope limits when necessary. The
broker rejects instead of reducing below `minimum_admitted_output_tokens`
(default 256). An explicit `max_tokens` is a hard request and is never reduced.
The `openai-chatgpt` Codex endpoint rejects the field, so its adapter strips it
and the broker enforces the reservation through reconciliation rather than an
HTTP output cap.

Isolated subprocesses attach to existing child scopes through
`VIBE_SPEND_CONTEXT`. Generic/Mistral redispatches authorize retry count, elapsed
policy, and another conservative token/USD exposure estimate against the
original reservation; opaque Mistral SDK retries are disabled. MCP sampling
remains the explicit unrouted boundary.
Non-token-priced text-to-speech is also an explicit paid-call boundary outside
the token ledger, as are real-time transcription and Mistral's model-backed web
search.

## Bash sandbox

`SandboxConfig` is nested under `[tools.bash.sandbox]` and defaults to
`enabled = true`, `allow_network = true`, `scrub_env = true`,
`require_backend = false`, `backend = "auto"`, and `seccomp = true`, with empty
`write_dirs` and `env_passthrough` lists. Ordinary sessions prefer Bubblewrap or
Seatbelt and may fall back. Linux `unshare` does not enforce filesystem or
network policy and is reported as a weak fallback.

Auto-approve, isolated/task-contracted work, topology-bound sessions, and
trusted checks are strict modes. Strict model Bash requires Bubblewrap or
Seatbelt; topology-bound model Bash rejects `background=true`, disables
network, and can write only its scratchpad. Trusted checks are narrower: they
run only on Linux Bubblewrap, never Seatbelt, with network off and only a
per-check disposable run root writable. All strict modes scrub credentials and
fail closed when confinement cannot start. Ordinary auto-approve retains
workspace writes, configured `write_dirs`, and Git commit compatibility.

`SandboxConfig.extra_args` remains an internal compatibility field for narrow
namespace/runtime flags. Bubblewrap rejects filesystem mounts, overlays, tmpfs,
device/proc mounts, chdir, argument-file expansion, and command terminators. It
cannot grant a path or alter the host-assembled command graph. See
`docs/design/sandbox.md` for the complete mode table.

## Purpose Model Routing

Aliases are opt-in and must name configured models:

```toml
[model_routing]
formatter_model = "cheap"
retrieval_model = "cheap"
mechanical_model = "cheap"
semantic_escalation_model = "strong"
```

Formatting receives only bounded raw output, schema, and diagnostics. Retrieval
routing is used only for ambiguous local memory matches. `mechanical_model`
overrides model requests for the trusted mechanical-edit manifest. The semantic
alias is used only after the bounded repair controller detects repeated semantic
no-progress.
An unavailable explicit optional alias fails closed; retrieval skips instead of
falling back to the active model.

## Trusted Verification Recipe

**Sources**: `vibe/core/config/_verification_config.py`,
`vibe/core/tools/builtins/verify_work.py`, `vibe/core/verification_state.py`

`trusted_verification_recipe` is an optional frozen config model containing a
recipe version, task brief, acceptance contract, allowed-path patterns, and one
or more exact `argv`/`cwd`/timeout checks. `AgentLoop` copies it into session
state at creation and preserves that original value across config reloads.
The recipe source must be user, `VIBE_` environment, or programmatic config;
project TOML entries are removed before settings validation. A bound recipe
forces the verification subsystem on. Managed reviewer and verifier children
inherit the same frozen value. Checks execute with `shell=False`; a shell or
`env` cannot be the executable, and either is rejected when selected behind
`uv run`. Every check requires a full lowercase 64-character
`executable_sha256` plus an absolute host-owned `environment_attestation_path`
and matching full lowercase `environment_attestation_sha256`. The runner reads
the executable through a descriptor-safe path, verifies its digest, copies it
to a private read-only path, and executes only that copy. Shebang wrappers are
rejected; use a pinned native interpreter with `-m <module>` or a script
argument. The source executable, private copy, and attestation are checked
before and after execution. Check configs may require or forbid output patterns.
A `test_count_pattern` must contain a named integer
`(?P<count>...)` group and be paired with `minimum_test_count >= 1`.
Non-executing help, version, collection, list, dry-run, no-run,
failure-masking, and structurally empty selectors are rejected. `dotnet test`,
pytest, unittest, and `cargo test` must report positive executed-test counts.
`go test` must emit at least one verbose `--- PASS:` record unless an explicit
count contract is supplied. npm/pnpm/yarn/bun test, `make test`, tox, nox,
Jest, and Vitest always require that contract. All observed custom counts must
agree and meet the minimum. Regex evaluation is killably time-bounded. Unknown
runners additionally require `custom_runner = true` and at least one
`required_output_patterns` entry. The executable and attestation must be
pre-provisioned outside the repository. Trusted recipes reject `uv` and
pre-commit entrypoints. Hosts must not configure another package-manager
installation as a check; the runtime does not classify every package-manager
CLI.

The attestation is a host-owned assertion, not a transitive digest of the
dynamic loader, shared libraries, language packages, or every other dependency.
Those runtime roots remain read-only to the sandboxed process but host-owned;
the host must provision them immutably or prevent concurrent changes for the
duration of an authority-bearing check.

```toml
[[trusted_verification_recipe.checks]]
name = "custom-tests"
argv = ["/opt/vibe-checks/bin/acme-test", "--ci"]
cwd = "."
timeout_seconds = 600
custom_runner = true
executable_sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
environment_attestation_path = "/opt/vibe-checks/environment.json"
environment_attestation_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
required_output_patterns = ["ACME RESULT: PASS"]
test_count_pattern = "ACME tests: (?P<count>[0-9]+)"
minimum_test_count = 1
```

An optional nested `execution_topology` binds managed work to a packet ID and
path, `active` or `verification` control state, exact control/candidate paths and
SHAs, candidate branch, durable evidence/run identity, `max_turns`, and
`max_session_tokens`. Root AgentLoop
construction validates both registered physical worktrees, cleanliness and HEAD,
packet/status agreement, dependencies, exact sorted scenarios, and an evidence
directory outside the
system temporary directory that neither contains nor is contained by any
worktree or Git metadata. Packet and status data are read as regular tracked
blobs from the exact control commit. Git probes ignore ambient `GIT_*` variables
and user/system Git configuration. Verification state requires the frozen
candidate SHA and `evidence_manifest_sha256`; active state forbids both. The
host first commits verification state, candidate SHA, and sorted scenarios,
finalizes evidence, then records the canonical manifest digest in packet and
status in a second final verification control commit. Only that final commit
can start a verification AgentLoop. Manifest validation holds the lock,
requires empty reservations and strict canonical JSON, and verifies exact
inventories, artifact hashes, identities, and the committed candidate
`uv.lock` digest. A failed topology probe aborts startup before a model turn.
Topology turn/token limits cap runtime even if a caller requests more.

Topology also installs an authoritative capability ceiling from canonical core
tools. Active roots have at most `bash`, `edit`, `glob`, `grep`, `read`,
`skill`, `task`, `todo`, and `write_file`. Verification roots have at most
`glob`, `grep`, `read`, `skill`, `task`, and `verify_work`. Availability,
permissions, and structured task manifests may narrow these sets. Project and
plugin tools, MCP/connectors, workflows, teams, web tools, `tool_search`, and
`land_work` cannot enter the managed catalog. Managed Task delegates only to
effective read-only built-in reviewer/verifier profiles; child tools are capped
at `bash`, `glob`, `grep`, `read`, and `skill`, plus contract-bound
`task_checks` for a structured task.
Managed reads are confined to the candidate/control/evidence roots, session
scratchpad, host skill roots, and active prompt files. Host logs, receipts,
runtime state, and unrelated paths are denied.

In an active worktree, a current verifier PASS makes the no-argument
`verify_work` tool eligible to execute the prebound plan. The tool has no command
or path fields in its model-visible schema. A passing receipt is bound to the
current main HEAD, candidate repository state, task, contract, recipe, and check
evidence. A topology-bound `verification` session uses the bound baseline and
candidate directly even when no worktree-manager session is active. `land_work`
revalidates receipts and reports the resulting merge commit only on its standard
active-worktree path.

Receipt checks run in an independent fail-closed Linux Bubblewrap sandbox; they
do not support Seatbelt. Each check runs against an exact-HEAD Git-exported
snapshot through a private copy of its digest-pinned native executable and a
pinned host environment attestation. The candidate and Git common directory
are not exposed inside the sandbox; only the frozen snapshot, copied executable,
and required runtime roots are readable, and only a disposable run directory is
writable. The attestation is not a transitive dependency hash.
Host credentials/config are scrubbed, caches are disposable, network is
disabled, and combined stdout/stderr is capped at 1 MiB before artifact
persistence. Repository state is captured before and after the
checks; a mutation, dirty candidate, or disallowed changed path fails the
receipt.

Non-trivial `land_work` requires a current trusted receipt. When no recipe was
bound at session start, only a locally checked documentation-only trivial
waiver may land; a legacy state-recorded verifier PASS and pasted verification
prose are not authority. Restart Vibe to adopt an intentional recipe change.

Verified delivery and landing use the exact authorized Git object ID and
compare-and-swap ref updates; a moved expected ref fails instead of resolving a
new target. Checked-out worktree materialization is necessarily a multi-file
operation. The merge lock coordinates Vibe landing operations, but external
editors and Git processes must remain idle during the approved landing window.

For every verifier attempt, the host records PASS, FAIL, PARTIAL, or INVALID.
Structured completion and outcome fields outrank raw task prose. Until the
current pass and any configured receipt are valid, AgentLoop suppresses the
model's tool-free completion and emits a host BLOCKED/PARTIAL status instead.

### Memory Types

`MemoryType` (StrEnum): `USER`, `FEEDBACK`, `PROJECT`, `REFERENCE`

### Scoping

- **Global** (default): shared across every project
- **Project**: `scope = "project"` writes to `~/.vibe/memory/projects/<hash>/` (keyed by git common dir; never committed)
- Project memories shadow same-id global memories for that project only

### Config

```toml
[memory]
enabled = true
select_mode = "per-turn"   # "per-turn" | "per-session" | "always"
selector_mode = "hybrid"   # "local" | "hybrid" | "llm"
local_min_score = 3.0
local_ambiguity_margin = 0.15
prefetch = true              # races only ambiguous hybrid / llm selection
model = ""                  # alias; falls back to compaction, then active model
max_selected = 2
max_inject_chars = 4000
timeout = 20.0
```

`local` never calls the memory selector model. `hybrid` calls it only to break an ambiguous local cutoff. `llm` restores the legacy selector-on-every-selection behavior. Auto-extraction, consolidation, and verification are controlled only by their explicit memory flags; `le-chaton` does not enable them implicitly.

## Where to Start When Changing Config/Backends

- **Config schema**: `vibe/core/config/_settings.py` → `vibe/core/config/vibe_schema.py`
- **Config loading**: `vibe/core/config/orchestrator.py` → `vibe/core/config/builder.py`
- **New backend**: `vibe/core/llm/backend/factory.py` → `vibe/core/llm/types.py` (BackendLike protocol)
- **Model failover**: `vibe/core/agent_loop_failover.py`
- **Skills**: `vibe/core/skills/manager.py` → `vibe/core/skills/parser.py`
- **Memory**: `vibe/core/memory/store.py` → `vibe/core/memory/local_selector.py` → `vibe/core/memory/selector.py`

## Tests

- `tests/core/test_config_resolution.py` (77 KB) — config layer merge tests
- `tests/core/test_config_orchestrator.py`, `test_config_layer.py`, `test_config_toml_merge.py`
- `tests/backend/` — backend adapter tests
- `tests/core/test_llm_exceptions_and_retry.py`, `test_model_fallback.py`, `test_prompt_caching.py`
- `tests/core/test_memory.py` (97 KB)
- `tests/skills/` — skill discovery and parsing tests
