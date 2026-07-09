# Configuration & LLM Backends

## Configuration System

**Source**: `vibe/core/config/`

Vibe uses a layered TOML configuration system built on Pydantic settings. Config is loaded from multiple sources and merged with well-defined precedence.

### VibeConfig

**Source**: `vibe/core/config/_settings.py` (line 706)

`VibeConfig(BaseSettings)` is the central config object. Key fields:

| Category | Fields |
|---|---|
| **Models/Providers** | `providers: list[ProviderConfig]`, `models: list[ModelConfig]`, `active_model: str`, `fallback_models: list[str]`, `compaction_model`, `subagent_model`, `grunt_model` |
| **Limits** | `api_timeout`, `auto_compact_threshold`, `max_output_escalation`, `context_shaping`, `spend`, `auxiliary_budget` |
| **Subsystems** | `memory: MemoryConfig`, `safety_judge: SafetyJudgeConfig`, `project_context: ProjectContextConfig`, `experiments: ExperimentsConfig`, `worktree: WorktreeConfig` |
| **Tools** | `tools` (dict), `tool_paths`, `enabled_tools`, `disabled_tools` |
| **UI/Behavior** | `theme`, `vim_keybindings`, `bypass_tool_permissions`, `enable_telemetry`, `caveman_thinking`, `include_project_context` |
| **Prompts** | `system_prompt_id`, `compaction_prompt_id` |
| **Workflows** | `disable_workflows`, `workflow_paths` |
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

1. `./.vibe/config.toml` — project-level (trusted folders only)
2. `~/.vibe/config.toml` — user-level (fallback)
3. Environment variables (override individual fields)
4. `~/.vibe/.env` — API keys (loaded via `load_dotenv_values()`)
5. `VIBE_HOME` — override the entire Vibe home directory

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
- `LSPServer` — LSP server configuration
- `ToolManifestConfig` — tool manifest pins

### Custom Agents

Custom agent configurations are TOML files in `~/.vibe/agents/` (or `.vibe/agents/`). Each can override `active_model`, `system_prompt_id`, `enabled_tools`, `disabled_tools`, and per-tool permissions. Select with `--agent <name>`.

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

[[models]]
name = "kimi-k2.7-code"
provider = "kimi"
alias = "kimi"
thinking = "high"
auto_compact_threshold = 200000
```

Key notes:
- `api_base` includes the version segment but NOT `/chat/completions` — Vibe appends that automatically
- Kimi requires `extra_headers = { User-Agent = "KimiCLI/1.47.0" }`
- A wrong endpoint (e.g., coding plan key sent to pay-as-you-go) returns HTTP 429 that looks like a rate limit

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
2. **After turn**: `MemoryExtractor` proposes new/updated memories from transcript → `MemoryStore` persists.
3. **Periodically**: `MemoryConsolidator` merges duplicates → `MemoryStore.trash()` for reversibility.
4. **Verification**: `MemoryVerifier` re-checks factual claims against current codebase → tags `STALE`/`BROKEN`.

All LLM components use **standalone backends** — failures are best-effort no-ops that never trigger model failover.

Standalone memory and safety-judge calls share `auxiliary_budget` (defaults:
50,000 tokens, 24 calls, and $1 per running agent). A zero limit disables
auxiliary dispatch. This local helper meter is an inner envelope in addition to
the shared session ledger. Restarting Vibe creates a fresh auxiliary envelope.

## Session Spend Broker

**Source**: `vibe/core/usage/`, `vibe/core/config/_spend_config.py`

`SpendConfig` has finite defaults of 400,000 prompt tokens, 100,000 completion
tokens, 500,000 total tokens, $10, 128 calls, two concurrent calls, and 16
retries. `SessionSpendAdapter` reserves against a file-lock-backed hierarchy
before primary, compaction, in-process task/workflow, memory, and safety-judge
dispatch, then reconciles provider usage. Missing usage is charged at the
estimate. Runtime `max_price` and `max_session_tokens` values can only reduce the
configured session envelope.

Isolated subprocesses, MCP sampling, narration, and backend-internal retry
attempts remain explicit unrouted boundaries.

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
